"""Technique 3: FoundAD-style few-shot anomaly detection on DINOv2 patch features.

Idea (Zhai et al., 2025, "Foundation visual encoders are secretly few-shot anomaly
detectors"): a frozen foundation encoder already describes every image patch well.
A small *predictor* network is trained, on a handful of good images, to map each
patch feature back onto itself. During training it is also shown synthetic defects
(CutPaste: a patch of the image copied somewhere else) while still being asked to
predict the *clean* features. It therefore learns to project features onto the
"normal manifold": for a normal patch, prediction == feature; for a defective
patch, the prediction is pulled towards normal, so the gap is large.

    score(image)  = mean of the K largest patch errors   (one bad spot is enough)
    heatmap       = the 32 x 32 grid of patch errors

What is trained: the predictor only (~5 M parameters). What is needed: a few good
images (default 4), no defect images, no labels. The threshold is still chosen on
split A, like the autoencoder's, so the comparison stays fair.

Reference implementation: github.com/ymxlzgy/FoundAD (MIT licence). Re-implemented
here against the repository's conventions (fixed seeds, CPU demo, unittest) rather
than vendored: its environment pins Python 3.8 / PyTorch 2.2 and 20 extra packages.
"""
import io
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.dino import PATCH, Block

FOUNDAD_SIZE = 448      # 448 / 14 = 32 x 32 patch grid (FoundAD uses 512 / 16 = the same grid)
TOP_K = 10              # image score = mean of the 10 highest patch errors (FoundAD's K for MVTec)
N_FROM_END = 3          # encoder block the features are taken from (3rd from the end, as FoundAD)


# ----------------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------------
class Predictor(nn.Module):
    """Patch features (b, n, dim) -> predicted patch features (b, n, dim).

    A narrow transformer (6 blocks, width 384) with the input embedding added back
    after every block, exactly as in FoundAD. No positional embedding is used
    (FoundAD's default): the predictor sees a bag of patches, so it cannot learn
    "position (x, y) is always dark" shortcuts and must rely on patch content.
    """

    def __init__(self, dim: int, width: int = 384, depth: int = 6, heads: int = 12):
        super().__init__()
        self.embed = nn.Linear(dim, width)
        self.blocks = nn.ModuleList(Block(width, heads, layer_scale=False) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.proj = nn.Linear(width, dim)
        self.apply(self._init)
        for i, block in enumerate(self.blocks, start=1):  # FoundAD's depth-scaled init of the residual paths
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * i))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * i))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, features):
        x = self.embed(features)
        residual = x
        for block in self.blocks:
            x = block(x) + residual
        return self.proj(self.norm(x))


class FoundAD(nn.Module):
    """Frozen DINOv2 encoder + trainable predictor. Only the predictor is saved."""

    def __init__(self, encoder, predictor: Predictor, n_from_end: int = N_FROM_END, top_k: int = TOP_K):
        super().__init__()
        self.encoder, self.predictor = encoder, predictor
        self.n_from_end, self.top_k = n_from_end, top_k

    def features(self, x):
        with torch.no_grad():
            return self.encoder(x, self.n_from_end)

    def patch_errors(self, x) -> torch.Tensor:
        """Squared error between each patch feature and its prediction: (b, n_patches)."""
        feats = self.features(x)
        return ((feats - self.predictor(feats)) ** 2).mean(dim=-1)


# ----------------------------------------------------------------------------
# Pre-processing and scoring
# ----------------------------------------------------------------------------
def foundad_transform(size: int = FOUNDAD_SIZE) -> T.Compose:
    """Resize to the patch grid and ImageNet-normalise (what DINOv2 was trained with)."""
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def score_from_errors(errors: torch.Tensor, top_k: int,
                      foreground: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """(image score, error map). Score = mean of the top-K patch errors; map = grid of errors.

    With a foreground mask (patch grid, True on the part) only the part's patches can
    score: the belt around the part is not inspected. Why: the line's frames carry codec
    noise and lighting that the dataset's pure-white PNG backgrounds never had, and on
    the demo feed those background patches alone lifted every good bottle above the
    REJECT line while the part itself scored as in the dataset. A defect is on the part.
    """
    errors = errors[0]
    side = int(math.sqrt(errors.numel()))
    grid = errors.reshape(side, side)
    if foreground is not None and foreground.any():
        grid = torch.where(torch.from_numpy(foreground).to(grid.device), grid, torch.zeros_like(grid))
    score = grid.flatten().topk(min(top_k, grid.numel())).values.mean()
    return float(score), grid.cpu().numpy()


def patch_foreground(image: np.ndarray, side: int) -> np.ndarray:
    """side x side patch mask of the part: a patch counts as part if more than half of its
    pixels are foreground (foreground_mask: Otsu, border = background)."""
    mask = foreground_mask(image)
    patch = mask.shape[0] // side
    return mask[:side * patch, :side * patch].reshape(side, patch, side, patch).mean(axis=(1, 3)) > 0.5


def foundad_inspect(model: FoundAD, image: Image.Image, size: int, device) -> tuple[float, np.ndarray]:
    """Score one PIL image. Returns (score, 32x32 patch-error map); only the part's patches score."""
    resized = image.convert("RGB").resize((size, size), Image.BILINEAR)
    foreground = patch_foreground(np.asarray(resized), patch_grid_side(size))
    x = foundad_transform(size)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        errors = model.patch_errors(x)
    return score_from_errors(errors, model.top_k, foreground)


def upsample_map(error_map: np.ndarray, size: int) -> np.ndarray:
    """Bilinear upsampling of the patch grid to size x size (for pixel-level evaluation)."""
    grid = torch.from_numpy(error_map)[None, None]
    return F.interpolate(grid, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


# ----------------------------------------------------------------------------
# Training-time augmentation: CutPaste synthetic defects
# ----------------------------------------------------------------------------
def foreground_mask(image: np.ndarray) -> np.ndarray:
    """Boolean mask of the part. Otsu's threshold splits the grey levels in two; the
    class that dominates the image border is taken to be the background. This is
    product-agnostic (dark bottle on a white belt, or the reverse)."""
    grey = np.asarray(Image.fromarray(image).convert("L"))
    threshold = otsu_threshold(grey)
    bright = grey > threshold
    border = np.concatenate([bright[0], bright[-1], bright[:, 0], bright[:, -1]])
    return ~bright if border.mean() > 0.5 else bright


def otsu_threshold(grey: np.ndarray) -> int:
    """Grey level that best separates the histogram into two classes (max between-class variance)."""
    hist = np.bincount(grey.ravel(), minlength=256).astype(np.float64)
    weight1 = np.cumsum(hist)
    weight2 = weight1[-1] - weight1
    mean1 = np.cumsum(hist * np.arange(256)) / np.maximum(weight1, 1)
    mean2 = (mean1[-1] * weight1[-1] - np.cumsum(hist * np.arange(256))) / np.maximum(weight2, 1)
    between = weight1 * weight2 * (mean1 - mean2) ** 2
    return int(np.argmax(between))


class CutPaste:
    """Paste a copied piece of the image onto the part to fake a defect (Li et al., 2021).

    Half the time a rectangle covering 2-25% of the image (a 'blob' defect), half the
    time a thin rotated strip (a 'scar' / crack). The piece is colour-jittered so it does
    not blend in. Only the *context* image gets the paste; the target features stay
    those of the clean image, which is what teaches the predictor to undo defects.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.jitter = T.ColorJitter(0.5, 0.5, 0.5, 0.5)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.blob(image) if self.rng.random() < 0.5 else self.scar(image)

    def _piece(self, image: np.ndarray, w: int, h: int) -> np.ndarray:
        ih, iw = image.shape[:2]
        x, y = self.rng.randint(0, iw - w), self.rng.randint(0, ih - h)
        piece = Image.fromarray(image[y:y + h, x:x + w])
        return np.asarray(self.jitter(piece))

    def _paste(self, image: np.ndarray, piece: np.ndarray) -> np.ndarray:
        """Paste the piece at a random position whose top-left corner lies on the part."""
        ih, iw = image.shape[:2]
        ph, pw = piece.shape[:2]
        ys, xs = np.nonzero(foreground_mask(image)[: ih - ph + 1, : iw - pw + 1])
        if len(ys) == 0:
            return image
        i = self.rng.randrange(len(ys))
        out = image.copy()
        out[ys[i]:ys[i] + ph, xs[i]:xs[i] + pw] = piece
        return out

    def blob(self, image: np.ndarray) -> np.ndarray:
        ih, iw = image.shape[:2]
        area = self.rng.uniform(0.02, 0.25) * ih * iw
        aspect = self.rng.uniform(0.3, 1 / 0.3)
        w = int(np.clip(round(math.sqrt(area * aspect)), 1, iw))
        h = int(np.clip(round(math.sqrt(area / aspect)), 1, ih))
        return self._paste(image, self._piece(image, w, h))

    def scar(self, image: np.ndarray) -> np.ndarray:
        w, h = self.rng.randint(2, 16), self.rng.randint(10, 25)
        piece = Image.fromarray(self._piece(image, w, h))
        piece = piece.rotate(self.rng.uniform(-45, 45), expand=True, fillcolor=None)
        return self._paste(image, np.asarray(piece))


def train_augment() -> T.Compose:
    """FoundAD's light augmentation of the few good images: 30% chance of a flip or a
    90-degree turn, 30% chance of a colour / grey / blur change. Applied before CutPaste,
    so target and context are made from the same augmented image. Randomness comes from
    torch's generator, seeded by set_seed()."""
    orientation = T.RandomChoice([T.RandomHorizontalFlip(1.0), T.RandomVerticalFlip(1.0),
                                  T.RandomRotation((90, 90)), T.RandomRotation((270, 270))])
    appearance = T.RandomChoice([T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.RandomGrayscale(1.0),
                                 T.GaussianBlur(23, sigma=(0.1, 2.0))])
    return T.Compose([T.RandomApply([orientation], p=0.3), T.RandomApply([appearance], p=0.3)])


COMPRESS_P = 0.5            # share of training pairs that go through a codec round trip
COMPRESS_QUALITY = (40, 95)  # JPEG quality range: USB cameras and video files sit in it


def jpeg_round_trip(image: np.ndarray, quality: int) -> np.ndarray:
    """The image after JPEG compression at this quality (uint8 RGB in, uint8 RGB out).

    Why: the line's frames come from a camera or a video file, so the model sees codec
    artefacts (8x8 blocks, chroma at half resolution) that the dataset's PNGs never had.
    Without this, a model trained on PNGs scored every bottle of the demo feed 4-6x higher
    than the same bottle as a PNG, and rejected all of them. Compressing both the target
    and the context of a training pair makes "compressed normal" part of normal."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "JPEG", quality=int(quality))
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def patch_grid_side(size: int) -> int:
    return size // PATCH
