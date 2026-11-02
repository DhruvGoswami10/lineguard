"""A minimal DINOv2 Vision Transformer that loads Meta's official weights.

DINOv2 (Oquab et al., 2024) is a self-supervised "foundation" image encoder:
trained on 142 million unlabelled images, its patch features already tell
apart textures, edges and materials, which is what makes few-shot anomaly
detection possible (technique 3, src/foundad.py).

Written here in ~120 lines instead of depending on the torch.hub repository so
that the demo runs offline, the dependency list stays as it is, and every team
member can explain the model. Only inference is needed: no dropout, no masking,
no training-time code. The layer names match the official checkpoint exactly,
so `load_state_dict(..., strict=True)` is the test that the architecture is right.

Weights: src/download_backbone.py fetches `dinov2_vits14_pretrain.pth` (88 MB)
once, with a SHA-256 check. Licence: Apache 2.0 (facebookresearch/dinov2).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 14  # pixels per patch: a 448x448 image becomes a 32x32 grid of patch tokens

# name -> (embedding dim, blocks, attention heads). Checkpoint file: dinov2_<name>_pretrain.pth
VARIANTS = {
    "vits14": (384, 12, 6),    # 21 M parameters, 88 MB: the default (fast on a CPU)
    "vitb14": (768, 12, 12),   # 86 M parameters, 346 MB: more accurate, ~4x slower
}


class Attention(nn.Module):
    """Multi-head self-attention: every patch token looks at every other one."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)    # one matrix produces queries, keys and values
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)  # softmax(q k^T / sqrt(head dim)) v
        return self.proj(x.transpose(1, 2).reshape(b, n, d))


class LayerScale(nn.Module):
    """Per-channel learned gain on each residual branch (helps deep ViTs train)."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + attention(norm(x)), then x + mlp(norm(x)).

    DINOv2's encoder blocks use LayerScale; FoundAD's small predictor (src/foundad.py)
    reuses this block without it, as in the original implementation.
    """

    def __init__(self, dim: int, heads: int, layer_scale: bool = True):
        super().__init__()
        scale = LayerScale if layer_scale else nn.Identity
        self.norm1, self.attn, self.ls1 = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads), scale(dim)
        self.norm2, self.mlp, self.ls2 = nn.LayerNorm(dim, eps=1e-6), Mlp(dim), scale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """Cut the image into 14x14 patches and project each to a token (one strided conv)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=PATCH, stride=PATCH)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (b, dim, h, w) -> (b, h*w, dim)


class DinoVisionTransformer(nn.Module):
    """DINOv2 ViT-S/14 or ViT-B/14. forward(x) -> patch tokens of block `-n_from_end`."""

    def __init__(self, variant: str = "vits14"):
        super().__init__()
        dim, depth, heads = VARIANTS[variant]
        self.variant, self.dim = variant, dim
        self.patch_embed = PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (518 // PATCH) ** 2, dim))  # trained at 518 px
        self.mask_token = nn.Parameter(torch.zeros(1, dim))  # unused at inference; kept so loading is strict
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def interpolate_pos_encoding(self, h: int, w: int) -> torch.Tensor:
        """Resample the 37x37 position grid to the input's patch grid (e.g. 32x32 for 448 px).

        DINOv2 passes a scale factor with a +0.1 offset instead of a target size (a
        historical workaround for a rounding bug); the same formula is used here so
        that the features match the official implementation bit for bit.
        """
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        m = int(math.sqrt(patch_pos.shape[1]))
        if (h, w) == (m, m):
            return self.pos_embed
        grid = patch_pos.reshape(1, m, m, self.dim).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, scale_factor=((h + 0.1) / m, (w + 0.1) / m), mode="bicubic")
        assert grid.shape[-2:] == (h, w)
        return torch.cat([cls_pos, grid.permute(0, 2, 3, 1).reshape(1, h * w, self.dim)], dim=1)

    def forward(self, x: torch.Tensor, n_from_end: int = 3) -> torch.Tensor:
        """Patch tokens (b, h*w, dim) taken from the block `n_from_end` positions before the
        last one, normalised with the final LayerNorm - exactly what DINOv2's
        get_intermediate_layers(n=n_from_end)[0] returns and what FoundAD scores on.
        An intermediate block keeps more local texture detail than the last block.
        """
        b, _, hp, wp = x.shape
        h, w = hp // PATCH, wp // PATCH
        tokens = torch.cat([self.cls_token.expand(b, -1, -1), self.patch_embed(x)], dim=1)
        tokens = tokens + self.interpolate_pos_encoding(h, w)
        for block in self.blocks[: len(self.blocks) - n_from_end + 1]:
            tokens = block(tokens)
        return self.norm(tokens)[:, 1:]  # drop the class token: one feature vector per patch


def load_dino(path, variant: str = "vits14", device="cpu") -> DinoVisionTransformer:
    """Build the encoder, load the official weights (strictly) and freeze it."""
    model = DinoVisionTransformer(variant)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)  # the encoder is never trained: only FoundAD's predictor is
    return model.to(device).eval()
