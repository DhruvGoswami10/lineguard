"""Technique 3 tests: the minimal DINOv2 encoder (shapes, strict weight loading), the
predictor, the top-K score, CutPaste synthesis and the Otsu foreground rule.
The strict-loading test needs weights/dinov2_vits14_pretrain.pth and is skipped without it."""
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dino import PATCH, VARIANTS, DinoVisionTransformer, load_dino
from src.download_backbone import BACKBONES, backbone_path
from src.foundad import (CutPaste, FoundAD, Predictor, foreground_mask, foundad_transform, otsu_threshold,
                         patch_foreground, score_from_errors, upsample_map)

REPO = Path(__file__).resolve().parents[1]
BACKBONE = REPO / backbone_path("vits14")


class TestDino(unittest.TestCase):
    def test_parameter_names_match_the_official_checkpoint_layout(self):
        names = {n for n, _ in DinoVisionTransformer("vits14").named_parameters()}
        for expected in ("cls_token", "pos_embed", "mask_token", "patch_embed.proj.weight", "blocks.0.attn.qkv.weight",
                         "blocks.0.ls1.gamma", "blocks.11.mlp.fc2.bias", "norm.weight"):
            self.assertIn(expected, names)
        self.assertEqual(sum(p.numel() for p in DinoVisionTransformer("vits14").parameters()), 22_056_576)

    def test_patch_grid_and_feature_shape(self):
        torch.manual_seed(0)
        model = DinoVisionTransformer("vits14").eval()
        for size in (224, 448):
            with torch.no_grad():
                feats = model(torch.randn(2, 3, size, size))
            self.assertEqual(tuple(feats.shape), (2, (size // PATCH) ** 2, VARIANTS["vits14"][0]))

    def test_position_grid_is_untouched_at_the_training_size(self):
        model = DinoVisionTransformer("vits14")
        self.assertIs(model.interpolate_pos_encoding(37, 37), model.pos_embed)   # 518 / 14 = 37
        self.assertEqual(tuple(model.interpolate_pos_encoding(32, 32).shape), (1, 1 + 32 * 32, 384))

    def test_intermediate_block_differs_from_last(self):
        torch.manual_seed(0)
        model = DinoVisionTransformer("vits14").eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            self.assertFalse(torch.allclose(model(x, n_from_end=3), model(x, n_from_end=1)))

    @unittest.skipUnless(BACKBONE.exists(), "run `python -m src.download_backbone` first")
    def test_official_weights_load_strictly_and_are_frozen(self):
        model = load_dino(BACKBONE, "vits14")
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertEqual(BACKBONE.stat().st_size, BACKBONES["vits14"]["bytes"])


class TestPredictorAndScore(unittest.TestCase):
    def test_predictor_keeps_the_feature_shape(self):
        torch.manual_seed(0)
        predictor = Predictor(dim=384, width=64, depth=2, heads=4)
        out = predictor(torch.randn(2, 16, 384))
        self.assertEqual(tuple(out.shape), (2, 16, 384))

    def test_only_the_predictor_is_trainable(self):
        torch.manual_seed(0)
        model = FoundAD(DinoVisionTransformer("vits14"), Predictor(384, 64, 2, 4))
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable and all(n.startswith("predictor.") for n in trainable))

    def test_score_is_mean_of_top_k_and_map_is_the_grid(self):
        errors = torch.tensor([[1.0, 5.0, 2.0, 9.0]])     # a 2x2 grid
        score, grid = score_from_errors(errors, top_k=2)
        self.assertAlmostEqual(score, 7.0)                 # (9 + 5) / 2
        self.assertEqual(grid.shape, (2, 2))
        self.assertEqual(grid[1, 1], 9.0)
        self.assertAlmostEqual(score_from_errors(errors, top_k=10)[0], 4.25)  # K capped at the grid size

    def test_only_the_part_can_score(self):
        errors = torch.tensor([[1.0, 5.0, 2.0, 9.0]])
        on_part = np.array([[True, True], [True, False]])  # the 9 is on the belt
        score, grid = score_from_errors(errors, top_k=2, foreground=on_part)
        self.assertAlmostEqual(score, 3.5)                 # (5 + 2) / 2
        self.assertEqual(grid[1, 1], 0.0)                  # the map shows the belt as clean
        empty = np.zeros((2, 2), bool)                     # no part found: every patch counts
        self.assertAlmostEqual(score_from_errors(errors, top_k=2, foreground=empty)[0], 7.0)

    def test_patch_foreground_follows_the_disc(self):
        grid = patch_foreground(bottle_like(), side=8)     # 128 px image -> 8x8 patches of 16 px
        self.assertEqual(grid.shape, (8, 8))
        self.assertTrue(grid[4, 4] and not grid[0, 0])
        self.assertTrue(10 < grid.sum() < 64)

    def test_upsample_keeps_the_hot_spot(self):
        grid = np.zeros((4, 4), np.float32)
        grid[3, 0] = 1.0
        big = upsample_map(grid, 32)
        self.assertEqual(big.shape, (32, 32))
        row, col = np.unravel_index(big.argmax(), big.shape)
        self.assertTrue(row >= 24 and col < 8)          # inside the bottom-left cell

    def test_transform_normalises_with_imagenet_statistics(self):
        x = foundad_transform(224)(Image.new("RGB", (900, 900), (124, 116, 104)))  # ~ the ImageNet mean colour
        self.assertEqual(tuple(x.shape), (3, 224, 224))
        self.assertLess(x.abs().max().item(), 0.05)


def bottle_like(size: int = 128) -> np.ndarray:
    """A dark disc on white, like a dataset photo (uint8 HxWx3)."""
    yy, xx = np.mgrid[:size, :size]
    disc = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size * 0.42) ** 2
    image = np.full((size, size, 3), 255, np.uint8)
    image[disc] = (40, 30, 30)
    return image


class TestSynthesis(unittest.TestCase):
    def test_otsu_splits_a_bimodal_histogram(self):
        grey = np.concatenate([np.full(500, 30, np.uint8), np.full(500, 220, np.uint8)]).reshape(20, 50)
        t = otsu_threshold(grey)
        self.assertTrue(30 <= t < 220)

    def test_foreground_is_the_class_not_on_the_border(self):
        image = bottle_like()
        mask = foreground_mask(image)
        self.assertTrue(mask[64, 64] and not mask[0, 0])
        inverted = 255 - image                      # bright part on a dark belt
        mask = foreground_mask(inverted)
        self.assertTrue(mask[64, 64] and not mask[0, 0])

    def test_cutpaste_changes_pixels_on_the_part_only(self):
        torch.manual_seed(0)
        rng = random.Random(0)
        image = bottle_like()
        pasted = CutPaste(rng)(image)
        changed = np.any(pasted != image, axis=2)
        self.assertTrue(changed.any())
        self.assertEqual(pasted.shape, image.shape)
        # the pasted piece's top-left corner is on the part, so the change starts inside the disc
        ys, xs = np.nonzero(changed)
        self.assertTrue(foreground_mask(image)[ys.min(), xs.min()])

    def test_blob_and_scar_both_produce_a_change(self):
        torch.manual_seed(0)
        image = bottle_like()
        cp = CutPaste(random.Random(1))
        self.assertTrue(np.any(cp.blob(image) != image))
        self.assertTrue(np.any(cp.scar(image) != image))


if __name__ == "__main__":
    unittest.main()
