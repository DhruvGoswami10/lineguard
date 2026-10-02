"""Tests for the two models, the hand-written Grad-CAM and the shared scoring code.

All models here are untrained (random weights) - these tests check shapes,
ranges and wiring, not accuracy. Accuracy is measured by src/evaluate.py.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.common import set_seed
from src.gradcam import GradCAM
from src.inference import (ae_inspect, best_f1_threshold, cnn_inspect, inspect, load_model,
                           save_checkpoint)
from src.models import ConvAutoencoder, build_resnet18

CPU = torch.device("cpu")


def random_image(size=64) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((size, size, 3)) * 255).astype(np.uint8))


class TestAutoencoder(unittest.TestCase):
    def test_output_matches_input_shape_and_range(self):
        model = ConvAutoencoder().eval()
        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(((y >= 0) & (y <= 1)).all())  # sigmoid output, same range as pixels

    def test_bottleneck_is_8x8(self):
        model = ConvAutoencoder().eval()
        with torch.no_grad():
            code = model.encoder(torch.rand(1, 3, 128, 128))
        self.assertEqual(tuple(code.shape[2:]), (8, 8))  # 4 stride-2 blocks: 128 -> 8


class TestResNet(unittest.TestCase):
    def test_two_class_head_builds_offline(self):
        model = build_resnet18(pretrained=False).eval()  # pretrained=False: no download
        with torch.no_grad():
            out = model(torch.rand(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 2))


class TestGradCAM(unittest.TestCase):
    def test_map_shape_and_range(self):
        set_seed(0)
        model = build_resnet18(pretrained=False).eval()
        cam = GradCAM(model, model.layer4)
        heat, logits = cam(torch.rand(1, 3, 224, 224), class_idx=1)
        cam.remove()
        self.assertEqual(heat.shape, (7, 7))  # layer4 grid for a 224x224 input
        self.assertTrue(np.isfinite(heat).all())
        self.assertGreaterEqual(float(heat.min()), 0.0)
        self.assertLessEqual(float(heat.max()), 1.0)
        self.assertEqual(tuple(logits.shape), (1, 2))

    def test_remove_detaches_hook(self):
        model = build_resnet18(pretrained=False).eval()
        GradCAM(model, model.layer4).remove()
        self.assertEqual(len(model.layer4._forward_hooks), 0)

    def test_no_grad_forward_still_works_while_attached(self):
        model = build_resnet18(pretrained=False).eval()
        cam = GradCAM(model, model.layer4)
        with torch.no_grad():
            model(torch.rand(1, 3, 224, 224))  # must not crash on the hook
        cam.remove()


class TestThreshold(unittest.TestCase):
    def test_separable_scores(self):
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        labels = [0, 0, 0, 1, 1, 1]
        self.assertAlmostEqual(best_f1_threshold(scores, labels), 0.7)

    def test_rule_is_score_at_or_above_threshold(self):
        # Best F1 (0.8) comes from rejecting every score >= 0.35, which includes
        # the defect scored exactly 0.35 - so the rule must be ">=".
        scores = [0.1, 0.4, 0.35, 0.9]
        labels = [0, 0, 1, 1]
        self.assertAlmostEqual(best_f1_threshold(scores, labels), 0.35)


class TestInspect(unittest.TestCase):
    def setUp(self):
        set_seed(0)
        self.image = random_image()

    def test_ae_score_is_max_of_heatmap(self):
        score, heat = ae_inspect(ConvAutoencoder().eval(), self.image, sigma=4.0, device=CPU)
        self.assertEqual(heat.shape, (128, 128))
        self.assertAlmostEqual(score, float(heat.max()), places=6)

    def test_cnn_probability_same_with_or_without_heatmap(self):
        model = build_resnet18(pretrained=False).eval()
        p_plain, nothing = cnn_inspect(model, self.image, CPU, with_heatmap=False)
        p_cam, heat = cnn_inspect(model, self.image, CPU, with_heatmap=True)
        self.assertIsNone(nothing)
        self.assertTrue(0.0 <= p_plain <= 1.0)
        self.assertAlmostEqual(p_plain, p_cam, places=5)
        self.assertEqual(heat.shape, (7, 7))

    def test_verdict_follows_threshold(self):
        model = ConvAutoencoder().eval()
        ckpt = {"kind": "ae", "threshold": 0.0, "config": {"sigma": 4.0}}
        self.assertEqual(inspect(model, ckpt, self.image, CPU)["verdict"], "REJECT")
        ckpt["threshold"] = 1e9
        self.assertEqual(inspect(model, ckpt, self.image, CPU)["verdict"], "OK")


class TestCheckpoint(unittest.TestCase):
    def test_round_trip_keeps_weights_and_threshold(self):
        model = ConvAutoencoder().eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ae.pt"
            save_checkpoint(path, model, kind="ae", threshold=0.123, config={"sigma": 4.0})
            loaded, ckpt = load_model(path, CPU)
        self.assertEqual(ckpt["threshold"], 0.123)
        self.assertEqual(ckpt["kind"], "ae")
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x), loaded(x)))

    def test_cnn_round_trip(self):
        model = build_resnet18(pretrained=False).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cnn.pt"
            save_checkpoint(path, model, kind="cnn", threshold=0.5, config={})
            loaded, ckpt = load_model(path, CPU)
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x), loaded(x)))

    def test_missing_file_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            load_model(Path("does/not/exist.pt"), CPU)


if __name__ == "__main__":
    unittest.main()
