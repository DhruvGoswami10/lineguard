"""Tests for the dataset layout check and the fixed split.

A fake MVTec 'bottle' folder with tiny images (same file names as the real
dataset) is built in a temp directory, so these tests run without the
real 157 MB download.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import (DEFECT_TYPES, EXPECTED_COUNTS, check_layout, load_mask, load_split,
                      make_split)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SPLIT = REPO_ROOT / "splits" / "bottle_split.json"


def build_fake_bottle(root: Path) -> None:
    """Create the MVTec bottle layout: 8x8 images named 000.png, 001.png, ... like the real data."""
    for rel, count in EXPECTED_COUNTS.items():
        folder = root / rel
        folder.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (8, 8), (i % 256, 0, 0)).save(folder / f"{i:03d}.png")
    for defect in DEFECT_TYPES:
        masks = root / "ground_truth" / defect
        masks.mkdir(parents=True)
        for i in range(EXPECTED_COUNTS[f"test/{defect}"]):
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:5, 2:5] = 255  # a 3x3 "defect" square
            Image.fromarray(mask).save(masks / f"{i:03d}_mask.png")


class FakeDataTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "bottle"
        build_fake_bottle(self.root)

    def tearDown(self):
        shutil.rmtree(self.tmp)


class TestCheckLayout(FakeDataTestCase):
    def test_valid_layout_returns_counts(self):
        counts = check_layout(self.root)
        self.assertEqual(counts, EXPECTED_COUNTS)

    def test_missing_folder_raises(self):
        shutil.rmtree(self.root / "test" / "contamination")
        with self.assertRaises(RuntimeError):
            check_layout(self.root)

    def test_missing_mask_raises(self):
        (self.root / "ground_truth" / "broken_small" / "000_mask.png").unlink()
        with self.assertRaises(RuntimeError):
            check_layout(self.root)

    def test_missing_root_raises(self):
        with self.assertRaises(RuntimeError):
            check_layout(self.tmp / "does_not_exist")


class TestSplit(FakeDataTestCase):
    def setUp(self):
        super().setUp()
        self.split = make_split(self.root, seed=42)

    def test_sizes(self):
        self.assertEqual(len(self.split["train"]), 188)
        self.assertEqual(len(self.split["val"]), 21)
        self.assertEqual(len(self.split["split_a"]), 41)
        self.assertEqual(len(self.split["split_b"]), 42)

    def test_stratified_by_folder(self):
        def count(part, kind):
            return sum(item["type"] == kind for item in self.split[part])
        expected = {"good": (10, 10), "broken_large": (10, 10),
                    "broken_small": (11, 11), "contamination": (10, 11)}
        for kind, (n_a, n_b) in expected.items():
            self.assertEqual((count("split_a", kind), count("split_b", kind)), (n_a, n_b), kind)

    def test_no_image_in_two_parts(self):
        parts = ["train", "val", "split_a", "split_b"]
        seen = [item["path"] for part in parts for item in self.split[part]]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(seen), sum(EXPECTED_COUNTS.values()))

    def test_labels_match_folders(self):
        for part in ["train", "val", "split_a", "split_b"]:
            for item in self.split[part]:
                self.assertEqual(item["label"], int(item["type"] != "good"))
                self.assertIn(f"/{item['type']}/", item["path"])

    def test_same_seed_same_split(self):
        self.assertEqual(make_split(self.root, seed=42), self.split)

    def test_different_seed_different_split(self):
        self.assertNotEqual(make_split(self.root, seed=7)["split_b"], self.split["split_b"])

    @unittest.skipUnless(COMMITTED_SPLIT.exists(), "split file not created yet")
    def test_committed_split_is_reproducible_from_its_seed(self):
        committed = load_split(COMMITTED_SPLIT)
        self.assertEqual(make_split(self.root, seed=committed["seed"]), committed)


class TestMasks(FakeDataTestCase):
    def test_good_image_mask_is_empty(self):
        item = {"path": "test/good/000.png", "label": 0, "type": "good"}
        mask = load_mask(self.root, item, size=8)
        self.assertEqual(mask.shape, (8, 8))
        self.assertFalse(mask.any())

    def test_defect_mask_is_loaded(self):
        item = {"path": "test/broken_large/003.png", "label": 1, "type": "broken_large"}
        mask = load_mask(self.root, item, size=8)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(int(mask.sum()), 9)


if __name__ == "__main__":
    unittest.main()
