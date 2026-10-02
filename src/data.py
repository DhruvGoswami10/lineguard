"""Data loading, transforms and the one fixed split every script uses.

MVTec AD 'bottle' layout expected under data/mvtec/bottle/:
    train/good/                       209 defect-free images
    test/good/                         20 defect-free images
    test/<defect>/                     images with one defect type each
    ground_truth/<defect>/*_mask.png   pixel masks (white = defect)

Usage:
    python -m src.data --check         print image counts, fail loudly if wrong
    python -m src.data --make-split    (re)create splits/bottle_split.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.common import SEED

DEFECT_TYPES = ("broken_large", "broken_small", "contamination")
TEST_FOLDERS = ("good",) + DEFECT_TYPES
EXPECTED_COUNTS = {
    "train/good": 209,
    "test/good": 20,
    "test/broken_large": 20,
    "test/broken_small": 22,
    "test/contamination": 21,
}
AE_SIZE = 128   # autoencoder input size (spec)
CNN_SIZE = 224  # ResNet-18 input size (what ImageNet weights were trained at)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_ROOT = "data/mvtec/bottle"
DEFAULT_SPLIT = "splits/bottle_split.json"


# ----------------------------------------------------------------------------
# Layout check
# ----------------------------------------------------------------------------
def list_images(folder: Path) -> list[str]:
    """Sorted PNG file names in a folder (sorting makes everything downstream repeatable)."""
    return sorted(p.name for p in Path(folder).glob("*.png"))


def mask_path(root: Path, item: dict) -> Path:
    """test/<defect>/007.png  ->  ground_truth/<defect>/007_mask.png"""
    return Path(root) / "ground_truth" / item["type"] / Path(item["path"]).name.replace(".png", "_mask.png")


def check_layout(root) -> dict:
    """Print image counts per folder and raise RuntimeError if the layout is wrong."""
    root = Path(root)
    hint = "Run `python -m src.download_data` to fetch the dataset."
    if not root.is_dir():
        raise RuntimeError(f"Dataset folder not found: {root}\n{hint}")

    counts, problems = {}, []
    for rel, expected in EXPECTED_COUNTS.items():
        counts[rel] = len(list_images(root / rel))
        if counts[rel] != expected:
            problems.append(f"{rel}: found {counts[rel]} images, expected {expected}")
    for defect in DEFECT_TYPES:
        for name in list_images(root / "test" / defect):
            item = {"path": f"test/{defect}/{name}", "type": defect}
            if not mask_path(root, item).is_file():
                problems.append(f"missing mask for test/{defect}/{name}")

    print(f"Dataset: {root}")
    for rel, n in counts.items():
        print(f"  {rel:<20} {n:>4} images")
    if problems:
        raise RuntimeError("MVTec AD layout is wrong:\n  " + "\n  ".join(problems[:10]) + f"\n{hint}")
    return counts


# ----------------------------------------------------------------------------
# The split
# ----------------------------------------------------------------------------
def make_item(folder: str, name: str, parent: str = "test") -> dict:
    """One split entry. label: 0 = good, 1 = defect. type: good or the defect name."""
    return {"path": f"{parent}/{folder}/{name}", "label": int(folder != "good"), "type": folder}


def make_split(root, seed: int = SEED, val_frac: float = 0.1) -> dict:
    """Build the fixed split.

    - train/good -> train (90%) + val (10%, used for autoencoder early stopping).
    - test/      -> split A / split B, 50/50 *within each folder* (stratified),
                    so both halves have the same mix of good bottles and of
                    each defect type. Odd-sized folders give the extra image to B.
    """
    root, rng = Path(root), random.Random(seed)

    good = list_images(root / "train" / "good")
    rng.shuffle(good)
    n_val = round(len(good) * val_frac)
    split = {
        "dataset": "MVTec AD / bottle",
        "seed": seed,
        "val_frac": val_frac,
        "train": [make_item("good", n, "train") for n in sorted(good[n_val:])],
        "val": [make_item("good", n, "train") for n in sorted(good[:n_val])],
        "split_a": [],
        "split_b": [],
    }
    for folder in TEST_FOLDERS:
        names = list_images(root / "test" / folder)
        rng.shuffle(names)
        half = len(names) // 2
        split["split_a"] += [make_item(folder, n) for n in sorted(names[:half])]
        split["split_b"] += [make_item(folder, n) for n in sorted(names[half:])]
    return split


def save_split(split: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")


def load_split(path=DEFAULT_SPLIT) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}. Run `python -m src.data --make-split`.")
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Images, masks, transforms
# ----------------------------------------------------------------------------
def load_image(path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_mask(root, item: dict, size: int) -> np.ndarray:
    """Ground-truth defect mask as a size x size boolean array (all False for good images)."""
    if item["label"] == 0:
        return np.zeros((size, size), dtype=bool)
    with Image.open(mask_path(root, item)) as mask:
        # NEAREST keeps the mask strictly black/white when shrinking it.
        mask = mask.convert("L").resize((size, size), Image.NEAREST)
    return np.array(mask) > 127


def ae_transform() -> T.Compose:
    """128x128, pixels scaled to [0, 1] - the same range as the decoder's sigmoid output."""
    return T.Compose([T.Resize((AE_SIZE, AE_SIZE)), T.ToTensor()])


def cnn_transform(train: bool) -> T.Compose:
    """224x224 with ImageNet normalisation; random augmentation only when training.

    Augmentation stretches 31 defect images further: a flipped or slightly
    rotated bottle is still the same bottle, so the label stays correct.
    """
    steps = [T.Resize((CNN_SIZE, CNN_SIZE))]
    if train:
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


class MVTecImages(Dataset):
    """Yields (image tensor, label, relative path) for a list of split items."""

    def __init__(self, root, items: list[dict], transform):
        self.root, self.items, self.transform = Path(root), items, transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        return self.transform(load_image(self.root / item["path"])), item["label"], item["path"]


def load_tensors(root, items: list[dict], transform) -> torch.Tensor:
    """Decode every image once and stack them (only for fixed, non-random transforms)."""
    return torch.stack([transform(load_image(Path(root) / item["path"])) for item in items])


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Check the dataset or create the fixed split.")
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--check", action="store_true",
                        help="print image counts (always done; with no other option this is all it does)")
    parser.add_argument("--make-split", action="store_true", help="write the split file")
    parser.add_argument("--force", action="store_true", help="overwrite a different existing split")
    args = parser.parse_args()

    check_layout(args.data_root)
    if not args.make_split:
        return
    split = make_split(args.data_root, seed=args.seed)
    path = Path(args.split)
    if path.is_file() and load_split(path) != split and not args.force:
        raise SystemExit(f"{path} exists and differs from a fresh split. Use --force to overwrite.")
    save_split(split, path)
    print(f"Split written to {path}: train {len(split['train'])}, val {len(split['val'])}, "
          f"A {len(split['split_a'])}, B {len(split['split_b'])}")


if __name__ == "__main__":
    main()
