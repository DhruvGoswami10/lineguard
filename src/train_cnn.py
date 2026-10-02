"""Train the supervised ResNet-18 classifier (technique 2): good vs defect.

    python -m src.train_cnn

Training images = the 188 'train' images (good) + all of split A (10 good, 31 defect).
The 21 'val' images (good) are only watched, to show how many unseen good bottles
it would wrongly reject; they are not used for any decision.

Why class-weighted loss: there are ~6x more good images than defect images, so an
unweighted model could look accurate by calling everything "good". Weighting each
class by inverse frequency makes the 31 defect images count as much as the 198 good ones.

The threshold is fixed at 0.5: split A is training data for this model, so a
threshold tuned on it would look perfect and mean nothing.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common import SEED, get_device, set_seed
from src.data import DEFAULT_ROOT, DEFAULT_SPLIT, MVTecImages, check_layout, cnn_transform, load_split
from src.inference import save_checkpoint
from src.models import build_resnet18
from src.visualize import plot_curves

AUGMENTATION = "horizontal + vertical flip, rotation +-15 deg, colour jitter 0.1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the ResNet-18 classifier (technique 2).")
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--out", default="weights/cnn.pt")
    p.add_argument("--results", default="results")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    return p.parse_args()


def class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse-frequency weights: n_images / (n_classes * n_images_in_class)."""
    counts = np.bincount(labels, minlength=2)
    return torch.tensor(len(labels) / (2 * counts), dtype=torch.float32)


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, float]:
    """One pass over the data; trains when an optimizer is given. Returns (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(x)
            correct += (logits.argmax(dim=1) == y).sum().item()  # argmax == probability >= 0.5
            count += len(x)
    return total_loss / count, correct / count


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    check_layout(args.data_root)
    split = load_split(args.split)

    items = split["train"] + split["split_a"]
    labels = [item["label"] for item in items]
    weights = class_weights(labels)
    n_defect = sum(labels)
    print(f"Training ResNet-18 on {len(items)} images ({len(items) - n_defect} good, {n_defect} defect) "
          f"on {device}. Class weights good/defect: {weights[0]:.3f}/{weights[1]:.3f}")

    shuffle_rng = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(MVTecImages(args.data_root, items, cnn_transform(train=True)),
                              batch_size=args.batch_size, shuffle=True, generator=shuffle_rng)
    val_loader = DataLoader(MVTecImages(args.data_root, split["val"], cnn_transform(train=False)),
                            batch_size=args.batch_size)

    model = build_resnet18(pretrained=True).to(device)  # downloads ImageNet weights once
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train loss": [], "validation loss (good only)": []}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        history["train loss"].append(train_loss)
        history["validation loss (good only)"].append(val_loss)
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f} acc {train_acc:.3f}  "
              f"val loss {val_loss:.4f}  val good bottles passed {val_acc:.3f}")

    config = {
        "input_size": 224, "architecture": "resnet18", "pretrained": "ImageNet (IMAGENET1K_V1)",
        "loss": "class-weighted cross-entropy", "class_weights": [round(float(w), 4) for w in weights],
        "optimizer": "Adam", "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "augmentation": AUGMENTATION, "n_train_good": len(items) - n_defect,
        "n_defect_images_used": n_defect, "final_train_loss": history["train loss"][-1],
        "final_val_good_pass_rate": val_acc, "threshold_rule": "fixed 0.5",
        "seed": args.seed, "trained_on": str(device), "torch": torch.__version__,
    }
    save_checkpoint(args.out, model, "cnn", 0.5, config)
    plot_curves(history, Path(args.results) / "cnn_training_curve.png",
                "ResNet-18 training", "cross-entropy loss")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
