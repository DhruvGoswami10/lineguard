"""Train the convolutional autoencoder on defect-free bottles, then pick its threshold.

    python -m src.train_ae

1. Train on the 188 'train' images (all good) to rebuild their own input (MSE loss).
2. After each epoch, measure the loss on the 21 'val' images (also good). Stop when it
   has not improved for --patience epochs, and keep the best epoch's weights.
3. Score split A and choose the threshold with the best F1. This is the
   autoencoder's ONLY use of split A; it never trains on a defect image.
4. Save weights + threshold to weights/ae.pt and the loss curve to results/.
"""
import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.common import SEED, get_device, set_seed
from src.data import (DEFAULT_ROOT, DEFAULT_SPLIT, ae_transform, check_layout, load_image,
                      load_split, load_tensors)
from src.inference import AE_SIGMA, ae_inspect, best_f1_threshold, save_checkpoint
from src.models import AE_CHANNELS, ConvAutoencoder
from src.visualize import plot_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the autoencoder (technique 1).")
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--out", default="weights/ae.pt")
    p.add_argument("--results", default="results")
    p.add_argument("--epochs", type=int, default=100, help="maximum epochs")
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=AE_SIGMA, help="error-map smoothing")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    return p.parse_args()


def run_epoch(model, loader, device, optimizer=None) -> float:
    """One pass over the data; trains when an optimizer is given. Returns the mean MSE."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for (x,) in loader:
            x = x.to(device)
            loss = F.mse_loss(model(x), x)  # how far the rebuilt image is from the input
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
    return total / count


def train(model, train_loader, val_loader, device, args) -> dict:
    """Train with early stopping. Restores the best epoch's weights into the model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = {"train": [], "validation": []}
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        history["train"].append(train_loss)
        history["validation"].append(val_loss)
        print(f"epoch {epoch:3d}  train MSE {train_loss:.5f}  val MSE {val_loss:.5f}")
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= args.patience:
            print(f"Early stop: validation loss has not improved for {args.patience} epochs.")
            break
    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_mse": best_loss}


def calibrate_threshold(model, root, items, sigma, device) -> float:
    """Score every split-A image and return the threshold with the best F1."""
    scores = [ae_inspect(model, load_image(Path(root) / item["path"]), sigma, device)[0]
              for item in items]
    return best_f1_threshold(scores, [item["label"] for item in items])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    check_layout(args.data_root)
    split = load_split(args.split)

    # The AE uses no augmentation, so each image is decoded and resized once, up front.
    x_train = load_tensors(args.data_root, split["train"], ae_transform())
    x_val = load_tensors(args.data_root, split["val"], ae_transform())
    shuffle_rng = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TensorDataset(x_train), batch_size=args.batch_size,
                              shuffle=True, generator=shuffle_rng)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=args.batch_size)

    model = ConvAutoencoder().to(device)
    print(f"Training autoencoder on {len(x_train)} good images, validating on {len(x_val)} ({device}).")
    result = train(model, train_loader, val_loader, device, args)
    model.eval()

    threshold = calibrate_threshold(model, args.data_root, split["split_a"], args.sigma, device)
    print(f"Kept epoch {result['best_epoch']} (val MSE {result['best_val_mse']:.5f}). "
          f"Threshold (best F1 on split A): {threshold:.5f}")

    config = {
        "input_size": 128, "channels": list(AE_CHANNELS), "loss": "MSE", "optimizer": "Adam",
        "lr": args.lr, "batch_size": args.batch_size, "max_epochs": args.epochs,
        "patience": args.patience, "epochs_run": len(result["history"]["train"]),
        "best_epoch": result["best_epoch"], "best_val_mse": result["best_val_mse"],
        "sigma": args.sigma, "threshold_rule": "best F1 on split A",
        "n_train": len(x_train), "n_val": len(x_val), "n_defect_images_used": 0,
        "seed": args.seed, "trained_on": str(device), "torch": torch.__version__,
    }
    save_checkpoint(args.out, model, "ae", threshold, config)
    plot_curves(result["history"], Path(args.results) / "ae_training_curve.png",
                "Autoencoder training (good images only)", "MSE per pixel", result["best_epoch"])
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
