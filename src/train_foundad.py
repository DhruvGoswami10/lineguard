"""Train technique 3 (FoundAD-style few-shot detector), then pick its thresholds.

    python -m src.train_foundad                       # 4 good images, 1000 steps
    python -m src.train_foundad --shots 8 --steps 4000
    python -m src.train_foundad --good-dir my_photos/ --out weights_myproduct/foundad.pt   # a new product

1. Pick --shots good images from the 'train' part of the split (seeded). Nothing else
   is ever trained on: the DINOv2 encoder stays frozen and sees no defect image.
2. Each step: lightly augment the images, make a CutPaste copy of each one with a
   synthetic defect pasted on the part, and train the predictor so that the features
   of the pasted image (and, half the time, of the clean image itself) predict the
   features of the clean image. MSE loss, AdamW, as in FoundAD. Half the pairs also go
   through a JPEG round trip, so camera / video codec artefacts count as normal
   (src/foundad.py, jpeg_round_trip: why this matters).
3. Score split A on the CPU and choose the threshold with the best F1, exactly as for
   the autoencoder (split A is the detector's only use of labelled images). This is the
   PASS line (t_low): the evaluation's binary metrics are computed at this cut.
4. Set the REJECT line (t_high) from good images the model never trained on (validation
   + split-A good): the lowest split-A defect score above all of them, so REJECT means
   "at least as anomalous as a known defect and more than any known good part", and no
   calibration good image would be rejected outright. Parts between the two lines go to
   REVIEW: a person decides instead of the part being passed or scrapped (src/decision.py).
5. Save the predictor + both thresholds to weights/foundad.pt and the loss curve to results/.

--calibrate-only skips the training and recomputes the thresholds for the predictor
already in --out (after adding good images, or after changing the rules above).

Commissioning a new product (--good-dir): the same training on photographs of your own
good parts, with no dataset and no labels. --shots of them train the predictor; the rest
calibrate the thresholds from good scores only: REVIEW from mean + 2 sigma, REJECT from
mean + 3 sigma (refine both in the dashboard once some parts have run). Then run the line
with --weights <that folder>; the DINOv2 file is still found in weights/.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.common import SEED, get_device, set_seed
from src.data import DEFAULT_ROOT, DEFAULT_SPLIT, check_layout, load_image, load_split
from src.download_backbone import BACKBONES, backbone_path
from src.foundad import (COMPRESS_P, COMPRESS_QUALITY, FOREGROUND_RULES, FOUNDAD_SIZE, N_FROM_END, TOP_K,
                         CutPaste, FoundAD, Predictor, foreground_rule, foundad_inspect, foundad_transform,
                         jpeg_round_trip, train_augment)
from src.inference import best_f1_threshold, save_checkpoint
from src.dino import load_dino
from src.visualize import plot_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the FoundAD-style detector (technique 3).")
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--good-dir", default=None,
                   help="folder of good photos of a new product (ignores the dataset and split)")
    p.add_argument("--backbone", default="vits14", choices=sorted(BACKBONES),
                   help="DINOv2 encoder (download first: python -m src.download_backbone)")
    p.add_argument("--weights-dir", default="weights", help="folder holding the DINOv2 file")
    p.add_argument("--out", default="weights/foundad.pt")
    p.add_argument("--results", default="results")
    p.add_argument("--shots", type=int, default=4, help="good images to train on (0 = all of 'train')")
    p.add_argument("--steps", type=int, default=1000, help="optimiser steps")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=FOUNDAD_SIZE, help="input size (multiple of 14)")
    p.add_argument("--predictor-width", type=int, default=384)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=12)
    p.add_argument("--n-from-end", type=int, default=N_FROM_END, help="encoder block to read features from")
    p.add_argument("--top-k", type=int, default=TOP_K, help="patch errors averaged into the image score")
    p.add_argument("--foreground", default="otsu", choices=FOREGROUND_RULES,  # added 2026-11-04 per Dhruv
                   help="which patches are the part: otsu (part on a plain belt) or disc (crops centred "
                        "on the part by a part finder, e.g. the egg line); stored in the checkpoint")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=100,
                   help="write a progress checkpoint (<out>.progress) every N steps")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>.progress if it exists (after an interrupted run)")
    p.add_argument("--calibrate-only", action="store_true",
                   help="skip training: recompute the thresholds of the predictor already in --out")
    return p.parse_args()


def choose_shots(items: list[dict], shots: int, seed: int) -> list[dict]:
    """The few good images to learn from: a seeded random sample of the 'train' part."""
    if shots <= 0 or shots >= len(items):
        return list(items)
    rng = random.Random(seed)
    return sorted(rng.sample(items, shots), key=lambda i: i["path"])


def make_pair(image: Image.Image, augment, cutpaste: CutPaste, rng: random.Random, transform):
    """(target tensor, context tensor): the same augmented image, the context half the
    time with a synthetic defect pasted on. The target is always the clean one. Half the
    pairs then go through a JPEG round trip at one random quality (both images alike), so
    the camera's and the video file's codec artefacts count as normal."""
    clean = np.asarray(augment(image))
    context = cutpaste(clean) if rng.random() < 0.5 else clean
    if rng.random() < COMPRESS_P:
        quality = rng.randint(*COMPRESS_QUALITY)
        clean, context = jpeg_round_trip(clean, quality), jpeg_round_trip(context, quality)
    return transform(Image.fromarray(clean)), transform(Image.fromarray(context))


def train(model: FoundAD, images: list[Image.Image], device, args) -> dict:
    """Train the predictor for args.steps steps. Returns the loss history.

    A progress checkpoint (predictor, optimiser, loss history, step) is written every
    --save-every steps so that a long run that is interrupted can be resumed with
    --resume. A resumed run re-seeds the augmentation stream at the resume step, so
    its result is not bit-identical to an uninterrupted run.
    """
    rng = random.Random(args.seed)
    augment, cutpaste, transform = train_augment(), CutPaste(rng, args.foreground), foundad_transform(args.size)
    dropout = nn.Dropout(0.2)  # on the context features, as FoundAD: a little noise regularises
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, window, start, first_step = [], [], time.perf_counter(), 1
    progress_path = Path(str(args.out) + ".progress")
    if args.resume and progress_path.is_file():
        progress = torch.load(progress_path, map_location=device, weights_only=True)
        model.predictor.load_state_dict(progress["predictor"])
        optimizer.load_state_dict(progress["optimizer"])
        history, first_step = list(progress["history"]), int(progress["step"]) + 1
        rng = random.Random(args.seed + first_step)
        cutpaste = CutPaste(rng, args.foreground)
        torch.manual_seed(args.seed + first_step)
        print(f"Resumed from {progress_path} at step {first_step - 1}.")
    model.predictor.train()
    for step in range(first_step, args.steps + 1):
        batch = rng.sample(images, min(args.batch_size, len(images)))
        pairs = [make_pair(img, augment, cutpaste, rng, transform) for img in batch]
        target = torch.stack([t for t, _ in pairs]).to(device)
        context = torch.stack([c for _, c in pairs]).to(device)
        with torch.no_grad():  # the encoder is frozen: no gradients, no graph
            h_target = model.encoder(target, model.n_from_end)
            h_context = model.encoder(context, model.n_from_end)
        loss = F.mse_loss(model.predictor(dropout(h_context)), h_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        window.append(loss.item())
        if step % args.log_every == 0 or step == args.steps:
            history.append(float(np.mean(window)))
            print(f"step {step:5d}/{args.steps}  loss {history[-1]:.5f}  "
                  f"({(time.perf_counter() - start) / (step - first_step + 1):.2f} s/step)", flush=True)
            window = []
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"predictor": model.predictor.state_dict(), "optimizer": optimizer.state_dict(),
                        "history": history, "step": step}, progress_path)
    progress_path.unlink(missing_ok=True)  # finished: the real checkpoint follows
    model.predictor.eval()
    return {"loss": history}


def scores_for(model, root, items, size, device, foreground: str = "otsu") -> tuple[np.ndarray, list[np.ndarray]]:
    """(image scores, patch-error maps) for a list of split items, under the foreground rule
    the checkpoint will store (so the thresholds come from the scores the line computes)."""
    results = [foundad_inspect(model, load_image(Path(root) / i["path"]), size, device, foreground) for i in items]
    return np.array([s for s, _ in results]), [m for _, m in results]


def reject_line(good_scores: np.ndarray, defect_scores: np.ndarray, threshold: float) -> float:
    """The REJECT line: the lowest defect score above the highest good score, so no
    calibration good image is rejected outright and a REJECT is at least as anomalous as
    a defect seen in calibration; never below the PASS line. (A midpoint between the two,
    like best_f1_threshold, coincides with the PASS line whenever split A is separable and
    leaves no REVIEW band.) With no defect above the highest good image, 10% above it."""
    top_good = float(good_scores.max())
    above = defect_scores[defect_scores > top_good]
    t_high = float(above.min()) if len(above) else top_good * 1.1
    return max(t_high, threshold)


def calibrate_on_split(model, args, split) -> dict:
    """Dataset mode: PASS line (the checkpoint's threshold) by best F1 on split A; REJECT
    line above every good image the model never trained on (validation + split-A good)."""
    cpu = torch.device("cpu")
    a_scores, _ = scores_for(model, args.data_root, split["split_a"], args.size, cpu, args.foreground)
    a_labels = np.array([i["label"] for i in split["split_a"]])
    threshold = best_f1_threshold(a_scores, a_labels)
    flagged = a_scores >= threshold
    tp, fp, fn = int((flagged & (a_labels == 1)).sum()), int((flagged & (a_labels == 0)).sum()), \
        int((~flagged & (a_labels == 1)).sum())
    unseen_good = split["val"] + [i for i in split["split_a"] if i["label"] == 0]
    good_scores, good_maps = scores_for(model, args.data_root, unseen_good, args.size, cpu, args.foreground)
    t_high = reject_line(good_scores, a_scores[a_labels == 1], threshold)
    in_band = int(((good_scores >= threshold) & (good_scores < t_high)).sum())
    print(f"PASS below {threshold:.5f} (best F1 on split A: {2 * tp / (2 * tp + fp + fn):.3f}); "
          f"REJECT from {t_high:.5f} (above all {len(good_scores)} unseen good images; "
          f"{in_band} of them would go to REVIEW)")
    return {"threshold": threshold, "t_high": t_high,
            "map_floor": float(np.median(np.stack(good_maps))),  # typical normal patch error: the heatmap's "blue"
            "threshold_rule": "best F1 on split A (midpoint between neighbouring scores); the PASS line",
            "split_a_f1": 2 * tp / (2 * tp + fp + fn),
            "t_high_rule": f"the lowest split-A defect score above all {len(good_scores)} unseen good images "
                           "(val + split-A good); the REJECT line",
            "n_unseen_good": len(good_scores), "unseen_good_in_review_band": in_band,
            "unseen_good_max": float(good_scores.max())}


def calibrate_on_good_only(model, args, held_out: list[Path], trained: list[Path]) -> dict:
    """New-product mode, no labels: thresholds from the spread of good scores. Uses the
    photos not trained on; falls back to the training photos (optimistic) if none are left."""
    cpu = torch.device("cpu")
    items = [{"path": p.name} for p in (held_out or trained)]
    scores, maps = scores_for(model, held_out[0].parent if held_out else trained[0].parent, items, args.size, cpu,
                              args.foreground)
    mean, std = float(scores.mean()), float(scores.std()) if len(scores) > 1 else float(scores.mean()) * 0.05
    threshold, t_high = mean + 2 * std, mean + 3 * std
    source = f"{len(scores)} held-out good photos" if held_out else f"the {len(scores)} training photos (no held-out ones: optimistic)"
    print(f"Label-free calibration on {source}: REVIEW from {threshold:.5f} (mean + 2 sigma), "
          f"REJECT from {t_high:.5f} (mean + 3 sigma). Refine in the dashboard once parts have run.")
    return {"threshold": threshold, "t_high": t_high, "map_floor": float(np.median(np.stack(maps))),
            "threshold_rule": f"mean + 2 sigma of good scores ({source}); the PASS line", "split_a_f1": None,
            "t_high_rule": f"mean + 3 sigma of good scores ({source}); the REJECT line",
            "n_unseen_good": len(scores)}


CALIBRATION_KEYS = ("t_low", "t_low_rule", "t_high", "t_high_rule", "threshold_rule", "split_a_f1", "map_floor", "map_floor_rule",
                    "n_unseen_good", "unseen_good_in_review_band", "unseen_good_max", "calibrated_on")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    previous = None
    if args.calibrate_only:  # the trained predictor and its settings come from the checkpoint
        previous = torch.load(args.out, map_location="cpu", weights_only=True)
        if previous.get("kind") != "foundad":
            raise SystemExit(f"{args.out} is not a FoundAD checkpoint")
        old = previous["config"]
        args.backbone, args.size, args.n_from_end, args.top_k = \
            old["backbone"], old["input_size"], old["n_from_end"], old["top_k"]
        args.predictor_width, args.predictor_depth, args.predictor_heads = \
            old["predictor_width"], old["predictor_depth"], old["predictor_heads"]
        args.foreground = foreground_rule(old)  # recalibrate with the rule it was trained under
    if args.size % 14:
        raise SystemExit("--size must be a multiple of 14 (the DINOv2 patch size).")
    if args.good_dir:
        photos = sorted(p for p in Path(args.good_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        if len(photos) < 2:
            raise SystemExit(f"--good-dir needs at least 2 photos of good parts (found {len(photos)} in {args.good_dir})")
        image_root, split = Path(args.good_dir), None
        shots = ([{"path": p} for p in previous["config"]["train_images"]] if previous
                 else choose_shots([{"path": p.name} for p in photos], args.shots, args.seed))
        shot_paths = [image_root / i["path"] for i in shots]
        held_out = [p for p in photos if p not in shot_paths]
    else:
        check_layout(args.data_root)
        split = load_split(args.split)
        image_root = Path(args.data_root)
        shots = ([{"path": p} for p in previous["config"]["train_images"]] if previous
                 else choose_shots(split["train"], args.shots, args.seed))

    encoder = load_dino(backbone_path(args.backbone, args.weights_dir), args.backbone, device)
    predictor = Predictor(encoder.dim, args.predictor_width, args.predictor_depth, args.predictor_heads)
    model = FoundAD(encoder, predictor.to(device), args.n_from_end, args.top_k)

    # Resize once; the augmentations work at the model's input size.
    images = [load_image(image_root / i["path"]).resize((args.size, args.size), Image.BILINEAR) for i in shots]
    n_params = sum(p.numel() for p in predictor.parameters())
    if previous:
        model.predictor.load_state_dict(previous["state_dict"])
        model.predictor.eval()
        print(f"Recalibrating the predictor in {args.out} ({n_params:,} parameters, "
              f"trained on {len(images)} good images); no training.")
    else:
        print(f"Training FoundAD predictor ({n_params:,} parameters) on {len(images)} good images, "
              f"DINOv2 {args.backbone} frozen, {args.size} px, {args.steps} steps ({device}).")
        result = train(model, images, device, args)

    # Calibrate on the CPU: evaluation and the demo run there, so the thresholds are
    # chosen from exactly the scores they will compute.
    model.cpu()
    calibration = (calibrate_on_split(model, args, split) if split
                   else calibrate_on_good_only(model, args, held_out, shot_paths))
    threshold = calibration.pop("threshold")
    calibration["map_floor_rule"] = "median patch error over the calibration good images (heatmap scale)"
    calibration["calibrated_on"] = "cpu"

    if previous:  # keep the training record, replace the calibration
        config = {k: v for k, v in previous["config"].items() if k not in CALIBRATION_KEYS}
        config.update(calibration)
        save_checkpoint(args.out, model.predictor, "foundad", threshold, config)
        print(f"Saved {args.out} (thresholds updated, weights unchanged)")
        return

    config = {
        "technique": "FoundAD-style few-shot detector (frozen DINOv2 + trained predictor)",
        "backbone": args.backbone, "backbone_file": BACKBONES[args.backbone]["file"],
        "input_size": args.size, "n_from_end": args.n_from_end, "top_k": args.top_k,
        "foreground": args.foreground,  # which patches score (src/foundad.py, part_mask); added 2026-11-04 per Dhruv
        "predictor_width": args.predictor_width, "predictor_depth": args.predictor_depth,
        "predictor_heads": args.predictor_heads, "predictor_params": n_params,
        "loss": "MSE between predicted and clean patch features", "optimizer": "AdamW",
        "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "steps": args.steps, "final_loss": result["loss"][-1],
        "augmentation": "flip/rot90 (p=0.3), colour/grey/blur (p=0.3); CutPaste blob or scar on 50% of contexts; "
                        f"JPEG round trip q{COMPRESS_QUALITY[0]}-{COMPRESS_QUALITY[1]} on {COMPRESS_P:.0%} of pairs",
        "n_train": len(images), "train_images": [i["path"] for i in shots],
        "train_source": args.good_dir or f"{args.data_root} ({args.split})",
        "n_defect_images_used": 0,
        **calibration,
        "seed": args.seed, "trained_on": str(device), "torch": torch.__version__,
    }
    save_checkpoint(args.out, model.predictor, "foundad", threshold, config)
    plot_curves({"train": result["loss"]}, Path(args.results) / "foundad_training_curve.png",
                f"FoundAD predictor training ({len(images)} good images, CutPaste)",
                f"feature MSE (mean per {args.log_every} steps)", xlabel=f"x {args.log_every} steps")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
