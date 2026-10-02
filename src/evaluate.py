"""Evaluate both models on split B - the images neither model ever learned from.

    python -m src.evaluate

Runs on CPU by default: the brief asks for CPU inference times, and CPU numbers
are what a marker can reproduce on any machine.

Writes to results/:
    metrics.csv            one row per model (all numbers below)
    scores_split_b.csv     per-image scores and verdicts for both models
    roc.png                ROC curves of both models
    score_hist_ae.png      score distributions (good vs defect) with the threshold
    score_hist_cnn.png
    confusion_matrices.png
    heatmaps/              6 examples per model, correct and incorrect
    SUMMARY.md             the handoff for the report writers
"""
import argparse
import csv
import platform
import sys
import time
from datetime import date
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.common import SEED, get_device, set_seed
from src.data import (AE_SIZE, DEFAULT_ROOT, DEFAULT_SPLIT, DEFECT_TYPES, check_layout, load_image,
                      load_mask, load_split)
from src.inference import inspect, load_model
from src.visualize import plt, save_figure, save_verdict_figure

MODELS = {"ae": "Convolutional autoencoder", "cnn": "ResNet-18 classifier"}
N_BOOT = 1000


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score_items(model, ckpt, root: Path, items: list[dict], device) -> dict:
    """Score every image once, timing each call (preprocessing + model + score)."""
    warm_up = load_image(root / items[0]["path"])
    for _ in range(3):  # the first calls pay one-off start-up costs; don't time those
        inspect(model, ckpt, warm_up, device, with_heatmap=False)
    scores, times, maps = [], [], []
    for item in items:
        image = load_image(root / item["path"])  # file loading is not part of the timing
        start = time.perf_counter()
        result = inspect(model, ckpt, image, device, with_heatmap=False)
        times.append((time.perf_counter() - start) * 1000)
        scores.append(result["score"])
        maps.append(result["heatmap"])  # AE error map (needed for its score); None for the CNN
    return {"scores": np.array(scores), "ms": np.array(times), "maps": maps}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Confusion-matrix counts and rates for the rule REJECT if score >= threshold."""
    reject = scores >= threshold
    tp = int(np.sum(reject & (labels == 1)))   # defect, rejected        (correct)
    fp = int(np.sum(reject & (labels == 0)))   # good, rejected          (false alarm)
    fn = int(np.sum(~reject & (labels == 1)))  # defect, passed as OK    (missed defect)
    tn = int(np.sum(~reject & (labels == 0)))  # good, passed as OK      (correct)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": f1, "specificity": specificity, "accuracy": (tp + tn) / len(labels)}


def bootstrap_ci(scores, labels, threshold, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """95% intervals: resample split B with replacement n_boot times, take the 2.5/97.5 percentiles.

    Split B has only 42 images (10 good), so every number is uncertain; the
    interval shows how much one more or one fewer mistake would move it.
    """
    rng = np.random.default_rng(seed)
    samples = {"auroc": [], "precision": [], "recall": [], "f1": [], "specificity": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        s, y = scores[idx], labels[idx]
        if y.min() == y.max():  # AUROC needs both good and defect images in the resample
            continue
        samples["auroc"].append(roc_auc_score(y, s))
        m = threshold_metrics(s, y, threshold)
        for key in ("precision", "recall", "f1", "specificity"):
            samples[key].append(m[key])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()}


def recall_per_type(scores, types, threshold) -> dict:
    """Share of each defect type that was rejected."""
    return {t: float(np.mean(scores[types == t] >= threshold)) for t in DEFECT_TYPES}


def pixel_auroc(maps, items, root) -> float:
    """How well the AE error map ranks defect pixels above normal pixels (128x128 masks)."""
    masks = [load_mask(root, item, AE_SIZE) for item in items]
    truth = np.concatenate([m.ravel() for m in masks])
    predicted = np.concatenate([p.ravel() for p in maps])
    return float(roc_auc_score(truth, predicted))


def evaluate_model(kind, scored, labels, types, threshold, items, root) -> dict:
    scores = scored["scores"]
    m = threshold_metrics(scores, labels, threshold)
    m.update({"auroc": float(roc_auc_score(labels, scores)), "threshold": threshold,
              "ci": bootstrap_ci(scores, labels, threshold), "per_type": recall_per_type(scores, types, threshold),
              "ms_per_image": float(scored["ms"].mean()),
              "pixel_auroc": pixel_auroc(scored["maps"], items, root) if kind == "ae" else None})
    return m


# ----------------------------------------------------------------------------
# Example selection for heatmaps
# ----------------------------------------------------------------------------
def pick_examples(scores, labels, types, threshold, n: int = 6) -> list[int]:
    """Up to 3 mistakes, alternating missed defects and false alarms (most confident
    first), then one clear catch per defect type (highest score), then the clearest
    good bottles (lowest score)."""
    reject = scores >= threshold
    by_confidence = lambda idx: sorted(idx, key=lambda i: -abs(scores[i] - threshold))  # noqa: E731
    missed = by_confidence(np.flatnonzero(~reject & (labels == 1)))
    false_alarms = by_confidence(np.flatnonzero(reject & (labels == 0)))
    wrong = [i for pair in zip_longest(missed, false_alarms) for i in pair if i is not None][:3]
    caught = [max(np.flatnonzero(reject & (types == t)), key=lambda i: scores[i], default=None)
              for t in DEFECT_TYPES]
    passed_good = sorted(np.flatnonzero(~reject & (labels == 0)), key=lambda i: scores[i])
    chosen = []
    for i in list(wrong) + [c for c in caught if c is not None] + list(passed_good):
        if int(i) not in chosen:
            chosen.append(int(i))
    return chosen[:n]


def outcome(label: int, verdict: str) -> str:
    if label == 1:
        return "correct-reject" if verdict == "REJECT" else "missed-defect"
    return "correct-ok" if verdict == "OK" else "false-alarm"


def save_heatmap_examples(kind, model, ckpt, root, items, scores, labels, types, out_dir, device):
    for n, i in enumerate(pick_examples(scores, labels, types, ckpt["threshold"]), start=1):
        item = items[i]
        image = load_image(root / item["path"])
        result = inspect(model, ckpt, image, device, with_heatmap=True)
        what = outcome(item["label"], result["verdict"])
        name = f"{kind}_{n}_{what}_{item['type']}_{Path(item['path']).stem}.png"
        save_verdict_figure(image, result, kind, out_dir / name,
                            f"{MODELS[kind]} | {item['path']} | {what}", truth=item["type"])


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_roc(labels, results: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for kind, r in results.items():
        fpr, tpr, _ = roc_curve(labels, r["scores"])
        ax.plot(fpr, tpr, label=f"{MODELS[kind]} (AUROC {r['metrics']['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="random guess")
    ax.set(xlabel="false-alarm rate (good bottles rejected)", ylabel="recall (defects caught)",
           title="ROC on split B (42 held-out images)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_score_hist(kind, scores, labels, threshold, out_path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(scores.min(), threshold), max(scores.max(), threshold), 25)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, color="tab:green", label="good")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, color="tab:red", label="defect")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold {threshold:.4g}")
    xlabel = "anomaly score (max smoothed rebuild error)" if kind == "ae" else "P(defect)"
    ax.set(title=f"{MODELS[kind]}: scores on split B", xlabel=xlabel, ylabel="images")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_confusion(results: dict, out_path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(9, 4))
    for ax, (kind, r) in zip(axes, results.items()):
        m = r["metrics"]
        grid = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(grid, cmap="Blues")
        for (row, col), value in np.ndenumerate(grid):
            ax.text(col, row, str(value), ha="center", va="center", fontsize=16,
                    color="white" if value > grid.max() / 2 else "black")
        ax.set_xticks([0, 1], ["OK", "REJECT"])
        ax.set_yticks([0, 1], ["good", "defect"])
        ax.set(xlabel="predicted", ylabel="truth", title=MODELS[kind])
    fig.tight_layout()
    save_figure(fig, out_path)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def metrics_row(kind: str, m: dict, config: dict) -> dict:
    row = {"model": kind, "technique": MODELS[kind],
           "defect_images_used_in_training": config["n_defect_images_used"],
           "threshold": m["threshold"], "auroc": m["auroc"],
           "auroc_ci_low": m["ci"]["auroc"][0], "auroc_ci_high": m["ci"]["auroc"][1]}
    for key in ("precision", "recall", "f1", "specificity"):
        row[key] = m[key]
        row[f"{key}_ci_low"], row[f"{key}_ci_high"] = m["ci"][key]
    row.update({"accuracy": m["accuracy"], "tp": m["tp"], "fn": m["fn"], "fp": m["fp"], "tn": m["tn"]})
    row.update({f"recall_{t}": m["per_type"][t] for t in DEFECT_TYPES})
    row.update({"ms_per_image_cpu": m["ms_per_image"],
                "pixel_auroc": "" if m["pixel_auroc"] is None else m["pixel_auroc"]})
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# SUMMARY.md
# ----------------------------------------------------------------------------
def fmt_ci(m: dict, key: str, digits: int = 3) -> str:
    low, high = m["ci"][key]
    return f"{m[key]:.{digits}f} [{low:.{digits}f}–{high:.{digits}f}]"


def fmt_value(value) -> str:
    """Hyperparameter table cell: floats to 6 significant digits, everything else as is."""
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def takeaway(kind: str, m: dict, config: dict) -> str:
    n_def, n_good = m["tp"] + m["fn"], m["tn"] + m["fp"]
    weakest = min(m["per_type"], key=m["per_type"].get)
    weak = (f"weakest defect type: {weakest} ({m['per_type'][weakest]:.0%} caught)"
            if m["per_type"][weakest] < 1 else "caught 100% of every defect type")
    if kind == "ae":
        trained = f"Learned from {config['n_train']} defect-free images only (0 defect images)"
        rule = "at its split-A threshold"
    else:
        trained = f"Trained with {config['n_defect_images_used']} labelled defect images"
        rule = "at threshold 0.5"
    return (f"{trained}. AUROC {m['auroc']:.3f}; {rule} it rejected {m['tp']}/{n_def} defective "
            f"and {m['fp']}/{n_good} good bottles; {weak}.")


def write_summary(path: Path, split: dict, results: dict, configs: dict) -> None:
    ae, cnn = results["ae"]["metrics"], results["cnn"]["metrics"]
    n_b = len(split["split_b"])
    n_b_good = sum(i["label"] == 0 for i in split["split_b"])
    rows = [
        ("Defect images used in training", "0", str(configs["cnn"]["n_defect_images_used"])),
        ("Image AUROC [95% CI]", fmt_ci(ae, "auroc"), fmt_ci(cnn, "auroc")),
        ("Threshold (REJECT if score ≥)", f"{ae['threshold']:.5f} (best F1 on split A)", "0.5 (fixed)"),
        ("Precision [95% CI]", fmt_ci(ae, "precision"), fmt_ci(cnn, "precision")),
        ("Recall (defects caught) [95% CI]", fmt_ci(ae, "recall"), fmt_ci(cnn, "recall")),
        ("F1 [95% CI]", fmt_ci(ae, "f1"), fmt_ci(cnn, "f1")),
        ("Specificity (good bottles passed) [95% CI]", fmt_ci(ae, "specificity"), fmt_ci(cnn, "specificity")),
        ("Accuracy", f"{ae['accuracy']:.3f}", f"{cnn['accuracy']:.3f}"),
        ("Confusion matrix TP / FN / FP / TN",
         f"{ae['tp']} / {ae['fn']} / {ae['fp']} / {ae['tn']}", f"{cnn['tp']} / {cnn['fn']} / {cnn['fp']} / {cnn['tn']}"),
    ]
    rows += [(f"Recall: {t}", f"{ae['per_type'][t]:.0%}", f"{cnn['per_type'][t]:.0%}") for t in DEFECT_TYPES]
    rows += [
        ("Mean CPU time per image (ms)", f"{ae['ms_per_image']:.1f}", f"{cnn['ms_per_image']:.1f}"),
        ("Pixel-level AUROC (defect localisation)", f"{ae['pixel_auroc']:.3f}", "n/a (Grad-CAM is not a pixel mask)"),
    ]
    table = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)

    lines = [
        "# LineGuard prototype: results summary",
        "",
        f"Generated by `python -m src.evaluate` on {date.today().isoformat()}. "
        f"All numbers are from **split B only**: {n_b} held-out images ({n_b_good} good, {n_b - n_b_good} defect) "
        "that neither model learned from.",
        "",
        "## Data",
        "",
        "| Part | Images | Used for |",
        "|---|---|---|",
        f"| train | {len(split['train'])} good | AE training; CNN training |",
        f"| val | {len(split['val'])} good | AE early stopping; CNN monitoring only |",
        f"| split A | {len(split['split_a'])} ({sum(i['label'] == 0 for i in split['split_a'])} good, "
        f"{sum(i['label'] for i in split['split_a'])} defect) | CNN training; AE threshold only |",
        f"| split B | {n_b} ({n_b_good} good, {n_b - n_b_good} defect) | evaluation of both models, nothing else |",
        "",
        "Dataset: MVTec AD, category `bottle` (CC BY-NC-SA 4.0). Split: `splits/bottle_split.json`, seed "
        f"{split['seed']}, stratified 50/50 by folder.",
        "",
        "## Results on split B",
        "",
        "| Metric | Autoencoder (technique 1) | ResNet-18 (technique 2) |",
        "|---|---|---|",
        table,
        "",
        f"95% CI = bootstrap percentile interval ({N_BOOT} resamples of split B). With only {n_b_good} good images, "
        f"one false alarm moves specificity by {100 / n_b_good:.0f} percentage points.",
        "",
        "## Takeaways",
        "",
        f"- **Autoencoder:** {takeaway('ae', ae, configs['ae'])}",
        f"- **ResNet-18:** {takeaway('cnn', cnn, configs['cnn'])}",
        "",
        "## Hyperparameters",
        "",
        "| Autoencoder | Value |",
        "|---|---|",
        *[f"| {k} | {fmt_value(v)} |" for k, v in configs["ae"].items()],
        "",
        "| ResNet-18 | Value |",
        "|---|---|",
        *[f"| {k} | {fmt_value(v)} |" for k, v in configs["cnn"].items()],
        "",
        "## Files",
        "",
        "- `metrics.csv`: every number above, one row per model",
        "- `scores_split_b.csv`: per-image scores and verdicts",
        "- `roc.png`, `score_hist_ae.png`, `score_hist_cnn.png`, `confusion_matrices.png`",
        "- `heatmaps/`: 6 examples per model (file name says correct-reject / correct-ok / missed-defect / false-alarm)",
        "- `ae_training_curve.png`, `cnn_training_curve.png`, `logs/`: training history",
        "",
        "## Environment",
        "",
        f"- Evaluated on: CPU ({platform.processor() or platform.machine()}), {torch.get_num_threads()} threads",
        f"- Python {sys.version.split()[0]}, PyTorch {torch.__version__}, {platform.system()} {platform.release()}",
        "- Timing = preprocessing + model + scoring for one image; excludes reading the file and Grad-CAM.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate both models on split B.")
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--weights", default="weights")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="cpu", help="cpu (default, for CPU timings) or cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = get_device(args.device)
    root, out = Path(args.data_root), Path(args.out)
    check_layout(root)
    split = load_split(args.split)
    items = split["split_b"]
    labels = np.array([i["label"] for i in items])
    types = np.array([i["type"] for i in items])
    print(f"Evaluating on split B: {len(items)} images ({int((labels == 0).sum())} good, "
          f"{int(labels.sum())} defect), device {device}")

    for old in (out / "heatmaps").glob("*.png"):  # examples can change between runs
        old.unlink()
    results, configs, per_image = {}, {}, {i["path"]: dict(i) for i in items}
    for kind in MODELS:
        model, ckpt = load_model(Path(args.weights) / f"{kind}.pt", device)
        scored = score_items(model, ckpt, root, items, device)
        metrics = evaluate_model(kind, scored, labels, types, ckpt["threshold"], items, root)
        results[kind] = {"scores": scored["scores"], "metrics": metrics}
        configs[kind] = ckpt["config"]
        for item, score in zip(items, scored["scores"]):
            per_image[item["path"]][f"{kind}_score"] = float(score)
            per_image[item["path"]][f"{kind}_verdict"] = "REJECT" if score >= ckpt["threshold"] else "OK"
        save_heatmap_examples(kind, model, ckpt, root, items, scored["scores"], labels, types,
                              out / "heatmaps", device)
        plot_score_hist(kind, scored["scores"], labels, ckpt["threshold"], out / f"score_hist_{kind}.png")
        print(f"{MODELS[kind]}: AUROC {metrics['auroc']:.3f}, precision {metrics['precision']:.3f}, "
              f"recall {metrics['recall']:.3f}, F1 {metrics['f1']:.3f}, {metrics['ms_per_image']:.1f} ms/image")

    out.mkdir(parents=True, exist_ok=True)
    write_csv([metrics_row(k, results[k]["metrics"], configs[k]) for k in MODELS], out / "metrics.csv")
    write_csv(list(per_image.values()), out / "scores_split_b.csv")
    plot_roc(labels, results, out / "roc.png")
    plot_confusion(results, out / "confusion_matrices.png")
    write_summary(out / "SUMMARY.md", split, results, configs)
    print(f"Results written to {out}/ (start with SUMMARY.md)")


if __name__ == "__main__":
    main()
