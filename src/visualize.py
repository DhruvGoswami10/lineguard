"""Figures: training curves, and the image | heatmap | verdict panel used by
both the evaluation (results/heatmaps/) and the demo (results/demo/)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files: works with no display (and on any OS)
import matplotlib.pyplot as plt  # noqa: E402  (must come after choosing the backend)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DISPLAY_SIZE = 448  # pixels per panel


def save_figure(fig, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_curves(curves: dict, out_path, title: str, ylabel: str, best_epoch: int | None = None) -> None:
    """Line plot of per-epoch values, e.g. {"train": [...], "validation": [...]}."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in curves.items():
        ax.plot(range(1, len(values) + 1), values, label=name)
    if best_epoch:
        ax.axvline(best_epoch, color="grey", linestyle="--", label=f"kept epoch ({best_epoch})")
    ax.set(title=title, xlabel="epoch", ylabel=ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_path)


def overlay(image: Image.Image, heatmap: np.ndarray, vmax: float) -> np.ndarray:
    """Blend a jet-coloured heatmap (scaled so vmax = red) over the image."""
    size = (DISPLAY_SIZE, DISPLAY_SIZE)
    base = np.asarray(image.convert("RGB").resize(size, Image.BILINEAR)) / 255.0
    scaled = np.clip(heatmap / vmax, 0, 1).astype(np.float32)
    heat = np.asarray(Image.fromarray(scaled).resize(size, Image.BILINEAR))  # small map -> image size
    return 0.55 * base + 0.45 * plt.get_cmap("jet")(heat)[..., :3]


def save_verdict_figure(image: Image.Image, result: dict, kind: str, out_path, title: str,
                        truth: str | None = None) -> None:
    """Three panels: input image | heatmap overlay | verdict with score and threshold.

    Autoencoder heatmaps are scaled so red means "error at or above the reject
    threshold". Grad-CAM maps are already scaled 0..1 per image (standard Grad-CAM).
    """
    vmax = result["threshold"] if kind == "ae" else 1.0
    heat_title = "Rebuild error (red = over threshold)" if kind == "ae" else "Grad-CAM: evidence for 'defect'"

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    axes[0].imshow(image.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR))
    axes[0].set_title("Input", fontsize=10)
    axes[1].imshow(overlay(image, result["heatmap"], vmax))
    axes[1].set_title(heat_title, fontsize=10)
    for ax in axes:
        ax.axis("off")

    colour = "green" if result["verdict"] == "OK" else "red"
    axes[2].text(0.5, 0.7, result["verdict"], color=colour, fontsize=30, weight="bold",
                 ha="center", va="center")
    fmt = ".4g" if kind == "ae" else ".4f"  # CNN probabilities: fixed decimals, so 0.9999 never shows as "1"
    lines = [f"score     {result['score']:{fmt}}", f"threshold {result['threshold']:{fmt}}"]
    if truth:
        lines.append(f"truth     {truth}")
    axes[2].text(0.5, 0.32, "\n".join(lines), family="monospace", fontsize=10, ha="center", va="center")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_path)
