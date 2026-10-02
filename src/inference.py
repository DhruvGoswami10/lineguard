"""One scoring path, shared by threshold calibration, evaluation and the demo.

Keeping it in one place guarantees the demo computes exactly the same score
as the evaluation that produced the reported numbers.

Decision rule for both models: REJECT if score >= threshold, otherwise OK.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve

from src.data import ae_transform, cnn_transform
from src.gradcam import GradCAM
from src.models import ConvAutoencoder, build_resnet18

AE_SIGMA = 4.0  # Gaussian smoothing of the error map, in pixels of the 128x128 map


def ae_inspect(model, image, sigma: float, device) -> tuple[float, np.ndarray]:
    """Autoencoder: score = the highest smoothed rebuild error anywhere in the image.

    Returns (score, error_map) with error_map being 128x128.
    """
    x = ae_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        rebuilt = model(x)
    # Squared error per pixel, averaged over R, G, B -> one 128x128 map.
    error = ((x - rebuilt) ** 2).mean(dim=1)[0].cpu().numpy()
    # Smoothing: a lone noisy pixel fades away, a real defect (a blob of error) stays a peak.
    error_map = gaussian_filter(error, sigma=sigma)
    return float(error_map.max()), error_map


def cnn_inspect(model, image, device, with_heatmap: bool = False):
    """CNN: score = softmax probability of 'defect'. Optional heatmap = Grad-CAM (7x7)."""
    x = cnn_transform(train=False)(image).unsqueeze(0).to(device)
    if not with_heatmap:
        with torch.no_grad():
            logits = model(x)
        return float(torch.softmax(logits, dim=1)[0, 1]), None
    cam = GradCAM(model, model.layer4)
    try:
        heatmap, logits = cam(x, class_idx=1)
    finally:
        cam.remove()
    return float(torch.softmax(logits, dim=1)[0, 1]), heatmap


def inspect(model, ckpt: dict, image, device, with_heatmap: bool = True) -> dict:
    """Score one PIL image with a loaded model; returns score, threshold, verdict, heatmap."""
    if ckpt["kind"] == "ae":
        score, heatmap = ae_inspect(model, image, ckpt["config"]["sigma"], device)
    else:
        score, heatmap = cnn_inspect(model, image, device, with_heatmap)
    threshold = ckpt["threshold"]
    verdict = "REJECT" if score >= threshold else "OK"
    return {"score": score, "threshold": threshold, "verdict": verdict, "heatmap": heatmap}


def best_f1_threshold(scores, labels) -> float:
    """The threshold (REJECT if score >= threshold) with the highest F1 on this data.

    The best cut lies between two neighbouring scores. The threshold is put halfway
    between them rather than on a score itself, so tiny numerical differences
    (e.g. GPU vs CPU arithmetic) cannot flip an image to the other side.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)  # thresholds: sorted unique scores
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))  # the final precision/recall pair has no threshold; skip it
    if best == 0:  # rejecting everything was best: there is no lower score to go halfway to
        return float(thresholds[0])
    return float((thresholds[best - 1] + thresholds[best]) / 2)


def save_checkpoint(path, model, kind: str, threshold: float, config: dict) -> None:
    """Save weights + the threshold the model must be used with, so the demo is self-contained."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}  # loadable without a GPU
    # The safe loader (weights_only=True) only accepts plain types, so round-trip the
    # config through JSON: tuples become lists, library objects become strings.
    config = json.loads(json.dumps(config, default=str))
    torch.save({"kind": kind, "state_dict": state, "threshold": float(threshold), "config": config}, path)


def build_model(kind: str) -> torch.nn.Module:
    if kind == "ae":
        return ConvAutoencoder()
    if kind == "cnn":
        return build_resnet18(pretrained=False)  # our own weights are loaded next: no download
    raise ValueError(f"Unknown model kind: {kind}")


def load_model(path, device) -> tuple[torch.nn.Module, dict]:
    """Rebuild a model from a checkpoint. Returns (model in eval mode, checkpoint dict)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Weights not found: {path}. They ship in weights/ - check the zip was fully "
            "extracted, or retrain with `python -m src.train_ae` / `python -m src.train_cnn`.")
    ckpt = torch.load(path, map_location=device, weights_only=True)  # weights_only: safe loading
    model = build_model(ckpt["kind"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
