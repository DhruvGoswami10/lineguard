"""Shared helpers: reproducible seeding and device selection.

Every script calls set_seed() first, so a re-run gives the same weight
initialisation, the same shuffling order and the same augmentations.
"""
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed the Python, NumPy and PyTorch (CPU + GPU) random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN normally benchmarks several convolution algorithms and keeps the
    # fastest, which can change results run to run. Pin it to deterministic ones.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    """Return the requested device. 'auto' means CUDA when available, else CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
