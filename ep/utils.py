"""Shared utilities."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when PyTorch is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
