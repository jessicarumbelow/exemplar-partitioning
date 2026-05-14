"""Shared geometric primitives for centered cosine operations.

Both `Dictionary.distances` (numpy, returns cosine distance) and
`EPDictionarySAE.encode` (torch, returns magnitude-weighted projection)
project inputs against the same centered geometry: subtract the dictionary's
fixed calibration center, then use stored unit directions in that centered
space.
"""

from __future__ import annotations

import numpy as np


def try_torch_gpu():
    """Return (torch, device) if CUDA is available, else (None, None).

    Shared by the distance/similarity paths in `dictionary`, `calibration`,
    and `eval` — gates whether GPU matmul kernels are used.
    """
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    return torch, torch.device("cuda")


def centered_unit(vecs: np.ndarray, mean: np.ndarray | None) -> np.ndarray:
    """Center `vecs` by `mean` (if not None) and L2-normalize each row.

    Args:
        vecs: (..., D) array.
        mean: (D,) array or None. If None, no centering is applied.

    Returns:
        Array with the same shape as `vecs`; each row lies on the unit
        sphere of (optionally) centered space.
    """
    if mean is not None:
        vecs = vecs - mean
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12
    return vecs / norms
