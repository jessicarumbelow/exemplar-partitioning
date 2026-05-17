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
    """Return (torch, device) if a GPU is available, else (None, None).

    CUDA is preferred; Apple Silicon MPS is used as a fallback. Shared by
    the distance/similarity paths in `dictionary`, `calibration`, and
    `eval` — gates whether GPU matmul kernels are used.

    Disable with ``EP_FORCE_CPU=1`` if you suspect a backend issue.
    """
    import os
    if os.environ.get("EP_FORCE_CPU"):
        return None, None
    try:
        import torch
    except ImportError:
        return None, None
    if torch.cuda.is_available():
        return torch, torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch, torch.device("mps")
    return None, None


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
