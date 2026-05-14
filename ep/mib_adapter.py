"""Adapter that exposes an exemplar-partition dictionary as a MIB Featurizer.

MIB's causal variable localisation track expects:
  - featurizer:         x -> (features, error)
  - inverse_featurizer: (features, error) -> x_reconstructed

Each partition's ``exemplar_direction`` (unit vector in centered space) is
used as a basis direction. The featurizer centers the input by the
dictionary's activation center, projects onto the basis, and returns the
reconstruction residual as ``error`` so the round-trip is lossless.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ep.discovery import Dictionary


class EPFeaturizerModule(nn.Module):
    def __init__(self, exemplars: torch.Tensor, center: torch.Tensor):
        super().__init__()
        self.register_buffer("exemplars", exemplars)
        self.register_buffer("center", center)

    def forward(self, x: torch.Tensor):
        e = self.exemplars.to(x.dtype)
        c = self.center.to(x.dtype)
        x_centered = x - c
        features = x_centered @ e.T
        error = x_centered - features @ e
        return features, error


class EPInverseFeaturizerModule(nn.Module):
    def __init__(self, exemplars: torch.Tensor, center: torch.Tensor):
        super().__init__()
        self.register_buffer("exemplars", exemplars)
        self.register_buffer("center", center)

    def forward(self, features: torch.Tensor, error) -> torch.Tensor:
        e = self.exemplars.to(features.dtype)
        c = self.center.to(features.dtype)
        return features @ e + error.to(features.dtype) + c


def make_featurizer(
    dictionary: Dictionary,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Build a MIB Featurizer from an exemplar-partition dictionary."""
    from CausalAbstraction.neural.featurizers import Featurizer

    if not dictionary.partitions:
        raise ValueError("EP MIB adapter requires a non-empty dictionary")

    raw = np.stack(
        [p.exemplar_direction for p in dictionary.partitions]
    ).astype(np.float32)
    exemplars = torch.from_numpy(raw).to(device=device, dtype=dtype)
    center = torch.from_numpy(dictionary.center.astype(np.float32)).to(
        device=device, dtype=dtype,
    )

    n = exemplars.shape[0]
    fwd = EPFeaturizerModule(exemplars, center)
    inv = EPInverseFeaturizerModule(exemplars, center)

    return Featurizer(fwd, inv, n_features=n, id="ep")
