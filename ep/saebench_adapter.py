"""SAEBench-compatible featurizer backed by an exemplar-partition dictionary.

SAEBench expects an nn.Module with encode/decode/forward and W_dec/W_enc.

Geometry mirrors `Dictionary.distances`: input is centered by the dictionary's
fixed activation center, then projected onto stored partition directions
(stored as unit directions in centered space). The dictionary itself stores no
magnitudes — only directions. Magnitude is supplied at inference time by the
input via `||x_c||` and threaded through whichever readout the downstream task
wants.

Two basis choices and six readouts are supported.

Bases:
- "exemplar" (Partition.exemplar_direction): the seed activation that spawned
  each partition, stored as a unit direction in centered space. Immutable for
  the life of the dictionary.
- "mean" (Partition.mean_member_direction): the equal-weighted spherical mean
  of all member directions. Each member contributes its unit direction (not
  its raw vector), so high-norm activations do not dominate.

Readouts (selected per evaluation):
- readout="topk", k=1 (default): top-1 / VQ. Encoder emits a single nonzero
  feature: the signed projection magnitude onto the assigned exemplar (ReLU'd).
  Matches the partition geometry — every activation belongs to exactly one
  cell. Reconstruction is angle-bounded; magnitude is preserved (the input
  supplies its own ||x_c||); exemplars are fixed points of encode∘decode for
  the exemplar basis. Use for SAE-style evals (core, scr, tpp, autointerp).
- readout="topk", k>1: keep the k largest signed projections, ReLU them, zero
  the rest. Loses the exemplar fixed-point property and re-introduces some
  frame-redundancy overshoot at decode, in exchange for a denser feature
  representation. Reserved as a knob for ablation.
- readout="signed": all signed projections, no ReLU, no masking. Equivalent
  to a length-K linear readout `x_c @ W_enc`. The "dense distance vector"
  framing — every input gets its alignment with every exemplar. Use for
  sparse_probing, where the probe benefits from alignment information across
  all partitions, not just the assigned one.
- readout="cosine": pure cosine similarity, magnitude removed. Equivalent to
  `(x_c / ||x_c||) @ W_enc`, returning values in [-1, 1] per direction. Use
  when averaging alignment across inputs of varying norm (concept selection,
  distance-to-cover signals) — high-norm tokens cannot dominate the average.
- readout="signed_norm": magnitude-removed signed projections. Same as
  `signed` but using the unit-normalised `x_c`, so values are in [-1, 1].
- readout="topk_norm": magnitude-removed top-k. Same as `topk` but on the
  unit-normalised `x_c`, so values are in [0, 1].
- readout="binary": single-hot assignment (requires k=1). Encoder emits a
  single 1.0 at the nearest-cosine partition; everything else is zero. Use
  when the downstream task cares only about cell identity, not magnitude.

Decode is symmetric: `feature_acts @ W_dec + center`. Under topk / topk_norm
/ binary, only the selected k entries contribute. Under signed / signed_norm
/ cosine, all K do (which under our non-orthonormal directions overshoots —
the dense readouts are intended for downstream consumers like probes, not
for round-trip reconstruction).
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import sae_bench.custom_saes.base_sae as base_sae
except ModuleNotFoundError:
    class _FallbackBaseSAE(torch.nn.Module):

        def __init__(self, d_in, d_sae, model_name, hook_layer, device, dtype, hook_name=None):
            super().__init__()
            self.d_in = d_in
            self.d_sae = d_sae
            self.model_name = model_name
            self.hook_layer = hook_layer
            self.hook_name = hook_name
            self.W_dec = torch.nn.Parameter(torch.empty(d_sae, d_in, device=device, dtype=dtype))
            self.W_enc = torch.nn.Parameter(torch.empty(d_in, d_sae, device=device, dtype=dtype))

    class base_sae:  # type: ignore[no-redef]
        BaseSAE = _FallbackBaseSAE

from ep.discovery import Dictionary


class EPDictionarySAE(base_sae.BaseSAE):
    """SAEBench-compatible featurizer backed by an exemplar-partition dictionary."""

    BASES = ("mean", "exemplar")
    READOUTS = ("topk", "signed", "cosine", "signed_norm", "topk_norm", "binary")

    def __init__(
        self,
        dictionary: Dictionary,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
        basis: str = "mean",
        readout: str = "topk",
        k: int = 1,
    ):
        if not dictionary.partitions:
            raise ValueError("EPDictionarySAE requires a non-empty dictionary")
        if basis not in self.BASES:
            raise ValueError(f"basis must be one of {self.BASES}, got {basis!r}")
        if readout not in self.READOUTS:
            raise ValueError(f"readout must be one of {self.READOUTS}, got {readout!r}")
        if readout in ("topk", "topk_norm") and k < 1:
            raise ValueError(f"k must be >= 1 for readout={readout!r}, got {k}")
        if readout == "binary" and k != 1:
            raise ValueError(f"binary readout requires k=1, got {k}")

        self._dictionary = dictionary
        self.basis = basis
        self.readout = readout
        self.k = k

        d_in = dictionary.center.shape[0]
        d_sae = len(dictionary.partitions)

        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        center = torch.from_numpy(dictionary.center.astype(np.float32)).to(
            device=device, dtype=dtype,
        )
        self.register_buffer("act_mean", center)

        attr = "mean_member_direction" if basis == "mean" else "exemplar_direction"
        # Both already stored as unit directions in centered space.
        unit = np.stack(
            [getattr(p, attr) for p in dictionary.partitions]
        ).astype(np.float32)

        W = torch.from_numpy(unit).to(device=device, dtype=dtype)
        self.W_dec.data = W
        self.W_enc.data = W.t().contiguous()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.act_mean
        if self.readout == "cosine":
            x_unit = torch.nn.functional.normalize(x_c, dim=-1, eps=1e-12)
            return x_unit @ self.W_enc
        if self.readout == "signed_norm":
            x_unit = torch.nn.functional.normalize(x_c, dim=-1, eps=1e-12)
            return x_unit @ self.W_enc
        if self.readout == "topk_norm":
            x_unit = torch.nn.functional.normalize(x_c, dim=-1, eps=1e-12)
            proj = x_unit @ self.W_enc
            k = min(self.k, proj.shape[-1])
            topk_vals, topk_idx = proj.topk(k, dim=-1)
            topk_vals = torch.relu(topk_vals)
            out = torch.zeros_like(proj)
            out.scatter_(-1, topk_idx, topk_vals)
            return out
        if self.readout == "binary":
            x_unit = torch.nn.functional.normalize(x_c, dim=-1, eps=1e-12)
            proj = x_unit @ self.W_enc
            _, top_idx = proj.topk(1, dim=-1)
            out = torch.zeros_like(proj)
            out.scatter_(-1, top_idx, torch.ones_like(top_idx, dtype=proj.dtype))
            return out
        proj = x_c @ self.W_enc
        if self.readout == "signed":
            return proj
        k = min(self.k, proj.shape[-1])
        topk_vals, topk_idx = proj.topk(k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        out = torch.zeros_like(proj)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.act_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
