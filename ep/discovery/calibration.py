"""Pre-compute the activation center and clustering threshold from a fixed
calibration sample, then cache them for reuse across runs.

Both quantities used to be derived as streaming approximations during
dictionary construction (running mean of activations; running average of the
within-batch p-th percentile). Streaming makes them drift early and bake in
the data ordering. Pre-computing once per (model, hook, percentile) and
freezing the values gives every dictionary built on the same activations the
same geometric reference — so runs across seeds, methods, and evals are
apples-to-apples.

The center is direction-first to stay consistent with the rest of the
method (which is fully cosine/direction-based):

    direction = normalize(mean(a / ||a||))      # spherical mean
    magnitude = <mean(a), direction>            # avg projection onto bulk
    center    = direction * magnitude

Reduces to the Euclidean mean when activations have uniform norms, but
robust to per-token norm variation that would otherwise let high-norm
activations skew the centering direction. Matters more on early layers,
smaller models, and unusual architectures.

Cache layout (versioned — bumping ``SCHEMA_VERSION`` invalidates old caches
on load with a warning):
    ~/.cache/ep/calibration/{model}__{hook}__pct{percentile}.npz
each file holds: ``center`` (D,), ``threshold`` (scalar), ``n_activations``,
``percentile``, ``version``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import centered_unit, try_torch_gpu

logger = logging.getLogger(__name__)

# Bumped when the center algorithm changes (loaded caches with a different
# version are ignored, forcing recompute). 1 was streaming/Euclidean mean;
# 2 is the spherical-direction × projection-magnitude center.
SCHEMA_VERSION = 2

EPS = 1e-12


@dataclass(frozen=True)
class Calibration:
    center: np.ndarray
    threshold: float
    n_activations: int
    percentile: float


def cache_dir() -> Path:
    root = os.environ.get("EP_CALIBRATION_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "ep" / "calibration"


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def _format_extras(extras: Mapping[str, Any] | None) -> str:
    """Stable suffix for extra cache-key fields (sorted by key)."""
    if not extras:
        return ""
    parts = [f"{_safe(k)}-{_safe(str(v))}" for k, v in sorted(extras.items())]
    return "__" + "__".join(parts)


def cache_path(
    model_name: str,
    hook_name: str,
    percentile: float,
    extras: Mapping[str, Any] | None = None,
) -> Path:
    fname = (
        f"{_safe(model_name)}__{_safe(hook_name)}"
        f"__pct{percentile:g}{_format_extras(extras)}.npz"
    )
    return cache_dir() / fname


def load(
    model_name: str,
    hook_name: str,
    percentile: float,
    extras: Mapping[str, Any] | None = None,
) -> Calibration | None:
    path = cache_path(model_name, hook_name, percentile, extras)
    if not path.exists():
        return None
    data = np.load(path)
    cached_version = int(data["version"]) if "version" in data.files else 1
    if cached_version != SCHEMA_VERSION:
        logger.warning(
            "calibration: cache at %s has schema version %d, expected %d; "
            "ignoring (will recompute)",
            path, cached_version, SCHEMA_VERSION,
        )
        return None
    return Calibration(
        center=data["center"].astype(np.float32),
        threshold=float(data["threshold"]),
        n_activations=int(data["n_activations"]),
        percentile=float(data["percentile"]),
    )


def save(
    model_name: str,
    hook_name: str,
    calibration: Calibration,
    extras: Mapping[str, Any] | None = None,
) -> Path:
    path = cache_path(model_name, hook_name, calibration.percentile, extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        center=calibration.center.astype(np.float32),
        threshold=np.float32(calibration.threshold),
        n_activations=np.int64(calibration.n_activations),
        percentile=np.float32(calibration.percentile),
        version=np.int32(SCHEMA_VERSION),
    )
    return path


def _percentile_pairwise_cosine(
    directions: np.ndarray, percentile: float,
) -> float:
    """p-th percentile of within-batch pairwise cosine distances.

    Uses GPU via torch if available — pdist on N=4000 unit vectors is
    O(N²) = 16M ops, trivially batched, and ~2 orders of magnitude faster
    on GPU than scipy.pdist on CPU.
    """
    n = len(directions)
    if n < 2:
        return float("nan")

    torch_mod, device = try_torch_gpu()
    if torch_mod is not None and device is not None:
        d = torch_mod.from_numpy(directions).to(device=device, dtype=torch_mod.float32)
        sim = d @ d.T
        iu = torch_mod.triu_indices(n, n, offset=1, device=device)
        dists = 1.0 - sim[iu[0], iu[1]]
        # torch.quantile matches np.percentile semantics (linear interpolation
        # between ranks) but caps input at ~16M elements. Above that, fall back
        # to kthvalue — the rank-vs-percentile discrepancy is bounded by the
        # gap between adjacent ranks, which is negligible when n_pairs is huge.
        n_pairs = dists.numel()
        if n_pairs <= 16_000_000:
            return float(torch_mod.quantile(dists, percentile / 100.0).item())
        k = max(1, min(n_pairs, int(round((percentile / 100.0) * (n_pairs - 1))) + 1))
        return float(torch_mod.kthvalue(dists, k).values.item())

    # CPU fallback: full sim matrix is O(N²) memory; pdist is the standard.
    from scipy.spatial.distance import pdist
    dists = pdist(directions, metric="cosine")
    return float(np.percentile(dists, percentile))


def calibrate(
    activation_batches: Iterable[np.ndarray],
    *,
    n_tokens: int = 200_000,
    percentile: float = 10.0,
) -> Calibration:
    """Compute (center, threshold) from at least ``n_tokens`` activations.

    ``center`` is the spherical mean direction (each activation contributes
    its unit-direction equally) scaled by the average projection of
    activations onto that direction. ``threshold`` is the mean across batches
    of the within-batch p-th percentile of cosine distances between
    unit-direction activations, computed against the final ``center``.

    Two passes are necessary so the threshold uses the final center rather
    than a streaming approximation; the second pass uses every activation
    in every batch.

    Args:
        activation_batches: iterable yielding (n_acts, d) numpy arrays.
        n_tokens: stop once at least this many activations have been seen
            (overshoots by at most one batch).
        percentile: which percentile of pairwise cosine distance to use.
    """
    sum_acts: np.ndarray | None = None
    sum_units: np.ndarray | None = None
    total = 0
    batches: list[np.ndarray] = []

    for i, batch in enumerate(activation_batches):
        if batch.ndim != 2:
            raise ValueError(
                f"calibration batch {i} must be 2-D (n_acts, d), got shape {batch.shape}"
            )
        batch = batch.astype(np.float32, copy=False)
        norms = np.linalg.norm(batch, axis=1, keepdims=True) + EPS
        b_sum = batch.sum(axis=0)
        b_unit_sum = (batch / norms).sum(axis=0)
        sum_acts = b_sum if sum_acts is None else sum_acts + b_sum
        sum_units = b_unit_sum if sum_units is None else sum_units + b_unit_sum
        total += len(batch)
        batches.append(batch)
        if total >= n_tokens:
            break

    if sum_acts is None or sum_units is None:
        raise ValueError("calibration received no activation batches")

    # Spherical mean direction: each activation contributes equally to direction.
    direction_raw = sum_units / total
    direction = direction_raw / (np.linalg.norm(direction_raw) + EPS)
    # Magnitude: avg projection of activations onto the bulk direction.
    # Equivalent to <mean(a), direction>; using the Euclidean mean here
    # gives the natural scale at which to subtract.
    mean_act = sum_acts / total
    proj_magnitude = float(mean_act @ direction)
    center = (direction * proj_magnitude).astype(np.float32)

    pct_per_batch: list[float] = []
    for batch in batches:
        directions = centered_unit(batch, center)
        if len(directions) < 2:
            continue
        pct_per_batch.append(_percentile_pairwise_cosine(directions, percentile))
    if not pct_per_batch:
        raise ValueError(
            "calibration produced no within-batch distances; need batches of size >= 2"
        )
    threshold = float(np.mean(pct_per_batch))

    logger.info(
        "calibration: %d activations (%d batches), "
        "center ||·||=%.4f, threshold@p%g=%.6f",
        total, len(batches),
        float(np.linalg.norm(center)), percentile, threshold,
    )
    return Calibration(
        center=center,
        threshold=threshold,
        n_activations=total,
        percentile=percentile,
    )


def load_or_calibrate(
    model_name: str,
    hook_name: str,
    activation_batches_fn,
    *,
    n_tokens: int = 200_000,
    percentile: float = 10.0,
    extras: Mapping[str, Any] | None = None,
    force: bool = False,
) -> Calibration:
    """Return cached calibration if present, else calibrate and cache.

    ``activation_batches_fn`` is a zero-arg callable that returns an iterable
    of activation batches (only invoked on cache miss).
    ``extras`` are extra cache-key fields (e.g. extractor, sampling_mode) so
    different activation sources don't share a cache slot.
    """
    if not force:
        cached = load(model_name, hook_name, percentile, extras)
        if cached is not None and cached.n_activations >= n_tokens:
            logger.info(
                "calibration: cache hit at %s (n_activations=%d, threshold=%.6f)",
                cache_path(model_name, hook_name, percentile, extras),
                cached.n_activations, cached.threshold,
            )
            return cached
    calibration = calibrate(
        activation_batches_fn(),
        n_tokens=n_tokens,
        percentile=percentile,
    )
    path = save(model_name, hook_name, calibration, extras)
    logger.info("calibration: cached to %s", path)
    return calibration
