"""Lightweight intrinsic evaluation of an exemplar-partition dictionary.

Computes quality metrics from the dictionary itself — no model inference,
no external datasets. Runs in seconds after `finalize()`.

Metrics serve as proxies for SAEBench-style evaluations:
- Cohesion (mean intra-partition distance) → sparse-probing quality
- Separation (nearest-neighbour exemplar distance) → feature independence
- Coverage (fraction of activations in nontrivial partitions) → completeness
- Redundancy (max exemplar similarity) → absorption risk
- Size distribution (Gini, effective count) → dictionary efficiency
"""

from __future__ import annotations

import logging

import numpy as np

from .dictionary import Dictionary, _cosine_pairwise
from .geometry import try_torch_gpu

logger = logging.getLogger(__name__)


def evaluate_dictionary(dictionary: Dictionary, min_members: int = 2) -> dict:
    """Compute quality metrics for a finalized dictionary.

    Returns a flat dict of scalar metrics suitable for logging.
    """
    partitions = dictionary.partitions
    if not partitions:
        return {"n_partitions": 0}

    members = np.array([p.member_count for p in partitions])
    total_members = int(members.sum())
    nontrivial_mask = members >= min_members
    n_nontrivial = int(nontrivial_mask.sum())
    nontrivial_members = members[nontrivial_mask]

    metrics: dict[str, float | int] = {
        "n_partitions": len(partitions),
        "n_partitions_nontrivial": n_nontrivial,
        "total_members": total_members,
        "threshold": dictionary.threshold or 0.0,
    }

    if n_nontrivial == 0:
        return metrics

    metrics["coverage"] = float(nontrivial_members.sum() / total_members)
    metrics["mean_members"] = float(nontrivial_members.mean())
    metrics["median_members"] = float(np.median(nontrivial_members))
    metrics["max_members"] = int(nontrivial_members.max())
    metrics["std_members"] = float(nontrivial_members.std())
    metrics["gini"] = _gini(nontrivial_members)

    p = nontrivial_members / nontrivial_members.sum()
    entropy = -float((p * np.log(p + 1e-12)).sum())
    metrics["effective_partitions"] = float(np.exp(entropy))
    metrics["partition_utilisation"] = metrics["effective_partitions"] / n_nontrivial

    intra = np.array([
        p.sum_dist_to_exemplar / p.member_count
        for p in partitions if p.member_count >= min_members
    ])
    metrics["mean_intra_dist"] = float(intra.mean())
    metrics["median_intra_dist"] = float(np.median(intra))

    if n_nontrivial < 2:
        metrics["mean_nearest_inter_dist"] = float("nan")
        metrics["separation_ratio"] = float("nan")
        metrics["silhouette"] = float("nan")
        metrics["max_exemplar_similarity"] = float("nan")
        metrics["mean_exemplar_similarity"] = float("nan")
        return metrics

    nontrivial_partitions = [p for p in partitions if p.member_count >= min_members]
    exemplar_dirs = np.stack([p.exemplar_direction for p in nontrivial_partitions])

    nearest_inter, max_off_sim, mean_off_sim = _exemplar_similarity_stats(exemplar_dirs)

    metrics["mean_nearest_inter_dist"] = float(nearest_inter.mean())
    metrics["median_nearest_inter_dist"] = float(np.median(nearest_inter))

    if dictionary.threshold and dictionary.threshold > 0:
        metrics["separation_ratio"] = float(nearest_inter.mean() / dictionary.threshold)

    sil = (nearest_inter - intra) / np.maximum(nearest_inter, intra)
    metrics["silhouette"] = float(sil.mean())

    metrics["max_exemplar_similarity"] = float(max_off_sim)
    metrics["mean_exemplar_similarity"] = float(mean_off_sim)

    return metrics


def _exemplar_similarity_stats(
    exemplar_dirs: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Compute (nearest_inter_dist[K], max_off_sim, mean_off_sim) from K unit dirs.

    Single K×K matmul on GPU when CUDA is available, then off-diagonal
    reductions. Avoids the scipy.pdist round-trip and the (1.6 GB at K=20k)
    host-side dist matrix.
    """
    n = len(exemplar_dirs)
    torch_mod, device = try_torch_gpu()
    if torch_mod is not None and device is not None:
        E = torch_mod.from_numpy(exemplar_dirs).to(device=device, dtype=torch_mod.float32)
        sim = E @ E.T
        sim.clamp_(-1.0, 1.0)
        diag = torch_mod.eye(n, dtype=torch_mod.bool, device=device)
        # Nearest-neighbour distance: mask the diagonal with +inf in dist space,
        # equivalently with -inf in similarity space.
        sim_off = sim.masked_fill(diag, float("-inf"))
        nearest_sim = sim_off.max(dim=1).values
        nearest_inter = (1.0 - nearest_sim).cpu().numpy().astype(np.float32, copy=False)
        # Off-diagonal max/mean similarity. Use upper triangle only (excludes diag).
        iu, ju = torch_mod.triu_indices(n, n, offset=1, device=device)
        off = sim[iu, ju]
        max_off_sim = float(off.max().item())
        mean_off_sim = float(off.mean().item())
        return nearest_inter, max_off_sim, mean_off_sim

    sim = exemplar_dirs @ exemplar_dirs.T
    np.clip(sim, -1.0, 1.0, out=sim)
    dist_matrix = 1.0 - sim
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_inter = dist_matrix.min(axis=1)
    pairwise_sim = 1.0 - _cosine_pairwise(exemplar_dirs)
    return nearest_inter, float(pairwise_sim.max()), float(pairwise_sim.mean())


def _gini(counts) -> float:
    """Gini coefficient of a non-negative array-like of member counts."""
    arr = np.sort(np.asarray(counts, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def format_eval_summary(metrics: dict) -> str:
    """Format evaluation metrics as a human-readable summary."""
    if metrics.get("n_partitions", 0) == 0:
        return "Dictionary: empty"

    lines = ["Dictionary eval:"]
    n = metrics["n_partitions"]
    nt = metrics.get("n_partitions_nontrivial", n)
    lines.append(f"  Partitions:  {nt} nontrivial / {n} total")

    if "coverage" in metrics:
        lines.append(f"  Coverage:    {metrics['coverage']:.1%}")
    if "mean_members" in metrics:
        med = metrics["median_members"]
        mx = metrics.get("max_members", "?")
        lines.append(f"  Members:     median {med:.0f}, max {mx}")
    if "gini" in metrics:
        lines.append(f"  Gini:        {metrics['gini']:.3f}")
    if "effective_partitions" in metrics:
        eff = metrics["effective_partitions"]
        util = metrics.get("partition_utilisation", 0)
        lines.append(f"  Effective:   {eff:.0f} partitions ({util:.1%} utilisation)")
    if "mean_intra_dist" in metrics:
        lines.append(f"  Cohesion:    {metrics['mean_intra_dist']:.4f} mean intra-dist")

    sep = metrics.get("mean_nearest_inter_dist")
    if sep is not None and not np.isnan(sep):
        sr = metrics.get("separation_ratio")
        sr_str = f" ({sr:.1f}x threshold)" if sr is not None else ""
        lines.append(f"  Separation:  {sep:.4f} mean nearest-inter{sr_str}")

    sil = metrics.get("silhouette")
    if sil is not None and not np.isnan(sil):
        lines.append(f"  Silhouette:  {sil:.3f}")

    sim = metrics.get("max_exemplar_similarity")
    if sim is not None and not np.isnan(sim):
        lines.append(f"  Redundancy:  {sim:.3f} max exemplar similarity")

    return "\n".join(lines)


def evaluate_and_log(
    dictionary: Dictionary,
    min_members: int = 2,
    log_fn=None,
) -> dict:
    """Evaluate the dictionary and print a summary. Returns the metric dict.

    If ``log_fn`` is provided (e.g. ``wandb.log``), metrics are logged with
    ``eval/{metric}`` keys.
    """
    metrics = evaluate_dictionary(dictionary, min_members=min_members)
    logger.info("\n%s", format_eval_summary(metrics))
    if log_fn is not None:
        log_fn({
            f"eval/{k}": v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not np.isnan(v)
        })
    return metrics
