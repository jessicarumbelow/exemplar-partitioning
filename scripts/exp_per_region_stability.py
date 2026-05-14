"""Per-region stability vs every Partition-stored metric.

For each region in each dictionary at a given (model, layer, percentile),
compute its empirical stability as the mean Hungarian-matched cosine
across all other-seed dictionaries at the same setting. Output a CSV
with every per-region predictor we can compute from a single dict, plus
the stability outcome. Optionally make scatter panels.

Usage:

    uv run python scripts/exp_per_region_stability.py \\
        --dicts /tmp/seed_stab/pile/p8_seed*.pkl \\
        --out outputs/per_region_stability/p8 \\
        --label "Pile L12 p=8"
"""

from __future__ import annotations

import argparse
import csv
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _per_region_metrics(d) -> dict:
    """Compute every single-dict scalar metric for each cell."""
    parts = d.partitions
    n = len(parts)
    means = np.stack([p.mean_member_direction.astype(np.float32) for p in parts])
    exemplars = np.stack([p.exemplar_direction.astype(np.float32) for p in parts])

    metrics = {
        "members": np.array([p.member_count for p in parts], dtype=np.float64),
        "log_members": np.array([np.log10(max(p.member_count, 1)) for p in parts]),
        "coherence": np.array([p.member_coherence for p in parts]),
        "s_self": (means * exemplars).sum(axis=1),
    }

    # Exemplar-centric distance stats (if accumulated)
    sum_dist = np.array([getattr(p, "sum_dist_to_exemplar", 0.0) for p in parts])
    sum_sq_dist = np.array([getattr(p, "sum_sq_dist_to_exemplar", 0.0) for p in parts])
    members_safe = np.maximum(metrics["members"], 1)
    mean_dist = sum_dist / members_safe
    var_dist = sum_sq_dist / members_safe - mean_dist ** 2
    metrics["mean_dist_to_exemplar"] = mean_dist
    metrics["var_dist_to_exemplar"] = np.clip(var_dist, 0.0, None)
    metrics["std_dist_to_exemplar"] = np.sqrt(metrics["var_dist_to_exemplar"])

    # Temporal/iteration stats
    metrics["n_source_iterations"] = np.array(
        [len(getattr(p, "source_iterations", set())) for p in parts]
    )
    iters = [getattr(p, "source_iterations", set()) for p in parts]
    metrics["first_iteration"] = np.array(
        [min(s) if s else 0 for s in iters]
    )
    metrics["last_iteration"] = np.array(
        [max(s) if s else 0 for s in iters]
    )
    metrics["iteration_span"] = metrics["last_iteration"] - metrics["first_iteration"]

    # Combined predictors
    metrics["density_log_Nc"] = np.log10(
        metrics["members"] * np.maximum(metrics["coherence"], 1e-9)
    )
    metrics["log_n_source_iters"] = np.log10(
        np.maximum(metrics["n_source_iterations"], 1)
    )
    return means, metrics


def _per_region_stability(dicts: list, labels: list[str]) -> list[dict]:
    """Per-region average matched-cosine across all other-seed dicts."""
    matrices_metrics = [_per_region_metrics(d) for d in dicts]
    rows: list[dict] = []
    for i, (m_a, met_a) in enumerate(matrices_metrics):
        # gather matched_cos for each cell in dict i across all other dicts
        per_cell_matched: list[list[float]] = [[] for _ in range(m_a.shape[0])]
        for j, (m_b, _) in enumerate(matrices_metrics):
            if i == j:
                continue
            sim = m_a @ m_b.T
            row, col = linear_sum_assignment(-sim)
            for k in range(len(row)):
                ia = int(row[k])
                jb = int(col[k])
                per_cell_matched[ia].append(float(sim[ia, jb]))
        for ia in range(m_a.shape[0]):
            ms = per_cell_matched[ia]
            if not ms:
                continue
            row_dict = {
                "dict_label": labels[i],
                "cell_id": ia,
                "stability_mean": float(np.mean(ms)),
                "stability_median": float(np.median(ms)),
                "stability_min": float(np.min(ms)),
                "n_other_seeds": len(ms),
            }
            for name, arr in met_a.items():
                row_dict[name] = float(arr[ia])
            rows.append(row_dict)
    return rows


def _scatter_grid(rows: list[dict], out_path: Path, label: str) -> None:
    metrics_to_plot = [
        ("log_members", "log10(N)"),
        ("coherence", "coherence c"),
        ("density_log_Nc", "log10(N·c)"),
        ("s_self", "s_self = cos(e, m)"),
        ("mean_dist_to_exemplar", "mean dist to exemplar"),
        ("std_dist_to_exemplar", "std dist to exemplar"),
        ("log_n_source_iters", "log10(# source iterations)"),
        ("iteration_span", "iteration span"),
    ]
    n = len(metrics_to_plot)
    cols = 4
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    y = np.array([r["stability_mean"] for r in rows])
    for ax, (key, prettyname) in zip(axes, metrics_to_plot):
        x = np.array([r[key] for r in rows])
        # Spearman, robust
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        ax.scatter(x, y, s=4, alpha=0.25, edgecolors="none")
        ax.set_xlabel(prettyname)
        ax.set_ylabel("stability (mean matched-cos)")
        ax.set_title(f"{prettyname}  (ρ={rho:+.3f})", fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicts", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="per-region-stability")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in args.dicts]
    dicts = [_load(p) for p in paths]
    labels = [p.stem for p in paths]
    rows = _per_region_stability(dicts, labels)

    csv_path = args.out / "per_region.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {csv_path}")

    fig_path = args.out / "scatter_grid.png"
    _scatter_grid(rows, fig_path, args.label)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
