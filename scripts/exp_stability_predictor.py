"""Test whether per-cell d(exemplar, mean) predicts cross-seed matched-mean cosine.

Hypothesis: cells whose exemplar lies close to the cell mean live in dense
directional regions, and dense regions are stable across seeds. So a single
dictionary's per-cell ``cos(exemplar, mean)`` should predict how well that
cell's mean direction matches the closest cell in a seed-shuffled rebuild.

If true, a single-seed dictionary can be filtered to "stable cells only"
without ever running a second seed.

Usage:

    uv run python scripts/exp_stability_predictor.py \\
        --dicts /tmp/seed_stab/pile/p8_seed*.pkl \\
        --out outputs/stability/p8 \\
        --label "Pile L12 p=8"
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, pearsonr


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _matrices(d) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.stack([p.mean_member_direction.astype(np.float32) for p in d.partitions])
    exemplars = np.stack([p.exemplar_direction.astype(np.float32) for p in d.partitions])
    s_self = (means * exemplars).sum(axis=1)  # cos(e_i, m_i) per cell
    coherence = np.array([p.member_coherence for p in d.partitions], dtype=np.float32)
    members = np.array([p.member_count for p in d.partitions], dtype=np.float32)
    return means, exemplars, s_self, coherence, members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicts", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="stability-predictor")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in args.dicts]
    dicts = [_load(p) for p in paths]
    labels = [p.stem for p in paths]
    sizes = [len(d.partitions) for d in dicts]

    # Pairwise: for each ordered (A, B) pair, Hungarian match A→B by mean.
    rows: list[dict] = []
    for (i, a), (j, b) in combinations(enumerate(dicts), 2):
        m_a, _, s_a, c_a, n_a = _matrices(a)
        m_b, _, s_b, c_b, n_b = _matrices(b)
        sim = m_a @ m_b.T
        row, col = linear_sum_assignment(-sim)
        for k in range(len(row)):
            ia, jb = int(row[k]), int(col[k])
            n_val = float(max(n_a[ia], 1))
            c_val = float(c_a[ia])
            rows.append({
                "pair": f"{labels[i]}__{labels[j]}",
                "i_a": ia, "i_b": jb,
                "s_self_a": float(s_a[ia]),
                "coherence_a": c_val,
                "members_a": int(n_a[ia]),
                "log_members_a": float(np.log10(n_val)),
                "density_n_times_c": float(np.log10(n_val * c_val + 1e-9)),
                "density_n_over_1minusC": float(np.log10(n_val / (1.0 - c_val + 0.01))),
                "matched_cos": float(sim[ia, jb]),
            })

    matched_arr = np.array([r["matched_cos"] for r in rows])

    def _corr(predictor_name: str) -> dict:
        x = np.array([r[predictor_name] for r in rows])
        rp, _ = pearsonr(x, matched_arr)
        rs, _ = spearmanr(x, matched_arr)
        return {"pearson": round(float(rp), 4), "spearman": round(float(rs), 4)}

    correlations = {
        "s_self_a": _corr("s_self_a"),
        "coherence_a": _corr("coherence_a"),
        "log_members_a": _corr("log_members_a"),
        "density_n_times_c": _corr("density_n_times_c"),
        "density_n_over_1minusC": _corr("density_n_over_1minusC"),
    }

    # Partial correlation of coherence with matched_cos, controlling for log_members
    def _partial_corr(y_name: str, x_name: str, z_name: str) -> dict:
        y = np.array([r[y_name] for r in rows])
        x = np.array([r[x_name] for r in rows])
        z = np.array([r[z_name] for r in rows])
        # residualise both x and y against z (linear regression), correlate residuals
        z1 = np.column_stack([np.ones_like(z), z])
        bx, *_ = np.linalg.lstsq(z1, x, rcond=None)
        by, *_ = np.linalg.lstsq(z1, y, rcond=None)
        rx = x - z1 @ bx
        ry = y - z1 @ by
        rp, _ = pearsonr(rx, ry)
        rs, _ = spearmanr(rx, ry)
        return {"pearson": round(float(rp), 4), "spearman": round(float(rs), 4)}

    partials = {
        "coherence_given_log_members": _partial_corr("matched_cos", "coherence_a", "log_members_a"),
        "s_self_given_log_members": _partial_corr("matched_cos", "s_self_a", "log_members_a"),
    }

    # Bin matched_cos by quintile of each predictor
    def _quintile_table(name: str) -> list[dict]:
        x = np.array([r[name] for r in rows])
        qs = np.percentile(x, [0, 20, 40, 60, 80, 100])
        out = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            mask = (x >= lo) & (x <= hi if hi == qs[-1] else x < hi)
            if mask.sum() == 0:
                out.append({"range": f"[{lo:.3f}, {hi:.3f}]", "n": 0,
                            "matched_cos_mean": None, "matched_cos_median": None})
                continue
            out.append({
                "range": f"[{lo:.3f}, {hi:.3f}]",
                "n": int(mask.sum()),
                "matched_cos_mean": round(float(matched_arr[mask].mean()), 4),
                "matched_cos_median": round(float(np.median(matched_arr[mask])), 4),
            })
        return out

    summary = {
        "label": args.label,
        "n_dicts": len(dicts),
        "dict_sizes": sizes,
        "n_matched_pairs_total": len(rows),
        "predictor_correlations": correlations,
        "partial_correlations_controlling_for_log_members": partials,
        "matched_cos_by_log_members_quintile": _quintile_table("log_members_a"),
        "matched_cos_by_coherence_quintile": _quintile_table("coherence_a"),
        "matched_cos_by_density_n_times_c_quintile": _quintile_table("density_n_times_c"),
        "matched_cos_by_density_n_over_1minusC_quintile": _quintile_table("density_n_over_1minusC"),
    }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    if rows:
        with (args.out / "matched_pairs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
