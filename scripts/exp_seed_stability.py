"""Quantify how much streaming order (and prompt-set sampling) changes EP geometry.

Hypothesis: cell mean directions are stable across seeds (spherical means over
many members average out the order noise), while first-arrival exemplars vary
(each exemplar is one sample). For each pair of dictionaries, Hungarian-match
cells by mean direction and report the matched-cos distributions for both
the mean basis and the exemplar basis.

Usage:

    uv run python scripts/exp_seed_stability.py \\
        --dicts /vol/.../seed0/.../layer20.pkl /vol/.../seed1/.../layer20.pkl ... \\
        --out outputs/seed_stability/p8 \\
        --label "p=8"
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


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _matrices(d) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack([p.mean_member_direction.astype(np.float32) for p in d.partitions])
    exemplars = np.stack([p.exemplar_direction.astype(np.float32) for p in d.partitions])
    return means, exemplars


def _match_pair(a, b, match_basis: str = "mean") -> dict:
    m_a, e_a = _matrices(a)
    m_b, e_b = _matrices(b)
    if match_basis == "mean":
        sim = m_a @ m_b.T
    else:
        sim = e_a @ e_b.T
    row, col = linear_sum_assignment(-sim)
    cos_mean = (m_a[row] * m_b[col]).sum(axis=1)
    cos_exemplar = (e_a[row] * e_b[col]).sum(axis=1)
    return {
        "k_a": int(m_a.shape[0]),
        "k_b": int(m_b.shape[0]),
        "n_matched": int(len(row)),
        "cos_mean": cos_mean,
        "cos_exemplar": cos_exemplar,
    }


def _stats(arr: np.ndarray) -> dict:
    return {
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicts", nargs="+", required=True,
                        help="Paths to pickled dictionaries (≥2).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="seed-stability")
    parser.add_argument("--match-basis", choices=("mean", "exemplar"),
                        default="mean")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dicts = [_load(Path(p)) for p in args.dicts]
    labels = [Path(p).parent.parent.name for p in args.dicts]
    sizes = [len(d.partitions) for d in dicts]
    thetas = [float(d.threshold) for d in dicts]

    pair_rows = []
    all_cos_mean = []
    all_cos_exemplar = []
    for (i, a), (j, b) in combinations(enumerate(dicts), 2):
        r = _match_pair(a, b, match_basis=args.match_basis)
        pair_rows.append({
            "label": args.label,
            "a": labels[i], "b": labels[j],
            "k_a": r["k_a"], "k_b": r["k_b"],
            "n_matched": r["n_matched"],
            "matched_by": args.match_basis,
            "cos_mean_median": float(np.median(r["cos_mean"])),
            "cos_mean_p10": float(np.percentile(r["cos_mean"], 10)),
            "cos_exemplar_median": float(np.median(r["cos_exemplar"])),
            "cos_exemplar_p10": float(np.percentile(r["cos_exemplar"], 10)),
            "gap_median": float(np.median(r["cos_mean"]) - np.median(r["cos_exemplar"])),
        })
        all_cos_mean.append(r["cos_mean"])
        all_cos_exemplar.append(r["cos_exemplar"])

    pooled_mean = np.concatenate(all_cos_mean)
    pooled_exemplar = np.concatenate(all_cos_exemplar)

    summary = {
        "label": args.label,
        "n_dicts": len(dicts),
        "matched_by": args.match_basis,
        "dictionary_sizes": sizes,
        "thresholds": thetas,
        "pooled_cos_mean": _stats(pooled_mean),
        "pooled_cos_exemplar": _stats(pooled_exemplar),
        "gap_median": round(float(np.median(pooled_mean) - np.median(pooled_exemplar)), 4),
    }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out / "pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
