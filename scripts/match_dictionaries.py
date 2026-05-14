"""Match exemplars between two EP dictionaries by Hungarian assignment.

Loads two pickled Dictionary objects (built with the same construction protocol
under fixed calibration), computes pairwise cosine similarity between their
exemplar directions, runs linear_sum_assignment to maximise matched cosine,
and writes a summary plus per-pair CSVs that localise persisted, dropped,
and introduced partitions.

Comparison is in centered-direction space — each dictionary stores
``exemplar_direction = (a - mu) / ||a - mu||`` with its own per-(model, layer)
calibration mean ``mu``. Cosine similarity between two centered-unit vectors
is well-defined as a similarity measure but has a slight bias when the two
``mu``s differ: it isolates structural alignment from any global activation-
mean shift between the two models. Distances are also reported in units of
the local threshold ``theta`` (the 10th-percentile pairwise distance on the
source distribution), so cross-layer comparison is meaningful even though
absolute cosine scale shifts with depth.

Usage (from the ep repo root):

    uv run python scripts/match_dictionaries.py \\
        --a path/to/base/layerN.pkl --label-a "base L12" \\
        --b path/to/it/layerN.pkl   --label-b "IT L12" \\
        --out outputs/drift/L12 --cutoff 0.7
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _exemplar_matrix(d) -> np.ndarray:
    rows = [p.exemplar_direction.astype(np.float32) for p in d.partitions]
    return np.stack(rows, axis=0)


def _top_prompt(p) -> str:
    if not p.sample_prompts:
        return ""
    # sample_prompts is a heap of (-distance, text, position); the smallest
    # distance is the closest member to the exemplar. The heap orders by
    # the first tuple element, so min over -distance picks the tightest.
    closest = min(p.sample_prompts, key=lambda t: -t[0])
    text = closest[1]
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if len(text) > 160:
        text = text[:157] + "..."
    return text


def _match(a, b, label_a: str, label_b: str, out_dir: Path, cutoff: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    e_a = _exemplar_matrix(a)
    e_b = _exemplar_matrix(b)
    k_a, k_b = e_a.shape[0], e_b.shape[0]
    theta_a, theta_b = float(a.threshold), float(b.threshold)

    # Cosine similarity between unit vectors == dot product.
    sim = e_a @ e_b.T  # (k_a, k_b)

    # linear_sum_assignment minimises cost; we maximise similarity by negating.
    row_ind, col_ind = linear_sum_assignment(-sim)
    n_pairs = len(row_ind)  # min(k_a, k_b)

    matched_cos = sim[row_ind, col_ind]
    matched_dist = 1.0 - matched_cos
    # Normalise by each side's own threshold so distributions are comparable
    # across layers and across base/finetune cells.
    norm_a = matched_dist / theta_a
    norm_b = matched_dist / theta_b

    # Index sets covered by Hungarian.
    matched_a_idx = set(int(i) for i in row_ind)
    matched_b_idx = set(int(j) for j in col_ind)

    # Per-pair CSV.
    pair_path = out_dir / "matches.csv"
    with pair_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "i_a", "i_b", "cos", "dist", "dist_over_theta_a", "dist_over_theta_b",
            "members_a", "members_b", "top_prompt_a", "top_prompt_b",
        ])
        order = np.argsort(-matched_cos)  # tightest matches first
        for k in order:
            i_a, i_b = int(row_ind[k]), int(col_ind[k])
            w.writerow([
                i_a, i_b,
                round(float(matched_cos[k]), 6),
                round(float(matched_dist[k]), 6),
                round(float(norm_a[k]), 4),
                round(float(norm_b[k]), 4),
                int(a.partitions[i_a].member_count),
                int(b.partitions[i_b].member_count),
                _top_prompt(a.partitions[i_a]),
                _top_prompt(b.partitions[i_b]),
            ])

    # Unmatched-A: partitions in A that Hungarian could not pair (when k_a > k_b)
    # plus pairs whose cosine fell below the --cutoff (treat as effective drop).
    unpaired_a = [i for i in range(k_a) if i not in matched_a_idx]
    weak_a = [int(row_ind[k]) for k in range(n_pairs) if matched_cos[k] < cutoff]
    dropped_a = sorted(set(unpaired_a) | set(weak_a))

    unpaired_b = [j for j in range(k_b) if j not in matched_b_idx]
    weak_b = [int(col_ind[k]) for k in range(n_pairs) if matched_cos[k] < cutoff]
    introduced_b = sorted(set(unpaired_b) | set(weak_b))

    def _write_unmatched(path: Path, dictionary, indices: list[int]) -> None:
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["i", "members", "top_prompt"])
            for i in sorted(indices, key=lambda x: -dictionary.partitions[x].member_count):
                p = dictionary.partitions[i]
                w.writerow([i, int(p.member_count), _top_prompt(p)])

    _write_unmatched(out_dir / "a_dropped.csv", a, dropped_a)
    _write_unmatched(out_dir / "b_introduced.csv", b, introduced_b)

    # Persisted = pairs whose cosine clears the cutoff.
    persisted_pairs = int(np.sum(matched_cos >= cutoff))

    summary = {
        "label_a": label_a,
        "label_b": label_b,
        "k_a": k_a,
        "k_b": k_b,
        "theta_a": theta_a,
        "theta_b": theta_b,
        "n_hungarian_pairs": int(n_pairs),
        "cutoff": cutoff,
        "persisted_above_cutoff": persisted_pairs,
        "dropped_a_total": len(dropped_a),
        "dropped_a_unpaired": len(unpaired_a),
        "dropped_a_below_cutoff": len(weak_a),
        "introduced_b_total": len(introduced_b),
        "introduced_b_unpaired": len(unpaired_b),
        "introduced_b_below_cutoff": len(weak_b),
        "matched_cos_stats": {
            "mean": round(float(matched_cos.mean()), 4),
            "median": round(float(np.median(matched_cos)), 4),
            "p10": round(float(np.percentile(matched_cos, 10)), 4),
            "p90": round(float(np.percentile(matched_cos, 90)), 4),
            "min": round(float(matched_cos.min()), 4),
            "max": round(float(matched_cos.max()), 4),
        },
        "matched_dist_over_theta_a_stats": {
            "mean": round(float(norm_a.mean()), 4),
            "median": round(float(np.median(norm_a)), 4),
            "p10": round(float(np.percentile(norm_a, 10)), 4),
            "p90": round(float(np.percentile(norm_a, 90)), 4),
        },
        "matched_dist_over_theta_b_stats": {
            "mean": round(float(norm_b.mean()), 4),
            "median": round(float(np.median(norm_b)), 4),
            "p10": round(float(np.percentile(norm_b, 10)), 4),
            "p90": round(float(np.percentile(norm_b, 90)), 4),
        },
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path, help="Pickled Dictionary A (e.g. base).")
    ap.add_argument("--b", required=True, type=Path, help="Pickled Dictionary B (e.g. finetune).")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", required=True, type=Path, help="Output directory.")
    ap.add_argument("--cutoff", type=float, default=0.7,
                    help="Matched cosine below this is treated as effective drop/introduction.")
    args = ap.parse_args()

    a = _load(args.a)
    b = _load(args.b)
    logger.info("loaded %s: %d partitions, theta=%.4f", args.label_a, len(a), a.threshold)
    logger.info("loaded %s: %d partitions, theta=%.4f", args.label_b, len(b), b.threshold)

    summary = _match(a, b, args.label_a, args.label_b, args.out, args.cutoff)
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
