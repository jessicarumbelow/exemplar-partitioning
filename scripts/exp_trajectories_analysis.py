"""Post-hoc analysis of trajectories.json — surfaces shared/divergent cells
across categories.

The most interesting trajectory finding from the L4/L12/L20 run is that
math and factual prompts share early- and late-layer cells (L4 cell 4,
L20 cell 22) but diverge at L12. The model uses L12 specifically to route
"answer math" vs "recall fact". This script computes that systematically:
for every pair of categories, which layers do they overlap on, which
do they split on?

Usage:
    uv run python -m scripts.exp_trajectories_analysis \
        --path /tmp/ep_results/trajectories.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True,
                        help="Path to trajectories.json")
    args = parser.parse_args()

    data = json.loads(args.path.read_text())
    layers = data["layers"]
    trajs = data["trajectories"]

    cats = sorted({t["category"] for t in trajs})

    # 1. For each (layer, category), what's the dominant cell?
    print("=" * 70)
    print(f"Dominant cell per (layer, category) — model: {data['model']}")
    print("=" * 70)
    for li, layer in enumerate(layers):
        print(f"\nL{layer}:")
        for cat in cats:
            paths = [t["path"][li] for t in trajs if t["category"] == cat]
            counter = Counter(paths)
            top_pid, top_n = counter.most_common(1)[0]
            print(f"  {cat:>9}: cell {top_pid:>4} dominates "
                  f"({top_n}/{len(paths)} = {top_n/len(paths):.0%})")

    # 2. Cross-category sharing: which (layer, cell) is shared by multiple cats?
    print()
    print("=" * 70)
    print("Cells shared across categories (≥10 prompts of each)")
    print("=" * 70)
    for li, layer in enumerate(layers):
        cell_to_cats: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for t in trajs:
            cell_to_cats[t["path"][li]][t["category"]] += 1
        shared = []
        for cell, cat_counts in cell_to_cats.items():
            big_cats = [c for c, n in cat_counts.items() if n >= 10]
            if len(big_cats) >= 2:
                shared.append((cell, dict(cat_counts), big_cats))
        if not shared:
            print(f"\nL{layer}: no shared cells (categories cleanly separated)")
            continue
        print(f"\nL{layer}: {len(shared)} cells shared by ≥2 categories")
        for cell, counts, big in sorted(shared)[:10]:
            counts_str = ", ".join(f"{c}={n}" for c, n in counts.items() if n)
            print(f"  cell {cell:>4}: {counts_str}")

    # 3. Pairwise category-overlap matrix per layer.
    print()
    print("=" * 70)
    print("Pairwise overlap (Jaccard of cell-id sets) per layer")
    print("=" * 70)
    for li, layer in enumerate(layers):
        cat_cells: dict[str, set[int]] = defaultdict(set)
        for t in trajs:
            cat_cells[t["category"]].add(t["path"][li])
        print(f"\nL{layer}:")
        print("           " + "  ".join(f"{c:>9}" for c in cats))
        for c1 in cats:
            row = []
            for c2 in cats:
                if c1 == c2:
                    row.append("       —")
                else:
                    a, b = cat_cells[c1], cat_cells[c2]
                    j = len(a & b) / len(a | b) if (a | b) else 0.0
                    row.append(f"     {j:.2f}")
            print(f"  {c1:>9}  " + "  ".join(row))

    # 4. Which categories diverge AT each layer? (i.e., shared at adjacent
    # layers but split here, or split here but reconverge later)
    print()
    print("=" * 70)
    print("Category-pair separation by layer")
    print("=" * 70)
    print("Lower Jaccard = more separated at that layer.")
    print()
    for c1, c2 in combinations(cats, 2):
        jaccards = []
        for li in range(len(layers)):
            a = {t["path"][li] for t in trajs if t["category"] == c1}
            b = {t["path"][li] for t in trajs if t["category"] == c2}
            j = len(a & b) / len(a | b) if (a | b) else 0.0
            jaccards.append(j)
        deepest = layers[jaccards.index(min(jaccards))]
        print(f"  {c1:>9} vs {c2:<9}: " +
              "  ".join(f"L{l}={j:.2f}" for l, j in zip(layers, jaccards)) +
              f"   most-separated@L{deepest}")

    # 5. Trajectory diversity per category.
    print()
    print("=" * 70)
    print("Trajectory diversity per category (unique paths / n prompts)")
    print("=" * 70)
    for cat in cats:
        paths = [tuple(t["path"]) for t in trajs if t["category"] == cat]
        unique = len(set(paths))
        common = Counter(paths).most_common(3)
        print(f"  {cat:>9}: {unique:>3} unique / {len(paths):>3} prompts")
        for path, n in common:
            print(f"             {list(path)} × {n}")


if __name__ == "__main__":
    main()
