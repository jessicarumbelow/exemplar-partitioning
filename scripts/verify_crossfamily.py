"""Recompute cross-family round-trip correspondence from assignment caches.

Expected files in ``cache_dir`` are ``cache_{g,l}_{base,it}.npz`` with arrays
``ids`` and ``dists``. They contain region assignments and nearest-exemplar
distances only; the harmful prompt corpus is not distributed.
"""
import argparse
from pathlib import Path

import numpy as np


N_PER_SIDE = 1176


def load(path):
    cached = np.load(path)
    ids_all, dists_all = cached["ids"], cached["dists"]
    harmful = np.arange(ids_all.shape[1]) < N_PER_SIDE
    exemplars, fractions = [], []
    for ids, dists in zip(ids_all, dists_all):
        layer_exemplars, layer_fractions = {}, {}
        for region in np.unique(ids):
            members = np.flatnonzero(ids == region)
            layer_exemplars[int(region)] = int(members[np.argmin(dists[members])])
            layer_fractions[int(region)] = float(harmful[members].mean())
        exemplars.append(layer_exemplars)
        fractions.append(layer_fractions)
    return ids_all, exemplars, fractions


def grid(anchor, target):
    ids_a, exemplars_a, fractions_a = anchor
    ids_b, exemplars_b, _ = target
    result = np.full((len(ids_a), len(ids_b)), np.nan)
    for layer_a in range(len(ids_a)):
        regions = [r for r, f in fractions_a[layer_a].items() if f == 1.0]
        if not regions:
            continue
        for layer_b in range(len(ids_b)):
            returned = 0
            for region in regions:
                prompt_a = exemplars_a[layer_a][region]
                region_b = int(ids_b[layer_b][prompt_a])
                prompt_b = exemplars_b[layer_b][region_b]
                returned += int(ids_a[layer_a][prompt_b] == region)
            result[layer_a, layer_b] = returned / len(regions)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()
    base = grid(load(args.cache_dir / "cache_g_base.npz"),
                load(args.cache_dir / "cache_l_base.npz"))
    tuned = grid(load(args.cache_dir / "cache_g_it.npz"),
                 load(args.cache_dir / "cache_l_it.npz"))
    print(f"base mean: {np.nanmean(base):.1%}")
    print(f"instruction-tuned mean: {np.nanmean(tuned):.1%}")
    print(f"Gemma L18 through Llama L17: {tuned[18, 17]:.1%}")


if __name__ == "__main__":
    main()
