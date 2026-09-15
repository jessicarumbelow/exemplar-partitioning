"""Build one cross-family assignment cache from saved activations.

The input is a NumPy array shaped ``(prompts, layers, hidden_dim)``. Prompt
order must match across all four checkpoints. The revision uses harmful prompts
first and benign prompts second; the cache itself stores no prompt text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ep.discovery.calibration import calibrate
from ep.discovery.dictionary import Dictionary


def build_cache(activations: np.ndarray, percentile: float = 12,
                seed: int = 0, batch_size: int = 16):
    if activations.ndim != 3:
        raise ValueError("activations must have shape (prompts, layers, hidden_dim)")
    n_prompts, n_layers, _ = activations.shape
    order = np.random.default_rng(seed).permutation(n_prompts)
    ids_all = np.empty((n_layers, n_prompts), dtype=np.int32)
    dists_all = np.empty((n_layers, n_prompts), dtype=np.float32)
    for layer in range(n_layers):
        layer_acts = np.asarray(activations[:, layer], dtype=np.float32)
        batches = (layer_acts[order][i:i + batch_size]
                   for i in range(0, n_prompts, batch_size))
        calibration = calibrate(
            batches, n_tokens=n_prompts, percentile=percentile,
        )
        dictionary = Dictionary(
            center=calibration.center, threshold=calibration.threshold,
        )
        for i in range(0, n_prompts, batch_size):
            dictionary.add_batch(layer_acts[order][i:i + batch_size])
        dictionary.finalize()
        ids_all[layer], dists_all[layer] = dictionary.assign(layer_acts)
        print(f"layer {layer}: K={len(dictionary.partitions)}")
    return ids_all, dists_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("activations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--percentile", type=float, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    acts = np.load(args.activations, mmap_mode="r")
    ids, dists = build_cache(
        acts, percentile=args.percentile, seed=args.seed,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=ids, dists=dists)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
