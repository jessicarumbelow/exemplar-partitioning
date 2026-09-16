"""Build EP refusal directions from local activations and assignment caches.

Prompt order is harmful first, then benign. The prompt JSON is supplied by the
user and is copied to the output because the intervention script needs the same
held-out prompts. No benchmark prompt corpus is distributed with this repo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def region_exemplars(ids, dists):
    return {
        int(region): (int(members[np.argmin(dists[members])]), members)
        for region in np.unique(ids)
        if len(members := np.flatnonzero(ids == region))
    }


def shuffled_exemplars(acts, ids, keep, rng):
    sizes = [
        len(members) for region in np.unique(ids)
        if len(members := np.intersect1d(np.flatnonzero(ids == region), keep))
    ]
    pool = rng.permutation(keep)
    exemplars, offset = [], 0
    for size in sizes:
        group = pool[offset:offset + size]
        offset += size
        if len(group) < size:
            break
        center = acts[group].mean(0)
        exemplars.append(int(group[np.argmin(np.linalg.norm(
            acts[group] - center, axis=1))]))
    return exemplars


def build_checkpoint(acts, ids_all, dists_all, is_harmful, held,
                     held_h, held_b, rng):
    n_layers, width = acts.shape[1:]
    allowed_h = np.array([i for i in np.flatnonzero(is_harmful) if i not in held])
    allowed_b = np.array([i for i in np.flatnonzero(~is_harmful) if i not in held])
    directions = np.zeros((n_layers, width), np.float32)
    prompt_mean = np.zeros_like(directions)
    shuffled = np.zeros_like(directions)
    norms = np.zeros(n_layers, np.float32)
    counts = []
    for layer in range(n_layers):
        layer_acts = np.asarray(acts[:, layer], dtype=np.float32)
        norms[layer] = np.median(np.linalg.norm(layer_acts, axis=1))
        regions = region_exemplars(ids_all[layer], dists_all[layer])
        h_ex = [e for e, m in regions.values()
                if is_harmful[m].all() and e not in held]
        b_ex = [e for e, m in regions.values()
                if (~is_harmful[m]).all() and e not in held]
        vector = layer_acts[h_ex].mean(0) - layer_acts[b_ex].mean(0)
        directions[layer] = vector / np.linalg.norm(vector)
        vector = layer_acts[allowed_h].mean(0) - layer_acts[allowed_b].mean(0)
        prompt_mean[layer] = vector / np.linalg.norm(vector)
        shuffled_h = shuffled_exemplars(layer_acts, ids_all[layer], allowed_h, rng)
        shuffled_b = shuffled_exemplars(layer_acts, ids_all[layer], allowed_b, rng)
        vector = layer_acts[shuffled_h].mean(0) - layer_acts[shuffled_b].mean(0)
        shuffled[layer] = vector / np.linalg.norm(vector)
        counts.append({
            "layer": layer, "K": len(regions),
            "n_harmful_regions": len(h_ex), "n_benign_regions": len(b_ex),
            "median_resid_norm": float(norms[layer]),
            "cos_ep_vs_promptmean": float(directions[layer] @ prompt_mean[layer]),
            "cos_ep_vs_shuffled": float(directions[layer] @ shuffled[layer]),
        })
    return directions, prompt_mean, shuffled, norms, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--g-it-activations", type=Path, required=True)
    parser.add_argument("--l-it-activations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-per-side", type=int, default=1176)
    parser.add_argument("--n-held-per-side", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--g-canonical-layer", type=int, default=18)
    parser.add_argument("--l-canonical-layer", type=int, default=17)
    args = parser.parse_args()

    prompts = json.loads(args.prompts_json.read_text())
    if len(prompts) != 2 * args.n_per_side:
        raise ValueError("prompt JSON must contain n_per_side harmful prompts followed by the same number of benign prompts")
    is_harmful = np.arange(len(prompts)) < args.n_per_side
    rng = np.random.default_rng(args.seed)
    held_h = np.sort(rng.choice(np.arange(args.n_per_side),
                                args.n_held_per_side, replace=False))
    held_b = np.sort(rng.choice(np.arange(args.n_per_side, len(prompts)),
                                args.n_held_per_side, replace=False))
    held = set(held_h.tolist()) | set(held_b.tolist())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompts.json").write_text(json.dumps(prompts))
    canonical = {"g_it": args.g_canonical_layer,
                 "l_it": args.l_canonical_layer}
    meta = {
        "n_per_side": args.n_per_side,
        "n_held_per_side": args.n_held_per_side,
        "seed": args.seed,
        "held_harmful_idx": held_h.tolist(),
        "held_benign_idx": held_b.tolist(),
        "canonical_layer": canonical,
        "checkpoints": {},
    }
    inputs = {
        "g_it": args.g_it_activations,
        "l_it": args.l_it_activations,
    }
    for tag, activation_path in inputs.items():
        acts = np.load(activation_path, mmap_mode="r")
        cache = np.load(args.cache_dir / f"cache_{tag}.npz")
        built = build_checkpoint(
            acts, cache["ids"], cache["dists"], is_harmful, held,
            held_h, held_b, canonical[tag], rng,
        )
        directions, prompt_mean, shuffled, norms, counts = built
        np.save(args.output_dir / f"dirs_{tag}.npy", directions)
        np.save(args.output_dir / f"dirs_pm_{tag}.npy", prompt_mean)
        np.save(args.output_dir / f"dirs_sh_{tag}.npy", shuffled)
        np.save(args.output_dir / f"norms_{tag}.npy", norms)
        meta["checkpoints"][tag] = {
            "n_layers": len(directions), "d": directions.shape[1],
            "layers": counts,
        }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
