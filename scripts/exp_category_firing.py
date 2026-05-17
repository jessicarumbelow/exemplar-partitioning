"""Category firing test: do prompts whose next token is in a given semantic
category land in geometrically-close EP partitions?

Pipeline:
1. For each of three categories (number, colour, country), construct a set of
   prompts where the next token is constrained to that category.
2. Forward each prompt through the model; capture the layer-L residual at
   the LAST position (the position whose activation produces the next token).
3. Assign each captured activation to its nearest partition in a built EP
   dictionary.
4. Compare within-category and between-category geometric distance among
   the hit partitions. If geometry encodes semantic category at this layer,
   within-category distance < between-category distance.

Run:
    uv run python -m scripts.exp_category_firing --dict-path path/to/dictionary.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


PROMPTS_BY_CATEGORY: dict[str, list[str]] = {
    "number": [
        "The total number of days in a week is",
        "Her age was",
        "He counted to",
        "Two plus two is",
        "How old are you? I am",
        "The room number is",
        "The price tag read $",
        "She finished in position number",
        "1, 2, 3, ",
        "How many fingers? I have",
        "We have approximately",
        "The score on the test was",
        "There were exactly",
        "He won by a margin of",
        "The temperature outside was",
        "She read up to chapter",
        "The total cost came to $",
        "The first three numbers are 1, 2,",
        "On a scale of 1 to 10, I'd give it a",
        "I'll take",
    ],
    "colour": [
        "Roses are red, violets are",
        "The sky on a sunny day is",
        "Grass is typically",
        "The colour of fresh snow is",
        "Lemons are usually",
        "The traffic light changed to",
        "Her dress was a vibrant",
        "He painted the wall",
        "The autumn leaves turned",
        "She had brown hair and bright",
        "The ocean appeared deep",
        "Stop signs are usually",
        "Coal is typically",
        "His shirt was bright",
        "The cherry was a deep",
        "An apple is often",
        "Bananas turn from green to",
        "Her favourite colour is",
        "He prefers shirts that are",
        "The sunset turned the sky",
    ],
    "country": [
        "Paris is the capital city of",
        "She moved abroad to live in",
        "The Eiffel Tower is located in",
        "Sushi originated in",
        "Tokyo is the capital of",
        "Big Ben can be found in",
        "Spaghetti and pizza come from",
        "Berlin is the capital of",
        "She visited the country of",
        "He emigrated from",
        "The Pyramids of Giza are in",
        "Ottawa is the capital of",
        "Madrid is the capital of",
        "She was born in the country of",
        "The Great Wall is in",
        "She's travelling next month to",
        "Athens is the capital of",
        "Stockholm is the capital of",
        "Mount Fuji is in",
        "The Amazon rainforest is mostly in",
    ],
}


def _capture_last_activation(model, prompt: str, hook_name: str) -> np.ndarray:
    captured: dict = {}

    def grab(act, hook):
        captured["a"] = act

    model.reset_hooks()
    model.add_hook(hook_name, grab, "fwd")
    try:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            model(tokens)
    finally:
        model.reset_hooks()
    a = captured["a"]                          # (1, T, D) bfloat16
    return a[0, -1].float().cpu().numpy()      # (D,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--model-short", default="gemma-2-2b")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--dict-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    np.random.seed(args.seed)

    if args.output_dir is None:
        args.output_dir = Path("results/exp_category_firing") / (
            f"{args.model_short}_L{args.layer}_seed{args.seed}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dictionary %s", args.dict_path)
    with open(args.dict_path, "rb") as f:
        dictionary = pickle.load(f)
    logger.info("Dictionary: %s", dictionary)
    K = len(dictionary.partitions)

    import transformer_lens as tl
    logger.info("Loading model %s on %s", args.model, args.device)
    t0 = time.time()
    model = tl.HookedTransformer.from_pretrained_no_processing(
        args.model, device=args.device, dtype=torch.bfloat16,
    )
    model.eval()
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    logger.info("Model loaded in %.1fs", time.time() - t0)

    logger.info("Running prompts; capturing L%d at last position", args.layer)
    per_prompt: list[dict] = []
    for category, prompts in PROMPTS_BY_CATEGORY.items():
        for prompt in prompts:
            act = _capture_last_activation(model, prompt, hook_name)
            pid, dist = dictionary.assign(act[None])      # ((1,), (1,))
            per_prompt.append({
                "category": category,
                "prompt": prompt,
                "partition_id": int(pid[0]),
                "distance": float(dist[0]),
            })

    # ---- analysis ----
    E = np.stack([p.exemplar_direction for p in dictionary.partitions])
    sim = E @ E.T
    np.clip(sim, -1.0, 1.0, out=sim)
    G = (1.0 - sim).astype(np.float32)
    np.fill_diagonal(G, 0.0)

    cats = list(PROMPTS_BY_CATEGORY.keys())
    by_cat_pids: dict[str, list[int]] = {c: [] for c in cats}
    for r in per_prompt:
        by_cat_pids[r["category"]].append(r["partition_id"])

    def _mean_pairs(pids_a: list[int], pids_b: list[int],
                    same: bool) -> float:
        d = []
        for i in pids_a:
            for j in pids_b:
                if same and i == j:
                    continue
                d.append(G[i, j])
        return float(np.mean(d)) if d else float("nan")

    cohesion = {c: _mean_pairs(by_cat_pids[c], by_cat_pids[c], same=True)
                for c in cats}
    separation: dict[str, float] = {}
    for i, ca in enumerate(cats):
        for cb in cats[i + 1:]:
            separation[f"{ca}|{cb}"] = _mean_pairs(
                by_cat_pids[ca], by_cat_pids[cb], same=False,
            )

    # Permutation null: shuffle category labels across all prompts, recompute
    # mean within-category geometric distance (averaged over categories).
    rng = np.random.default_rng(args.seed)
    all_pids = [r["partition_id"] for r in per_prompt]
    all_cats = [r["category"] for r in per_prompt]
    n = len(all_pids)
    obs_within = float(np.mean(list(cohesion.values())))
    null = np.zeros(args.n_permutations, dtype=np.float32)
    for k in range(args.n_permutations):
        perm_cats = list(all_cats)
        rng.shuffle(perm_cats)
        d = {c: [] for c in cats}
        for c, p in zip(perm_cats, all_pids):
            d[c].append(p)
        within = []
        for c in cats:
            pids = d[c]
            if len(pids) < 2:
                continue
            ds = []
            for i in pids:
                for j in pids:
                    if i != j:
                        ds.append(G[i, j])
            if ds:
                within.append(float(np.mean(ds)))
        null[k] = float(np.mean(within)) if within else float("nan")
    p_value = float((null <= obs_within).mean())

    # Top-3 most-frequent partitions per category
    from collections import Counter
    top_partitions = {}
    for c in cats:
        ctr = Counter(by_cat_pids[c])
        top = ctr.most_common(3)
        top_partitions[c] = [
            {"partition_id": int(pid),
             "n_hits": int(n),
             "member_count": int(dictionary.partitions[pid].member_count)}
            for pid, n in top
        ]

    out = {
        "model": args.model_short,
        "layer": args.layer,
        "K": K,
        "n_prompts_per_category": {c: len(by_cat_pids[c]) for c in cats},
        "n_unique_partitions_per_category": {
            c: len(set(by_cat_pids[c])) for c in cats
        },
        "cohesion": cohesion,
        "separation": separation,
        "obs_within_mean": obs_within,
        "null_within_mean": float(null.mean()),
        "null_within_std": float(null.std()),
        "p_value_left_tail": p_value,
        "top_partitions_per_category": top_partitions,
        "per_prompt": per_prompt,
    }
    out_path = args.output_dir / "category_firing.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_path)

    logger.info("Within-category mean geom distance:")
    for c, v in cohesion.items():
        logger.info("  %-8s  %.4f  (%d unique partitions / %d prompts)",
                    c, v, len(set(by_cat_pids[c])), len(by_cat_pids[c]))
    logger.info("Between-category mean geom distance:")
    for k, v in separation.items():
        logger.info("  %-20s  %.4f", k, v)
    logger.info("Permutation: obs within=%.4f, null mean=%.4f std=%.4f, p=%.4f",
                obs_within, null.mean(), null.std(), p_value)


if __name__ == "__main__":
    main()
