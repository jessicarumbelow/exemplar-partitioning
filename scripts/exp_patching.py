"""Experiment: causal partition patching on factual completion pairs.

Setup: pairs of prompts that share grammatical structure but differ in
expected answer (e.g., "The capital of France is" → Paris vs "The capital of
Germany is" → Berlin). At a given layer L, find each prompt's EP partition
assignment at the last position. Replace prompt-A's L activation with
prompt-B's partition exemplar direction (norm-preserving in centered space)
and measure how the next-token distribution shifts toward the target-B
answer.

Tests whether EP partitions are *causally load-bearing* at specific
(layer, position) pairs, not just correlated with output. Direct counterpart
to classic activation patching but with EP partitions as the swap unit.

Reports:
- flip_rate: fraction of pairs where the top-1 next token becomes the
  target-B answer
- logit_diff_shift: change in (logit(target_B) - logit(target_A)) before
  vs after patch
- kl_to_baseline_B: KL between patched-A distribution and baseline-B
  distribution (lower = patch successfully recovers B's behaviour)

Baselines:
- random_partition: patch with a randomly chosen partition's exemplar
  (controls for direction magnitude)
- self_partition: patch with prompt-A's *own* partition exemplar
  (controls for any-direction-at-this-norm effect)

Run on Modal:
    modal run ep_modal_experiments.py::patching
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

EPS = 1e-12


# Single-token-friendly factual completion templates. Each entry: (template,
# fill, target_answer). Targets are picked to be one BPE token under the
# Gemma tokenizer (verified at runtime; pairs that fail are dropped).
CAPITALS = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Portugal", "Lisbon"),
    ("Russia", "Moscow"),
    ("Japan", "Tokyo"),
    ("China", "Beijing"),
    ("India", "Delhi"),
    ("Egypt", "Cairo"),
    ("Greece", "Athens"),
    ("Turkey", "Ankara"),
    ("Poland", "Warsaw"),
    ("Sweden", "Stockholm"),
    ("Norway", "Oslo"),
    ("Finland", "Helsinki"),
    ("Austria", "Vienna"),
    ("Hungary", "Budapest"),
    ("Ireland", "Dublin"),
    ("Iceland", "Reykjavik"),
]

LANGUAGES = [
    ("France", "French"),
    ("Germany", "German"),
    ("Italy", "Italian"),
    ("Spain", "Spanish"),
    ("Portugal", "Portuguese"),
    ("Russia", "Russian"),
    ("Japan", "Japanese"),
    ("China", "Chinese"),
    ("Greece", "Greek"),
    ("Turkey", "Turkish"),
    ("Poland", "Polish"),
    ("Sweden", "Swedish"),
    ("Norway", "Norwegian"),
    ("Finland", "Finnish"),
    ("Hungary", "Hungarian"),
]

COLORS = [
    ("sky on a clear day", "blue"),
    ("grass", "green"),
    ("ripe banana", "yellow"),
    ("fresh blood", "red"),
    ("snow", "white"),
    ("coal", "black"),
    ("ripe lemon", "yellow"),
    ("dandelion", "yellow"),
    ("eggplant", "purple"),
    ("ripe tomato", "red"),
    ("milk", "white"),
    ("ocean", "blue"),
]

MONTHS_AFTER = [
    ("January", "February"),
    ("February", "March"),
    ("March", "April"),
    ("April", "May"),
    ("May", "June"),
    ("June", "July"),
    ("July", "August"),
    ("August", "September"),
    ("September", "October"),
    ("October", "November"),
    ("November", "December"),
]

DAYS_AFTER = [
    ("Monday", "Tuesday"),
    ("Tuesday", "Wednesday"),
    ("Wednesday", "Thursday"),
    ("Thursday", "Friday"),
    ("Friday", "Saturday"),
    ("Saturday", "Sunday"),
    ("Sunday", "Monday"),
]

CATEGORIES = {
    "capitals":  ("The capital of {} is",                   CAPITALS),
    "languages": ("The official language of {} is",          LANGUAGES),
    "colors":    ("The colour of a {} is typically",         COLORS),
    "months":    ("The month after {} is",                   MONTHS_AFTER),
    "days":      ("The day after {} is",                     DAYS_AFTER),
}


@dataclass
class Item:
    category: str
    fill: str
    target: str
    prompt: str
    target_token_id: int       # first BPE id of " {target}"
    target_str_repr: str       # decoded form of that token (for sanity)


def _build_items(tokenizer) -> list[Item]:
    items: list[Item] = []
    for cat, (tmpl, pairs) in CATEGORIES.items():
        for fill, target in pairs:
            prompt = tmpl.format(fill)
            # Tokenise " <target>" — leading space matters for BPE.
            tok_ids = tokenizer.encode(" " + target, add_special_tokens=False)
            if not tok_ids:
                continue
            tid = tok_ids[0]
            items.append(Item(
                category=cat, fill=fill, target=target, prompt=prompt,
                target_token_id=tid,
                target_str_repr=tokenizer.decode([tid]),
            ))
    logger.info("Built %d items across %d categories", len(items),
                len(CATEGORIES))
    return items


def _find_fill_position(model, prompt: str, fill: str) -> int:
    """Locate the last token of " {fill}" inside the BOS-prefixed tokenisation.

    We patch at the position where the fill-word's identity is concentrated
    in the residual stream — that's where downstream layers read "which X is
    this?". For multi-token fills we use the last subword (the position that
    gathers the full fill identity by attention).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)[0].tolist()
    fill_ids = model.tokenizer.encode(" " + fill, add_special_tokens=False)
    # Search for fill_ids as a contiguous subsequence in tokens.
    for i in range(len(tokens) - len(fill_ids), -1, -1):
        if tokens[i:i + len(fill_ids)] == fill_ids:
            return i + len(fill_ids) - 1
    raise ValueError(f"Could not find fill {fill!r} in prompt {prompt!r}")


def _forward_capture(model, prompt: str, hook_name: str, fill_pos: int,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward `prompt`, return (final-pos logits, fill-pos activation, final-pos activation)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    cache_act = {}

    def cache_hook(act, hook):
        cache_act["a"] = act.detach().cpu().float().numpy()

    model.reset_hooks()
    model.add_hook(hook_name, cache_hook, "fwd")
    try:
        with torch.no_grad():
            logits = model(tokens)
    finally:
        model.reset_hooks()
    final_logits = logits[0, -1].detach().cpu().float().numpy()
    fill_act = cache_act["a"][0, fill_pos]   # activation at fill token
    final_act = cache_act["a"][0, -1]
    return final_logits, fill_act, final_act


def _patched_logits(model, prompt: str, hook_name: str, patch_pos: int,
                    new_act: np.ndarray) -> np.ndarray:
    """Forward `prompt` with a hook that replaces activation at patch_pos."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    new_act_t = torch.from_numpy(new_act.astype(np.float32))

    def patch_hook(act, hook, _new=new_act_t, _pos=patch_pos):
        out = act.clone()
        out[:, _pos] = _new.to(device=act.device, dtype=act.dtype)
        return out

    model.reset_hooks()
    model.add_hook(hook_name, patch_hook, "fwd")
    try:
        with torch.no_grad():
            logits = model(tokens)
    finally:
        model.reset_hooks()
    return logits[0, -1].detach().cpu().float().numpy()


def _norm_preserving_swap(orig_act: np.ndarray,
                          new_direction: np.ndarray,
                          center: np.ndarray) -> np.ndarray:
    """Replace the centered-direction of orig_act with new_direction, keeping centered norm.

    EP exemplars live as unit directions in centered space — this is the
    natural "swap" for activation patching: keep the magnitude of how-far-
    from-the-mean-this-activation-was, change which direction it points.
    """
    centered = orig_act - center
    norm = float(np.linalg.norm(centered)) + EPS
    new_centered = norm * new_direction.astype(np.float64)
    return (new_centered + center).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) over a probability distribution. NaN-safe."""
    eps = 1e-30
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--model-short", default="gemma-2-2b")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--dict-path", type=str, required=True,
                        help="Path to a pickled Dictionary, e.g. "
                             "/vol/dictionaries/gemma-2-2b_L12_p10_..._full/"
                             "dictionaries/gemma-2-2b_layer12.pkl")
    parser.add_argument("--max-pairs", type=int, default=400,
                        help="Cap total (A→B) ordered pairs evaluated.")
    parser.add_argument("--n-random-controls", type=int, default=1,
                        help="Random-partition control draws per pair.")
    parser.add_argument("--basis", choices=("mean", "exemplar"),
                        default="mean",
                        help="Which partition direction to patch with.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ep-properties")
    parser.add_argument("--wandb-entity", default="jessicamarycooper")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = Path("results/exp_patching") / (
            f"{args.model_short}_L{args.layer}_seed{args.seed}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dictionary ---
    logger.info("Loading dictionary from %s", args.dict_path)
    with open(args.dict_path, "rb") as f:
        dictionary = pickle.load(f)
    logger.info("Dictionary: %s", dictionary)
    n_partitions = len(dictionary.partitions)

    # --- Load model ---
    import transformer_lens as tl
    logger.info("Loading model %s on %s", args.model, args.device)
    t0 = time.time()
    model = tl.HookedTransformer.from_pretrained_no_processing(
        args.model, device=args.device, dtype=torch.bfloat16,
    )
    model.eval()
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    logger.info("Model loaded in %.1fs", time.time() - t0)

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"patching_{args.model_short}_L{args.layer}_seed{args.seed}",
            config=vars(args), job_type="patching",
        )

    # --- Build items, run baseline forward pass for each ---
    items = _build_items(model.tokenizer)

    logger.info("Baseline forward pass on %d items (probe at fill-token position)",
                len(items))
    base_logits = np.zeros((len(items), model.cfg.d_vocab), dtype=np.float32)
    base_fill_acts = np.zeros((len(items), model.cfg.d_model), dtype=np.float32)
    fill_positions = np.zeros(len(items), dtype=np.int64)
    base_top1 = np.zeros(len(items), dtype=np.int64)
    base_target_correct = np.zeros(len(items), dtype=bool)
    for i, it in enumerate(items):
        fp = _find_fill_position(model, it.prompt, it.fill)
        fill_positions[i] = fp
        logits, fill_act, _ = _forward_capture(model, it.prompt, hook_name, fp)
        base_logits[i] = logits
        base_fill_acts[i] = fill_act
        base_top1[i] = int(np.argmax(logits))
        base_target_correct[i] = (base_top1[i] == it.target_token_id)
        if (i + 1) % 20 == 0:
            logger.info("  %d/%d  baseline target-correct so far: %.2f",
                        i + 1, len(items), base_target_correct[:i+1].mean())
    logger.info("Baseline target-top1 rate: %.2f", base_target_correct.mean())

    # Partition assignment for each item, AT THE FILL POSITION.
    pids, _ = dictionary.assign(base_fill_acts)
    pid_counts = np.bincount(pids[pids >= 0], minlength=n_partitions)
    logger.info("Items hit %d distinct partitions at fill position; "
                "mode partition has %d items",
                int((pid_counts > 0).sum()), int(pid_counts.max()))

    if args.wandb:
        import wandb
        wandb.summary["baseline/target_correct_rate"] = float(
            base_target_correct.mean()
        )
        wandb.summary["baseline/n_items"] = len(items)
        wandb.summary["baseline/n_distinct_partitions"] = int(
            (pid_counts > 0).sum()
        )

    # --- Build pair list: same-category, distinct partitions, distinct fills.
    pairs: list[tuple[int, int]] = []
    for cat in CATEGORIES:
        idxs = [i for i, it in enumerate(items) if it.category == cat]
        for a in idxs:
            for b in idxs:
                if a == b:
                    continue
                if items[a].fill == items[b].fill:
                    continue
                if pids[a] == pids[b]:
                    # Different prompts in same partition is interesting but
                    # not a useful test of the causal claim — skip.
                    continue
                pairs.append((a, b))
    rng.shuffle(pairs)
    if len(pairs) > args.max_pairs:
        pairs = pairs[:args.max_pairs]
    logger.info("Evaluating %d (A→B) pairs across %d categories",
                len(pairs), len(CATEGORIES))

    # --- Run patches ---
    results = []
    for k, (a, b) in enumerate(pairs):
        item_a, item_b = items[a], items[b]
        ta, tb = item_a.target_token_id, item_b.target_token_id
        patch_pos = int(fill_positions[a])  # patch A's fill-token position

        # 1) Target patch: replace A's fill-token L activation with B's
        #    partition direction, norm-preserving.
        def _dir(pid):
            p = dictionary.partitions[pid]
            return (p.mean_member_direction if args.basis == "mean"
                    else p.exemplar_direction)

        target_dir = _dir(pids[b])
        new_act = _norm_preserving_swap(base_fill_acts[a], target_dir,
                                        dictionary.center)
        patched_logits = _patched_logits(model, item_a.prompt, hook_name,
                                         patch_pos, new_act)

        # 2) Random-partition control: average over a few random partitions.
        rand_logit_arr = np.zeros((args.n_random_controls,
                                   model.cfg.d_vocab), dtype=np.float32)
        for r in range(args.n_random_controls):
            rand_pid = int(rng.integers(0, n_partitions))
            while rand_pid == pids[b] or rand_pid == pids[a]:
                rand_pid = int(rng.integers(0, n_partitions))
            rand_dir = _dir(rand_pid)
            new_rand = _norm_preserving_swap(base_fill_acts[a], rand_dir,
                                             dictionary.center)
            rand_logit_arr[r] = _patched_logits(model, item_a.prompt,
                                                hook_name, patch_pos,
                                                new_rand)
        rand_logits = rand_logit_arr.mean(axis=0)

        # 3) Self-partition control: replace with A's own partition direction.
        self_dir = _dir(pids[a])
        new_self = _norm_preserving_swap(base_fill_acts[a], self_dir,
                                         dictionary.center)
        self_logits = _patched_logits(model, item_a.prompt, hook_name,
                                      patch_pos, new_self)

        # ----- Metrics -----
        # Flip rates: did top-1 become target_b?
        def flipped(logits): return int(np.argmax(logits) == tb)

        # Logit-diff shift: (logit_b - logit_a)_patched - (logit_b - logit_a)_baseline.
        def lg_diff(logits): return float(logits[tb] - logits[ta])

        # KL between patched and baseline-B distribution at the same position.
        # Baseline-B's logits live at base_logits[b], same position semantics.
        base_b_probs = _softmax(base_logits[b])
        patched_probs = _softmax(patched_logits)
        rand_probs    = _softmax(rand_logits)
        self_probs    = _softmax(self_logits)
        base_a_probs  = _softmax(base_logits[a])

        results.append({
            "pair_idx": k,
            "category": item_a.category,
            "a_fill": item_a.fill, "b_fill": item_b.fill,
            "a_target": item_a.target, "b_target": item_b.target,
            "a_pid": int(pids[a]), "b_pid": int(pids[b]),
            "baseline_a_top1_correct": bool(base_target_correct[a]),
            "baseline_b_top1_correct": bool(base_target_correct[b]),
            "flip_target":    flipped(patched_logits),
            "flip_random":    flipped(rand_logits),
            "flip_self":      flipped(self_logits),
            "flip_baseline":  flipped(base_logits[a]),
            "lgdiff_baseline_a": lg_diff(base_logits[a]),
            "lgdiff_target":    lg_diff(patched_logits),
            "lgdiff_random":    lg_diff(rand_logits),
            "lgdiff_self":      lg_diff(self_logits),
            "kl_target_to_baseline_b": _kl(patched_probs, base_b_probs),
            "kl_random_to_baseline_b": _kl(rand_probs, base_b_probs),
            "kl_self_to_baseline_b":   _kl(self_probs, base_b_probs),
            "kl_baseline_a_to_baseline_b": _kl(base_a_probs, base_b_probs),
        })

        if (k + 1) % 25 == 0:
            sub = results
            f_t = np.mean([r["flip_target"] for r in sub])
            f_r = np.mean([r["flip_random"] for r in sub])
            f_s = np.mean([r["flip_self"]   for r in sub])
            f_b = np.mean([r["flip_baseline"] for r in sub])
            logger.info("  %d/%d  flip: target=%.2f random=%.2f self=%.2f baseline=%.2f",
                        k + 1, len(pairs), f_t, f_r, f_s, f_b)

    # --- Aggregate ---
    def agg(field):
        return float(np.mean([r[field] for r in results]))

    summary = {
        "n_pairs": len(results),
        "flip_target":   agg("flip_target"),
        "flip_random":   agg("flip_random"),
        "flip_self":     agg("flip_self"),
        "flip_baseline": agg("flip_baseline"),
        "lgdiff_baseline_a": agg("lgdiff_baseline_a"),
        "lgdiff_target":     agg("lgdiff_target"),
        "lgdiff_random":     agg("lgdiff_random"),
        "lgdiff_self":       agg("lgdiff_self"),
        "kl_target_to_baseline_b": agg("kl_target_to_baseline_b"),
        "kl_random_to_baseline_b": agg("kl_random_to_baseline_b"),
        "kl_self_to_baseline_b":   agg("kl_self_to_baseline_b"),
        "kl_baseline_a_to_baseline_b": agg("kl_baseline_a_to_baseline_b"),
    }

    # Per-category breakdown.
    by_cat = {}
    for cat in CATEGORIES:
        sub = [r for r in results if r["category"] == cat]
        if not sub:
            continue
        by_cat[cat] = {
            "n": len(sub),
            "flip_target":   float(np.mean([r["flip_target"] for r in sub])),
            "flip_random":   float(np.mean([r["flip_random"] for r in sub])),
            "flip_self":     float(np.mean([r["flip_self"]   for r in sub])),
            "lgdiff_target": float(np.mean([r["lgdiff_target"] for r in sub])),
            "lgdiff_random": float(np.mean([r["lgdiff_random"] for r in sub])),
        }

    logger.info("\n=== SUMMARY ===")
    for k, v in summary.items():
        logger.info("  %-32s %s", k, v)
    logger.info("\n=== BY CATEGORY ===")
    for cat, m in by_cat.items():
        logger.info("  %-12s n=%d  flip_target=%.2f  flip_random=%.2f  flip_self=%.2f",
                    cat, m["n"], m["flip_target"], m["flip_random"], m["flip_self"])

    # --- Persist ---
    payload = {
        "config": vars(args) | {"output_dir": str(args.output_dir),
                                "dict_path": str(args.dict_path)},
        "n_partitions": n_partitions,
        "summary": summary,
        "by_category": by_cat,
        "rows": results,
    }
    out_path = args.output_dir / "patching.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Saved results to %s", out_path)

    if args.wandb:
        import wandb
        for k, v in summary.items():
            wandb.summary[f"summary/{k}"] = v
        for cat, m in by_cat.items():
            for k, v in m.items():
                wandb.summary[f"by_cat/{cat}/{k}"] = v
        # Pair-level table for inspection.
        cols = list(results[0].keys())
        tbl = wandb.Table(columns=cols,
                          data=[[r[c] for c in cols] for r in results])
        wandb.log({"pair_results": tbl})
        wandb.finish()


if __name__ == "__main__":
    main()
