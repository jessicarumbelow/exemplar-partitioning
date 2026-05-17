"""Held-out coverage / OOD experiment.

For each (dictionary, corpus) pair, stream a fixed number of activations
through the target model, assign each to its nearest exemplar via
``Dictionary.assign``, and report:
- within-threshold rate: fraction of activations whose nearest-exemplar
  cosine distance is ≤ the calibrated threshold (i.e. the activation
  falls inside an existing partition cell)
- distribution stats: mean / median / p90 / p99 of nearest-exemplar
  distance
- per-dictionary breakdown when given multiple dicts

Corpora:
  pile: in-distribution sample of `monology/pile-uncopyrighted`
  bulgarian_wiki: Bulgarian Wikipedia (different language distribution
    from Pile-dominant English; partial coverage expected)
  random_tokens: uniform-random vocab IDs (out-of-distribution; coverage
    should be near zero)

The point: at sufficient resolution, EP partitions cover the real
held-out distribution but reject random activations. The within-threshold
rate functions as a free distribution-shift / OOD signal — no
auxiliary classifier or confidence head needed.

Run:
    uv run python -m scripts.exp_coverage --dict-paths d1.pkl,d2.pkl,d3.pkl
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _pile_stream(tokenizer, context_length: int, seed: int) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    for item in ds:
        text = item.get("text", "")
        if len(text) < 200:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= context_length:
            yield tokenizer.decode(ids[:context_length])


def _bulgarian_wiki_stream(tokenizer, context_length: int,
                           seed: int) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.bg",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    for item in ds:
        text = item.get("text", "")
        if len(text) < 200:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= context_length:
            yield tokenizer.decode(ids[:context_length])


def _random_token_batches(tokenizer, vocab_size: int,
                          context_length: int, batch_size: int,
                          n_batches: int, seed: int,
                          ) -> Iterator[torch.Tensor]:
    """Yield random-token batches as raw token id tensors (B, T).

    Bypasses the text path entirely — text-decode-then-re-encode would
    discard out-of-vocab ids and thereby smuggle structure into the
    'random' baseline. Instead we go straight to forward(input_ids).
    """
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        # Avoid special tokens (BOS/EOS/PAD if known).
        special = set()
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            v = getattr(tokenizer, attr, None)
            if v is not None:
                special.add(v)
        # Reject sample loop is fine — special set is small.
        ids = torch.randint(0, vocab_size, (batch_size, context_length),
                            generator=g)
        if special:
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for sid in special:
                mask |= (ids == sid)
            # Replace special tokens with id 0 + 1 = 1 (or another safe id).
            ids[mask] = (ids[mask] + 1) % vocab_size
        yield ids


def _activations_from_text(model, prompts: list[str], hook_name: str,
                           context_length: int,
                           batch_size: int = 8) -> np.ndarray:
    """Forward `prompts` and gather L_hook activations across all token
    positions (excluding BOS). Returns ``(N, D)`` numpy array."""
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    out_acts = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        tokenizer.padding_side = "right"
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=context_length,
                        add_special_tokens=False)
        input_ids = enc["input_ids"].to(model.cfg.device)
        attn = enc["attention_mask"].to(model.cfg.device)

        cache = {}

        def cache_hook(act, hook):
            cache["a"] = act.detach().cpu().float()

        model.reset_hooks()
        model.add_hook(hook_name, cache_hook, "fwd")
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            model.reset_hooks()
        a = cache["a"]                         # (B, T, D)
        # Mask out padding tokens. Skip position 0 (BOS-like) too.
        for i in range(a.shape[0]):
            valid = attn[i].cpu().bool().numpy()
            if not valid.any():
                continue
            valid[0] = False
            out_acts.append(a[i][valid].numpy())
    return np.concatenate(out_acts, axis=0)


def _activations_from_token_ids(model, token_id_batches,
                                hook_name: str) -> np.ndarray:
    """Forward random token ids directly, gather activations."""
    out_acts = []
    for input_ids in token_id_batches:
        input_ids = input_ids.to(model.cfg.device)
        cache = {}

        def cache_hook(act, hook):
            cache["a"] = act.detach().cpu().float()

        model.reset_hooks()
        model.add_hook(hook_name, cache_hook, "fwd")
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            model.reset_hooks()
        a = cache["a"]                         # (B, T, D)
        # Skip position 0 (still treated as 'first position' by the model).
        for i in range(a.shape[0]):
            out_acts.append(a[i, 1:].numpy())
    return np.concatenate(out_acts, axis=0)


def _gather_corpus(corpus: str, model, hook_name: str,
                   n_target: int, context_length: int,
                   batch_size: int, seed: int) -> np.ndarray:
    """Gather ~n_target activations from the chosen corpus."""
    if corpus == "random_tokens":
        vocab_size = model.cfg.d_vocab
        # Need ~ceil(n_target / (T-1)) batches for T-1 valid positions per row.
        per_row = max(1, context_length - 1)
        rows_needed = (n_target + per_row - 1) // per_row
        n_batches = (rows_needed + batch_size - 1) // batch_size
        gen = _random_token_batches(model.tokenizer, vocab_size,
                                    context_length, batch_size,
                                    n_batches, seed)
        return _activations_from_token_ids(model, gen, hook_name)[:n_target]

    # Text corpora
    if corpus == "pile":
        stream = _pile_stream(model.tokenizer, context_length, seed)
    elif corpus == "bulgarian_wiki":
        stream = _bulgarian_wiki_stream(model.tokenizer, context_length, seed)
    else:
        raise ValueError(f"unknown corpus: {corpus!r}")

    # Pull enough prompts to fill n_target activations.
    per_row = max(1, context_length - 1)
    rows_needed = (n_target + per_row - 1) // per_row
    prompts = []
    for p in stream:
        prompts.append(p)
        if len(prompts) >= rows_needed:
            break
    return _activations_from_text(model, prompts, hook_name,
                                  context_length, batch_size)[:n_target]


def _coverage_stats(distances: np.ndarray, threshold: float) -> dict:
    distances = np.asarray(distances, dtype=np.float64)
    return {
        "n":                     int(distances.size),
        "threshold":             float(threshold),
        "within_threshold_rate": float((distances <= threshold).mean()),
        "mean_distance":         float(distances.mean()),
        "median_distance":       float(np.median(distances)),
        "p90_distance":          float(np.percentile(distances, 90)),
        "p99_distance":          float(np.percentile(distances, 99)),
        "min_distance":          float(distances.min()),
        "max_distance":          float(distances.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--model-short", default="gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument(
        "--dict-paths", required=True,
        help="Comma-separated paths to dictionary .pkl files (one per "
             "resolution). Labels read from filename.")
    parser.add_argument("--corpora", default="pile,bulgarian_wiki,random_tokens")
    parser.add_argument("--n-activations", type=int, default=20000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ep-properties")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    torch.manual_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = Path("results/exp_coverage") / (
            f"{args.model_short}_L{args.layer}_seed{args.seed}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dict_paths = [p.strip() for p in args.dict_paths.split(",") if p.strip()]
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]

    # --- Load model once ---
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
            name=f"coverage_{args.model_short}_L{args.layer}_seed{args.seed}",
            config=vars(args), job_type="coverage",
        )

    # --- Gather activations once per corpus, reuse across dicts ---
    corpus_acts: dict[str, np.ndarray] = {}
    for corpus in corpora:
        logger.info("Gathering %d activations from %r", args.n_activations, corpus)
        t = time.time()
        acts = _gather_corpus(
            corpus, model, hook_name, args.n_activations,
            args.context_length, args.batch_size, args.seed,
        )
        logger.info("  %r: %d activations in %.1fs (shape %s)",
                    corpus, len(acts), time.time() - t, acts.shape)
        corpus_acts[corpus] = acts

    # --- For each dictionary, assign each corpus's activations ---
    results = {}
    for dict_path in dict_paths:
        path = Path(dict_path)
        # Extract the percentile token (p4, p8, p10, ...) from the parent
        # directory name. Pattern: <model>_L<layer>_p<pct>_ctx<...>_...
        import re as _re
        parts = path.parts
        label = None
        for s in parts:
            m = _re.match(r".*_L\d+_(p\d+(?:p\d+)?)_", s)
            if m:
                label = m.group(1)
                break
        if label is None:
            label = path.stem

        logger.info("Loading dictionary %s (%s)", path, label)
        with open(path, "rb") as f:
            dictionary = pickle.load(f)
        n_partitions = len(dictionary.partitions)
        threshold = dictionary.threshold
        logger.info("  %s: %d partitions, threshold=%.4f",
                    label, n_partitions, threshold)

        per_corpus = {}
        for corpus in corpora:
            acts = corpus_acts[corpus]
            _, dists = dictionary.assign(acts)
            stats = _coverage_stats(dists, threshold)
            per_corpus[corpus] = stats
            logger.info("    %r: within=%.3f  mean_d=%.4f  median_d=%.4f  p90=%.4f  p99=%.4f",
                        corpus, stats["within_threshold_rate"],
                        stats["mean_distance"], stats["median_distance"],
                        stats["p90_distance"], stats["p99_distance"])
            if args.wandb:
                import wandb
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        wandb.summary[f"{label}/{corpus}/{k}"] = v

        results[label] = {
            "dict_path": str(path),
            "n_partitions": n_partitions,
            "threshold": threshold,
            "per_corpus": per_corpus,
        }

    payload = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "results": results,
    }
    out_path = args.output_dir / "coverage.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Saved results to %s", out_path)

    # --- Concise headline table ---
    logger.info("\n=== HEADLINE: within-threshold rate ===")
    logger.info(f"{'dictionary':<10} {'n_part':>7} {'threshold':>10}  " +
                "  ".join(f"{c:>14}" for c in corpora))
    for label, r in results.items():
        cells = "  ".join(
            f"{r['per_corpus'][c]['within_threshold_rate']:>14.3f}"
            for c in corpora
        )
        logger.info(f"{label:<10} {r['n_partitions']:>7} {r['threshold']:>10.4f}  {cells}")

    if args.wandb:
        import wandb
        cols = ["dictionary", "n_partitions", "threshold"] + [
            f"within_{c}" for c in corpora
        ]
        rows = [
            [label, r["n_partitions"], r["threshold"]] +
            [r["per_corpus"][c]["within_threshold_rate"] for c in corpora]
            for label, r in results.items()
        ]
        wandb.log({"coverage": wandb.Table(columns=cols, data=rows)})
        wandb.finish()


if __name__ == "__main__":
    main()
