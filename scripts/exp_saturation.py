"""Experiment 2: domain saturation curves.

Build EP libraries online on three domain streams (math, code, chat) using a
single shared geometric reference (calibrated on the Pile). Plot |library|
vs activations-seen for each domain. Differing saturation shapes = different
representational diversity per domain at fixed resolution.

The calibration is computed once on the Pile and reused across domains so
the threshold is identical — without that, per-domain calibration would
shift the radius and saturation curves wouldn't be comparable.

Run:
    uv run python -m scripts.exp_saturation --max-tokens 50000
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)

DOMAINS = ("math", "code", "chat")


def _domain_stream(name: str, seed: int = 0) -> Iterator[str]:
    """Infinite text iterator for one domain."""
    from datasets import load_dataset

    rng = np.random.default_rng(seed)

    def cycle(items: list[str]):
        idx = list(range(len(items)))
        while True:
            rng.shuffle(idx)
            for i in idx:
                yield items[i]

    if name == "math":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        prompts = [f"{r['question']}\n{r['answer']}" for r in ds]
    elif name == "code":
        ds = load_dataset("openai_humaneval", split="test")
        prompts = [r["prompt"] + r["canonical_solution"] for r in ds]
        # HumanEval is small (164); also pull mbpp for more variety.
        try:
            mbpp = load_dataset("mbpp", "sanitized", split="train")
            prompts.extend([r["text"] + "\n" + r["code"] for r in mbpp])
        except Exception:
            pass
    elif name == "chat":
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        # Materialise a fixed pool so we can cycle reproducibly.
        pool = []
        for i, row in enumerate(ds):
            pool.append(row["chosen"])
            if i >= 5000:
                break
        prompts = pool
    else:
        raise ValueError(f"Unknown domain: {name}")

    logger.info("Domain %s: loaded %d prompts", name, len(prompts))
    yield from cycle(prompts)


def _pile_stream(tokenizer, context_length: int = 128, seed: int = 0) -> Iterator[str]:
    """Pile decoder, used only for calibration."""
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    for item in ds:
        text = item.get("text", "")
        if len(text) < 200:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= context_length:
            yield tokenizer.decode(ids[:context_length])


def _saturation_figure(curves: dict[str, list[tuple[int, int]]], title: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for domain, pairs in curves.items():
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, label=domain, linewidth=2)
    ax.set_xlabel("activations seen")
    ax.set_ylabel("|library|")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--model-short", default="gemma-2-2b")
    parser.add_argument("--layers", default="12",
                        help="Comma-separated layer indices. Default '12' = single layer.")
    parser.add_argument("--percentile", type=float, default=8.0)
    parser.add_argument("--calibration-tokens", type=int, default=200_000)
    parser.add_argument("--force-recalibrate", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=10_000_000,
                        help="Per-(domain, layer) budget; matches build_dictionary default.")
    parser.add_argument("--saturation-window", type=int, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domains", default=",".join(DOMAINS),
                        help="Comma-separated subset of math,code,chat")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ep-properties")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layers = [int(x) for x in args.layers.split(",")]

    if args.output_dir is None:
        args.output_dir = Path("results/exp_saturation") / args.model_short
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    import transformer_lens as tl
    logger.info("Loading model %s on %s", args.model, args.device)
    t0 = time.time()
    model = tl.HookedTransformer.from_pretrained_no_processing(
        args.model, device=args.device, dtype=torch.bfloat16,
    )
    model.eval()
    logger.info("Model loaded in %.1fs", time.time() - t0)

    from ep.discovery.calibration import load_or_calibrate
    from ep.discovery.extraction import extract_per_position
    from ep.discovery.pipeline import _iter_activation_batches, discover

    # --- wandb init (one run, log every (domain, layer)) ---
    if args.wandb:
        import wandb
        layers_tag = "_".join(f"L{l}" for l in layers)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"saturation_{args.model_short}_{layers_tag}_p{args.percentile:g}",
            config=vars(args),
            job_type="saturation",
        )

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    # Indexed as curves_by_layer[layer][domain] = [(n_acts, n_partitions), ...].
    curves_by_layer: dict[int, dict[str, list[tuple[int, int]]]] = {}
    summary_rows: list[list] = []

    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        logger.info("============ L%d ============", layer)

        def _calib_batches(_hook=hook_name):
            return _iter_activation_batches(
                model, _pile_stream(model.tokenizer, args.context_length, args.seed),
                _hook,
                extract_fn=extract_per_position,
                extract_kwargs={"batch_size": args.prompt_batch_size},
                prompt_batch_size=args.prompt_batch_size,
                seed=args.seed,
            )

        calibration = load_or_calibrate(
            args.model, hook_name, _calib_batches,
            n_tokens=args.calibration_tokens,
            percentile=args.percentile,
            force=args.force_recalibrate,
        )
        logger.info("Calibration L%d ready: threshold=%.6f",
                    layer, calibration.threshold)

        layer_curves: dict[str, list[tuple[int, int]]] = {}
        for domain in domains:
            logger.info("=== L%d / %s ===", layer, domain)
            snapshots: list[tuple[int, int]] = []

            def log_fn(metrics: dict, _domain=domain, _layer=layer, _snap=snapshots):
                _snap.append((metrics["n_acts"], metrics["n_partitions"]))
                if args.wandb:
                    import wandb
                    payload = {f"saturation/L{_layer}/{_domain}/{k}": v
                               for k, v in metrics.items()}
                    payload[f"saturation/L{_layer}/{_domain}/x_n_acts"] = metrics["n_acts"]
                    wandb.log(payload)

            result = discover(
                model=model,
                texts=_domain_stream(domain, seed=args.seed),
                hook_name=hook_name,
                calibration=calibration,
                extract_fn=extract_per_position,
                extract_kwargs={"batch_size": args.prompt_batch_size},
                log_cadence=1,
                checkpoint_cadence=10_000,
                saturation_window=args.saturation_window,
                max_tokens=args.max_tokens,
                prompt_batch_size=args.prompt_batch_size,
                log_fn=log_fn,
                seed=args.seed,
            )
            layer_curves[domain] = snapshots
            summary_rows.append([
                layer, domain, result.n_activations, len(result.dictionary),
                bool(result.saturated),
            ])
            logger.info(
                "L%d / %s done: %d acts, %d partitions, saturated=%s",
                layer, domain, result.n_activations,
                len(result.dictionary), result.saturated,
            )
        curves_by_layer[layer] = layer_curves

    # --- Final figures + tables ---
    if args.wandb:
        import wandb
        for layer, layer_curves in curves_by_layer.items():
            fig = _saturation_figure(layer_curves,
                                     title=f"L{layer} domain saturation (p={args.percentile:g})")
            wandb.log({f"saturation_curves/L{layer}": wandb.Image(fig)})

        wandb.log({
            "saturation_summary": wandb.Table(
                columns=["layer", "domain", "n_acts", "n_partitions", "saturated"],
                data=summary_rows,
            )
        })
        wandb.finish()

    # Always save curves to disk for reuse.
    import json
    out_path = args.output_dir / "saturation_curves.json"
    payload = {
        "config": {
            "model": args.model, "layers": layers, "percentile": args.percentile,
            "max_tokens": args.max_tokens,
            "saturation_window": args.saturation_window,
            "calibration_tokens": args.calibration_tokens,
            "domains": domains,
        },
        "curves_by_layer": {
            str(layer): {
                d: [[int(a), int(p)] for (a, p) in pairs]
                for d, pairs in layer_curves.items()
            }
            for layer, layer_curves in curves_by_layer.items()
        },
        "summary": [
            {"layer": l, "domain": d, "n_acts": int(na), "n_partitions": int(np_),
             "saturated": bool(s)}
            for (l, d, na, np_, s) in summary_rows
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved curves to %s", out_path)


if __name__ == "__main__":
    main()
