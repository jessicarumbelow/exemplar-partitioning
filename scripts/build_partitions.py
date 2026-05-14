"""Build an exemplar-partition dictionary, optionally followed by SAEBench evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Callable, Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "baselines" / "SAEBench"))

DEFAULT_MODEL = "google/gemma-2-2b"
DEFAULT_MODEL_SHORT = "gemma-2-2b"
DEFAULT_LAYER = 12

import ep  # noqa: F401 - apply any compatibility shims
from ep.discovery.dictionary import _cosine_pairwise
from ep.discovery.pipeline import DiscoveryResult, discover


def _partition_diameter(p) -> float:
    """Max pairwise cosine distance between sampled member directions.

    True diameter on the sphere, in cosine units. Approximate when
    member_count exceeds the reservoir cap (SAMPLE_RESERVOIR_CAP=30).
    Returns 0.0 if fewer than 2 members were sampled.
    """
    if len(p.sample_members) < 2:
        return 0.0
    members = np.asarray(p.sample_members)
    return round(float(_cosine_pairwise(members).max()), 4)

ALL_EVALS = [
    "autointerp",
    "core",
    "ravel",
    "scr",
    "tpp",
    "sparse_probing",
    "sparse_probing_sae_probes",
    "unlearning",
    "compare_sae",
    "axbench",
]
SAEBENCH_EVALS = {
    "autointerp", "core", "ravel", "scr", "tpp",
    "sparse_probing", "sparse_probing_sae_probes", "unlearning",
}


def parse_eval_types(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(ALL_EVALS)
    evals = [e.strip() for e in raw.split(",") if e.strip()]
    unknown = sorted(set(evals) - set(ALL_EVALS))
    if unknown:
        raise ValueError(f"Unknown eval(s): {unknown}. Options: all,{','.join(ALL_EVALS)}")
    return evals


def format_number(value: float | int | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def config_slug(args: argparse.Namespace) -> str:
    threshold = f"p{format_number(args.percentile)}"
    prompt_cap = f"_mp{args.max_prompts}" if args.max_prompts is not None else ""
    flags = ""
    if args.merge_close:
        flags += "_merge"
    return (
        f"{args.model_short}_L{args.layer}_{threshold}"
        f"_ctx{args.context_length}_mt{args.max_tokens}{prompt_cap}_bs{args.model_batch_size}"
        f"_seed{args.seed}_{args.extractor}_{args.sampling_mode}{flags}"
    )


def activations_slug(args: argparse.Namespace) -> str:
    """Slug for the raw-activation shard cache.

    Drops percentile and merge_close — neither affects the activations that
    discover() sees, so percentile sweeps and merge-variant runs at the same
    seed/model/layer share one cache.
    """
    prompt_cap = f"_mp{args.max_prompts}" if args.max_prompts is not None else ""
    hook_part = ""
    if args.hook_name:
        # Default hook is implied by `layer`; only encode non-default hooks.
        hook_part = "_" + args.hook_name.replace("/", "_").replace(".", "_")
    return (
        f"{args.model_short}_L{args.layer}{hook_part}"
        f"_ctx{args.context_length}_mt{args.max_tokens}{prompt_cap}"
        f"_bs{args.model_batch_size}_seed{args.seed}"
        f"_{args.extractor}_{args.sampling_mode}"
    )


def activations_cache_dir_for(args: argparse.Namespace) -> Path:
    """Path to the shared raw-activation shard cache.

    Sibling of ``args.output_dir`` (so on Modal: ``/vol/dictionaries/<slug>``
    builds shard into ``/vol/dictionaries/activations_cache/<activations_slug>``),
    keyed only on the args that determine the activations themselves.
    """
    return args.output_dir.parent / "activations_cache" / activations_slug(args)


def stream_batches(
    tokenizer,
    context_length: int,
    batch_size: int,
    sampling_mode: Literal["fixed", "random", "full"] = "full",
    seed: int = 0,
    min_chars: int = 200,
):
    """Yield decoded text strings from the Pile, cycling indefinitely.

    - random: context_length × batch_size fresh texts per cycle. Each sequence
      is a random span of length l from a different source text.
    - fixed: batch_size fresh texts per cycle. Each sequence is a prefix t[0:l].
    - full: batch_size full-length texts per cycle. One forward pass per text
      harvests all positions. Use with extract_per_position.
    """
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    pile_iter = iter(ds)

    def load_texts(n: int) -> list[list[int]]:
        t0 = time.time()
        texts = []
        for item in pile_iter:
            text = item.get("text", "")
            if len(text) < min_chars:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) >= context_length:
                texts.append(ids)
                if len(texts) >= n:
                    break
        logger.info("Loaded %d Pile texts in %.1fs", len(texts), time.time() - t0)
        return texts

    if sampling_mode == "full":
        texts_per_cycle = batch_size
    elif sampling_mode == "random":
        texts_per_cycle = context_length * batch_size
    else:  # fixed
        texts_per_cycle = batch_size

    lengths = list(range(1, context_length + 1))

    while True:
        source_texts = load_texts(texts_per_cycle)
        rng.shuffle(lengths)

        if sampling_mode == "full":
            for ids in source_texts:
                yield tokenizer.decode(ids[:context_length])
        elif sampling_mode == "random":
            text_ix = 0
            for seq_len in lengths:
                for _ in range(batch_size):
                    ids = source_texts[text_ix]
                    start = int(rng.integers(0, len(ids) - seq_len + 1))
                    yield tokenizer.decode(ids[start:start + seq_len])
                    text_ix += 1
        else:  # fixed
            for seq_len in lengths:
                for ids in source_texts:
                    yield tokenizer.decode(ids[:seq_len])


# -------------------------------------------------------------- I/O paths

def dictionary_path(output_dir: Path, model_short: str, layer: int) -> Path:
    return output_dir / "dictionaries" / f"{model_short}_layer{layer}.pkl"


def metadata_path(output_dir: Path, model_short: str, layer: int) -> Path:
    return output_dir / "dictionaries" / f"{model_short}_layer{layer}_metadata.json"


def discovery_config(args: argparse.Namespace) -> dict:
    # Identity fields only — anything that affects the resulting dictionary.
    # log_cadence / checkpoint_cadence are observability knobs and don't change
    # the output, so they're deliberately excluded to avoid spurious cache misses.
    return {
        "model": args.model,
        "model_short": args.model_short,
        "layer": args.layer,
        "hook_name": args.hook_name,
        "context_length": args.context_length,
        "model_batch_size": args.model_batch_size,
        "max_tokens": args.max_tokens,
        "max_prompts": args.max_prompts,
        "sampling_mode": args.sampling_mode,
        "percentile": args.percentile,
        "calibration_tokens": args.calibration_tokens,
        "seed": args.seed,
        "extractor": args.extractor,
        "merge_close": args.merge_close,
    }


def cached_config_is_compatible(args: argparse.Namespace) -> bool:
    path = metadata_path(args.output_dir, args.model_short, args.layer)
    if not path.exists():
        logger.info("No metadata at %s; cached dictionary will not be reused", path)
        return False
    with path.open() as f:
        metadata = json.load(f)
    cached = metadata.get("discovery_config", {})
    expected = discovery_config(args)
    mismatches = {
        key: {"cached": cached.get(key), "expected": value}
        for key, value in expected.items()
        if cached.get(key) != value
    }
    if mismatches:
        logger.info("Cached metadata does not match requested discovery config:")
        for key, values in mismatches.items():
            logger.info("  %s: cached=%r expected=%r",
                        key, values["cached"], values["expected"])
        return False
    return True


def load_dictionary(output_dir: Path, model_short: str, layer: int):
    path = dictionary_path(output_dir, model_short, layer)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        dictionary = pickle.load(f)
    logger.info("Loaded dictionary (%d partitions) from %s", len(dictionary), path)
    return dictionary


def save_dictionary(
    output_dir: Path,
    model_short: str,
    layer: int,
    result: DiscoveryResult,
    args: argparse.Namespace,
) -> dict[str, object]:
    out_dir = output_dir / "dictionaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = dictionary_path(output_dir, model_short, layer)
    # Atomic write: dump to .tmp, then rename. A crash mid-dump leaves no
    # .pkl on the volume, so the cache stays consistent and downstream
    # `load_dictionary` calls raise FileNotFoundError instead of EOFError on
    # a truncated file.
    tmp_path = pkl_path.with_suffix(pkl_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(result.dictionary, f)
    tmp_path.replace(pkl_path)
    logger.info("  dictionary -> %s", pkl_path)

    metadata = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "discovery_config": discovery_config(args),
        "discovery": {
            "n_partitions": len(result.dictionary),
            "n_activations": result.n_activations,
            "saturated": result.saturated,
            "elapsed_s": result.elapsed_s,
            "clustering_time_s": result.clustering_time_s,
            "extraction_time_s": result.extraction_time_s,
            "threshold": result.dictionary.threshold,
            "snapshots": [
                {
                    "n_acts": s.n_activations,
                    "n_prompts": s.n_prompts,
                    "n_partitions": s.n_partitions,
                    "new_rate": s.new_partition_rate,
                    "elapsed_s": s.elapsed_s,
                }
                for s in result.snapshots
            ],
        },
        "n_forward_passes": result.n_forward_passes,
        "n_prompts": result.n_prompts,
        "n_tokens": result.n_tokens,
    }

    with metadata_path(output_dir, model_short, layer).open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    run_path = output_dir / f"run_{int(time.time())}.json"
    with run_path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved run metadata to %s", run_path)

    return {"dictionary": result.dictionary, "metadata": metadata}


# ----------------------------------------------------------- wandb logging

def _wandb_init(args: argparse.Namespace, default_name: str) -> None:
    """Init or resume a wandb run, depending on whether --wandb-run-id is set.

    Resume mode is used by the Modal aggregator so the build phase + every
    fan-out eval ultimately log into a single run instead of one-per-stage.
    """
    import wandb
    init_kwargs = dict(
        project=args.wandb_project,
        entity="jessicamarycooper",
        config=vars(args),
    )
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "must"
    else:
        init_kwargs["name"] = default_name
    wandb.init(**init_kwargs)


def _save_wandb_run_id(output_dir: Path) -> None:
    """Persist the active wandb run id so subsequent stages can resume it."""
    import wandb
    if wandb.run is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wandb_run.json").write_text(
        json.dumps({"id": wandb.run.id, "name": wandb.run.name})
    )


def _top_attributed_tokens(
    prompt_scores: list[tuple[list[str], np.ndarray]], k: int = 8,
) -> str:
    """Top-k tokens by attribution aggregated across prompts.

    Per-prompt min-max normalisation so prompts contribute comparably regardless
    of absolute gradient magnitude, then sum across positions and prompts."""
    from collections import defaultdict
    totals: dict[str, float] = defaultdict(float)
    for token_strs, scores in prompt_scores:
        if scores.size == 0:
            continue
        m = float(scores.max()) or 1.0
        for tok, s in zip(token_strs, scores):
            t = tok.strip()
            if not t:
                continue
            totals[t] += float(s) / m
    if not totals:
        return ""
    top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return ", ".join(t for t, _ in top)


def _top_vocab_tokens(vec: np.ndarray, model, k: int = 8) -> str:
    with torch.no_grad():
        v = torch.tensor(vec, dtype=model.W_U.dtype, device=model.W_U.device)
        v = model.ln_final(v)
        logits = v @ model.W_U
        top_ids = logits.topk(k).indices.cpu().tolist()
    tokens = [model.tokenizer.decode([tid]).strip() for tid in top_ids]
    return ", ".join(t for t in tokens if t)


def _gradient_attributions(
    model,
    prompt: str,
    position: int,
    exemplar_direction: np.ndarray,
    hook_name: str,
    center: np.ndarray,
) -> tuple[list[str], np.ndarray] | None:
    """Per-token attribution by gradient of cosine-distance(h(x)[pos], exemplar) wrt input embeddings.

    Faithful to assignment: applies the same centering the dictionary used.
    Score per token is |emb · ∂dist/∂emb| (input × gradient). Returns
    ``(token_strs, scores)`` or ``None`` if the prompt/position is unusable.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    L = tokens.shape[1]
    if position < 1 or position >= L:
        return None

    embed_device = model.W_E.device
    embeds = model.W_E[tokens[0]].clone().detach().requires_grad_(True)  # (L, d)

    captured: dict = {}

    def replace_embed(act, hook):
        return embeds.unsqueeze(0)

    def grab(act, hook):
        captured["a"] = act

    model.reset_hooks()
    model.add_hook("hook_embed", replace_embed, "fwd")
    model.add_hook(hook_name, grab, "fwd")
    try:
        model(tokens.to(embed_device))
    finally:
        model.reset_hooks()

    a_pos = captured["a"][0, position, :]
    c = torch.tensor(center, dtype=a_pos.dtype, device=a_pos.device)
    e_dir = torch.tensor(exemplar_direction, dtype=a_pos.dtype, device=a_pos.device)
    a_centered = a_pos - c
    a_dir = a_centered / (a_centered.norm() + 1e-12)
    dist = 1.0 - (a_dir * e_dir).sum()

    dist.backward()
    grad = embeds.grad
    scores = (grad * embeds).sum(dim=-1).abs().detach().float().cpu().numpy()
    token_strs = [model.tokenizer.decode([t.item()]) for t in tokens[0]]
    return token_strs, scores


def _render_highlighted_html(
    token_strs: list[str], scores: np.ndarray, focus_position: int | None = None
) -> str:
    """Tokens with orange backgrounds proportional to attribution score.

    Score is min-max normalised within the prompt. ``focus_position`` (the
    position whose activation defined the distance) gets a blue underline so
    the reader can see where the partition was assigned.
    """
    import html as _html
    if scores.size == 0:
        return ""
    max_score = float(scores.max()) or 1.0
    parts = ['<div style="line-height:1.6;'
             'white-space:pre-wrap;word-break:break-word;">']
    for i, (tok, s) in enumerate(zip(token_strs, scores)):
        alpha = float(s) / max_score
        bg = f"rgba(255,140,0,{alpha:.3f})"
        safe = _html.escape(tok).replace("\n", "↵")
        underline = ("border-bottom:2px solid #1e88e5;"
                     if focus_position is not None and i == focus_position else "")
        parts.append(
            f'<span style="background:{bg};padding:0 1px;{underline}">{safe}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


def _log_partition_attributions(
    dictionary,
    model,
    hook_name: str,
    prompts_per_partition: int = 5,
    sample_size: int = 200,
    top_k: int = 20,
    seed: int = 0,
) -> None:
    """One row per sampled partition with gradient-attributed nearest + boundary prompts.

    Cost is ``sample_size × prompts_per_partition × 2`` forward+backward
    passes, so for large dictionaries we sample rather than render all
    partitions. The sample combines the ``top_k`` largest partitions
    (by member count) with a uniform-random draw from the rest — keeps
    the heavy hitters and surfaces small clusters, which are often the
    most distinctive. Logged at finalise as wandb table
    ``all_partitions_attribution``. Uses the dictionary's center so
    attribution is faithful to the assignment that put each prompt in
    this partition.
    """
    import wandb

    if not dictionary.partitions:
        return

    center = dictionary.center
    n_partitions = len(dictionary.partitions)

    # Pick which partitions to render: top_k by member count, plus a random
    # sample from the rest. Reproducible via seed.
    if n_partitions <= sample_size:
        sampled_idxs = np.arange(n_partitions, dtype=np.int64)
    else:
        member_counts = np.array(
            [p.member_count for p in dictionary.partitions], dtype=np.int64,
        )
        order = np.argsort(-member_counts, kind="stable")
        k = min(top_k, sample_size)
        top = order[:k]
        rest = order[k:]
        n_random = sample_size - k
        rng = np.random.default_rng(seed)
        random_pick = rng.choice(rest, size=n_random, replace=False)
        sampled_idxs = np.sort(np.concatenate([top, random_pick]))
    n_to_render = len(sampled_idxs)

    def _render_cell(
        items: list[tuple[float, str, int]], reverse: bool,
    ) -> tuple[str, list[tuple[list[str], np.ndarray]]]:
        chosen = sorted(items, reverse=reverse)[:prompts_per_partition]
        rendered: list[str] = []
        attributions: list[tuple[list[str], np.ndarray]] = []
        for entry in chosen:
            _, prompt, pos = entry
            dist = -entry[0] if not reverse else entry[0]
            attr = _gradient_attributions(
                model, prompt, pos, p.exemplar_direction, hook_name, center,
            )
            if attr is None:
                continue
            token_strs, scores = attr
            attributions.append((token_strs, scores))
            html = _render_highlighted_html(token_strs, scores, focus_position=pos)
            rendered.append(
                f'<div style="margin-bottom:10px;">'
                f'<div style="color:#666;">d={dist:.4f} pos={pos}</div>'
                f'{html}</div>'
            )
        return "".join(rendered) if rendered else "", attributions

    rows = []
    t_start = time.time()
    for n_done, idx in enumerate(sampled_idxs, start=1):
        idx = int(idx)
        p = dictionary.partitions[idx]
        diameter = _partition_diameter(p)
        top_toks = _top_vocab_tokens(p.exemplar_direction, model)
        top_toks_mean = _top_vocab_tokens(p.mean_member_direction, model)
        nearest_html, nearest_attr = _render_cell(p.sample_prompts, reverse=False)
        boundary_html, boundary_attr = _render_cell(p.boundary_prompts, reverse=True)
        top_attr_nearest = _top_attributed_tokens(nearest_attr)
        top_attr_boundary = _top_attributed_tokens(boundary_attr)
        rows.append([
            idx, p.member_count, diameter, top_toks, top_toks_mean,
            top_attr_nearest, top_attr_boundary,
            wandb.Html(nearest_html), wandb.Html(boundary_html),
        ])
        if n_done % 50 == 0:
            logger.info(
                "  attribution: %d/%d sampled partitions (%.0fs)",
                n_done, n_to_render, time.time() - t_start,
            )

    columns = ["partition_id", "members", "diameter",
               "unembed_top_tokens", "unembed_top_tokens_member_mean",
               "top_attributed_tokens_nearest", "top_attributed_tokens_boundary",
               "highlighted_nearest", "highlighted_boundary"]
    table = wandb.Table(columns=columns, data=rows)
    wandb.log({"all_partitions_attribution": table})
    logger.info(
        "Logged attribution table: %d/%d partitions in %.0fs",
        n_to_render, n_partitions, time.time() - t_start,
    )


def _clip_polygon_to_bbox(
    poly: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float,
) -> np.ndarray:
    """Sutherland-Hodgman clip of a 2D polygon to an axis-aligned bbox."""
    def clip(pts: np.ndarray, axis: int, value: float, keep_greater: bool) -> np.ndarray:
        if len(pts) == 0:
            return pts
        out = []
        n = len(pts)
        for i in range(n):
            curr = pts[i]
            prev = pts[(i - 1) % n]
            curr_in = curr[axis] >= value if keep_greater else curr[axis] <= value
            prev_in = prev[axis] >= value if keep_greater else prev[axis] <= value
            if curr_in:
                if not prev_in:
                    t = (value - prev[axis]) / (curr[axis] - prev[axis])
                    out.append(prev + t * (curr - prev))
                out.append(curr)
            elif prev_in:
                t = (value - prev[axis]) / (curr[axis] - prev[axis])
                out.append(prev + t * (curr - prev))
        return np.array(out) if out else np.empty((0, 2))

    pts = clip(poly, 0, xmin, keep_greater=True)
    pts = clip(pts, 0, xmax, keep_greater=False)
    pts = clip(pts, 1, ymin, keep_greater=True)
    pts = clip(pts, 1, ymax, keep_greater=False)
    return pts


def _polygon_centroid(poly: np.ndarray) -> np.ndarray | None:
    """Area-weighted centroid of a 2D polygon. Falls back to the vertex mean
    for degenerate (collinear or empty) polygons."""
    if len(poly) == 0:
        return None
    if len(poly) < 3:
        return poly.mean(axis=0)
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * np.sum(cross)
    if abs(area) < 1e-12:
        return poly.mean(axis=0)
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6 * area)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6 * area)
    return np.array([cx, cy])


def _voronoi_polygons(coords: np.ndarray, radius: float) -> list[np.ndarray]:
    """Finite Voronoi polygon (one per input point) for 2D coords. Unbounded
    regions are closed off by extending the missing ridge by `radius`; choose
    radius >> viewport so matplotlib's axis clipping does the trimming."""
    from scipy.spatial import Voronoi
    vor = Voronoi(coords)
    centre = coords.mean(axis=0)

    ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges.setdefault(p1, []).append((p2, v1, v2))
        ridges.setdefault(p2, []).append((p1, v1, v2))

    new_vertices = vor.vertices.tolist()
    polygons: list[np.ndarray] = []
    for p_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        verts = [v for v in region if v >= 0]
        for p2, v1, v2 in ridges.get(p_idx, []):
            if v1 >= 0 and v2 >= 0:
                continue
            finite_v = v1 if v1 >= 0 else v2
            tangent = coords[p2] - coords[p_idx]
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = (coords[p_idx] + coords[p2]) / 2
            direction = np.sign(np.dot(midpoint - centre, normal)) or 1
            far_point = vor.vertices[finite_v] + direction * normal * radius
            new_vertices.append(far_point.tolist())
            verts.append(len(new_vertices) - 1)
        pts = np.array([new_vertices[v] for v in verts])
        anchor = coords[p_idx]
        angles = np.arctan2(pts[:, 1] - anchor[1], pts[:, 0] - anchor[0])
        polygons.append(np.array([new_vertices[v] for _, v in sorted(zip(angles, verts))]))
    return polygons


def _wandb_checkpoint(
    dictionary,
    stats: dict,
    model=None,
    hook_name: str | None = None,
    device: str = "cpu",
) -> None:
    """Log a snapshot of the dictionary to wandb: size histogram, top-partitions
    table, intra/inter distance histogram, coverage curve, exemplar PCA scatter."""
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = stats["total_acts"]
    if not dictionary.partitions:
        return

    members = sorted([p.member_count for p in dictionary.partitions], reverse=True)
    n_partitions = len(members)

    # Size histogram.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(members, bins=min(50, n_partitions), color="#4a90d9",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Members per partition")
    ax.set_ylabel("Count")
    ax.set_title(f"Partition size distribution ({n_partitions} partitions, {step} acts)")
    ax.set_yscale("log")
    fig.tight_layout()
    wandb.log({"size_histogram": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    # Top-partitions table.
    top_n = min(20, n_partitions)
    ranked_ids = sorted(
        range(n_partitions),
        key=lambda j: dictionary.partitions[j].member_count,
        reverse=True,
    )[:top_n]

    table_rows = []
    for rank, idx in enumerate(ranked_ids):
        p = dictionary.partitions[idx]
        nearest = [text for _, text, _ in sorted(p.sample_prompts)[:5]]
        boundary = [text for _, text, _ in sorted(p.boundary_prompts, reverse=True)[:5]]
        diameter = _partition_diameter(p)
        top_toks = _top_vocab_tokens(p.exemplar_direction, model) if model is not None else ""
        top_toks_mean = (
            _top_vocab_tokens(p.mean_member_direction, model)
            if model is not None else ""
        )
        table_rows.append([
            rank + 1, idx, p.member_count, diameter, top_toks, top_toks_mean,
            " | ".join(nearest), " | ".join(boundary),
        ])

    columns = ["rank", "partition_id", "members", "diameter",
               "unembed_top_tokens", "unembed_top_tokens_member_mean",
               "nearest_prompts", "boundary_prompts"]
    table = wandb.Table(columns=columns, data=table_rows)
    wandb.log({"top_partitions": table}, commit=False)

    # Intra/inter histogram.
    dd = dictionary.distance_distributions(min_members=2)
    if len(dd["intra"]) >= 2 and len(dd["inter"]) >= 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dd["intra"], bins=30, alpha=0.6, color="#4a90d9",
                label="intra (member→exemplar)", density=True)
        ax.hist(dd["inter"], bins=30, alpha=0.6, color="#e74c3c",
                label="inter (exemplar↔exemplar)", density=True)
        ax.axvline(x=dictionary.threshold, color="black", linestyle="--",
                   linewidth=1.5, label=f"θ = {dictionary.threshold:.4f}")
        ax.set_xlabel("Cosine distance")
        ax.set_ylabel("Density")
        ax.set_title(f"Intra vs inter distance ({n_partitions} partitions)")
        ax.legend()
        fig.tight_layout()
        wandb.log({"separation": wandb.Image(fig)}, commit=False)
        plt.close(fig)

        wandb.log({
            "mean_intra_dist": float(np.mean(dd["intra"])),
            "mean_inter_dist": float(np.mean(dd["inter"])),
            "separation_ratio": float(np.mean(dd["inter"])
                                      / max(float(np.mean(dd["intra"])), 1e-12)),
        }, commit=False)

    # Coverage curve.
    cumsum = np.cumsum(members)
    total = cumsum[-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, n_partitions + 1), cumsum / total * 100,
            color="#4a90d9", linewidth=2)
    ax.set_xlabel("Top N partitions (ranked by size)")
    ax.set_ylabel("% of all activations covered")
    ax.set_title(f"Cumulative coverage ({step} acts)")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.5)
    n_50 = int(np.searchsorted(cumsum / total, 0.5)) + 1
    n_90 = int(np.searchsorted(cumsum / total, 0.9)) + 1
    ax.annotate(f"50% at {n_50}", xy=(n_50, 50), fontsize=9, color="gray")
    ax.annotate(f"90% at {n_90}", xy=(n_90, 90), fontsize=9, color="gray")
    fig.tight_layout()
    wandb.log({"coverage_curve": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    # Exemplar PCA scatter.
    if n_partitions >= 5:
        from sklearn.decomposition import PCA
        unit = np.stack([p.exemplar_direction for p in dictionary.partitions])
        sizes = np.array([p.member_count for p in dictionary.partitions])
        coords = PCA(n_components=2).fit_transform(unit)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)

        # Lock viewport before drawing Voronoi: infinite ridges otherwise
        # expand the data limits and squish the exemplars into the middle.
        # Percentile bbox keeps a few outlier exemplars from dominating.
        bbox_min = np.percentile(coords, 1, axis=0)
        bbox_max = np.percentile(coords, 99, axis=0)
        centre_xy = (bbox_min + bbox_max) / 2
        half = float((bbox_max - bbox_min).max()) / 2
        pad = 0.08 * (half * 2) or 1.0
        xlim = (centre_xy[0] - half - pad, centre_xy[0] + half + pad)
        ylim = (centre_xy[1] - half - pad, centre_xy[1] + half + pad)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_autoscale_on(False)

        # Cells coloured by member count (cividis_r so darker = denser); no
        # exemplar dots — overlap in the 2D PCA projection implies a precision
        # the projection doesn't have.
        from matplotlib.collections import PolyCollection
        polygons = _voronoi_polygons(coords, radius=(half + pad) * 8)
        cells = PolyCollection(
            polygons, array=np.log1p(sizes), cmap="cividis_r",
            edgecolors="none", alpha=0.9, zorder=1,
        )
        ax.add_collection(cells)
        cbar = fig.colorbar(cells, ax=ax, shrink=0.6)
        cbar.set_label("log(1 + members)")

        # Random sample (not top-N): largest partitions tend to capture diffuse
        # structure (punctuation, function words); a random sample shows breadth.
        n_labels = min(15, n_partitions)
        rng = np.random.default_rng(seed=step)
        label_idxs = rng.choice(n_partitions, size=n_labels, replace=False)
        label_pairs: list[tuple[int, str]] = []
        for idx in label_idxs:
            p = dictionary.partitions[int(idx)]
            text = p.label
            if not text and model is not None:
                try:
                    text = _top_vocab_tokens(p.exemplar_direction, model, k=3)
                except Exception:
                    text = None
            if text:
                label_pairs.append((int(idx), text))

        # Stack labels on the left margin sorted by y so leader lines don't
        # cross more than necessary; xytext in axes-fraction lands them outside
        # the data area. Anchor leaders at the visible polygon centroid (clipped
        # to viewport) so the line lands inside the cell, not at the off-centre
        # exemplar point.
        if label_pairs:
            centroids = []
            for poly in polygons:
                clipped = _clip_polygon_to_bbox(poly, xlim[0], xlim[1],
                                                ylim[0], ylim[1])
                c = _polygon_centroid(clipped)
                centroids.append(c if c is not None else poly.mean(axis=0))
            centroids = np.array(centroids)

            label_pairs.sort(key=lambda pair: centroids[pair[0], 1], reverse=True)
            label_ys = np.linspace(0.97, 0.03, len(label_pairs))
            for (idx, text), ly in zip(label_pairs, label_ys):
                ax.annotate(
                    text,
                    xy=(centroids[idx, 0], centroids[idx, 1]), xycoords="data",
                    xytext=(-0.16, ly), textcoords="axes fraction",
                    ha="right", va="center", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="0.15",
                                    linewidth=0.7, alpha=0.85,
                                    relpos=(1.0, 0.5)),
                    annotation_clip=False,
                    parse_math=False,
                )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Exemplar PCA ({n_partitions} partitions, {step} acts)")
        fig.tight_layout()
        if label_pairs:
            fig.subplots_adjust(left=0.34)
        wandb.log({"exemplar_pca": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    wandb.log({}, commit=True)


# ------------------------------------------------------- build entry point

def build_dictionary(
    model,
    texts,
    *,
    hook_name: str,
    output_dir: Path | None = None,
    model_short: str = DEFAULT_MODEL_SHORT,
    layer: int = DEFAULT_LAYER,
    model_name: str | None = None,
    percentile: float = 10,
    calibration_tokens: int = 200_000,
    calibration_extras: Mapping[str, object] | None = None,
    force_recalibrate: bool = False,
    max_tokens: int = 1_000_000,
    max_prompts: int | None = None,
    prompt_batch_size: int = 128,
    log_cadence: int = 1,
    checkpoint_cadence: int = 10,
    saturation_window: int = 1,
    extract_fn: Callable | None = None,
    extract_kwargs: dict | None = None,
    seed: int = 0,
    device: str = "cpu",
    use_wandb: bool = False,
    merge_close: bool = False,
    activations_cache_dir: Path | None = None,
    log_attribution: bool = True,
    attribution_prompts_per_partition: int = 5,
    attribution_sample_size: int = 200,
    attribution_top_k: int = 20,
    skip_logs: bool = False,
) -> DiscoveryResult:
    """Stream activations from texts and build an exemplar-partition dictionary.

    If ``use_wandb`` is True, expects ``wandb`` to already be initialised.
    If ``activations_cache_dir`` is set, raw activations are sharded to disk
    so downstream tools (compare_sae) can skip re-running the model.
    """
    log_fn = None
    if use_wandb:
        import wandb
        log_fn = wandb.log

    if output_dir is not None:
        output_dir = Path(output_dir)

    # Periodic in-run wandb snapshot only when wandb is on. The final
    # dictionary is pickled once at finalise via save_dictionary; intermediate
    # disk pickles are not consumed downstream, so we don't write them.
    checkpoint_fn = None
    if use_wandb and not skip_logs:
        def checkpoint_fn(dictionary, snapshots, stats):
            _wandb_checkpoint(dictionary, stats, model=model,
                              hook_name=hook_name, device=device)
            logger.info("Wandb snapshot at %d acts: %d partitions",
                        stats["total_acts"], len(dictionary))

    from ep.discovery.calibration import (
        load as _load_calibration, save as _save_calibration,
    )
    from ep.discovery.pipeline import calibrate_pipeline

    cache_model_name = model_name or model_short
    cal_extras = dict(calibration_extras) if calibration_extras else None
    calibration = (
        None if force_recalibrate
        else _load_calibration(cache_model_name, hook_name, percentile, cal_extras)
    )
    if calibration is None or calibration.n_activations < calibration_tokens:
        logger.info("Calibrating: %d tokens @ p%g (cache miss or insufficient)",
                    calibration_tokens, percentile)
        calibration = calibrate_pipeline(
            model=model, texts=texts, hook_name=hook_name,
            n_tokens=calibration_tokens, percentile=percentile,
            extract_fn=extract_fn, extract_kwargs=extract_kwargs or {},
            prompt_batch_size=prompt_batch_size, seed=seed,
        )
        _save_calibration(cache_model_name, hook_name, calibration, cal_extras)
    else:
        logger.info("Calibration cache hit (n_activations=%d)", calibration.n_activations)
    logger.info("Calibration: ||center||=%.4f θ=%.6f (n_acts=%d)",
                float(np.linalg.norm(calibration.center)), calibration.threshold,
                calibration.n_activations)

    result = discover(
        model=model,
        texts=texts,
        hook_name=hook_name,
        calibration=calibration,
        extract_fn=extract_fn,
        extract_kwargs=extract_kwargs or {},
        log_cadence=log_cadence,
        checkpoint_cadence=checkpoint_cadence,
        saturation_window=saturation_window,
        max_tokens=max_tokens,
        max_prompts=max_prompts,
        prompt_batch_size=prompt_batch_size,
        checkpoint_fn=checkpoint_fn,
        log_fn=log_fn,
        seed=seed,
        merge_close=merge_close,
        activations_cache_dir=activations_cache_dir,
    )

    if use_wandb:
        import wandb
        members = sorted(
            [p.member_count for p in result.dictionary.partitions], reverse=True
        ) if result.dictionary.partitions else []
        wandb.summary["final_partitions"] = len(result.dictionary)
        wandb.summary["threshold"] = result.dictionary.threshold
        if members:
            wandb.summary["largest_partition"] = members[0]
            wandb.summary["median_members"] = float(np.median(members))
            wandb.summary["mean_members"] = float(np.mean(members))
            wandb.summary["singletons"] = sum(1 for m in members if m == 1)
            wandb.summary["partitions_gt10"] = sum(1 for m in members if m > 10)
            wandb.summary["partitions_gt100"] = sum(1 for m in members if m > 100)
        wandb.summary["extraction_time_s"] = result.extraction_time_s
        wandb.summary["clustering_time_s"] = result.clustering_time_s
        wandb.summary["method_runtime_s"] = (
            result.extraction_time_s + result.clustering_time_s
        )
        _wandb_checkpoint(result.dictionary, {"total_acts": result.n_activations},
                          model=model, hook_name=hook_name, device=device)
        if log_attribution:
            _log_partition_attributions(
                result.dictionary, model, hook_name,
                prompts_per_partition=attribution_prompts_per_partition,
                sample_size=attribution_sample_size,
                top_k=attribution_top_k,
                seed=seed,
            )

    return result


# ----------------------------------------------------------- SAEBench eval

# Sparse-probing-style evals benefit from full alignment information across
# every partition, so they get the dense signed-projection readout. All other
# SAEBench evals expect a sparse code and get top-1 VQ.
PROBING_EVALS = {"sparse_probing", "sparse_probing_sae_probes"}


def _readout_for(eval_type: str) -> tuple[str, int]:
    if eval_type in PROBING_EVALS:
        return ("signed", 1)
    return ("topk", 1)


def _run_scr_or_tpp(eval_type: str, args, cfg, selected_saes) -> None:
    """Direct invocation of scr_and_tpp.run_eval with n_values filtered to the
    smallest d_sae across selected_saes. Bypasses the runner wrapper, which
    hard-codes the default n_values=[..., 500] and asserts on dictionaries
    smaller than that."""
    from sae_bench.custom_saes.run_all_evals_custom_saes import RANDOM_SEED
    from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
    from sae_bench.evals.scr_and_tpp.main import run_eval as scr_run_eval

    default_n_values = ScrAndTppEvalConfig.model_fields["n_values"].default_factory()
    min_d_sae = min(sae.cfg.d_sae for _, sae in selected_saes)
    n_values = [n for n in default_n_values if n <= min_d_sae]
    if not n_values:
        logger.warning(
            "%s: no n_values fit (min d_sae=%d); skipping.", eval_type, min_d_sae,
        )
        return
    if n_values != default_n_values:
        logger.info(
            "%s: filtered n_values to %s (min d_sae=%d).",
            eval_type, n_values, min_d_sae,
        )

    config = ScrAndTppEvalConfig(
        model_name=args.model_short,
        random_seed=RANDOM_SEED,
        perform_scr=(eval_type == "scr"),
        llm_batch_size=cfg["batch_size"],
        llm_dtype=cfg["dtype"],
        n_values=n_values,
    )
    scr_run_eval(
        config,
        selected_saes,
        args.device,
        "eval_results",
        args.force_rerun,
        clean_up_activations=True,
        save_activations=args.save_activations,
    )


def _run_saebench(args, dictionary, eval_types: list[str]) -> None:
    import os
    import sae_bench.custom_saes.run_all_evals_custom_saes as saebench_runner
    import sae_bench.custom_saes.run_all_evals_dictionary_learning_saes as saebench_dict
    import sae_bench.sae_bench_utils.general_utils as general_utils
    from ep.saebench_adapter import EPDictionarySAE

    saebench_configs = dict(saebench_runner.MODEL_CONFIGS)
    saebench_configs.update(saebench_dict.MODEL_CONFIGS)
    if args.model_short not in saebench_configs:
        logger.warning(
            "SAEBench eval skipped: model_short=%s not supported "
            "(supported: %s).",
            args.model_short, sorted(saebench_configs),
        )
        return
    cfg = saebench_configs[args.model_short]
    dtype = general_utils.str_to_dtype(cfg["dtype"])

    api_key = args.api_key
    if api_key is None and args.api_key_file is not None and args.api_key_file.exists():
        api_key = args.api_key_file.read_text().strip()

    # Identity SAE is readout-independent — build once, reuse across evals.
    identity_entry = None
    if not args.no_identity_baseline:
        from sae_bench.custom_saes.identity_sae import IdentitySAE

        d_in = dictionary.center.shape[0]
        identity_name = f"identity_{args.model_short}_layer_{args.layer}"
        if not args.aggregate_only:
            identity_sae = IdentitySAE(
                d_in=d_in,
                model_name=args.model_short,
                hook_layer=args.layer,
                device=torch.device(args.device),
                dtype=dtype,
                hook_name=args.hook_name,
            )
            identity_entry = (identity_name, identity_sae)

    sae_meta: dict[str, dict[str, str]] = {}
    for basis in EPDictionarySAE.BASES:
        sae_name = f"cas_{args.model_short}_layer_{args.layer}_{basis}"
        sae_meta[sae_name] = {"kind": "ep", "basis": basis}
    if identity_entry is not None:
        sae_meta[identity_entry[0]] = {"kind": "identity"}

    def _build_cas_saes(readout: str, k: int) -> list:
        saes = []
        for basis in EPDictionarySAE.BASES:
            sae_name = f"cas_{args.model_short}_layer_{args.layer}_{basis}"
            sae = EPDictionarySAE(
                dictionary,
                model_name=args.model_short,
                hook_layer=args.layer,
                hook_name=args.hook_name,
                device=torch.device(args.device),
                dtype=dtype,
                basis=basis,
                readout=readout,
                k=k,
            )
            saes.append((sae_name, sae))
        return saes

    if args.aggregate_only:
        logger.info("--aggregate-only: skipping eval execution.")
        return

    # Eval-time params that affect numerical results. If these change between
    # runs (e.g. SAEBench upstream bumps a model's batch_size or dtype), the
    # cached JSONs are stale even though the dictionary identity hasn't changed.
    eval_meta = {"batch_size": cfg["batch_size"], "dtype": cfg["dtype"]}

    def _meta_path(eval_type: str) -> Path:
        return args.output_dir / saebench_runner.output_folders[eval_type] / "_eval_meta.json"

    def _meta_matches(eval_type: str) -> bool:
        path = _meta_path(eval_type)
        if not path.exists():
            return False
        try:
            with path.open() as f:
                return json.load(f) == eval_meta
        except (OSError, json.JSONDecodeError):
            return False

    def _write_meta(eval_type: str) -> None:
        path = _meta_path(eval_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(eval_meta, f)

    evals_to_run = list(eval_types)
    if not args.force_rerun:
        remaining = []
        for eval_type in evals_to_run:
            folder = saebench_runner.output_folders[eval_type]
            paths = [
                args.output_dir / folder / f"{name}_custom_sae_eval_results.json"
                for name in sae_meta
            ]
            if all(p.exists() for p in paths) and _meta_matches(eval_type):
                logger.info("Skipping %s: results already exist", eval_type)
            else:
                if all(p.exists() for p in paths):
                    logger.info(
                        "Re-running %s: cached results' eval cfg doesn't match current "
                        "(%s)", eval_type, eval_meta,
                    )
                remaining.append(eval_type)
        evals_to_run = remaining

    if not evals_to_run:
        logger.info("All SAEBench eval results already exist; nothing to run.")
        return

    if args.force_rerun:
        for eval_type in evals_to_run:
            folder = saebench_runner.output_folders[eval_type]
            for name in sae_meta:
                p = args.output_dir / folder / f"{name}_custom_sae_eval_results.json"
                if p.exists():
                    logger.info("Removing %s for force rerun", p)
                    p.unlink()

    logger.info("Running SAEBench evals: %s", ",".join(evals_to_run))
    prev_dir = Path.cwd()
    os.chdir(args.output_dir)
    failed = []
    try:
        for eval_type in evals_to_run:
            readout, k = _readout_for(eval_type)
            selected_saes = _build_cas_saes(readout, k)
            if identity_entry is not None:
                selected_saes.append(identity_entry)
            logger.info("  %s: readout=%s k=%d", eval_type, readout, k)
            try:
                if eval_type in ("scr", "tpp"):
                    _run_scr_or_tpp(
                        eval_type, args, cfg, selected_saes,
                    )
                else:
                    saebench_runner.run_evals(
                        model_name=args.model_short,
                        selected_saes=selected_saes,
                        llm_batch_size=cfg["batch_size"],
                        llm_dtype=cfg["dtype"],
                        device=args.device,
                        eval_types=[eval_type],
                        api_key=api_key,
                        force_rerun=args.force_rerun,
                        save_activations=args.save_activations,
                    )
                _write_meta(eval_type)
            except Exception as e:
                logger.error("Eval %s failed: %s", eval_type, e)
                failed.append(eval_type)
    finally:
        os.chdir(prev_dir)
    if failed:
        logger.warning("Failed evals: %s", ", ".join(failed))


# ----------------------------------------------------- compare-sae trigger

# scripts/compare_sae.py hard-codes SAE availability per model in SAE_CONFIGS
# and _gemma_canonical_widths. Mirror that here so we can guard at the build
# layer too.
_COMPARE_SAE_SUPPORTED = {
    "pythia-70m-deduped",
    "pythia-160m-deduped",
    "gemma-2-2b",
}


def _run_compare_sae(args) -> None:
    """Subprocess scripts.compare_sae against the just-built dictionary.

    Writes `{output_dir}/sae_comparison.json` for the headline table to pick
    up. Skipped if the JSON already exists (cheap rerun) unless --force-rerun.
    Uses the existing dictionary pickle, so no rebuild — only model load +
    activation collection + SAE/EP encode.
    """
    import subprocess

    if args.model_short not in _COMPARE_SAE_SUPPORTED:
        logger.warning(
            "compare_sae skipped: model_short=%s not supported "
            "(no SAEs configured in compare_sae.SAE_CONFIGS; supported: %s).",
            args.model_short, sorted(_COMPARE_SAE_SUPPORTED),
        )
        return

    out_path = args.output_dir / "sae_comparison.json"
    if out_path.exists() and not args.force_rerun:
        # Result cached at a smaller token budget would silently mislead the
        # headline table — check the recorded n_tokens before reusing.
        try:
            with out_path.open() as f:
                cached_cfg = json.load(f).get("config", {})
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("compare_sae: cannot read %s (%s); will rerun", out_path, e)
        else:
            requested = args.compare_sae_n_tokens
            cached_n = cached_cfg.get("n_tokens")
            if requested is None or (cached_n is not None and cached_n >= requested):
                logger.info(
                    "compare_sae results exist at %s (n_tokens=%s); skipping",
                    out_path, cached_n,
                )
                return
            logger.info(
                "compare_sae result at %s used n_tokens=%s but %s requested; rerunning",
                out_path, cached_n, requested,
            )

    pkl_path = dictionary_path(args.output_dir, args.model_short, args.layer)
    if not pkl_path.exists():
        logger.warning("compare_sae: no dictionary pickle at %s; skipping", pkl_path)
        return

    cmd = [
        sys.executable, "-m", "scripts.compare_sae",
        "--model", args.model,
        "--model-short", args.model_short,
        "--layer", str(args.layer),
        "--ep-dictionary", str(pkl_path),
        "--ep-percentile", str(args.percentile),
        "--output", str(out_path),
        "--device", args.device,
        "--seed", str(args.seed),
        "--batch-size", str(args.model_batch_size),
        "--context-length", str(args.context_length),
    ]
    if args.compare_sae_n_tokens is not None:
        cmd.extend(["--n-tokens", str(args.compare_sae_n_tokens)])
    if args.hook_name:
        cmd.extend(["--hook-name", args.hook_name])

    # Width-match the SAE to the partition count so the compare_sae row and the
    # "Best SAE @ <width>" SOTA row agree. Pass closest_width's pick (4k/16k/65k
    # — the SAEBench-published widths) rather than n_partitions itself, so
    # compare_sae.pick_release_for_width lands on the same width as the SOTA row.
    meta = _read_build_metadata(args.output_dir, args.model_short, args.layer)
    n_partitions = meta.get("discovery", {}).get("n_partitions")
    if n_partitions:
        from ep.saebench_sota import closest_width
        cmd.extend(["--n-concepts", str(closest_width(n_partitions))])

    # Reuse cached activations from the build phase if they exist — skips
    # the model load + repeated forward pass that compare_sae would otherwise do.
    cache_dir = activations_cache_dir_for(args)
    if cache_dir.exists() and any(cache_dir.glob("shard_*.npz")):
        cmd.extend(["--activations-cache", str(cache_dir)])

    logger.info("Running compare_sae → %s", out_path)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        logger.warning("compare_sae exited %d; headline columns will be blank",
                       result.returncode)


# ----------------------------------------------------------- AxBench eval

# Each (model_short, layer) here has a pre-shipped concept dataset bundled
# with AxBench at axbench/concept500/prod_<model>_l<layer>_v1/. Other configs
# would require running their generate.py (OpenAI-credit cost), so we skip.
AXBENCH_SUPPORTED: dict[str, dict[int, str]] = {
    "gemma-2-2b-it": {
        10: "prod_2b_l10_v1",
        20: "prod_2b_l20_v1",
    },
    "gemma-2-9b-it": {
        20: "prod_9b_l20_v1",
        31: "prod_9b_l31_v1",
    },
}

AXBENCH_BASES = ("mean", "exemplar")  # mirrors EPDictionarySAE.BASES


def _ensure_axbench_ep_module(axbench_root: Path) -> None:
    """Sync our vendored EP module into the AxBench checkout.

    The canonical source lives at ``ep/_axbench_ep.py`` (tracked). The
    deployed copy at ``baselines/axbench/axbench/models/ep.py`` is gitignored
    and untracked in axbench's own tree, so we copy on every run to keep
    them in lockstep. Idempotent: skipped when contents already match.
    """
    import shutil
    import ep

    src = Path(ep.__file__).parent / "_axbench_ep.py"
    dst = axbench_root / "axbench" / "models" / "ep.py"
    if not src.exists():
        raise RuntimeError(f"Vendored AxBench EP source missing at {src}")
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    logger.info("Installed AxBench EP module: %s -> %s", src, dst)


def _ensure_axbench_seed_data(axbench_root: Path) -> Path:
    """Ensure seed_sentences/seed_instructions/alpaca_eval.json exist; return master_data_dir.

    AxBench's DatasetFactory unconditionally loads seed_sentences and
    seed_instructions from ``master_data_dir`` (axbench/utils/dataset.py:202-203),
    and steering inference additionally reads ``alpaca_eval.json`` from the
    same dir (axbench/utils/dataset.py:677-678). The repo only ships download
    scripts (download-seed-sentences.py, download-alpaca.sh), and the Modal
    mount of the axbench source is read-only, so we cache the HuggingFace
    pulls on the writable /vol volume.
    """
    import subprocess
    import urllib.request

    cache_dir = Path("/vol/axbench-data") if Path("/vol").exists() \
        else axbench_root / "axbench" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not ((cache_dir / "seed_sentences").exists() and (cache_dir / "seed_instructions").exists()):
        download_script = axbench_root / "axbench" / "data" / "download-seed-sentences.py"
        logger.info("Downloading AxBench seed datasets to %s", cache_dir)
        result = subprocess.run(
            [sys.executable, str(download_script)],
            cwd=str(cache_dir),
        )
        if result.returncode != 0:
            raise RuntimeError(f"AxBench seed download failed (exit {result.returncode})")

    alpaca_path = cache_dir / "alpaca_eval.json"
    if not alpaca_path.exists():
        url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
        logger.info("Downloading AxBench alpaca_eval.json to %s", alpaca_path)
        urllib.request.urlretrieve(url, alpaca_path)

    return cache_dir


def _run_axbench(args, dictionary) -> None:
    """Run AxBench's train→inference→evaluate pipeline against the EP library,
    once per basis (mean, exemplar).

    The leaderboard's headline metric is Overall Score = max_lm_judge_rating,
    computed in axbench/scripts/analyse.ipynb (cell 4 / format_df). Reproduced
    by _read_axbench_metrics: argmax-factor on steering.jsonl, lookup
    that index in steering_test.jsonl. So we run evaluate twice for steering
    (modes "steering" and "steering_test") plus once for "latent".

    Skips cleanly if (model_short, layer) is not an AxBench-supported config.
    """
    import os
    import shutil
    import subprocess
    import yaml

    layers_for_model = AXBENCH_SUPPORTED.get(args.model_short)
    if layers_for_model is None:
        logger.warning(
            "AxBench eval skipped: model_short=%s not supported "
            "(supported: %s).",
            args.model_short, sorted(AXBENCH_SUPPORTED),
        )
        return
    if args.layer not in layers_for_model:
        logger.warning(
            "AxBench eval skipped: %s@L%d not supported "
            "(supported layers: %s).",
            args.model_short, args.layer, sorted(layers_for_model),
        )
        return

    pkl_path = dictionary_path(args.output_dir, args.model_short, args.layer)
    if not pkl_path.exists():
        logger.warning("AxBench: no dictionary pickle at %s; skipping", pkl_path)
        return

    axbench_root = REPO_ROOT / "baselines" / "axbench"
    if not axbench_root.exists():
        logger.warning("AxBench repo not at %s; skipping", axbench_root)
        return

    prod_subdir = layers_for_model[args.layer]
    prod_data_dir = axbench_root / "axbench" / "concept500" / prod_subdir / "generate"
    if not prod_data_dir.exists():
        logger.warning(
            "AxBench: pre-shipped concept data missing at %s; skipping",
            prod_data_dir,
        )
        return

    dump_dir = args.output_dir / "axbench"
    dump_dir.mkdir(parents=True, exist_ok=True)

    _ensure_axbench_ep_module(axbench_root)
    master_data_dir = _ensure_axbench_seed_data(axbench_root)

    # Write a sweep YAML pointing at our EP library and the chosen layer.
    # Both bases are evaluated together; AxBench namespaces outputs by model_name.
    method_block = {
        "batch_size": 32,
        "n_epochs": 1,
        "low_rank_dimension": 1,
        "intervention_positions": "all",
        "intervention_type": "addition",
        # Required: prepare_df only emits the `labels` column when binarize=True,
        # and EP delegates to AxBench's probe make_data_module / consumes 0-1
        # labels in train() itself. Matches LinearProbe's sweep configs.
        "binarize_dataset": True,
        "train_on_negative": True,
        "exclude_bos": True,
        "ep_library_path": str(pkl_path),
    }
    # Train only the EP variants. PromptSteering needs no training but is
    # included at inference/evaluate time as the win-rate baseline (matches
    # the AxBench leaderboard convention).
    train_methods = ["EPExemplar", "EPMean"]
    inference_methods = train_methods + ["PromptSteering"]
    sweep = {
        "train": {
            "model_name": args.model,
            "layer": args.layer,
            "component": "res",
            "seed": args.seed,
            "use_bf16": True,
            "output_length": 128,
            "models": {name: dict(method_block) for name in train_methods},
        },
        "inference": {
            "use_bf16": True,
            "models": inference_methods,
            "model_name": args.model,
            "output_length": 128,
            "latent_num_of_examples": 36,
            "latent_batch_size": 16,
            "steering_intervention_type": "addition",
            "steering_model_name": args.model,
            "steering_datasets": ["AlpacaEval"],
            "steering_batch_size": 32,
            "steering_output_length": 128,
            "steering_layers": [args.layer],
            "steering_num_of_examples": getattr(args, "axbench_steering_examples", 10),
            "steering_factors": [
                0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
                2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0,
            ],
            "master_data_dir": str(master_data_dir),
            "seed": args.seed,
            "lm_model": "gpt-4o-mini",
            "temperature": 1.0,
        },
        "evaluate": {
            "models": inference_methods,
            "latent_evaluators": ["AUCROCEvaluator", "HardNegativeEvaluator"],
            "steering_evaluators": ["PerplexityEvaluator", "LMJudgeEvaluator"],
            # winrate_split_ratio splits the steering data per-concept into a
            # factor-selection partition (mode="steering") and an evaluation
            # partition (mode="steering_test"). 0.5 mirrors the production
            # sweep configs in axbench/sweep/aryaman/.
            "winrate_split_ratio": 0.5,
            "num_of_workers": 32,
            "lm_model": "gpt-4o-mini",
            "master_data_dir": str(master_data_dir),
        },
    }
    max_concepts = getattr(args, "axbench_max_concepts", None)
    if max_concepts is not None:
        sweep["train"]["max_concepts"] = max_concepts
        sweep["inference"]["max_concepts"] = max_concepts
        sweep["evaluate"]["max_concepts"] = max_concepts

    sweep_path = dump_dir / "sweep.yaml"
    with sweep_path.open("w") as f:
        yaml.safe_dump(sweep, f, sort_keys=False)
    logger.info("Wrote AxBench sweep config to %s", sweep_path)

    # Skip if the test-half jsonl already exists (cheap rerun).
    eval_steering_test = dump_dir / "evaluate" / "steering_test.jsonl"
    if eval_steering_test.exists() and not args.force_rerun:
        logger.info("AxBench results exist at %s; skipping", eval_steering_test)
        return
    if args.force_rerun and dump_dir.exists():
        for sub in ("train", "inference", "evaluate"):
            d = dump_dir / sub
            if d.exists():
                logger.info("Removing %s for force rerun", d)
                shutil.rmtree(d)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{axbench_root}:{axbench_root}/axbench/scripts:{env.get('PYTHONPATH', '')}"
    # AxBench's ModelParams whitelist drops unknown YAML keys, so the
    # ep_library_path field in the sweep never reaches the EP model. Pass it
    # via env var instead.
    env["EP_LIBRARY_PATH"] = str(pkl_path)
    # evaluate.py:384 hardcodes master_data_dir="axbench/data" for the LMJudge
    # cache, which resolves under the read-only Modal mount. Override via env
    # so the LM cache lands on the writable /vol volume alongside the seed data.
    env["AXBENCH_MASTER_DATA_DIR"] = str(master_data_dir)

    def _torchrun(module: str, *extra: str) -> int:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node=1",
            "-m", module,
            "--config", str(sweep_path),
            "--dump_dir", str(dump_dir),
            "--overwrite_data_dir", str(prod_data_dir),
            "--overwrite_metadata_dir", str(prod_data_dir),
            *extra,
        ]
        logger.info("AxBench: %s %s", module, " ".join(extra))
        return subprocess.run(cmd, cwd=str(axbench_root), env=env).returncode

    modes = [m.strip() for m in args.axbench_modes.split(",") if m.strip()]
    if _torchrun("ep._axbench_train") != 0:
        logger.warning("AxBench train failed; aborting eval")
        return
    if "latent" in modes:
        if _torchrun("ep._axbench_inference", "--mode", "latent") != 0:
            logger.warning("AxBench latent inference failed; aborting eval")
            return
    if "steering" in modes or "steering_test" in modes:
        if _torchrun("ep._axbench_inference", "--mode", "steering") != 0:
            logger.warning("AxBench steering inference failed; aborting eval")
            return
    # evaluate is single-process (no torchrun). Three passes:
    # - latent → latent.jsonl, latent_data.parquet
    # - steering → steering.jsonl (factor-selection partition)
    # - steering_test → steering_test.jsonl (held-out partition, used for
    #   the leaderboard's max_lm_judge_rating headline metric)
    for mode in modes:
        cmd = [
            sys.executable, "-m", "ep._axbench_evaluate",
            "--config", str(sweep_path),
            "--dump_dir", str(dump_dir),
            "--mode", mode,
        ]
        logger.info("AxBench: evaluate --mode %s", mode)
        if subprocess.run(cmd, cwd=str(axbench_root), env=env).returncode != 0:
            logger.warning("AxBench evaluate --mode %s failed", mode)


def _read_axbench_metrics(output_dir: Path) -> dict[str, dict[str, float]]:
    """Pull per-method headline metrics from AxBench's evaluate output.

    Reproduces axbench/scripts/analyse.ipynb cell 4 (format_df):
    - Overall Score (max_lm_judge_rating): for each (concept, method) pick the
      argmax-factor on steering.jsonl (factor-selection partition), look up
      lm_judge_rating at that index in steering_test.jsonl (held-out
      partition). Mean across concepts.
    - Latent AUROC, Hard-negative accuracy: pulled from latent.jsonl directly.

    Returns ``{method_name: {metric: value}}``. Missing files / methods just
    don't populate the dict — caller treats absence as NaN.
    """
    eval_dir = output_dir / "axbench" / "evaluate"
    if not eval_dir.exists():
        return {}

    def load_lmjudge(path: Path) -> dict[int, dict[str, dict]]:
        """concept_id → method → LMJudgeEvaluator result dict."""
        out: dict[int, dict[str, dict]] = {}
        if not path.exists():
            return out
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                out[int(d["concept_id"])] = d.get("results", {}).get("LMJudgeEvaluator", {})
        return out

    train = load_lmjudge(eval_dir / "steering.jsonl")
    test = load_lmjudge(eval_dir / "steering_test.jsonl")

    # Per-concept overall score: argmax-factor on the factor-selection
    # partition, look up the lm_judge_rating at that index in the held-out
    # partition. Skip the concept entirely if either side is missing —
    # falling back to the train side would leak factor selection into the
    # reported score.
    overall_scores: dict[str, list[float]] = {}
    for cid, train_methods in train.items():
        test_methods = test.get(cid, {})
        for method, train_result in train_methods.items():
            train_ratings = train_result.get("lm_judge_rating")
            test_ratings = test_methods.get(method, {}).get("lm_judge_rating")
            if not train_ratings or not test_ratings:
                continue
            if len(train_ratings) != len(test_ratings):
                continue
            idx = int(np.argmax(train_ratings))
            overall_scores.setdefault(method, []).append(float(test_ratings[idx]))

    out: dict[str, dict[str, float]] = {}
    for method, scores in overall_scores.items():
        out[method] = {"overall_score": float(np.mean(scores))}

    # Latent metrics from latent.jsonl (AUROC + hard-negative accuracy).
    latent_path = eval_dir / "latent.jsonl"
    if latent_path.exists():
        per_method: dict[str, dict[str, list[float]]] = {}
        with latent_path.open() as f:
            for line in f:
                results = json.loads(line).get("results", {})
                for method, m in results.get("AUCROCEvaluator", {}).items():
                    per_method.setdefault(method, {}).setdefault("auroc", []).append(float(m["roc_auc"]))
                for method, m in results.get("HardNegativeEvaluator", {}).items():
                    per_method.setdefault(method, {}).setdefault("hardneg_acc", []).append(float(m["macro_avg_accuracy"]))
        for method, metrics in per_method.items():
            for k, vs in metrics.items():
                out.setdefault(method, {})[k] = float(np.mean(vs))
    return out


# ---------------------------------------------------- wandb headline table

def _log_per_eval_scalars(args, eval_types: list[str]) -> None:
    """Log scalar headline metrics for the SAEBench / AxBench evals this worker
    just ran, so each fan-out worker contributes its own slice of the wandb
    timeline. Keys are namespaced (`saebench/<eval>/<col>/<sae>`,
    `axbench/<method>/<metric>`) so concurrent workers don't collide.

    No tables here — the unified table is the aggregator's job, since wandb
    Tables overwrite on every log call and can't be merged across processes.
    """
    if not args.wandb:
        return
    import wandb
    from ep.saebench_adapter import EPDictionarySAE
    from ep.saebench_sota import HEADLINE_METRICS

    if wandb.run is None:
        _wandb_init(args, default_name=f"{args.model_short}_L{args.layer}_evals")

    sae_names = [
        f"cas_{args.model_short}_layer_{args.layer}_{basis}"
        for basis in EPDictionarySAE.BASES
    ]
    if not args.no_identity_baseline:
        sae_names.append(f"identity_{args.model_short}_layer_{args.layer}")

    payload: dict[str, float] = {}
    for et in eval_types:
        if et == "axbench":
            metrics = _read_axbench_metrics(args.output_dir)
            for method, m in metrics.items():
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        payload[f"axbench/{method}/{k}"] = float(v)
            continue
        for col, (eval_type, category, key, _) in HEADLINE_METRICS.items():
            if eval_type != et:
                continue
            for sae in sae_names:
                p = (
                    args.output_dir / "eval_results" / eval_type
                    / f"{sae}_custom_sae_eval_results.json"
                )
                if not p.exists():
                    continue
                with p.open() as f:
                    cat = (json.load(f).get("eval_result_metrics") or {}).get(category) or {}
                val = cat.get(key)
                if isinstance(val, (int, float)):
                    payload[f"saebench/{et}/{col}/{sae}"] = float(val)

    if payload:
        wandb.log(payload)
        logger.info("Logged %d scalar metrics to wandb for evals %s",
                    len(payload), eval_types)


def _read_eval_metrics(output_dir: Path, sae_name: str) -> dict[str, float]:
    """Pull every column in HEADLINE_METRICS for one sae_name.

    Reads `eval_results/{eval_type}/{sae_name}_custom_sae_eval_results.json`
    once per distinct eval_type and extracts the configured (category, key).
    Missing files / missing metrics are silently skipped.
    """
    from ep.saebench_sota import HEADLINE_METRICS

    cache: dict[str, dict] = {}
    metrics: dict[str, float] = {}
    for col, (eval_type, category, key, _) in HEADLINE_METRICS.items():
        if eval_type not in cache:
            path = (
                output_dir / "eval_results" / eval_type
                / f"{sae_name}_custom_sae_eval_results.json"
            )
            if not path.exists():
                cache[eval_type] = {}
                continue
            with path.open() as f:
                cache[eval_type] = json.load(f).get("eval_result_metrics", {})
        cat = cache[eval_type].get(category) or {}
        val = cat.get(key)
        if isinstance(val, (int, float)):
            metrics[col] = float(val)
    return metrics


def _read_compare_sae(output_dir: Path) -> tuple[dict[str, dict[str, float]], int | None]:
    """Pull aggregate F1s per basis and the matched SAE width from sae_comparison.json.

    Returns ({basis: {sae_to_ep_mean_f1: ..., ...}}, sae_d_sae). Empty dict +
    None if the JSON is missing. Handles both new schema
    (results_by_percentile[p][basis] = r) and old schema
    (results_by_percentile[p] = r, mean basis only).
    """
    path = output_dir / "sae_comparison.json"
    if not path.exists():
        return {}, None
    with path.open() as f:
        data = json.load(f)
    sae_d_sae = data.get("config", {}).get("sae_d_sae")
    by_p = data.get("results_by_percentile") or {}
    if not by_p:
        return {}, sae_d_sae
    # Take the lowest percentile run (= tightest threshold) if multiple.
    p_key = sorted(by_p.keys(), key=float)[0]
    p_entry = by_p[p_key]

    # Detect schema: new schema has basis keys, old schema has sae_to_ep directly.
    if isinstance(p_entry, dict) and "sae_to_ep" not in p_entry:
        per_basis = p_entry  # {basis: results}
    else:
        per_basis = {"mean": p_entry}  # legacy

    def _agg(r: dict) -> dict[str, float]:
        sae_to_ep = [m["f1"] for m in r.get("sae_to_ep", []) if "f1" in m]
        ep_to_sae = [m["f1"] for m in r.get("ep_to_sae", []) if "f1" in m]
        out: dict[str, float] = {}
        if sae_to_ep:
            out["sae_to_ep_mean_f1"] = float(np.mean(sae_to_ep))
            out["sae_to_ep_frac>0.5"] = float(sum(1 for f in sae_to_ep if f > 0.5) / len(sae_to_ep))
        if ep_to_sae:
            out["ep_to_sae_mean_f1"] = float(np.mean(ep_to_sae))
            out["ep_to_sae_frac>0.5"] = float(sum(1 for f in ep_to_sae if f > 0.5) / len(ep_to_sae))
        return out

    return {basis: _agg(r) for basis, r in per_basis.items()}, sae_d_sae


def _read_build_metadata(output_dir: Path, model_short: str, layer: int) -> dict:
    """Pull n_partitions and elapsed_s from the dictionary metadata."""
    path = metadata_path(output_dir, model_short, layer)
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _log_headline_table(
    args, dictionary, *,
    saebench_evals: bool, run_compare_sae: bool, run_axbench: bool,
) -> None:
    """One paper-ready table covering SAEBench, compare_sae, and AxBench results.

    Rows (union — present only if their eval ran):
      - Partitions (mean) / (exemplar) — fully populated when all evals ran
      - Identity SAE, Best SAE @ <width>          (saebench-only)
      - Prompt (ours), each published AxBench leaderboard method (axbench-only)
    Cols:
      method | n | build_h | <SAEBench HEADLINE_METRICS> | compare_sae F1s
             | axbench_overall | axbench_aucroc | axbench_hardneg

    Cells are None where the row's eval didn't produce that column (e.g. SAEBench
    rows have empty AxBench cells). SOTA detail companions are logged separately
    by ``_log_saebench_sota_detail``.
    """
    if not args.wandb or not (saebench_evals or run_compare_sae or run_axbench):
        return
    import wandb

    if wandb.run is None:
        _wandb_init(args, default_name=f"{args.model_short}_L{args.layer}_headline")

    from ep.saebench_sota import HEADLINE_METRICS, closest_width, headline_baselines
    from ep.saebench_adapter import EPDictionarySAE

    metric_cols = list(HEADLINE_METRICS.keys()) if saebench_evals else []
    compare_cols = (
        ["sae_width",
         "sae_to_ep_mean_f1", "sae_to_ep_frac>0.5",
         "ep_to_sae_mean_f1", "ep_to_sae_frac>0.5"]
        if run_compare_sae else []
    )
    axbench_cols = (
        ["axbench_overall", "axbench_aucroc", "axbench_hardneg"]
        if run_axbench else []
    )
    cols = ["method", "n", "build_h"] + metric_cols + compare_cols + axbench_cols

    n_partitions = len(dictionary) if dictionary is not None else None
    meta = _read_build_metadata(args.output_dir, args.model_short, args.layer)
    elapsed_s = meta.get("discovery", {}).get("elapsed_s")
    build_h = round(elapsed_s / 3600.0, 4) if elapsed_s else None
    if run_compare_sae:
        compare, compare_sae_width = _read_compare_sae(args.output_dir)
    else:
        compare, compare_sae_width = {}, None
    axbench_metrics = _read_axbench_metrics(args.output_dir) if run_axbench else {}

    # AxBench's per-method JSON keys for the EP entries (wired in _run_axbench).
    AXBENCH_EP_NAMES = {"mean": "EPMean", "exemplar": "EPExemplar"}

    table_data: list[list] = []

    # --- Partition rows (one per basis) ---
    for basis in EPDictionarySAE.BASES:
        row = [f"Partitions ({basis})", n_partitions, build_h]
        if saebench_evals:
            sae_name = f"cas_{args.model_short}_layer_{args.layer}_{basis}"
            m = _read_eval_metrics(args.output_dir, sae_name)
            row.extend(m.get(c) for c in metric_cols)
        if run_compare_sae:
            basis_compare = compare.get(basis, {})
            # sae_width is global (same SAE across both basis rows); F1 cols are per-basis.
            row.extend(
                compare_sae_width if c == "sae_width" else basis_compare.get(c)
                for c in compare_cols
            )
        if run_axbench:
            axb = axbench_metrics.get(AXBENCH_EP_NAMES[basis], {})
            row.extend([axb.get("overall_score"), axb.get("auroc"), axb.get("hardneg_acc")])
        table_data.append(row)

    # --- Identity SAE row (SAEBench-only) ---
    if saebench_evals and not args.no_identity_baseline:
        identity_name = f"identity_{args.model_short}_layer_{args.layer}"
        m = _read_eval_metrics(args.output_dir, identity_name)
        # For identity, n = d_model; build cost is zero (no fitting).
        d_in = dictionary.center.shape[0] if dictionary is not None else None
        row = ["Identity SAE", d_in, 0.0]
        row.extend(m.get(c) for c in metric_cols)
        row.extend([None] * (len(compare_cols) + len(axbench_cols)))
        table_data.append(row)

    # --- Best published SAE @ closest width (SAEBench-only) ---
    sota_width = (
        closest_width(n_partitions) if (saebench_evals and n_partitions) else None
    )
    sota_best = headline_baselines(args.model_short, sota_width) if sota_width else {}
    if sota_width and sota_best:
        row = [f"Best SAE @ {sota_width}", sota_width, None]
        row.extend(sota_best[c].value if c in sota_best else None for c in metric_cols)
        row.extend([None] * (len(compare_cols) + len(axbench_cols)))
        table_data.append(row)
    elif sota_width:
        logger.warning(
            "No published SAEBench baselines for %s @ width %d "
            "(model not in PUBLISHED_LAYER or HF download failed); "
            "headline table will lack a SOTA row.",
            args.model_short, sota_width,
        )

    # --- AxBench Prompt baseline we ran ourselves ---
    if run_axbench and "PromptSteering" in axbench_metrics:
        m = axbench_metrics["PromptSteering"]
        row = ["Prompt (ours)", None, None]
        row.extend([None] * (len(metric_cols) + len(compare_cols)))
        row.extend([m.get("overall_score"), m.get("auroc"), m.get("hardneg_acc")])
        table_data.append(row)

    # --- Published AxBench leaderboard rows ---
    if run_axbench:
        published = AXBENCH_LEADERBOARD.get((args.model_short, args.layer), {})
        for method, score in sorted(published.items(), key=lambda kv: -kv[1]):
            row = [method, None, None]
            row.extend([None] * (len(metric_cols) + len(compare_cols)))
            row.extend([score, None, None])
            table_data.append(row)

    table = wandb.Table(columns=cols, data=table_data)
    wandb.log({"headline": table})
    logger.info("Logged headline (%d rows × %d cols)", len(table_data), len(cols))

    if saebench_evals and sota_width and sota_best:
        _log_saebench_sota_detail(args, sota_width, sota_best, metric_cols)


def _log_saebench_sota_detail(args, sota_width, sota_best, metric_cols) -> None:
    """SOTA companion tables — per-metric winners, full distribution, and
    ours-vs-SOTA min/median/max + percentile rank. Logged alongside `headline`
    so the unified table stays compact while the supporting context is one
    click away.
    """
    import statistics
    import wandb
    from ep.saebench_sota import HEADLINE_METRICS, all_runs_metrics, percentile_rank
    from ep.saebench_adapter import EPDictionarySAE

    winners_cols = ["metric", "value", "architecture", "trainer", "width"]
    winners_data = [
        [c, sota_best[c].value, sota_best[c].architecture, sota_best[c].trainer, sota_best[c].width]
        for c in metric_cols
        if c in sota_best
    ]
    wandb.log({"saebench/sota_winners": wandb.Table(columns=winners_cols, data=winners_data)})

    all_runs = all_runs_metrics(args.model_short, sota_width)
    if not all_runs:
        return

    full_cols = ["architecture", "trainer", "width"] + metric_cols
    full_data = [
        [r["architecture"], r["trainer"], r["width"]] + [r.get(c) for c in metric_cols]
        for r in all_runs
    ]
    wandb.log({"saebench/sota_full": wandb.Table(columns=full_cols, data=full_data)})

    # One row per (basis, metric) so wandb can sort/filter by basis.
    summary_cols = [
        "basis", "metric", "ours", "sota_min", "sota_median", "sota_max",
        "ours_percentile", "lower_is_better",
    ]
    summary_data = []
    for basis in EPDictionarySAE.BASES:
        ep_metrics = _read_eval_metrics(
            args.output_dir,
            f"cas_{args.model_short}_layer_{args.layer}_{basis}",
        )
        for c in metric_cols:
            _, _, _, lower = HEADLINE_METRICS[c]
            vals = [r[c] for r in all_runs if c in r]
            if not vals:
                continue
            ours = ep_metrics.get(c)
            pct = (
                percentile_rank(ours, vals, lower_is_better=lower)
                if isinstance(ours, (int, float))
                else None
            )
            summary_data.append([
                basis, c, ours, min(vals), statistics.median(vals), max(vals), pct, lower,
            ])
    wandb.log({"saebench/sota_summary": wandb.Table(columns=summary_cols, data=summary_data)})
    logger.info(
        "Logged saebench/sota_winners (%d rows), saebench/sota_full (%d SAEs), "
        "saebench/sota_summary (%d rows across %d bases)",
        len(winners_data), len(all_runs), len(summary_data), len(EPDictionarySAE.BASES),
    )


# Published AxBench leaderboard numbers (README, May 2025). These are
# Overall Score = max_lm_judge_rating averaged across concepts (NOT win-rate
# — that's a separate column the analyse.ipynb notebook computes for some
# sub-tables but isn't the headline metric). Cells with no published
# submission are absent.
AXBENCH_LEADERBOARD: dict[tuple[str, int], dict[str, float]] = {
    ("gemma-2-2b-it", 10): {
        "Prompt": 0.698, "RePS": 0.756, "ReFT-r1": 0.633,
        "DiffMean": 0.297, "SAE": 0.177, "SAE-A": 0.166,
        "LAT": 0.117, "PCA": 0.107, "Probe": 0.095,
    },
    ("gemma-2-2b-it", 20): {
        "HyperSteer": 0.742, "Prompt": 0.731, "RePS": 0.606,
        "ReFT-r1": 0.509, "DiffMean": 0.178, "SAE": 0.151,
        "SAE-A": 0.132, "LAT": 0.130, "PCA": 0.083, "Probe": 0.091,
    },
    ("gemma-2-9b-it", 20): {
        "HyperSteer": 1.091, "Prompt": 1.075, "RePS": 0.892,
        "ReFT-r1": 0.630, "SAE-filtered": 0.546, "DiffMean": 0.322,
        "SAE": 0.191, "SAE-A": 0.186, "LAT": 0.127, "PCA": 0.128,
        "Probe": 0.108,
    },
    ("gemma-2-9b-it", 31): {
        "Prompt": 1.072, "RePS": 0.624, "ReFT-r1": 0.401,
        "SAE-filtered": 0.470, "DiffMean": 0.158, "SAE": 0.140,
        "SAE-A": 0.143, "LAT": 0.134, "PCA": 0.104, "Probe": 0.099,
    },
}


# ----------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an exemplar-partition dictionary"
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Defaults to results/dictionaries/<config-slug>.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--model-short", type=str, default=DEFAULT_MODEL_SHORT)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--hook-name", type=str, default=None)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--sampling-mode", choices=("fixed", "random", "full"),
                        default="full",
                        help="full: full-length texts, one per forward pass "
                             "(default; pairs with extract_per_position). "
                             "random: full-length random spans. fixed: prefixes "
                             "t[0:l] of batch_size texts (pairs with "
                             "extract_final_position).")
    parser.add_argument("--log-cadence", type=int, default=1)
    # Controls how often the in-run wandb snapshot table fires; no-op without --wandb.
    parser.add_argument("--checkpoint-cadence", type=int, default=10)
    parser.add_argument("--skip-logs", action="store_true",
                        help="Skip intermediate wandb snapshots (size hist, top "
                             "table, distance dist, voronoi). The final snapshot "
                             "still runs at the end. No-op without --wandb.")
    parser.add_argument("--model-batch-size", "--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--percentile", type=float, default=10,
                        help="Cosine-distance percentile for fixed threshold calibration.")
    parser.add_argument("--calibration-tokens", type=int, default=200_000,
                        help="Activation budget for calibration (center, threshold).")
    parser.add_argument("--force-recalibrate", action="store_true",
                        help="Recompute calibration even if cached.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-discovery", action="store_true",
                        help="Rebuild even if a compatible cached dictionary exists.")
    parser.add_argument("--extractor", choices=("per-position", "final-position"),
                        default="per-position")
    parser.add_argument("--merge-close", action="store_true",
                        help="After each batch, merge partition pairs whose exemplar "
                             "directions lie within θ (demotion strategy).")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb-project", type=str, default="ep")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Resume an existing wandb run by id. Used by the "
                             "Modal aggregator step so build + eval results all "
                             "land in the same run.")
    parser.add_argument("--no-attribution", action="store_true",
                        help="Skip the per-partition gradient-attribution table at "
                             "finalise (saves ~N_partitions × 6 forward+backward passes).")
    parser.add_argument("--attribution-prompts", type=int, default=5,
                        help="Per partition, render this many nearest + this many "
                             "boundary prompts with gradient highlighting.")

    eval_group = parser.add_argument_group("SAEBench evaluation")
    eval_group.add_argument("--eval", type=str, default=None,
                            help=f"Run SAEBench evals after building. "
                                 f"Comma-separated or 'all'. Options: {','.join(ALL_EVALS)}")
    eval_group.add_argument("--api-key", type=str, default=None,
                            help="OpenAI API key for autointerp eval.")
    eval_group.add_argument("--api-key-file", type=Path, default=None)
    eval_group.add_argument("--force-rerun", action="store_true",
                            help="Overwrite existing SAEBench eval results.")
    eval_group.add_argument("--no-save-activations", action="store_false",
                            dest="save_activations",
                            help="Disable activation caching across SAEs within an eval.")
    eval_group.set_defaults(save_activations=True)
    eval_group.add_argument("--build-only", action="store_true",
                            help="Build dictionary and exit; skip evals.")
    eval_group.add_argument("--aggregate-only", action="store_true",
                            help="Skip build and eval; just dump existing eval JSONs.")
    eval_group.add_argument("--no-identity-baseline", action="store_true",
                            help="Skip the IdentitySAE baseline row.")
    eval_group.add_argument("--compare-sae-n-tokens", type=int, default=None,
                            help="Activation budget for compare_sae's "
                                 "SAE-vs-EP correspondence (separate from "
                                 "--max-tokens). Default: scripts.compare_sae's "
                                 "own default (500_000).")
    eval_group.add_argument("--axbench-max-concepts", type=int, default=None,
                            help="Cap concepts per AxBench run (smoke testing). "
                                 "None = full 500.")
    eval_group.add_argument("--axbench-steering-examples", type=int, default=10,
                            help="AxBench steering_num_of_examples per concept "
                                 "per factor. Default 10 matches their leaderboard.")
    eval_group.add_argument("--axbench-modes", type=str, default="latent,steering,steering_test",
                            help="Comma-separated list of AxBench modes to run. "
                                 "Default 'latent,steering,steering_test' runs all. "
                                 "Use 'latent' alone for the headline AUROC only.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.context_length < 1:
        raise ValueError("--context-length must be at least 1")
    if args.sampling_mode == "full":
        # Full-text sampling pairs with per-position extraction.
        args.extractor = "per-position"

    if args.output_dir is None:
        args.output_dir = REPO_ROOT / "results" / "dictionaries" / config_slug(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only and args.eval is None:
        args.eval = "all"

    eval_types = parse_eval_types(args.eval) if args.eval else []

    use_cache = not args.force_discovery and cached_config_is_compatible(args)
    if use_cache and not args.aggregate_only:
        try:
            dictionary = load_dictionary(args.output_dir, args.model_short, args.layer)
            logger.info("Reusing cached dictionary at %s",
                        dictionary_path(args.output_dir, args.model_short, args.layer))
        except FileNotFoundError:
            use_cache = False

    will_build = not use_cache and not args.aggregate_only
    if args.wandb and not args.aggregate_only and (will_build or args.build_only):
        _wandb_init(args, default_name=f"{args.model_short}_L{args.layer}_{config_slug(args)}")
        _save_wandb_run_id(args.output_dir)

    if will_build:
        # --- Load model ---
        import transformer_lens as tl
        logger.info("Loading model %s on %s", args.model, args.device)
        t0 = time.time()
        model = tl.HookedTransformer.from_pretrained_no_processing(
            args.model, device=args.device, dtype=torch.bfloat16)
        model.eval()
        logger.info("Model loaded in %.1fs (d_model=%d)",
                    time.time() - t0, model.cfg.d_model)

        hook_name = args.hook_name or f"blocks.{args.layer}.hook_resid_post"

        # --- Choose extractor ---
        from ep.discovery.extraction import (
            extract_final_position,
            extract_per_position,
        )
        if args.extractor == "per-position":
            extract_fn = extract_per_position
            extract_kwargs = {"batch_size": args.model_batch_size}
        else:
            extract_fn = extract_final_position
            extract_kwargs = {"batch_size": args.model_batch_size}

        # --- Stream prompts and run discovery ---
        text_stream = stream_batches(
            model.tokenizer,
            context_length=args.context_length,
            batch_size=args.model_batch_size,
            sampling_mode=args.sampling_mode,
            seed=args.seed,
        )

        result = build_dictionary(
            model=model,
            texts=text_stream,
            hook_name=hook_name,
            output_dir=args.output_dir,
            model_short=args.model_short,
            layer=args.layer,
            model_name=args.model,
            percentile=args.percentile,
            calibration_tokens=args.calibration_tokens,
            calibration_extras={
                "extractor": args.extractor,
                "sampling": args.sampling_mode,
                "ctx": args.context_length,
            },
            force_recalibrate=args.force_recalibrate,
            max_tokens=args.max_tokens,
            max_prompts=args.max_prompts,
            prompt_batch_size=args.model_batch_size,
            log_cadence=args.log_cadence,
            checkpoint_cadence=args.checkpoint_cadence,
            extract_fn=extract_fn,
            extract_kwargs=extract_kwargs,
            seed=args.seed,
            device=args.device,
            use_wandb=args.wandb,
            merge_close=args.merge_close,
            activations_cache_dir=activations_cache_dir_for(args),
            log_attribution=not args.no_attribution,
            attribution_prompts_per_partition=args.attribution_prompts,
            skip_logs=args.skip_logs,
        )
        save_dictionary(args.output_dir, args.model_short, args.layer, result, args)
        dictionary = result.dictionary
    elif args.aggregate_only:
        try:
            dictionary = load_dictionary(args.output_dir, args.model_short, args.layer)
        except FileNotFoundError:
            logger.warning(
                "--aggregate-only: no dictionary found at %s — aggregation may "
                "be partial",
                dictionary_path(args.output_dir, args.model_short, args.layer),
            )
            dictionary = None

    logger.info("Dictionary ready in %s: %d partitions",
                args.output_dir, len(dictionary) if dictionary else 0)

    if args.build_only:
        return

    saebench_evals = [e for e in eval_types if e in SAEBENCH_EVALS]
    run_compare_sae = "compare_sae" in eval_types
    run_axbench = "axbench" in eval_types

    if not args.aggregate_only:
        # --- SAEBench evaluation ---
        if saebench_evals and dictionary is not None and len(dictionary) > 0:
            _run_saebench(args, dictionary, saebench_evals)
        elif saebench_evals and (dictionary is None or len(dictionary) == 0):
            logger.warning("Dictionary empty or missing; skipping SAEBench evals.")

        # --- Compare-SAE correspondence ---
        if run_compare_sae and dictionary is not None and len(dictionary) > 0:
            _run_compare_sae(args)
        elif run_compare_sae and (dictionary is None or len(dictionary) == 0):
            logger.warning("Dictionary empty or missing; skipping compare_sae.")

        # --- AxBench steering eval ---
        if run_axbench and dictionary is not None and len(dictionary) > 0:
            _run_axbench(args, dictionary)
        elif run_axbench and (dictionary is None or len(dictionary) == 0):
            logger.warning("Dictionary empty or missing; skipping AxBench eval.")

        # --- Per-eval scalar logging (each fan-out worker writes its own slice) ---
        _log_per_eval_scalars(args, eval_types)
    else:
        # --- Unified headline table (aggregator only — tables overwrite, so
        # this has to run exactly once after every eval JSON has landed). ---
        _log_headline_table(
            args, dictionary,
            saebench_evals=bool(saebench_evals),
            run_compare_sae=run_compare_sae,
            run_axbench=run_axbench,
        )

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
