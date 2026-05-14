"""Streaming exemplar-partition discovery pipeline.

Extracts activations from a model, feeds them to a `Dictionary` via leader
clustering, and stops when the dictionary saturates (no new partitions in the
last `saturation_window` batches).

Calibration (activation center + cosine threshold) is precomputed by the
caller via :func:`calibrate_pipeline` (or loaded from cache), then passed to
:func:`discover` as a fixed ``Calibration`` object. This makes every
dictionary built on the same (model, hook, percentile) use the same
geometric reference, regardless of seed or data ordering.
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .calibration import Calibration, calibrate
from .dictionary import Dictionary
from .eval import _gini

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """Dictionary state at a point during discovery."""
    n_activations: int
    n_prompts: int
    n_partitions: int
    new_partition_rate: float
    elapsed_s: float


@dataclass
class DiscoveryResult:
    """Result from a discovery run."""
    dictionary: Dictionary
    n_activations: int
    n_prompts: int
    n_tokens: int
    n_forward_passes: int
    elapsed_s: float
    extraction_time_s: float
    clustering_time_s: float
    saturated: bool
    snapshots: list[Snapshot]


def _iter_prompt_batches(
    texts: list[str] | Iterator[str],
    prompt_batch_size: int,
    seed: int,
) -> Iterator[list[str]]:
    """Yield prompt batches from ``texts`` (lists are shuffled by ``seed``)."""
    if isinstance(texts, list):
        all_texts = list(texts)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_texts)
        text_iter = iter(all_texts)
    else:
        text_iter = iter(texts)
    while True:
        prompt_batch: list[str] = []
        try:
            while len(prompt_batch) < prompt_batch_size:
                prompt_batch.append(next(text_iter))
        except StopIteration:
            if prompt_batch:
                yield prompt_batch
            return
        yield prompt_batch


def _iter_activation_batches(
    model,
    texts: list[str] | Iterator[str],
    hook_name: str,
    *,
    extract_fn: Callable,
    extract_kwargs: dict,
    prompt_batch_size: int,
    seed: int,
):
    """Yield activation batches by extracting from texts (until exhausted)."""
    from .extraction import normalize_extractor_output

    for prompt_batch in _iter_prompt_batches(texts, prompt_batch_size, seed):
        raw = extract_fn(model, prompt_batch, hook_name, **extract_kwargs)
        result = normalize_extractor_output(raw)
        if len(result.x) > 0:
            yield result.x


def calibrate_pipeline(
    model,
    texts: list[str] | Iterator[str],
    hook_name: str,
    *,
    n_tokens: int = 200_000,
    percentile: float = 10.0,
    extract_fn: Callable | None = None,
    extract_kwargs: dict | None = None,
    prompt_batch_size: int = 16,
    seed: int = 0,
) -> Calibration:
    """Extract activations until ``n_tokens`` are seen, then compute (center, threshold).

    Identical extraction settings to :func:`discover` so the calibration
    matches what the dictionary will see.
    """
    from .extraction import extract_per_position as _default_extract

    if extract_fn is None:
        extract_fn = _default_extract
    if extract_kwargs is None:
        extract_kwargs = {}

    batches = _iter_activation_batches(
        model, texts, hook_name,
        extract_fn=extract_fn, extract_kwargs=extract_kwargs,
        prompt_batch_size=prompt_batch_size,
        seed=seed,
    )
    return calibrate(batches, n_tokens=n_tokens, percentile=percentile)


def discover(
    model,
    texts: list[str] | Iterator[str],
    hook_name: str,
    calibration: Calibration,
    *,
    extract_fn: Callable | None = None,
    extract_kwargs: dict | None = None,
    log_cadence: int = 1,
    checkpoint_cadence: int = 10,
    saturation_window: int = 1,
    max_tokens: int | None = None,
    max_prompts: int | None = None,
    prompt_batch_size: int = 16,
    checkpoint_fn: Callable | None = None,
    log_fn: Callable[[dict], None] | None = None,
    seed: int = 0,
    merge_close: bool = False,
    activations_cache_dir: Path | None = None,
) -> DiscoveryResult:
    """Run streaming exemplar-partition discovery against a fixed calibration.

    Args:
        model: TransformerLens HookedTransformer.
        texts: list of prompt strings or an iterator that yields them.
        hook_name: hook point (e.g. "blocks.8.hook_resid_post").
        calibration: precomputed (center, threshold). Use
            :func:`calibrate_pipeline` or
            :func:`calibration.load_or_calibrate` to obtain.
        extract_fn: activation extractor. Defaults to extract_per_position.
        extract_kwargs: extra kwargs for the extractor.
        log_cadence: extraction batches between log/snapshot events.
        checkpoint_cadence: extraction batches between checkpoint saves.
        saturation_window: stop after this many consecutive batches with no
            new partitions.
        max_tokens: stop after this many tokens processed (may overshoot by
            one batch).
        max_prompts: optional cap on number of prompts.
        prompt_batch_size: prompts per extractor call.
        checkpoint_fn: called every checkpoint_cadence batches with
            (dictionary, snapshots, stats_dict).
        log_fn: called every log_cadence batches with a flat metric dict.
        seed: random seed for prompt ordering.
        merge_close: if True, run a post-batch merge pass on close-pair
            partitions (§3.5 merge-sensitivity ablation). Default False.
        activations_cache_dir: if set, write each batch's raw activations
            (plus prompt/position metadata) as a sharded `.npz`.
    """
    from pathlib import Path
    from .extraction import (
        extract_per_position as _default_extract,
        normalize_extractor_output,
    )

    if extract_fn is None:
        extract_fn = _default_extract
    if extract_kwargs is None:
        extract_kwargs = {}
    if prompt_batch_size < 1:
        raise ValueError("prompt_batch_size must be at least 1")

    dictionary = Dictionary(
        center=calibration.center,
        threshold=calibration.threshold,
        merge_close=merge_close,
    )

    activations_cache_dir = (
        Path(activations_cache_dir) if activations_cache_dir is not None else None
    )
    if activations_cache_dir is not None:
        activations_cache_dir.mkdir(parents=True, exist_ok=True)

    total_acts = 0
    total_tokens = 0
    total_prompts = 0
    extract_time = 0.0
    clustering_time = 0.0
    n_forward_passes = 0

    MAX_EXAMPLES_PER_PARTITION = 10
    snapshots: list[Snapshot] = []

    t_start = time.time()
    saturated = False
    last_partition_batch = 0
    batch_count = 0

    for prompt_batch in _iter_prompt_batches(texts, prompt_batch_size, seed):
        if saturated:
            break
        if max_prompts is not None:
            remaining = max_prompts - total_prompts
            if remaining <= 0:
                break
            if len(prompt_batch) > remaining:
                prompt_batch = prompt_batch[:remaining]

        t0 = time.time()
        raw = extract_fn(model, prompt_batch, hook_name, **extract_kwargs)
        result = normalize_extractor_output(raw)
        extract_time += time.time() - t0
        n_forward_passes += result.n_forward_passes
        total_tokens += result.n_tokens

        x_new = result.x
        n_acts = len(x_new)

        if n_acts == 0:
            total_prompts += len(prompt_batch)
            continue

        if activations_cache_dir is not None:
            shard_path = activations_cache_dir / f"shard_{batch_count:06d}.npz"
            np.savez(
                shard_path,
                x=x_new.astype(np.float32),
                prompt_ids=result.prompt_ids,
                position_ids=result.position_ids,
                prompts=np.array(prompt_batch, dtype=object),
            )

        n_before = len(dictionary)
        t_clust = time.time()
        assignments = dictionary.add_batch(
            x_batch=x_new,
            iteration=batch_count,
            global_index_start=total_acts,
        )
        clustering_time += time.time() - t_clust
        if len(dictionary) > n_before:
            last_partition_batch = batch_count

        # Per-partition closest/furthest prompt heaps for visualisation.
        # Reuse the per-batch distances stashed by add_batch — no recompute.
        prompt_ids = result.prompt_ids
        position_ids = result.position_ids
        per_act_dists = dictionary._last_dists
        for act_idx, pids in enumerate(assignments):
            if not pids:
                continue
            pid_in_batch = int(prompt_ids[act_idx]) if len(prompt_ids) else 0
            pos = int(position_ids[act_idx]) if len(position_ids) else 0
            if pid_in_batch >= len(prompt_batch):
                continue
            dist = float(per_act_dists[act_idx])
            for partition_id in pids:
                p = dictionary.partitions[partition_id]
                if len(p.sample_prompts) < MAX_EXAMPLES_PER_PARTITION:
                    heapq.heappush(p.sample_prompts,
                                   (-dist, prompt_batch[pid_in_batch], pos))
                elif dist < -p.sample_prompts[0][0]:
                    heapq.heapreplace(p.sample_prompts,
                                      (-dist, prompt_batch[pid_in_batch], pos))
                if len(p.boundary_prompts) < MAX_EXAMPLES_PER_PARTITION:
                    heapq.heappush(p.boundary_prompts,
                                   (dist, prompt_batch[pid_in_batch], pos))
                elif dist > p.boundary_prompts[0][0]:
                    heapq.heapreplace(p.boundary_prompts,
                                      (dist, prompt_batch[pid_in_batch], pos))

        total_acts += n_acts
        total_prompts += len(prompt_batch)
        batch_count += 1

        if batch_count % log_cadence == 0:
            elapsed = time.time() - t_start
            logger.info("  %d acts | %d tokens | %d partitions | %.0fs",
                        total_acts, total_tokens, len(dictionary), elapsed)
            snapshots.append(Snapshot(
                n_activations=total_acts,
                n_prompts=total_prompts,
                n_partitions=len(dictionary),
                new_partition_rate=dictionary.new_partition_rate,
                elapsed_s=elapsed,
            ))
            if log_fn is not None:
                members = sorted(
                    [p.member_count for p in dictionary.partitions], reverse=True
                ) if dictionary.partitions else []
                metrics = {
                    "n_acts": total_acts,
                    "n_tokens": total_tokens,
                    "n_prompts": total_prompts,
                    "elapsed_s": elapsed,
                    "extract_time_s": extract_time,
                    "n_forward_passes": n_forward_passes,
                    "n_partitions": len(dictionary),
                    "new_partition_rate": dictionary.new_partition_rate,
                    "clustering_time_s": clustering_time,
                    "threshold": dictionary.threshold,
                }
                if members:
                    metrics["largest_partition"] = members[0]
                    metrics["top5_members"] = sum(members[:5])
                    metrics["top10_members"] = sum(members[:10])
                    metrics["median_members"] = float(np.median(members))
                    metrics["mean_members"] = float(np.mean(members))
                    metrics["singletons"] = sum(1 for m in members if m == 1)
                    metrics["partitions_gt10"] = sum(1 for m in members if m > 10)
                    metrics["partitions_gt100"] = sum(1 for m in members if m > 100)
                    metrics["gini"] = _gini(members)
                log_fn(metrics)

        if batch_count - last_partition_batch > saturation_window:
            saturated = True
            logger.info(
                "Saturated at %d acts (%d partitions)",
                total_acts, len(dictionary),
            )

        if batch_count % checkpoint_cadence == 0 and checkpoint_fn is not None:
            checkpoint_fn(dictionary, snapshots, {
                "total_acts": total_acts,
                "total_tokens": total_tokens,
                "total_prompts": total_prompts,
                "extract_time_s": extract_time,
                "clustering_time_s": clustering_time,
                "n_forward_passes": n_forward_passes,
                "batch_count": batch_count,
            })

        if max_tokens and total_tokens >= max_tokens:
            logger.info("Reached max_tokens=%d (%d prompts)", max_tokens, total_prompts)
            break
        if max_prompts and total_prompts >= max_prompts:
            logger.info("Reached max_prompts=%d (%d acts)", max_prompts, total_acts)
            break

    dictionary.finalize()

    elapsed = time.time() - t_start
    logger.info(
        "Discovery complete: %.1fs (%.1fs extraction, %d fwd), "
        "%d prompts, %d tokens, %d acts, %d partitions",
        elapsed, extract_time, n_forward_passes,
        total_prompts, total_tokens, total_acts, len(dictionary),
    )

    return DiscoveryResult(
        dictionary=dictionary,
        n_activations=total_acts,
        n_prompts=total_prompts,
        n_tokens=total_tokens,
        n_forward_passes=n_forward_passes,
        elapsed_s=elapsed,
        extraction_time_s=extract_time,
        clustering_time_s=clustering_time,
        saturated=saturated,
        snapshots=snapshots,
    )
