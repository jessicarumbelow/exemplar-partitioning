"""Compare exemplar-partition dictionaries against pretrained SAEs on the same model/layer.

Both sides are evaluated as soft activation matrices (n_samples × n_features):
the SAE via its native encode, EP via EPDictionarySAE.encode (the same adapter
SAEBench uses). For each feature on each side we take its top-k activating
inputs; correspondence is overlap of those sets. Symmetric: SAE-vs-EP uses
the exact same machinery as EP-vs-SAE.

Usage:
    uv run python -m scripts.compare_sae --model google/gemma-2-2b --layer 12
    uv run python -m scripts.compare_sae --model google/gemma-2-2b --layer 12 --sae-release gemma-scope-2b-pt-res-canonical --sae-id layer_12/width_16k/canonical
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

SAE_CONFIGS = {
    "pythia-70m-deduped": {
        "release": "pythia-70m-deduped-res-sm",
        "sae_id_template": "blocks.{layer}.hook_resid_post",
        "model_id": "EleutherAI/pythia-70m-deduped",
    },
    "pythia-160m-deduped": {
        "release": "pythia-160m-deduped-res-sm",
        "sae_id_template": "blocks.{layer}.hook_resid_post",
        "model_id": "EleutherAI/pythia-160m-deduped",
    },
    "gemma-2-2b": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical",
        "model_id": "google/gemma-2-2b",
    },
}

# Width tokens used in gemma-scope canonical sae_ids (e.g. layer_12/width_16k/canonical).
_GEMMA_WIDTH_TOKENS = {
    4096: "4k", 16384: "16k", 32768: "32k", 65536: "65k",
    131072: "131k", 262144: "262k", 524288: "524k", 1048576: "1m",
}


def _gemma_canonical_widths(layer: int) -> dict[int, str]:
    """Return {numeric_width: sae_id} for canonical gemma-scope SAEs available at this layer."""
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    lookup = get_pretrained_saes_directory()["gemma-scope-2b-pt-res-canonical"]
    out = {}
    for num, tok in _GEMMA_WIDTH_TOKENS.items():
        sae_id = f"layer_{layer}/width_{tok}/canonical"
        if sae_id in lookup.saes_map:
            out[num] = sae_id
    return out


def available_widths(model_short: str, layer: int) -> dict[int, tuple[str, str]]:
    """Return {numeric_width: (release, sae_id)} for SAEs we know how to load at this model+layer.

    Pythia res-sm is single-width — we can't introspect the trained width without loading,
    so we expose it under a sentinel key (-1) meaning "the only width available."
    """
    if model_short == "gemma-2-2b":
        release = "gemma-scope-2b-pt-res-canonical"
        return {w: (release, sid) for w, sid in _gemma_canonical_widths(layer).items()}
    if model_short in SAE_CONFIGS:
        cfg = SAE_CONFIGS[model_short]
        return {-1: (cfg["release"], cfg["sae_id_template"].format(layer=layer))}
    return {}


def pick_release_for_width(
    model_short: str, layer: int, n_concepts: int,
) -> tuple[str, str, int | None]:
    """Pick the (release, sae_id, width) whose width is closest to n_concepts in log space.

    Returns width=None for single-width models (no matching possible).
    Falls back to SAE_CONFIGS default if model_short isn't in available_widths.
    """
    import math
    widths = available_widths(model_short, layer)
    if -1 in widths:
        release, sae_id = widths[-1]
        return release, sae_id, None
    if not widths:
        cfg = SAE_CONFIGS[model_short]
        return cfg["release"], cfg["sae_id_template"].format(layer=layer), None
    n = max(int(n_concepts), 1)
    best = min(widths, key=lambda w: abs(math.log2(w) - math.log2(n)))
    release, sae_id = widths[best]
    return release, sae_id, best


def load_sae(model_short: str, layer: int, release: str | None = None, sae_id: str | None = None,
             n_concepts: int | None = None):
    """Load an SAE for (model_short, layer).

    - If release+sae_id are both given, load that exact SAE (no width matching).
    - Else if n_concepts is given, pick the width closest to n_concepts (log-space).
    - Else fall back to SAE_CONFIGS default.
    """
    from sae_lens import SAE

    if release is None or sae_id is None:
        if n_concepts is not None:
            release, sae_id, matched_width = pick_release_for_width(model_short, layer, n_concepts)
            if matched_width is not None:
                logger.info("Width-matched SAE: %d concepts → width %d", n_concepts, matched_width)
        else:
            cfg = SAE_CONFIGS[model_short]
            release = release or cfg["release"]
            sae_id = sae_id or cfg["sae_id_template"].format(layer=layer)

    logger.info("Loading SAE: release=%s, sae_id=%s", release, sae_id)
    t0 = time.time()
    sae, cfg_dict, sparsity = SAE.from_pretrained(release=release, sae_id=sae_id)
    logger.info("SAE loaded in %.1fs — %d features, d_in=%d", time.time() - t0, sae.cfg.d_sae, sae.cfg.d_in)
    return sae, cfg_dict, release, sae_id


def load_cached_activations(cache_dir: Path, n_tokens: int):
    """Load sharded activations written by `discover(activations_cache_dir=...)`.

    Returns (activations, prompts) compatible with `collect_activations`.
    Reads shards in order and stops once `n_tokens` is reached. Each shard
    is `{x: (n, D) float32, prompt_ids, position_ids, prompts}`.
    """
    shards = sorted(cache_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No activation shards in {cache_dir}")

    xs: list[np.ndarray] = []
    prompts: list[str] = []
    seen_prompts: set[str] = set()
    total = 0

    logger.info("Loading cached activations from %d shards in %s",
                len(shards), cache_dir)
    t0 = time.time()
    for shard in shards:
        if total >= n_tokens:
            break
        data = np.load(shard, allow_pickle=True)
        x = data["x"]
        xs.append(x)
        total += len(x)
        for p in data["prompts"]:
            if p not in seen_prompts:
                prompts.append(str(p))
                seen_prompts.add(str(p))

    activations = np.concatenate(xs, axis=0)[:n_tokens]
    logger.info("Loaded %d cached activation vectors in %.1fs from %d prompts",
                len(activations), time.time() - t0, len(prompts))
    return activations, prompts


def collect_activations(model, tokenizer, hook_name: str, n_tokens: int,
                        context_length: int, batch_size: int, device: str, seed: int = 0):
    """Collect residual stream activations from the Pile. Returns (activations, prompts).

    Pulls ``batch_size`` prompts from the stream, batch-tokenizes once, and
    runs a single forward pass per batch. Activations are gathered with a
    device-side keep-mask (skipping BOS and padding) and transferred to host
    in one tensor per batch.
    """
    from scripts.build_partitions import stream_batches

    text_stream = stream_batches(
        tokenizer, context_length, batch_size=batch_size,
        sampling_mode="full", seed=seed,
    )

    all_acts: list[np.ndarray] = []
    all_prompts: list[str] = []
    total_tokens = 0

    logger.info("Collecting %d tokens of activations at %s...", n_tokens, hook_name)
    t0 = time.time()

    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    while total_tokens < n_tokens:
        prompt_batch = [next(text_stream) for _ in range(batch_size)]
        batch = model.to_tokens(prompt_batch, prepend_bos=True, padding_side="right")
        lengths = (batch != pad_id).sum(dim=1)

        acts_store: dict = {}
        def hook_fn(act, hook):
            acts_store[hook.name] = act

        model.reset_hooks()
        model.add_hook(hook_name, hook_fn, "fwd")
        with torch.no_grad():
            model(batch)
        model.reset_hooks()

        act_tensor = acts_store[hook_name]  # (B, max_len, D), on-device
        max_len = act_tensor.shape[1]
        positions = torch.arange(max_len, device=act_tensor.device)
        keep_mask = (positions[None, :] < lengths[:, None]) & (positions[None, :] >= 1)
        # Transfer in source dtype, upcast on CPU — saves bandwidth for bf16 models.
        flat = act_tensor[keep_mask].detach().cpu().float().numpy()
        all_acts.append(flat)
        all_prompts.extend(prompt_batch)
        total_tokens += int((lengths - 1).clamp(min=0).sum().item())

    activations = np.concatenate(all_acts, axis=0)[:n_tokens]
    logger.info("Collected %d activation vectors in %.1fs from %d prompts",
                len(activations), time.time() - t0, len(all_prompts))
    return activations, all_prompts


def get_sae_features(sae, activations: np.ndarray, device: str, batch_size: int = 512,
                     top_k: int = 0) -> np.ndarray:
    """Encode activations through SAE, return (n_samples, n_features) activation matrix."""
    sae = sae.to(device)
    sae.eval()
    n = len(activations)
    all_feature_acts = []

    for start in range(0, n, batch_size):
        batch = torch.tensor(activations[start:start + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            feature_acts = sae.encode(batch)
        all_feature_acts.append(feature_acts.cpu().numpy())

    feature_acts = np.concatenate(all_feature_acts, axis=0)
    logger.info("SAE encoding done: %d samples × %d features, %.1f%% nonzero",
                feature_acts.shape[0], feature_acts.shape[1],
                100 * (feature_acts > 0).mean())
    return feature_acts


def get_ep_features(dictionary, activations: np.ndarray, device: str,
                    basis: str = "mean", batch_size: int = 512) -> np.ndarray:
    """Encode activations through an EP dictionary as soft activations.

    Uses EPDictionarySAE in signed-readout mode: returns (n_samples,
    n_partitions) of (x − center) · basis_i — the signed projection onto each
    partition's unit direction. _top_k_indicator filters to acts > 0 and ranks
    by magnitude, so the resulting top-k input set is identical to a ReLU-on-
    all readout but exposes the full directional alignment SAE features get
    compared against. The default top-1 VQ readout would constrain each EP
    feature's top-activating inputs to its own cell's members, artificially
    capping correspondence with SAE features that co-fire across inputs.

    `basis` selects the per-partition basis vector ("mean" = spherical mean
    of all member directions; "exemplar" = first-arrival activation, stored
    as a unit direction in centered space). Both are unit-norm.
    """
    from ep.saebench_adapter import EPDictionarySAE

    if not dictionary.partitions:
        return np.zeros((len(activations), 0), dtype=np.float32)

    ep_sae = EPDictionarySAE(
        dictionary=dictionary,
        model_name="",
        hook_layer=0,
        device=torch.device(device),
        dtype=torch.float32,
        basis=basis,
        readout="signed",
    )
    ep_sae.eval()

    n_partitions = len(dictionary.partitions)
    n = len(activations)
    all_acts = []
    for start in range(0, n, batch_size):
        batch = torch.tensor(activations[start:start + batch_size],
                             dtype=torch.float32, device=device)
        with torch.no_grad():
            feature_acts = ep_sae.encode(batch)
        all_acts.append(feature_acts[:, :n_partitions].cpu().numpy())

    ep_features = np.concatenate(all_acts, axis=0)
    logger.info("EP encoding done: %d samples × %d partitions, %.1f%% nonzero",
                ep_features.shape[0], ep_features.shape[1],
                100 * (ep_features > 0).mean())
    return ep_features


def _top_k_indicator(features: np.ndarray, top_k: int):
    """Sparse (n_samples, n_features) indicator: 1 if input is in feature's top-k.

    Top-k is restricted to inputs with positive activation (after ReLU).
    Returns (csr_matrix, n_active_per_feature).
    """
    from scipy.sparse import csr_matrix

    n_samples, n_features = features.shape
    rows: list[int] = []
    cols: list[int] = []
    n_active = np.zeros(n_features, dtype=np.int64)
    for f in range(n_features):
        acts = features[:, f]
        active_idx = np.where(acts > 0)[0]
        n_active[f] = len(active_idx)
        if len(active_idx) == 0:
            continue
        k_eff = min(top_k, len(active_idx))
        top_local = np.argpartition(acts[active_idx], -k_eff)[-k_eff:]
        top_global = active_idx[top_local]
        rows.extend(int(i) for i in top_global)
        cols.extend([f] * k_eff)
    M = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(n_samples, n_features),
    )
    return M, n_active


def compute_correspondence(sae_features: np.ndarray, ep_features: np.ndarray,
                           min_feature_activations: int = 20,
                           top_k_sae: int = 100) -> dict:
    """Symmetric feature correspondence between SAE features and EP partitions.

    Both sides are soft activations. For each feature on each side we take its
    top-k activating inputs (top-k restricted to positive activations). Best
    match for an A-feature is the B-feature with maximum top-k overlap.
    Precision/recall use each side's actual top-k size (capped by n_active).
    """
    n_samples, n_sae = sae_features.shape
    _, n_ep = ep_features.shape

    M_sae, sae_n_active = _top_k_indicator(sae_features, top_k_sae)
    M_ep, ep_n_active = _top_k_indicator(ep_features, top_k_sae)

    sae_set_sizes = np.minimum(sae_n_active, top_k_sae)
    ep_set_sizes = np.minimum(ep_n_active, top_k_sae)

    overlap = (M_sae.T @ M_ep).toarray()

    sae_eligible = sae_n_active >= min_feature_activations
    ep_eligible = ep_n_active >= min_feature_activations

    overlap_masked = overlap.copy()
    overlap_masked[:, ~ep_eligible] = -1
    overlap_masked[~sae_eligible, :] = -1

    sae_to_ep = []
    for i in np.where(sae_eligible)[0]:
        if not ep_eligible.any():
            break
        j = int(overlap_masked[i].argmax())
        if overlap[i, j] == 0:
            continue
        ov = int(overlap[i, j])
        precision = ov / max(int(sae_set_sizes[i]), 1)
        recall = ov / max(int(ep_set_sizes[j]), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        sae_to_ep.append({
            "sae_feature": int(i),
            "ep_partition": j,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_active": int(sae_n_active[i]),
            "overlap": ov,
        })

    ep_to_sae = []
    for j in np.where(ep_eligible)[0]:
        if not sae_eligible.any():
            break
        i = int(overlap_masked[:, j].argmax())
        if overlap[i, j] == 0:
            continue
        ov = int(overlap[i, j])
        precision = ov / max(int(ep_set_sizes[j]), 1)
        recall = ov / max(int(sae_set_sizes[i]), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        ep_to_sae.append({
            "ep_partition": int(j),
            "sae_feature": i,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_members": int(ep_n_active[j]),
            "overlap": ov,
        })

    hungarian = _hungarian_match(
        overlap, sae_set_sizes, ep_set_sizes, sae_eligible, ep_eligible,
    )

    coherence = _top1_coherence(sae_features, ep_features, top_k=5)

    return {
        "sae_to_ep": sae_to_ep,
        "ep_to_sae": ep_to_sae,
        "hungarian": hungarian,
        "coherence": coherence,
    }


def _top1_coherence(sae_features: np.ndarray, ep_features: np.ndarray,
                    top_k: int = 5) -> dict:
    """Per-partition concept-coherence: for each EP partition, take its top-K
    most-activating samples, ask which SAE feature is top-1 on each, count
    distinct features across the K.

    1 distinct = all K exemplars peak on the same SAE feature → clean concept match.
    K distinct = exemplars split across K different SAE features → partition is
    finer than SAE, or composite, or noisy.

    Concept-level: insensitive to which exact tokens overlap, sensitive to whether
    the SAE recognises the partition's exemplars as the same thing.
    """
    from collections import Counter

    n_samples, n_ep = ep_features.shape
    histogram = Counter()
    n_evaluated = 0

    for j in range(n_ep):
        col = ep_features[:, j]
        # Need at least top_k positive activations to evaluate
        n_pos = int((col > 0).sum())
        if n_pos < top_k:
            continue
        top_idx = np.argpartition(col, -top_k)[-top_k:]
        sae_top1 = sae_features[top_idx, :].argmax(axis=1)
        n_distinct = len(set(int(x) for x in sae_top1))
        histogram[n_distinct] += 1
        n_evaluated += 1

    return {
        "top_k": top_k,
        "n_evaluated": n_evaluated,
        "n_partitions": n_ep,
        "histogram": {k: int(histogram[k]) for k in range(1, top_k + 1)},
        "histogram_frac": {
            k: float(histogram[k] / max(n_evaluated, 1)) for k in range(1, top_k + 1)
        },
        "frac_all_same": float(histogram[1] / max(n_evaluated, 1)),
        "mean_distinct": float(
            sum(k * histogram[k] for k in histogram) / max(n_evaluated, 1)
        ),
    }


def _hungarian_match(overlap: np.ndarray, sae_set_sizes: np.ndarray,
                     ep_set_sizes: np.ndarray, sae_eligible: np.ndarray,
                     ep_eligible: np.ndarray) -> list[dict]:
    """Optimal one-to-one SAE↔EP matching via Hungarian, scored by pairwise F1.

    F1 here uses the dice form 2·overlap / (|A|+|B|), equivalent to harmonic
    mean of precision and recall when set sizes are the per-feature top-k sizes.
    Returns ≤ min(n_sae_eligible, n_ep_eligible) pairs (zero-overlap matches
    are dropped). Result is direction-agnostic: same answer regardless of which
    side is rows.
    """
    from scipy.optimize import linear_sum_assignment

    sae_idx = np.where(sae_eligible)[0]
    ep_idx = np.where(ep_eligible)[0]
    if len(sae_idx) == 0 or len(ep_idx) == 0:
        return []

    sub_overlap = overlap[np.ix_(sae_idx, ep_idx)].astype(np.float64)
    denom = sae_set_sizes[sae_idx, None] + ep_set_sizes[None, ep_idx]
    f1_mat = 2.0 * sub_overlap / np.maximum(denom, 1)

    row_ind, col_ind = linear_sum_assignment(-f1_mat)

    matches = []
    for r, c in zip(row_ind, col_ind):
        i = int(sae_idx[r])
        j = int(ep_idx[c])
        ov = int(overlap[i, j])
        if ov == 0:
            continue
        precision = ov / max(int(sae_set_sizes[i]), 1)
        recall = ov / max(int(ep_set_sizes[j]), 1)
        matches.append({
            "sae_feature": i,
            "ep_partition": j,
            "precision": precision,
            "recall": recall,
            "f1": float(f1_mat[r, c]),
            "overlap": ov,
            "sae_set_size": int(sae_set_sizes[i]),
            "ep_set_size": int(ep_set_sizes[j]),
        })
    return matches


def print_summary(results: dict, sae_features: np.ndarray, n_ep_partitions: int):
    sae_to_ep = results["sae_to_ep"]
    ep_to_sae = results["ep_to_sae"]
    hungarian = results.get("hungarian", [])
    coherence = results.get("coherence", {})

    print("\n" + "=" * 70)
    print("FEATURE CORRESPONDENCE SUMMARY")
    print("=" * 70)

    print(f"\nSAE features evaluated: {len(sae_to_ep)} "
          f"(of {sae_features.shape[1]} total, filtered by min activations)")
    print(f"EP partitions evaluated: {len(ep_to_sae)} (of {n_ep_partitions} total)")

    if sae_to_ep:
        f1s = [m["f1"] for m in sae_to_ep]
        print("\n--- SAE → EP (each SAE feature's best EP match) ---")
        print(f"  Mean F1:   {np.mean(f1s):.3f}")
        print(f"  Median F1: {np.median(f1s):.3f}")
        print(f"  F1 > 0.5:  {sum(1 for f in f1s if f > 0.5)} / {len(f1s)} "
              f"({100 * sum(1 for f in f1s if f > 0.5) / len(f1s):.1f}%)")
        print(f"  F1 > 0.3:  {sum(1 for f in f1s if f > 0.3)} / {len(f1s)} "
              f"({100 * sum(1 for f in f1s if f > 0.3) / len(f1s):.1f}%)")

        top = sorted(sae_to_ep, key=lambda x: x["f1"], reverse=True)[:10]
        print("\n  Top 10 SAE→EP matches:")
        for m in top:
            print(f"    SAE feat {m['sae_feature']:5d} → EP partition {m['ep_partition']:4d}  "
                  f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    if ep_to_sae:
        f1s = [m["f1"] for m in ep_to_sae]
        print("\n--- EP → SAE (each EP partition's best SAE match) ---")
        print(f"  Mean F1:   {np.mean(f1s):.3f}")
        print(f"  Median F1: {np.median(f1s):.3f}")
        print(f"  F1 > 0.5:  {sum(1 for f in f1s if f > 0.5)} / {len(f1s)} "
              f"({100 * sum(1 for f in f1s if f > 0.5) / len(f1s):.1f}%)")
        print(f"  F1 > 0.3:  {sum(1 for f in f1s if f > 0.3)} / {len(f1s)} "
              f"({100 * sum(1 for f in f1s if f > 0.3) / len(f1s):.1f}%)")

        top = sorted(ep_to_sae, key=lambda x: x["f1"], reverse=True)[:10]
        print("\n  Top 10 EP→SAE matches:")
        for m in top:
            print(f"    EP partition {m['ep_partition']:4d} → SAE feat {m['sae_feature']:5d}  "
                  f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    if hungarian:
        f1s = [m["f1"] for m in hungarian]
        print("\n--- Hungarian (optimal one-to-one SAE↔EP) ---")
        print(f"  Matched pairs: {len(hungarian)}")
        print(f"  Mean F1:   {np.mean(f1s):.3f}")
        print(f"  Median F1: {np.median(f1s):.3f}")
        print(f"  F1 > 0.5:  {sum(1 for f in f1s if f > 0.5)} / {len(f1s)} "
              f"({100 * sum(1 for f in f1s if f > 0.5) / len(f1s):.1f}%)")
        print(f"  F1 > 0.3:  {sum(1 for f in f1s if f > 0.3)} / {len(f1s)} "
              f"({100 * sum(1 for f in f1s if f > 0.3) / len(f1s):.1f}%)")

        top = sorted(hungarian, key=lambda x: x["f1"], reverse=True)[:10]
        print("\n  Top 10 Hungarian matches:")
        for m in top:
            print(f"    EP partition {m['ep_partition']:4d} ↔ SAE feat {m['sae_feature']:5d}  "
                  f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    if coherence and coherence.get("n_evaluated", 0) > 0:
        k = coherence["top_k"]
        n = coherence["n_evaluated"]
        print(f"\n--- Per-partition top-1 SAE coherence (top-{k} exemplars) ---")
        print(f"  Partitions evaluated: {n} / {coherence['n_partitions']}")
        print(f"  Mean distinct top-1 SAE features per partition: "
              f"{coherence['mean_distinct']:.2f}  (1 = clean concept match, "
              f"{k} = totally diffuse)")
        print(f"  Frac with all {k} exemplars sharing top-1: "
              f"{coherence['frac_all_same']:.3f}")
        print("  Histogram:")
        for d in range(1, k + 1):
            count = coherence["histogram"][d]
            frac = coherence["histogram_frac"][d]
            bar = "█" * int(40 * frac)
            print(f"    {d}: {count:5d} ({frac:.3f}) {bar}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare EP concepts against pretrained SAE features")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--model-short", type=str, default=None)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--hook-name", type=str, default=None)
    parser.add_argument("--sae-release", type=str, default=None)
    parser.add_argument("--sae-id", type=str, default=None)
    parser.add_argument("--n-concepts", type=int, default=None,
                        help="Number of EP partitions; selects the SAE width "
                             "closest to this in log space. Ignored if "
                             "--sae-release and --sae-id are both given.")
    parser.add_argument("--n-tokens", type=int, default=500_000,
                        help="Number of activation tokens to collect for comparison.")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ep-dictionary", type=Path, default=None,
                        help="Path to a pre-built EP dictionary pickle. If not provided, builds one.")
    parser.add_argument("--ep-percentile", type=float, default=None,
                        help="Single percentile for EP dictionary. Mutually exclusive with --sweep.")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Comma-separated percentiles to sweep (e.g. '5,10,15,20,30').")
    parser.add_argument("--ep-max-tokens", type=int, default=None,
                        help="Max tokens for EP dictionary building. Defaults to --n-tokens.")
    parser.add_argument("--top-k-sae", type=int, default=100,
                        help="Top-k activating inputs per SAE feature for overlap computation.")
    parser.add_argument("--min-activations", type=int, default=20,
                        help="Minimum activations/members to include a feature/concept in comparison.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save results JSON to this path.")
    parser.add_argument("--activations-cache", type=Path, default=None,
                        help="Directory of sharded activation npz files written by "
                             "build_partitions. If set, skip the model-load + "
                             "collect_activations step and read shards instead.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ep")
    parser.add_argument("--wandb-entity", default=None,
                        help="wandb entity (defaults to your wandb auth's default).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)

    if args.model_short is None:
        args.model_short = args.model.split("/")[-1]
    hook_name = args.hook_name or f"blocks.{args.layer}.hook_resid_post"
    if args.ep_max_tokens is None:
        args.ep_max_tokens = args.n_tokens
    if args.ep_percentile is None and args.sweep is None:
        args.ep_percentile = 10
    percentiles = (
        [float(p) for p in args.sweep.split(",")]
        if args.sweep else [args.ep_percentile]
    )

    # --- Load SAE (always needed) ---
    sae, sae_cfg, resolved_release, resolved_sae_id = load_sae(
        args.model_short, args.layer, args.sae_release, args.sae_id,
        n_concepts=args.n_concepts,
    )

    # --- Activations: load cache if present, else run model ---
    if args.activations_cache is not None and args.activations_cache.exists():
        activations, prompts = load_cached_activations(
            args.activations_cache, args.n_tokens,
        )
    else:
        if args.activations_cache is not None:
            logger.warning(
                "--activations-cache %s does not exist; falling back to model run",
                args.activations_cache,
            )
        import transformer_lens as tl

        logger.info("Loading model %s on %s", args.model, args.device)
        t0 = time.time()
        model = tl.HookedTransformer.from_pretrained_no_processing(
            args.model, device=args.device, dtype=torch.bfloat16)
        model.eval()
        logger.info("Model loaded in %.1fs (d_model=%d)",
                    time.time() - t0, model.cfg.d_model)
        activations, prompts = collect_activations(
            model, model.tokenizer, hook_name, args.n_tokens,
            args.context_length, args.batch_size, args.device, args.seed,
        )

    # --- SAE encode ---
    sae_features = get_sae_features(sae, activations, args.device)

    # --- Run EP at each percentile, both bases ---
    from ep.discovery import Dictionary, calibrate
    from ep.saebench_adapter import EPDictionarySAE

    BASES = EPDictionarySAE.BASES  # ("mean", "exemplar")
    all_results: dict[float, dict[str, dict]] = {}

    batch_sz = args.batch_size

    def _yield_calibration_batches(n_batches: int):
        for i in range(n_batches):
            start = i * batch_sz
            end = min(start + batch_sz, len(activations))
            if start >= len(activations):
                return
            yield activations[start:end]

    for percentile in percentiles:
        print(f"\n{'=' * 70}")
        print(f"EP percentile = {percentile}")
        print("=" * 70)

        if args.ep_dictionary is not None and len(percentiles) == 1:
            import pickle
            logger.info("Loading EP dictionary from %s", args.ep_dictionary)
            with open(args.ep_dictionary, "rb") as f:
                dictionary = pickle.load(f)
        else:
            logger.info("Building EP dictionary (p=%.1f) from %d activations...",
                        percentile, len(activations))
            t0 = time.time()
            calibration_n = min(100, max(1, len(activations) // batch_sz))
            calibration = calibrate(
                _yield_calibration_batches(calibration_n),
                n_tokens=min(len(activations), calibration_n * batch_sz),
                percentile=percentile,
            )
            dictionary = Dictionary(
                center=calibration.center,
                threshold=calibration.threshold,
            )
            for start in range(0, min(len(activations), args.ep_max_tokens), batch_sz):
                end = min(start + batch_sz, len(activations), args.ep_max_tokens)
                dictionary.add_batch(
                    activations[start:end], iteration=start // batch_sz,
                )

            dictionary.finalize()
            logger.info("EP dictionary built in %.1fs: %d partitions",
                        time.time() - t0, len(dictionary))

        per_basis: dict[str, dict] = {}
        for basis in BASES:
            print(f"\n--- basis = {basis} ---")
            ep_features = get_ep_features(
                dictionary, activations, args.device, basis=basis,
            )
            results = compute_correspondence(
                sae_features, ep_features,
                min_feature_activations=args.min_activations,
                top_k_sae=args.top_k_sae,
            )
            results["n_partitions"] = len(dictionary.partitions)
            results["percentile"] = percentile
            results["threshold"] = dictionary.threshold
            results["basis"] = basis
            print_summary(results, sae_features, len(dictionary.partitions))
            per_basis[basis] = results

        all_results[percentile] = per_basis

    if len(percentiles) > 1:
        print(f"\n{'=' * 70}")
        print("SWEEP SUMMARY: percentile × basis → n_partitions → F1")
        print("=" * 70)
        print(f"  {'p':>5}  {'basis':>9}  {'partitions':>11}  {'threshold':>10}  "
              f"{'SAE→EP F1':>11}  {'EP→SAE F1':>11}  {'Hung F1':>9}  {'n_match':>8}  "
              f"{'all_same':>8}  {'mean_dist':>9}")
        print(f"  {'-' * 5}  {'-' * 9}  {'-' * 11}  {'-' * 10}  {'-' * 11}  "
              f"{'-' * 11}  {'-' * 9}  {'-' * 8}  {'-' * 8}  {'-' * 9}")
        for p in percentiles:
            for basis, r in all_results[p].items():
                sae_f1 = np.mean([m["f1"] for m in r["sae_to_ep"]]) if r["sae_to_ep"] else 0
                ep_f1 = np.mean([m["f1"] for m in r["ep_to_sae"]]) if r["ep_to_sae"] else 0
                hung = r.get("hungarian", [])
                hung_f1 = np.mean([m["f1"] for m in hung]) if hung else 0
                coh = r.get("coherence", {})
                all_same = coh.get("frac_all_same", 0.0)
                mean_dist = coh.get("mean_distinct", 0.0)
                print(f"  {p:>5.0f}  {basis:>9}  {r['n_partitions']:>11d}  {r['threshold']:>10.4f}  "
                      f"{sae_f1:>11.3f}  {ep_f1:>11.3f}  {hung_f1:>9.3f}  {len(hung):>8d}  "
                      f"{all_same:>8.3f}  {mean_dist:>9.2f}")
        print()

    if args.output is not None:
        import json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "config": {
                "model": args.model,
                "layer": args.layer,
                "n_tokens": len(activations),
                "n_sae_features": sae_features.shape[1],
                "sae_release": resolved_release,
                "sae_id": resolved_sae_id,
                "sae_d_sae": int(sae.cfg.d_sae),
                "n_concepts_requested": args.n_concepts,
                "top_k_sae": args.top_k_sae,
                "percentiles": percentiles,
            },
            "bases": list(BASES),
            "results_by_percentile": {
                # nested as {percentile: {basis: results}} so each percentile
                # carries both bases — readers must iterate over basis keys.
                str(p): per_basis for p, per_basis in all_results.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2)
        logger.info("Results saved to %s", args.output)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=f"compare_sae_{args.model_short}_L{args.layer}")
        for p, per_basis in all_results.items():
            for basis, r in per_basis.items():
                sae_f1s = [m["f1"] for m in r["sae_to_ep"]]
                ep_f1s = [m["f1"] for m in r["ep_to_sae"]]
                hung_f1s = [m["f1"] for m in r.get("hungarian", [])]
                coh = r.get("coherence", {})
                log_dict = {
                    "percentile": p,
                    "basis": basis,
                    "n_partitions": r["n_partitions"],
                    "threshold": r["threshold"],
                    f"{basis}/sae_to_ep/mean_f1": np.mean(sae_f1s) if sae_f1s else 0,
                    f"{basis}/sae_to_ep/median_f1": np.median(sae_f1s) if sae_f1s else 0,
                    f"{basis}/sae_to_ep/frac_above_0.5": sum(1 for f in sae_f1s if f > 0.5) / max(len(sae_f1s), 1),
                    f"{basis}/ep_to_sae/mean_f1": np.mean(ep_f1s) if ep_f1s else 0,
                    f"{basis}/ep_to_sae/median_f1": np.median(ep_f1s) if ep_f1s else 0,
                    f"{basis}/ep_to_sae/frac_above_0.5": sum(1 for f in ep_f1s if f > 0.5) / max(len(ep_f1s), 1),
                    f"{basis}/hungarian/mean_f1": np.mean(hung_f1s) if hung_f1s else 0,
                    f"{basis}/hungarian/median_f1": np.median(hung_f1s) if hung_f1s else 0,
                    f"{basis}/hungarian/frac_above_0.5": sum(1 for f in hung_f1s if f > 0.5) / max(len(hung_f1s), 1),
                    f"{basis}/hungarian/n_matched": len(hung_f1s),
                    f"{basis}/coherence/mean_distinct": coh.get("mean_distinct", 0.0),
                    f"{basis}/coherence/frac_all_same": coh.get("frac_all_same", 0.0),
                    f"{basis}/coherence/n_evaluated": coh.get("n_evaluated", 0),
                }
                for d, frac in coh.get("histogram_frac", {}).items():
                    log_dict[f"{basis}/coherence/hist_k{d}_frac"] = frac
                wandb.log(log_dict)
        wandb.finish()


if __name__ == "__main__":
    main()
