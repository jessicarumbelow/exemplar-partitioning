"""Resolution-path experiment: pairs of exemplars are direct neighbours in
a coarse EP dictionary; trace the shortest path between them in
finer-resolution dictionaries.

Pipeline (single model load):
1. Load the language model once.
2. For each dictionary at the same (model, layer): logit-lens every
   exemplar to get its top tokens, build the cosine-distance matrix on
   exemplars.
3. In the *seed* (coarsest) dictionary, pick top-N
   cleanly-interpretable direct-neighbour pairs as anchor pairs.
4. For each pair, take their exemplar directions as fixed anchors. For
   each (finer) dictionary, find the partitions whose exemplars are
   closest to those two anchors (i.e., assign the coarse exemplar
   direction to the fine dictionary). Compute shortest path in the
   k-NN graph between the two assignments.
5. Save per-dictionary summaries + per-pair path data.

Run:
    uv run python -m scripts.exp_resolution_paths --dict-paths d1.pkl,d2.pkl,d3.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _logit_lens(exemplars: np.ndarray, model, batch_size: int = 64,
                k_top: int = 8) -> list[str]:
    device = model.W_U.device
    dtype = model.W_U.dtype
    K = exemplars.shape[0]
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, K, batch_size):
            batch = torch.tensor(
                exemplars[i:i + batch_size], dtype=dtype, device=device,
            )
            normed = model.ln_final(batch)
            logits = normed @ model.W_U
            top_ids = logits.topk(k_top, dim=-1).indices.cpu().tolist()
            for ids in top_ids:
                toks = [model.tokenizer.decode([t]).strip() for t in ids]
                out.append(", ".join(t for t in toks if t)[:200])
    return out


def _cleanliness(s: str) -> float:
    parts = [p.strip() for p in s.split(",")]
    if not parts:
        return 0.0
    return sum(1 for p in parts if re.match(r"^[a-zA-Z]{2,15}$", p)) / len(parts)


def _knn_graph(G: np.ndarray, k: int = 3):
    from scipy.sparse import csr_matrix
    K = G.shape[0]
    nbr = np.argsort(G, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(K), k)
    cols = nbr.flatten()
    A = csr_matrix((np.ones_like(rows, dtype=float), (rows, cols)),
                   shape=(K, K))
    return A.maximum(A.T)


def _shortest_path(A, src: int, dst: int):
    from scipy.sparse.csgraph import shortest_path
    H, preds = shortest_path(A, directed=False, unweighted=True,
                             return_predecessors=True)
    if not np.isfinite(H[src, dst]) or src == dst:
        return [src] if src == dst else None, int(H[src, dst]) if np.isfinite(H[src, dst]) else None
    path = [dst]
    cur = dst
    while cur != src:
        cur = int(preds[src, cur])
        if cur < 0:
            return None, None
        path.append(cur)
    path.reverse()
    return path, len(path) - 1


def _percentile_tag(dict_path: Path) -> str:
    name = dict_path.parent.parent.name
    m = re.search(r"_p(\d+(?:p\d+)?)_", name)
    return f"p{m.group(1).replace('p', '.')}" if m else "p?"


def _build_dict_record(dict_path: Path, model, basis: str = "exemplar") -> dict:
    """Logit-lens + geom matrix for one dictionary, using either
    exemplar_direction (first-arrival) or mean_member_direction (cell mean).

    ``exemplar_directions`` (first-arrival) is always returned alongside
    ``directions`` (the basis used for geom + logit-lens) so identity
    tracking can run regardless of the rendering basis.
    """
    with open(dict_path, "rb") as f:
        dictionary = pickle.load(f)
    K = len(dictionary.partitions)
    exemplar_directions = np.stack(
        [p.exemplar_direction for p in dictionary.partitions]
    )
    if basis == "exemplar":
        directions = exemplar_directions
    elif basis == "mean":
        directions = np.stack(
            [p.mean_member_direction for p in dictionary.partitions]
        )
    else:
        raise ValueError(f"unknown basis: {basis!r}")
    member_counts = np.array(
        [p.member_count for p in dictionary.partitions], dtype=np.int64,
    )
    t0 = time.time()
    top_tokens = _logit_lens(directions, model)
    logger.info("  Logit-lensed %d %s directions in %.1fs",
                K, basis, time.time() - t0)
    sim = directions @ directions.T
    np.clip(sim, -1.0, 1.0, out=sim)
    G = (1.0 - sim).astype(np.float32)
    np.fill_diagonal(G, 0.0)
    return {
        "dict": dictionary,
        "K": K,
        "threshold": float(dictionary.threshold),
        "exemplars": directions,
        "exemplar_directions": exemplar_directions,
        "member_counts": member_counts,
        "top_tokens": top_tokens,
        "G": G,
        "basis": basis,
    }


def _identity_match(
    src_exemplars: np.ndarray,
    dst_exemplars: np.ndarray,
    cosine_eps: float = 1e-4,
) -> np.ndarray:
    """For each row in ``src_exemplars`` return the index in ``dst_exemplars``
    with cosine ≥ 1 - eps (identity), else -1.

    Same first-arrival activation under the same calibration produces an
    identical exemplar_direction (centered + L2-normalised on construction),
    so identity is a 1:1 match modulo float jitter rather than a near-neighbour
    search.
    """
    if src_exemplars.shape[0] == 0 or dst_exemplars.shape[0] == 0:
        return np.full(src_exemplars.shape[0], -1, dtype=np.int64)
    sim = src_exemplars @ dst_exemplars.T
    best_idx = sim.argmax(axis=1)
    best_sim = sim[np.arange(sim.shape[0]), best_idx]
    matched = best_sim >= 1.0 - cosine_eps
    out = np.where(matched, best_idx, -1).astype(np.int64)
    return out


def _pick_anchor_pairs(
    seed: dict,
    n_pairs: int = 5,
    candidate_mask: np.ndarray | None = None,
    knn_k: int = 3,
) -> list[tuple[int, int]]:
    """Pick top-N most cleanly-interpretable direct-neighbour pairs in
    seed dict's k-NN graph, restricted to ``candidate_mask`` if given.

    Both endpoints of a pair must be candidates. Edges where either endpoint
    is masked out are skipped.
    """
    K = seed["K"]
    G = seed["G"]
    top = seed["top_tokens"]
    A = _knn_graph(G, k=knn_k)
    if candidate_mask is None:
        candidate_mask = np.ones(K, dtype=bool)
    pairs = []
    coo = A.tocoo()
    for i, j in zip(coo.row, coo.col):
        if i >= j:
            continue
        if not (candidate_mask[i] and candidate_mask[j]):
            continue
        score = (_cleanliness(top[i]) + _cleanliness(top[j])) / 2
        pairs.append((float(score), float(G[i, j]), int(i), int(j)))
    pairs.sort(key=lambda x: (-x[0], x[1]))
    return [(i, j) for _, _, i, j in pairs[:n_pairs]]


def _assign_direction(direction: np.ndarray, target_exemplars: np.ndarray) -> int:
    """Find the partition in target_exemplars whose exemplar is closest in
    cosine distance to the given direction. direction is already a centered
    unit direction (since exemplars from any percentile build at the same
    (model, layer) live in the same centered space)."""
    sim = target_exemplars @ direction
    return int(np.argmax(sim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--model-short", default="gemma-2-2b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--dict-paths", required=True,
                    help="Comma-separated. ORDER MATTERS: seed first (coarsest), "
                         "then progressively finer.")
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--basis", choices=("exemplar", "mean"), default="exemplar",
                    help="Direction basis: 'exemplar' (first-arrival) or "
                         "'mean' (spherical mean of cell members).")
    ap.add_argument("--identity-track", action="store_true",
                    help="Restrict anchor candidates to seed exemplars whose "
                         "first-arrival direction also appears as an exemplar "
                         "in every other listed dictionary (cosine identity). "
                         "Per-resolution anchor IDs become genuine cross-"
                         "dictionary partition matches rather than nearest-"
                         "direction projections.")
    ap.add_argument("--max-members", type=int, default=0,
                    help="If > 0, restrict anchor candidates to seed partitions "
                         "with member_count <= this value. Filters out the "
                         "high-N grammatical / function-word attractors so the "
                         "surfaced anchors are content-coherent.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)

    dict_paths = [Path(p.strip()) for p in args.dict_paths.split(",") if p.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)

    import transformer_lens as tl
    logger.info("Loading model %s on %s", args.model, args.device)
    t0 = time.time()
    model = tl.HookedTransformer.from_pretrained_no_processing(
        args.model, device=args.device, dtype=torch.bfloat16,
    )
    model.eval()
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # Build records for every dictionary (logit-lens + geom)
    records: list[dict] = []
    for dp in dict_paths:
        tag = _percentile_tag(dp)
        logger.info("=== %s   %s   basis=%s ===", tag, dp.name, args.basis)
        rec = _build_dict_record(dp, model, basis=args.basis)
        rec["tag"] = tag
        rec["path"] = str(dp)
        records.append(rec)

    # Seed = coarsest = first
    seed = records[0]
    logger.info("Seed dictionary: %s, K=%d", seed["tag"], seed["K"])

    # Build identity maps seed → each finer dict (by exemplar_direction).
    # identity_map[k] is a length-K_seed int array; entry i is the partition
    # ID in records[k] whose first-arrival exemplar matches seed exemplar i,
    # or -1 if no identity match exists at that resolution.
    identity_map: list[np.ndarray] = []
    seed_e = seed["exemplar_directions"]
    for rec in records:
        m = _identity_match(seed_e, rec["exemplar_directions"])
        identity_map.append(m)
        n_matched = int((m >= 0).sum())
        logger.info(
            "  identity %s → %s: %d / %d seed exemplars survive",
            seed["tag"], rec["tag"], n_matched, seed["K"],
        )

    # Build candidate mask over seed exemplars.
    candidate_mask = np.ones(seed["K"], dtype=bool)
    if args.identity_track:
        survives_all = np.ones(seed["K"], dtype=bool)
        for k, rec in enumerate(records):
            if rec is seed:
                continue
            survives_all &= identity_map[k] >= 0
        candidate_mask &= survives_all
        logger.info(
            "identity-track: %d / %d seed exemplars survive in every dict",
            int(candidate_mask.sum()), seed["K"],
        )
    if args.max_members > 0:
        small = seed["member_counts"] <= args.max_members
        candidate_mask &= small
        logger.info(
            "max-members %d: %d / %d seed exemplars are small",
            args.max_members, int(small.sum()), seed["K"],
        )
        logger.info(
            "candidate mask after both filters: %d / %d",
            int(candidate_mask.sum()), seed["K"],
        )

    pairs = _pick_anchor_pairs(
        seed, n_pairs=args.n_pairs, candidate_mask=candidate_mask,
    )
    logger.info("Picked %d direct-neighbour anchor pairs in seed:", len(pairs))
    for a, b in pairs:
        logger.info(
            "  p%d (N=%d, %s)  ↔  p%d (N=%d, %s)",
            a, int(seed["member_counts"][a]), seed["top_tokens"][a][:50],
            b, int(seed["member_counts"][b]), seed["top_tokens"][b][:50],
        )

    # For each pair × resolution: trace the path. In identity-track mode
    # the per-resolution anchor IDs are taken from the identity map (so the
    # same first-arrival activation anchors the cell at every resolution);
    # otherwise fall back to nearest-direction projection.
    pair_results = []
    for (sa, sb) in pairs:
        anchor_a_dir = seed["exemplars"][sa]
        anchor_b_dir = seed["exemplars"][sb]
        per_resolution = []
        for k, rec in enumerate(records):
            if args.identity_track:
                ta = int(identity_map[k][sa])
                tb = int(identity_map[k][sb])
                if ta < 0 or tb < 0:
                    # Should not happen under identity_track filtering, but
                    # guard against masking bugs.
                    raise RuntimeError(
                        f"identity-track lost anchor at {rec['tag']}: "
                        f"seed {sa}→{ta}, seed {sb}→{tb}"
                    )
            else:
                ta = _assign_direction(anchor_a_dir, rec["exemplars"])
                tb = _assign_direction(anchor_b_dir, rec["exemplars"])
            A = _knn_graph(rec["G"], k=3)
            path, hops = _shortest_path(A, ta, tb)
            per_resolution.append({
                "tag": rec["tag"],
                "K": rec["K"],
                "anchor_a_pid": int(ta),
                "anchor_b_pid": int(tb),
                "anchor_a_decode": rec["top_tokens"][ta],
                "anchor_b_decode": rec["top_tokens"][tb],
                "anchor_a_members": int(rec["member_counts"][ta]),
                "anchor_b_members": int(rec["member_counts"][tb]),
                "path_pids": path,
                "hops": hops,
                "path_decodes": [rec["top_tokens"][p] for p in (path or [])],
            })
        pair_results.append({
            "seed_anchor_a_pid": int(sa),
            "seed_anchor_b_pid": int(sb),
            "seed_anchor_a_decode": seed["top_tokens"][sa],
            "seed_anchor_b_decode": seed["top_tokens"][sb],
            "seed_anchor_a_members": int(seed["member_counts"][sa]),
            "seed_anchor_b_members": int(seed["member_counts"][sb]),
            "per_resolution": per_resolution,
        })

    # Save per-dict summaries
    for rec in records:
        out_dir = args.output_root / rec["tag"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "top_tokens.json", "w") as f:
            json.dump([
                {"partition_id": i, "exemplar_top_tokens": rec["top_tokens"][i]}
                for i in range(rec["K"])
            ], f, indent=2)
        np.savez(out_dir / "geom.npz", geom=rec["G"])

    out = {
        "model": args.model_short,
        "layer": args.layer,
        "basis": args.basis,
        "dict_tags": [r["tag"] for r in records],
        "dict_K": {r["tag"]: r["K"] for r in records},
        "seed_tag": seed["tag"],
        "pairs": pair_results,
    }
    with open(args.output_root / "resolution_paths.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s/resolution_paths.json", args.output_root)


if __name__ == "__main__":
    main()
