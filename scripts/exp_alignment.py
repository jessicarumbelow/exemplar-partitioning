"""Geometric-behavioural alignment of an EP dictionary.

For each partition in a built dictionary, compute (i) the geometric position
of its exemplar on the unit sphere and (ii) a behavioural signature obtained
by projecting the exemplar through ln_final · W_U (logit lens) into a
distribution over the vocabulary.

Headline metric: Spearman rank correlation between the pairwise geometric
distance matrix (cosine on exemplars) and the pairwise behavioural distance
matrix (Jensen-Shannon divergence on softmaxed logits), with a permutation
null. Property test of the dictionary: does its geometry track its
behaviour?

Builds on Li et al. 2025 ("The Geometry of Concepts"), which reports
qualitative semantic-geometric structure in SAE feature dictionaries.

Run locally:
    uv run python -m scripts.exp_alignment --dict-path path/to/library.pkl
Run on Modal:
    modal run ep_modal_experiments.py::alignment
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


def _logit_lens(
    exemplars: np.ndarray,           # (K, D), centered unit directions
    model,
    batch_size: int = 64,
) -> torch.Tensor:
    """Project each exemplar back to vocab logits via ln_final · W_U.

    Matches the convention in scripts/build_partitions.py::_top_vocab_tokens —
    feed the centered unit direction directly through ln_final and W_U,
    without re-adding the activation center. ln_final normalises out
    magnitude so the result is dominated by the direction. Returns a
    (K, V) tensor of pre-softmax logits in fp32 on CPU (memory-friendly).
    """
    device = model.W_U.device
    dtype = model.W_U.dtype
    K = exemplars.shape[0]
    out = []
    with torch.no_grad():
        for i in range(0, K, batch_size):
            batch = torch.tensor(
                exemplars[i:i + batch_size], dtype=dtype, device=device,
            )  # (b, D)
            normed = model.ln_final(batch)
            logits = normed @ model.W_U  # (b, V)
            out.append(logits.float().cpu())
    return torch.cat(out, dim=0)         # (K, V)


def _js_divergence_matrix(probs: torch.Tensor) -> np.ndarray:
    """Pairwise Jensen-Shannon divergence on (K, V) probability matrix.

    JS(p, q) = 0.5 * (KL(p || m) + KL(q || m)) where m = 0.5 (p + q).
    Symmetric, bounded in [0, log 2]. Returns (K, K) ndarray, fp32.
    """
    K = probs.shape[0]
    log_p = torch.log(probs + 1e-30)             # (K, V)
    out = np.zeros((K, K), dtype=np.float32)
    # Pairwise: for each i, compute KL(p_i || m) + KL(p_j || m) for all j>i.
    # Done row-wise to keep memory at O(V) per row instead of O(KV).
    for i in range(K):
        p = probs[i]                             # (V,)
        # m_ij = 0.5(p_i + p_j); shape (K, V) for all j
        m = 0.5 * (probs + p)
        log_m = torch.log(m + 1e-30)
        # KL(p || m) and KL(q || m) along V
        kl_p = (p * (log_p[i] - log_m)).sum(dim=1)         # (K,)
        kl_q = (probs * (log_p - log_m)).sum(dim=1)        # (K,)
        js = 0.5 * (kl_p + kl_q)
        out[i] = js.numpy()
    out = 0.5 * (out + out.T)            # symmetrize residual numerical error
    np.fill_diagonal(out, 0.0)
    return out


def _spearman_upper(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ on the upper triangles of two square matrices."""
    K = a.shape[0]
    iu, ju = np.triu_indices(K, k=1)
    av = a[iu, ju]
    bv = b[iu, ju]
    # rank both
    ar = np.argsort(np.argsort(av))
    br = np.argsort(np.argsort(bv))
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = np.sqrt((ar * ar).sum() * (br * br).sum())
    if denom == 0:
        return 0.0
    return float((ar * br).sum() / denom)


def _permutation_null(
    geom: np.ndarray, beh: np.ndarray, n_perm: int = 1000, seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Permute partition labels and recompute Spearman to build the null."""
    rng = np.random.default_rng(seed)
    K = geom.shape[0]
    null = np.zeros(n_perm, dtype=np.float32)
    for i in range(n_perm):
        perm = rng.permutation(K)
        null[i] = _spearman_upper(geom, beh[perm][:, perm])
    rho = _spearman_upper(geom, beh)
    p = float((np.abs(null) >= abs(rho)).mean())
    return rho, null, p           # type: ignore[return-value]


def _top_tokens_str(logits_row: torch.Tensor, tokenizer, k: int = 8) -> str:
    top_ids = logits_row.topk(k).indices.tolist()
    toks = [tokenizer.decode([t]).strip() for t in top_ids]
    return ", ".join(t for t in toks if t)[:200]


def _pick_anchors(dictionary, n_anchors: int) -> list[int]:
    """Pick anchor partition indices: the n_anchors largest by member count."""
    sizes = [(i, p.member_count) for i, p in enumerate(dictionary.partitions)]
    sizes.sort(key=lambda x: -x[1])
    return [i for i, _ in sizes[:n_anchors]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--model-short", default="gemma-2-2b")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--dict-path", required=True,
                        help="Path to library.pkl on disk or /vol")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-anchors", type=int, default=5)
    parser.add_argument("--n-neighbours", type=int, default=3)
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is None:
        args.output_dir = Path("results/exp_alignment") / (
            f"{args.model_short}_L{args.layer}_seed{args.seed}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dictionary %s", args.dict_path)
    with open(args.dict_path, "rb") as f:
        dictionary = pickle.load(f)
    logger.info("Dictionary: %s", dictionary)
    K = len(dictionary.partitions)
    if K < 4:
        raise SystemExit(f"K={K}: too few partitions for a meaningful alignment test.")

    import transformer_lens as tl
    logger.info("Loading model %s on %s", args.model, args.device)
    t0 = time.time()
    model = tl.HookedTransformer.from_pretrained_no_processing(
        args.model, device=args.device, dtype=torch.bfloat16,
    )
    model.eval()
    logger.info("Model loaded in %.1fs", time.time() - t0)

    logger.info("Logit-lensing %d exemplars", K)
    exemplars = np.stack([p.exemplar_direction for p in dictionary.partitions])  # (K, D)
    logits = _logit_lens(exemplars, model)                                        # (K, V)
    probs = torch.softmax(logits, dim=-1)

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"alignment_{args.model_short}_L{args.layer}_K{K}",
            config=vars(args), job_type="alignment",
        )

    logger.info("Computing geometric distance matrix (cosine on exemplars)")
    sim = exemplars @ exemplars.T
    np.clip(sim, -1.0, 1.0, out=sim)
    geom = (1.0 - sim).astype(np.float32)
    np.fill_diagonal(geom, 0.0)

    logger.info("Computing behavioural distance matrix (JS divergence)")
    t0 = time.time()
    beh = _js_divergence_matrix(probs)
    logger.info("JS matrix in %.1fs", time.time() - t0)

    logger.info("Spearman rank correlation + permutation null (n=%d)",
                args.n_permutations)
    rho, null, p = _permutation_null(
        geom, beh, n_perm=args.n_permutations, seed=args.seed,
    )
    logger.info("Spearman ρ = %.4f, p = %.4f, null mean = %.4f, null std = %.4f",
                rho, p, null.mean(), null.std())

    anchors = _pick_anchors(dictionary, args.n_anchors)
    anchor_panel = []
    for a in anchors:
        # Top-N geometric neighbours (excluding self): smallest geom[a, j], j != a
        g_order = np.argsort(geom[a])
        g_neighbours = [j for j in g_order if j != a][:args.n_neighbours]
        # Top-N behavioural neighbours
        b_order = np.argsort(beh[a])
        b_neighbours = [j for j in b_order if j != a][:args.n_neighbours]
        anchor_panel.append({
            "partition_id": a,
            "member_count": int(dictionary.partitions[a].member_count),
            "top_tokens": _top_tokens_str(logits[a], model.tokenizer),
            "geometric_neighbours": [
                {"partition_id": int(j),
                 "geom_dist": float(geom[a, j]),
                 "beh_dist": float(beh[a, j]),
                 "top_tokens": _top_tokens_str(logits[j], model.tokenizer)}
                for j in g_neighbours
            ],
            "behavioural_neighbours": [
                {"partition_id": int(j),
                 "geom_dist": float(geom[a, j]),
                 "beh_dist": float(beh[a, j]),
                 "top_tokens": _top_tokens_str(logits[j], model.tokenizer)}
                for j in b_neighbours
            ],
        })

    out = {
        "model": args.model_short,
        "layer": args.layer,
        "K": K,
        "spearman_rho": rho,
        "p_value": p,
        "n_permutations": args.n_permutations,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "anchor_panel": anchor_panel,
    }
    out_path = args.output_dir / "alignment.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_path)

    np.savez(
        args.output_dir / "matrices.npz",
        geom=geom, beh=beh, null=null,
    )

    top_tokens = [
        {"partition_id": i,
         "member_count": int(dictionary.partitions[i].member_count),
         "exemplar_top_tokens": _top_tokens_str(logits[i], model.tokenizer)}
        for i in range(K)
    ]
    with open(args.output_dir / "top_tokens.json", "w") as f:
        json.dump(top_tokens, f, indent=2)

    if args.wandb:
        import wandb
        wandb.log({
            "spearman_rho": rho, "p_value": p,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "K": K,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
