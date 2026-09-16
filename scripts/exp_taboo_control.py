"""Build Taboo dictionaries from assistant control-token activations.

The surface text at these positions is identical across transcripts. The base
model reads the same transcripts and provides an assignment control.

The script:
  1. Reuse the hint-game transcripts from an exp_taboo run.
  2. Extract per-position activations at --layer (paper uses 32) for the taboo
     and stock models; keep only assistant control positions.
  3. Build an EP dictionary on the fine-tuned model's control activations and
     assign the base model's activations into it.
  4. Save all assignments and counts. The top occupancy regions are also read
     out for the paper's pilot comparison and auditor experiment.

``exp_taboo_inventory`` lists every region with fine-tuned support and zero
base support for the paper's main 21-organism analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.exp_taboo import (
    build_dictionary,
    extract,
    load_merged_hf,
    load_stock,
    logit_lens_rank,
    region_mean_dirs,
    secret_token_ids,
    wrap_tl,
)


def control_indices(prov):
    """Indices of assistant control tokens: '<start_of_turn>' immediately
    followed by 'model', and that 'model' token itself."""
    keep = []
    for i, (t, p, s) in enumerate(prov):
        if s == "model" and i > 0 and prov[i - 1][2] == "<start_of_turn>" \
                and prov[i - 1][0] == t and prov[i - 1][1] == p - 1:
            keep.extend([i - 1, i])
    return np.array(sorted(set(keep)), dtype=int)


def embedding_rank(w_e, tokenizer, vector, secret_ids, top_k=30):
    """Rank of the secret under cosine(vector, token embedding rows)."""
    v = torch.tensor(vector, dtype=torch.float32, device=w_e.device)
    v = v / (v.norm() + 1e-8)
    rows = w_e / (w_e.norm(dim=1, keepdim=True) + 1e-8)
    sims = rows @ v
    order = torch.argsort(sims, descending=True)
    rank_of = torch.empty_like(order)
    rank_of[order] = torch.arange(len(order), device=order.device)
    best = min(int(rank_of[t]) for t in secret_ids)
    top = [tokenizer.decode([int(t)]) for t in order[:top_k]]
    return best, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secret", required=True)
    ap.add_argument("--base-model", default="google/gemma-2-9b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--percentile", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-regions", type=int, default=10)
    ap.add_argument("--transcripts-json", required=True)
    ap.add_argument(
        "--exclude-secret-text",
        action="store_true",
        help="Drop transcripts containing the secret string, case-insensitively.",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = args.adapter or f"bcywinski/gemma-2-9b-it-taboo-{args.secret}"

    print(f"=== taboo-control secret={args.secret} layer={args.layer} "
          f"p={args.percentile} ===")
    texts = json.loads(Path(args.transcripts_json).read_text())
    n_transcripts_input = len(texts)
    if args.exclude_secret_text:
        texts = [t for t in texts if args.secret.lower() not in t.lower()]
    n_transcripts_excluded = n_transcripts_input - len(texts)
    print(f"{len(texts)} transcripts reused from {args.transcripts_json}"
          f" ({n_transcripts_excluded} excluded for secret text)")

    merged, tok = load_merged_hf(args.base_model, adapter, args.device)
    secret_ids = secret_token_ids(tok, args.secret)
    taboo = wrap_tl(args.base_model, merged, tok, args.device)
    del merged
    torch.cuda.empty_cache()

    acts_taboo_all, prov = extract(taboo, tok, texts, args.layer, args.device)
    ctl = control_indices(prov)
    acts_taboo = acts_taboo_all[ctl]
    print(f"{len(ctl)} assistant control positions of {len(prov)} total")
    w_e = taboo.W_E.detach().float()
    del acts_taboo_all

    stock = load_stock(args.base_model, args.device)
    acts_stock_all, prov_stock = extract(stock, tok, texts, args.layer,
                                         args.device)
    acts_stock = acts_stock_all[control_indices(prov_stock)]
    del stock, acts_stock_all
    torch.cuda.empty_cache()

    d, cal, ids_taboo, _ = build_dictionary(acts_taboo, args.percentile,
                                            args.seed)
    k = len(d.partitions)
    ids_stock, _ = d.assign(acts_stock)
    n_taboo = np.bincount(ids_taboo, minlength=k)
    n_stock = np.bincount(ids_stock, minlength=k)
    diff = n_taboo.astype(int) - n_stock.astype(int)
    print(f"dict: K={k}, theta={cal.threshold:.3f}, "
          f"N_taboo={len(acts_taboo)}, N_stock={len(acts_stock)}")

    dirs = region_mean_dirs(acts_taboo, ids_taboo, d.center, k)
    report = []
    for r in np.argsort(-diff)[:args.top_regions]:
        r = int(r)
        raw_mean = acts_taboo[ids_taboo == r].mean(0)
        ll_rank, ll_top = logit_lens_rank(taboo, dirs[r], secret_ids)
        ed_rank, ed_top = embedding_rank(w_e, tok, dirs[r], secret_ids)
        er_rank, er_top = embedding_rank(w_e, tok, raw_mean, secret_ids)
        report.append({
            "region": r, "n_taboo": int(n_taboo[r]), "n_stock": int(n_stock[r]),
            "occupancy_diff": int(diff[r]),
            "logit_lens": {"secret_rank": ll_rank, "top_tokens": ll_top},
            "embed_dir": {"secret_rank": ed_rank, "top_tokens": ed_top},
            "embed_rawmean": {"secret_rank": er_rank, "top_tokens": er_top},
        })
        print(f"region {r:3d} taboo={n_taboo[r]:4d} stock={n_stock[r]:4d} "
              f"ll_rank={ll_rank:6d} embdir_rank={ed_rank:6d} "
              f"embraw_rank={er_rank:6d}")
        print(f"   ll_top:     {ll_top[:10]}")
        print(f"   embdir_top: {ed_top[:10]}")
        print(f"   embraw_top: {er_top[:10]}")

    out = {
        "secret": args.secret, "adapter": adapter, "layer": args.layer,
        "percentile": args.percentile, "seed": args.seed,
        "n_transcripts": len(texts),
        "n_transcripts_input": n_transcripts_input,
        "n_transcripts_excluded_secret_text": n_transcripts_excluded,
        "n_control_taboo": int(len(acts_taboo)),
        "n_control_stock": int(len(acts_stock)),
        "K": k, "theta": float(cal.threshold),
        "top_by_occupancy": report,
        "all_occupancy_taboo": n_taboo.tolist(),
        "all_occupancy_stock": n_stock.tolist(),
    }
    (args.output_dir / "results.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(args.output_dir / "acts_control.npz",
                        taboo=acts_taboo, stock=acts_stock,
                        ids_taboo=ids_taboo, ids_stock=ids_stock,
                        prov=np.array(["|".join(map(str, prov[i]))
                                       for i in ctl]))
    print(f"wrote {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
