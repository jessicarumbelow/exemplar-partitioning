"""Occupancy-difference selector for the taboo experiment.

The SAE selector that worked compares the SAME tokens under both models and
asks what fires more. This is the EP-native analogue: assign the stock model's
activations (identical tokens) into the TABOO dictionary and rank regions by
how many taboo activations they hold minus how many stock activations land
there. Regions that exist for the fine-tuned model's processing of the text
but not the base model's are the fine-tune's new behaviour.

Reads acts.npz + dict_taboo.pkl written by exp_taboo.py. The logit lens uses
the tied unembedding + final norm read from the cached safetensors shards;
all regions are lensed in one GPU matmul.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch


def load_lens(model_name, device):
    """(embed (V, D), final norm weight (D,)) as float32 torch tensors."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    idx = json.load(open(hf_hub_download(model_name, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    out = {}
    for name in ("model.embed_tokens.weight", "model.norm.weight"):
        shard = hf_hub_download(model_name, wm[name])
        with safe_open(shard, framework="pt") as f:
            out[name] = f.get_tensor(name).float().to(device)
    return out["model.embed_tokens.weight"], out["model.norm.weight"]


def lens_all(dirs, embed, norm_w, secret_ids, device, top_k=15):
    """Secret rank + top tokens for every row of dirs (K, D), one matmul."""
    d = torch.tensor(dirs, dtype=torch.float32, device=device)
    d = d / (d.norm(dim=1, keepdim=True) + 1e-8)
    rms = d.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8
    logits = ((d / rms) * (1.0 + norm_w)) @ embed.T          # (K, V)
    order = torch.argsort(logits, dim=1, descending=True)
    rank_of = torch.empty_like(order)
    k = order.shape[0]
    rank_of.scatter_(1, order, torch.arange(order.shape[1], device=device)
                     .expand(k, -1))
    secret = torch.tensor(secret_ids, device=device)
    ranks = rank_of[:, secret].min(dim=1).values.cpu().numpy()
    return ranks, order[:, :top_k].cpu().numpy()


def secret_token_ids(tok, secret):
    variants = {secret, secret.capitalize(), secret.upper(),
                " " + secret, " " + secret.capitalize(), " " + secret.upper()}
    out = set()
    for v in variants:
        ids = tok(v, add_special_tokens=False)["input_ids"]
        if ids:
            out.add(ids[0])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--top-regions", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run = args.run_dir

    from transformers import AutoTokenizer

    from ep.discovery.geometry import centered_unit

    tok = AutoTokenizer.from_pretrained(args.model)
    embed, norm_w = load_lens(args.model, args.device)

    r = json.load(open(run / "results.json"))
    secret = r["secret"]
    secret_ids = secret_token_ids(tok, secret)
    z = np.load(run / "acts.npz", allow_pickle=True)
    with open(run / "dict_taboo.pkl", "rb") as f:
        d_taboo = pickle.load(f)
    prov = [s.split("|", 2) for s in z["prov"]]

    ids_taboo = z["ids_taboo"]
    stock_in_taboo, _ = d_taboo.assign(z["stock"])
    k = len(d_taboo.partitions)
    c_taboo = np.bincount(ids_taboo, minlength=k)
    c_stock = np.bincount(stock_in_taboo, minlength=k)
    diff = c_taboo.astype(int) - c_stock.astype(int)

    units = centered_unit(z["taboo"], d_taboo.center)
    dirs = np.zeros((k, units.shape[1]), np.float32)
    for reg in range(k):
        m = units[ids_taboo == reg]
        if len(m):
            v = m.mean(0)
            dirs[reg] = v / (np.linalg.norm(v) + 1e-8)

    ranks, top_ids = lens_all(dirs, embed, norm_w, secret_ids, args.device)
    empty = c_taboo == 0
    ranks[empty] = embed.shape[0]  # empty regions can't carry the secret

    print(f"=== {run.name} (secret={secret}) K={k} ===")
    order = np.argsort(-diff)
    report = []
    for reg in order[:args.top_regions]:
        toks = [prov[i][2] for i in np.nonzero(ids_taboo == reg)[0][:10]]
        report.append({
            "region": int(reg), "n_taboo": int(c_taboo[reg]),
            "n_stock": int(c_stock[reg]), "occupancy_diff": int(diff[reg]),
            "secret_rank": int(ranks[reg]),
            "top_tokens": [tok.decode([t]) for t in top_ids[reg]],
            "member_tokens": toks,
        })
        print(f"region {reg:5d} taboo={c_taboo[reg]:5d} stock={c_stock[reg]:5d} "
              f"secret_rank={ranks[reg]:6d} members={toks[:6]} "
              f"top={report[-1]['top_tokens'][:6]}")

    best = int(np.argmin(ranks))
    pos = int(np.where(order == best)[0][0])
    print(f"best-secret region {best}: lens rank {int(ranks[best])}, "
          f"occupancy position {pos + 1}/{k}, "
          f"diff={diff[best]} (taboo={c_taboo[best]}, stock={c_stock[best]})")

    (run / "occupancy.json").write_text(json.dumps({
        "secret": secret, "top_by_occupancy": report,
        "best_secret_region": best,
        "best_secret_rank": int(ranks[best]),
        "best_secret_occupancy_position": pos + 1,
        "all_region_occupancy_diff": diff.tolist(),
        "all_region_secret_ranks": ranks.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()

