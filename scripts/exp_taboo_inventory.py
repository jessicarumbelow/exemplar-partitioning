"""Read every fine-tune-added Taboo control-token region.

This is an inventory, not a secret selector. Regions are built on the
fine-tuned activations exactly as in ``exp_taboo_control``; a region is listed
when the sampled base activations never enter it.  Rows are displayed by
fine-tuned support only for readability. ``--evaluate-secret`` adds ranks
after construction; those ranks must not be used as a discovery selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from ep.discovery.calibration import calibrate
from ep.discovery.geometry import centered_unit


def region_directions(acts: np.ndarray, ids: np.ndarray, center: np.ndarray):
    units = centered_unit(acts, center)
    directions = []
    for region in range(int(ids.max()) + 1):
        members = units[ids == region]
        mean = members.mean(0)
        directions.append(mean / (np.linalg.norm(mean) + 1e-8))
    return np.asarray(directions, dtype=np.float32)


def reconstructed_center(acts: np.ndarray, percentile: float, seed: int):
    order = np.random.default_rng(seed).permutation(len(acts))
    shuffled = acts[order]
    result = calibrate(
        (shuffled[i:i + 256] for i in range(0, len(shuffled), 256)),
        n_tokens=len(acts), percentile=percentile,
    )
    return result.center


def embedding_shard(index: dict):
    weight_map = index["weight_map"]
    for key in ("model.embed_tokens.weight", "embed_tokens.weight"):
        if key in weight_map:
            return key, weight_map[key]
    raise KeyError("model embedding weight not present in safetensors index")


def load_embedding(model_name: str, device: str):
    index_path = hf_hub_download(model_name, "model.safetensors.index.json")
    key, shard = embedding_shard(json.loads(Path(index_path).read_text()))
    shard_path = hf_hub_download(model_name, shard)
    with safe_open(shard_path, framework="pt", device="cpu") as archive:
        weight = archive.get_tensor(key)
    weight = weight.to(device=device, dtype=torch.float32)
    weight.div_(weight.norm(dim=1, keepdim=True) + 1e-8)
    return weight


def top_tokens(weight: torch.Tensor, tokenizer, direction: np.ndarray,
               n_tokens: int):
    vector = torch.as_tensor(direction, dtype=torch.float32, device=weight.device)
    vector = vector / (vector.norm() + 1e-8)
    indices = torch.topk(weight @ vector, n_tokens).indices.tolist()
    return [tokenizer.decode([token]) for token in indices]


def secret_token_ids(tokenizer, secret: str):
    variants = {secret, secret.capitalize(), secret.upper(),
                " " + secret, " " + secret.capitalize(), " " + secret.upper()}
    return sorted({
        token_ids[0] for variant in variants
        if (token_ids := tokenizer(variant, add_special_tokens=False)["input_ids"])
    })


def secret_rank(weight: torch.Tensor, direction: np.ndarray, token_ids: list[int]):
    vector = torch.as_tensor(direction, dtype=torch.float32, device=weight.device)
    vector = vector / (vector.norm() + 1e-8)
    scores = weight @ vector
    return int((scores > scores[token_ids].max()).sum().item())


def inventory(run_dir: Path, weight: torch.Tensor, tokenizer, n_tokens: int,
              evaluate_secret: bool):
    result = json.loads((run_dir / "results.json").read_text())
    saved = np.load(run_dir / "acts_control.npz")
    acts, ids_taboo, ids_stock = saved["taboo"], saved["ids_taboo"], saved["ids_stock"]
    center = reconstructed_center(acts, result["percentile"], result["seed"])
    directions = region_directions(acts, ids_taboo, center)
    k = len(directions)
    n_taboo = np.bincount(ids_taboo, minlength=k)
    n_stock = np.bincount(ids_stock, minlength=k)
    token_ids = secret_token_ids(tokenizer, result["secret"]) if evaluate_secret else []
    rows = []
    for region in np.flatnonzero((n_taboo > 0) & (n_stock == 0)):
        row = {
            "region": int(region),
            "taboo_members": int(n_taboo[region]),
            "base_members": 0,
            "top_embedding_tokens": top_tokens(
                weight, tokenizer, directions[region], n_tokens),
        }
        if evaluate_secret:
            row["secret_embedding_rank"] = secret_rank(
                weight, directions[region], token_ids)
        rows.append(row)
    rows.sort(key=lambda row: -row["taboo_members"])
    return {
        "secret": result["secret"],
        "layer": result["layer"],
        "percentile": result["percentile"],
        "n_control_taboo": result["n_control_taboo"],
        "n_control_stock": result["n_control_stock"],
        "dictionary_size": result["K"],
        "added_regions": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", required=True,
                        help="Comma-separated control-run directories.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-tokens", type=int, default=10)
    parser.add_argument("--evaluate-secret", action="store_true",
                        help="Record secret rank for evaluation only; never selects rows.")
    args = parser.parse_args()

    weight = load_embedding(args.model, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    reports = [inventory(Path(path), weight, tokenizer, args.top_tokens,
                         args.evaluate_secret)
               for path in args.run_dirs.split(",")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
