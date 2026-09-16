"""Save the prompt split used by the paper's region-ablation experiment.

The input JSON contains 1,176 harmful prompts followed by 1,176 benign prompts.
The prompt corpus is supplied by the user and is not distributed here.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def held_out_indices(n_per_side: int, n_held_per_side: int, seed: int):
    rng = np.random.default_rng(seed)
    harmful = np.sort(rng.choice(np.arange(n_per_side),
                                 n_held_per_side, replace=False))
    benign = np.sort(rng.choice(np.arange(n_per_side, 2 * n_per_side),
                                n_held_per_side, replace=False))
    return harmful.tolist(), benign.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-per-side", type=int, default=1176)
    parser.add_argument("--n-held-per-side", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompts = json.loads(args.prompts_json.read_text())
    if len(prompts) != 2 * args.n_per_side:
        raise ValueError("expected n_per_side harmful prompts followed by n_per_side benign prompts")
    harmful, benign = held_out_indices(args.n_per_side, args.n_held_per_side,
                                        args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompts.json").write_text(json.dumps(prompts))
    (args.output_dir / "meta.json").write_text(json.dumps({
        "n_per_side": args.n_per_side,
        "n_held_per_side": args.n_held_per_side,
        "seed": args.seed,
        "held_harmful_idx": harmful,
        "held_benign_idx": benign,
    }, indent=2))


if __name__ == "__main__":
    main()
