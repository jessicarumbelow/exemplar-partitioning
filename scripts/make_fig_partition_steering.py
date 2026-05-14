"""Build the partition-steering α-curve figure for the paper.

Loads the three resolution runs (p4 / p8 / p10) of
`exp_partition_steering`, plots Cyrillic-fraction vs α for EP at each
resolution against the (resolution-independent) DiffMean baseline and a
random-partition control. Saves to `paper/figures/partition_steering_cyrillic.pdf`
(plus a PNG copy for quick inspection).

Run locally:
    uv run python -m scripts.make_fig_partition_steering
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _series(json_path: Path) -> tuple[list[float], dict]:
    """Return (alphas, results-dict-for-cyrillic) from a steering JSON."""
    d = json.loads(json_path.read_text())
    alphas = sorted(float(a) for a in d["results"]["cyrillic"]["per_alpha"])
    return alphas, d["results"]["cyrillic"]


def _curve(per_alpha: dict, alphas: list[float], key: str) -> list[float]:
    return [per_alpha[str(a)].get(key, float("nan")) for a in alphas]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p4",  default="results/exp_partition_steering/p4_seed2.json",
        help="Path to p4 partition_steering.json (seed 2 — has DiffMean)")
    parser.add_argument(
        "--p4-clean-random",
        default="results/exp_partition_steering/p4_seed1.json",
        help="Optional separate p4 run whose random control is uncontaminated. "
             "Used only for the random-control curve. Set to same as --p4 to skip.")
    parser.add_argument(
        "--p8",  default="results/exp_partition_steering/p8_seed2.json")
    parser.add_argument(
        "--p10", default="results/exp_partition_steering/p10_seed2.json")
    parser.add_argument("--out-pdf", default="paper/figures/partition_steering_cyrillic.pdf")
    parser.add_argument("--out-png", default="paper/figures/partition_steering_cyrillic.png")
    args = parser.parse_args()

    alphas_p4,  cyr_p4  = _series(Path(args.p4))
    alphas_p8,  cyr_p8  = _series(Path(args.p8))
    alphas_p10, cyr_p10 = _series(Path(args.p10))

    # All three runs share the same DiffMean direction by construction
    # (DiffMean uses labelled seeds, not the dictionary). Take p4's curve.
    pa_p4  = cyr_p4["per_alpha"]
    pa_p8  = cyr_p8["per_alpha"]
    pa_p10 = cyr_p10["per_alpha"]

    ep_p4   = _curve(pa_p4,  alphas_p4,  "ep_score_mean")
    ep_p8   = _curve(pa_p8,  alphas_p8,  "ep_score_mean")
    ep_p10  = _curve(pa_p10, alphas_p10, "ep_score_mean")
    dm_p4   = _curve(pa_p4,  alphas_p4,  "dm_score_mean")

    # Use a separate seed for the random control if the main seed's random
    # samples were contaminated (e.g. landed on another Cyrillic partition).
    # If the path matches the main p4 path, we just use p4's own random.
    if Path(args.p4_clean_random).resolve() == Path(args.p4).resolve():
        rand_p4 = _curve(pa_p4, alphas_p4, "rand_score_mean")
        rand_alphas = alphas_p4
    else:
        rand_alphas, rand_cyr = _series(Path(args.p4_clean_random))
        rand_p4 = _curve(rand_cyr["per_alpha"], rand_alphas,
                         "rand_score_mean")

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    # EP at three resolutions
    ax.plot(alphas_p4,  ep_p4,  "o-",  color="#1f77b4",
            label=r"EP ($p_{4}$, 738 regions)", lw=2.0, ms=5)
    ax.plot(alphas_p8,  ep_p8,  "s--", color="#1f77b4", alpha=0.6,
            label=r"EP ($p_{8}$, 252 regions)", lw=1.6, ms=4.5)
    ax.plot(alphas_p10, ep_p10, "^:",  color="#1f77b4", alpha=0.4,
            label=r"EP ($p_{10}$, 176 regions)", lw=1.4, ms=4.5)

    # DiffMean (supervised, resolution-independent)
    ax.plot(alphas_p4, dm_p4, "D-", color="#d62728",
            label="DiffMean (supervised)", lw=2.0, ms=5)

    # Random control
    ax.plot(rand_alphas, rand_p4, "x-", color="#7f7f7f", alpha=0.7,
            label="Random region", lw=1.4, ms=5)

    ax.set_xlabel(r"Steering scale $\alpha$")
    ax.set_ylabel("Fraction of output letters that are Cyrillic")
    ax.set_title(
        r"Cyrillic-steering causal sweep, Gemma-2-2B-it L20"
    )
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
