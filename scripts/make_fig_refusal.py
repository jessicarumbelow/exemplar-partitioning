"""Build the refusal-ablation Δ-vs-percentile figure.

Two lines: exemplar-basis ablation Δ and mean-basis ablation Δ, both at
K=1 (single highest-loading region's basis direction projected out).
The figure shows: a broad sweet spot in p where exemplar dominates;
the p=8 fragmentation accident where ablation collapses; the p=20
contamination where the single refusal region is too coarse to act on.

Run:
    uv run python -m scripts.make_fig_refusal
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCES = {
    5:  ("sweep_p5.json",        "sweep_by_basis"),
    6:  ("sweep_p6.json",        "sweep_by_basis"),
    8:  ("behavioral.json",      None),  # K-sweep, no per-basis split
    10: ("behavioral_p10b.json", "sweep_by_basis"),
    12: ("sweep_p12.json",       "sweep_by_basis"),
    15: ("sweep_p15.json",       "sweep_by_basis"),
    20: ("sweep_p20.json",       "sweep_by_basis"),
}

# At p=8 the refusal cluster fragments across 3 regions; K=1 ablation collapses.
# Numbers from paper tab:ablation; the JSON only has flat structure here.
P8_FALLBACK = {"exemplar": 0.00, "mean": -0.04}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="/tmp/ep_results")
    parser.add_argument(
        "--out-pdf",
        default="paper/figures/refusal_ablation.pdf")
    parser.add_argument(
        "--out-png",
        default="paper/figures/refusal_ablation.png")
    args = parser.parse_args()

    pcts = sorted(SOURCES.keys())
    K_per_p: dict[int, int] = {}
    delta_exemplar: dict[int, float] = {}
    delta_mean: dict[int, float] = {}

    for p, (filename, mode) in SOURCES.items():
        d = json.loads((Path(args.in_dir) / filename).read_text())
        K_per_p[p] = d["n_partitions"]
        if mode == "sweep_by_basis":
            sbb = d["ablation"]["sweep_by_basis"]
            entry_mean = next(e for e in sbb["mean"] if e["k"] == 1)
            entry_exem = next(e for e in sbb["exemplar"] if e["k"] == 1)
            delta_mean[p] = entry_mean["delta"]
            delta_exemplar[p] = entry_exem["delta"]
        else:
            delta_exemplar[p] = P8_FALLBACK["exemplar"]
            delta_mean[p] = P8_FALLBACK["mean"]

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 11, "axes.titlesize": 11,
        "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    xs = pcts
    y_exem = [delta_exemplar[p] for p in xs]
    y_mean = [delta_mean[p] for p in xs]

    ax.plot(xs, y_exem, "o-", color="#d62728",
            label="Exemplar basis", lw=2.0, ms=7)
    ax.plot(xs, y_mean, "s--", color="#1f77b4",
            label="Mean-member basis", lw=1.8, ms=6)

    # Annotate the failure modes.
    ax.annotate("p=8: refusal cluster\nfragments across 3 regions",
                xy=(8, delta_exemplar[8]), xytext=(8, -0.35),
                fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#666"))
    ax.annotate("p=20: single region\nis 0.52 refusal\n(contaminated)",
                xy=(20, delta_exemplar[20]), xytext=(18, -0.40),
                fontsize=8.5, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#666"))

    # Reference line at zero (no effect).
    ax.axhline(0.0, color="#888", lw=0.8, ls=":", zorder=1)

    # Annotate the dictionary size at each x for context.
    for p in xs:
        ax.text(p, 0.04, f"K={K_per_p[p]}", ha="center", va="bottom",
                fontsize=7.5, color="#555")

    ax.set_xlabel(r"Calibration percentile $p$")
    ax.set_ylabel(r"$\Delta$ refusal rate (held-out, $K=1$ ablated)")
    ax.set_title(
        "Refusal ablation across resolution, Gemma-2-2B-it L20\n"
        "(baseline refusal $0.98$; lower $\\Delta$ = stronger ablation)")
    ax.set_ylim(-0.85, 0.15)
    ax.set_xlim(3.5, 22)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="lower left", framealpha=0.95)

    fig.tight_layout()

    out_pdf = Path(args.out_pdf); out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = Path(args.out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
