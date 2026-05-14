"""Build the per-layer per-domain saturation figure.

Layout: one row of three panels (one per layer), each plotting region
count vs activations seen for math / code / chat. The point is that under
a single calibration, different domains saturate at different region counts
— a model-and-data-distribution-dependent quantity not accessible to a
fixed-size SAE.

Run:
    uv run python -m scripts.make_fig_saturation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="/tmp/sat_p8.json",
        help="Path to saturation_curves.json")
    parser.add_argument("--out-pdf", default="paper/figures/saturation.pdf")
    parser.add_argument("--out-png", default="paper/figures/saturation.png")
    args = parser.parse_args()

    sat = json.loads(Path(args.input).read_text())
    curves = sat["curves_by_layer"]
    layers = sorted(int(k) for k in curves.keys())

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(1, len(layers), figsize=(11.0, 3.2),
                             sharey=False)
    if len(layers) == 1:
        axes = [axes]

    palette = {"math": "#1f77b4", "code": "#2ca02c", "chat": "#d62728"}
    markers = {"math": "o", "code": "s", "chat": "^"}

    for ax, layer in zip(axes, layers):
        by_dom = curves[str(layer)]
        for dom in ("math", "code", "chat"):
            pts = by_dom.get(dom)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "-", marker=markers[dom], color=palette[dom],
                    label=dom, lw=1.7, ms=4.5)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Activations seen")
        if ax is axes[0]:
            ax.set_ylabel("Regions")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle(
        r"Per-layer, per-domain saturation under a single calibration "
        r"(Gemma-2-2B, $p_{8}$)", fontsize=11, y=1.02,
    )
    fig.tight_layout()

    out_pdf = Path(args.out_pdf); out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = Path(args.out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
