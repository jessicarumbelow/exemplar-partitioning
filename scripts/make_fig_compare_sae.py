"""Build the SAE-correspondence figure.

Two-panel layout: mean F1 of (EP→SAE) and (SAE→EP) per percentile, plus
fraction of EP partitions with strong (F1 > 0.5) match. Points labelled
by basis (mean / exemplar). All on one model+layer (gemma-2-2b L12,
mt10M, vs gemma-scope-2b-pt-res-canonical 16k SAE).

Run:
    uv run python -m scripts.make_fig_compare_sae
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="/tmp/cs",
        help="Directory containing pX.json files (one per percentile).")
    parser.add_argument(
        "--percentiles", default="p1,p2,p4,p8,p10,p16,p32",
        help="Comma-separated percentile labels to plot.")
    parser.add_argument("--out-pdf", default="paper/figures/compare_sae.pdf")
    parser.add_argument("--out-png", default="paper/figures/compare_sae.png")
    args = parser.parse_args()

    labels = args.percentiles.split(",")
    rows = []
    for lbl in labels:
        path = Path(args.input_dir) / f"{lbl}.json"
        d = _load(path)
        pct_key = list(d["results_by_percentile"].keys())[0]
        for basis in ["mean", "exemplar"]:
            r = d["results_by_percentile"][pct_key].get(basis)
            if r is None:
                continue
            s2e = [x["f1"] for x in r["sae_to_ep"]]
            e2s = [x["f1"] for x in r["ep_to_sae"]]
            rows.append({
                "label": lbl,
                "pct":   float(lbl.lstrip("p").replace("p", ".")),
                "basis": basis,
                "K":     r["n_partitions"],
                "s2e_mean":  statistics.mean(s2e) if s2e else 0,
                "e2s_mean":  statistics.mean(e2s) if e2s else 0,
                "frac_strong": sum(1 for x in e2s if x > 0.5) / max(1, len(e2s)),
                "frac_very":   sum(1 for x in e2s if x > 0.8) / max(1, len(e2s)),
                "max_f1":      max(e2s) if e2s else 0,
            })

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.4), sharex=True)

    palette = {"mean": "#1f77b4", "exemplar": "#d62728"}
    markers = {"mean": "o", "exemplar": "s"}

    for basis in ["mean", "exemplar"]:
        sub = [r for r in rows if r["basis"] == basis]
        sub.sort(key=lambda r: r["pct"])
        x = [r["pct"] for r in sub]
        axL.plot(x, [r["e2s_mean"] for r in sub], "-",
                 marker=markers[basis], color=palette[basis],
                 label=f"EP→SAE mean F1 ({basis})", lw=2.0, ms=5.5)
        axL.plot(x, [r["s2e_mean"] for r in sub], "--",
                 marker=markers[basis], color=palette[basis], alpha=0.55,
                 label=f"SAE→EP mean F1 ({basis})", lw=1.5, ms=4.5)
        axR.plot(x, [r["frac_strong"] for r in sub], "-",
                 marker=markers[basis], color=palette[basis],
                 label=f"frac F1 > 0.5 ({basis})", lw=2.0, ms=5.5)

    axL.set_xlabel(r"Dictionary percentile $p$")
    axL.set_ylabel("mean F1 (token-overlap, top-100)")
    axL.set_title("EP–SAE feature correspondence (mean F1)")
    axL.set_xscale("log")
    axL.set_ylim(0, 0.45)
    axL.grid(True, alpha=0.25, linestyle=":")
    axL.legend(loc="lower right", framealpha=0.9, fontsize=7)

    axR.set_xlabel(r"Dictionary percentile $p$")
    axR.set_ylabel(r"fraction of EP partitions with $\mathrm{max\;F1} > 0.5$")
    axR.set_title("Strong-match fraction per percentile")
    axR.set_xscale("log")
    axR.set_ylim(0, 0.25)
    axR.grid(True, alpha=0.25, linestyle=":")
    axR.legend(loc="upper right", framealpha=0.9, fontsize=7)

    fig.suptitle(
        "EP partition correspondence to GemmaScope canonical 16k SAE "
        "(gemma-2-2b L12)", fontsize=11, y=1.00,
    )
    fig.tight_layout()

    out_pdf = Path(args.out_pdf); out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = Path(args.out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
