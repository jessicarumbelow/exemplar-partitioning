"""Build the held-out coverage figure for the paper.

Two-panel layout: within-threshold rate (left) and mean nearest-exemplar
distance (right) versus dictionary percentile, on three corpora — Pile
(in-distribution), Bulgarian Wikipedia (cross-language), random tokens
(no semantic structure). L4 p4 is overlaid as a separate marker on the
right panel for the layer-comparison.

Saves to paper/figures/coverage.{pdf,png}.

Run:
    uv run python -m scripts.make_fig_coverage
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
        "--l20",  default="results/exp_coverage/l20.json",
        help="L20 multi-percentile coverage JSON.")
    parser.add_argument(
        "--l4",   default="results/exp_coverage/l4.json",
        help="L4 single-percentile coverage JSON (for layer comparison).")
    parser.add_argument("--out-pdf", default="paper/figures/coverage.pdf")
    parser.add_argument("--out-png", default="paper/figures/coverage.png")
    args = parser.parse_args()

    l20 = json.loads(Path(args.l20).read_text())
    l4 = json.loads(Path(args.l4).read_text())

    corpora = l20["config"]["corpora"].split(",")

    # Sort dictionaries by percentile (numeric).
    def _pct(label: str) -> float:
        # labels look like "p1", "p2", "p4", "p10". Strip leading p.
        s = label.lstrip("p").replace("p", ".")
        try:
            return float(s)
        except ValueError:
            return 0.0

    dicts = sorted(l20["results"].items(), key=lambda kv: _pct(kv[0]))
    pcts   = [_pct(label) for label, _ in dicts]
    K_list = [r["n_partitions"] for _, r in dicts]

    # Per-corpus stat lists, ordered by ascending percentile.
    within_by_corpus: dict[str, list[float]] = {c: [] for c in corpora}
    meand_by_corpus:  dict[str, list[float]] = {c: [] for c in corpora}
    for _, r in dicts:
        for c in corpora:
            within_by_corpus[c].append(r["per_corpus"][c]["within_threshold_rate"])
            meand_by_corpus[c].append(r["per_corpus"][c]["mean_distance"])

    # L4 single-percentile reference points.
    l4_dict = list(l4["results"].values())[0]
    l4_pct  = _pct(list(l4["results"].keys())[0])
    l4_within = {c: l4_dict["per_corpus"][c]["within_threshold_rate"] for c in corpora}
    l4_meand  = {c: l4_dict["per_corpus"][c]["mean_distance"] for c in corpora}

    # ---- Plot ----
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.4), sharex=True)

    colors = {
        "pile":           "#1f77b4",
        "bulgarian_wiki": "#2ca02c",
        "random_tokens":  "#d62728",
    }
    labels = {
        "pile":           "Pile (in-distribution)",
        "bulgarian_wiki": "Bulgarian Wikipedia",
        "random_tokens":  "Random tokens",
    }
    markers_l20 = {"pile": "o", "bulgarian_wiki": "s", "random_tokens": "^"}

    for c in corpora:
        axL.plot(pcts, within_by_corpus[c], "-",
                 marker=markers_l20[c], color=colors[c],
                 label=f"L20 — {labels[c]}", lw=2.0, ms=5.5)
        axR.plot(pcts, meand_by_corpus[c], "-",
                 marker=markers_l20[c], color=colors[c],
                 label=f"L20 — {labels[c]}", lw=2.0, ms=5.5)

    # L4 reference points
    for c in corpora:
        axR.plot([l4_pct], [l4_meand[c]], marker="*", color=colors[c],
                 markersize=14, markeredgecolor="black", markeredgewidth=0.8,
                 linestyle="none",
                 label=f"L4 ($p_{{4}}$) — {labels[c]}")

    axL.set_xlabel(r"Dictionary percentile $p$")
    axL.set_ylabel("Within-threshold rate")
    axL.set_title("Within-threshold rate (L20)")
    axL.set_xscale("log")
    axL.set_ylim(0.85, 1.005)
    axL.grid(True, alpha=0.25, linestyle=":")
    axL.legend(loc="lower right", framealpha=0.9, fontsize=7)
    # Annotate the binary-signal break at p1.
    axL.annotate(
        r"$p_{1}$: random drops to 0.875",
        xy=(1, 0.875), xytext=(1.6, 0.91),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#555"),
    )

    axR.set_xlabel(r"Dictionary percentile $p$")
    axR.set_ylabel("Mean nearest-exemplar distance")
    axR.set_title("Mean nearest-exemplar distance (L20 + L4 markers)")
    axR.set_xscale("log")
    axR.grid(True, alpha=0.25, linestyle=":")
    axR.legend(loc="lower right", framealpha=0.9, fontsize=7, ncol=1)

    fig.suptitle(
        "Held-out coverage and OOD signal across resolution and layer "
        "(gemma-2-2b-it)", fontsize=11, y=1.00,
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
