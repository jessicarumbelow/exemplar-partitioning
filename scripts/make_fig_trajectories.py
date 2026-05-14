"""Build the cross-layer trajectory alluvial figure.

Three columns (L4, L12, L20). Per-layer stacked region bars sized by how
many prompts visit each region. Ribbons between columns coloured by
category; ribbon width proportional to prompt count.

The headline structure: math collapses onto a single dominant path;
factual is more diverse; harmful (base-model probe) is scattered; code
takes a different route entirely.

Run:
    uv run python -m scripts.make_fig_trajectories
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


CATEGORY_LABEL = {"math": "math", "code": "code",
                  "factual": "factual", "refusal": "harmful"}
CATEGORY_COLOR = {"math": "#1f77b4", "code": "#2ca02c",
                  "factual": "#9467bd", "refusal": "#d62728"}
CATEGORY_ORDER = ["math", "factual", "code", "refusal"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/tmp/ep_results/trajectories.json")
    parser.add_argument(
        "--out-pdf",
        default="paper/figures/trajectories.pdf")
    parser.add_argument(
        "--out-png",
        default="paper/figures/trajectories.png")
    parser.add_argument("--top-per-layer", type=int, default=6,
                        help="Show top-N regions per layer; rest pooled into 'other'.")
    args = parser.parse_args()

    d = json.loads(Path(args.input).read_text())
    layers = d["layers"]
    trajs = d["trajectories"]

    # Build per-layer region counts (sum over all categories).
    per_layer_total: list[Counter] = [Counter() for _ in layers]
    for t in trajs:
        for li, region in enumerate(t["path"]):
            per_layer_total[li][region] += 1

    # Pick top regions per layer; pool rest into a single "other" bucket.
    top_per_layer = []
    for layer_counts in per_layer_total:
        top = [r for r, _ in layer_counts.most_common(args.top_per_layer)]
        top_per_layer.append(top)

    def _reg(layer_idx: int, region_id: int):
        if region_id in top_per_layer[layer_idx]:
            return region_id
        return "other"

    # Per-(layer, region, category) counts.
    layer_region_cat: list[dict] = [defaultdict(lambda: defaultdict(int))
                                     for _ in layers]
    for t in trajs:
        cat = t["category"]
        for li, region in enumerate(t["path"]):
            r = _reg(li, region)
            layer_region_cat[li][r][cat] += 1

    # Per-(L_a, region_a, L_b, region_b, category) counts for ribbons.
    flow_counts: list[dict] = [defaultdict(int), defaultdict(int)]
    for t in trajs:
        cat = t["category"]
        path = t["path"]
        for hop in range(len(path) - 1):
            src = _reg(hop, path[hop])
            dst = _reg(hop + 1, path[hop + 1])
            flow_counts[hop][(src, dst, cat)] += 1

    # Layout: 3 columns, each with stacked region bars.
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    fig, ax = plt.subplots(figsize=(11.0, 1.7))

    # Column x-positions and bar width.
    col_x = [0.0, 0.5, 1.0]
    bar_w = 0.06
    n_prompts = len(trajs)
    gap = 0.012  # gap between region rectangles within a layer
    pad_top = 0.04

    # Determine display order per layer: top regions sorted by total count desc,
    # then "other" at the bottom.
    layer_order = []
    for li in range(len(layers)):
        order = list(top_per_layer[li])
        order.sort(key=lambda r: -per_layer_total[li][r])
        if any(r == "other" for r in layer_region_cat[li].keys()):
            order.append("other")
        layer_order.append(order)

    # Compute y-spans for each (layer, region) and within-region category
    # sub-spans (so ribbons can leave from category-coloured slices).
    region_spans: list[dict] = [{} for _ in layers]   # region -> (y0, y1)
    region_cat_spans: list[dict] = [{} for _ in layers]  # (region, cat) -> (y0, y1)
    region_cat_in_offsets: list[dict] = [{} for _ in layers]  # incoming ribbon offset

    for li in range(len(layers)):
        order = layer_order[li]
        # Find total height: 1.0 minus pad_top minus n_regions*gap.
        total = sum(per_layer_total[li][r] if r != "other"
                    else sum(layer_region_cat[li][r].values()) for r in order)
        avail = 1.0 - pad_top - gap * (len(order) - 1)
        # y starts from top.
        y_cursor = 1.0 - pad_top / 2.0
        for r in order:
            n = (per_layer_total[li][r] if r != "other"
                 else sum(layer_region_cat[li][r].values()))
            h = (n / total) * avail
            y_top = y_cursor
            y_bot = y_cursor - h
            region_spans[li][r] = (y_bot, y_top)
            # Within region, allocate by category (in CATEGORY_ORDER).
            cat_dict = layer_region_cat[li][r]
            yc = y_top
            for cat in CATEGORY_ORDER:
                if cat not in cat_dict:
                    continue
                cn = cat_dict[cat]
                ch = (cn / n) * h
                region_cat_spans[li][(r, cat)] = (yc - ch, yc)
                yc -= ch
            y_cursor = y_bot - gap

    # Draw region bars.
    for li, x in enumerate(col_x):
        for r, (y0, y1) in region_spans[li].items():
            ax.add_patch(plt.Rectangle((x - bar_w / 2, y0), bar_w, y1 - y0,
                                       facecolor="#444444", edgecolor="black",
                                       linewidth=0.6, zorder=3))
            # Region label
            label = "other" if r == "other" else f"#{r}"
            ax.text(x, y1 + 0.005, label, ha="center", va="bottom",
                    fontsize=7.5, zorder=4)

    # Draw category-coloured slices on top of bars.
    for li, x in enumerate(col_x):
        for (r, cat), (y0, y1) in region_cat_spans[li].items():
            ax.add_patch(plt.Rectangle((x - bar_w / 2, y0), bar_w, y1 - y0,
                                       facecolor=CATEGORY_COLOR[cat],
                                       edgecolor="none", alpha=0.95, zorder=4))

    # Draw ribbons between consecutive layers.
    # For each (src_region, src_cat) we need to track where ribbons LEAVE within
    # the source region's category-slice; same for arrivals on target side.
    # Per (layer, region, cat), maintain a running "y cursor" for outgoing/incoming.
    out_cursor: list[dict] = [{} for _ in layers]  # at layer li, going to li+1
    in_cursor: list[dict] = [{} for _ in layers]   # at layer li, coming from li-1
    for li in range(len(layers)):
        for k, (y0, y1) in region_cat_spans[li].items():
            out_cursor[li][k] = y1  # start at top of slice
            in_cursor[li][k] = y1

    for hop in range(len(layers) - 1):
        # Order ribbons by category for stable stacking (same order both sides).
        items = sorted(flow_counts[hop].items(),
                       key=lambda kv: (CATEGORY_ORDER.index(kv[0][2]),
                                       layer_order[hop].index(kv[0][0]),
                                       layer_order[hop + 1].index(kv[0][1])))
        # Sum widths into source/target slots so we know each ribbon's vertical extent.
        for (src, dst, cat), cnt in items:
            # Source slice (region, cat) at layer hop.
            src_slice = region_cat_spans[hop].get((src, cat))
            dst_slice = region_cat_spans[hop + 1].get((dst, cat))
            if src_slice is None or dst_slice is None:
                continue
            sy0, sy1 = src_slice
            dy0, dy1 = dst_slice
            # Compute heights proportional to cnt within the slice.
            src_total = layer_region_cat[hop][src][cat]
            dst_total = layer_region_cat[hop + 1][dst][cat]
            if src_total == 0 or dst_total == 0:
                continue
            sh = (cnt / src_total) * (sy1 - sy0)
            dh = (cnt / dst_total) * (dy1 - dy0)
            # Use cursors to stack ribbons within the slice from top to bottom.
            s_top = out_cursor[hop][(src, cat)]
            s_bot = s_top - sh
            d_top = in_cursor[hop + 1][(dst, cat)]
            d_bot = d_top - dh
            out_cursor[hop][(src, cat)] = s_bot
            in_cursor[hop + 1][(dst, cat)] = d_bot

            # Bezier ribbon: from (xs, s_top..s_bot) to (xd, d_top..d_bot).
            xs = col_x[hop] + bar_w / 2
            xd = col_x[hop + 1] - bar_w / 2
            xm = (xs + xd) / 2
            verts = [
                (xs, s_top),
                (xm, s_top), (xm, d_top), (xd, d_top),  # top curve
                (xd, d_bot),
                (xm, d_bot), (xm, s_bot), (xs, s_bot),  # bottom curve
                (xs, s_top),
            ]
            codes = [
                MplPath.MOVETO,
                MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                MplPath.LINETO,
                MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                MplPath.CLOSEPOLY,
            ]
            ax.add_patch(PathPatch(MplPath(verts, codes),
                                   facecolor=CATEGORY_COLOR[cat],
                                   edgecolor="none", alpha=0.45, zorder=2))

    # Column labels at top.
    for x, layer in zip(col_x, layers):
        ax.text(x, 1.06, f"L{layer}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    # Category legend.
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLOR[c],
                             edgecolor="none", alpha=0.9)
               for c in CATEGORY_ORDER]
    labels = [CATEGORY_LABEL[c] for c in CATEGORY_ORDER]
    ax.legend(handles, labels, loc="lower right", framealpha=0.95,
              fontsize=9, title="Category")

    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.02, 1.12)
    ax.set_axis_off()

    fig.suptitle(
        "Cross-layer region trajectories on Gemma-2-2B "
        r"($p_{8}$ saturated, 100 prompts $\times$ 4 categories)",
        fontsize=11, y=0.99,
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
