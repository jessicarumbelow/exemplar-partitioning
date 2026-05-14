"""Graphical (matplotlib) version of the shared-neighbours / lens figure.

Layout: three resolutions stacked vertically. Each row is a 3-column
GridSpec (anchor A | lens cells stacked | anchor B). Each cell is rendered
as a labelled rounded rectangle. Connector lines fan from anchor A to
each lens cell and from each lens cell to anchor B.

Usage:
    uv run python -m scripts.make_fig_shared_lens \\
        --geom-root /tmp/ep_check/resolution_paths_geom/mean \\
        --paths-json /tmp/ep_check/resolution_paths_mean/resolution_paths.json \\
        --pair-index 2 --resolutions p10,p4,p2 \\
        --output paper/figures/shared_lens_cognition.pdf \\
        --display-cap 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np


def _short_tokens(decode: str, n: int = 3, char_limit: int = 28) -> str:
    seen: list[str] = []
    for t in decode.split(","):
        t = t.strip()
        if not t or t in seen:
            continue
        seen.append(t)
        if len(seen) >= n:
            break
    s = ", ".join(seen)
    if len(s) > char_limit:
        s = s[:char_limit - 1] + "…"
    return s


def _strip_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_box(ax, x, y, w, h, pid_label, decode_label, face, edge,
              fontsize_id=7, fontsize_decode=8.5):
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, facecolor=face, edgecolor=edge, zorder=2,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)
    ax.text(x, y + h / 2 - 0.05, pid_label,
            ha="center", va="top", fontsize=fontsize_id,
            color="#444", family="monospace", zorder=3,
            transform=ax.transAxes)
    ax.text(x, y - 0.04, decode_label,
            ha="center", va="center", fontsize=fontsize_decode,
            family="DejaVu Sans", zorder=3,
            transform=ax.transAxes)


def _render_panel(fig, gs_panel, geom, tok_by_pid, a_pid, b_pid,
                  display_cap, res_tag, K_total):
    """Build one resolution row inside an outer gridspec slot."""
    # Uniform face for all cells; anchors distinguished by position, not colour.
    anchor_a_face = "#ececec"
    anchor_b_face = "#ececec"
    lens_face = "#ececec"

    d_ab = float(geom[a_pid, b_pid])
    mask = (geom[a_pid] < d_ab) & (geom[b_pid] < d_ab)
    mask[a_pid] = False
    mask[b_pid] = False
    cells = sorted(np.flatnonzero(mask).tolist(),
                   key=lambda c: 0.5 * (geom[a_pid, c] + geom[b_pid, c]))
    total = len(cells)
    show = cells[:display_cap]
    truncated = total > display_cap

    # 3 columns inside this panel: anchor-A | lens-stack | anchor-B
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_panel,
        width_ratios=[1.1, 1.6, 1.1], wspace=0.04,
    )
    ax_a = fig.add_subplot(inner[0, 0])
    ax_l = fig.add_subplot(inner[0, 1])
    ax_b = fig.add_subplot(inner[0, 2])
    for ax in (ax_a, ax_l, ax_b):
        _strip_axis(ax)

    # Title across the panel (use ax_l)
    note = (f"|lens|={total}" if not truncated
            else f"|lens|={total} (closest {display_cap} shown)")
    ax_l.set_title(f"{res_tag}    K={K_total}    "
                   f"d(A,B)={d_ab:.3f}    {note}",
                   fontsize=10, pad=6)

    # Anchor A
    _draw_box(ax_a, 0.5, 0.5, w=0.92, h=0.55,
              pid_label=f"p{a_pid}",
              decode_label=_short_tokens(tok_by_pid[a_pid], n=3, char_limit=26),
              face=anchor_a_face, edge="#888")
    # Anchor B
    _draw_box(ax_b, 0.5, 0.5, w=0.92, h=0.55,
              pid_label=f"p{b_pid}",
              decode_label=_short_tokens(tok_by_pid[b_pid], n=3, char_limit=26),
              face=anchor_b_face, edge="#888")

    # Lens cells stacked
    n = len(show)
    if n == 0:
        ax_l.text(0.5, 0.5, "(lens empty)",
                  ha="center", va="center", fontsize=9,
                  color="#888", style="italic", transform=ax_l.transAxes)
    else:
        # box height in axes coords; pack cells into the column
        max_box_h = 0.85 / n
        box_h = min(0.20, max_box_h)
        if box_h < 0.06:
            box_h = 0.06
        gap_h = (0.85 - n * box_h) / max(n - 1, 1) if n > 1 else 0
        top = 0.5 + ((n - 1) * (box_h + gap_h)) / 2 + box_h / 2
        ys = [top - i * (box_h + gap_h) for i in range(n)]
        # connector lines from anchor centres into each lens box
        # we draw lines using figure-level coordinates so they cross axes
        trans_a = ax_a.transAxes
        trans_l = ax_l.transAxes
        trans_b = ax_b.transAxes

        for c, y in zip(show, ys):
            # lens box
            _draw_box(ax_l, 0.5, y, w=0.95, h=box_h,
                      pid_label=f"p{c}",
                      decode_label=_short_tokens(tok_by_pid[c], n=3,
                                                  char_limit=30),
                      face=lens_face, edge="#888",
                      fontsize_id=6.5, fontsize_decode=7.5)
            # connector A → lens
            xy_a = trans_a.transform((1.0, 0.5))
            xy_l_left = trans_l.transform((0.025, y))
            xy_l_right = trans_l.transform((0.975, y))
            xy_b = trans_b.transform((0.0, 0.5))
            inv = fig.transFigure.inverted()
            (sa_x, sa_y) = inv.transform(xy_a)
            (la_x, la_y) = inv.transform(xy_l_left)
            (lb_x, lb_y) = inv.transform(xy_l_right)
            (sb_x, sb_y) = inv.transform(xy_b)
            line_a = plt.Line2D([sa_x, la_x], [sa_y, la_y],
                                color="#999", linewidth=0.7,
                                transform=fig.transFigure, zorder=1)
            line_b = plt.Line2D([lb_x, sb_x], [lb_y, sb_y],
                                color="#999", linewidth=0.7,
                                transform=fig.transFigure, zorder=1)
            fig.add_artist(line_a)
            fig.add_artist(line_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom-root", type=Path, required=True)
    ap.add_argument("--paths-json", type=Path, required=True)
    ap.add_argument("--pair-index", type=int, required=True)
    ap.add_argument("--resolutions", default="p10,p4,p2")
    ap.add_argument("--display-cap", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    res_tags = [t.strip() for t in args.resolutions.split(",") if t.strip()]
    d = json.load(open(args.paths_json))
    pair = d["pairs"][args.pair_index]
    res_by_tag = {r["tag"]: r for r in pair["per_resolution"]}

    # Pre-compute lens sizes so we know how tall each panel needs to be
    panel_lens = []
    for t in res_tags:
        r = res_by_tag[t]
        g = np.load(args.geom_root / t / "geom.npz")["geom"]
        a, b = r["anchor_a_pid"], r["anchor_b_pid"]
        d_ab = float(g[a, b])
        n_lens = int(((g[a] < d_ab) & (g[b] < d_ab)).sum()) - 2  # exclude a, b
        panel_lens.append(max(1, min(args.display_cap, n_lens)))

    n_panels = len(res_tags)
    fig_w = 9.5
    # Each panel ~1.5" base + 0.32" per lens cell
    height_ratios = [1.5 + 0.32 * n for n in panel_lens]
    fig_h = sum(height_ratios) + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = gridspec.GridSpec(n_panels, 1, figure=fig, hspace=0.45,
                              height_ratios=height_ratios)

    if args.title:
        fig.suptitle(args.title, fontsize=12, fontweight="bold", y=0.995)
        outer.update(top=0.95)

    for i, tag in enumerate(res_tags):
        r = res_by_tag[tag]
        geom = np.load(args.geom_root / tag / "geom.npz")["geom"]
        tok = json.load(open(args.geom_root / tag / "top_tokens.json"))
        tok_by_pid = {t["partition_id"]: t["exemplar_top_tokens"] for t in tok}
        _render_panel(fig, outer[i, 0], geom, tok_by_pid,
                      r["anchor_a_pid"], r["anchor_b_pid"],
                      args.display_cap, tag, r["K"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
