"""Render a single anchor pair's resolution-paths as a 5-row figure.

Each row = one EP percentile; cells along the path are labelled with their
top three logit-lens tokens. Anchor cells (start/end) are red; intermediate
cells are grey. Rows stack p16 → p2, showing path length grow with
resolution.

Usage:
    uv run python -m scripts.make_fig_resolution_paths \\
        --paths-json /tmp/ep_check/resolution_paths_mean/resolution_paths.json \\
        --pair-index 2 \\
        --output figures/respath_mean_pair3.pdf \\
        --title "Cognition/discourse walk (mean basis)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _short_tokens(decode: str, n: int = 3, char_limit: int = 32) -> str:
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


def render(paths_json: Path, pair_index: int, output: Path,
           title: str | None = None) -> None:
    d = json.load(open(paths_json))
    pair = d["pairs"][pair_index]
    per_res = pair["per_resolution"]
    n_rows = len(per_res)

    # Find the longest path for x-axis sizing
    max_path_len = max(len(r["path_pids"] or []) for r in per_res)

    fig_w = max(8.5, 1.8 * max_path_len + 2.5)
    fig_h = 1.4 * n_rows + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, max_path_len)
    ax.set_ylim(-0.5, n_rows + 0.4)
    ax.invert_yaxis()
    ax.axis("off")

    # Title
    if title:
        ax.text(max_path_len / 2 - 0.5, -0.3, title,
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    cell_w, cell_h = 1.55, 0.85
    for row, r in enumerate(per_res):
        path = r["path_pids"] or []
        decodes = r["path_decodes"] or []
        n = len(path)
        # Row label
        label = f"{r['tag']}\nK={r['K']}\nhops={r['hops']}"
        ax.text(-0.4, row + 0.5, label, ha="right", va="center",
                fontsize=9, family="monospace")

        # Center the path horizontally if shorter than max
        x_start = (max_path_len - n) / 2

        for k, (pid, dec) in enumerate(zip(path, decodes)):
            x = x_start + k
            y = row + 0.5
            is_anchor = (k == 0 or k == n - 1)
            face = "#fadcdc" if is_anchor else "#eeeeee"
            edge = "#a01818" if is_anchor else "#777777"
            box = patches.FancyBboxPatch(
                (x - cell_w / 2, y - cell_h / 2), cell_w, cell_h,
                boxstyle="round,pad=0.05,rounding_size=0.08",
                linewidth=1.2, facecolor=face, edgecolor=edge,
            )
            ax.add_patch(box)
            ax.text(x, y - 0.15, _short_tokens(dec, n=3, char_limit=28),
                    ha="center", va="center", fontsize=8,
                    family="DejaVu Sans")
            ax.text(x, y + 0.22, f"p{pid}",
                    ha="center", va="center", fontsize=7,
                    color="#444", family="monospace")

            if k < n - 1:
                # Arrow to next cell
                nx = x_start + (k + 1)
                ax.annotate(
                    "", xy=(nx - cell_w / 2 + 0.02, y),
                    xytext=(x + cell_w / 2 - 0.02, y),
                    arrowprops=dict(arrowstyle="->", color="#888",
                                    lw=1.0),
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-json", type=Path, required=True)
    ap.add_argument("--pair-index", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    render(args.paths_json, args.pair_index, args.output, args.title)


if __name__ == "__main__":
    main()
