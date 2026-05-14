"""Text-mode "constellation" rendering of shared neighbours between two
close partitions. For each (anchor_A, anchor_B) pair, list the cells that
appear in the top-K cosine-nearest of both endpoints, drawn as ASCII tree.

Output: plain-text file suitable for LaTeX \\begin{verbatim} or markdown
fenced code block.

Usage:
    uv run python -m scripts.make_fig_shared_neighbours \\
        --geom-root /tmp/ep_check/resolution_paths_geom \\
        --paths-mean /tmp/ep_check/resolution_paths_mean/resolution_paths.json \\
        --paths-exemplar /tmp/ep_check/resolution_paths/resolution_paths.json \\
        --output paper/figures/shared_neighbours.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _short_tokens(decode: str, n: int = 3, char_limit: int = 36) -> str:
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


def constellation(anchor_a: str, anchor_b: str,
                  shared: list[tuple[str, float, float]],
                  *, anchor_w: int = 36, mid_w: int = 38) -> str:
    """Render an ASCII constellation: two anchors on left/right, shared
    neighbours stacked between them with box-drawing connectors. Each
    `shared` entry is (decode_string, dist_to_a, dist_to_b)."""
    a = _short_tokens(anchor_a, n=4, char_limit=anchor_w)
    b = _short_tokens(anchor_b, n=4, char_limit=anchor_w)
    n = len(shared)
    if n == 0:
        return f"{a:<{anchor_w}}  ─x─  {b}\n  (no shared top-8 neighbours)\n"

    a_pad = a.ljust(anchor_w)
    b_pad = b
    out_lines: list[str] = []

    for i, (s_dec, d_a, d_b) in enumerate(shared):
        s = _short_tokens(s_dec, n=3, char_limit=mid_w)
        s_pad = s.ljust(mid_w)
        if n == 1:
            out_lines.append(f"{a_pad}  ──── {s_pad} ────  {b_pad}")
        else:
            if i == 0:
                left, lbr, rbr = a_pad, " ─┬── ", " ──┬─ "
                right = b_pad
            elif i == n - 1:
                left = " " * anchor_w
                lbr = "  └── "
                rbr = " ──┘  "
                right = ""
            else:
                left = " " * anchor_w
                lbr = "  ├── "
                rbr = " ──┤  "
                right = ""
            if i == 0:
                out_lines.append(f"{left}{lbr}{s_pad}{rbr}{right}")
            else:
                out_lines.append(f"{left}{lbr}{s_pad}{rbr}{right}")
    return "\n".join(out_lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom-root", type=Path, required=True)
    ap.add_argument("--paths-mean", type=Path, required=True)
    ap.add_argument("--paths-exemplar", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k-nearest", type=int, default=20)
    ap.add_argument("--criterion", choices=("topk", "lens"), default="lens",
                    help="'topk': cells in the top-K nearest of both. "
                         "'lens': cells with d(A,C)<d(A,B) and d(B,C)<d(A,B).")
    ap.add_argument("--display-cap", type=int, default=8,
                    help="When |lens| exceeds this, show only the closest "
                         "--display-cap cells (by mean distance) in the "
                         "constellation; the lens-population is reported in "
                         "the heading.")
    args = ap.parse_args()

    out_lines: list[str] = []
    out_lines.append("Shared-neighbour constellations across resolutions")
    out_lines.append("=" * 80)
    out_lines.append("")
    out_lines.append("For each anchor pair, cells listed in the middle column appear in")
    if args.criterion == "topk":
        out_lines.append(f"the top-{args.k_nearest} cosine-nearest neighbours of BOTH anchors. d_a, d_b")
        out_lines.append("are the cosine distances from the shared cell to anchor A and B.")
    else:
        out_lines.append("a 'lens' between the two anchors: every cell C with d(A,C) < d(A,B)")
        out_lines.append("AND d(B,C) < d(A,B). I.e., cells that are closer to BOTH endpoints")
        out_lines.append("than the endpoints are to each other. Listed by mean distance.")
    out_lines.append("")

    for basis, paths_path in (("mean", args.paths_mean),
                              ("exemplar", args.paths_exemplar)):
        d = json.load(open(paths_path))
        out_lines.append("")
        out_lines.append("#" * 80)
        out_lines.append(f"BASIS = {basis}")
        out_lines.append("#" * 80)
        out_lines.append("")

        for pair_idx, pair in enumerate(d["pairs"]):
            for r in pair["per_resolution"]:
                tag = r["tag"]
                geom = np.load(args.geom_root / basis / tag / "geom.npz")["geom"]
                tok = json.load(open(args.geom_root / basis / tag / "top_tokens.json"))
                tok_by_pid = {t["partition_id"]: t["exemplar_top_tokens"] for t in tok}

                a_pid = r["anchor_a_pid"]
                b_pid = r["anchor_b_pid"]
                d_ab = float(geom[a_pid, b_pid])
                if args.criterion == "topk":
                    K_top = args.k_nearest
                    nbr_a = set(np.argsort(geom[a_pid])[1:K_top + 1].tolist())
                    nbr_b = set(np.argsort(geom[b_pid])[1:K_top + 1].tolist())
                    shared = (nbr_a & nbr_b) - {a_pid, b_pid}
                else:
                    # lens: cells that are closer to both A and B than they are to each other
                    mask = (geom[a_pid] < d_ab) & (geom[b_pid] < d_ab)
                    mask[a_pid] = False
                    mask[b_pid] = False
                    shared = set(np.flatnonzero(mask).tolist())
                if not shared:
                    continue
                shared_sorted = sorted(
                    shared,
                    key=lambda c: 0.5 * (geom[a_pid, c] + geom[b_pid, c]),
                )
                lens_total = len(shared_sorted)
                truncated = lens_total > args.display_cap
                display_cells = shared_sorted[: args.display_cap]
                shared_data = [
                    (tok_by_pid[c], float(geom[a_pid, c]), float(geom[b_pid, c]))
                    for c in display_cells
                ]
                shown_note = (f"showing closest {len(display_cells)}"
                              if truncated else "all shown")
                heading = (f"Pair {pair_idx + 1}  ·  {tag}  (K={r['K']}, "
                           f"d(a,b)={d_ab:.3f}, "
                           f"|lens|={lens_total} [{args.criterion}], "
                           f"{shown_note})")
                out_lines.append(heading)
                out_lines.append("-" * len(heading))
                out_lines.append(constellation(
                    tok_by_pid[a_pid], tok_by_pid[b_pid], shared_data,
                ))
                # Distance footer per shared cell
                for s_dec, d_a, d_b in shared_data:
                    out_lines.append(
                        f"   · d_a={d_a:.3f}  d_b={d_b:.3f}    "
                        f"{_short_tokens(s_dec, n=4, char_limit=58)}"
                    )
                out_lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
