"""TikZ Voronoi version of the shared-neighbours / lens figure.

Per panel: anchors A, B placed at left/right; lens cells stacked vertically
in the middle. scipy 2D Voronoi gives polygons; we emit TikZ \\fill plots
with text labels inside each cell. Three resolutions stacked vertically.

The Voronoi here is over the 2D LAYOUT positions, not over the actual
high-D directions — it's a schematic of cell adjacency, not a faithful
projection of activation-space geometry. Caption should make this clear.

Usage:
    uv run python -m scripts.make_fig_lens_voronoi_tikz \\
        --geom-root /tmp/ep_check/resolution_paths_geom/mean \\
        --paths-json /tmp/ep_check/resolution_paths_mean/resolution_paths.json \\
        --pair-index 2 --resolutions p10,p4,p2 \\
        --output paper/figures/shared_lens_cognition.tex \\
        --display-cap 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Voronoi


_LATEX_ESCAPES = {
    "\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$",
    "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}


def _latex_escape(s: str) -> str:
    return "".join(_LATEX_ESCAPES.get(c, c) for c in s)


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


def _clip_polygon(poly: np.ndarray, xmin, xmax, ymin, ymax) -> np.ndarray:
    def clip(pts, axis, value, keep_greater):
        if len(pts) == 0:
            return pts
        out = []
        n = len(pts)
        for i in range(n):
            curr = pts[i]
            prev = pts[(i - 1) % n]
            curr_in = (curr[axis] >= value) if keep_greater else (curr[axis] <= value)
            prev_in = (prev[axis] >= value) if keep_greater else (prev[axis] <= value)
            if curr_in:
                if not prev_in:
                    if curr[axis] != prev[axis]:
                        t = (value - prev[axis]) / (curr[axis] - prev[axis])
                        out.append(prev + t * (curr - prev))
                out.append(curr)
            elif prev_in:
                if curr[axis] != prev[axis]:
                    t = (value - prev[axis]) / (curr[axis] - prev[axis])
                    out.append(prev + t * (curr - prev))
        return np.array(out) if out else np.empty((0, 2))

    poly = clip(poly, 0, xmin, True)
    poly = clip(poly, 0, xmax, False)
    poly = clip(poly, 1, ymin, True)
    poly = clip(poly, 1, ymax, False)
    return poly


def _voronoi_polygons(points: np.ndarray, bbox):
    """Return list of (point_index, polygon_xy) clipped to bbox."""
    xmin, xmax, ymin, ymax = bbox
    radius = 4.0 * max(xmax - xmin, ymax - ymin)
    vor = Voronoi(points)

    centre_to_ridges: dict[int, list] = {p: [] for p in range(len(vor.points))}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        centre_to_ridges[p1].append((p2, v1, v2))
        centre_to_ridges[p2].append((p1, v1, v2))

    out = []
    for p in range(len(vor.points)):
        region_idx = vor.point_region[p]
        verts = vor.regions[region_idx]
        if -1 not in verts and len(verts) > 0:
            poly = np.array([vor.vertices[i] for i in verts])
        else:
            ridges = centre_to_ridges[p]
            poly_pts = []
            for other, v1, v2 in ridges:
                if v1 != -1 and v2 != -1:
                    poly_pts.append(vor.vertices[v1])
                    poly_pts.append(vor.vertices[v2])
                else:
                    finite = vor.vertices[v2 if v1 == -1 else v1]
                    direction = vor.points[other] - vor.points[p]
                    n = np.array([-direction[1], direction[0]])
                    n = n / (np.linalg.norm(n) + 1e-9)
                    midpoint = 0.5 * (vor.points[p] + vor.points[other])
                    far = midpoint + n * radius
                    if np.dot(far - midpoint, finite - midpoint) < 0:
                        far = midpoint - n * radius
                    poly_pts.append(finite)
                    poly_pts.append(far)
            if not poly_pts:
                continue
            pts = np.array(poly_pts)
            ctr = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
            poly = pts[np.argsort(ang)]
        poly = _clip_polygon(poly, xmin, xmax, ymin, ymax)
        if len(poly) >= 3:
            out.append((p, poly))
    return out


def _layout_points(n_lens: int, anchor_x: float = 4.0, lens_y_span: float = 3.0):
    """Place anchor A at (-anchor_x, 0), anchor B at (+anchor_x, 0),
    lens cells stacked vertically at x=0."""
    points = [(-anchor_x, 0.0), (anchor_x, 0.0)]
    if n_lens > 0:
        ys = np.linspace(lens_y_span / 2, -lens_y_span / 2, n_lens) if n_lens > 1 else [0.0]
        for y in ys:
            points.append((0.0, float(y)))
    return np.array(points)


def _emit_panel(panel: dict, y_offset: float, panel_height: float,
                display_cap: int) -> str:
    """Emit TikZ for one panel."""
    out: list[str] = []

    BBOX = (-6.0, 6.0, -2.0, 2.0)
    n_lens = len(panel["lens_show"])
    points = _layout_points(n_lens)

    # Voronoi requires ≥4 points. If we have only 2 anchors and ≤1 lens,
    # add ghost points far outside bbox to stabilise the diagram.
    if len(points) < 4:
        ghosts = np.array([[0.0, 100.0], [0.0, -100.0],
                           [100.0, 0.0], [-100.0, 0.0]])
        all_points = np.vstack([points, ghosts])
        n_real = len(points)
    else:
        all_points = points
        n_real = len(points)

    polygons = _voronoi_polygons(all_points, BBOX)

    # Heading
    pct_num = panel["tag"].lstrip("p").replace("p", ".")
    head_text = (f"$p_{{{pct_num}}}$  "
                 f"\\quad $K = {panel['K']}$ "
                 f"\\quad $d(A, B) = {panel['d_ab']:.3f}$ "
                 f"\\quad $|\\mathrm{{lens}}| = {panel['lens_total']}$"
                 + (f" (closest {display_cap} shown)"
                    if panel['lens_total'] > display_cap else ""))
    head_y = y_offset + (BBOX[3] - BBOX[2]) / 2 + 0.45
    out.append(f"  \\node[head] at (0, {head_y:.2f}) {{{head_text}}};")

    # Translate y by y_offset so panel sits below previous
    panel_centre_y = y_offset

    # Cell colours: anchors slightly tinted; lens light grey
    for p_idx, poly in polygons:
        if p_idx >= n_real:
            continue   # ghost
        # translate y
        verts = [(x, y + panel_centre_y) for x, y in poly]
        if p_idx == 0:    # anchor A
            face = "anchorAfill"
        elif p_idx == 1:  # anchor B
            face = "anchorBfill"
        else:
            face = "lensfill"
        path = " -- ".join(f"({vx:.3f}, {vy:.3f})" for vx, vy in verts)
        out.append(f"  \\fill[{face}, draw=cellline, line width=0.4pt] {path} -- cycle;")

    # Cell labels
    A_dec = _short_tokens(panel["a_dec"], n=3, char_limit=24)
    B_dec = _short_tokens(panel["b_dec"], n=3, char_limit=24)
    out.append(f"  \\node[anchorlabel] at (-4, {y_offset:.2f}) "
               f"{{\\textbf{{region {panel['a_pid']}}} \\\\ "
               f"{_latex_escape(A_dec)}}};")
    out.append(f"  \\node[anchorlabel] at (4, {y_offset:.2f}) "
               f"{{\\textbf{{region {panel['b_pid']}}} \\\\ "
               f"{_latex_escape(B_dec)}}};")
    if n_lens == 0:
        out.append(f"  \\node[lenslabel, gray] at (0, {y_offset:.2f}) "
                   f"{{\\textit{{(lens empty)}}}};")
    else:
        ys = np.linspace(1.5, -1.5, n_lens) if n_lens > 1 else [0.0]
        for (pid, dec), y_local in zip(panel["lens_show"], ys):
            out.append(f"  \\node[lenslabel] at (0, {y_offset + y_local:.2f}) "
                       f"{{\\textcolor{{black!50}}{{\\scriptsize {pid}:}}~"
                       f"{_latex_escape(_short_tokens(dec, n=3, char_limit=26))}}};")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom-root", type=Path, required=True)
    ap.add_argument("--paths-json", type=Path, required=True)
    ap.add_argument("--pair-index", type=int, required=True)
    ap.add_argument("--resolutions", default="p10,p4,p2")
    ap.add_argument("--display-cap", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    res_tags = [t.strip() for t in args.resolutions.split(",") if t.strip()]
    d = json.load(open(args.paths_json))
    pair = d["pairs"][args.pair_index]
    res_by_tag = {r["tag"]: r for r in pair["per_resolution"]}

    panels: list[dict] = []
    for tag in res_tags:
        r = res_by_tag[tag]
        geom = np.load(args.geom_root / tag / "geom.npz")["geom"]
        tok = json.load(open(args.geom_root / tag / "top_tokens.json"))
        tok_by_pid = {t["partition_id"]: t["exemplar_top_tokens"] for t in tok}
        a, b = r["anchor_a_pid"], r["anchor_b_pid"]
        d_ab = float(geom[a, b])
        mask = (geom[a] < d_ab) & (geom[b] < d_ab)
        mask[a] = False
        mask[b] = False
        cells = sorted(np.flatnonzero(mask).tolist(),
                       key=lambda c: 0.5 * (geom[a, c] + geom[b, c]))
        lens_total = len(cells)
        show = cells[: args.display_cap]
        panels.append({
            "tag": tag,
            "K": int(r["K"]),
            "d_ab": d_ab,
            "a_pid": int(a),
            "a_dec": tok_by_pid[a],
            "b_pid": int(b),
            "b_dec": tok_by_pid[b],
            "lens_total": lens_total,
            "lens_show": [(int(c), tok_by_pid[c]) for c in show],
        })

    parts: list[str] = []
    parts.append("% Auto-generated by scripts.make_fig_lens_voronoi_tikz; do not edit by hand.")
    parts.append("\\begin{tikzpicture}[")
    parts.append("    every node/.style={font=\\small, inner sep=0.5pt},")
    parts.append("    head/.style={font=\\small\\bfseries},")
    parts.append("    anchorlabel/.style={align=center, font=\\footnotesize},")
    parts.append("    lenslabel/.style={align=center, font=\\footnotesize, text=black!75},")
    parts.append("]")
    parts.append("  \\definecolor{anchorAfill}{HTML}{D6DBE7}")
    parts.append("  \\definecolor{anchorBfill}{HTML}{E5DDC4}")
    parts.append("  \\definecolor{lensfill}{HTML}{F1F1F1}")
    parts.append("  \\definecolor{cellline}{HTML}{888888}")

    PANEL_H = 4.6
    y = 0.0
    for panel in panels:
        parts.append(_emit_panel(panel, y, PANEL_H, args.display_cap))
        y -= PANEL_H

    parts.append("\\end{tikzpicture}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
