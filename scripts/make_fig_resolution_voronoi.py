"""2D Voronoi rendering of a single anchor pair's resolution-paths.

Each panel shows one resolution as a 2D Voronoi tessellation. The path
cells are coloured (anchors red, intermediates salmon); their immediate
neighbours form the surrounding context cells in light grey. Path is
connected with arrows. Top tokens labelled inside the path cells.

Geometry: per-panel subset = path cells ∪ top-k nearest of each path
cell. The subset's pairwise cosine-distance submatrix is fed into MDS
to get 2D positions, then scipy 2D Voronoi tessellates. Unbounded
ridges are clipped to a padded bounding box.

Usage:
    uv run python -m scripts.make_fig_resolution_voronoi \\
        --basis-dir /tmp/ep_check/resolution_paths_geom/mean \\
        --paths-json /tmp/ep_check/resolution_paths_mean/resolution_paths.json \\
        --pair-index 2 --output figures/respath_voronoi_mean_cognition.pdf \\
        --title "Cognition / discourse walk (mean basis)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from sklearn.manifold import MDS


def _short_tokens(decode: str, n: int = 3, char_limit: int = 30) -> str:
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


def _bounded_voronoi_polygons(vor: Voronoi, bbox: tuple[float, ...]):
    """Yield (point_index, polygon_xy) for each input point, with unbounded
    cells clipped to bbox = (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = bbox
    radius = 2.0 * max(xmax - xmin, ymax - ymin)

    centre_to_ridges: dict[int, list] = {p: [] for p in range(len(vor.points))}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        centre_to_ridges[p1].append((p2, v1, v2))
        centre_to_ridges[p2].append((p1, v1, v2))

    for p in range(len(vor.points)):
        region_idx = vor.point_region[p]
        verts = vor.regions[region_idx]
        if -1 not in verts and len(verts) > 0:
            poly = np.array([vor.vertices[i] for i in verts])
        else:
            # unbounded → reconstruct using ridges
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
            # angular sort around centroid
            ctr = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
            poly = pts[np.argsort(ang)]
        # clip to bbox via Sutherland-Hodgman
        poly = _clip_to_bbox(poly, xmin, xmax, ymin, ymax)
        if len(poly) >= 3:
            yield p, poly


def _clip_to_bbox(poly: np.ndarray, xmin, xmax, ymin, ymax):
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


def render_panel(ax, geom: np.ndarray, decodes: list[str], path: list[int],
                 k_context: int = 3, seed: int = 0):
    if not path or len(path) < 2:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "(no path)", ha="center", va="center")
        return

    # Subset = path ∪ top-k nearest of each path cell
    subset = list(path)
    for p in path:
        nbr = np.argsort(geom[p])[1:k_context + 1].tolist()
        for n in nbr:
            if n not in subset:
                subset.append(n)
    # Cap to keep visualisation legible
    subset = subset[:60]
    sub_idx = {pid: i for i, pid in enumerate(subset)}
    sub_geom = geom[np.ix_(subset, subset)]

    # MDS to 2D
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=seed, normalized_stress="auto")
    pts = mds.fit_transform(sub_geom)

    # Voronoi
    vor = Voronoi(pts)
    pad = 0.15 * max(pts[:, 0].ptp(), pts[:, 1].ptp())
    bbox = (pts[:, 0].min() - pad, pts[:, 0].max() + pad,
            pts[:, 1].min() - pad, pts[:, 1].max() + pad)

    # Cell colours: path cells use cividis_r (matches wandb partition plots),
    # context cells neutral light grey.
    cmap = plt.get_cmap("cividis_r")
    n_path = len(path)
    path_set = set(path)
    for p, poly in _bounded_voronoi_polygons(vor, bbox):
        pid = subset[p]
        if pid in path_set:
            i_in_path = path.index(pid)
            t = i_in_path / max(n_path - 1, 1)
            face = cmap(t)
            edge = "#222"
        else:
            face, edge = "#eaeaea", "#aaaaaa"
        ax.fill(poly[:, 0], poly[:, 1], facecolor=face, edgecolor=edge,
                linewidth=0.7, zorder=1)

    # Path arrows
    for k in range(len(path) - 1):
        p1 = pts[sub_idx[path[k]]]
        p2 = pts[sub_idx[path[k + 1]]]
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="-|>", color="#4a1010",
                                    lw=1.6, mutation_scale=14,
                                    shrinkA=12, shrinkB=12),
                    zorder=3)

    # Path cell labels
    for k, pid in enumerate(path):
        i = sub_idx[pid]
        x, y = pts[i]
        label = _short_tokens(decodes[k], n=3, char_limit=24)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
                family="DejaVu Sans", zorder=4,
                path_effects=[],
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.85))

    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render(basis_dir: Path, paths_json: Path, pair_index: int,
           output: Path, title: str | None = None) -> None:
    d = json.load(open(paths_json))
    pair = d["pairs"][pair_index]
    per_res = pair["per_resolution"]
    n = len(per_res)

    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, per_res):
        tag = r["tag"]
        geom = np.load(basis_dir / tag / "geom.npz")["geom"]
        path = r["path_pids"] or []
        decodes = r["path_decodes"] or []
        render_panel(ax, geom, decodes, path)
        ax.set_title(f"{tag} (K={r['K']}, hops={r['hops']})",
                     fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12, y=0.98, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(output, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis-dir", type=Path, required=True,
                    help="Local dir containing per-tag {geom.npz, top_tokens.json}")
    ap.add_argument("--paths-json", type=Path, required=True)
    ap.add_argument("--pair-index", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    render(args.basis_dir, args.paths_json, args.pair_index,
           args.output, args.title)


if __name__ == "__main__":
    main()
