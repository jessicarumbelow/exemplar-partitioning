"""3D spherical Voronoi figure highlighting one anchor partition and its
top-k cosine-nearest neighbours, with their logit-lens top tokens.

For paper §3.6.5 (function and content in partition geometry): generate
two contrasting figures — one anchor whose neighbours are content-coherent,
one whose neighbours are function-coherent — to illustrate the heterogeneity
of EP neighbour structure.

Usage:
    uv run python -m scripts.make_fig_neighbourhood \\
        --dict-path /tmp/ep_check/dict/gemma-2-2b_layer12.pkl \\
        --top-tokens-json /tmp/ep_check/alignment/top_tokens.json \\
        --anchors 170,17 \\
        --output-dir figures/
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import SphericalVoronoi
from sklearn.decomposition import PCA


def _project_to_sphere(directions: np.ndarray, seed: int = 0) -> np.ndarray:
    coords = PCA(n_components=3, random_state=seed).fit_transform(directions)
    return coords / np.linalg.norm(coords, axis=1, keepdims=True)


def _slerp_arc(a, b, n_segments: int = 8):
    ts = np.linspace(0.0, 1.0, n_segments + 1)
    pts = (1.0 - ts)[:, None] * a + ts[:, None] * b
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _shell_for_cells(sv: SphericalVoronoi, cell_colors: list[str],
                     arc_segments: int = 8):
    """Build per-cell triangulated mesh, with colour per cell."""
    xyz: list[np.ndarray] = []
    i_idx, j_idx, k_idx = [], [], []
    face_color: list[str] = []
    for region_idx, region in enumerate(sv.regions):
        if len(region) < 3:
            continue
        verts = sv.vertices[region]
        perim_pts: list[np.ndarray] = []
        for t in range(len(verts)):
            arc = _slerp_arc(verts[t], verts[(t + 1) % len(verts)], arc_segments)
            perim_pts.extend(arc[:-1])
        perim = np.asarray(perim_pts)
        centroid = perim.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        c_idx = len(xyz)
        xyz.append(centroid)
        ring_start = len(xyz)
        for v in perim:
            xyz.append(v)
        n = len(perim)
        col = cell_colors[region_idx]
        for t in range(n):
            i_idx.append(c_idx)
            j_idx.append(ring_start + t)
            k_idx.append(ring_start + (t + 1) % n)
            face_color.append(col)
    return (np.asarray(xyz), np.asarray(i_idx), np.asarray(j_idx),
            np.asarray(k_idx), face_color)


def render_one(
    points: np.ndarray, sv: SphericalVoronoi, partitions, top_tokens_by_pid,
    anchor: int, neighbours: list[int], output: Path, label_chars: int = 30,
):
    K = len(partitions)
    # cell_colors: anchor = strong red, neighbours = orange, others = grey
    cell_colors = ["rgba(225,225,230,0.55)"] * K
    cell_colors[anchor] = "rgba(220, 50, 50, 0.85)"
    for nb in neighbours:
        cell_colors[nb] = "rgba(245, 165, 70, 0.85)"

    # SphericalVoronoi.regions is a list with len K; same order as input points
    # (which match partitions). So index alignment is direct.
    sxyz, si, sj, sk, fc = _shell_for_cells(sv, cell_colors)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=sxyz[:, 0], y=sxyz[:, 1], z=sxyz[:, 2],
        i=si, j=sj, k=sk,
        facecolor=fc,
        flatshading=False,
        lighting=dict(ambient=0.95, diffuse=0.05, specular=0.0),
        opacity=1.0,
        hoverinfo="skip",
        showlegend=False,
    ))

    # Cell edges
    edge_x, edge_y, edge_z = [], [], []
    seen: set[tuple[int, int]] = set()
    for region in sv.regions:
        n = len(region)
        for t in range(n):
            a, b = region[t], region[(t + 1) % n]
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            arc = _slerp_arc(sv.vertices[a], sv.vertices[b], 8)
            edge_x.extend(arc[:, 0].tolist() + [None])
            edge_y.extend(arc[:, 1].tolist() + [None])
            edge_z.extend(arc[:, 2].tolist() + [None])
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(0,0,0,0.55)", width=1),
        hoverinfo="skip", showlegend=False,
    ))

    # Anchor + neighbour exemplar dots, with token labels floated above
    highlight_idxs = [anchor] + list(neighbours)
    anchor_xyz = points[highlight_idxs] * 1.02
    label_xyz = points[highlight_idxs] * 1.16
    labels = []
    for pid in highlight_idxs:
        toks = top_tokens_by_pid.get(pid, "")
        # show the top three tokens, deduped
        seen_t = []
        for t in toks.split(","):
            t = t.strip()
            if not t or t in seen_t:
                continue
            seen_t.append(t)
            if len(seen_t) >= 3:
                break
        short = ", ".join(seen_t)
        if len(short) > label_chars:
            short = short[:label_chars] + "…"
        labels.append(f"p{pid}: {short}")

    colors = ["#b30c0c"] + ["#cf7c10"] * len(neighbours)

    fig.add_trace(go.Scatter3d(
        x=anchor_xyz[:, 0], y=anchor_xyz[:, 1], z=anchor_xyz[:, 2],
        mode="markers",
        marker=dict(size=6, color=colors),
        hoverinfo="text", text=labels,
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=label_xyz[:, 0], y=label_xyz[:, 1], z=label_xyz[:, 2],
        mode="text",
        text=labels,
        textfont=dict(size=12, color="#111"),
        hoverinfo="skip", showlegend=False,
    ))

    anchor_decode = top_tokens_by_pid.get(anchor, "?").split(",")[0].strip()
    fig.update_layout(
        title=(f"Neighbourhood of partition {anchor} (\"{anchor_decode}\")  "
               f"— anchor red, top-{len(neighbours)} cosine neighbours orange"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs="cdn")
    print(f"wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict-path", type=Path, required=True)
    ap.add_argument("--top-tokens-json", type=Path, required=True,
                    help="JSON with [{partition_id, exemplar_top_tokens}]")
    ap.add_argument("--anchors", type=str, required=True,
                    help="Comma-separated anchor partition IDs")
    ap.add_argument("--n-neighbours", type=int, default=5)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("paper/figures"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.dict_path, "rb") as f:
        dictionary = pickle.load(f)
    partitions = dictionary.partitions
    K = len(partitions)
    print(f"Loaded {K} partitions from {args.dict_path}")

    top_raw = json.load(open(args.top_tokens_json))
    top_by_pid = {t["partition_id"]: t["exemplar_top_tokens"] for t in top_raw}

    # Project all exemplars
    directions = np.stack([p.exemplar_direction for p in partitions])
    points = _project_to_sphere(directions, seed=args.seed)

    sv = SphericalVoronoi(points, radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()

    # Pairwise cosine on full-D directions for finding neighbours
    sim = directions @ directions.T
    G = 1.0 - sim
    np.fill_diagonal(G, np.inf)

    anchors = [int(a) for a in args.anchors.split(",") if a.strip()]
    for a in anchors:
        nb_idx = np.argsort(G[a])[:args.n_neighbours].tolist()
        out_path = args.output_dir / f"neighbourhood_p{a}.html"
        render_one(points, sv, partitions, top_by_pid,
                   anchor=a, neighbours=nb_idx, output=out_path)


if __name__ == "__main__":
    main()
