"""Render the exemplar dictionary as a spherical Voronoi diagram.

Geometry: exemplar directions are L2-normalised, so the natural ambient space
is S^(d-1). We PCA-project to R^3, radially push to S^2, then tessellate. Cell
boundaries are an approximation of the high-D partition (same caveat as the
2D PCA Voronoi in build_partitions.py:_wandb_checkpoint).

Output: interactive plotly HTML.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import SphericalVoronoi
from sklearn.decomposition import PCA


def _project_to_sphere(directions: np.ndarray) -> np.ndarray:
    coords = PCA(n_components=3).fit_transform(directions)
    return coords / np.linalg.norm(coords, axis=1, keepdims=True)


def _slerp_arc(a: np.ndarray, b: np.ndarray, n_segments: int) -> np.ndarray:
    """n_segments+1 points along the great-circle arc from unit-vector a to b
    via lerp-then-renormalise. Spacing is slightly non-uniform vs true SLERP
    but visually indistinguishable for our segment counts."""
    ts = np.linspace(0.0, 1.0, n_segments + 1)
    pts = (1.0 - ts)[:, None] * a + ts[:, None] * b
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _partition_snippet(p, max_chars: int = 60) -> str:
    """Short human-readable label for a partition. Prefers an LLM-set
    p.label; otherwise falls back to a window of the closest sample
    prompt centered on the activating token (approximate via ~4 chars
    per BPE token — good enough for hover/screenshot use)."""
    if p.label:
        return p.label
    if not p.sample_prompts:
        return f"n={p.member_count}"
    # sample_prompts is a heap of (-distance, text, position); smallest
    # by -distance == closest member to exemplar.
    closest = min(p.sample_prompts, key=lambda t: -t[0])
    text, pos = closest[1], int(closest[2])
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ").strip()
    if not text:
        return f"n={p.member_count}"
    char_offset = min(len(text), max(0, pos * 4))
    half = max_chars // 2
    start = max(0, char_offset - half)
    end = min(len(text), char_offset + half)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _click_toggle_post_script(markers_idx: int) -> str:
    """JS injected into the HTML.

    On every mousemove we project all exemplar 3D points to screen
    coords using plotly's internal camera matrices, find the nearest
    one to the cursor, and pop a tooltip for it. This gives "fuzzy"
    hover (no need to land exactly on a marker) which is essential
    when the markers are invisible. Click does the same — pin a
    tooltip for the nearest exemplar; click again on the same nearest
    to unpin.

    Camera moves clear hover and pinned tooltips (their fixed screen
    positions stop corresponding to underlying 3D points).
    """
    return f"""
var gd = document.getElementById('{{plot_id}}');
var MARKERS = {markers_idx};
var pinned = new Map();
var hoverDiv = null;

// Cache the exemplar 3D coords once; they're immutable for the plot.
var EX_X = Array.prototype.slice.call(gd.data[MARKERS].x);
var EX_Y = Array.prototype.slice.call(gd.data[MARKERS].y);
var EX_Z = Array.prototype.slice.call(gd.data[MARKERS].z);
var EX_N = EX_X.length;

if (!document.getElementById('sphere-tooltip-style')) {{
    var s = document.createElement('style');
    s.id = 'sphere-tooltip-style';
    s.textContent = `
.sphere-tooltip {{
    position: absolute;
    transform: translate(-50%, -100%);
    background: rgba(255,255,255,0.94);
    border: 1px solid #888;
    border-radius: 3px;
    padding: 3px 7px;
    font-size: 12px;
    font-family: system-ui, sans-serif;
    color: #111;
    pointer-events: none;
    z-index: 10000;
    white-space: nowrap;
}}
.sphere-tooltip::after {{
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -6px;
    border-width: 6px 6px 0 6px;
    border-style: solid;
    border-color: #888 transparent transparent transparent;
}}
.sphere-tooltip::before {{
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    margin-top: -1px;
    border-width: 5px 5px 0 5px;
    border-style: solid;
    border-color: rgba(255,255,255,0.94) transparent transparent transparent;
    z-index: 1;
}}
`;
    document.head.appendChild(s);
}}

function getCanvasRect() {{
    var c = gd.querySelector('canvas');
    return c ? c.getBoundingClientRect() : gd.getBoundingClientRect();
}}

// Apply a 4x4 column-major matrix to a 4-vec (homogeneous).
function mat4Apply(M, x, y, z, w) {{
    return [
        M[0]*x + M[4]*y + M[8]*z + M[12]*w,
        M[1]*x + M[5]*y + M[9]*z + M[13]*w,
        M[2]*x + M[6]*y + M[10]*z + M[14]*w,
        M[3]*x + M[7]*y + M[11]*z + M[15]*w,
    ];
}}

// Plotly's gl-plot3d operates on "scene-world" coords, NOT raw data
// coords. With aspectmode='cube' it normalizes each axis from
// [rangeMin, rangeMax] to [-aspect/2, +aspect/2] before applying the
// view+projection matrices. Skipping this step makes the projected
// sphere 2x too big (data range [-1,1] vs scene range [-0.5,0.5]).
function getSceneScaler() {{
    var sl = gd._fullLayout && gd._fullLayout.scene;
    if (!sl) return null;
    var xR = sl.xaxis.range, yR = sl.yaxis.range, zR = sl.zaxis.range;
    var ar = sl.aspectratio || {{x:1, y:1, z:1}};
    return function(x, y, z) {{
        return [
            (x - (xR[0] + xR[1]) / 2) / ((xR[1] - xR[0]) / 2) * (ar.x / 2),
            (y - (yR[0] + yR[1]) / 2) / ((yR[1] - yR[0]) / 2) * (ar.y / 2),
            (z - (zR[0] + zR[1]) / 2) / ((zR[1] - zR[0]) / 2) * (ar.z / 2),
        ];
    }};
}}

function projectAll() {{
    var scene = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
    if (!scene || !scene.glplot) return null;
    var cam = scene.glplot.cameraParams;
    if (!cam || !cam.view || !cam.projection) return null;
    var scale = getSceneScaler();
    if (!scale) return null;
    var V = cam.view, P = cam.projection;
    var rect = getCanvasRect();
    var W = rect.width, H = rect.height;
    var ox = rect.left + window.scrollX;
    var oy = rect.top + window.scrollY;
    var out = new Array(EX_N);
    for (var i = 0; i < EX_N; i++) {{
        var s = scale(EX_X[i], EX_Y[i], EX_Z[i]);
        var vp = mat4Apply(V, s[0], s[1], s[2], 1);
        var cp = mat4Apply(P, vp[0], vp[1], vp[2], vp[3]);
        if (cp[3] <= 0) {{ out[i] = null; continue; }}
        var ndcX = cp[0] / cp[3], ndcY = cp[1] / cp[3], ndcZ = cp[2] / cp[3];
        out[i] = [
            (ndcX * 0.5 + 0.5) * W + ox,
            (-ndcY * 0.5 + 0.5) * H + oy,
            ndcZ,
        ];
    }}
    return out;
}}

// 4x4 column-major matrix utilities for cursor->sphere ray-casting.
function mat4Mul(a, b) {{
    var o = new Array(16);
    for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) {{
        var s = 0;
        for (var k = 0; k < 4; k++) s += a[k*4 + j] * b[i*4 + k];
        o[i*4 + j] = s;
    }}
    return o;
}}
function mat4Invert(m) {{
    var inv = new Array(16);
    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];
    var det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (Math.abs(det) < 1e-12) return null;
    var d = 1.0 / det;
    for (var i = 0; i < 16; i++) inv[i] *= d;
    return inv;
}}

// Find the cell to label for cursor position (px, py): nearest
// front-facing exemplar in screen space. Strict bbox check ensures
// labels never appear when the cursor is far from the sphere outline.
// (We tried 3D ray-cast cell-membership but plotly's reported camera
// matrices didn't reliably correspond to the rendered viewport, so
// rays from off-sphere cursors were spuriously hitting the sphere.
// Forward projection works fine, so the simpler 2D-nearest approach
// is the robust choice.)
function findCell(px, py) {{
    var proj = projectAll();
    if (!proj) return -1;
    var scene = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
    var cam = scene && scene.glplot && scene.glplot.cameraParams;
    if (!cam || !cam.view) return -1;
    var V = cam.view;

    // First pass: bbox of front-facing exemplar projections. If cursor
    // outside the bbox (with a small pad), bail.
    var minX = 1e18, maxX = -1e18, minY = 1e18, maxY = -1e18;
    for (var i = 0; i < EX_N; i++) {{
        var p = proj[i]; if (!p) continue;
        var vzRel = V[2]*EX_X[i] + V[6]*EX_Y[i] + V[10]*EX_Z[i];
        if (vzRel <= 0) continue;
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
    }}
    var pad = 8;
    if (px < minX - pad || px > maxX + pad
        || py < minY - pad || py > maxY + pad) return -1;

    // Second pass: nearest front-facing exemplar to (px, py).
    var best = -1, bestD = 1e18;
    for (var i = 0; i < EX_N; i++) {{
        var p = proj[i]; if (!p) continue;
        var vzRel = V[2]*EX_X[i] + V[6]*EX_Y[i] + V[10]*EX_Z[i];
        if (vzRel <= 0) continue;
        var dx = p[0] - px, dy = p[1] - py;
        var d = dx*dx + dy*dy;
        if (d < bestD) {{ bestD = d; best = i; }}
    }}
    return best;
}}

function showHover(px, py, text) {{
    if (!hoverDiv) {{
        hoverDiv = document.createElement('div');
        hoverDiv.className = 'sphere-tooltip';
        hoverDiv.style.opacity = '0.85';
        document.body.appendChild(hoverDiv);
    }}
    hoverDiv.textContent = text;
    hoverDiv.style.left = px + 'px';
    hoverDiv.style.top = (py - 6) + 'px';
    hoverDiv.style.display = 'block';
}}
function hideHover() {{ if (hoverDiv) hoverDiv.style.display = 'none'; }}
function clearPinned() {{
    pinned.forEach(function(div) {{ div.remove(); }});
    pinned.clear();
}}
// Re-project all pinned exemplars on camera change so labels stay
// anchored to their cells. Front/back test uses the view matrix
// directly — gd.layout.scene.camera.eye lags during drag, which made
// labels "fly off" when their underlying point was actually on the
// back. In view space the camera looks down -z, so a point with
// view-space z > origin's view-space z is closer to the camera (front).
function repositionPinned() {{
    if (pinned.size === 0) return;
    var scene = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
    var cam = scene && scene.glplot && scene.glplot.cameraParams;
    if (!cam || !cam.view) return;
    var V = cam.view;
    var proj = projectAll();
    if (!proj) return;
    pinned.forEach(function(div, i) {{
        var p = proj[i];
        // view-space z of point i, relative to origin (which is at V[14])
        var vzRel = V[2]*EX_X[i] + V[6]*EX_Y[i] + V[10]*EX_Z[i];
        if (!p || vzRel <= 0) {{
            div.style.display = 'none';
            return;
        }}
        div.style.display = 'block';
        div.style.left = p[0] + 'px';
        div.style.top = (p[1] - 6) + 'px';
    }});
}}

// Pin every cell's label. repositionPinned hides labels for
// back-facing cells so only the visible hemisphere shows at any time.
function pinAll() {{
    var proj = projectAll();
    if (!proj) {{ setTimeout(pinAll, 200); return; }}
    clearPinned();
    for (var i = 0; i < EX_N; i++) {{
        var p = proj[i]; if (!p) continue;
        var div = document.createElement('div');
        div.className = 'sphere-tooltip';
        div.style.left = p[0] + 'px';
        div.style.top = (p[1] - 6) + 'px';
        div.textContent = gd.data[MARKERS].customdata[i];
        document.body.appendChild(div);
        pinned.set(i, div);
    }}
    repositionPinned();
}}
setTimeout(pinAll, 600);

// Show-all / hide-all toggle button, fixed top-right.
(function() {{
    var allShown = true;
    var btn = document.createElement('button');
    btn.textContent = 'Hide all labels';
    btn.style.position = 'fixed';
    btn.style.top = '10px';
    btn.style.right = '10px';
    btn.style.zIndex = '20000';
    btn.style.padding = '6px 12px';
    btn.style.background = 'white';
    btn.style.border = '1px solid #888';
    btn.style.borderRadius = '4px';
    btn.style.cursor = 'pointer';
    btn.style.fontFamily = 'system-ui, sans-serif';
    btn.style.fontSize = '12px';
    btn.addEventListener('click', function() {{
        if (allShown) {{
            clearPinned();
            btn.textContent = 'Show all labels';
            allShown = false;
        }} else {{
            pinAll();
            btn.textContent = 'Hide all labels';
            allShown = true;
        }}
    }});
    document.body.appendChild(btn);
}})();

// Track drag vs click so camera-orbit drags don't fire pin events.
var downPos = null, isDrag = false;
gd.addEventListener('mousedown', function(e) {{
    downPos = {{x: e.pageX, y: e.pageY, t: Date.now()}};
    isDrag = false;
}});
gd.addEventListener('mousemove', function(e) {{
    if (downPos) {{
        var dx = e.pageX - downPos.x, dy = e.pageY - downPos.y;
        if (dx*dx + dy*dy > 25) isDrag = true;
    }}
    var i = findCell(e.pageX, e.pageY);
    if (i < 0) {{ hideHover(); return; }}
    var proj = projectAll();
    var p = proj && proj[i];
    if (!p) {{ hideHover(); return; }}
    showHover(p[0], p[1], gd.data[MARKERS].customdata[i]);
}});
gd.addEventListener('mouseleave', hideHover);
gd.addEventListener('mouseup', function(e) {{
    var dragged = isDrag;
    downPos = null; isDrag = false;
    if (dragged) return;
    var i = findCell(e.pageX, e.pageY);
    if (i < 0) return;
    if (pinned.has(i)) {{
        pinned.get(i).remove();
        pinned.delete(i);
        return;
    }}
    var proj = projectAll();
    var p = proj && proj[i];
    if (!p) return;
    var div = document.createElement('div');
    div.className = 'sphere-tooltip';
    div.style.left = p[0] + 'px';
    div.style.top = (p[1] - 6) + 'px';
    div.textContent = gd.data[MARKERS].customdata[i];
    document.body.appendChild(div);
    pinned.set(i, div);
}});
// plotly_relayouting fires continuously during 3D camera drag (newer
// plotly); plotly_relayout fires after. Hook both so labels track
// smoothly during the drag and land in the right place at the end.
function onCameraChange(e) {{
    if (!e) return;
    if (e['scene.camera'] || e['scene.camera.eye']
        || (typeof e === 'object' && Object.keys(e).some(function(k) {{
            return k.indexOf('scene.camera') === 0;
        }}))) {{
        repositionPinned();
        hideHover();
    }}
}}
gd.on('plotly_relayout', onCameraChange);
if (gd.on) {{
    try {{ gd.on('plotly_relayouting', onCameraChange); }} catch (e) {{}}
}}
// Pin position needs updating during the user's mouse drag too (plotly
// relayout events on 3D may lag); requestAnimationFrame while a drag
// is active keeps things fluid.
function rafLoop() {{
    if (isDrag) repositionPinned();
    requestAnimationFrame(rafLoop);
}}
requestAnimationFrame(rafLoop);
""".strip()


def render(
    dictionary,
    output: Path,
    n_labels: int = 15,  # kept for back-compat; unused in click-toggle mode
    seed: int = 0,
    label_mode: str = "click",
    labels: list[str] | None = None,
) -> None:
    """Render an interactive sphere with click-to-toggle labels.

    If ``labels`` is provided, those strings are used directly (one per
    partition). Otherwise we fall back to a snippet of the closest sample
    prompt — useful for back-compat / local runs without a model loaded.
    """
    partitions = list(dictionary.partitions)
    if len(partitions) < 5:
        raise SystemExit(f"need >=5 partitions for a sphere tessellation, got {len(partitions)}")

    directions = np.stack([p.exemplar_direction for p in partitions])
    points = _project_to_sphere(directions)

    sv = SphericalVoronoi(points, radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()

    fig = go.Figure()

    # Radial lines from origin out to each exemplar.
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    for j in range(len(points)):
        line_x.extend([0.0, points[j, 0], None])
        line_y.extend([0.0, points[j, 1], None])
        line_z.extend([0.0, points[j, 2], None])
    fig.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode="lines",
        line=dict(color="rgba(0,0,0,0.25)", width=1),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Cell edges as great-circle arcs (subdivided so they follow the sphere).
    # No white-shell occluder here — it intercepted WebGL clicks on the
    # markers behind it. Without it, back-of-sphere edges show through but
    # the markers are reliably clickable.
    edge_x, edge_y, edge_z = [], [], []
    seen: set[tuple[int, int]] = set()
    for region in sv.regions:
        n_r = len(region)
        for t in range(n_r):
            a, b = region[t], region[(t + 1) % n_r]
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
        line=dict(color="rgba(0,0,0,0.6)", width=2),
        hoverinfo="skip",
        showlegend=False,
    ))

    if labels is None:
        labels = [_partition_snippet(p) for p in partitions]

    # Small black markers at exemplar positions so each cell has a
    # visible anchor for its label. Hover/click handled in JS via
    # mousemove + nearest-projected-exemplar (see post_script).
    markers_trace_idx = len(fig.data)
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=3, color="rgba(0,0,0,0.85)"),
        customdata=labels,
        hoverinfo="skip",
        showlegend=False,
        name="exemplars",
    ))

    fig.update_layout(
        title=f"Spherical Voronoi · {len(partitions)} partitions  (click a dot to toggle its label)",
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
    fig.write_html(
        output,
        include_plotlyjs="cdn",
        post_script=_click_toggle_post_script(markers_trace_idx),
    )
    print(f"wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="dictionary root (same as build_partitions --output-dir)")
    ap.add_argument("--model-short", type=str, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("sphere_voronoi.html"))
    ap.add_argument("--n-labels", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-mode", choices=["click"], default="click",
                    help="click: every partition is a clickable marker; "
                         "click toggles its top-prompt snippet as a label.")
    args = ap.parse_args()
    from scripts.build_partitions import load_dictionary
    dictionary = load_dictionary(args.output_dir, args.model_short, args.layer)
    render(dictionary, args.out, n_labels=args.n_labels, seed=args.seed,
           label_mode=args.label_mode)


if __name__ == "__main__":
    main()
