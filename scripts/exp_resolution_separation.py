"""Resolution-separation toy (paper §6 and appendix H).

A controlled toy on real activations, testing whether EP can recover
fine-grained or compositional features that never appear in isolation.

Design (colours only)
---------------------
Colours ``{red, blue, yellow}``. Each prompt is a random integer then an ordered
pair of two distinct colours, the backgrounded one bracketed::

    "{n} ({c1}) {c2}"

We read the final-position activation, which sits on ``c2`` -- the recency-
prominent colour. ``c1`` is present in context but is NEVER the read token:
it is the backgrounded, never-isolated feature. Both orders of every pair
are present, so:

  (1) Prominence = recency. "blue red" and "red blue" (same colour set,
      opposite order) separate because the prominent (final) colour differs.
  (2) Never-isolated feature recovery. c1 is never the read token, yet at fine
      resolution (small p) each final-colour region splits by c1.

Resolution p
------------
The threshold is calibrated on THIS toy: it is the p-th percentile of pairwise
cosine distances among the toy's centred-unit activations (single-batch
calibrate, so p is exactly a percentile over all pairs). Small p -> small
threshold -> more regions (fine); large p -> fewer regions (coarse). We expect
a distance ladder

    within-condition (random-int noise)
        <  same-final-colour / different-first (backgrounded colour)
        <  different-final-colour (prominent colour)

so the backgrounded colour is resolved only while the threshold sits below the
top rung -- i.e. at small enough p. Fine-grained recovery is a matter of
choosing a small enough p.

The build reproduces discover()'s streaming leader-clustering exactly: discover
only ever mutates the dictionary via Dictionary.add_batch in a loop over
prompt-batches, so we extract activations once and call add_batch over
fixed-order chunks. Deterministic, and identical partitions to a discover() run.

Metrics (per p)
---------------
- K: number of regions.
- ARI of the region assignment against three label sets:
    * final colour  (c2, prominent)   -- high across most of the sweep
    * first colour  (c1, backgrounded) -- high only at small p (the claim)
    * full condition (c1>c2)          -- high only at small p
- purity of regions against the full condition label.

Dense readout (paper §6)
--------------------
Per-condition centroids in centred-unit space; their 6x6 pairwise cosine
distances. Prediction: conditions sharing the final (prominent) colour are
closest; opposite-order pairs are farthest -- the readout is organised by the
prominent feature, with the backgrounded feature as a secondary axis.

Run:
    uv run python -m scripts.exp_resolution_separation \
        --output-root results/resolution_separation
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemma-2-2b"
DEFAULT_MODEL_SHORT = "gemma-2-2b"
DEFAULT_LAYER = 12
DEFAULT_COLORS = ["red", "blue", "yellow"]
DEFAULT_PERCENTILES = [10, 12, 14, 16, 18, 20, 22, 25, 30, 40, 55]


def build_dataset(
    colors: list[str], final_colors: list[str], n_per_condition: int,
    rng: np.random.Generator, max_int: int,
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    """Ordered colour pairs ``"{n} ({c1}) {c2}"``; n prompts each, random-int prefix.

    The random integer gives within-condition variation (noise) so each condition
    is a realistic cloud of activations rather than a single point. The
    backgrounded colour c1 is bracketed. ``final_colors`` restricts which colours
    may be the final (prominent) token; colours in ``colors`` but not
    ``final_colors`` appear only as the earlier, backgrounded token.
    """
    conditions = [(a, b) for a in colors for b in final_colors if a != b]
    prompts: list[str] = []
    labels: list[tuple[str, str]] = []
    for (c1, c2) in conditions:
        for _ in range(n_per_condition):
            n = int(rng.integers(0, max_int))
            prompts.append(f"{n} ({c1}) {c2}")
            labels.append((c1, c2))
    return prompts, labels, conditions


def build_dictionary_from_acts(X, center, threshold, batch_size, seed):
    """Reproduce discover()'s streaming leader-cluster over pre-extracted acts.

    discover() mutates the dictionary only through Dictionary.add_batch, called
    once per prompt-batch. extract_final_position yields one activation per
    prompt, so a prompt-batch of B prompts is B activations. We therefore chunk
    the activations into batch_size-sized batches in a fixed (seeded) order and
    add each -- the same partition a discover() run would produce.
    """
    from ep import Dictionary

    order = np.random.default_rng(seed).permutation(len(X))
    d = Dictionary(center=center, threshold=float(threshold))
    for bi, s in enumerate(range(0, len(order), batch_size)):
        idx = order[s:s + batch_size]
        d.add_batch(x_batch=X[idx], iteration=bi, global_index_start=int(s))
    d.finalize()
    return d


def write_readout(coarse_d, fine_d, X, full_lbl, prompts,
                  coarse_p, fine_p, output_root: Path):
    """Terse two-resolution readout.

    Both blocks have the same shape: each region, a couple of example member
    prompts, and its distances to the other regions. COARSE regions are the
    prominent (final) colour; FINE regions split by the backgrounded colour.
    Distances use each region's centroid (mean_member_direction, the average
    over its members); region membership is EP's nearest-exemplar assignment.
    """
    from collections import Counter

    def block(d, coarse: bool, p: float) -> list[str]:
        ids, _ = d.assign(X)
        K = len(d.partitions)
        E = np.stack([q.mean_member_direction for q in d.partitions])
        dm = np.clip(1.0 - E @ E.T, 0.0, 2.0)
        label, examples = {}, {}
        for r in range(K):
            mem = np.where(ids == r)[0]
            c1, c2 = Counter(full_lbl[mem]).most_common(1)[0][0].split(">")
            label[r] = c2 if coarse else f"({c1}) {c2}"
            pick = mem[np.linspace(0, len(mem) - 1, min(2, len(mem))).astype(int)]
            examples[r] = ", ".join(f'"{prompts[i]}"' for i in pick)
        order = sorted(range(K), key=lambda r: label[r])
        out = [f"{'COARSE' if coarse else 'FINE'}  (p={p:g}, {K} regions)"]
        for r in order:
            out.append(f"  {label[r]}   e.g. {examples[r]}")
            nbrs = "   ".join(f"{label[j]} {dm[r, j]:.2f}"
                              for j in np.argsort(dm[r]) if j != r)
            out.append(f"    dist:  {nbrs}")
        return out

    L = block(coarse_d, True, coarse_p)
    L.append("")
    L += block(fine_d, False, fine_p)

    text = "\n".join(L)
    path = output_root / "readout.txt"
    path.write_text(text)
    logger.info("Wrote %s", path)
    logger.info("\n%s", text)


def _p_grid(spec: str) -> list[float]:
    lo, hi, step = (float(x) for x in spec.split(","))
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def find_p_for_k(X, center, batch_size, seed, target_k, p_grid, full_lbl=None):
    """Scan p for a dictionary with exactly ``target_k`` regions.

    K is non-increasing in p. Among the p that give exactly target_k regions,
    return the one whose regions are cleanest (highest mean purity against
    ``full_lbl``) when labels are given, else the plateau's middle p. If
    target_k is skipped entirely, return the nearest-K build with exact=False.

    Returns (p, dictionary, threshold, exact).
    """
    from collections import Counter

    from ep import calibrate

    def purity(d):
        ids, _ = d.assign(X)
        total = 0
        for r in range(len(d.partitions)):
            c = Counter(full_lbl[ids == r])
            total += c.most_common(1)[0][1] if c else 0
        return total / len(X)

    builds = []
    for p in p_grid:
        thr = calibrate([X], n_tokens=len(X), percentile=p).threshold
        d = build_dictionary_from_acts(X, center, thr, batch_size, seed)
        builds.append((p, len(d), thr, d))
    exact = [b for b in builds if b[1] == target_k]
    if exact:
        if full_lbl is not None:
            p, _, thr, d = max(exact, key=lambda b: purity(b[3]))
        else:
            p, _, thr, d = exact[len(exact) // 2]
        return p, d, thr, True
    p, _, thr, d = min(builds, key=lambda b: (abs(b[1] - target_k), b[0]))
    return p, d, thr, False


def background_feature_consistency(dirs, full_lbl, colors, final_colors, conditions):
    """Is a never-prominent colour a cohesive feature -- i.e. is "add colour y as
    background" the same displacement whichever prominent colour it sits behind?

    For each background-only colour y and each prominent context f, the
    displacement is centroid(y>f) minus the mean centroid of the other
    backgrounds in that same context. High cosine between contexts means y is
    recovered as a consistent direction (a cohesive feature) even without its
    own region. Returns {y: {contexts, pairwise_cosine, mean_cosine}}.
    """
    cond_set = {f"{a}>{b}" for a, b in conditions}

    def centroid(cond):
        m = dirs[full_lbl == cond].mean(0)
        return m / (np.linalg.norm(m) + 1e-12)

    out = {}
    for y in [c for c in colors if c not in final_colors]:
        disp = {}
        for f in final_colors:
            if y == f or f"{y}>{f}" not in cond_set:
                continue
            others = [b for b in colors if b not in (f, y) and f"{b}>{f}" in cond_set]
            if not others:
                continue
            ref = np.mean([centroid(f"{b}>{f}") for b in others], axis=0)
            disp[f] = centroid(f"{y}>{f}") - ref
        ctx = list(disp)
        cosines = [
            float(disp[ctx[i]] @ disp[ctx[j]]
                  / (np.linalg.norm(disp[ctx[i]]) * np.linalg.norm(disp[ctx[j]]) + 1e-12))
            for i in range(len(ctx)) for j in range(i + 1, len(ctx))
        ]
        if cosines:
            out[y] = {"contexts": ctx, "pairwise_cosine": cosines,
                      "mean_cosine": float(np.mean(cosines))}
    return out


def purity(region_ids: np.ndarray, labels: np.ndarray) -> float:
    """Weighted region purity: sum over regions of the majority-label count / N."""
    total = 0
    for r in np.unique(region_ids):
        mask = region_ids == r
        _, counts = np.unique(labels[mask], return_counts=True)
        total += int(counts.max())
    return total / len(region_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model-short", default=DEFAULT_MODEL_SHORT)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--hook-name", default=None,
                    help="Defaults to blocks.{layer}.hook_resid_post.")
    ap.add_argument("--colors", default=",".join(DEFAULT_COLORS),
                    help="Comma-separated colour words.")
    ap.add_argument("--final-colours", default="",
                    help="Colours allowed as the final (prominent) token. Default: "
                         "same as --colors. A subset makes the excluded colours "
                         "background-only (never prominent).")
    ap.add_argument("--n-per-condition", type=int, default=50)
    ap.add_argument("--percentiles", default=",".join(str(p) for p in DEFAULT_PERCENTILES),
                    help="Comma-separated p sweep (percentiles of pairwise distance).")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Activations per add_batch chunk (streaming granularity).")
    ap.add_argument("--coarse-k", type=int, default=3,
                    help="Target region count for the coarse readout (search p).")
    ap.add_argument("--fine-k", type=int, default=6,
                    help="Target region count for the fine readout (search p).")
    ap.add_argument("--p-grid", default="6,60,0.5",
                    help="min,max,step for the p search that hits --coarse-k/--fine-k.")
    ap.add_argument("--max-int", type=int, default=100,
                    help="Random int prefix drawn from [0, max-int); sets the "
                         "within-condition noise level. Keep small: too much "
                         "jitter misassigns activations across weak splits.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-root", type=Path,
                    default=Path("results/resolution_separation"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)

    import torch
    import transformer_lens as tl
    from ep import calibrate, extract_final_position, set_seed
    from ep.discovery.geometry import centered_unit
    from sklearn.metrics import adjusted_rand_score

    set_seed(args.seed)
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    final_colors = [c.strip() for c in args.final_colours.split(",") if c.strip()] or colors
    unknown = [c for c in final_colors if c not in colors]
    if unknown:
        raise ValueError(f"--final-colours not in --colors: {unknown}")
    percentiles = [float(p) for p in args.percentiles.split(",") if p.strip()]
    hook_name = args.hook_name or f"blocks.{args.layer}.hook_resid_post"
    args.output_root.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    rng = np.random.default_rng(args.seed)
    prompts, labels, conditions = build_dataset(
        colors, final_colors, args.n_per_condition, rng, args.max_int)
    logger.info("Dataset: %d prompts, %d conditions (%s)",
                len(prompts), len(conditions),
                ", ".join(f"{a}>{b}" for a, b in conditions))

    # --- Model ---
    logger.info("Loading %s on %s", args.model, args.device)
    t0 = time.time()
    model = tl.HookedTransformer.from_pretrained_no_processing(
        args.model, device=args.device, dtype=torch.bfloat16)
    model.eval()
    logger.info("Model loaded in %.1fs (d_model=%d)", time.time() - t0, model.cfg.d_model)

    # --- Extract final-position activations once ---
    res = extract_final_position(model, prompts, hook_name)
    X = res.x  # (N, D)
    assert len(X) == len(prompts), (
        f"expected one activation per prompt, got {len(X)} for {len(prompts)} prompts"
    )
    logger.info("Extracted %d final-position activations (D=%d)", len(X), X.shape[1])

    # Label arrays aligned to X (original prompt order).
    first_lbl = np.array([c1 for c1, _ in labels])
    final_lbl = np.array([c2 for _, c2 in labels])
    full_lbl = np.array([f"{c1}>{c2}" for c1, c2 in labels])

    # --- Calibration center (percentile-independent) + distance ladder ---
    cal_ref = calibrate([X], n_tokens=len(X), percentile=50.0)
    center = cal_ref.center
    dirs = centered_unit(X, center)
    sim = np.clip(dirs @ dirs.T, -1.0, 1.0)
    iu = np.triu_indices(len(X), k=1)
    pair_d = (1.0 - sim)[iu]
    ladder = {int(q): float(np.percentile(pair_d, q))
              for q in (1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)}
    logger.info("Pairwise cosine-distance percentiles (guide for choosing p):")
    for q, v in ladder.items():
        logger.info("  p%-3d = %.5f", q, v)

    # --- Sweep p ---
    rows = []
    for p in percentiles:
        cal = calibrate([X], n_tokens=len(X), percentile=p)
        d = build_dictionary_from_acts(X, center, cal.threshold, args.batch_size, args.seed)
        region_ids, _ = d.assign(X)
        row = {
            "p": p,
            "threshold": float(cal.threshold),
            "K": len(d),
            "ari_final": float(adjusted_rand_score(final_lbl, region_ids)),
            "ari_first": float(adjusted_rand_score(first_lbl, region_ids)),
            "ari_full": float(adjusted_rand_score(full_lbl, region_ids)),
            "purity_full": purity(region_ids, full_lbl),
        }
        rows.append(row)
        logger.info(
            "p%-4g  thr=%.5f  K=%-4d  ARI final=%.3f first=%.3f full=%.3f  purity=%.3f",
            p, row["threshold"], row["K"], row["ari_final"], row["ari_first"],
            row["ari_full"], row["purity_full"],
        )

    # --- Dense readout: per-condition centroids, 6x6 cosine-distance matrix ---
    # Order conditions by (final, first) so shared-final-colour pairs are
    # adjacent and the block structure is visible.
    ordered_conditions = sorted(conditions, key=lambda ab: (ab[1], ab[0]))
    centroids = []
    for (c1, c2) in ordered_conditions:
        mask = full_lbl == f"{c1}>{c2}"
        m = dirs[mask].mean(axis=0)
        centroids.append(m / (np.linalg.norm(m) + 1e-12))
    C = np.stack(centroids)
    dense = np.clip(1.0 - C @ C.T, 0.0, 2.0)

    # --- Cohesive background feature: is a never-prominent colour a consistent
    #     displacement across prominent contexts? ---
    bg_consistency = background_feature_consistency(
        dirs, full_lbl, colors, final_colors, conditions)
    for y, info in bg_consistency.items():
        logger.info("background-only '%s': add-'%s' direction is consistent across "
                    "%s at cosine %.3f (1.0 = a single cohesive feature)",
                    y, y, info["contexts"], info["mean_cosine"])

    cond_lbls = [f"{c1}>{c2}" for c1, c2 in ordered_conditions]
    logger.info("Dense readout (centroid cosine distance), ordered by (final, first):")
    logger.info("  %-12s%s", "", "".join(f"{lbl:>12}" for lbl in cond_lbls))
    for i, lbl in enumerate(cond_lbls):
        logger.info("  %-12s%s", lbl, "".join(f"{dense[i, j]:12.3f}" for j in range(len(cond_lbls))))

    # --- Save numbers ---
    out = {
        "model": args.model_short,
        "layer": args.layer,
        "hook_name": hook_name,
        "colors": colors,
        "n_per_condition": args.n_per_condition,
        "n_prompts": len(prompts),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "distance_ladder": ladder,
        "sweep": rows,
        "dense_readout": {
            "conditions": [f"{c1}>{c2}" for c1, c2 in ordered_conditions],
            "distance_matrix": dense.tolist(),
        },
        "background_feature_consistency": bg_consistency,
    }
    json_path = args.output_root / "resolution_separation.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", json_path)

    # --- Terse two-resolution readout ---
    # Search p for exactly coarse_k and fine_k regions.
    p_grid = _p_grid(args.p_grid)
    cp, coarse_d, _, c_exact = find_p_for_k(
        X, center, args.batch_size, args.seed, args.coarse_k, p_grid, full_lbl)
    fp, fine_d, _, f_exact = find_p_for_k(
        X, center, args.batch_size, args.seed, args.fine_k, p_grid, full_lbl)
    logger.info("coarse: p=%g -> K=%d (target %d)", cp, len(coarse_d), args.coarse_k)
    logger.info("fine:   p=%g -> K=%d (target %d)", fp, len(fine_d), args.fine_k)
    if not c_exact:
        logger.warning("No p in grid gives exactly K=%d regions; using nearest "
                       "(K=%d at p=%g).", args.coarse_k, len(coarse_d), cp)
    if not f_exact:
        logger.warning("No p in grid gives exactly K=%d regions; using nearest "
                       "(K=%d at p=%g). Try a finer --p-grid step.",
                       args.fine_k, len(fine_d), fp)
    from collections import Counter
    fids, _ = fine_d.assign(X)
    for r in range(len(fine_d.partitions)):
        comp = Counter(full_lbl[fids == r])
        pur = comp.most_common(1)[0][1] / sum(comp.values())
        if pur < 0.9:
            logger.warning("Fine region %d impure (purity %.2f, dominant %s); "
                           "lower --max-int.", r, pur, comp.most_common(1)[0][0])
    write_readout(coarse_d, fine_d, X, full_lbl, prompts,
                  cp, fp, args.output_root)

    _make_figure(rows, dense, ordered_conditions, args.output_root)


def _make_figure(rows, dense, ordered_conditions, output_root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = [r["p"] for r in rows]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: separation curve.
    axA.plot(ps, [r["ari_final"] for r in rows], "-o", label="final colour (prominent)")
    axA.plot(ps, [r["ari_first"] for r in rows], "-s", label="first colour (backgrounded)")
    axA.plot(ps, [r["ari_full"] for r in rows], "-^", label="full condition")
    axA.set_xlabel("resolution p  (percentile of pairwise distance)")
    axA.set_ylabel("ARI vs region assignment")
    axA.set_ylim(-0.05, 1.05)
    axA.set_title("Separation vs resolution")
    axA.legend(loc="lower left", fontsize=8)
    axK = axA.twinx()
    axK.plot(ps, [r["K"] for r in rows], ":", color="grey", label="K (regions)")
    axK.set_ylabel("K (number of regions)", color="grey")
    axK.tick_params(axis="y", labelcolor="grey")

    # Panel B: dense-readout heatmap.
    lbls = [f"{c1}>{c2}" for c1, c2 in ordered_conditions]
    im = axB.imshow(dense, cmap="viridis")
    axB.set_xticks(range(len(lbls)))
    axB.set_yticks(range(len(lbls)))
    axB.set_xticklabels(lbls, rotation=45, ha="right", fontsize=8)
    axB.set_yticklabels(lbls, fontsize=8)
    axB.set_title("Dense readout: centroid cosine distance")
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = output_root / f"resolution_separation.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Wrote %s", path)
    plt.close(fig)


if __name__ == "__main__":
    main()

