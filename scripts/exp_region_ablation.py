"""Run the cross-family region intervention and its controls.

Each model's assignment cache comes from its own p=12 final-token clustering
over 1,176 harmful and 1,176 benign prompts. A region is wholly harmful or
benign when every assigned prompt has that label. At L14 the paper's main
condition is ``swap-c``: project each residual activation off the span of the
harmful-region means in centred space, then add the benign-region mean position
within that span. The script also supports the reported projection and shift
controls and layer sweeps. The opt-in ``swap-global`` control requires
``--holdout-centroids`` and uses corpus-wide harmful and benign means.

``prepare_region_ablation`` produces the prompt split consumed here. Evaluation
prompts exclude region exemplars. With ``--holdout-centroids``, they are also
excluded from the region means. Calibration, clustering, and region selection
still use the full corpus; this is not an independently held-out dataset.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from ep.discovery.calibration import calibrate
from ep.discovery.geometry import centered_unit
from scripts.refusal_eval import generate_hooked as _generate_hooked, score as _score

logger = logging.getLogger(__name__)

CANON = {"g_it": 18, "l_it": 17}
BS = 16   # streaming batch size build_cache.py calibrated with


def calibration_centre(a):
    """The same centre build_cache.py clustered with: same permutation, batch
    size and percentile, so the exemplar directions here are the ones EP used."""
    o = np.random.default_rng(0).permutation(len(a))
    c = calibrate((a[o][i:i + BS] for i in range(0, len(a), BS)),
                  n_tokens=len(a), percentile=12)
    return c.center.astype(np.float32)


def regions(ids, dists):
    """region id -> (exemplar prompt index, member indices), largest first."""
    out = {int(r): (int(m[np.argmin(dists[m])]), m)
           for r in np.unique(ids) if len(m := np.where(ids == r)[0])}
    return dict(sorted(out.items(), key=lambda kv: -len(kv[1][1])))


def project_off(basis, centre, put_back=None):
    """Remove the component in `basis` (measured from `centre`); `put_back` is an
    optional vector already inside the span to add in its place."""
    def hook(act, hook):
        shape = act.shape
        x = act.float().reshape(-1, shape[-1]) - centre
        x = x - (x @ basis) @ basis.T + centre
        if put_back is not None:
            x = x + put_back
        return x.reshape(shape).to(act.dtype)
    hook.n = basis.shape[1]
    return hook


def add_shift(vec):
    """Add a fixed vector at every position (mean-shift, no projection)."""
    def hook(act, hook):
        return act + vec.to(device=act.device, dtype=act.dtype)
    hook.n = 1
    return hook


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


CANDIDATES_BASE = ["harmful-span", "benign-span", "random-span", "swap", "meanshift",
                   "harmful-only-dir", "harmful-only-span",
                   "harmful-span-c", "benign-span-c", "swap-c", "meanshift-c"]


def condition_todo(out_dir, selected, force, candidates, holdout_centroids=False):
    """What is left to compute at out_dir, from the results file alone (no model,
    no activations) so we can decide whether to load the model at all."""
    existing = set()
    rpath = out_dir / "results.json"
    if rpath.exists():
        saved = json.loads(rpath.read_text())
        if saved.get("holdout_centroids") != holdout_centroids:
            raise ValueError("Centroid exclusion differs from saved results; use a different output directory")
        existing = {r["condition"] for r in saved.get("rows", [])}
    return [c for c in candidates
            if (selected is None or c in selected) and (force or c not in existing)]


def run_layer(model, args, L, out_dir, acts_all, z, meta, prompts, candidates):
    """Compute + write all wanted conditions at one layer. Model is passed in so a
    sweep loads it once."""
    selected = None if args.conditions == "all" else set(args.conditions.split(","))
    condition_todo(out_dir, selected, args.force, candidates, args.holdout_centroids)
    held_h, held_b = meta["held_harmful_idx"], meta["held_benign_idx"]
    a = np.asarray(acts_all[:, L], dtype=np.float32)
    is_harmful = np.arange(len(a)) < meta["n_per_side"]
    if len(a) != 2 * meta["n_per_side"]:
        raise ValueError("activation rows do not match the prompt split")
    ids, dists = z["ids"][L], z["dists"][L]
    centre = calibration_centre(a)
    reg = regions(ids, dists)
    h_ex = [e for e, m in reg.values() if is_harmful[m].all()]
    b_ex = [e for e, m in reg.values() if (~is_harmful[m]).all()]
    exemplars = set(h_ex) | set(b_ex)
    eval_h = [i for i in held_h if i not in exemplars][:args.n_eval]
    eval_b = [i for i in held_b if i not in exemplars][:args.n_eval]
    sides = {"harmful": [prompts[i] for i in eval_h],
             "benign": [prompts[i] for i in eval_b]}
    n = len(h_ex)
    if n < 2 or len(b_ex) < 2:
        logger.info("L%d: only %d harmful / %d benign pure regions; skipping",
                    L, n, len(b_ex))
        return
    rng = np.random.default_rng(0)
    dirs = {
        "harmful-span": centered_unit(a[h_ex], centre),
        "benign-span": centered_unit(a[b_ex[:n]], centre),
        "random-span": rng.normal(size=(n, a.shape[1])).astype(np.float32),
    }
    norm = float(np.median(np.linalg.norm(a, axis=1)))
    steer_names = [c for c in candidates if c.startswith("harmful-only-steer@")]

    selected = None if args.conditions == "all" else set(args.conditions.split(","))
    rpath = out_dir / "results.json"
    existing = set()
    if rpath.exists():
        existing = {r["condition"] for r in json.loads(rpath.read_text()).get("rows", [])}
    def want(c):
        return (selected is None or c in selected) and (args.force or c not in existing)
    todo = [c for c in candidates if want(c)]
    if not todo:
        logger.info("L%d: nothing to do (all requested already in %s)", L, rpath)
        return

    logger.info("L%d: K=%d, %d harmful / %d benign regions; eval %d/%d; running: %s",
                L, len(reg), n, len(b_ex), len(eval_h), len(eval_b), ", ".join(todo))
    hook_name = f"blocks.{L}.hook_resid_post"
    centre_t = torch.tensor(centre, device=args.device)
    rows, completions, t0 = [], {}, time.time()

    def run(cond, side, hooks):
        gens = _generate_hooked(model, sides[side], hooks, args.max_new_tokens,
                                args.batch_size)
        rows.append({"condition": cond, "side": side, "layer": L,
                     "n_directions": 0 if not hooks else hooks[0][1].n, **_score(gens)})
        completions[f"{cond}|{side}"] = gens
        logger.info("  %-22s %-7s refusal %.2f  uniq %.2f", cond, side,
                    rows[-1]["refusal_rate"], rows[-1]["unique_token_ratio"])
        return rows[-1]

    # Baseline always runs when we compute anything: it is the abort guard.
    for side in sides:
        run("baseline", side, [])
    base_h = next(r for r in rows if r["condition"] == "baseline"
                  and r["side"] == "harmful")["refusal_rate"]
    if base_h < args.min_baseline_refusal:
        raise SystemExit(f"L{L}: unsteered harmful refusal {base_h:.2f} < "
                         f"{args.min_baseline_refusal:.2f}; the harness is altering "
                         "behaviour before any intervention.")

    bases = {c: np.linalg.qr(D.T)[0] for c, D in dirs.items()}
    for cond, basis_np in bases.items():
        if not want(cond):
            continue
        basis = torch.tensor(basis_np, dtype=torch.float32, device=args.device)
        hooks = [(hook_name, project_off(basis, centre_t))]
        run(cond, "harmful", hooks)
        if cond == "harmful-span":
            run(cond, "benign", hooks)

    if want("swap"):
        Bh = bases["harmful-span"]
        put_back = torch.tensor(Bh @ (Bh.T @ (a[b_ex].mean(0) - centre)),
                                dtype=torch.float32, device=args.device)
        basis = torch.tensor(Bh, dtype=torch.float32, device=args.device)
        run("swap", "harmful", [(hook_name, project_off(basis, centre_t, put_back))])
        run("swap", "benign", [(hook_name, project_off(basis, centre_t, put_back))])

    if want("meanshift"):
        shift = torch.tensor(a[b_ex].mean(0) - a[h_ex].mean(0),
                             dtype=torch.float32, device=args.device)
        run("meanshift", "harmful", [(hook_name, add_shift(shift))])
        run("meanshift", "benign", [(hook_name, add_shift(-shift))])

    Hc = centered_unit(a[h_ex], centre)
    Bb_full, _ = np.linalg.qr(centered_unit(a[b_ex], centre).T)
    muH = Hc.mean(0)
    muH /= np.linalg.norm(muH)
    v = muH - Bb_full @ (Bb_full.T @ muH)
    v /= np.linalg.norm(v)
    if want("harmful-only-dir"):
        Bhod = torch.tensor(v[:, None], dtype=torch.float32, device=args.device)
        run("harmful-only-dir", "harmful", [(hook_name, project_off(Bhod, centre_t))])
        run("harmful-only-dir", "benign", [(hook_name, project_off(Bhod, centre_t))])
    if want("harmful-only-span"):
        resid = Hc.T - Bb_full @ (Bb_full.T @ Hc.T)
        U, sv, _ = np.linalg.svd(resid, full_matrices=False)
        k = int((sv > 0.3).sum())
        Bperp = torch.tensor(U[:, :k], dtype=torch.float32, device=args.device)
        run("harmful-only-span", "harmful", [(hook_name, project_off(Bperp, centre_t))])
        run("harmful-only-span", "benign", [(hook_name, project_off(Bperp, centre_t))])
    if any(want(sn) for sn in steer_names):
        v_t = torch.tensor(v, dtype=torch.float32, device=args.device)
        for sn in steer_names:
            if not want(sn):
                continue
            alpha = float(sn.split("@")[1])
            run(sn, "harmful", [(hook_name, add_shift(-alpha * norm * v_t))])
            run(sn, "benign", [(hook_name, add_shift(alpha * norm * v_t))])

    # Centroid variants: each region described by the mean of its members.
    # Held-out: drop the evaluated prompts from the means the swap is scored
    # against. Calibration and region construction still use all prompts.
    # Each pure region retains its exemplar, which evaluation excludes.
    drop = (set(eval_h) | set(eval_b)) if args.holdout_centroids else set()
    if selected and "swap-global" in selected and want("swap-global"):
        if not args.holdout_centroids:
            raise ValueError("swap-global requires --holdout-centroids")
        keep = np.array([i not in drop for i in range(len(a))])
        harmful_mean = a[keep & is_harmful].mean(0)
        benign_mean = a[keep & ~is_harmful].mean(0)
        direction = centered_unit(harmful_mean[None, :], centre).T
        basis = torch.tensor(direction, dtype=torch.float32, device=args.device)
        offset = torch.tensor(benign_mean - centre, dtype=torch.float32,
                              device=args.device)
        put_back = basis @ (basis.T @ offset)
        hooks = [(hook_name, project_off(basis, centre_t, put_back))]
        run("swap-global", "harmful", hooks)
        run("swap-global", "benign", hooks)
    _fell_back = [0]
    def _cmean(m):
        keep = [i for i in m if i not in drop]
        if not keep:
            _fell_back[0] += 1
            keep = list(m)
        return a[keep].mean(0)
    cH = np.stack([_cmean(m) for _, m in reg.values() if is_harmful[m].all()])
    cB = np.stack([_cmean(m) for _, m in reg.values() if (~is_harmful[m]).all()])
    if args.holdout_centroids:
        logger.info("L%d: held-out centroids (dropped %d eval prompts; %d regions "
                    "fell back to full mean)", L, len(drop), _fell_back[0])
    bases_c = {"harmful-span-c": np.linalg.qr(centered_unit(cH, centre).T)[0],
               "benign-span-c": np.linalg.qr(centered_unit(cB[:n], centre).T)[0]}
    for cond, basis_np in bases_c.items():
        if not want(cond):
            continue
        basis = torch.tensor(basis_np, dtype=torch.float32, device=args.device)
        hooks = [(hook_name, project_off(basis, centre_t))]
        run(cond, "harmful", hooks)
        if cond == "harmful-span-c":
            run(cond, "benign", hooks)
    if want("swap-c"):
        Bh = bases_c["harmful-span-c"]
        put_back = torch.tensor(Bh @ (Bh.T @ (cB.mean(0) - centre)),
                                dtype=torch.float32, device=args.device)
        basis = torch.tensor(Bh, dtype=torch.float32, device=args.device)
        run("swap-c", "harmful", [(hook_name, project_off(basis, centre_t, put_back))])
        run("swap-c", "benign", [(hook_name, project_off(basis, centre_t, put_back))])
    if want("meanshift-c"):
        shift = torch.tensor(cB.mean(0) - cH.mean(0),
                             dtype=torch.float32, device=args.device)
        run("meanshift-c", "harmful", [(hook_name, add_shift(shift))])
        run("meanshift-c", "benign", [(hook_name, add_shift(-shift))])

    # Merge into any existing results/completions so a partial rerun keeps the rest.
    out_dir.mkdir(parents=True, exist_ok=True)
    cpath = out_dir / "completions.json"
    ran = {r["condition"] for r in rows}
    if rpath.exists():
        prev = json.loads(rpath.read_text())
        rows = [r for r in prev.get("rows", []) if r["condition"] not in ran] + rows
    if cpath.exists():
        prevc = json.loads(cpath.read_text())
        for key, val in prevc.items():
            if key.split("|")[0] not in ran and key not in completions:
                completions[key] = val
    rpath.write_text(json.dumps(
        {"model": args.model, "tag": args.tag, "layer": L, "K": len(reg),
         "n_harmful_regions": n, "n_benign_regions": len(b_ex),
         "n_eval_harmful": len(eval_h), "n_eval_benign": len(eval_b),
         "max_new_tokens": args.max_new_tokens,
         "holdout_centroids": args.holdout_centroids,
         "harmful_exemplar_idx": h_ex, "benign_exemplar_idx": b_ex[:n],
         "elapsed_s": time.time() - t0, "rows": rows}, indent=1))
    cpath.write_text(json.dumps(completions, indent=1))
    logger.info("L%d: wrote %s (%.0f s)", L, out_dir, time.time() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--tag", default="g_it", choices=["g_it", "l_it"])
    p.add_argument("--layer", type=int, default=-1, help="-1: best-separated layer")
    p.add_argument("--layers", default="",
                   help="sweep these layers instead of --layer, e.g. '8-20' or "
                        "'8,10,12'. Writes one dir per layer under --output-dir.")
    p.add_argument("--acts", type=Path, required=True,
                   help="acts_all_layers.npy (N, n_layers, d) from the layer sweep")
    p.add_argument("--cache", type=Path, required=True, help="crossmodel/cache_<tag>.npz")
    p.add_argument("--split-dir", type=Path, required=True,
                   help="prompts.json and meta.json from prepare_region_ablation")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="single-layer: the output dir. sweep (--layers): the root; "
                        "each layer writes <root>/<tag>_L<L>_n<n_eval>.")
    p.add_argument("--holdout-centroids", action="store_true",
                   help="exclude evaluated prompts from region means, not dictionary construction")
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-baseline-refusal", type=float, default=0.85)
    p.add_argument("--conditions", default="all",
                   help="comma-separated conditions to (re)run; 'all' runs everything. "
                        "baseline always runs. Others merge into the existing file, so "
                        "a rerun does not recompute conditions already present.")
    p.add_argument("--steer-alphas", default="0.5,1.0,2.0",
                   help="magnitudes for harmful-only-steer, x the layer's median "
                        "residual norm.")
    p.add_argument("--force", action="store_true",
                   help="recompute conditions even if already in the results file.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sweep = bool(args.layers)
    layers = parse_layers(args.layers) if sweep else \
        [CANON[args.tag] if args.layer < 0 else args.layer]
    candidates = CANDIDATES_BASE + [f"harmful-only-steer@{float(x):g}"
                                    for x in args.steer_alphas.split(",")]
    selected = None if args.conditions == "all" else set(args.conditions.split(","))

    if selected and "swap-global" in selected:
        candidates = candidates + ["swap-global"]

    meta = json.loads((args.split_dir / "meta.json").read_text())
    prompts = json.loads((args.split_dir / "prompts.json").read_text())
    z = np.load(args.cache)
    acts_all = np.load(args.acts, mmap_mode="r")

    def out_dir_for(L):
        return (args.output_dir / f"{args.tag}_L{L}_n{args.n_eval}") if sweep \
            else args.output_dir

    # Decide, without the model, which layers have work left.
    work = [(L, out_dir_for(L)) for L in layers
            if condition_todo(out_dir_for(L), selected, args.force, candidates,
                              args.holdout_centroids)]
    if not work:
        logger.info("nothing to do for layers %s (use --force to recompute)", layers)
        return
    logger.info("layers with work: %s", [L for L, _ in work])

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, device=args.device, dtype=torch.bfloat16)
    model.eval()

    for L, od in work:
        run_layer(model, args, L, od, acts_all, z, meta, prompts, candidates)


if __name__ == "__main__":
    main()
