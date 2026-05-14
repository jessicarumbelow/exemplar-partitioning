"""Pull behavioural.json from the original (Table 2) cells and the matching
_reanchor cells on the Modal volume, print a per-(p, seed) comparison table
of refusal-ablation Δ across {mean, exemplar, exemplar_reanchored} bases plus
the size-and-coherence-matched null.

Usage:
    cd ~/research && modal run ep_modal_experiments.py::aggregate_reanchor
or local:
    cd ~/research/ep && uv run python -m scripts.aggregate_reanchor \\
        --vol-mount /path/to/locally/mounted/vol
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PERCENTILES = (8, 10, 12, 16, 18, 20)
SEEDS = (0, 1, 2, 3)
MODEL_SHORT = "gemma-2-2b-it"
LAYER = 20
N_PROMPTS = 300


def _path(vol_root: Path, p: int, seed: int, suffix: str) -> Path:
    return (
        vol_root / "ep_experiments" / "behavioral"
        / f"{MODEL_SHORT}_L{LAYER}_p{p}_n{N_PROMPTS}_seed{seed}{suffix}"
        / "behavioral.json"
    )


def _delta(payload: dict, basis: str) -> float | None:
    abl = payload.get("ablation") or {}
    sbb = abl.get("sweep_by_basis") or {}
    runs = sbb.get(basis) or []
    if not runs:
        return None
    k1 = next((e for e in runs if e["k"] == 1), None)
    return None if k1 is None else float(k1["delta"])


def _null_delta(payload: dict, basis: str) -> float | None:
    abl = payload.get("ablation") or {}
    null = abl.get("null_ablation") or {}
    sbb = (null.get("sweep_by_basis") or {}).get(basis) or None
    return None if sbb is None else float(sbb["delta"])


def _fmt(d: float | None) -> str:
    return "  ---" if d is None else f"{d:+.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol-mount", type=Path, required=True,
                        help="Local mount of the ep-results Modal volume.")
    args = parser.parse_args()

    rows = []
    for p in PERCENTILES:
        for seed in SEEDS:
            orig = _path(args.vol_mount, p, seed, "")
            reanc = _path(args.vol_mount, p, seed, "_reanchor")

            orig_payload = json.loads(orig.read_text()) if orig.exists() else {}
            reanc_payload = (
                json.loads(reanc.read_text()) if reanc.exists() else {}
            )

            row = {
                "p": p,
                "seed": seed,
                "delta_mean_orig": _delta(orig_payload, "mean"),
                "delta_exemplar_orig": _delta(orig_payload, "exemplar"),
                "delta_mean_re": _delta(reanc_payload, "mean"),
                "delta_exemplar_re": _delta(reanc_payload, "exemplar"),
                "delta_reanchored_re":
                    _delta(reanc_payload, "exemplar_reanchored"),
                "null_reanchored_re":
                    _null_delta(reanc_payload, "exemplar_reanchored"),
            }
            rows.append(row)

    # Wide table
    cols = ("p", "seed",
            "Δ_mean(orig)", "Δ_exem(orig)",
            "Δ_mean(re)", "Δ_exem(re)", "Δ_REANCHOR(re)",
            "null_REANCHOR(re)")
    widths = (3, 4, 12, 12, 10, 10, 14, 17)
    sep = "  "
    print(sep.join(c.rjust(w) for c, w in zip(cols, widths)))
    print(sep.join("-" * w for w in widths))
    for r in rows:
        print(sep.join([
            str(r["p"]).rjust(widths[0]),
            str(r["seed"]).rjust(widths[1]),
            _fmt(r["delta_mean_orig"]).rjust(widths[2]),
            _fmt(r["delta_exemplar_orig"]).rjust(widths[3]),
            _fmt(r["delta_mean_re"]).rjust(widths[4]),
            _fmt(r["delta_exemplar_re"]).rjust(widths[5]),
            _fmt(r["delta_reanchored_re"]).rjust(widths[6]),
            _fmt(r["null_reanchored_re"]).rjust(widths[7]),
        ]))

    # W1 verdict per cell
    print()
    print("W1 verdict (does on-mean re-anchor flip a previously-failing seed?)")
    flips = 0
    eligible = 0
    for r in rows:
        d_old = r["delta_exemplar_orig"]
        d_new = r["delta_reanchored_re"]
        if d_old is None or d_new is None:
            continue
        if d_old > -0.5:  # original ≈ failed (Δ closer to 0 than -0.5)
            eligible += 1
            verdict = "FLIPPED" if d_new <= -0.5 else "still failed"
            print(f"  p={r['p']:>2} seed={r['seed']}: "
                  f"Δ_old={d_old:+.2f}  Δ_reanchored={d_new:+.2f}  → {verdict}")
            if d_new <= -0.5:
                flips += 1
    print()
    print(f"Total: {flips} of {eligible} previously-failing cells flip "
          f"under deterministic re-anchor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
