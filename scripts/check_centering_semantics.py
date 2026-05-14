"""Diagnostic for §3.5 centering ablation.

Two questions:
  1. Does centering reorganise the neighbour graph, or just rescale it?
     -> Spearman rank correlation between centered and uncentered pairwise
        distance matrices. ~1.0 means rescaling. <<1.0 means reorganisation.
  2. Does the reorganisation align with semantic structure?
     -> Within-category vs between-category mean cosine distance, both metrics.

Setup: Pythia-70m-deduped, layer 4 resid_post, final-position activations on
"The {word}" prompts across 5 semantic categories.

Outputs:
  - Console: Spearman rho, within/between gap (uncentered + centered).
  - Figure: docs/figures/fig_centering_semantic.png — paired heatmaps of
    pairwise distance grouped by category.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformer_lens as tl
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ep.discovery.extraction import extract_final_position  # noqa: E402

CATEGORIES: dict[str, list[str]] = {
    "animal":  ["dog", "cat", "horse", "bird", "fish", "tiger", "bear", "rabbit", "mouse", "elephant"],
    "food":    ["bread", "apple", "cheese", "pizza", "soup", "cake", "pasta", "salad", "rice", "butter"],
    "vehicle": ["car", "truck", "bike", "plane", "boat", "train", "bus", "ship", "scooter", "motorcycle"],
    "body":    ["hand", "foot", "head", "arm", "leg", "knee", "elbow", "finger", "eye", "nose"],
    "color":   ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white"],
}


def pairwise_cos_dist(V: np.ndarray) -> np.ndarray:
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Vn @ Vn.T


def within_between(D: np.ndarray, label_idx: np.ndarray) -> tuple[float, float, float]:
    n = D.shape[0]
    same = label_idx[:, None] == label_idx[None, :]
    off_diag = ~np.eye(n, dtype=bool)
    within = D[same & off_diag].mean()
    between = D[~same & off_diag].mean()
    gap = (between - within) / between
    return float(within), float(between), float(gap)


def knn_purity(D: np.ndarray, label_idx: np.ndarray, k: int = 5) -> float:
    """Mean fraction of each point's k nearest neighbours sharing its label."""
    n = D.shape[0]
    D_self = D.copy()
    np.fill_diagonal(D_self, np.inf)
    nn = np.argsort(D_self, axis=1)[:, :k]
    matches = (label_idx[nn] == label_idx[:, None]).mean(axis=1)
    return float(matches.mean())


def run_one_layer(model, layer: int, prompts: list[str], label_idx: np.ndarray, device: str):
    hook_name = f"blocks.{layer}.hook_resid_post"
    result = extract_final_position(
        model, prompts, hook_name=hook_name, device=device, batch_size=64,
    )
    order = np.argsort(result.prompt_ids)
    X = np.asarray(result.x)[order]
    n = X.shape[0]

    D_u = pairwise_cos_dist(X)
    D_c = pairwise_cos_dist(X - X.mean(axis=0, keepdims=True))

    iu = np.triu_indices(n, k=1)
    rho, _ = spearmanr(D_u[iu], D_c[iu])

    return {
        "X": X, "D_u": D_u, "D_c": D_c, "rho": float(rho),
        "wb_u": within_between(D_u, label_idx),
        "wb_c": within_between(D_c, label_idx),
        "knn_u": {k: knn_purity(D_u, label_idx, k) for k in (3, 5, 9)},
        "knn_c": {k: knn_purity(D_c, label_idx, k) for k in (3, 5, 9)},
    }


def main() -> None:
    # NOTE: TL+transformers version mismatch breaks Pythia loading
    # (`GPTNeoXConfig` lost `rotary_pct`). GPT-2 small works and gives a
    # clean diagnostic for the same claim (centering reorganises neighbours
    # and aligns with semantic structure).
    model_name = "gpt2"
    layers = [0, 2, 4, 6, 8, 10]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cat_list = list(CATEGORIES.keys())
    prompts: list[str] = []
    label_idx_list: list[int] = []
    for ci, cat in enumerate(cat_list):
        for w in CATEGORIES[cat]:
            prompts.append(f"The {w}")
            label_idx_list.append(ci)
    label_idx = np.asarray(label_idx_list)
    n = len(prompts)

    print(f"Loading {model_name} on {device}...")
    model = tl.HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, dtype=torch.float32,
    )
    model.eval()
    print(f"  d_model={model.cfg.d_model}  n_layers={model.cfg.n_layers}")

    print(f"\nSweeping layers {layers}, {n} prompts, \"The {{noun}}\" template...\n")
    print(f"{'layer':>5}  {'Spearman':>9}  "
          f"{'gap_u':>6}  {'gap_c':>6}  "
          f"{'knn3_u':>6}  {'knn3_c':>6}  "
          f"{'knn9_u':>6}  {'knn9_c':>6}")
    print("-" * 70)

    results: dict[int, dict] = {}
    for L in layers:
        r = run_one_layer(model, L, prompts, label_idx, device)
        results[L] = r
        print(
            f"{L:>5}  {r['rho']:>9.3f}  "
            f"{r['wb_u'][2] * 100:>5.1f}%  {r['wb_c'][2] * 100:>5.1f}%  "
            f"{r['knn_u'][3]:>6.3f}  {r['knn_c'][3]:>6.3f}  "
            f"{r['knn_u'][9]:>6.3f}  {r['knn_c'][9]:>6.3f}"
        )

    # Pick the layer with the biggest k-NN(k=9) lift from centering for the heatmap figure.
    best_layer = max(layers, key=lambda L: results[L]["knn_c"][9] - results[L]["knn_u"][9])
    r = results[best_layer]
    D_u, D_c = r["D_u"], r["D_c"]
    rho, (_, _, g_u), (_, _, g_c) = r["rho"], r["wb_u"], r["wb_c"]
    p9_u, p9_c = r["knn_u"][9], r["knn_c"][9]

    print(f"\nMost dramatic centering effect at layer {best_layer} (k-NN k=9 lift)")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    titles = [
        f"Uncentered\nk-NN(9) purity: {p9_u:.2f}",
        f"Centered\nk-NN(9) purity: {p9_c:.2f}",
    ]
    boundaries = np.cumsum([len(CATEGORIES[c]) for c in cat_list])[:-1]
    midpoints = [0] + boundaries.tolist() + [n]
    centers = [(midpoints[i] + midpoints[i + 1] - 1) / 2 for i in range(len(midpoints) - 1)]

    for ax, D, title in zip(axes, [D_u, D_c], titles):
        im = ax.imshow(D, cmap="viridis")
        ax.set_title(title, fontsize=12)
        for b in boundaries:
            ax.axhline(b - 0.5, color="white", linewidth=0.7)
            ax.axvline(b - 0.5, color="white", linewidth=0.7)
        ax.set_xticks(centers)
        ax.set_xticklabels(cat_list, fontsize=10)
        ax.set_yticks(centers)
        ax.set_yticklabels(cat_list, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"{model_name}, layer {best_layer} resid_post — final-position activations on \"The {{noun}}\"\n"
        f"Pairwise cosine distance grouped by category. "
        f"Spearman(centered, uncentered) = {rho:.3f}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = REPO_ROOT / "docs" / "figures" / "fig_centering_semantic.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved heatmap figure to {out}")

    # Layer-sweep plot: how does centering's effect scale with depth?
    fig2, ax = plt.subplots(figsize=(8, 5))
    Ls = list(results.keys())
    ax.plot(Ls, [results[L]["knn_u"][9] for L in Ls], "o-", label="uncentered", color="C3")
    ax.plot(Ls, [results[L]["knn_c"][9] for L in Ls], "o-", label="centered", color="C2")
    ax2 = ax.twinx()
    ax2.plot(Ls, [results[L]["rho"] for L in Ls], "s--", color="gray", alpha=0.7, label="Spearman ρ")
    ax.set_xlabel("layer (resid_post)", fontsize=11)
    ax.set_ylabel("k-NN(9) category purity", fontsize=11)
    ax2.set_ylabel("Spearman ρ (uncentered, centered)", fontsize=11, color="gray")
    ax.set_title(
        f"{model_name} — centering's effect by layer\n"
        f"\"The {{noun}}\" prompts, 5 categories × 10 words", fontsize=12)
    ax.legend(loc="lower left")
    ax2.legend(loc="lower right")
    ax.grid(alpha=0.2)
    out2 = REPO_ROOT / "docs" / "figures" / "fig_centering_layer_sweep.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"Saved sweep figure to {out2}")


if __name__ == "__main__":
    main()
