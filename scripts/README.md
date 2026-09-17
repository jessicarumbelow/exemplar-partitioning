# Scripts

Scripts are grouped by purpose. Most have a top-of-file docstring and a `Run:` line; this index is for finding the right one fast.

Every script is invoked as a module from the repo root:

```bash
python -m scripts.<name> [flags]
```

The figure-makers all write into `figures/` by default.

## Build & evaluate dictionaries

| Script                 | What it does                                                                                                                          |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `build_partitions.py`  | The main entrypoint. Stream Pile activations, calibrate threshold, grow a dictionary, optionally run SAEBench / AxBench evals.        |
| `compare_sae.py`       | Cross-decomposition F1: per-EP-partition F1 against the best-matching Gemma Scope SAE feature, and vice versa. Used for §6 + app. §B.  |
| `label_dictionary.py`  | Generate human-readable labels per partition from sample prompts via Anthropic API.                                                   |
| `match_dictionaries.py`| Bipartite-match partitions across two dictionaries by exemplar similarity. Used for §2.2 + app. §A.4 (cross-seed stability). |

## Paper experiments

These produce the JSON / NPZ inputs that the figure-makers below consume.

| Script                            | Paper section                | Topic                                                                                                                        |
|-----------------------------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `exp_saturation.py`               | §6 + app. §A.1               | Online partition growth on math / code / chat streams — does dictionary size saturate?                                       |
| `exp_resolution_paths.py`         | app. §A.3                    | Pick two partitions in a coarse dictionary; trace the shortest path between them in finer-resolution dictionaries.           |
| `exp_coverage.py`                 | Exploratory; not in revised paper | Measure within-threshold rates and nearest-exemplar distances; these do not establish an OOD detector. |
| `exp_seed_stability.py`           | §2.2 + app. §A.4             | Test whether mean partition directions are stable across random seeds.                                                       |
| `exp_per_region_stability.py`     | §2.2 + app. §A.4             | Per-region (not aggregate) stability across builds.                                                                          |
| `exp_stability_predictor.py`      | §2.2 + app. §A.4             | Whether $D_i = \log_{10}(N_i c_i^2)$ predicts cross-seed region stability.                                                   |
| `prepare_region_ablation.py`      | §4 + app. §E                  | Recreate the paper's held-out prompt split from a user-supplied harmful/benign prompt JSON. |
| `exp_region_ablation.py`          | §4 + app. §E                  | Test the centroid-span swap, controls, and layer sweep from the causal intervention section. Requires local prompt and activation data. |
| `exp_taboo.py`                    | §5 + app. §G                  | Generate Taboo hint transcripts and build fine-tuned/base dictionaries. |
| `exp_taboo_control.py`            | §5 + app. §G                  | Extract the two fixed assistant-prefix activations used by the 21-organism result. |
| `exp_taboo_inventory.py`          | §5 + app. §G                  | List all regions with fine-tuned support and zero base support; optionally score secret ranks after construction. |
| `exp_taboo_audit.py`              | §5 + app. §G                  | Compare transcript-only and transcript-plus-region secret auditors. |
| `exp_resolution_separation.py`    | §6 + app. §H                  | Run the ordered-colour toy and report when coarse identities split into ordered pairs. |
| `verify_text_baseline.py`         | §4 + app. §E                  | TF--IDF $k$-means control for harmful/benign separation. Accepts a locally constructed prompt JSON; the harmful corpus is not distributed. |
| `build_crossfamily_cache.py`      | §4 + app. §E                  | Build the per-layer assignment cache from a saved `(prompts, layers, hidden_dim)` activation array, with no hard-coded scratch paths. |
| `verify_crossfamily.py`           | §4 + app. §E                  | Recompute the 100%-harmful round-trip grid from the four assignment caches without model inference. |

## Figures

Each `make_fig_*` reads JSON / NPZ produced by an `exp_*` (or a dictionary directly) and writes one figure.

| Figure script                          | Reads from                            | Paper section / topic                                              |
|----------------------------------------|---------------------------------------|--------------------------------------------------------------------|
| `make_fig_saturation.py`               | `exp_saturation.py`                   | §6 + app. §A.1: saturation curves                                  |
| `make_fig_compare_sae.py`              | `compare_sae.py`                      | §6 + app. §B: EP↔SAE F1 match                                      |
| `make_fig_neighbourhood.py`            | a dictionary                          | app. §A.2: top-k cosine neighbours + logit-lens labels per anchor  |
| `make_fig_shared_neighbours.py`        | a dictionary                          | app. §A.3: ASCII tree of cells appearing in top-K of two anchors   |
| `make_fig_lens_voronoi_tikz.py`        | a dictionary                          | app. §A.3: TikZ source for paper-quality lens-Voronoi panels       |
| `make_fig_shared_lens_tikz.py`         | a dictionary                          | app. §A.3: TikZ source for the same                                |

## Diagnostics and utilities

| Script                            | What it does                                                                                       |
|-----------------------------------|----------------------------------------------------------------------------------------------------|
| `sphere_voronoi.py`               | Standalone 3D-PCA spherical Voronoi plotter with logit-lens labels. Used for the splash figure.    |
