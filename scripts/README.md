# Scripts

Forty scripts, grouped by purpose. Most have a top-of-file docstring and a `Run:` line; this index is for finding the right one fast.

Every script is invoked as a module from the repo root:

```bash
python -m scripts.<name> [flags]
```

The figure-makers all write into `figures/` by default.

## Build & evaluate dictionaries

| Script                 | What it does                                                                                                                          |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `build_partitions.py`  | The main entrypoint. Stream Pile activations, calibrate threshold, grow a dictionary, optionally run SAEBench / AxBench evals.        |
| `compare_sae.py`       | Cross-decomposition F1: per-EP-partition F1 against the best-matching Gemma Scope SAE feature, and vice versa. Used for §6 + app. §B. |
| `label_dictionary.py`  | Generate human-readable labels per partition from sample prompts via Anthropic API.                                                   |
| `match_dictionaries.py`| Bipartite-match partitions across two dictionaries by exemplar similarity. Used for §9 + app. §A.3 + app. §A.6 (cross-checkpoint), and §A.7 (cross-seed stability). |
| `aggregate_reanchor.py`| Reanchor cells under a results root and tabulate `{mean, exemplar, exemplar_reanchored}` × `{p, seed}` for ablation-Δ comparison. |
| `run_all.py`           | Convenience wrapper: build a sweep of percentiles for one (model, layer), optionally with eval.                                       |

## Paper experiments

These produce the JSON / NPZ inputs that the figure-makers below consume.

| Script                            | Paper section                | Topic                                                                                                                        |
|-----------------------------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `exp_saturation.py`               | §8 + app. §A.1               | Online partition growth on math / code / chat streams — does dictionary size saturate?                                       |
| `exp_resolution_paths.py`         | §3 + app. §A.5               | Pick two partitions in a coarse dictionary; trace the shortest path between them in finer-resolution dictionaries.           |
| `exp_coverage.py`                 | §7 + app. §C                 | Per (dictionary, corpus): within-threshold rate and mean nearest-exemplar distance — OOD geometry.                           |
| `exp_seed_stability.py`           | §2.2 + app. §A.7             | Test whether mean partition directions are stable across random seeds.                                                       |
| `exp_per_region_stability.py`     | §2.2 + app. §A.7             | Per-region (not aggregate) stability across builds.                                                                          |
| `exp_stability_predictor.py`      | §2.2 + app. §A.7             | Whether $D_i = \log_{10}(N_i c_i^2)$ predicts cross-seed region stability.                                                   |
| `exp_behavioral.py`               | §4.1 + app. §A.2             | Refusal collapse. Build EP on AdvBench + Alpaca at L20 of `gemma-2-2b-it`, score partitions by member refusal rate, ablate the top one. |

## Exploratory experiments (not in the published paper)

These ran but didn't make the final paper, or only appear as future-work directions in §E. Kept in-tree because they're useful starting points for follow-up work.

| Script                            | Status                                                                                                          |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `exp_trajectories.py`             | Cross-layer Sankey. Future-work direction at §E (line 804); no figure in main paper.                            |
| `exp_trajectories_analysis.py`    | Post-process for the above.                                                                                     |
| `exp_alignment.py`                | Geometric--behavioural alignment. Future-work direction at §E (line 794).                                       |
| `exp_partition_steering.py`       | Steer the model along a single partition's exemplar direction. Not in paper.                                    |
| `exp_concept_steering.py`         | Steering along supervised concept-difference directions, head-to-head with EP. Not in paper.                    |
| `exp_patching.py`                 | Activation patching across prompt pairs that share structure but differ in one feature. Not in paper.           |
| `exp_category_firing.py`          | Do prompts in the same category land in geometrically-close partitions? Not in paper.                           |
| `aggregate_reanchor.py`           | Reanchor-cell ablation-Δ comparison. Not in paper.                                                              |

## Figures

Each `make_fig_*` reads JSON / NPZ produced by an `exp_*` (or a dictionary directly) and writes one figure. Marked **(exploratory)** if the corresponding experiment isn't in the published paper.

| Figure script                          | Reads from                            | Paper section / topic                                              |
|----------------------------------------|---------------------------------------|--------------------------------------------------------------------|
| `make_fig_saturation.py`               | `exp_saturation.py`                   | §8 + app. §A.1: saturation curves                                  |
| `make_fig_resolution_paths.py`         | `exp_resolution_paths.py`             | §3 + app. §A.5: path through finer resolutions                     |
| `make_fig_resolution_voronoi.py`       | `exp_resolution_paths.py`             | §3 + app. §A.5: 2D Voronoi panels of the same path                 |
| `make_fig_coverage.py`                 | `exp_coverage.py`                     | §7 + app. §C: OOD coverage                                         |
| `make_fig_compare_sae.py`              | `compare_sae.py`                      | §6 + app. §B: EP↔SAE F1 match                                      |
| `make_fig_refusal.py`                  | `exp_behavioral.py`                   | §4.1 + app. §A.2: refusal-collapse Δ per percentile                |
| `make_fig_neighbourhood.py`            | a dictionary                          | §3: top-k cosine neighbours + logit-lens labels per anchor         |
| `make_fig_shared_neighbours.py`        | a dictionary                          | §3: ASCII tree of cells appearing in top-K of two anchors          |
| `make_fig_lens_voronoi_tikz.py`        | a dictionary                          | §3: TikZ source for paper-quality lens-Voronoi panels              |
| `make_fig_shared_lens.py`              | a dictionary                          | §3: shared-lens projection across resolutions                      |
| `make_fig_shared_lens_tikz.py`         | a dictionary                          | §3: TikZ source for the same                                       |
| `make_fig_centering.py`                | (self-contained)                      | Toy showing centred vs uncentred unit-norm geometry                |
| `make_fig_trajectories.py`             | `exp_trajectories.py`                 | **(exploratory)** cross-layer alluvial; not in paper               |
| `make_fig_partition_steering.py`       | `exp_partition_steering.py`           | **(exploratory)** steering response curves; not in paper           |

## Diagnostics and utilities

| Script                            | What it does                                                                                       |
|-----------------------------------|----------------------------------------------------------------------------------------------------|
| `check_centering_semantics.py`    | Sanity check: do `(x - center) / ||x - center||` and `x / ||x||` give the same cosine geometry? (No — that's the point.) |
| `sphere_voronoi.py`               | Standalone 3D-PCA spherical Voronoi plotter with logit-lens labels. Used for the splash figure.    |
| `mib_fast_test.py`                | MCQA on Gemma-2-2b at one layer with DBM+EP plus full-vector baselines.                            |

## One-shot migration scripts (kept for provenance)

These ran once on 2026-05-16 to repair the post-rename HF dataset. They are kept in-tree so the history of the public dataset is reproducible, but should not need to run again.

| Script                       | What it did                                                                                  |
|------------------------------|----------------------------------------------------------------------------------------------|
| `repickle_hub.py`            | Re-pickled all 13 HF dictionary blobs under the `ep.*` namespace (post `cas` → `ep` rename). |
| `patch_hub_metadata.py`      | Renamed `wandb_project: cas` → `wandb_project: ep` in the metadata JSONs.                    |
