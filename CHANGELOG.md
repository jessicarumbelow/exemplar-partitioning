# Changelog

All notable changes to this project will be documented here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `calibrate_pipeline` now accepts `cache_model_name` / `cache_extras` / `force_recalibrate`, so caching across runs no longer requires manually constructing an `activation_batches_fn` for `load_or_calibrate`.
- `try_torch_gpu` now picks up Apple Silicon (MPS) when CUDA is unavailable. Disable with `EP_FORCE_CPU=1`.
- `--wandb-entity` flag on `build_partitions.py` and `compare_sae.py` (was a hardcoded entity, now defaults to your wandb auth).
- `--readout-override` choices include `signed_norm` (was missing from the CLI even though the adapter exposed it).
- `partition.label` documented in the README "What's in a dictionary" section.

### Fixed
- Cosine distances clamped to `[0, 2]` at every entry point (`assign`, `distances`, `_nearest_exemplar`, `closest_prompts`, `farthest_prompts`) so float-noise on near-parallel unit vectors no longer produces displayed `d=-0.000` artifacts.
- Calibration percentile uses `torch.quantile` (matching `np.percentile` semantics) instead of `torch.kthvalue` when input fits the 16M element cap. Was a latent CPU/GPU disagreement that MPS exposed.
- `Dictionary.from_hub` docstring no longer overclaims "full p sweep" for `gemma-2-2b-it` L4/L12 (only L20 has the full sweep).
- `pyproject.toml` comment referenced removed `from_pretrained`; now points at `from_hub`.
- `extract_per_position` docstring now explains that `n_tokens` counts BOS even though BOS positions are skipped from extraction.
- `saebench_adapter` module docstring lists all six readouts (`topk`, `signed`, `cosine`, `signed_norm`, `topk_norm`, `binary`), not just three.

### Removed
- `Run on Modal:` recipes from every `scripts/exp_*.py` docstring. The referenced Modal driver lives outside the public repo, so the recipes were dead ends for fresh researchers. Local invocation is the only documented path now.
- Hardcoded personal wandb entity from `build_partitions.py`, `compare_sae.py`, and all `exp_*.py` scripts.
- Internal Modal-volume comments from `build_partitions.py` and `_axbench_evaluate.py` docstrings (replaced with filesystem-agnostic wording).

## [0.1.0-post1] – 2026-05-16

### Fixed
- `Dictionary.from_hub` raised `ModuleNotFoundError: No module named 'cas'` on a fresh install because the published HuggingFace blobs were pickled before the 2026-05-03 `cas` → `ep` rename. Two-part fix:
  - All 13 prebuilt dictionaries on [`J-RUM/exemplar-partitioning`](https://huggingface.co/datasets/J-RUM/exemplar-partitioning) re-pickled under the `ep.*` namespace and re-uploaded.
  - `_LegacyCASCompatUnpickler` added inside `from_hub` so any stale local caches (and users pinned to older `ep` versions) still deserialise.
- README "Prebuilt dictionaries" matrix corrected to match what's actually on HF (previously over-claimed the `gemma-2-2b-it` grid).
- README `__repr__` example output corrected to match the real L12 p10 build.
- README `pytest` timing corrected (`~30s` → `~10s`).
- `wandb_project` in 9 metadata.json files on HF renamed `cas` → `ep`.

### Added
- `plotly` added to the `[scripts]` extra — two figure scripts import it and it was missing.
- `scripts/repickle_hub.py`, `scripts/patch_hub_metadata.py` — one-shot migration scripts kept in-tree for provenance.
- README intervention example now includes `alpha`-scale guidance.
- CLI section reorganised by research goal (build / AxBench / SAEBench / resume).
- `notebooks/walkthrough.ipynb` — CPU-runnable tour of `from_hub`, partition inspection, OOD distance, and the intervention pattern.
- `scripts/README.md` — script-to-figure and script-to-paper-section map.
- This file.

## [0.1.0] – 2026-05-11

Initial public release. See the [paper](https://arxiv.org/abs/2605.14347) for method details.
