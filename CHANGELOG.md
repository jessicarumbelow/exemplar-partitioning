# Changelog

All notable changes to this project will be documented here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
