# Exemplar Partitioning

We introduce Exemplar Partitioning (EP), an unsupervised method for constructing interpretable feature dictionaries from Large Language Model (LLM) activations with ~10³× fewer tokens than comparable sparse autoencoders. An EP dictionary is a Voronoi partition of centered, unit-norm activation space, built by leader-clustering streamed activations within a cosine-distance threshold. Each region is anchored by an observed exemplar that serves as both its membership criterion and intervention direction; dictionary size is not prespecified, but determined by the activation geometry at that threshold. Because exemplars are observed rather than learned, dictionaries built from the same data stream are directly comparable across layers, models, and training checkpoints.

This paper characterises EP as an interpretability object via targeted demonstrations of properties newly accessible through this construction, plus one head-to-head benchmark. In Gemma-2-2b, we find that EP dictionary regions are interpretable and support causal interventions: refusal in instruction-tuned Gemma concentrates in a region whose exemplar ablation can collapse held-out refusal. Cross-checkpoint matching between base and instruction-tuned dictionaries separates the directions preserved through finetuning from those introduced by it. EP regions and Gemma Scope SAE features decompose activation space differently, but agree on a shared core: ~20% of EP regions match an SAE feature at F₁ > 0.5, and EP one-hot probes retain ~97% of raw-activation probe accuracy at ℓ₀ = 1: the linearly-decodable identity that probing tests is largely preserved by density structure alone. Nearest-exemplar distance provides a free out-of-distribution signal at inference. On AxBench latent concept detection at Gemma-2-2B-it L20, EP at p₁ reaches mean AUROC 0.881, +0.126 over the canonical GemmaScope SAE leaderboard entry and within 0.030 of SAE-A's 0.911, at ~10³× less build compute. Code: [github.com/jessicarumbelow/exemplar-partitioning](https://github.com/jessicarumbelow/exemplar-partitioning).

> **Paper:** ["Exemplar Partitioning for Mechanistic Interpretability"](https://arxiv.org/abs/2605.14347) (arXiv:2605.14347).
>
> **Prebuilt dictionaries:** [`J-RUM/exemplar-partitioning`](https://huggingface.co/datasets/J-RUM/exemplar-partitioning) on HuggingFace — Gemma-2-2B (L12 across $p \in \{1, 2, 4, 8, 10\}$, plus L20 at $p=10$) and Gemma-2-2B-it (L4 at $p=4$, L12 at $p=10$, L20 across $p \in \{1, 2, 4, 8, 10\}$). EP has no training step; the dictionaries are streamed partitions over Pile activations, distributed so you can skip the build. Files are Python pickles — verify the blob SHA on the dataset page before loading.

## Install

```bash
pip install -e .                # core (includes transformer-lens for live extraction)
pip install -e ".[sae]"         # + SAE comparison baselines (sae-lens, scikit-learn)
pip install -e ".[scripts]"     # + paper-figure / eval scripts (datasets, wandb, ...)
pip install -e ".[all]"         # everything
```

Python ≥ 3.11. CUDA optional but recommended for any model larger than ~160M.

## Quickstart: load a prebuilt dictionary

The fastest way in is to load one of the published Gemma-2-2B dictionaries — no model load, no build pass, ~10s on a fresh machine:

```python
import ep

# (model_short, layer, percentile). See "Prebuilt dictionaries" below for
# the full matrix.
d = ep.Dictionary.from_hub("gemma-2-2b", layer=12, percentile=10)
print(d)
# → Dictionary(203 partitions, 203 with ≥2 members, θ=0.8744, ||center||=88.7901)

# Inspect the largest partitions.
for p in sorted(d.partitions, key=lambda p: -p.member_count)[:3]:
    print(f"K={p.member_count}, coherence={p.member_coherence:.2f}")
    for dist, prompt, pos in p.closest_prompts[:3]:
        print(f"  d={dist:.3f}  pos={pos}  {prompt[:80]!r}")
```

Assign new activations to their nearest partition:

```python
import numpy as np
new_activations = np.random.randn(100, 2304).astype(np.float32)
partition_ids, distances = d.assign(new_activations)
```

`distances` doubles as a free OOD signal — a large distance to the nearest exemplar means the activation falls outside the training distribution.

## Build a dictionary from a live model

```python
import ep
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gemma-2-2b", device="cuda")
texts = [...]  # any iterable of strings — Pile, your own corpus, etc.
hook  = "blocks.12.hook_resid_post"
extract_fn = ep.extract_per_position  # also: ep.extract_final_position

# 1. Calibrate: choose a distance threshold from activation geometry.
#    `percentile` is the p-th percentile of within-batch pairwise cosine
#    distances after centering — smaller p = tighter cells, more partitions.
calibration = ep.calibrate_pipeline(
    model, texts, hook,
    n_tokens=200_000, percentile=10.0,
    extract_fn=extract_fn,
)

# 2. Discover: stream activations, grow the dictionary.
result = ep.discover(
    model, texts, hook, calibration,
    max_tokens=10_000_000,
    extract_fn=extract_fn,
)
dictionary = result.dictionary
```

Computation runs wherever the model lives; CUDA is detected automatically.

**Calibration and discovery must use the same extractor.** The threshold is calibrated against the distribution of activations the extractor produces; mixing per-position calibration with final-position discovery (or different context lengths) silently produces meaningless cells. The CLI handles this for you; in Python, pass the same `extract_fn` to both calls.

If you want the calibration cached across runs, use `ep.load_or_calibrate` instead of `ep.calibrate_pipeline`. The cache key is `(model_name, hook_name, percentile, extras)` — pass any extractor- or sampling-specific knobs (e.g. `extras={"extractor": "per-position", "ctx": 128}`) so two calibrations with different settings don't share a cache slot. The CLI uses `extras={"extractor", "sampling", "ctx"}` by default; match those keys to reuse its cache from Python.

## CLI

For full reproducibility runs (Pile streaming, SAEBench / AxBench evals), use the scripts:

```bash
# Build a dictionary at the canonical p=10 setting.
python -m scripts.build_partitions \
    --model google/gemma-2-2b --layer 12 \
    --percentile 10 --max-tokens 10_000_000

# Reproduce the headline AxBench AUROC (Gemma-2-2B-it L20 p=1).
python -m scripts.build_partitions \
    --model google/gemma-2-2b-it --model-short gemma-2-2b-it \
    --layer 20 --percentile 1 --max-tokens 10_000_000 \
    --eval axbench

# SAEBench sparse-probing eval.
python -m scripts.build_partitions \
    --model google/gemma-2-2b --layer 12 \
    --percentile 10 --eval sparse_probing
```

The eval pathways need third-party repos checked out under `baselines/` — the script prints the exact `git clone` command on first invocation. SAEBench: `https://github.com/adamkarvonen/SAEBench`. AxBench: `https://github.com/stanfordnlp/axbench`.

Useful flags: `--model`, `--layer`, `--percentile`, `--max-tokens`, `--max-prompts`, `--seed`, `--device`, `--extractor {per-position,final-position}`.

## Prebuilt dictionaries

`Dictionary.from_hub` pulls from [`J-RUM/exemplar-partitioning`](https://huggingface.co/datasets/J-RUM/exemplar-partitioning). The matrix:

| Model            | Layer | Percentiles    |
|------------------|-------|----------------|
| `gemma-2-2b`     | 12    | 1, 2, 4, 8, 10 |
| `gemma-2-2b`     | 20    | 10             |
| `gemma-2-2b-it`  | 4     | 4              |
| `gemma-2-2b-it`  | 12    | 10             |
| `gemma-2-2b-it`  | 20    | 1, 2, 4, 8, 10 |

## What's in a dictionary

```python
dictionary.partitions                       # list[Partition]
dictionary.center                           # (dim,) activation centroid from calibration
dictionary.threshold                        # cosine distance threshold (scalar)

partition.exemplar_direction                # (dim,) unit vector — the centered, L2-
                                            #   normalised form of the first-arrival
                                            #   activation that created this partition
partition.mean_member_direction             # (dim,) spherical mean of member directions
partition.member_count                      # int
partition.member_coherence                  # float in [0, 1]; 1 = all members agree
partition.closest_prompts                   # list of (dist, prompt, position) — closest first
partition.farthest_prompts                  # list of (dist, prompt, position) — farthest first
```

A partition has two candidate "representative directions". `exemplar_direction` is the first-arrival activation that anchored the cell — observed, immutable, and the direction used for intervention examples below. `mean_member_direction` is the spherical mean of everything assigned to the cell — a denoised consensus that drifts as the cell fills. The paper benchmarks both at AxBench in §4.1; neither dominates uniformly, so default to `exemplar_direction` for causal interventions (you know exactly what you're injecting) and `mean_member_direction` for read-out / probing (lower variance).

### Intervention with an exemplar

`exemplar_direction` lives in the centered, unit-norm space the dictionary clusters in. To inject (or ablate) a partition in the model's raw activation space, undo the centering and pick a scale that matches the layer's typical activation norm:

```python
import torch
p = d.partitions[42]
e = torch.from_numpy(p.exemplar_direction)
c = torch.from_numpy(d.center)

# Add a centroid-scale push along the exemplar at hook time:
def steer(act, hook, alpha=float(torch.linalg.norm(c))):
    return act + alpha * e.to(act.device, act.dtype)
model.add_hook("blocks.12.hook_resid_post", steer, "fwd")

# To ablate the partition's direction instead, project it out:
def ablate(act, hook):
    x = (act - c.to(act)).to(torch.float32)
    proj = (x @ e.to(act.device, torch.float32))[..., None] * e.to(act.device, torch.float32)
    return (x - proj).to(act.dtype) + c.to(act)
```

The paper's refusal-collapse result (§4.3) uses exactly this ablation pattern on the partition whose exemplar matches the refusal direction in Gemma-2-2B-it L20. To reproduce end-to-end — build the dictionary, score partitions by member refusal rate, ablate the top one on a held-out harmful set — run [`scripts/exp_behavioral.py`](scripts/exp_behavioral.py); the per-percentile plotting (`make_fig_refusal.py`) consumes its JSON outputs.

## Repository layout

```
ep/                       # The package
├── discovery/
│   ├── extraction.py        # Activation extractors (per-position, final-position)
│   ├── dictionary.py        # Streaming exemplar-partition dictionary (+ from_hub)
│   ├── pipeline.py          # calibrate_pipeline, discover
│   ├── calibration.py       # Threshold calibration + on-disk cache
│   ├── eval.py              # Intrinsic dictionary metrics
│   └── geometry.py          # Centred unit-norm primitives + GPU detection
├── saebench_adapter.py      # SAEBench-compatible EPDictionarySAE wrapper
├── saebench_sota.py         # Cached SAEBench leaderboard numbers for headline tables
├── mib_adapter.py           # Featurizer / inverse-featurizer for MIB causal-variable track
└── utils.py                 # set_seed

scripts/                  # Build, evaluate, and reproduce the paper figures
tests/                    # pytest suite (run `pytest`)
```

## Tests

```bash
pytest                 # ~10s, no GPU required
```

## Citation

```bibtex
@misc{rumbelow2026exemplar,
  title         = {Exemplar Partitioning for Mechanistic Interpretability},
  author        = {Rumbelow, Jessica},
  year          = {2026},
  eprint        = {2605.14347},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.14347},
}
```

## License

MIT.
