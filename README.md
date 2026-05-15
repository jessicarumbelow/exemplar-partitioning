# Exemplar Partitioning

We introduce Exemplar Partitioning (EP), an unsupervised method for constructing interpretable feature dictionaries from Large Language Model (LLM) activations with ~10³× fewer tokens than comparable sparse autoencoders. An EP dictionary is a Voronoi partition of activation space, built by leader-clustering streamed activations within a distance threshold. Each region is anchored by an observed exemplar that serves as both its membership criterion and intervention direction; dictionary size is not prespecified, but determined by the activation geometry at that threshold. Because exemplars are observed rather than learned, dictionaries built from the same data stream are directly comparable across layers, models, and training checkpoints.

This paper characterises EP as an interpretability object via targeted demonstrations of properties newly accessible through this construction, plus one head-to-head benchmark. In Gemma-2-2b, we find that EP dictionary regions are interpretable and support causal interventions: refusal in instruction-tuned Gemma concentrates in a region whose exemplar ablation can collapse held-out refusal. Cross-checkpoint matching between base and instruction-tuned dictionaries separates the directions preserved through finetuning from those introduced by it. EP regions and Gemma Scope SAE features decompose activation space differently, but agree on a shared core: ~20% of EP regions match an SAE feature at F₁ > 0.5, and EP one-hot probes retain ~97% of raw-activation probe accuracy at ℓ₀ = 1: the linearly-decodable identity that probing tests is largely preserved by density structure alone. Nearest-exemplar distance provides a free out-of-distribution signal at inference. On AxBench latent concept detection at Gemma-2-2B-it L20, EP at p₁ reaches mean AUROC 0.881, +0.126 over the canonical GemmaScope SAE leaderboard entry and within 0.030 of SAE-A's 0.911, at ~10³× less build compute. Code: [github.com/jessicarumbelow/exemplar-partitioning](https://github.com/jessicarumbelow/exemplar-partitioning).

> **Paper:** ["Exemplar Partitioning for Mechanistic Interpretability"](https://arxiv.org/abs/2605.14347) (arXiv:2605.14347).
>
> **Pretrained dictionaries:** [`J-RUM/exemplar-partitioning`](https://huggingface.co/datasets/J-RUM/exemplar-partitioning) on HuggingFace — Gemma-2-2B and Gemma-2-2B-it at L4/L12/L20, percentiles $p \in \{1, 2, 4, 8, 10\}$.

## Install

```bash
pip install -e .                # core
pip install -e ".[sae]"         # + SAE comparison baselines (sae-lens, transformer-lens)
```

Python ≥ 3.11. CUDA optional but recommended for any model larger than ~160M.

## Quickstart

```python
import ep
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gemma-2-2b", device="cuda")
texts = [...]  # any iterable of strings — Pile, your own corpus, etc.
hook  = "blocks.12.hook_resid_post"

# 1. Calibrate: choose a distance threshold from activation geometry.
calibration = ep.calibrate_pipeline(
    model, texts, hook,
    n_tokens=200_000, percentile=10.0,
)

# 2. Discover: stream activations, grow the dictionary.
result = ep.discover(
    model, texts, hook, calibration,
    max_tokens=10_000_000,
)
dictionary = result.dictionary
print(dictionary)
# → Dictionary(5129 partitions, 4231 with ≥2 members, θ=0.0832, ||center||=2.3001)

# 3. Use it. Assign new activations to their nearest partition:
import numpy as np
new_activations = np.random.randn(100, 2304).astype(np.float32)
partition_ids, distances = dictionary.assign(new_activations)
```

The same `distances` are also your free OOD signal — large distance to the nearest exemplar means the activation falls outside the training distribution. Computation runs wherever the model lives; CUDA is detected automatically.

## CLI

For full reproducibility runs (Pile streaming, SAEBench / AxBench evals), use the scripts:

```bash
# Build a dictionary at the canonical p=10 setting.
python -m scripts.build_partitions \
    --model google/gemma-2-2b --layer 12 \
    --percentile 10 --max-tokens 10_000_000

# Add a SAEBench evaluation.
python -m scripts.build_partitions \
    --model google/gemma-2-2b --layer 12 \
    --percentile 10 --eval sparse_probing

# Run all SAEBench evals.
python -m scripts.build_partitions --eval all
```

Useful flags: `--model`, `--layer`, `--percentile`, `--max-tokens`, `--max-prompts`, `--seed`, `--device`, `--extractor {per-position,final-position}`, `--merge-close`.

## What's in a dictionary

```python
dictionary.partitions                       # list[Partition]
partition.exemplar_direction                # (dim,) unit vector in centred space —
                                            #   the first-arrival activation that
                                            #   created this partition
partition.mean_member_direction             # (dim,) spherical mean of all member
                                            #   directions, renormalised
partition.member_count                      # int
partition.member_coherence                  # float in [0, 1]; 1 = all members
                                            #   point the same way
partition.sample_prompts                    # heap of (-dist, prompt, pos) — the
                                            #   closest prompts to this partition
partition.boundary_prompts                  # heap of (dist, prompt, pos) — the
                                            #   farthest prompts in this partition
```

You can swap `exemplar_direction` for `mean_member_direction` as the partition's representative direction and get a slightly different downstream signal (the paper compares both at AxBench in §4.1).

## Repository layout

```
ep/                       # The package
├── discovery/
│   ├── extraction.py        # Activation extractors (per-position, final-position)
│   ├── dictionary.py        # Streaming exemplar-partition dictionary
│   ├── pipeline.py          # calibrate_pipeline, discover
│   ├── calibration.py       # Threshold calibration + on-disk cache
│   ├── eval.py              # Intrinsic dictionary metrics
│   └── geometry.py          # Centred unit-norm primitives + GPU detection
├── saebench_adapter.py      # SAEBench-compatible EPDictionarySAE
├── mib_adapter.py           # MIB Featurizer wrapper
└── utils.py                 # set_seed

scripts/                  # Build, evaluate, and reproduce the paper figures
tests/                    # pytest suite (run `pytest`)
```

## Tests

```bash
pytest                 # ~30s, no GPU required
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
