# Exemplar Partitioning

We introduce Exemplar Partitioning (EP), an unsupervised method for constructing interpretable feature dictionaries from Large Language Model (LLM) activations with ~10³× fewer tokens than comparable sparse autoencoders. An EP dictionary is a Voronoi partition of centered, unit-norm activation space, built by leader-clustering streamed activations within a cosine-distance threshold. Each region is anchored by an observed exemplar that serves as both its membership criterion and intervention direction; dictionary size is not prespecified, but determined by the activation geometry at that threshold. Because exemplars are observed rather than learned, dictionaries built from the same data stream are directly comparable across layers, models, and training checkpoints.

This paper characterises EP through targeted demonstrations and one head-to-head benchmark. On AxBench latent concept detection at Gemma-2-2B-it L20, EP at p₁ reaches mean AUROC 0.937 with the region mean as detector under SAE-A's own selection rule on AxBench's shipped held-out set, above SAE-A's 0.911 and +0.182 over the canonical GemmaScope SAE leaderboard entry, at ~10³× less build compute (0.881 with the exemplar under the token-mean contrast rule). Shared prompt exemplars also make independently built dictionaries directly comparable: mean round-trip correspondence between wholly harmful Gemma and Llama regions rises from 19% in the base models to 44% after instruction tuning. Across 21 Taboo organisms, inspecting every region with fine-tuned support and zero base support finds a human-readable secret region in 19 cases. The secret is used only to evaluate the list after construction; EP does not automatically select the right region. EP regions and Gemma Scope SAE features agree selectively: roughly 20% of EP regions have a strong SAE counterpart at p₁₀. Under the native one-hot readout, EP retains 98% of raw-activation top-1 probe accuracy and 82% of test accuracy at p₁₀; test-accuracy retention rises to 91% at p₁. Nearest-exemplar distance provides an out-of-distribution signal at inference. Code: [github.com/jessicarumbelow/exemplar-partitioning](https://github.com/jessicarumbelow/exemplar-partitioning).

> **Paper:** ["Exemplar Partitioning for Mechanistic Interpretability"](https://arxiv.org/abs/2605.14347) (arXiv:2605.14347).
>
> **Prebuilt dictionaries:** [`J-RUM/exemplar-partitioning`](https://huggingface.co/datasets/J-RUM/exemplar-partitioning) on HuggingFace — Gemma-2-2B (L12 across $p \in \{1, 2, 4, 8, 10\}$, plus L20 at $p=10$) and Gemma-2-2B-it (L4 at $p=4$, L12 at $p=10$, L20 across $p \in \{1, 2, 4, 8, 10\}$). EP has no training step; the dictionaries are streamed partitions over Pile activations, distributed so you can skip the build. Files are Python pickles — verify the blob SHA on the dataset page before loading.

The Gemma-2-2B-it L20 $p=0.5$ build reported in the paper is not in the dataset; reproduce it with `scripts.build_partitions`.

## Install

```bash
pip install -e .                # core (includes transformer-lens for live extraction)
pip install -e ".[sae]"         # + SAE comparison baselines (sae-lens, scikit-learn)
pip install -e ".[scripts]"     # + paper-figure / eval scripts (datasets, wandb, ...)
pip install -e ".[all]"         # everything
```

Python ≥ 3.11. CUDA optional but recommended for any model larger than ~160M.

## Quickstart: load a prebuilt dictionary

The fastest way in is to load one of the published Gemma-2-2B dictionaries — no model load, no build pass. The L12 p=10 build below is ~65 MB and downloads in seconds; the tighter p=1 builds reach 6 GB and take a few minutes on first download. For an interactive tour with cell outputs, open [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb).

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

To cache calibration across runs, pass `cache_model_name` to `calibrate_pipeline`:

```python
calibration = ep.calibrate_pipeline(
    model, texts, hook,
    n_tokens=200_000, percentile=10.0,
    extract_fn=extract_fn,
    cache_model_name="google/gemma-2-2b",
    cache_extras={"extractor": "per-position", "ctx": 128},
)
```

The cache key is `(cache_model_name, hook_name, percentile, cache_extras)` under `~/.cache/ep/calibration/` (override with `EP_CALIBRATION_CACHE`). Pass any extractor- or sampling-specific knobs in `cache_extras` so two calibrations with different settings don't share a slot. The CLI uses `{"extractor", "sampling", "ctx"}` by default — match those keys to reuse its cache from Python.

## CLI

For full reproducibility runs (Pile streaming, SAEBench / AxBench evals), use the scripts. `python -m scripts.build_partitions --help` lists every flag; the recipes below cover the common research goals.

**Just build a dictionary (no eval):**

```bash
python -m scripts.build_partitions \
    --model google/gemma-2-2b --layer 12 \
    --percentile 10 --max-tokens 10_000_000
```

The build flags that control what you get: `--model`, `--model-short` (alias used in output paths), `--layer`, `--percentile` (cell tightness — smaller = more partitions), `--max-tokens` (build budget), `--extractor {per-position,final-position}` (which activations to cluster), `--seed`.

**Reproduce the headline AxBench AUROC (§3.1; Gemma-2-2B-it L20 p=1):**

```bash
python -m scripts.build_partitions \
    --model google/gemma-2-2b-it --model-short gemma-2-2b-it \
    --layer 20 --percentile 1 --max-tokens 100_000_000 \
    --eval axbench --axbench-modes latent \
    --axbench-selection auroc \
    --axbench-act-cache-dir ~/.cache/ep/axbench-acts
```

The test rows are AxBench's shipped held-out set for this model and layer, so no LLM calls are needed. The p₂/p₄/p₈ rows of the table use `--max-tokens 10_000_000`. Rows scored on LLM-regenerated test texts (the contrast-rule table in appendix F) are archived at `J-RUM/exemplar-partitioning` (`axbench/gemma-2-2b-it_L20_p{1,2,4,8}_latent_data.parquet`) and can be replayed with `--axbench-latent-data`.

How the region is chosen per concept is `--axbench-selection`. `auroc` is AxBench's own SAE-A rule (`GemmaScopeSAEMaxAUC`): max cosine over the positions of each training sequence, then the region with the highest training-set AUROC. It is the protocol the paper reports. `contrast` is the token-mean rule reported in appendix F: mean cosine over all positive tokens minus mean over all negative tokens. Both representatives (exemplar and region mean) are scored in the same run; the selection always uses the representative that is then scored.

`--axbench-latent-data` replays the test rows from an earlier run's `inference/latent_data.parquet` verbatim. Without it the eval scores AxBench's own held-out set (`concept500/<config>/inference/latent_eval_data.parquet`), the same rows as the published leaderboard. The archived parquet above was produced by regenerating the test set through the OpenAI API instead. `--axbench-act-cache-dir` stores every residual activation the eval computes, keyed by token ids, so a second selection rule or representative reruns in minutes. `--axbench-dump-tag NAME` writes outputs to `axbench_NAME/` instead of `axbench/`, so reruns never overwrite an earlier result.

Other options: `--axbench-max-concepts` (smoke test on a prefix of the 500 concepts), `--axbench-steering-examples`, `--axbench-modes` (`latent,steering,steering_test`; the steering modes need OpenAI for the LM judge). Partition labelling needs an Anthropic API key: `--api-key-file path/to/key`.

**SAEBench sparse-probing eval (appendix §D):**

```bash
python -m scripts.build_partitions \
    --model google/gemma-2-2b --layer 12 \
    --percentile 10 --eval sparse_probing
```

Adds: `--eval sparse_probing` and `--readout-override`, `--readout-k`.

**Resume / inspect a previous run:**

`--build-only` stops after the dictionary is written; `--aggregate-only` skips the build and just runs the eval against an existing dictionary; `--force-rerun` ignores cached calibration and partial outputs.

**Eval prerequisites.** The eval pathways need third-party repos checked out under `baselines/` — the script prints the exact `git clone` command on first invocation. SAEBench: `https://github.com/adamkarvonen/SAEBench`. AxBench: `https://github.com/stanfordnlp/axbench`.

See [`scripts/README.md`](scripts/README.md) for the full script-to-figure / script-to-section map.

The Taboo, resolution-toy and cross-family experiments are also available as module entrypoints. The Taboo pipeline generates transcripts, extracts the two assistant control-token activations, then inventories every region with fine-tuned support and zero assigned base support:

```bash
python -m scripts.exp_taboo --help
python -m scripts.exp_taboo_control --help
python -m scripts.exp_taboo_inventory --help
python -m scripts.exp_taboo_audit --help
```

The inventory reads `results.json` and `acts_control.npz` from each `exp_taboo_control` run. Its default output lists candidate regions without using the secret. Pass `--evaluate-secret` to add secret embedding ranks for post-hoc evaluation. The saved [21-organism summary](results/summaries/taboo_21_summary.json) records the best post-hoc row for each organism; the [occupancy comparison](results/summaries/taboo_occupancy_comparison.json) records the older selector reported as a control in the paper.

```bash
python -m scripts.exp_taboo_inventory \
    --run-dirs outputs/blue_control,outputs/book_control \
    --output outputs/taboo_inventory.json --evaluate-secret
```

Run `python -m scripts.exp_resolution_separation --help` for the ordered-colour toy. For the causal intervention, prepare the prompt split with `python -m scripts.prepare_region_ablation --help`, then run `python -m scripts.exp_region_ablation --help` with your activation arrays and assignment cache. At L14, the paper's main intervention is `--conditions swap-c`; `--layers 8-20` reproduces the layer sweep. The harmful-prompt corpus and activation arrays are not distributed. Small saved summaries are in [`results/summaries/`](results/summaries/).

```bash
python -m scripts.prepare_region_ablation \
    --prompts-json data/prompts.json --output-dir outputs/region_split
python -m scripts.exp_region_ablation \
    --model google/gemma-2-2b-it --tag g_it --layer 14 \
    --acts data/acts_g_it.npy --cache data/cache_g_it.npz \
    --split-dir outputs/region_split --output-dir outputs/ablation_g_it_L14 \
    --conditions swap-c
```

GPU reproduction for the Taboo and ordered-colour experiments is available through [`modal/experiments.py`](modal/experiments.py). The saved Taboo resolution, layer, and secret-text controls are in [`results/summaries/taboo_robustness.json`](results/summaries/taboo_robustness.json).

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
partition.label                             # Optional[str]. None on hub dictionaries;
                                            #   populate with scripts/label_dictionary.py
                                            #   (needs an Anthropic API key).
```

A partition has two candidate representatives, which the paper calls the exemplar and the region mean. `exemplar_direction` is the first-arrival activation that anchored the cell — observed, immutable, traceable to the prompt and token that produced it, and the one used for the intervention examples below. `mean_member_direction` is the mean of everything assigned to the cell — smoother, but not a real activation and with no prompt behind it. The paper benchmarks both at AxBench in §3.1: under SAE-A's selection rule the region mean is the better detector at every resolution (0.935 vs 0.828 at p₁), because a single activation gives a spiky max-over-positions score that lets one generic region win the training-set selection for many concepts. So default to `mean_member_direction` for read-out / probing and `exemplar_direction` for causal interventions and for anything you need to trace back to data.

### Intervention with an exemplar

`exemplar_direction` lives in the centered, unit-norm space the dictionary clusters in. To inject (or ablate) a partition in the model's raw activation space, undo the centering and pick a scale that matches the layer's typical activation norm.

**On `alpha`.** `e` is unit-norm, so `alpha` directly sets how strongly you push along the exemplar in raw activation units. The centroid norm `||d.center||` is a reasonable default — it matches the layer's mean activation magnitude — but the right value depends on what you're testing. A push of `0.5 · ||d.center||` is a gentle nudge; `2-3 · ||d.center||` is a hard override. If you see no behavioural effect, the most common cause is `alpha` too small; the next most common is intervening at the wrong layer.

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

The paper's causal result uses a different intervention: it projects off the span of harmful-region **means** and adds the benign regions' mean position within that span. See [`scripts/exp_region_ablation.py`](scripts/exp_region_ablation.py) for that centroid swap and its controls. The single-direction hook above is an illustrative intervention recipe.

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
