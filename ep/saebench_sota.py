"""Width-matched SAEBench published baselines.

Lazily downloads the published SAEBench results from HuggingFace and exposes
the best score across all (architecture, trainer) at a chosen width. The EP
build script picks the width closest to the number of partitions discovered,
so the comparison is "best published SAE at our width" rather than
"best published SAE at 16k width."

Source: https://huggingface.co/datasets/adamkarvonen/sae_bench_results_0125
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

HF_REPO = "adamkarvonen/sae_bench_results_0125"
WIDTHS = (4096, 16384, 65536)
WIDTH_TAGS = {4096: "2pow12", 16384: "2pow14", 65536: "2pow16"}
PUBLISHED_LAYER = {"gemma-2-2b": 12, "pythia-160m-deduped": 8}


def closest_width(n_concepts: int) -> int:
    """Pick the published width closest to n_concepts in log space."""
    n = max(int(n_concepts), 1)
    return min(WIDTHS, key=lambda w: abs(math.log2(w) - math.log2(n)))


def _folder(model: str, width: int) -> str:
    return f"saebench_{model}_width-{WIDTH_TAGS[width]}_date-0108"


@lru_cache(maxsize=32)
def _download(model: str, eval_type: str, width: int) -> Path | None:
    """Snapshot-download the (model, eval_type, width) folder. Returns None on failure."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub unavailable; can't fetch SAEBench baselines")
        return None

    folder = _folder(model, width)
    try:
        local_root = snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=[f"{eval_type}/{folder}/*.json"],
        )
    except Exception as e:
        logger.warning("Failed to download %s/%s: %s", eval_type, folder, e)
        return None

    folder_path = Path(local_root) / eval_type / folder
    if not folder_path.is_dir():
        logger.warning("Downloaded but missing folder: %s", folder_path)
        return None
    return folder_path


@dataclass
class BestScore:
    value: float
    architecture: str
    trainer: int
    width: int


def _iter_results(model: str, eval_type: str, width: int):
    folder = _download(model, eval_type, width)
    if folder is None:
        return
    for path in sorted(folder.glob("*.json")):
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            continue
        # Filename: ..._{ARCH}_{model}__0108_resid_post_layer_{L}_trainer_{N}_eval_results.json
        stem = path.stem
        parts = stem.split("_")
        try:
            trainer_idx = parts.index("trainer")
            trainer_num = int(parts[trainer_idx + 1])
        except (ValueError, IndexError):
            trainer_num = -1
        # Architecture is the token after `date-{stamp}`.
        try:
            date_idx = next(i for i, p in enumerate(parts) if p.startswith("date-"))
            architecture = parts[date_idx + 1]
        except (StopIteration, IndexError):
            architecture = "unknown"
        yield architecture, trainer_num, data


def best_at_width(
    model: str,
    width: int,
    eval_type: str,
    category: str,
    key: str,
    lower_is_better: bool = False,
) -> BestScore | None:
    """Best published score for (model, width, eval_type, category, key).

    Iterates every architecture×trainer JSON in the folder, pulls
    `eval_result_metrics[category][key]`, returns the best.
    """
    best: BestScore | None = None
    for architecture, trainer, data in _iter_results(model, eval_type, width):
        metrics = data.get("eval_result_metrics") or {}
        cat = metrics.get(category) or {}
        val = cat.get(key)
        if not isinstance(val, (int, float)):
            continue
        v = float(val)
        if best is None:
            best = BestScore(v, architecture, trainer, width)
            continue
        better = (v < best.value) if lower_is_better else (v > best.value)
        if better:
            best = BestScore(v, architecture, trainer, width)
    return best


HEADLINE_METRICS: dict[str, tuple[str, str, str, bool]] = {
    # column_name -> (eval_type, category, key, lower_is_better)
    "autointerp": ("autointerp", "autointerp", "autointerp_score", False),
    "probe_k1": ("sparse_probing", "sae", "sae_top_1_test_accuracy", False),
    "probe_k2": ("sparse_probing", "sae", "sae_top_2_test_accuracy", False),
    "probe_k5": ("sparse_probing", "sae", "sae_top_5_test_accuracy", False),
    "ce_loss_score": ("core", "model_performance_preservation", "ce_loss_score", False),
    "explained_var": ("core", "reconstruction_quality", "explained_variance", False),
    "cossim": ("core", "reconstruction_quality", "cossim", False),
    "l0": ("core", "sparsity", "l0", False),
    "l2_ratio": ("core", "shrinkage", "l2_ratio", False),
    "mse": ("core", "reconstruction_quality", "mse", True),
    "scr_k10": ("scr", "scr_metrics", "scr_metric_threshold_10", False),
    "tpp_k10": ("tpp", "tpp_metrics", "tpp_threshold_10_total_metric", False),
    "ravel": ("ravel", "ravel", "disentanglement_score", False),
}


def headline_baselines(model: str, width: int) -> dict[str, BestScore]:
    """Return best-at-width score for every headline metric."""
    if model not in PUBLISHED_LAYER:
        return {}
    out: dict[str, BestScore] = {}
    for col, (eval_type, category, key, lower) in HEADLINE_METRICS.items():
        b = best_at_width(model, width, eval_type, category, key, lower_is_better=lower)
        if b is not None:
            out[col] = b
    return out


def all_runs_metrics(model: str, width: int) -> list[dict]:
    """Every published (architecture, trainer) at width, joined across eval types.

    Returns one dict per SAE: {"architecture": ..., "trainer": ..., "<metric>": value, ...}
    using the same column names as HEADLINE_METRICS. Missing metrics are absent
    from the dict (use .get(metric) at read time). Used for percentile-rank
    framing: "we land at the Nth percentile of published SAEs at this width."
    """
    if model not in PUBLISHED_LAYER:
        return []
    runs: dict[tuple[str, int], dict] = {}
    for col, (eval_type, category, key, _lower) in HEADLINE_METRICS.items():
        for arch, trainer, data in _iter_results(model, eval_type, width):
            metrics = data.get("eval_result_metrics") or {}
            cat = metrics.get(category) or {}
            val = cat.get(key)
            if not isinstance(val, (int, float)):
                continue
            row = runs.setdefault(
                (arch, trainer),
                {"architecture": arch, "trainer": trainer, "width": width},
            )
            row[col] = float(val)
    return sorted(runs.values(), key=lambda r: (r["architecture"], r["trainer"]))


def percentile_rank(value: float, distribution: list[float], lower_is_better: bool = False) -> float:
    """Percentile rank of value within distribution. 100 = best.

    For lower-is-better metrics, percentile is inverted so 100 still means "best".
    """
    if not distribution:
        return float("nan")
    if lower_is_better:
        better_count = sum(1 for d in distribution if d > value)
    else:
        better_count = sum(1 for d in distribution if d < value)
    return 100.0 * better_count / len(distribution)
