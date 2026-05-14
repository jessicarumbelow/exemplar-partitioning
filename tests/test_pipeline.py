"""Smoke test for the discover() pipeline using a stub extractor."""

from __future__ import annotations

import numpy as np

from ep.discovery import (
    DiscoveryResult,
    calibrate_pipeline,
    discover,
)
from ep.discovery.extraction import ExtractionResult


def _make_stub_extractor(rng, d: int = 16):
    """Build a stub extractor that yields k=4 fixed-cluster activations."""
    centers = rng.standard_normal((4, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= 3.0

    def stub_extract(model, prompts, hook_name, device="cpu", **kw):
        n = len(prompts)
        rng_local = np.random.default_rng(hash(prompts[0]) & 0xFFFFFFFF)
        ks = rng_local.integers(0, 4, size=n)
        x = centers[ks] + 0.05 * rng_local.standard_normal((n, d)).astype(np.float32)
        return ExtractionResult(
            x=x,
            prompt_ids=np.arange(n, dtype=np.int64),
            position_ids=np.zeros(n, dtype=np.int64),
            n_forward_passes=1,
            n_tokens=n,
        )
    return stub_extract


class TestDiscover:
    def test_returns_discovery_result(self):
        rng = np.random.default_rng(0)
        extract = _make_stub_extractor(rng)
        texts = [f"prompt {i}" for i in range(60)]

        cal = calibrate_pipeline(
            model=None, texts=list(texts), hook_name="blocks.0.hook_resid_post",
            extract_fn=extract, prompt_batch_size=10,
            n_tokens=20, percentile=10,
        )
        result = discover(
            model=None,
            texts=texts,
            hook_name="blocks.0.hook_resid_post",
            calibration=cal,
            extract_fn=extract,
            prompt_batch_size=10,
        )
        assert isinstance(result, DiscoveryResult)
        assert result.n_activations > 0
        assert len(result.dictionary) >= 1
        assert result.snapshots
        assert result.dictionary.threshold > 0

    def test_max_prompts_caps_run(self):
        rng = np.random.default_rng(0)
        extract = _make_stub_extractor(rng)
        texts = [f"prompt {i}" for i in range(200)]
        cal = calibrate_pipeline(
            model=None, texts=list(texts), hook_name="blocks.0.hook_resid_post",
            extract_fn=extract, prompt_batch_size=10,
            n_tokens=20, percentile=10,
        )
        result = discover(
            model=None,
            texts=texts,
            hook_name="blocks.0.hook_resid_post",
            calibration=cal,
            extract_fn=extract,
            prompt_batch_size=10,
            max_prompts=30,
        )
        assert result.n_prompts <= 30 + 10

    def test_log_fn_receives_metrics(self):
        rng = np.random.default_rng(0)
        extract = _make_stub_extractor(rng)
        seen = []

        def log_fn(metrics):
            seen.append(metrics)

        texts = [f"p {i}" for i in range(40)]
        cal = calibrate_pipeline(
            model=None, texts=list(texts), hook_name="blocks.0.hook_resid_post",
            extract_fn=extract, prompt_batch_size=10,
            n_tokens=20, percentile=10,
        )
        discover(
            model=None,
            texts=texts,
            hook_name="blocks.0.hook_resid_post",
            calibration=cal,
            extract_fn=extract,
            prompt_batch_size=10,
            log_fn=log_fn,
        )
        assert seen
        assert "n_partitions" in seen[0]
        assert "n_acts" in seen[0]
