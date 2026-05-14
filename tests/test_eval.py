"""Smoke tests for evaluate_dictionary."""

from __future__ import annotations

import numpy as np

from ep.discovery import Dictionary, evaluate_dictionary, format_eval_summary


def _make_clustered(k: int, n_per: int, d: int, rng, noise: float = 0.05):
    centers = rng.standard_normal((k, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= 3.0
    blocks = [c + noise * rng.standard_normal((n_per, d)).astype(np.float32)
              for c in centers]
    return np.vstack(blocks)


class TestEvaluateDictionary:
    def test_empty_dictionary(self):
        d = Dictionary(center=np.zeros(4, dtype=np.float32), threshold=0.1)
        m = evaluate_dictionary(d)
        assert m == {"n_partitions": 0}

    def test_basic_metrics_present(self):
        rng = np.random.default_rng(42)
        x = _make_clustered(4, 30, 16, rng)
        d = Dictionary(center=x.mean(axis=0), threshold=0.05)
        d.add_batch(x)
        d.finalize()
        m = evaluate_dictionary(d, min_members=2)
        for key in ("n_partitions", "coverage", "mean_members", "gini",
                    "effective_partitions", "mean_intra_dist"):
            assert key in m

    def test_format_summary_handles_empty(self):
        out = format_eval_summary({"n_partitions": 0})
        assert "empty" in out.lower()

    def test_format_summary_renders_metrics(self):
        rng = np.random.default_rng(42)
        x = _make_clustered(4, 30, 16, rng)
        d = Dictionary(center=x.mean(axis=0), threshold=0.05)
        d.add_batch(x)
        d.finalize()
        m = evaluate_dictionary(d, min_members=2)
        out = format_eval_summary(m)
        assert "Partitions" in out
