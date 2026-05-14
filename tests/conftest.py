"""Shared fixtures for EP tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dim():
    return 64


@pytest.fixture
def random_vectors(rng, dim):
    """Factory: generate (N, D) random vectors."""
    def _make(n: int, d: int | None = None) -> np.ndarray:
        return rng.standard_normal((n, d or dim)).astype(np.float32)
    return _make


@pytest.fixture
def clustered_vectors(rng, dim):
    """Generate vectors from K well-separated clusters."""
    def _make(k: int = 4, per_cluster: int = 50, d: int | None = None, noise: float = 0.05):
        d = d or dim
        centers = rng.standard_normal((k, d)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        centers *= 3.0

        vecs = []
        labels = []
        for i, c in enumerate(centers):
            cluster_vecs = c + noise * rng.standard_normal((per_cluster, d)).astype(np.float32)
            vecs.append(cluster_vecs)
            labels.extend([i] * per_cluster)
        return np.concatenate(vecs), np.array(labels)
    return _make
