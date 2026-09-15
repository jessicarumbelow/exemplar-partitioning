import numpy as np

from scripts.build_crossfamily_cache import build_cache


def test_build_cache_shapes_and_determinism():
    rng = np.random.default_rng(7)
    acts = rng.normal(size=(40, 3, 8)).astype(np.float32)
    ids_a, dists_a = build_cache(acts, percentile=12, seed=0, batch_size=8)
    ids_b, dists_b = build_cache(acts, percentile=12, seed=0, batch_size=8)
    assert ids_a.shape == (3, 40)
    assert dists_a.shape == (3, 40)
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_allclose(dists_a, dists_b)
