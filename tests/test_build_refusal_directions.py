import numpy as np

from scripts.build_refusal_directions import build_checkpoint


def test_build_checkpoint_outputs_unit_directions():
    rng = np.random.default_rng(3)
    acts = rng.normal(size=(20, 2, 6)).astype(np.float32)
    acts[:10, :, 0] += 2
    ids = np.vstack([
        np.repeat(np.arange(10), 2),
        np.repeat(np.arange(10), 2),
    ])
    dists = np.tile(np.array([0.0, 0.1] * 10), (2, 1))
    is_harmful = np.arange(20) < 10
    built = build_checkpoint(
        acts, ids, dists, is_harmful, set(), np.array([], dtype=int),
        np.array([], dtype=int), 0, rng,
    )
    directions, prompt_mean, shuffled, norms, counts = built
    assert directions.shape == (2, 6)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(prompt_mean, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(shuffled, axis=1), 1, atol=1e-6)
    assert norms.shape == (2,)
    assert len(counts) == 2
