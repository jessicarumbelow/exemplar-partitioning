"""Smoke tests for the Dictionary clustering API."""

from __future__ import annotations

import numpy as np

from ep.discovery import Dictionary, Partition


def _make_clustered(k: int, n_per: int, d: int, rng, noise: float = 0.05):
    centers = rng.standard_normal((k, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= 3.0
    blocks = []
    labels = []
    for i, c in enumerate(centers):
        block = c + noise * rng.standard_normal((n_per, d)).astype(np.float32)
        blocks.append(block)
        labels.extend([i] * n_per)
    return np.vstack(blocks), np.array(labels)


def _zero_center(d: int) -> np.ndarray:
    return np.zeros(d, dtype=np.float32)


class TestInit:
    def test_required_args(self):
        d = Dictionary(center=_zero_center(8), threshold=0.1)
        assert d.threshold == 0.1
        assert d.center.shape == (8,)
        assert len(d) == 0
        assert d.partitions == []

    def test_rejects_bad_center_shape(self):
        import pytest
        with pytest.raises(ValueError):
            Dictionary(center=np.zeros((4, 4), dtype=np.float32), threshold=0.1)

    def test_rejects_nonpositive_threshold(self):
        import pytest
        with pytest.raises(ValueError):
            Dictionary(center=_zero_center(4), threshold=0.0)


class TestAddBatch:
    def test_recovers_distinct_clusters(self):
        rng = np.random.default_rng(42)
        x, _ = _make_clustered(4, 50, 32, rng, noise=0.02)
        d = Dictionary(center=x.mean(axis=0), threshold=0.05)
        d.add_batch(x)
        d.finalize()
        assert 2 <= len(d) <= 60

    def test_first_activation_becomes_exemplar_direction(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((10, 16)).astype(np.float32)
        center = x.mean(axis=0)
        d = Dictionary(center=center, threshold=0.05)
        d.add_batch(x)
        # Partition 0's exemplar_direction should be the unit vector of
        # (x[0] - center).
        centered_first = x[0] - center
        expected = centered_first / np.linalg.norm(centered_first)
        np.testing.assert_allclose(
            d.partitions[0].exemplar_direction, expected, atol=1e-6,
        )

    def test_tighter_threshold_more_partitions(self):
        rng = np.random.default_rng(42)
        x, _ = _make_clustered(4, 50, 32, rng)
        center = x.mean(axis=0)
        tight = Dictionary(center=center, threshold=0.001)
        tight.add_batch(x.copy())
        loose = Dictionary(center=center, threshold=0.5)
        loose.add_batch(x.copy())
        assert len(tight) >= len(loose)


class TestPartitionState:
    def test_member_counts_sum_to_input(self):
        rng = np.random.default_rng(42)
        x, _ = _make_clustered(3, 30, 16, rng)
        d = Dictionary(center=x.mean(axis=0), threshold=0.05)
        d.add_batch(x)
        total_members = sum(p.member_count for p in d.partitions)
        assert total_members == len(x)

    def test_exemplar_direction_is_unit_norm(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((20, 16)).astype(np.float32)
        d = Dictionary(center=x.mean(axis=0), threshold=0.1)
        d.add_batch(x)
        for p in d.partitions:
            np.testing.assert_allclose(
                np.linalg.norm(p.exemplar_direction), 1.0, atol=1e-5,
            )

    def test_mean_member_direction_is_unit_norm(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((20, 16)).astype(np.float32)
        d = Dictionary(center=x.mean(axis=0), threshold=0.1)
        d.add_batch(x)
        for p in d.partitions:
            np.testing.assert_allclose(
                np.linalg.norm(p.mean_member_direction), 1.0, atol=1e-5,
            )

    def test_member_coherence_in_unit_interval(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((50, 16)).astype(np.float32) * 0.05
        d = Dictionary(center=_zero_center(16), threshold=10.0)
        d.add_batch(x)
        for p in d.partitions:
            assert 0.0 <= p.member_coherence <= 1.0 + 1e-6

    def test_sample_members_reservoir_capped(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((200, 16)).astype(np.float32) * 0.05
        d = Dictionary(center=_zero_center(16), threshold=10.0)
        d.add_batch(x)
        assert len(d) == 1
        assert len(d.partitions[0].sample_members) <= 30


class TestInference:
    def test_assign_returns_valid_partition_ids(self):
        rng = np.random.default_rng(42)
        x, _ = _make_clustered(3, 30, 16, rng)
        d = Dictionary(center=x.mean(axis=0), threshold=0.05)
        d.add_batch(x)
        d.finalize()
        ids, dists = d.assign(x[:5])
        assert ids.shape == (5,)
        assert dists.shape == (5,)
        assert (ids >= 0).all()
        assert (ids < len(d)).all()

    def test_assign_empty_dictionary(self):
        d = Dictionary(center=_zero_center(4), threshold=0.1)
        ids, dists = d.assign(np.zeros((3, 4), dtype=np.float32))
        assert (ids == -1).all()
        assert np.isinf(dists).all()

    def test_distances_shape(self):
        rng = np.random.default_rng(42)
        x, _ = _make_clustered(3, 30, 16, rng)
        d = Dictionary(center=x.mean(axis=0), threshold=0.05)
        d.add_batch(x)
        d.finalize()
        D = d.distances(x[:5])
        assert D.shape == (5, len(d))

    def test_to_directions_returns_unit_norm(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((10, 16)).astype(np.float32)
        d = Dictionary(center=_zero_center(16), threshold=0.1)
        directions = d.to_directions(x)
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestFinalize:
    def test_drops_below_min_members(self):
        rng = np.random.default_rng(42)
        x, _ = _make_clustered(3, 30, 16, rng, noise=0.02)
        d = Dictionary(center=x.mean(axis=0), threshold=0.05)
        d.add_batch(x)
        n_before = len(d)
        d.finalize(min_members=10)
        assert len(d) <= n_before
        for p in d.partitions:
            assert p.member_count >= 10


class TestDistanceDistributions:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        x, _ = _make_clustered(4, 30, 16, rng)
        d = Dictionary(center=x.mean(axis=0), threshold=0.05)
        d.add_batch(x)
        dd = d.distance_distributions(min_members=2)
        assert set(dd.keys()) == {"intra", "inter", "nearest_inter"}


class TestRepr:
    def test_repr_has_partitions(self):
        d = Dictionary(center=_zero_center(4), threshold=0.1)
        assert "Dictionary(" in repr(d)
        assert "partitions" in repr(d)


class TestPartitionDataclass:
    def test_default_factories(self):
        x = np.zeros(4, dtype=np.float32)
        p = Partition(member_count=1, exemplar_direction=x, mean_member_direction=x)
        assert p.source_iterations == set()
        assert p.constituent_sample_indices == []
        assert p.sample_prompts == []
        assert p.boundary_prompts == []
        assert p.sample_members == []
        assert p.label is None
        assert p.member_coherence == 1.0


class TestMergeClose:
    def test_merge_demotion_keeps_first_arrival(self):
        rng = np.random.default_rng(0)
        # Two clusters far enough apart that they spawn separately at θ=0.001.
        c1 = np.tile(np.array([10.0] + [0.0] * 15, dtype=np.float32), (5, 1))
        c1[:, 1:] += 0.0001 * rng.standard_normal((5, 15)).astype(np.float32)
        c2 = np.tile(np.array([0.0, 10.0] + [0.0] * 14, dtype=np.float32), (3, 1))
        c2[:, 2:] += 0.0001 * rng.standard_normal((3, 14)).astype(np.float32)

        # Wide threshold so they fuse on merge_close.
        d = Dictionary(center=_zero_center(16), threshold=2.0, merge_close=True)
        d.add_batch(c1)
        d.add_batch(c2)
        # After merge, only one partition; its exemplar_direction must equal
        # the unit-direction of the larger source's first-arrival activation.
        assert len(d) == 1
        expected = c1[0] / np.linalg.norm(c1[0])
        np.testing.assert_allclose(
            d.partitions[0].exemplar_direction, expected, atol=1e-5,
        )
        assert d.partitions[0].member_count == len(c1) + len(c2)

    def test_merge_keeps_closest_sample_prompts(self):
        """_merge_into_target must keep the K closest prompts in sample_prompts,
        not the K farthest. Regression test for the bug introduced when
        merge_close was first added: the sort key was double-negated, so the
        slice kept the largest dist instead of the smallest.
        """
        import heapq
        from ep.discovery.dictionary import MAX_PROMPT_EXAMPLES

        d = Dictionary(center=_zero_center(4), threshold=0.5, merge_close=False)

        def _mk_partition(label_prefix: str, dists: list[float]) -> Partition:
            ex = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            p = Partition(member_count=1, exemplar_direction=ex,
                          mean_member_direction=ex.copy())
            for dist in dists:
                # sample_prompts heap convention: (-dist, prompt, pos)
                heapq.heappush(p.sample_prompts, (-dist, f"{label_prefix}@{dist:.2f}", 0))
            return p

        # target has the closer prompts (small dists), source the farther ones.
        # Total entries > MAX_PROMPT_EXAMPLES so the merge truncation fires.
        n_each = MAX_PROMPT_EXAMPLES + 2
        target = _mk_partition("close", [0.01 + 0.01 * i for i in range(n_each)])
        source = _mk_partition("far", [0.50 + 0.01 * i for i in range(n_each)])
        d.partitions = [target, source]

        d._merge_into_target(target, source)

        kept_labels = {entry[1] for entry in target.sample_prompts}
        # All kept prompts must come from the "close" cluster.
        assert all(label.startswith("close@") for label in kept_labels), (
            f"sample_prompts merge kept entries from the far cluster: {kept_labels}"
        )
        assert len(target.sample_prompts) == MAX_PROMPT_EXAMPLES

    def test_merge_keeps_farthest_boundary_prompts(self):
        """Symmetric to the sample_prompts test: boundary_prompts must keep
        the K farthest prompts after merging."""
        import heapq
        from ep.discovery.dictionary import MAX_PROMPT_EXAMPLES

        d = Dictionary(center=_zero_center(4), threshold=0.5, merge_close=False)

        def _mk_partition(label_prefix: str, dists: list[float]) -> Partition:
            ex = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            p = Partition(member_count=1, exemplar_direction=ex,
                          mean_member_direction=ex.copy())
            for dist in dists:
                # boundary_prompts heap convention: (dist, prompt, pos)
                heapq.heappush(p.boundary_prompts, (dist, f"{label_prefix}@{dist:.2f}", 0))
            return p

        n_each = MAX_PROMPT_EXAMPLES + 2
        target = _mk_partition("near", [0.01 + 0.01 * i for i in range(n_each)])
        source = _mk_partition("far", [0.50 + 0.01 * i for i in range(n_each)])
        d.partitions = [target, source]

        d._merge_into_target(target, source)

        kept_labels = {entry[1] for entry in target.boundary_prompts}
        # All kept prompts must come from the "far" cluster (largest dists).
        assert all(label.startswith("far@") for label in kept_labels), (
            f"boundary_prompts merge kept entries from the near cluster: {kept_labels}"
        )
        assert len(target.boundary_prompts) == MAX_PROMPT_EXAMPLES


class TestCorrectness:
    def test_spherical_mean_matches_handcalc(self):
        """mean_member_direction should equal the spherical mean of (x - center) / ||x - center||."""
        center = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        # Three activations all assigned to one partition (very loose threshold).
        x = np.array([
            [3.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        d = Dictionary(center=center, threshold=2.0)
        d.add_batch(x)
        assert len(d) == 1
        directions = (x - center) / np.linalg.norm(x - center, axis=1, keepdims=True)
        sum_dirs = directions.sum(axis=0)
        expected_coherence = float(np.linalg.norm(sum_dirs) / len(x))
        expected_unit = sum_dirs / np.linalg.norm(sum_dirs)
        np.testing.assert_allclose(
            d.partitions[0].mean_member_direction, expected_unit, atol=1e-5,
        )
        np.testing.assert_allclose(
            d.partitions[0].member_coherence, expected_coherence, atol=1e-5,
        )

    def test_assignment_matches_distance_argmin(self):
        """Partition IDs from add_batch should match argmin of distances()."""
        rng = np.random.default_rng(7)
        x_seed, _ = _make_clustered(5, 20, 16, rng, noise=0.01)
        center = x_seed.mean(axis=0)
        d = Dictionary(center=center, threshold=0.05)
        d.add_batch(x_seed)
        d.finalize()
        # Now query with fresh activations — direct assignment via add_batch
        # (no library mutation: we use assign instead) should match argmin of distances.
        x_query = x_seed[:10] + 0.005 * rng.standard_normal((10, 16)).astype(np.float32)
        ids, _ = d.assign(x_query)
        D = d.distances(x_query)
        argmin_ids = D.argmin(axis=1)
        np.testing.assert_array_equal(ids, argmin_ids)

    def test_member_count_matches_total_inputs_across_batches(self):
        """Multiple add_batch calls must accumulate member_count exactly."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((30, 8)).astype(np.float32) * 0.05
        d = Dictionary(center=_zero_center(8), threshold=10.0)
        for batch in (x[:10], x[10:20], x[20:]):
            d.add_batch(batch)
        assert len(d) == 1
        assert d.partitions[0].member_count == 30

    def test_spherical_mean_matches_handcalc_across_batches(self):
        """Spherical mean should remain exact across multiple add_batch calls."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((30, 8)).astype(np.float32) * 0.05
        d = Dictionary(center=_zero_center(8), threshold=10.0)
        for batch in (x[:7], x[7:19], x[19:]):
            d.add_batch(batch)
        directions = x / np.linalg.norm(x, axis=1, keepdims=True)
        sum_dirs = directions.sum(axis=0)
        expected_coherence = float(np.linalg.norm(sum_dirs) / len(x))
        expected_unit = sum_dirs / np.linalg.norm(sum_dirs)
        np.testing.assert_allclose(
            d.partitions[0].mean_member_direction, expected_unit, atol=1e-5,
        )
        np.testing.assert_allclose(
            d.partitions[0].member_coherence, expected_coherence, atol=1e-5,
        )


class TestCalibration:
    def test_calibrate_threshold_matches_handcalc_percentile(self):
        """Threshold must equal mean across batches of the per-batch p-th
        percentile of cosine distances between (x - center) directions.
        """
        from ep.discovery import calibrate
        from scipy.spatial.distance import pdist
        rng = np.random.default_rng(0)
        batches = [rng.standard_normal((30, 8)).astype(np.float32) for _ in range(4)]
        cal = calibrate(batches, n_tokens=120, percentile=20.0)

        # Recompute by hand using the actual final center.
        per_batch_pct = []
        for b in batches:
            centered = b - cal.center
            units = centered / np.linalg.norm(centered, axis=1, keepdims=True)
            d = pdist(units, metric="cosine")
            per_batch_pct.append(np.percentile(d, 20.0))
        expected = float(np.mean(per_batch_pct))
        np.testing.assert_allclose(cal.threshold, expected, atol=1e-5)

    def test_calibrate_center_is_spherical_direction_scaled_projection(self):
        """Center follows the direction-first calibration formula."""
        from ep.discovery import calibrate
        rng = np.random.default_rng(0)
        batches = [
            rng.standard_normal((10, 4)).astype(np.float32),
            rng.standard_normal((20, 4)).astype(np.float32),
            rng.standard_normal((15, 4)).astype(np.float32),
        ]
        cal = calibrate(batches, n_tokens=45, percentile=10.0)
        acts = np.concatenate(batches)
        units = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-12)
        direction = units.mean(axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        expected = direction * float(acts.mean(axis=0) @ direction)
        np.testing.assert_allclose(cal.center, expected, atol=1e-5)

    def test_calibration_round_trip(self, tmp_path, monkeypatch):
        from ep.discovery import calibrate, Calibration
        from ep.discovery.calibration import save, load
        monkeypatch.setenv("EP_CALIBRATION_CACHE", str(tmp_path))
        rng = np.random.default_rng(0)
        batches = [rng.standard_normal((20, 8)).astype(np.float32) for _ in range(3)]
        cal = calibrate(batches, n_tokens=60, percentile=15.0)
        save("test-model", "blocks.0.hook_resid_post", cal)
        loaded = load("test-model", "blocks.0.hook_resid_post", percentile=15.0)
        assert loaded is not None
        np.testing.assert_allclose(loaded.center, cal.center)
        np.testing.assert_allclose(loaded.threshold, cal.threshold, atol=1e-6)
        assert loaded.percentile == 15.0
