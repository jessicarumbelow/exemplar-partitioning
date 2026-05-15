"""Streaming exemplar-partition dictionary on the centered unit sphere.

Each new activation is converted to a direction in centered space — subtract
the precomputed activation center, then L2-normalise. Cosine distance to
existing partition directions decides cell membership: if the nearest
exemplar direction is within the precomputed threshold, the activation joins
that partition; otherwise it spawns a new one with itself as exemplar.

The center and threshold are computed once per (model, hook, percentile) by
the calibration module and passed in at construction. They are immutable for
the life of the dictionary, so every activation — past, present, future —
sees the same geometric reference.

Each partition stores two unit directions:
- ``exemplar_direction``: the first-arrival activation that anchored the
  cell, converted to a centered unit direction.
- ``mean_member_direction``: equal-weighted spherical mean of all member
  directions (renormalised to unit length on read).

Plus ``member_coherence`` ∈ [0, 1]: ``||sum_member_directions|| / N``,
which equals 1 only when all members agree on direction. Low coherence =
loose cell.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field

import numpy as np

from .geometry import try_torch_gpu

logger = logging.getLogger(__name__)

EPS = 1e-12


def _to_directions(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Subtract ``center`` and L2-normalise each row."""
    centered = x - center
    norms = np.linalg.norm(centered, axis=-1, keepdims=True) + EPS
    return centered / norms


def _cosine_pairwise(directions: np.ndarray) -> np.ndarray:
    """Within-batch pairwise cosine distances (condensed, length n*(n-1)/2)."""
    from scipy.spatial.distance import pdist
    return pdist(directions, metric="cosine")


@dataclass
class Partition:
    """One partition cell: two unit directions plus membership statistics.

    The two ``*_prompts`` lists are min-heaps in the heapq convention; for
    plain sorted access, use :attr:`closest_prompts` and
    :attr:`farthest_prompts`.
    """

    member_count: int
    exemplar_direction: np.ndarray            # unit vector, centered space
    mean_member_direction: np.ndarray         # unit vector, centered space
    member_coherence: float = 1.0             # [0, 1]; 1 = all members agree
    sum_dist_to_exemplar: float = 0.0
    sum_sq_dist_to_exemplar: float = 0.0
    source_iterations: set[int] = field(default_factory=set)
    constituent_sample_indices: list[int] = field(default_factory=list)
    sample_prompts: list[tuple[float, str, int]] = field(default_factory=list)
    boundary_prompts: list[tuple[float, str, int]] = field(default_factory=list)
    sample_members: list[np.ndarray] = field(default_factory=list)
    label: str | None = None

    @property
    def closest_prompts(self) -> list[tuple[float, str, int]]:
        """Member prompts closest to the exemplar, sorted nearest-first.

        Returns ``(distance, prompt, position)`` tuples with positive
        distances. Wraps the ``sample_prompts`` heap (whose internal sign
        convention is a ``heapq`` implementation detail).
        """
        return sorted(
            ((-neg_dist, prompt, pos) for neg_dist, prompt, pos in self.sample_prompts),
            key=lambda t: t[0],
        )

    @property
    def farthest_prompts(self) -> list[tuple[float, str, int]]:
        """Member prompts farthest from the exemplar, sorted farthest-first.

        Returns ``(distance, prompt, position)`` tuples. Wraps the
        ``boundary_prompts`` heap.
        """
        return sorted(self.boundary_prompts, key=lambda t: t[0], reverse=True)


SAMPLE_RESERVOIR_CAP = 30
MAX_PROMPT_EXAMPLES = 10


class Dictionary:
    """Streaming exemplar-partition dictionary built by leader clustering.

    Args:
        center: precomputed activation mean. All incoming activations are
            centered by this before distance computation.
        threshold: precomputed cosine-distance threshold for cell membership.
            An activation joins an existing partition if the nearest
            exemplar direction is within this distance; otherwise it spawns
            a new partition.
        merge_close: if True, run a post-batch pass that merges any pair of
            partitions whose exemplar directions are within ``threshold``.
            The larger partition keeps its first-arrival exemplar; the
            smaller is dissolved into it (demotion strategy). Off by
            default; canonical EP keeps the full leader-clustering set.

    The ``exemplar_direction`` of each partition is the unit-direction form
    of its first-arrival activation, immutable for the life of the cell.
    """

    def __init__(
        self,
        center: np.ndarray,
        threshold: float,
        *,
        merge_close: bool = False,
    ):
        if center.ndim != 1:
            raise ValueError(f"center must be 1-D, got shape {center.shape}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.center = center.astype(np.float32, copy=True)
        self.threshold = float(threshold)
        self.merge_close = merge_close
        self.partitions: list[Partition] = []
        self._last_batch_new_partition_rate = 0.0

        # Incremental exemplar-direction cache: contiguous (capacity, D) buffer,
        # appended on _create_partition. Avoids re-stacking K exemplars on every
        # add_batch call. Rebuilt only when partitions are mutated by merge.
        self._exemplars: np.ndarray | None = None
        self._exemplars_capacity = 0

        # Optional GPU mirror of the exemplar buffer. When CUDA is available
        # the per-batch nearest-exemplar matmul runs on device; the numpy
        # buffer remains the canonical store (used by merge, distances,
        # serialization). Mirror is kept in sync on append/rebuild.
        self._torch, self._torch_device = try_torch_gpu()
        self._use_torch = self._torch is not None
        self._exemplars_t = None  # torch.Tensor on GPU when _use_torch
        self._exemplars_t_capacity = 0

        # Stashed per-batch outputs for the pipeline to reuse without
        # recomputing directions or distances. Set at the end of add_batch.
        self._last_directions: np.ndarray | None = None
        self._last_dists: np.ndarray | None = None

    # ------------------------------------------------------------------ utils

    def to_directions(self, x: np.ndarray) -> np.ndarray:
        """Convert raw activations to centered unit directions."""
        return _to_directions(x, self.center)

    def _exemplar_direction_matrix(self) -> np.ndarray:
        """Return a view of the cached exemplar-direction matrix (K, D)."""
        if self._exemplars is None or not self.partitions:
            return np.zeros((0, self.center.shape[0]), dtype=np.float32)
        return self._exemplars[:len(self.partitions)]

    def _pairwise_sim(self, dirs: np.ndarray) -> np.ndarray:
        """Within-batch pairwise cosine similarity (N, N).

        GPU when available; result is moved to host as float32 numpy. Used
        by the leader-clustering inner loop to avoid per-iter gemv calls.
        """
        if self._use_torch:
            torch = self._torch
            x_t = torch.from_numpy(dirs).to(
                device=self._torch_device, dtype=torch.float32,
            )
            sim = x_t @ x_t.T
            return sim.cpu().numpy().astype(np.float32, copy=False)
        return dirs @ dirs.T

    def _nearest_exemplar(
        self, batch_dirs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-row min cosine distance and argmin partition index against exemplars.

        Hot path of add_batch: scales O(B × D × K). Runs on GPU when CUDA is
        available and never materialises the (B, K) intermediate on host.
        Returns numpy arrays so the rest of add_batch is unchanged.
        """
        n_partitions = len(self.partitions)
        if self._use_torch and n_partitions > 0:
            torch = self._torch
            E_t = self._exemplars_t[:n_partitions]
            x_t = torch.from_numpy(batch_dirs).to(
                device=self._torch_device, dtype=torch.float32,
            )
            sim = x_t @ E_t.T
            best_sim, best_idx = sim.max(dim=1)
            min_dists = (1.0 - best_sim).cpu().numpy().astype(np.float32, copy=False)
            best_idxs = best_idx.cpu().numpy().astype(np.int64, copy=False)
            return min_dists, best_idxs

        E = self._exemplar_direction_matrix()
        all_dists = 1.0 - batch_dirs @ E.T
        return all_dists.min(axis=1), all_dists.argmin(axis=1)

    def _ensure_exemplar_capacity(self, size: int, sample: np.ndarray) -> None:
        if self._exemplars_capacity >= size:
            return
        new_capacity = max(size, max(16, self._exemplars_capacity * 2))
        buf = np.empty((new_capacity, sample.shape[0]), dtype=np.float32)
        if self._exemplars is not None and self._exemplars_capacity:
            buf[:self._exemplars_capacity] = self._exemplars[:self._exemplars_capacity]
        self._exemplars = buf
        self._exemplars_capacity = new_capacity
        if self._use_torch:
            self._grow_torch_buffer(new_capacity, sample.shape[0])

    def _grow_torch_buffer(self, new_capacity: int, dim: int) -> None:
        torch = self._torch
        buf = torch.empty((new_capacity, dim), dtype=torch.float32,
                          device=self._torch_device)
        if self._exemplars_t is not None and self._exemplars_t_capacity:
            buf[:self._exemplars_t_capacity] = self._exemplars_t[:self._exemplars_t_capacity]
        self._exemplars_t = buf
        self._exemplars_t_capacity = new_capacity

    def _rebuild_exemplar_cache(self) -> None:
        """Rebuild the contiguous cache from current partitions (used after merge)."""
        n = len(self.partitions)
        if n == 0:
            self._exemplars = None
            self._exemplars_capacity = 0
            self._exemplars_t = None
            self._exemplars_t_capacity = 0
            return
        sample = self.partitions[0].exemplar_direction
        self._exemplars = np.empty((n, sample.shape[0]), dtype=np.float32)
        for i, p in enumerate(self.partitions):
            self._exemplars[i] = p.exemplar_direction
        self._exemplars_capacity = n
        if self._use_torch:
            torch = self._torch
            self._exemplars_t = torch.from_numpy(self._exemplars[:n]).to(
                device=self._torch_device, dtype=torch.float32,
            )
            self._exemplars_t_capacity = n

    # The torch module reference and the CUDA exemplar buffer aren't
    # pickleable; per-batch scratch is large and disposable. Strip them on
    # save and rehydrate from the numpy mirror on load.
    _UNPICKLEABLE_ATTRS = (
        "_torch", "_torch_device", "_use_torch",
        "_exemplars_t", "_exemplars_t_capacity",
        "_last_directions", "_last_dists",
    )

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if k not in self._UNPICKLEABLE_ATTRS}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._torch, self._torch_device = try_torch_gpu()
        self._use_torch = self._torch is not None
        self._exemplars_t = None
        self._exemplars_t_capacity = 0
        self._last_directions = None
        self._last_dists = None
        if self._use_torch and self._exemplars is not None and self.partitions:
            self._rebuild_exemplar_cache()

    # ------------------------------------------------------------ public API

    def add(self, x: np.ndarray, iteration: int = 0, global_index: int = -1) -> list[int]:
        """Add a single activation. Returns its assigned partition IDs (length 1)."""
        return self.add_batch(
            x[None], iteration=iteration, global_index_start=global_index,
        )[0]

    def add_batch(
        self,
        x_batch: np.ndarray,
        iteration: int = 0,
        global_index_start: int = -1,
    ) -> list[list[int]]:
        """Add a batch of activations.

        Returns a list of length ``len(x_batch)``; each entry is a 1-element
        list ``[partition_id]`` indicating which partition that activation
        joined (or created).

        After return, ``self._last_directions`` (N, D) and ``self._last_dists``
        (N,) hold the per-batch unit-direction matrix and the per-activation
        distance to the assigned partition exemplar — the pipeline reuses these
        for downstream bookkeeping without re-deriving them.
        """
        n = len(x_batch)
        assignments: list[list[int]] = [[] for _ in range(n)]
        if n == 0:
            self._last_directions = np.zeros((0, self.center.shape[0]), dtype=np.float32)
            self._last_dists = np.zeros((0,), dtype=np.float32)
            return assignments

        directions = self.to_directions(x_batch)
        per_act_dists = np.zeros(n, dtype=np.float32)

        n_new = 0
        if not self.partitions:
            gi = global_index_start if global_index_start >= 0 else -1
            self._create_partition(directions[0], iteration, gi)
            n_new += 1
            assignments[0] = [0]
            start = 1
        else:
            start = 0

        if start < n:
            batch_dirs = directions[start:]
            min_dists, best_idxs = self._nearest_exemplar(batch_dirs)
            match_mask = min_dists <= self.threshold

            # Existing-partition assignments.
            match_indices = np.where(match_mask)[0]
            if len(match_indices) > 0:
                abs_indices = match_indices + start
                partition_ids = best_idxs[match_indices]
                matched_dists = min_dists[match_indices]
                per_act_dists[abs_indices] = matched_dists
                for pid in np.unique(partition_ids):
                    mask = partition_ids == pid
                    idxs = abs_indices[mask]
                    p_dists = matched_dists[mask]
                    p = self.partitions[pid]
                    self._record_membership(
                        p, directions=directions, idxs=idxs, dists=p_dists,
                        iteration=iteration, global_index_start=global_index_start,
                    )
                    for idx in idxs:
                        assignments[idx].append(int(pid))

            # New-partition spawning. Mini leader-cluster among the unmatched.
            no_match_indices = np.where(~match_mask)[0] + start
            n_unmatched = len(no_match_indices)
            if n_unmatched > 0:
                unmatched = directions[no_match_indices]
                # Pairwise within-batch similarity precomputed once (GPU when
                # available). The sequential leader-selection loop below then
                # only does fancy-indexed argmax over a row slice — no per-
                # iter gemv. Preserves "best-fit" semantics exactly.
                pair_sim = self._pairwise_sim(unmatched)
                sim_threshold = 1.0 - self.threshold

                leader_local_buf = np.empty(n_unmatched, dtype=np.int64)
                leader_count = 0
                leader_local_idx: list[int] = []
                follower_local_idx: list[int] = []
                follower_leader: list[int] = []
                follower_dist: list[float] = []

                for i in range(n_unmatched):
                    if leader_count > 0:
                        leaders_so_far = leader_local_buf[:leader_count]
                        sims_to_leaders = pair_sim[i, leaders_so_far]
                        best = int(np.argmax(sims_to_leaders))
                        best_sim = float(sims_to_leaders[best])
                        if best_sim >= sim_threshold:
                            follower_local_idx.append(i)
                            follower_leader.append(best)
                            follower_dist.append(1.0 - best_sim)
                            continue
                    leader_local_buf[leader_count] = i
                    leader_local_idx.append(i)
                    leader_count += 1

                # Spawn one new partition per leader; record assignment + dist (0).
                leader_to_pid: list[int] = [-1] * leader_count
                for slot, li in enumerate(leader_local_idx):
                    idx = int(no_match_indices[li])
                    gi = global_index_start + idx if global_index_start >= 0 else -1
                    pid = len(self.partitions)
                    self._create_partition(directions[idx], iteration, gi)
                    assignments[idx] = [pid]
                    leader_to_pid[slot] = pid
                    n_new += 1
                    # per_act_dists[idx] already 0.0

                # Batch followers by leader pid: one _record_membership call per leader.
                if follower_local_idx:
                    fi_arr = np.asarray(follower_local_idx, dtype=np.int64)
                    fl_arr = np.asarray(follower_leader, dtype=np.int64)
                    fd_arr = np.asarray(follower_dist, dtype=np.float32)
                    abs_idxs = no_match_indices[fi_arr]
                    per_act_dists[abs_idxs] = fd_arr
                    for slot in np.unique(fl_arr):
                        mask = fl_arr == slot
                        group_abs = abs_idxs[mask]
                        group_dists = fd_arr[mask]
                        pid = leader_to_pid[int(slot)]
                        p = self.partitions[pid]
                        explicit = (
                            [int(global_index_start + a) for a in group_abs]
                            if global_index_start >= 0 else None
                        )
                        self._record_membership(
                            p, directions=directions,
                            idxs=group_abs.astype(np.int64),
                            dists=group_dists,
                            iteration=iteration,
                            global_index_start=-1,
                            explicit_global_indices=explicit,
                        )
                        for a in group_abs:
                            assignments[int(a)] = [pid]

        self._reservoir_sample_members(assignments, directions)
        if self.merge_close:
            self._merge_close_pairs()
        self._last_batch_new_partition_rate = n_new / n
        self._last_directions = directions
        self._last_dists = per_act_dists
        return assignments

    # ------------------------------------------------------ partition writes

    def _create_partition(
        self, direction: np.ndarray, iteration: int, global_index: int,
    ) -> None:
        d = direction.astype(np.float32, copy=True)
        idx = len(self.partitions)
        self._ensure_exemplar_capacity(idx + 1, d)
        assert self._exemplars is not None
        self._exemplars[idx] = d
        if self._use_torch:
            torch = self._torch
            self._exemplars_t[idx] = torch.from_numpy(d).to(
                device=self._torch_device, dtype=torch.float32,
            )
        self.partitions.append(Partition(
            member_count=1,
            exemplar_direction=d,
            mean_member_direction=d.copy(),
            member_coherence=1.0,
            source_iterations={iteration},
            constituent_sample_indices=[global_index] if global_index >= 0 else [],
        ))

    def _record_membership(
        self,
        p: Partition,
        directions: np.ndarray,
        idxs: np.ndarray,
        dists: np.ndarray,
        iteration: int,
        global_index_start: int,
        explicit_global_indices: list[int] | None = None,
    ) -> None:
        new_dirs = directions[idxs]
        n_new = len(idxs)
        n_old = p.member_count
        n_total = n_old + n_new

        # Equal-weighted spherical mean: reconstruct the stored sum-of-units,
        # append the new unit directions, then renormalise.
        old_sum = p.member_coherence * n_old * p.mean_member_direction
        new_sum = old_sum + new_dirs.sum(axis=0)
        mean_direction = new_sum / n_total
        coherence = float(np.linalg.norm(mean_direction))
        # Store as unit direction; coherence is the magnitude before renorm.
        p.mean_member_direction = (mean_direction / (coherence + EPS)).astype(
            p.mean_member_direction.dtype, copy=False,
        )
        p.member_coherence = coherence
        p.member_count = n_total
        p.sum_dist_to_exemplar += float(dists.sum())
        p.sum_sq_dist_to_exemplar += float((dists ** 2).sum())
        p.source_iterations.add(iteration)
        if explicit_global_indices is not None:
            p.constituent_sample_indices.extend(explicit_global_indices)
        elif global_index_start >= 0:
            p.constituent_sample_indices.extend(
                int(gi) for gi in idxs + global_index_start
            )

    def _reservoir_sample_members(
        self, assignments: list[list[int]], directions: np.ndarray,
    ) -> None:
        """Per-partition reservoir of member directions (for viz)."""
        cap = SAMPLE_RESERVOIR_CAP
        n = len(assignments)
        if n == 0:
            return
        rng = np.random.default_rng()
        # Pre-draw RNG once per batch instead of two python-level calls per act.
        rolls = rng.random(n)
        slots = rng.integers(0, cap, n)
        for idx, pids in enumerate(assignments):
            for pid in pids:
                if pid >= len(self.partitions):
                    continue
                p = self.partitions[pid]
                if len(p.sample_members) < cap:
                    p.sample_members.append(directions[idx].copy())
                elif rolls[idx] < cap / max(p.member_count, 1):
                    p.sample_members[int(slots[idx])] = directions[idx].copy()

    # ----------------------------------------------------------- merge

    def _close_exemplar_pairs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Upper-triangle pairs (i, j) with cosine distance <= threshold.

        Runs the K×K matmul on GPU when CUDA is available and only the
        matching pair indices cross back to host — avoids the K×K dist
        matrix (1.6 GB at K=20k) on CPU.
        """
        sim_threshold = 1.0 - self.threshold
        if self._use_torch:
            torch = self._torch
            E_t = self._exemplars_t[:n]
            sim = E_t @ E_t.T
            sim.clamp_(-1.0, 1.0)
            mask = torch.triu(sim >= sim_threshold, diagonal=1)
            pairs = mask.nonzero(as_tuple=False)
            if pairs.numel() == 0:
                return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
            pairs_cpu = pairs.cpu().numpy()
            return pairs_cpu[:, 0], pairs_cpu[:, 1]

        E = self._exemplar_direction_matrix()
        sim = E @ E.T
        np.clip(sim, -1.0, 1.0, out=sim)
        iu, ju = np.triu_indices(n, k=1)
        close = sim[iu, ju] >= sim_threshold
        return iu[close], ju[close]

    def _merge_close_pairs(self) -> int:
        """Merge any pair of partitions whose exemplar directions are within θ.

        Demotion strategy: the larger partition keeps its first-arrival
        exemplar; the smaller is dissolved into it. Returns the number of
        merge operations performed.
        """
        n = len(self.partitions)
        if n < 2:
            return 0

        iu, ju = self._close_exemplar_pairs(n)
        if iu.size == 0:
            return 0

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in zip(iu, ju):
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[rj] = ri

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        new_partitions: list[Partition] = []
        n_merges = 0
        for members in groups.values():
            if len(members) == 1:
                new_partitions.append(self.partitions[members[0]])
                continue
            members.sort(key=lambda i: -self.partitions[i].member_count)
            target = self.partitions[members[0]]
            for source_idx in members[1:]:
                self._merge_into_target(target, self.partitions[source_idx])
                n_merges += 1
            new_partitions.append(target)

        if n_merges:
            self.partitions = new_partitions
            self._rebuild_exemplar_cache()
            logger.info("merge_close: merged %d pair(s) → %d partitions remain",
                        n_merges, len(self.partitions))
        return n_merges

    def _merge_into_target(self, target: Partition, source: Partition) -> None:
        n_a, n_b = target.member_count, source.member_count
        n_total = n_a + n_b

        # Demotion: target.exemplar_direction (first-arrival) is preserved.
        # mean_member_direction: combine the two (renormalised) directional means.
        # Both target and source store unit-mean directions with separate coherences;
        # reconstruct the underlying sums to combine properly.
        target_sum = target.member_coherence * n_a * target.mean_member_direction
        source_sum = source.member_coherence * n_b * source.mean_member_direction
        combined_sum = (target_sum + source_sum) / n_total
        coherence = float(np.linalg.norm(combined_sum))
        target.mean_member_direction = (combined_sum / (coherence + EPS)).astype(
            target.mean_member_direction.dtype, copy=False,
        )
        target.member_coherence = coherence
        target.member_count = n_total
        target.sum_dist_to_exemplar += source.sum_dist_to_exemplar
        target.sum_sq_dist_to_exemplar += source.sum_sq_dist_to_exemplar
        target.source_iterations |= source.source_iterations
        target.constituent_sample_indices.extend(source.constituent_sample_indices)

        # sample_prompts stores (-dist, prompt, pos); keep the K with largest
        # x[0] (= largest -dist = smallest dist = closest to exemplar).
        combined_sp = target.sample_prompts + source.sample_prompts
        combined_sp.sort(key=lambda x: x[0], reverse=True)
        target.sample_prompts = combined_sp[:MAX_PROMPT_EXAMPLES]
        heapq.heapify(target.sample_prompts)

        combined_bp = target.boundary_prompts + source.boundary_prompts
        combined_bp.sort(key=lambda x: x[0], reverse=True)
        target.boundary_prompts = combined_bp[:MAX_PROMPT_EXAMPLES]
        heapq.heapify(target.boundary_prompts)

        combined_m = target.sample_members + source.sample_members
        if len(combined_m) <= SAMPLE_RESERVOIR_CAP:
            target.sample_members = combined_m
        else:
            rng = np.random.default_rng()
            keep = rng.choice(len(combined_m), size=SAMPLE_RESERVOIR_CAP, replace=False)
            target.sample_members = [combined_m[i] for i in keep]

    # --------------------------------------------------------- finalization

    def finalize(self, min_members: int = 1) -> None:
        """Drop partitions with fewer than ``min_members`` members."""
        small = [i for i, p in enumerate(self.partitions) if p.member_count < min_members]
        if not small:
            return
        big = [i for i, p in enumerate(self.partitions) if p.member_count >= min_members]
        if not big:
            logger.warning(
                "finalize: no partitions with >= %d members; dropping %d",
                min_members, len(small),
            )
            self.partitions = []
            self._rebuild_exemplar_cache()
            return
        logger.info("finalize: dropping %d partitions below min_members=%d",
                    len(small), min_members)
        self.partitions = [self.partitions[i] for i in big]
        self._rebuild_exemplar_cache()

    # ------------------------------------------------------------ inference

    def assign(self, vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Assign activations to their nearest partition (read-only).

        ``vecs`` are raw activations; centering and normalising are applied
        internally. Returns ``(partition_ids, distances)``, both shape ``(N,)``.
        The dictionary itself is not mutated by this call.
        """
        if not self.partitions:
            return (np.full(len(vecs), -1, dtype=np.int64),
                    np.full(len(vecs), np.inf))
        directions = self.to_directions(vecs)
        min_dists, best_idxs = self._nearest_exemplar(directions)
        return best_idxs.astype(np.int64, copy=False), min_dists

    def distances(self, x: np.ndarray) -> np.ndarray:
        """Cosine distance from each activation to each partition exemplar.

        Returns an ``(N, K)`` array of distances. Runs on GPU when CUDA is
        available; the (N, K) result is then moved to host. For very large N
        and K this can exceed host memory — callers that only need argmin or
        a per-row reduction should use ``assign`` instead.
        """
        if not self.partitions:
            return np.full((len(x), 0), np.inf)
        directions = self.to_directions(x)
        n_partitions = len(self.partitions)
        if self._use_torch:
            torch = self._torch
            E_t = self._exemplars_t[:n_partitions]
            x_t = torch.from_numpy(directions).to(
                device=self._torch_device, dtype=torch.float32,
            )
            sim = x_t @ E_t.T
            return (1.0 - sim).cpu().numpy().astype(np.float32, copy=False)
        E = self._exemplar_direction_matrix()
        return 1.0 - directions @ E.T

    # ------------------------------------------------------- diagnostics

    def distance_distributions(self, min_members: int = 2) -> dict:
        """Intra-partition mean-distance and inter-partition pairwise distances."""
        partitions = [p for p in self.partitions if p.member_count >= min_members]
        if len(partitions) < 2:
            return {"intra": np.array([]), "inter": np.array([]),
                    "nearest_inter": np.array([])}

        intra = np.array([p.sum_dist_to_exemplar / p.member_count for p in partitions])
        E = np.stack([p.exemplar_direction for p in partitions])
        inter = _cosine_pairwise(E)

        sim = E @ E.T
        np.clip(sim, -1.0, 1.0, out=sim)
        dist_matrix = 1.0 - sim
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_inter = dist_matrix.min(axis=1)

        return {"intra": intra, "inter": inter, "nearest_inter": nearest_inter}

    @property
    def new_partition_rate(self) -> float:
        """Fraction of activations in the most recent batch that spawned new partitions."""
        return self._last_batch_new_partition_rate

    # --------------------------------------------------------- hub load

    @classmethod
    def from_hub(
        cls,
        model_short: str = "gemma-2-2b",
        layer: int = 12,
        percentile: int = 10,
        *,
        repo_id: str = "J-RUM/exemplar-partitioning",
        cache_dir: str | None = None,
    ) -> "Dictionary":
        """Load a published EP dictionary from the HuggingFace Hub.

        EP dictionaries are built (streaming partition over observed
        activations), not trained, so this method intentionally avoids the
        ``from_pretrained`` naming used for learned models.

        Args:
            model_short: ``gemma-2-2b`` or ``gemma-2-2b-it``.
            layer: 4, 12, or 20.
            percentile: 1, 2, 4, 8, or 10. Note ``gemma-2-2b`` L20 only
                ships ``p10``.
            repo_id: HF dataset repo. Override only to point at a fork.
            cache_dir: passed through to ``hf_hub_download``.

        Returns:
            The unpickled :class:`Dictionary`.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "Dictionary.from_hub requires huggingface_hub. "
                "Install it (or any of: transformers, datasets) to fetch "
                "prebuilt dictionaries."
            ) from e
        import pickle

        subdir = f"{model_short}_L{layer}_p{percentile}"
        fname = f"{model_short}_layer{layer}.pkl"
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subdir}/{fname}",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        with open(path, "rb") as f:
            return pickle.load(f)

    def __len__(self) -> int:
        return len(self.partitions)

    def __repr__(self) -> str:
        n_real = sum(1 for p in self.partitions if p.member_count >= 2)
        return (f"Dictionary({len(self.partitions)} partitions, "
                f"{n_real} with ≥2 members, θ={self.threshold:.4f}, "
                f"||center||={float(np.linalg.norm(self.center)):.4f})")
