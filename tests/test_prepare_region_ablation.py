from scripts.prepare_region_ablation import held_out_indices


def test_split_matches_paper_counts_and_keeps_sides_disjoint():
    harmful, benign = held_out_indices(n_per_side=1176, n_held_per_side=256, seed=0)

    assert len(harmful) == len(benign) == 256
    assert harmful == sorted(set(harmful))
    assert benign == sorted(set(benign))
    assert max(harmful) < 1176 <= min(benign)
    assert (harmful[:3], benign[:3]) == ([2, 5, 7], [1176, 1179, 1188])
