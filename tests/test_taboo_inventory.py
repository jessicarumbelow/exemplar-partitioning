import json

import numpy as np
import torch

from scripts.exp_taboo import build_dictionary
from scripts.exp_taboo_inventory import inventory, reconstructed_center


class Tokenizer:
    def decode(self, ids):
        return str(ids[0])

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0]}


def test_inventory_lists_all_added_regions_without_secret_selection(tmp_path, monkeypatch):
    (tmp_path / "results.json").write_text(json.dumps({
        "secret": "blue", "layer": 32, "percentile": 12, "seed": 0,
        "n_control_taboo": 4, "n_control_stock": 4, "K": 3,
    }))
    np.savez(tmp_path / "acts_control.npz",
             taboo=np.array([[1., 0.], [1., 0.], [0., 1.], [-1., 0.]]),
             ids_taboo=np.array([0, 0, 1, 2]),
             ids_stock=np.array([0, 0, 0, 2]))
    monkeypatch.setattr("scripts.exp_taboo_inventory.reconstructed_center",
                        lambda acts, percentile, seed: np.zeros(2))
    weight = torch.tensor([[0., 1.], [1., 0.], [-1., 0.]])

    report = inventory(tmp_path, weight, Tokenizer(), 2, evaluate_secret=False)

    assert [row["region"] for row in report["added_regions"]] == [1]
    assert report["added_regions"][0]["base_members"] == 0
    assert "secret_embedding_rank" not in report["added_regions"][0]


def test_inventory_uses_the_dictionary_calibration_center():
    acts = np.random.default_rng(0).normal(size=(20, 6)).astype(np.float32)
    dictionary, _, _, _ = build_dictionary(acts, percentile=12, seed=0)

    np.testing.assert_allclose(reconstructed_center(acts, 12, 0),
                               dictionary.center)
