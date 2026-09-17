import torch

from scripts.exp_region_ablation import project_off
from scripts.refusal_eval import generate_hooked


def test_centroid_swap_projects_and_adds_benign_position():
    basis = torch.tensor([[1.0], [0.0]])
    centre = torch.tensor([2.0, 3.0])
    benign_position = torch.tensor([-1.0, 0.0])
    hook = project_off(basis, centre, benign_position)

    result = hook(torch.tensor([[[5.0, 7.0]]]), None)

    torch.testing.assert_close(result, torch.tensor([[[1.0, 7.0]]]))


def test_generation_installs_projection_hook(monkeypatch):
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "left"

        def __call__(self, text, **kwargs):
            if isinstance(text, str):
                return {"input_ids": [2, 3]}
            return {"input_ids": torch.tensor([[2, 3]])}

        def decode(self, ids, **kwargs):
            return "reply"

    class Model:
        tokenizer = Tokenizer()
        cfg = type("Config", (), {"device": "cpu"})()
        hook = None
        hook_result = None

        def reset_hooks(self):
            self.hook = None

        def add_hook(self, name, hook, direction):
            self.hook = hook

        def generate(self, ids, **kwargs):
            self.hook_result = self.hook(torch.tensor([[[5.0, 7.0]]]), None)
            return torch.cat([ids, torch.tensor([[4]])], dim=1)

    monkeypatch.setattr("scripts.refusal_eval.format_chat", lambda model, prompt: prompt)
    model = Model()
    hook = project_off(torch.tensor([[1.0], [0.0]]), torch.tensor([2.0, 3.0]))

    assert generate_hooked(model, ["prompt"], [("resid", hook)], 1, 1) == ["reply"]
    torch.testing.assert_close(model.hook_result, torch.tensor([[[2.0, 7.0]]]))


def test_swap_excludes_evaluation_activations_from_region_means(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from scripts import exp_region_ablation as experiment

    # Each pure region retains its exemplar; its other member is evaluated.
    acts = np.array([[1, 0, 0], [1, 9, 9], [0, 1, 0], [9, 1, 9],
                     [0, 0, 1], [9, 9, 1], [1, 0, 1], [9, 9, 9]],
                    dtype=np.float32)[:, None, :]
    cache = {"ids": np.array([[0, 0, 1, 1, 2, 2, 3, 3]]),
             "dists": np.array([[0, 1, 0, 1, 0, 1, 0, 1]])}
    meta = {"n_per_side": 4, "held_harmful_idx": [1, 3],
            "held_benign_idx": [5, 7]}
    args = SimpleNamespace(n_eval=2, conditions="swap-c", force=True,
                           device="cpu", batch_size=2, max_new_tokens=1,
                           min_baseline_refusal=0.85, holdout_centroids=True,
                           model="toy", tag="g_it")
    observed = []

    def generate(model, prompts, hooks, *unused):
        if hooks:
            observed.append(hooks[0][1](torch.tensor([[[2., 3., 4.]]]), None))
        return ["refusal"] * len(prompts)

    monkeypatch.setattr(experiment, "calibration_centre", lambda a: np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(experiment, "_generate_hooked", generate)
    monkeypatch.setattr(experiment, "_score", lambda texts: {"refusal_rate": 1., "unique_token_ratio": 1.})
    experiment.run_layer(None, args, 0, tmp_path, acts, cache, meta,
                         [str(i) for i in range(8)], ["swap-c"])
    # Harmful exemplars span x/y; benign exemplar mean is (0.5, 0, 1).
    for result in observed:
        torch.testing.assert_close(result, torch.tensor([[[0.5, 0., 4.]]]))
    assert len(observed) == 2


def test_resume_rejects_a_different_centroid_population(tmp_path):
    import json
    import pytest
    from scripts.exp_region_ablation import condition_todo

    path = tmp_path / 'results.json'
    for previous in (False, True):
        path.write_text(json.dumps({'holdout_centroids': previous,
                                    'rows': [{'condition': 'swap-c'}]}))
        assert condition_todo(tmp_path, {'swap-c'}, False, ['swap-c'], previous) == []
        for force in (False, True):
            with pytest.raises(ValueError, match='different output directory'):
                condition_todo(tmp_path, {'swap-c'}, force, ['swap-c'], not previous)
    # Older files cannot establish whether exclusion was applied.
    path.write_text(json.dumps({'rows': [{'condition': 'swap-c'}]}))
    for requested in (False, True):
        with pytest.raises(ValueError, match='different output directory'):
            condition_todo(tmp_path, {'swap-c'}, False, ['swap-c'], requested)
