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
