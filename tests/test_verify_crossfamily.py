import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_crossfamily.py"
spec = importlib.util.spec_from_file_location("verify_crossfamily", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_grid_counts_prompt_identity_round_trips():
    ids = np.array([[0, 0, 1, 1]])
    exemplars = [{0: 0, 1: 2}]
    fractions = [{0: 1.0, 1: 0.0}]
    result = module.grid((ids, exemplars, fractions),
                         (ids, exemplars, fractions))
    assert result[0, 0] == 1.0
