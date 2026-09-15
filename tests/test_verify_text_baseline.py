import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_text_baseline.py"
spec = importlib.util.spec_from_file_location("verify_text_baseline", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_harmful_coverage_counts_only_qualifying_regions():
    labels = np.array([0, 0, 1, 1, 1, 1])
    assert module.harmful_coverage(labels, n_harmful=4, threshold=0.9) == 0.5
