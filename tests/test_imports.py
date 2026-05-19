"""Verify that the public surface of the package imports cleanly post-rename."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_top_level_imports():
    from ep import (  # noqa: F401
        Calibration,
        Dictionary,
        DiscoveryResult,
        ExtractionResult,
        Partition,
        Snapshot,
        calibrate,
        calibrate_pipeline,
        discover,
        evaluate_and_log,
        evaluate_dictionary,
        extract_final_position,
        extract_per_position,
        format_eval_summary,
        load_or_calibrate,
        normalize_extractor_output,
        set_seed,
    )


def test_discovery_subpackage_imports():
    from ep.discovery import (  # noqa: F401
        Calibration,
        Dictionary,
        DiscoveryResult,
        ExtractionResult,
        Partition,
        Snapshot,
        calibrate,
        calibrate_pipeline,
        discover,
        evaluate_and_log,
        evaluate_dictionary,
        extract_final_position,
        extract_per_position,
        format_eval_summary,
        load_or_calibrate,
        normalize_extractor_output,
    )


def test_dictionary_module_internals():
    from ep.discovery.dictionary import (  # noqa: F401
        Dictionary,
        Partition,
        _cosine_pairwise,
        _to_directions,
    )


def test_calibration_module_internals():
    from ep.discovery.calibration import (  # noqa: F401
        Calibration,
        cache_path,
        calibrate,
        load,
        load_or_calibrate,
        save,
    )


def test_geometry_helper_imports():
    from ep.discovery.geometry import centered_unit  # noqa: F401


def test_no_legacy_modules():
    import importlib
    for name in ("ep.discovery.library", "ep.discovery.signatures",
                 "ep.discovery.clustering", "ep.discovery.stats"):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        else:
            raise AssertionError(f"{name} should have been removed")


def test_dictionary_import_without_torch_available():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        import numpy as np


        class BlockTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ModuleNotFoundError("No module named 'torch'", name="torch")
                return None


        sys.meta_path.insert(0, BlockTorch())

        import ep
        from ep.discovery import Dictionary

        ep.set_seed(123)
        d = Dictionary(center=np.zeros(2, dtype=np.float32), threshold=0.5)
        print("ok", len(d))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "ok 0"
