"""Verify that the public surface of the package imports cleanly post-rename."""

from __future__ import annotations


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
