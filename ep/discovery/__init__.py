"""Exemplar Partitioning: gradient-free partition discovery in activation space.

Usage:

    >>> from ep.discovery import calibrate_pipeline, discover
    >>> calibration = calibrate_pipeline(model, texts, hook_name="blocks.4.hook_resid_post")
    >>> result = discover(model, texts, hook_name="blocks.4.hook_resid_post",
    ...                   calibration=calibration)
    >>> print(f"{len(result.dictionary)} partitions")
"""

from .calibration import Calibration, calibrate, load_or_calibrate
from .dictionary import Dictionary, Partition
from .pipeline import DiscoveryResult, Snapshot, calibrate_pipeline, discover
from .eval import evaluate_and_log, evaluate_dictionary, format_eval_summary
from .extraction import (
    ExtractionResult,
    extract_final_position,
    extract_per_position,
    normalize_extractor_output,
)


__all__ = [
    "Calibration",
    "Dictionary",
    "DiscoveryResult",
    "ExtractionResult",
    "Partition",
    "Snapshot",
    "calibrate",
    "calibrate_pipeline",
    "discover",
    "evaluate_and_log",
    "evaluate_dictionary",
    "extract_final_position",
    "extract_per_position",
    "format_eval_summary",
    "load_or_calibrate",
    "normalize_extractor_output",
]
