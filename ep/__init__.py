"""Exemplar Partitioning: training-free feature discovery for LLM activations."""

from .discovery import (
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
from .utils import set_seed

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
    "set_seed",
]
