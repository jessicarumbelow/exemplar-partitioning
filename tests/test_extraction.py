"""Tests for ExtractionResult and normalize_extractor_output."""

from __future__ import annotations

import numpy as np
import pytest

from ep.discovery.extraction import (
    ExtractionResult,
    normalize_extractor_output,
)


class TestExtractionResult:
    def test_basic_construction(self):
        n, d = 10, 8
        x = np.random.default_rng(0).standard_normal((n, d)).astype(np.float32)
        result = ExtractionResult(x=x)
        assert result.x.shape == (n, d)
        assert result.prompt_ids.shape == (0,)
        assert result.position_ids.shape == (0,)

    def test_with_ids(self):
        x = np.ones((3, 4))
        pids = np.array([0, 0, 1], dtype=np.int64)
        posids = np.array([0, 1, 0], dtype=np.int64)
        result = ExtractionResult(x=x, prompt_ids=pids, position_ids=posids)
        np.testing.assert_array_equal(result.prompt_ids, pids)
        np.testing.assert_array_equal(result.position_ids, posids)

    def test_pass_counts(self):
        x = np.ones((3, 4))
        result = ExtractionResult(x=x, n_forward_passes=5, n_tokens=100)
        assert result.n_forward_passes == 5
        assert result.n_tokens == 100


class TestNormalizeExtractorOutput:
    def test_passthrough_extraction_result(self):
        x = np.ones((3, 4))
        result = ExtractionResult(x=x)
        assert normalize_extractor_output(result) is result

    def test_ndarray_conversion(self):
        x = np.ones((3, 4))
        result = normalize_extractor_output(x)
        assert isinstance(result, ExtractionResult)
        np.testing.assert_allclose(result.x, x)

    def test_one_tuple(self):
        x = np.ones((3, 4))
        result = normalize_extractor_output((x,))
        assert isinstance(result, ExtractionResult)
        np.testing.assert_allclose(result.x, x)

    def test_three_tuple_conversion(self):
        x = np.ones((3, 4))
        pids = np.zeros(3, dtype=np.int64)
        posids = np.arange(3, dtype=np.int64)
        result = normalize_extractor_output((x, pids, posids))
        assert isinstance(result, ExtractionResult)
        np.testing.assert_allclose(result.x, x)
        np.testing.assert_array_equal(result.prompt_ids, pids)
        np.testing.assert_array_equal(result.position_ids, posids)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            normalize_extractor_output("invalid")
