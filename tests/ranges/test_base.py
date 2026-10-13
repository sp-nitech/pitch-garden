"""Tests for the base class of the F0 range estimators."""

import logging

import numpy as np
import pytest

from ..dummy import MedianScaleRangeEstimator


def test_range_is_clipped_to_default() -> None:
    """Test that the estimated range does not exceed the first-pass range."""
    values = np.full(100, 1500.0)
    estimator = MedianScaleRangeEstimator()
    result = estimator(values, default=(40.0, 2000.0))
    assert result.fmax == 2000.0


def test_fallback_without_voiced_frames(caplog) -> None:
    """Test that the first-pass range is used with a warning every time."""
    estimator = MedianScaleRangeEstimator()
    with caplog.at_level(logging.WARNING, logger="pitch_garden"):
        results = [estimator(np.zeros(5), default=(40.0, 2000.0)) for _ in "abc"]
    result = results[-1]
    assert (result.fmin, result.fmax) == (40.0, 2000.0)
    assert result.fallback
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3


def test_single_voiced_frame_is_used() -> None:
    """Test that the range is estimated even from a single voiced frame."""
    estimator = MedianScaleRangeEstimator()
    result = estimator(np.array([0.0, 200.0, 0.0]), default=(40.0, 2000.0))
    assert (result.fmin, result.fmax) == (150.0, 300.0)
    assert not result.fallback


def test_narrow_range_is_widened() -> None:
    """Test that a range narrower than the minimum is widened around its center."""
    values = np.full(100, 200.0)
    estimator = MedianScaleRangeEstimator(1.0, 1.0)
    result = estimator(values, default=(40.0, 2000.0))
    width = estimator.min_octaves / 2
    assert result.fmin == pytest.approx(200.0 * 2**-width)
    assert result.fmax == pytest.approx(200.0 * 2**width)
    assert not result.fallback
