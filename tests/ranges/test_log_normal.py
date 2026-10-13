"""Tests for the F0 range estimator based on the log-normal distribution."""

import numpy as np
import pytest

from pitch_garden.ranges.log_normal import LogNormalRangeEstimator


def test_log_normal() -> None:
    """Test the estimator based on the mean and the standard deviation of log F0."""
    rng = np.random.default_rng(0)
    values = np.exp(rng.normal(np.log(200.0), 0.1, size=1000))
    estimator = LogNormalRangeEstimator(gamma=3.0)
    result = estimator(values, default=(40.0, 2000.0))
    mu, sigma = np.mean(np.log(values)), np.std(np.log(values))
    assert result.fmin == pytest.approx(np.exp(mu - 3.0 * sigma))
    assert result.fmax == pytest.approx(np.exp(mu + 3.0 * sigma))


def test_log_normal_invalid_parameters() -> None:
    """Test that invalid parameters are rejected."""
    with pytest.raises(ValueError):
        LogNormalRangeEstimator(gamma=0.0)
    with pytest.raises(ValueError):
        LogNormalRangeEstimator(min_octaves=-1.0)


def test_log_normal_steady_tone() -> None:
    """Test that the range of a steady tone has the minimum width."""
    estimator = LogNormalRangeEstimator(min_octaves=1.0)
    result = estimator(np.full(100, 150.0), default=(40.0, 2000.0))
    assert result.fmin == pytest.approx(150.0 / np.sqrt(2.0))
    assert result.fmax == pytest.approx(150.0 * np.sqrt(2.0))
