"""Tests for the registry of the F0 range estimators."""

import pytest

from pitch_garden.ranges import RANGE_ESTIMATOR_MAP, create_range_estimator
from pitch_garden.ranges.log_normal import LogNormalRangeEstimator


def test_registered() -> None:
    """Test that the estimators are registered."""
    assert RANGE_ESTIMATOR_MAP["log-normal"] is LogNormalRangeEstimator
    estimator = create_range_estimator("log-normal", gamma=2.0)
    assert isinstance(estimator, LogNormalRangeEstimator)
    with pytest.raises(ValueError):
        create_range_estimator("unknown")
