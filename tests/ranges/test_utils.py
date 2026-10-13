"""Tests for the utility functions for the F0 search range."""

import numpy as np

from pitch_garden import Pitch
from pitch_garden.ranges._utils import estimate_f0_range

from ..dummy import MedianScaleRangeEstimator


def make_pitch(f0: np.ndarray) -> Pitch:
    """Create a pitch object."""
    return Pitch(f0=f0, frame_period=5.0, sample_rate=16000, source="test")


def test_estimate_from_multiple_pitches() -> None:
    """Test that voiced frames of multiple contours are pooled."""
    a = make_pitch(np.array([0.0, 100.0, 100.0]))
    b = make_pitch(np.array([200.0, 0.0, 200.0]))
    estimator = MedianScaleRangeEstimator(lower=0.5, upper=2.0)
    result = estimate_f0_range([a, b], estimator, default=(40.0, 2000.0))
    # The median of the pooled frames [100, 100, 200, 200] is 150 Hz.
    assert result.num_voiced_frames == 4
    assert (result.fmin, result.fmax) == (75.0, 300.0)
