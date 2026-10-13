"""Tests for the utilities for loading the estimated F0 contours."""

import numpy as np

from pitch_garden import Pitch

from .loader import load_module

estimates = load_module("estimates")


def test_load_estimate(tmp_path) -> None:
    """Test that the estimate is put on the grid without interpolation."""
    path = tmp_path / "a.npz"
    f0 = np.array([100.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 300.0])
    pitch = Pitch(f0=f0, frame_period=4.0, sample_rate=16000, source="t")
    pitch.tofile(str(path), output_format="linear", file_type="npz")
    np.testing.assert_allclose(
        estimates.load_estimate(str(path), 4, 16.0), [100, 200, 300, 0]
    )
