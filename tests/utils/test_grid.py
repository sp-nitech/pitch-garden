"""Tests for the frame grid utilities."""

import numpy as np
import pytest

from pitch_garden.utils.grid import get_num_frames, resample


@pytest.mark.parametrize(
    "num_samples, expected", [(1599, 20), (1600, 20), (1601, 21), (1, 1), (0, 0)]
)
def test_get_num_frames(num_samples: int, expected: int) -> None:
    """Test that the number of frames is ceil(T / hop) like SPTK."""
    assert get_num_frames(num_samples, 16000, 5.0) == expected


def test_get_num_frames_with_fractional_hop() -> None:
    """Test the number of frames when the hop size is not an integer."""
    # 5 ms at 44.1 kHz is 220.5 samples.
    assert get_num_frames(441, 44100, 5.0) == 2
    assert get_num_frames(442, 44100, 5.0) == 3


def test_resample_identity() -> None:
    """Test that resampling onto the same grid keeps the values."""
    times = np.arange(4) * 0.005
    f0 = np.array([0.0, 100.0, 200.0, 0.0])
    out = resample(times, f0, num_frames=4, frame_period=5.0)
    np.testing.assert_allclose(out.f0, f0)


def test_resample_log_linear_interpolation() -> None:
    """Test that voiced frames are interpolated in the log domain."""
    times = np.array([0.0, 0.016])
    f0 = np.array([100.0, 400.0])
    out = resample(times, f0, num_frames=4, frame_period=8.0)
    # The frame at 8 ms is the geometric mean of 100 and 400 Hz.
    np.testing.assert_allclose(out.f0[:3], [100.0, 200.0, 400.0])
    # The frame at 24 ms is beyond the last native frame.
    assert out.f0[3] == pytest.approx(400.0)


def test_resample_voicing_follows_nearest_frame() -> None:
    """Test that the voicing decision follows the nearest native frame."""
    times = np.array([0.0, 0.010])
    f0 = np.array([100.0, 0.0])
    out = resample(times, f0, num_frames=6, frame_period=2.0)
    # Frames at 0, 2, 4 ms are close to the voiced frame (ties go to the left).
    np.testing.assert_array_equal(out.f0, [100, 100, 100, 0, 0, 0])


def test_resample_score() -> None:
    """Test that the score follows the nearest native frame."""
    times = np.array([0.0, 0.010])
    f0 = np.array([0.0, 200.0])
    score = np.array([0.2, 0.8])
    out = resample(times, f0, score=score, num_frames=3, frame_period=5.0)
    assert out.score is not None
    np.testing.assert_allclose(out.score, [0.2, 0.2, 0.8])
    np.testing.assert_allclose(out.f0, [0.0, 0.0, 200.0])


def test_resample_empty() -> None:
    """Test that resampling an empty contour returns unvoiced frames."""
    out = resample(np.zeros(0), np.zeros(0), num_frames=3, frame_period=5.0)
    np.testing.assert_array_equal(out.f0, [0, 0, 0])


def test_resample_does_not_interpolate_with_unvoiced_frame() -> None:
    """Test that a voiced frame next to an unvoiced one takes the nearest value."""
    times = np.array([0.0, 0.016])
    f0 = np.array([200.0, 0.0])
    out = resample(times, f0, num_frames=3, frame_period=5.0)
    np.testing.assert_allclose(out.f0, [200.0, 200.0, 0.0])
