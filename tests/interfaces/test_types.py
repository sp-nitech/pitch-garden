"""Tests for the data types."""

from typing import Any, get_args

import numpy as np
import pytest

from pitch_garden import Audio, Pitch
from pitch_garden.interfaces.types import (
    LOG_ZERO,
    FileType,
    NativeContour,
    OutputFormat,
)


def make_pitch(**kwargs) -> Pitch:
    """Create a small pitch object for testing."""
    params: dict[str, Any] = dict(
        f0=np.array([0.0, 100.0, 200.0, 0.0]),
        frame_period=5.0,
        sample_rate=16000,
        source="test",
    )
    params.update(kwargs)
    return Pitch(**params)


def test_audio_is_mono_float64() -> None:
    """Test that the audio is converted to a 1-D float64 array."""
    audio = Audio(np.zeros(10, dtype=np.float32), 16000)
    assert audio.data.dtype == np.float64
    assert audio.length == 10
    assert audio.duration == pytest.approx(10 / 16000)


def test_audio_rejects_invalid_input() -> None:
    """Test that invalid audio is rejected."""
    with pytest.raises(ValueError):
        Audio(np.zeros((2, 10)), 16000)
    with pytest.raises(ValueError):
        Audio(np.zeros(10), 0)


def test_pitch_voiced_flags() -> None:
    """Test that voiced flags are derived from F0."""
    pitch = make_pitch()
    np.testing.assert_array_equal(pitch.voiced, [False, True, True, False])
    assert pitch.length == 4


def test_pitch_rejects_negative_f0() -> None:
    """Test that negative F0 values are rejected."""
    with pytest.raises(ValueError):
        make_pitch(f0=np.array([-1.0]))


@pytest.mark.parametrize("frame_period", [0.0, -5.0, np.nan, np.inf])
def test_pitch_rejects_invalid_frame_period(frame_period: float) -> None:
    """Test that the frame period must be a positive finite number."""
    with pytest.raises(ValueError, match="frame_period"):
        make_pitch(frame_period=frame_period)


def test_pitch_to_linear() -> None:
    """Test conversion to F0 in Hz."""
    np.testing.assert_array_equal(make_pitch().to("linear"), [0, 100, 200, 0])


def test_pitch_to_log() -> None:
    """Test conversion to log F0 with the SPTK magic number."""
    lf0 = make_pitch().to("log")
    np.testing.assert_allclose(lf0, [LOG_ZERO, np.log(100), np.log(200), LOG_ZERO])


def test_pitch_to_inverse() -> None:
    """Test conversion to the pitch period in samples (Fs/F0)."""
    np.testing.assert_allclose(make_pitch().to("inverse"), [0, 160, 80, 0])


def test_pitch_to_unknown_format() -> None:
    """Test that an unknown format is rejected with the supported ones."""
    with pytest.raises(ValueError) as e:
        make_pitch().to("hz")
    assert str(list(get_args(OutputFormat))) in str(e.value)


def test_pitch_tofile_unknown_file_type(tmp_path) -> None:
    """Test that an unknown file type is rejected with the supported ones."""
    with pytest.raises(ValueError) as e:
        make_pitch().tofile(str(tmp_path / "a"), file_type="text")
    assert str(list(get_args(FileType))) in str(e.value)


def test_pitch_with_threshold() -> None:
    """Test re-thresholding the voicing decision on the native frames."""
    native = NativeContour(
        times=np.arange(4) * 0.005,
        f0=np.array([0.0, 100.0, 200.0, 0.0]),
        score=np.array([0.1, 0.6, 0.9, 0.4]),
        candidate=np.array([90.0, 100.0, 200.0, 300.0]),
    )
    pitch = make_pitch(native=native)
    np.testing.assert_array_equal(pitch.with_threshold(0.3).f0, [0, 100, 200, 300])
    np.testing.assert_array_equal(pitch.with_threshold(0.8).f0, [0, 0, 200, 0])
    # Re-thresholding is repeatable from the original native contour.
    again = pitch.with_threshold(0.8).with_threshold(0.3)
    np.testing.assert_array_equal(again.f0, [0, 100, 200, 300])


def test_pitch_with_threshold_requires_score() -> None:
    """Test that re-thresholding requires the score with a hint of re-extraction."""
    native = NativeContour(times=np.zeros(4), f0=np.zeros(4))
    for pitch in (make_pitch(), make_pitch(native=native)):
        with pytest.raises(ValueError, match="replace"):
            pitch.with_threshold(0.5)


def test_pitch_with_f0_drops_native() -> None:
    """Test that replacing F0 drops the native contour, which becomes stale."""
    native = NativeContour(times=np.zeros(1), f0=np.zeros(1))
    assert make_pitch(native=native).with_f0(np.zeros(4)).native is None


def test_pitch_tofile(tmp_path) -> None:
    """Test saving pitch to files."""
    pitch = make_pitch()

    pitch.tofile(str(tmp_path / "a.lf0"), output_format="log", file_type="double")
    data = np.fromfile(tmp_path / "a.lf0", dtype=np.float64)
    np.testing.assert_allclose(data, pitch.to("log"))

    pitch.tofile(str(tmp_path / "a.f0"), output_format="linear", file_type="float")
    data = np.fromfile(tmp_path / "a.f0", dtype=np.float32)
    np.testing.assert_allclose(data, pitch.to("linear"))

    pitch.tofile(str(tmp_path / "a.npz"), output_format="linear", file_type="npz")
    with np.load(tmp_path / "a.npz") as data:
        np.testing.assert_allclose(data["f0"], pitch.to("linear"))
        assert data["frame_period"] == 5.0
        assert str(data["source"]) == "test"


def test_pitch_repr_is_concise() -> None:
    """Test that the representation summarizes the contour without arrays."""
    text = repr(make_pitch(score=np.zeros(4)))
    assert text == (
        "Pitch(source='test', num_frames=4, num_voiced_frames=2, "
        "frame_period=5.0, sample_rate=16000)"
    )
    assert str(make_pitch()) == repr(make_pitch())


def test_pitch_to_grid() -> None:
    """Test that a contour is put on another grid, unvoiced beyond its end."""
    pitch = make_pitch(f0=np.array([0.0, 100.0, 100.0, 100.0]), frame_period=10.0)
    grid = pitch.to_grid(9, frame_period=4.0)
    assert (grid.length, grid.frame_period) == (9, 4.0)
    # The frames at 0-32 ms, where 4 ms is nearest to the unvoiced frame at 0 ms,
    # and 32 ms is beyond the last frame at 30 ms.
    np.testing.assert_allclose(grid.f0, [0.0, 0.0] + [100.0] * 6 + [0.0])
    assert make_pitch(f0=np.zeros(0)).to_grid(3, frame_period=5.0).length == 3
