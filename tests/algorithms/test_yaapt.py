"""Tests for the YAAPT algorithm."""

import warnings

import numpy as np
import pytest

from pitch_garden import Tracker

from ..helpers import cent_error, generate_pulse_train


def test_identical_to_amfm_decompy() -> None:
    """Test that the F0 and the frame positions are those of AMFM_decompy."""
    import amfm_decompy.basic_tools as basic
    import amfm_decompy.pYAAPT as pyaapt

    x = generate_pulse_train(150.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = pyaapt.yaapt(
            basic.SignalObj(x.copy(), 16000),
            frame_space=10.0,
            f0_min=40.0,
            f0_max=800.0,
            nlfer_thresh1=0.5,
        )
    tracker = Tracker("yaapt", voicing_threshold=0.5, two_pass_policy=None)
    pitch = tracker(x, 16000, frame_period=10.0)
    assert pitch.native is not None
    np.testing.assert_array_equal(pitch.native.f0, expected.samp_values)
    np.testing.assert_array_equal(pitch.native.times, expected.frames_pos / 16000)


def test_utterance() -> None:
    """Test that the F0 is tracked in an utterance surrounded by silence."""
    silence = np.zeros(8000)
    x = np.concatenate([silence, generate_pulse_train(150.0), silence])
    pitch = Tracker("yaapt")(x, 16000, frame_period=10.0)
    inner = pitch.f0[60:141]  # 0.6-1.4 seconds
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)
    np.testing.assert_array_equal(pitch.f0[:40], 0.0)  # 0-0.4 seconds


def test_search_range() -> None:
    """Test that the F0 out of the search range is not estimated."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("yaapt", fmin=200.0, fmax=400.0)(x, 16000, frame_period=10.0)
    voiced = pitch.f0[0 < pitch.f0]
    assert np.all((190.0 <= voiced) & (voiced <= 410.0))


@pytest.mark.parametrize("duration_ms", [10, 30, 60, 100])
def test_short_audio(duration_ms: int) -> None:
    """Test that short audio, which AMFM_decompy fails on, is tracked."""
    x = generate_pulse_train(150.0)[: 16 * duration_ms]
    pitch = Tracker("yaapt")(x, 16000, frame_period=5.0)
    assert pitch.length == int(np.ceil(duration_ms / 5))


def test_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the warnings and the messages of AMFM_decompy are suppressed."""
    x = np.concatenate([np.zeros(8000), generate_pulse_train(150.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Tracker("yaapt", fmin=10.0, fmax=8000.0)(x, 16000, frame_period=10.0)
    assert capsys.readouterr().out == ""
