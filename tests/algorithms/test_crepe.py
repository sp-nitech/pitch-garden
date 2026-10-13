"""Tests for the CREPE algorithm."""

import numpy as np

from pitch_garden import Tracker
from pitch_garden.algorithms.crepe import filter_window, median_filter

from ..helpers import cent_error, generate_pulse_train


def test_tone() -> None:
    """Test that the F0 of a voice-like pulse train is estimated within 50 cents."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("crepe", model="tiny")(x, 16000, frame_period=10.0)
    inner = pitch.f0[10:-10]
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)


def test_utterance() -> None:
    """Test that the F0 is tracked in an utterance surrounded by silence."""
    silence = np.zeros(8000)
    x = np.concatenate([silence, generate_pulse_train(150.0), silence])
    pitch = Tracker("crepe", model="tiny")(x, 16000, frame_period=10.0)
    inner = pitch.f0[60:141]  # 0.6-1.4 seconds
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)


def test_deterministic() -> None:
    """Test that the results are identical at every call."""
    x = generate_pulse_train(150.0)
    tracker = Tracker("crepe", model="tiny")
    np.testing.assert_array_equal(tracker(x, 16000).f0, tracker(x, 16000).f0)


def test_search_range() -> None:
    """Test that the F0 out of the search range is not estimated."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("crepe", model="tiny", fmin=200.0, fmax=400.0)(x, 16000)
    assert pitch.native is not None and pitch.native.candidate is not None
    assert np.all(190.0 <= pitch.native.candidate)


def test_auto_device() -> None:
    """Test that a GPU is used by default if available."""
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert Tracker("crepe").config.device == expected


def test_not_parallel_safe_on_gpu() -> None:
    """Test that CREPE on a GPU is run in the main process."""
    assert Tracker("crepe", device="cpu").parallel_safe
    assert not Tracker("crepe", device="cuda").parallel_safe


def test_silence_has_no_periodicity() -> None:
    """Test that the periodicity of silent frames is zero if the threshold is given."""
    rng = np.random.default_rng(0)
    noise = 1e-4 * rng.standard_normal(8000)
    x = np.concatenate([noise, generate_pulse_train(150.0)])
    tracker = Tracker("crepe", model="tiny", silence_threshold=-60.0)
    pitch = tracker(x, 16000, frame_period=10.0)
    assert pitch.score is not None
    np.testing.assert_array_equal(pitch.score[:40], 0.0)  # 0-0.4 seconds
    assert np.all(0 < pitch.score[60:140])
    tracker = Tracker("crepe", model="tiny", silence_threshold=None)
    ungated = tracker(x, 16000, frame_period=10.0)
    assert ungated.score is not None
    assert np.all(0 < ungated.score[:40])


def test_median_filter() -> None:
    """Test the median filter truncated at the edges as in torchcrepe."""
    x = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    # The lower median is taken for the truncated windows of two values.
    np.testing.assert_array_equal(median_filter(x, 3), [0.0, 0.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(median_filter(x, 1), x)


def test_filter_length_in_milliseconds() -> None:
    """Test that the filter length is converted into an odd number of frames."""
    assert filter_window(15.0, 5.0) == 3
    assert filter_window(15.0, 1.0) == 15
    assert filter_window(15.0, 10.0) == 1
    assert filter_window(0.0, 5.0) == 1
