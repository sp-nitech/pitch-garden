"""Tests for the RMVPE algorithm."""

import numpy as np

from pitch_garden import Tracker

from ..helpers import cent_error, generate_pulse_train


def generate_utterance(f0: float) -> np.ndarray:
    """Generate a pulse train of 1 second surrounded by silence of 0.5 seconds."""
    silence = np.zeros(8000)
    return np.concatenate([silence, generate_pulse_train(f0), silence])


def test_utterance() -> None:
    """Test that the F0 is tracked in an utterance surrounded by silence."""
    pitch = Tracker("rmvpe")(generate_utterance(150.0), 16000, frame_period=10.0)
    inner = pitch.f0[60:141]  # 0.6-1.4 seconds
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)


def test_search_range() -> None:
    """Test that the F0 out of the search range is not estimated."""
    tracker = Tracker("rmvpe", fmin=200.0, fmax=400.0)
    pitch = tracker(generate_utterance(150.0), 16000, frame_period=10.0)
    assert pitch.native is not None and pitch.native.candidate is not None
    assert np.all(190.0 <= pitch.native.candidate)


def test_auto_device() -> None:
    """Test that a GPU is used by default if available."""
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert Tracker("rmvpe").config.device == expected
