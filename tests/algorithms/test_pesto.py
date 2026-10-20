"""Tests for the PESTO algorithm."""

import importlib.util
import sys

import numpy as np
import pytest

from pitch_garden import Tracker

from ..helpers import cent_error, generate_pulse_train

# pesto-pitch is not installed on Python 3.15 or later.
requires_pesto = pytest.mark.skipif(
    importlib.util.find_spec("pesto") is None, reason="pesto-pitch is not installed."
)


def generate_utterance(f0: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a pulse train of 1 second surrounded by silence of 0.5 seconds."""
    silence = np.zeros(sample_rate // 2)
    voice = generate_pulse_train(f0, sample_rate=sample_rate)
    return np.concatenate([silence, voice, silence])


@requires_pesto
def test_utterance() -> None:
    """Test that the F0 is tracked in an utterance surrounded by silence."""
    pitch = Tracker("pesto")(generate_utterance(150.0), 16000, frame_period=10.0)
    inner = pitch.f0[60:141]  # 0.6-1.4 seconds
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)


@requires_pesto
def test_search_range() -> None:
    """Test that the F0 out of the search range is not estimated."""
    tracker = Tracker("pesto", fmin=200.0, fmax=400.0)
    pitch = tracker(generate_utterance(150.0), 16000, frame_period=10.0)
    assert pitch.native is not None and pitch.native.candidate is not None
    candidate = pitch.native.candidate
    assert np.all((candidate == 0) | (190.0 <= candidate))


@requires_pesto
def test_frame_period_can_be_changed() -> None:
    """Test that the frame period can be changed at every call."""
    tracker = Tracker("pesto")
    x = generate_utterance(150.0)
    for frame_period in (10.0, 5.0):
        pitch = tracker(x, 16000, frame_period=frame_period)
        assert pitch.native is not None
        np.testing.assert_allclose(np.diff(pitch.native.times), frame_period / 1000)


@requires_pesto
def test_any_sample_rate() -> None:
    """Test that the audio is processed at its own sample rate."""
    pitch = Tracker("pesto")(generate_utterance(150.0, 44100), 44100, frame_period=10.0)
    assert np.all(cent_error(pitch.f0[60:141], 150.0) < 50)


def test_missing_package(monkeypatch) -> None:
    """Test that the error tells the Python versions supporting pesto-pitch."""
    monkeypatch.setitem(sys.modules, "pesto", None)
    with pytest.raises(ImportError, match="Python 3.14 or earlier"):
        Tracker("pesto").load()


def test_auto_device() -> None:
    """Test that a GPU is used by default if available."""
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert Tracker("pesto").config.device == expected
