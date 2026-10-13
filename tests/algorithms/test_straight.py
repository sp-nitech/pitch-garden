"""Tests for the STRAIGHT algorithm."""

import numpy as np
import pytest

from pitch_garden import Tracker

from ..helpers import generate_pulse_train

pyls = pytest.importorskip("pylstraight")


def generate_utterance(f0: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a pulse train of 1 second surrounded by silence of 0.5 seconds.

    The voicing decision of STRAIGHT relies on the contrast with the silence.

    """
    silence = np.zeros(sample_rate // 2)
    voice = generate_pulse_train(f0, sample_rate=sample_rate)
    return np.concatenate([silence, voice, silence])


def test_identical_to_pylstraight() -> None:
    """Test that the native contour is identical to that of pylstraight."""
    x = generate_utterance(200.0)
    pitch = Tracker("straight", fmin=60.0, fmax=500.0)(x, 16000, frame_period=5.0)
    expected = pyls.extract_f0(
        x, 16000, frame_shift=5.0, f0_range=(60.0, 500.0), seed=0
    )
    assert pitch.native is not None
    np.testing.assert_array_equal(pitch.native.f0, expected)
    np.testing.assert_allclose(pitch.native.times, np.arange(len(expected)) * 0.005)
