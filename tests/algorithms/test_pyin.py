"""Tests for the pYIN algorithm."""

import numpy as np

from pitch_garden import Tracker

from ..helpers import generate_pulse_train


def generate_glide(sample_rate: int = 16000) -> np.ndarray:
    """Generate a harmonic tone gliding from 120 to 180 Hz surrounded by silence."""
    t = np.arange(sample_rate) / sample_rate
    phase = 2 * np.pi * np.cumsum(120 + 60 * t) / sample_rate
    x = sum(np.sin(k * phase) / k for k in range(1, 11))
    x = x + 0.01 * np.random.default_rng(0).standard_normal(sample_rate)
    silence = np.zeros(sample_rate // 4)
    return np.concatenate([silence, x, silence])


def test_identical_to_librosa() -> None:
    """Test that the F0 is identical to that of librosa.

    The expected values are given by librosa 1.0.0 with `librosa.pyin(x, fmin=40,
    fmax=800, sr=16000, frame_length=1024, hop_length=80)`.

    """
    tracker = Tracker("pyin", fmin=40.0, fmax=800.0)
    pitch = tracker(generate_glide(), 16000, frame_period=5.0)
    assert pitch.native is not None
    f0 = pitch.native.f0
    voiced = np.flatnonzero(0 < f0)
    assert (len(f0), len(voiced), voiced[0], voiced[-1]) == (301, 210, 46, 255)
    np.testing.assert_allclose(
        f0[[60, 100, 150, 200, 240]],
        [
            123.37686603263526,
            135.32282858103176,
            150.1500794673829,
            164.6883578629587,
            177.53111553085517,
        ],
        rtol=1e-9,
    )
    np.testing.assert_allclose(f0.sum(), 31547.342135830608, rtol=1e-9)


def test_search_range() -> None:
    """Test that the F0 out of the search range is not estimated."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("pyin", fmin=200.0, fmax=400.0)(x, 16000, frame_period=10.0)
    voiced = pitch.f0[0 < pitch.f0]
    assert np.all((200.0 <= voiced) & (voiced <= 400.0))
