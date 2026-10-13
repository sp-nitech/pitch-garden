"""Tests for the YIN algorithm."""

import numpy as np

from pitch_garden import Tracker

from ..helpers import cent_error, generate_pulse_train


def generate_glide(sample_rate: int = 16000) -> np.ndarray:
    """Generate a harmonic tone gliding from 120 to 180 Hz in 1 second with noise."""
    t = np.arange(sample_rate) / sample_rate
    phase = 2 * np.pi * np.cumsum(120 + 60 * t) / sample_rate
    x = sum(np.sin(k * phase) / k for k in range(1, 11))
    return x + 0.01 * np.random.default_rng(0).standard_normal(sample_rate)


def test_identical_to_librosa() -> None:
    """Test that the F0 candidates are identical to those of librosa.

    The expected values are given by librosa 1.0.0 with `librosa.yin(x, fmin=40,
    fmax=800, sr=16000, frame_length=1024, hop_length=80)`.

    """
    tracker = Tracker("yin", fmin=40.0, fmax=800.0)
    pitch = tracker(generate_glide(), 16000, frame_period=5.0)
    assert pitch.native is not None and pitch.native.candidate is not None
    candidate = pitch.native.candidate
    assert len(candidate) == 201
    np.testing.assert_allclose(
        candidate[[0, 10, 50, 100, 150, 190, 200]],
        [
            121.1759727918583,
            123.33464006689742,
            135.1206252367632,
            150.05383379722488,
            165.11587205128146,
            177.06104249180413,
            179.1731060640721,
        ],
        rtol=1e-9,
    )
    np.testing.assert_allclose(candidate.sum(), 30166.086949052253, rtol=1e-9)


def test_utterance() -> None:
    """Test that the F0 is tracked and silence is unvoiced by the aperiodicity."""
    silence = np.zeros(8000)
    x = np.concatenate([silence, generate_pulse_train(150.0), silence])
    pitch = Tracker("yin")(x, 16000, frame_period=10.0)
    inner = pitch.f0[60:141]  # 0.6-1.4 seconds
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)
    np.testing.assert_array_equal(pitch.f0[:40], 0.0)  # 0-0.4 seconds


def test_search_range() -> None:
    """Test that the F0 out of the search range is not estimated."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("yin", fmin=200.0, fmax=400.0)(x, 16000, frame_period=10.0)
    assert pitch.native is not None and pitch.native.candidate is not None
    assert np.all(190.0 <= pitch.native.candidate)
