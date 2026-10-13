"""Tests for the PiENet algorithm."""

import numpy as np

from pitch_garden import Tracker

from ..helpers import cent_error, generate_pulse_train


def generate_glide(sample_rate: int = 16000) -> np.ndarray:
    """Generate a harmonic tone gliding from 120 to 180 Hz in 1 second with noise."""
    t = np.arange(sample_rate) / sample_rate
    phase = 2 * np.pi * np.cumsum(120 + 60 * t) / sample_rate
    x = sum(np.sin(k * phase) / k for k in range(1, 11))
    return x + 0.01 * np.random.default_rng(0).standard_normal(sample_rate)


def test_identical_to_pienet() -> None:
    """Test that the F0 and the voicing probability are identical to those of PiENet.

    The expected values are given by `F0Estimator.probabilities` of PiENet at commit
    2c5e7eb with the frames of `frame_audio`, where the frequencies of the bins are
    in single precision.

    """
    pitch = Tracker("pienet")(generate_glide(), 16000, frame_period=10.0)
    native = pitch.native
    assert native is not None
    assert native.candidate is not None and native.score is not None
    assert len(native.candidate) == 101
    index = [0, 10, 50, 100]
    np.testing.assert_allclose(
        native.candidate[index],
        [121.0386734008789, 126.75977325439453, 150.48043823242188, 187.08367919921875],
        rtol=1e-6,
    )
    np.testing.assert_allclose(native.candidate.sum(), 15198.006004333496, rtol=1e-6)
    np.testing.assert_allclose(
        native.score[index],
        [
            0.9946742467582226,
            0.9999999999999997,
            0.9999999999999998,
            0.059664368629455566,
        ],
        atol=1e-4,
    )
    np.testing.assert_allclose(native.score.sum(), 98.04212720591138, rtol=1e-4)


def test_independent_of_batch_size() -> None:
    """Test that the frames analyzed in batches are identical to those at once."""
    x = np.tile(generate_glide(), 3)
    native = [
        Tracker("pienet", batch_size=batch_size, device="cpu")(x, 16000).native
        for batch_size in (37, 1000)
    ]
    assert native[0] is not None and native[1] is not None
    assert native[0].score is not None and native[1].score is not None
    np.testing.assert_array_equal(native[0].candidate, native[1].candidate)
    np.testing.assert_allclose(native[0].score, native[1].score, atol=1e-6)


def test_fixed_frame_period() -> None:
    """Test that the frames are analyzed every 10 milliseconds for any frame period."""
    pitch = Tracker("pienet")(generate_glide(), 16000, frame_period=5.0)
    assert pitch.native is not None
    np.testing.assert_allclose(np.diff(pitch.native.times), 0.01)
    assert pitch.length == 200


def test_utterance() -> None:
    """Test that the F0 is tracked in an utterance surrounded by silence."""
    silence = np.zeros(8000)
    x = np.concatenate([silence, generate_pulse_train(150.0), silence])
    pitch = Tracker("pienet")(x, 16000, frame_period=10.0)
    inner = pitch.f0[60:141]  # 0.6-1.4 seconds
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)
    np.testing.assert_array_equal(pitch.f0[:40], 0.0)  # 0-0.4 seconds


def test_search_range() -> None:
    """Test that the F0 out of the search range is not estimated."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("pienet", fmin=200.0, fmax=400.0)(x, 16000)
    assert pitch.native is not None and pitch.native.candidate is not None
    assert np.all((200.0 <= pitch.native.candidate) & (pitch.native.candidate <= 400.0))


def test_auto_device() -> None:
    """Test that a GPU is used by default if available."""
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert Tracker("pienet").config.device == expected


def test_not_parallel_safe_on_gpu() -> None:
    """Test that PiENet on a GPU is run in the main process."""
    assert Tracker("pienet", device="cpu").parallel_safe
    assert not Tracker("pienet", device="cuda").parallel_safe
