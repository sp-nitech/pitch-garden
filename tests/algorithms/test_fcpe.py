"""Tests for the FCPE algorithm."""

import numpy as np

from pitch_garden import Tracker

from ..helpers import cent_error, generate_pulse_train


def generate_glide(sample_rate: int = 16000) -> np.ndarray:
    """Generate a harmonic tone gliding from 120 to 180 Hz in 1 second with noise."""
    t = np.arange(sample_rate) / sample_rate
    phase = 2 * np.pi * np.cumsum(120 + 60 * t) / sample_rate
    x = sum(np.sin(k * phase) / k for k in range(1, 11)) / 3
    return x + 0.01 * np.random.default_rng(0).standard_normal(sample_rate)


def test_identical_to_fcpe() -> None:
    """Test that the F0 and the confidence are identical to those of FCPE.

    The expected values are given by torchfcpe 0.0.4 with the local argmax decoder
    of the bundled model, where the confidence is the maximum of the latent.

    """
    pitch = Tracker("fcpe")(generate_glide(), 16000, frame_period=10.0)
    native = pitch.native
    assert native is not None
    assert native.candidate is not None and native.score is not None
    assert len(native.candidate) == 101
    index = [0, 10, 50, 100]
    np.testing.assert_allclose(
        native.candidate[index],
        [120.51902193251132, 126.10994166061118, 150.0803050676314, 180.34261699132284],
        rtol=1e-5,
    )
    np.testing.assert_allclose(native.candidate.sum(), 15153.421918236123, rtol=1e-5)
    np.testing.assert_allclose(
        native.score[index],
        [
            0.7996346354484558,
            0.9851963520050049,
            0.9487770795822144,
            0.8616669774055481,
        ],
        atol=1e-4,
    )
    np.testing.assert_allclose(native.score.sum(), 97.84013992547989, rtol=1e-4)


def test_frames_at_hops() -> None:
    """Test that the outputs are at the hops as trained in FCPE.

    The windows are centered 80 samples after the hops, but the model is trained to
    give the F0 at the hops, and the times of the centers degrade the accuracy.

    """
    pitch = Tracker("fcpe")(generate_glide(), 16000, frame_period=5.0)
    assert pitch.native is not None
    np.testing.assert_allclose(pitch.native.times, np.arange(101) * 0.01, atol=1e-12)
    assert pitch.length == 200


def test_utterance() -> None:
    """Test that the F0 is tracked in an utterance surrounded by silence."""
    silence = np.zeros(8000)
    x = np.concatenate([silence, generate_pulse_train(150.0), silence])
    pitch = Tracker("fcpe")(x, 16000, frame_period=10.0)
    inner = pitch.f0[60:141]  # 0.6-1.4 seconds
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)
    np.testing.assert_array_equal(pitch.f0[:40], 0.0)  # 0-0.4 seconds


def test_search_range() -> None:
    """Test that the F0 out of the search range is not estimated."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("fcpe", fmin=200.0, fmax=400.0)(x, 16000)
    assert pitch.native is not None and pitch.native.candidate is not None
    assert np.all((190.0 <= pitch.native.candidate) & (pitch.native.candidate <= 410.0))


def test_auto_device() -> None:
    """Test that a GPU is used by default if available."""
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert Tracker("fcpe").config.device == expected


def test_not_parallel_safe_on_gpu() -> None:
    """Test that FCPE on a GPU is run in the main process."""
    assert Tracker("fcpe", device="cpu").parallel_safe
    assert not Tracker("fcpe", device="cuda").parallel_safe
