"""Tests for the FCN-f0 algorithm."""

import numpy as np
import pytest

from pitch_garden import Tracker

from ..helpers import cent_error, generate_pulse_train

# The F0 and the confidence of the frames 0, 10, 50, and 100 and their sums given by
# the Keras models of FCN-f0 at commit 8a2b530 with the frames of `frame_audio`.
EXPECTED = {
    "929": (
        [119.4060126228288, 125.91793247143235, 149.62344886183695, 179.3571263634999],
        15127.00033337452,
        [
            0.7707332968711853,
            0.9091925024986267,
            0.9045819044113159,
            0.7112302780151367,
        ],
        92.07348638772964,
    ),
    "993": (
        [
            120.90023984700375,
            125.52907267836068,
            149.80322876419146,
            179.32005626581105,
        ],
        15130.458027364912,
        [
            0.7520391345024109,
            0.9247187376022339,
            0.9404357075691223,
            0.7252493500709534,
        ],
        92.66327077150345,
    ),
    "1953": (
        [
            120.42560261461766,
            125.81630196343295,
            149.70251965717807,
            179.26562569444803,
        ],
        15134.448482773514,
        [0.732187807559967, 0.9093276262283325, 0.9093937277793884, 0.8030596971511841],
        91.1370057463646,
    ),
}


def generate_glide(sample_rate: int = 8000) -> np.ndarray:
    """Generate a harmonic tone gliding from 120 to 180 Hz in 1 second with noise."""
    t = np.arange(sample_rate) / sample_rate
    phase = 2 * np.pi * np.cumsum(120 + 60 * t) / sample_rate
    x = sum(np.sin(k * phase) / k for k in range(1, 11))
    return x + 0.01 * np.random.default_rng(0).standard_normal(sample_rate)


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_identical_to_fcn_f0(model: str) -> None:
    """Test that the F0 and the confidence are identical to those of FCN-f0."""
    f0, f0_sum, confidence, confidence_sum = EXPECTED[model]
    pitch = Tracker("fcn-f0", model=model)(generate_glide(), 8000, frame_period=10.0)
    native = pitch.native
    assert native is not None
    assert native.candidate is not None and native.score is not None
    assert len(native.candidate) == 101
    index = [0, 10, 50, 100]
    np.testing.assert_allclose(native.candidate[index], f0, rtol=1e-4)
    np.testing.assert_allclose(native.candidate.sum(), f0_sum, rtol=1e-4)
    np.testing.assert_allclose(native.score[index], confidence, atol=1e-4)
    np.testing.assert_allclose(native.score.sum(), confidence_sum, rtol=1e-4)


def test_utterance() -> None:
    """Test that the F0 is tracked in an utterance surrounded by silence."""
    silence = np.zeros(8000)
    x = np.concatenate([silence, generate_pulse_train(150.0), silence])
    pitch = Tracker("fcn-f0")(x, 16000, frame_period=10.0)
    inner = pitch.f0[60:141]  # 0.6-1.4 seconds
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)
    np.testing.assert_array_equal(pitch.f0[:40], 0.0)  # 0-0.4 seconds


def test_search_range() -> None:
    """Test that the F0 out of the search range is not estimated."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("fcn-f0", fmin=200.0, fmax=400.0)(x, 16000)
    assert pitch.native is not None and pitch.native.candidate is not None
    assert np.all(190.0 <= pitch.native.candidate)


def test_invalid_model() -> None:
    """Test that an unknown model is rejected."""
    with pytest.raises(ValueError, match="Unsupported model"):
        Tracker("fcn-f0", model="1024")


def test_auto_device() -> None:
    """Test that a GPU is used by default if available."""
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert Tracker("fcn-f0").config.device == expected


def test_not_parallel_safe_on_gpu() -> None:
    """Test that FCN-f0 on a GPU is run in the main process."""
    assert Tracker("fcn-f0", device="cpu").parallel_safe
    assert not Tracker("fcn-f0", device="cuda").parallel_safe


def test_constants_are_read_only() -> None:
    """Test that the cached cents of the bins and the architectures are immutable."""
    from pitch_garden.algorithms.fcn_f0 import _ARCHITECTURES, _bin_cents

    cents = _bin_cents()
    assert cents is _bin_cents()
    with pytest.raises(ValueError, match="read-only"):
        cents[0] = 0.0
    for filters, widths, _ in _ARCHITECTURES.values():
        assert isinstance(filters, tuple) and isinstance(widths, tuple)


def test_range_narrower_than_bin() -> None:
    """Test that the candidates stay near a range narrower than the bin interval."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("fcn-f0", fmin=300.0, fmax=300.5)(x, 16000, frame_period=10.0)
    assert pitch.native is not None and pitch.native.candidate is not None
    assert np.all(cent_error(pitch.native.candidate, 300.0) < 25)
