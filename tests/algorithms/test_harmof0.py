"""Tests for the HarmoF0 algorithm."""

import numpy as np
import pytest

from pitch_garden import Tracker

from ..helpers import cent_error, generate_pulse_train

# The F0 and the confidence of the frames 0, 50, 150, and 300 and their sums given
# by the models of HarmoF0 at commit 3b22236 with the frames of `frame_audio` every
# 46 samples.
EXPECTED = {
    "mdb-stem-synth": (
        [
            121.70064861777168,
            128.93734577011764,
            146.83238395870376,
            172.11070782485515,
        ],
        52209.39406017809,
        [
            0.8366420269012451,
            0.9998062252998352,
            0.9862944483757019,
            0.9999910593032837,
        ],
        346.5360722541809,
    ),
    "mir-1k": (
        [
            121.70064861777168,
            128.93734577011764,
            146.83238395870376,
            172.11070782485515,
        ],
        52259.54614766066,
        [
            0.9661077260971069,
            0.9967443943023682,
            0.9441410303115845,
            0.9999958276748657,
        ],
        344.6178589463234,
    ),
}


def generate_glide(sample_rate: int = 16000) -> np.ndarray:
    """Generate a harmonic tone gliding from 120 to 180 Hz in 1 second with noise."""
    t = np.arange(sample_rate) / sample_rate
    phase = 2 * np.pi * np.cumsum(120 + 60 * t) / sample_rate
    x = sum(np.sin(k * phase) / k for k in range(1, 11))
    return x + 0.01 * np.random.default_rng(0).standard_normal(sample_rate)


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_identical_to_harmof0(model: str) -> None:
    """Test that the F0 and the confidence are identical to those of HarmoF0."""
    f0, f0_sum, confidence, confidence_sum = EXPECTED[model]
    pitch = Tracker("harmof0", model=model)(generate_glide(), 16000, frame_period=10.0)
    native = pitch.native
    assert native is not None
    assert native.candidate is not None and native.score is not None
    assert len(native.candidate) == 348
    index = [0, 50, 150, 300]
    np.testing.assert_allclose(native.candidate[index], f0, rtol=1e-9)
    np.testing.assert_allclose(native.candidate.sum(), f0_sum, rtol=1e-9)
    np.testing.assert_allclose(native.score[index], confidence, atol=1e-4)
    np.testing.assert_allclose(native.score.sum(), confidence_sum, rtol=1e-4)


def test_independent_of_batch_size() -> None:
    """Test that the frames analyzed in batches are identical to those at once."""
    x = np.tile(generate_glide(), 3)
    native = [
        Tracker("harmof0", batch_size=batch_size, device="cpu")(x, 16000).native
        for batch_size in (37, 1000)
    ]
    assert native[0] is not None and native[1] is not None
    assert native[0].score is not None and native[1].score is not None
    np.testing.assert_array_equal(native[0].candidate, native[1].candidate)
    np.testing.assert_allclose(native[0].score, native[1].score, atol=1e-6)


def test_fixed_frame_period() -> None:
    """Test that the frames are analyzed every 46 samples for any frame period."""
    pitch = Tracker("harmof0")(generate_glide(), 16000, frame_period=5.0)
    assert pitch.native is not None
    np.testing.assert_allclose(np.diff(pitch.native.times), 46 / 16000)
    assert pitch.length == 200


def test_utterance() -> None:
    """Test that the F0 is tracked in an utterance surrounded by silence."""
    silence = np.zeros(8000)
    x = np.concatenate([silence, generate_pulse_train(150.0), silence])
    pitch = Tracker("harmof0")(x, 16000, frame_period=10.0)
    inner = pitch.f0[60:141]  # 0.6-1.4 seconds
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)
    np.testing.assert_array_equal(pitch.f0[:40], 0.0)  # 0-0.4 seconds


def test_search_range() -> None:
    """Test that the F0 out of the search range is not estimated."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("harmof0", fmin=200.0, fmax=400.0)(x, 16000)
    assert pitch.native is not None and pitch.native.candidate is not None
    assert np.all((200.0 <= pitch.native.candidate) & (pitch.native.candidate <= 400.0))


def test_invalid_model() -> None:
    """Test that an unknown model is rejected."""
    with pytest.raises(ValueError, match="Unsupported model"):
        Tracker("harmof0", model="mdb")


def test_auto_device() -> None:
    """Test that a GPU is used by default if available."""
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert Tracker("harmof0").config.device == expected


def test_not_parallel_safe_on_gpu() -> None:
    """Test that HarmoF0 on a GPU is run in the main process."""
    assert Tracker("harmof0", device="cpu").parallel_safe
    assert not Tracker("harmof0", device="cuda").parallel_safe


def test_range_narrower_than_bins() -> None:
    """Test that the bin nearest to a range narrower than the bins is used."""
    x = generate_pulse_train(150.0)
    pitch = Tracker("harmof0", fmin=150.0, fmax=150.5)(x, 16000)
    assert pitch.native is not None and pitch.native.candidate is not None
    np.testing.assert_allclose(pitch.native.candidate, 27.5 * 2 ** (118 / 48))


def test_bin_frequencies_are_read_only() -> None:
    """Test that the cached frequencies of the pitch bins cannot be modified."""
    from pitch_garden.algorithms.harmof0 import _bin_frequencies

    frequencies = _bin_frequencies()
    assert frequencies is _bin_frequencies()
    with pytest.raises(ValueError, match="read-only"):
        frequencies[0] = 0.0
