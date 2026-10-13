"""Tests for the algorithms of SPTK."""

from typing import cast

import numpy as np
import pytest

from pitch_garden import Audio, RawFormat, Tracker, load_audio
from pitch_garden.algorithms.sptk import SptkConfig

from ..helpers import cent_error, generate_pulse_train, generate_tone

pytest.importorskip("pitch_garden_sptk")

SPTK_ALGORITHMS = ["rapt", "swipe", "reaper", "dio", "harvest"]


@pytest.mark.parametrize("algorithm", SPTK_ALGORITHMS)
@pytest.mark.parametrize("sample_rate", [16000, 44100])
def test_tone(algorithm: str, sample_rate: int) -> None:
    """Test that the F0 of a voice-like pulse train is estimated within 50 cents."""
    x = generate_pulse_train(150.0, sample_rate=sample_rate)
    pitch = Tracker(algorithm)(x, sample_rate, frame_period=5.0)
    assert pitch.length == 200
    inner = pitch.f0[30:-30]
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)


@pytest.mark.parametrize("algorithm", SPTK_ALGORITHMS)
@pytest.mark.parametrize("sample_rate", [6000, 192000])
def test_unsupported_sample_rate_is_resampled(algorithm: str, sample_rate: int) -> None:
    """Test that the audio out of the supported sample rates is resampled."""
    from pitch_garden_sptk import MAX_SAMPLE_RATE, MIN_SAMPLE_RATE

    assert not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE
    x = generate_pulse_train(150.0, sample_rate=sample_rate)
    pitch = Tracker(algorithm)(x, sample_rate, frame_period=5.0)
    assert pitch.length == 200
    assert pitch.sample_rate == sample_rate
    inner = pitch.f0[30:-30]
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 150.0) < 50)


@pytest.mark.parametrize(
    "algorithm, low, high",
    [
        ("rapt", -0.6, 0.7),
        ("swipe", 0.2, 0.5),
        ("reaper", -0.5, 1.6),
        ("dio", 0.02, 0.2),
        ("harvest", 0.0, 0.2),
    ],
)
def test_voicing_threshold_out_of_range(
    algorithm: str, low: float, high: float
) -> None:
    """Test that a voicing threshold out of the recommended range is accepted."""
    x = generate_pulse_train(150.0)
    for threshold in (low - 1.0, high + 1.0):
        pitch = Tracker(algorithm, voicing_threshold=threshold)(x, 16000)
        assert pitch.length == 200


@pytest.mark.parametrize("algorithm", SPTK_ALGORITHMS)
def test_upper_bound_is_clipped_below_nyquist(algorithm: str) -> None:
    """Test that the upper bound is clipped to be lower than the Nyquist frequency."""
    x = generate_pulse_train(150.0, sample_rate=8000)
    pitch = Tracker(algorithm, fmax=5000.0)(x, 8000)
    voiced = pitch.f0[0 < pitch.f0]
    assert np.all(voiced < 4000.0)
    assert cent_error(np.median(voiced), 150.0) < 50


def test_range_above_nyquist() -> None:
    """Test that a range entirely above the Nyquist frequency is an error."""
    tracker = Tracker("rapt", fmin=4500.0, fmax=5000.0)
    with pytest.raises(ValueError, match="Nyquist"):
        tracker(np.zeros(8000), 8000)


def test_lower_bound_below_sptk_limit_is_clipped() -> None:
    """Test that a lower bound below the exclusive limit of SPTK is clipped."""
    x = generate_pulse_train(150.0)
    assert Tracker("rapt", fmin=5.0)(x, 16000).length == 200


@pytest.mark.parametrize("algorithm", SPTK_ALGORITHMS)
def test_identical_to_sptk(tmp_path, algorithm: str) -> None:
    """Test that a 16-bit raw file gives the same result as the binding of SPTK.

    The binding is tested to be identical to the pitch command of SPTK in
    bindings/sptk/tests.

    """
    from pitch_garden_sptk import extract_f0

    rng = np.random.default_rng(0)
    x = generate_tone(150.0) + 0.003 * rng.standard_normal(16000)
    path = tmp_path / "a.raw"
    samples = np.round(x * 32768).astype("<i2")
    samples.tofile(path)

    audio = load_audio(str(path), RawFormat(sample_rate=16000, data_type="s"))
    assert isinstance(audio, Audio)
    tracker = Tracker(algorithm, fmin=60.0, fmax=400.0)
    pitch = tracker(audio, frame_period=5.0)

    threshold = cast(SptkConfig, tracker.config).voicing_threshold
    waveform = samples.astype(np.float64)
    expected = extract_f0(waveform, 16000, 80, 60.0, 400.0, threshold, algorithm)
    np.testing.assert_array_equal(pitch.f0, expected)


def test_sptk_failure_has_context() -> None:
    """Test that a failure of SPTK reports the arguments."""
    pytest.importorskip("pitch_garden_sptk")
    with pytest.raises(RuntimeError, match=r"rapt.*frame shift.*samples"):
        Tracker("rapt")(np.zeros(10), 16000)
