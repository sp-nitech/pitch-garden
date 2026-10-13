"""Tests for the utilities shared by the algorithms."""

import numpy as np
import pytest

from pitch_garden.algorithms._utils import (
    frame_audio,
    mask_bins_out_of_range,
    mel_filterbank,
)


def test_frames_are_centered_at_multiples_of_hop() -> None:
    """Test that frame t is centered at sample t * hop of the zero-padded audio."""
    data = np.arange(1, 11, dtype=np.float64)
    frames, hop = frame_audio(data, 1000, 3.0, 4)
    assert hop == 3
    np.testing.assert_array_equal(
        frames,
        [[0, 0, 1, 2], [2, 3, 4, 5], [5, 6, 7, 8], [8, 9, 10, 0]],
    )


def test_reflection_padding() -> None:
    """Test that the audio can be padded by reflection."""
    data = np.arange(1, 11, dtype=np.float64)
    frames, _ = frame_audio(data, 1000, 3.0, 4, pad_mode="reflect")
    np.testing.assert_array_equal(frames[0], [3, 2, 1, 2])
    np.testing.assert_array_equal(frames[-1], [8, 9, 10, 9])


@pytest.mark.parametrize(
    ("frame_period", "expected"), [(10.0, 160), (0.01, 1), (5.03, 80)]
)
def test_hop_is_rounded_to_samples(frame_period: float, expected: int) -> None:
    """Test that the hop is an integer number of samples of at least one."""
    _, hop = frame_audio(np.zeros(1600), 16000, frame_period, 1024)
    assert hop == expected


def test_number_of_frames() -> None:
    """Test that the number of frames covers the audio including its end."""
    frames, _ = frame_audio(np.zeros(1600), 16000, 10.0, 1024)
    assert frames.shape == (11, 1024)


def test_odd_frame_length() -> None:
    """Test that the last frame is available for an odd frame length."""
    data = np.arange(1, 10, dtype=np.float64)
    frames, hop = frame_audio(data, 1000, 3.0, 5)
    assert hop == 3
    np.testing.assert_array_equal(
        frames,
        [[0, 0, 1, 2, 3], [2, 3, 4, 5, 6], [5, 6, 7, 8, 9], [8, 9, 0, 0, 0]],
    )


def test_mask_bins_out_of_range() -> None:
    """Test that the bins out of the search range are -inf."""
    salience = np.ones((2, 4))
    masked = mask_bins_out_of_range(
        salience, np.array([50.0, 100.0, 200.0, 400.0]), (90.0, 250.0)
    )
    np.testing.assert_array_equal(masked, [[-np.inf, 1, 1, -np.inf]] * 2)
    np.testing.assert_array_equal(salience, 1.0)  # The input is unchanged.


def test_mask_bins_out_of_narrow_range() -> None:
    """Test that the bin nearest to a range narrower than the bins is kept."""
    frequencies = np.array([50.0, 100.0, 200.0, 400.0])
    masked = mask_bins_out_of_range(np.ones((1, 4)), frequencies, (120.0, 130.0))
    np.testing.assert_array_equal(masked, [[-np.inf, 1, -np.inf, -np.inf]])


@pytest.mark.parametrize(
    ("fmin", "htk", "total", "peaks", "values"),
    [
        (
            30.0,
            True,
            8.189247468380245,
            [3, 118, 501],
            [0.056785039603710175, 0.02016606368124485, 0.005860890727490187],
        ),
        (
            0.0,
            False,
            8.18838044066149,
            [1, 109, 500],
            [0.028577780351042747, 0.020523004233837128, 0.005223188549280167],
        ),
    ],
)
def test_mel_filterbank(
    fmin: float, htk: bool, total: float, peaks: list[int], values: list[float]
) -> None:
    """Test that the mel filterbank is identical to that of librosa.

    The expected values are given by librosa 1.0.0 with `librosa.filters.mel(
    sr=16000, n_fft=1024, n_mels=128, fmin=fmin, fmax=8000, htk=htk)`, i.e., the
    settings of RMVPE and FCPE.

    """
    weights = mel_filterbank(16000, 1024, 128, (fmin, 8000.0), htk=htk)
    assert weights.shape == (128, 513)
    np.testing.assert_allclose(weights.sum(), total, rtol=1e-6)
    assert [int(np.argmax(weights[i])) for i in (0, 64, 127)] == peaks
    np.testing.assert_allclose(
        [weights[i, j] for i, j in zip((0, 64, 127), peaks)], values, rtol=1e-6
    )


def test_mask_bins_out_of_range_with_fill() -> None:
    """Test that the masked bins are filled with the given value."""
    frequencies = np.array([100.0, 200.0, 400.0])
    masked = mask_bins_out_of_range(np.ones((1, 3)), frequencies, (150.0, 250.0), 0.0)
    np.testing.assert_array_equal(masked, [[0.0, 1.0, 0.0]])
