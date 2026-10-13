"""Tests for the median voter."""

import numpy as np
import pytest

from pitch_garden import Pitch
from pitch_garden.voters.median import MedianVoter


def make_pitches(*contours: list[float]) -> dict[str, Pitch]:
    """Create pitch objects from F0 lists."""
    return {
        f"t{i}": Pitch(f0=np.array(c), frame_period=5.0, sample_rate=16000, source="")
        for i, c in enumerate(contours)
    }


def test_single_frame_median() -> None:
    """Test the median across trackers without temporal smoothing."""
    pitches = make_pitches([100.0, 0.0], [200.0, 0.0], [400.0, 300.0])
    voted = MedianVoter(filter_length=1)(pitches)
    # Frame 0: median of 3 values. Frame 1: 2 unvoiced votes win over 1 voiced vote.
    np.testing.assert_allclose(voted.f0, [200.0, 0.0])
    assert voted.source == "median"


def test_tie_is_voiced_with_log_mean() -> None:
    """Test that a tie is voiced and that an even count gives the log-domain mean."""
    pitches = make_pitches([100.0], [400.0], [0.0], [0.0])
    voted = MedianVoter(filter_length=1)(pitches)
    np.testing.assert_allclose(voted.f0, [200.0])


def test_temporal_window_is_truncated_at_edges() -> None:
    """Test the temporal window and its truncation at the edges."""
    pitches = make_pitches([100.0, 200.0, 400.0, 800.0])
    voted = MedianVoter(filter_length=3)(pitches)
    # Edge frames use two values (log-domain mean); middle frames use three.
    np.testing.assert_allclose(
        voted.f0, [np.sqrt(100 * 200), 200.0, 400.0, np.sqrt(400 * 800)]
    )


def test_invalid_inputs() -> None:
    """Test that inconsistent inputs are rejected."""
    with pytest.raises(ValueError):
        MedianVoter(filter_length=0)
    with pytest.raises(ValueError):
        MedianVoter()({})
    with pytest.raises(ValueError):
        MedianVoter()(make_pitches([100.0], [100.0, 100.0]))


@pytest.mark.parametrize(
    "filter_length, expected",
    [
        (1, [200.0, 115.0, 0.0, 130.0, 149.6662954709576, 152.47950681976906]),
        (
            2,
            [
                200.0,
                117.4734012447073,
                117.4734012447073,
                183.84776310850233,
                140.0,
                152.47950681976906,
            ],
        ),
        (
            3,
            [
                117.4734012447073,
                120.0,
                125.0,
                149.6662954709576,
                150.0,
                152.47950681976906,
            ],
        ),
    ],
)
def test_identical_to_sptk(filter_length: int, expected: list[float]) -> None:
    """Test that the result is identical to `medfilt -w 1 -magic` of SPTK.

    The expected values are the outputs of `medfilt -l 4 -k <filter_length - 1>
    -magic -1e10 -w 1` of SPTK for the log F0 of the input.

    """
    pitches = make_pitches(
        [100.0, 110.0, 0.0, 130.0, 0.0, 150.0],
        [200.0, 0.0, 0.0, 260.0, 140.0, 0.0],
        [0.0, 120.0, 0.0, 125.0, 160.0, 155.0],
        [400.0, 115.0, 300.0, 0.0, 0.0, 0.0],
    )
    voted = MedianVoter(filter_length=filter_length)(pitches)
    np.testing.assert_allclose(voted.f0, expected)
