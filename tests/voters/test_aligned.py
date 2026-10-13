"""Tests for the voter aligning the trackers before voting."""

from typing import Any

import numpy as np
import pytest

from pitch_garden import Pipeline, Pitch, Tracker
from pitch_garden.voters import VOTER_MAP, create_voter
from pitch_garden.voters.aligned import (
    AlignedVoter,
    Alignment,
    apply_alignment,
    estimate_alignment,
    shift_contour,
)
from pitch_garden.voters.median import MedianVoter

from ..dummy import ConstantAlgorithm, FailingAlgorithm

FRAME_PERIOD = 5.0
NUM_FRAMES = 400


def contour(delay: float = 0.0, bias: float = 0.0) -> Pitch:
    """Create a vibrato contour delayed by frames and biased in cents."""
    t = (np.arange(NUM_FRAMES) - delay) * FRAME_PERIOD / 1000
    cents = 300 * np.sin(2 * np.pi * 6 * t) + bias
    f0 = 200 * 2 ** (cents / 1200)
    f0[:20] = 0.0  # A leading unvoiced segment.
    return Pitch(f0=f0, frame_period=FRAME_PERIOD, sample_rate=16000, source="t")


def test_estimate_fractional_shift_and_bias() -> None:
    """Test that a fractional delay and a bias are recovered."""
    alignment = estimate_alignment(
        contour(delay=0.4, bias=20.0), contour(), tolerance=5.0
    )
    assert alignment.applied
    assert alignment.shift == pytest.approx(-0.4)
    assert alignment.bias == pytest.approx(20.0, abs=1.0)


def test_shift_step_is_configurable() -> None:
    """Test that the resolution of the shift can be changed."""
    alignment = estimate_alignment(
        contour(delay=1.0), contour(), shift_step=0.5, tolerance=5.0
    )
    assert alignment.shift == pytest.approx(-1.0)
    coarse = estimate_alignment(
        contour(delay=0.4), contour(), shift_step=1.0, tolerance=50.0
    )
    assert coarse.shift in (-1.0, 0.0)


def test_no_correction_with_few_overlapping_frames() -> None:
    """Test that the alignment is skipped if voiced frames rarely overlap."""
    reference = contour()
    estimate = contour(delay=0.4, bias=20.0)
    estimate = estimate.with_f0(np.where(np.arange(NUM_FRAMES) < 50, estimate.f0, 0))
    alignment = estimate_alignment(estimate, reference, min_overlap=50)
    assert alignment == Alignment(shift=0.0, bias=0.0, applied=False)


def test_apply_alignment() -> None:
    """Test that the shift and the bias are applied to voiced frames."""
    pitch = contour(bias=20.0)
    aligned = apply_alignment(pitch, Alignment(shift=-2.0, bias=20.0, applied=True))
    np.testing.assert_allclose(aligned.f0[18:-2], contour().f0[20:], rtol=1e-6)
    # The frames shifted in from outside are unvoiced.
    np.testing.assert_array_equal(aligned.f0[-2:], 0.0)
    assert aligned.length == pitch.length


def test_invalid_parameters() -> None:
    """Test that invalid parameters are rejected."""
    invalid: list[dict[str, Any]] = [
        {"max_shift": -1.0},
        {"shift_step": 0.0},
        {"tolerance": 0.0},
        {"min_overlap": 0},
    ]
    for params in invalid:
        with pytest.raises(ValueError):
            estimate_alignment(contour(), contour(), **params)
        with pytest.raises(ValueError):
            AlignedVoter(reference="a", **params)


def test_aligned_voter() -> None:
    """Test that the trackers are aligned to the reference before voting."""
    pitches = {
        "ref": contour(),
        # Both trackers are delayed and biased in the same direction, so that the
        # errors are not canceled by the median without the alignment.
        "late": contour(delay=1.5, bias=30.0),
        "later": contour(delay=1.0, bias=40.0),
    }
    reference = pitches["ref"].f0[40:-10]  # Exclude the edges of voiced segments.

    def max_error(voted: Pitch) -> float:
        cents = 1200 * np.log2(voted.f0[40:-10] / reference)
        return float(np.max(np.abs(cents)))

    # The temporal window of the median voter is disabled, which smooths the
    # extrema of the vibrato regardless of the alignment.
    params = {"filter_length": 1}
    voter = AlignedVoter(
        reference="ref", voter="median", voter_params=params, tolerance=5.0
    )
    assert max_error(voter(pitches)) < 2.0
    # Without the alignment, the delays and the biases remain.
    assert 20.0 < max_error(MedianVoter(**params)(pitches))

    alignments = voter.align(pitches)
    assert alignments["late"][1].shift == pytest.approx(-1.5)
    assert alignments["ref"][1] == Alignment(shift=0.0, bias=0.0, applied=False)


def test_registered_and_created_by_name() -> None:
    """Test that the voter is registered with the inner voter by name."""
    assert VOTER_MAP["aligned"] is AlignedVoter
    voter = create_voter("aligned", reference="a", voter="median", shift_step=0.5)
    assert isinstance(voter, AlignedVoter) and voter.shift_step == 0.5


def test_reference_must_be_a_tracker() -> None:
    """Test that the reference tracker is checked by the pipeline."""
    trackers = [
        Tracker(ConstantAlgorithm, name="a"),
        Tracker(ConstantAlgorithm, name="b"),
    ]
    with pytest.raises(ValueError, match="reference"):
        Pipeline(trackers, voter=AlignedVoter(reference="c"))
    Pipeline(trackers, voter=AlignedVoter(reference="a"))


def test_empty_contours() -> None:
    """Test that empty contours are voted without alignment."""
    empty = Pitch(f0=np.zeros(0), frame_period=5.0, sample_rate=16000, source="a")
    voter = AlignedVoter(reference="a")
    assert voter({"a": empty, "b": empty}).length == 0
    assert shift_contour(empty, 0.5).shape == (0,)


def test_failed_reference_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """Test that the trackers are voted without the alignment if the reference fails."""
    trackers = [
        Tracker(FailingAlgorithm, name="a"),
        Tracker(ConstantAlgorithm, name="b"),
        Tracker(ConstantAlgorithm, name="c"),
    ]
    x = np.zeros(16000)
    with Pipeline(trackers, voter=AlignedVoter(reference="a"), num_workers=1) as p:
        result = p(x, 16000)
    assert list(result.errors) == ["a"]
    assert result.pitch is not None
    np.testing.assert_allclose(result.pitch.f0, result.each["b"].f0)
    assert "reference" in caplog.text
