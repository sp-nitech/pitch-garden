"""Tests for the evaluation of all metrics."""

import numpy as np
import pytest

from pitch_garden import Pitch
from pitch_garden.metrics import evaluate

REF = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 0.0])


EST = np.array([0.0, 101.0, 200.0, 0.0, 100.0, 150.0])


def test_evaluate_with_pitch_objects() -> None:
    """Test the evaluation with pitch objects of different lengths."""
    ref = Pitch(f0=REF, frame_period=5.0, sample_rate=16000, source="ref")
    est = Pitch(
        f0=np.append(EST, 100.0), frame_period=5.0, sample_rate=16000, source=""
    )
    result = evaluate(ref, est)
    assert set(result) == {
        "vde",
        "voicing_precision",
        "voicing_recall",
        "voicing_f1",
        "gpe",
        "fpe",
        "rpa",
        "rca",
        "oa",
        "f1",
    }
    assert result["gpe"] == pytest.approx(1 / 3)


def test_mismatched_frame_period() -> None:
    """Test that a mismatch of the frame period is rejected."""
    ref = Pitch(f0=REF, frame_period=5.0, sample_rate=16000, source="")
    est = Pitch(f0=EST, frame_period=10.0, sample_rate=16000, source="")
    with pytest.raises(ValueError):
        evaluate(ref, est)


def test_mask_shorter_than_contours() -> None:
    """Test that a mask shorter than the evaluated frames is an error."""
    with pytest.raises(ValueError, match="mask"):
        evaluate(REF, EST, mask=np.ones(3, dtype=bool))
