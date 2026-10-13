"""Tests for the metrics on the F0 values."""

import numpy as np
import pytest

from pitch_garden.metrics import (
    f1_score,
    fpe,
    gpe,
    oa,
    rca,
    rpa,
    vde,
)

REF = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 0.0])


EST = np.array([0.0, 101.0, 200.0, 0.0, 100.0, 150.0])


def test_gpe() -> None:
    """Test the gross pitch error."""
    # Frames 1, 2, 4 are voiced in both, and frame 2 is a gross error.
    assert gpe(REF, EST) == pytest.approx(1 / 3)


def test_fpe() -> None:
    """Test the fine pitch error in cents."""
    errors = 1200 * np.log2(np.array([101.0, 100.0]) / 100.0)
    assert fpe(REF, EST) == pytest.approx(np.std(errors))


def test_rpa_and_rca() -> None:
    """Test the raw pitch and chroma accuracies."""
    # Frames 1 and 4 are within 50 cents among four reference voiced frames.
    assert rpa(REF, EST) == pytest.approx(2 / 4)
    # Frame 2 is correct after octave folding.
    assert rca(REF, EST) == pytest.approx(3 / 4)


def test_f1_score() -> None:
    """Test the pitch F1 score at 50 cents."""
    # Hits are frames 1 and 4; four estimated and four reference voiced frames.
    assert f1_score(REF, EST) == pytest.approx(2 / 4)


def test_mask() -> None:
    """Test that the masked frames are excluded."""
    mask = np.array([True, True, False, True, True, True])
    assert gpe(REF, EST, mask=mask) == 0.0
    assert vde(REF, EST, mask=mask) == pytest.approx(2 / 5)


def test_empty_denominator_is_nan() -> None:
    """Test that undefined metrics are NaN."""
    zeros = np.zeros(3)
    assert np.isnan(gpe(zeros, zeros))


def test_oa() -> None:
    """Test the overall accuracy."""
    # Frame 0 is correctly unvoiced, and frames 1 and 4 are within 50 cents.
    assert oa(REF, EST) == pytest.approx(3 / 6)
    mask = np.array([True, True, True, False, False, False])
    assert oa(REF, EST, mask) == pytest.approx(2 / 3)
