"""Tests for the metrics on the voicing decision."""

import numpy as np
import pytest

from pitch_garden.metrics import (
    vde,
    voicing_scores,
)

REF = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 0.0])


EST = np.array([0.0, 101.0, 200.0, 0.0, 100.0, 150.0])


def test_vde() -> None:
    """Test the voicing decision error."""
    # Frames 3 and 5 are wrong.
    assert vde(REF, EST) == pytest.approx(2 / 6)


def test_voicing_scores() -> None:
    """Test the precision, recall, and F1 of the voicing decision."""
    scores = voicing_scores(REF, EST)
    assert scores["precision"] == pytest.approx(3 / 4)
    assert scores["recall"] == pytest.approx(3 / 4)
    assert scores["f1"] == pytest.approx(3 / 4)
