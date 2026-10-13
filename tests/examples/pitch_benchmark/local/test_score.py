"""Tests for the scoring script of the pitch-benchmark recipe."""

import importlib.util
import os
import sys

import numpy as np
import pytest

from pitch_garden import Pitch
from pitch_garden.metrics import evaluate

EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples"
)
SCRIPT = os.path.join(EXAMPLES_DIR, "pitch_benchmark", "local", "score.py")
# The scripts import the shared utilities linked from 'local/utils'.
sys.path.insert(0, os.path.dirname(SCRIPT))
spec = importlib.util.spec_from_file_location("pb_score", SCRIPT)
assert spec is not None and spec.loader is not None
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def write_label(path, f0: list[float], trusted: list[bool] | None = None) -> None:
    """Write a label file in the format of pitch-benchmark."""
    trusted = [True] * len(f0) if trusted is None else trusted
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, pitch=np.array(f0, dtype=np.float32), trusted=np.array(trusted))


def write_estimate(path, f0: list[float], frame_period: float = 4.0) -> None:
    """Write an estimated F0 file in the NPZ format of pitch-garden."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pitch = Pitch(
        f0=np.array(f0), frame_period=frame_period, sample_rate=16000, source="t"
    )
    pitch.tofile(str(path), output_format="linear", file_type="npz")


def test_evaluate_corpus_with_trusted_mask(tmp_path) -> None:
    """Test that the frames of a corpus are pooled with the trusted mask."""
    db, est = tmp_path / "db", tmp_path / "est"
    write_label(db / "A" / "test" / "labels" / "0.npz", [100.0, 100.0], [True, False])
    write_label(db / "A" / "test" / "labels" / "1.npz", [0.0, 200.0])
    write_estimate(est / "A" / "0.npz", [100.0] * 4 + [300.0] * 4)
    write_estimate(est / "A" / "1.npz", [0.0] * 4 + [0.0] * 4)

    metrics = score.evaluate_corpus(str(db), "A", "test", str(est))
    expected = evaluate(
        np.array([100.0, 100.0, 0.0, 200.0]),
        np.array([100.0, 300.0, 0.0, 0.0]),
        mask=np.array([True, False, True, True]),
    )
    for key in expected:
        assert metrics[key] == pytest.approx(expected[key], nan_ok=True)


def test_evaluate_corpus_requires_all_estimates(tmp_path) -> None:
    """Test that a missing estimate is reported."""
    db, est = tmp_path / "db", tmp_path / "est"
    write_label(db / "A" / "test" / "labels" / "0.npz", [100.0])
    with pytest.raises(FileNotFoundError, match="0.npz"):
        score.evaluate_corpus(str(db), "A", "test", str(est))


def test_macro_average() -> None:
    """Test that the metrics are averaged over corpora."""
    result = score.macro_average({"A": {"oa": 0.5}, "B": {"oa": 1.0}})
    assert result["oa"] == pytest.approx(0.75)


def test_evaluate_tracker_over_given_corpora(tmp_path) -> None:
    """Test that the metrics are averaged over the given corpora only."""
    db, est = tmp_path / "db", tmp_path / "est"
    for corpus, f0 in (("A", 100.0), ("B", 0.0), ("C", 300.0)):
        write_label(db / corpus / "test" / "labels" / "0.npz", [100.0])
        write_estimate(est / corpus / "0.npz", [f0] * 4)
    result = score.evaluate_tracker(str(db), "test", str(est), corpora=["A", "B"])
    assert result["oa"] == pytest.approx(0.5)


def test_format_comparison_table() -> None:
    """Test formatting the comparison of the settings of the two-pass tracking."""
    metrics = {"rpa": 0.876, "rca": 0.9, "vde": 0.1234, "oa": 0.85}
    results = {("rapt", "one-pass"): metrics, ("rapt", "two-pass-1st"): metrics}
    lines = score.format_comparison_table(results).splitlines()
    assert lines[0] == "| Tracker | Setting | RPA [%] | RCA [%] | VDE [%] | OA [%] |"
    assert lines[2] == "| `rapt` | one-pass | 87.6 | 90.0 | 12.3 | 85.0 |"
    assert lines[3] == "| `rapt` | two-pass-1st | 87.6 | 90.0 | 12.3 | 85.0 |"
