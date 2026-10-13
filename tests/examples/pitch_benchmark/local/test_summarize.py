"""Tests for the summarizing script of the pitch-benchmark recipe."""

import importlib.util
import os
import sys

import pytest

EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples"
)
SCRIPT = os.path.join(EXAMPLES_DIR, "pitch_benchmark", "local", "summarize.py")
# The scripts import the shared utilities linked from 'local/utils'.
sys.path.insert(0, os.path.dirname(SCRIPT))
spec = importlib.util.spec_from_file_location("pb_summarize", SCRIPT)
assert spec is not None and spec.loader is not None
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)


def metrics(oa: float) -> dict[str, float]:
    """Create the metrics whose values are all the given one."""
    return {"rpa": oa, "rca": oa, "vde": oa, "oa": oa}


def test_best_settings_averaged_over_splits() -> None:
    """Test that the best setting of each tracker is chosen by the averaged OA."""
    results = {
        "valid": {
            ("a", "one-pass"): metrics(0.8),
            ("a", "two-pass-1st"): metrics(0.9),
            ("b", "one-pass"): metrics(0.95),
            ("b", "two-pass-1st"): metrics(0.7),
        },
        "test": {
            ("a", "one-pass"): metrics(0.9),
            ("a", "two-pass-1st"): metrics(0.7),
            ("b", "one-pass"): metrics(0.85),
            ("b", "two-pass-1st"): metrics(0.9),
        },
    }
    best = summarize.best_settings(results)
    # The trackers are sorted in descending order of the averaged OA.
    assert list(best) == [("b", "one-pass"), ("a", "one-pass")]
    assert best[("b", "one-pass")]["oa"] == pytest.approx(0.9)
    assert best[("a", "one-pass")]["rpa"] == pytest.approx(0.85)


def test_ties_are_sorted_by_name() -> None:
    """Test that the trackers with the same OA are sorted by name."""
    results = {
        "test": {("b", "one-pass"): metrics(0.8), ("a", "one-pass"): metrics(0.8)}
    }
    assert list(summarize.best_settings(results)) == [
        ("a", "one-pass"),
        ("b", "one-pass"),
    ]
