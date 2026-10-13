"""Tests for the utilities for reporting the results of the recipes."""

import pytest

from .loader import load_module

report = load_module("report")


def test_format_table() -> None:
    """Test formatting the results as a Markdown table."""
    metrics = {"rpa": 0.876, "rca": 0.9, "vde": 0.1234, "oa": 0.85}
    lines = report.format_table({"swift-f0": metrics, "rapt": metrics}).splitlines()
    assert lines[0] == "| Tracker | RPA [%] | RCA [%] | VDE [%] | OA [%] |"
    assert lines[2] == "| `rapt` | 87.6 | 90.0 | 12.3 | 85.0 |"


def test_update_readme(tmp_path) -> None:
    """Test that only the named section of the README is replaced."""
    path = tmp_path / "README.md"
    path.write_text(
        "<!-- a:begin -->\n<!-- a:end -->\n<!-- b:begin -->\nold\n<!-- b:end -->\n"
    )
    report.update_readme(str(path), "b", "| new |")
    assert path.read_text() == (
        "<!-- a:begin -->\n<!-- a:end -->\n"
        "<!-- b:begin -->\n\n| new |\n\n<!-- b:end -->\n"
    )
    with pytest.raises(ValueError, match="c:begin"):
        report.update_readme(str(path), "c", "| new |")
