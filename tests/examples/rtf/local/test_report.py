"""Tests for the reporting script of the RTF recipe."""

import importlib.util
import json
import os
import sys

EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples"
)
SCRIPT = os.path.join(EXAMPLES_DIR, "rtf", "local", "report.py")
# The scripts import the shared utilities linked from 'local/utils'.
sys.path.insert(0, os.path.dirname(SCRIPT))
spec = importlib.util.spec_from_file_location("rtf_report", SCRIPT)
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_result(path, algorithm: str, device: str, rtf: float, machine: str) -> None:
    """Write a result file in the format of the measuring script."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = {"algorithm": algorithm, "device": device, "rtf": rtf, "machine": machine}
    with open(path, "w") as f:
        json.dump(result, f)


def test_format_table(tmp_path) -> None:
    """Test that the RTFs on CPU and GPU are tabulated with a dash for missing ones."""
    write_result(tmp_path / "cpu" / "yin.json", "yin", "cpu", 0.01234, "Xeon")
    write_result(tmp_path / "cpu" / "crepe.json", "crepe", "cpu", 1.5, "Xeon")
    write_result(tmp_path / "cuda" / "crepe.json", "crepe", "cuda", 0.0123, "RTX")
    lines = report.format_table(report.collect_results(str(tmp_path))).splitlines()
    assert lines[0] == "| Tracker | RTF (CPU) | RTF (GPU) |"
    assert lines[2] == "| `crepe` | 1.5000 | 0.0123 |"
    assert lines[3] == "| `yin` | 0.0123 | — |"


def test_machines(tmp_path) -> None:
    """Test that the machines of the measurements are collected per device."""
    write_result(tmp_path / "cpu" / "yin.json", "yin", "cpu", 0.01, "Xeon")
    write_result(tmp_path / "cuda" / "crepe.json", "crepe", "cuda", 0.01, "RTX")
    assert report.collect_machines(str(tmp_path)) == {"cpu": ["Xeon"], "cuda": ["RTX"]}
