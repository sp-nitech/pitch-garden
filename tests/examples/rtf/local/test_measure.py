"""Tests for the measuring script of the RTF recipe."""

import importlib.util
import json
import os
import sys

import numpy as np

EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples"
)
SCRIPT = os.path.join(EXAMPLES_DIR, "rtf", "local", "measure.py")
# The scripts import the shared utilities linked from 'local/utils'.
sys.path.insert(0, os.path.dirname(SCRIPT))
spec = importlib.util.spec_from_file_location("rtf_measure", SCRIPT)
assert spec is not None and spec.loader is not None
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


def test_signal_is_deterministic() -> None:
    """Test that the signal of 30 seconds at 16 kHz is the same at every call."""
    x = measure.generate_signal()
    assert x.shape == (30 * 16000,)
    np.testing.assert_array_equal(x, measure.generate_signal())


def test_signal_alternates_voice_and_silence() -> None:
    """Test that 2 seconds of voice and 0.5 seconds of silence alternate."""
    x = measure.generate_signal()
    rms = np.sqrt(np.mean(x.reshape(-1, 1600) ** 2, axis=1))  # Every 0.1 seconds.
    # The first segment is voiced in 0-2 seconds and silent in 2-2.5 seconds.
    assert np.all(0.1 < rms[:20])
    assert np.all(rms[20:25] < 0.01)
    assert np.all(0.1 < rms[25:45])


def test_measure(tmp_path) -> None:
    """Test that the real-time factor and the environment are saved."""
    path = tmp_path / "cpu" / "yin.json"
    measure.measure("yin", "cpu", str(path), repeats=2)
    with open(path) as f:
        result = json.load(f)
    assert result["algorithm"] == "yin"
    assert result["device"] == "cpu"
    assert len(result["times"]) == 2
    assert result["rtf"] == np.median(result["times"]) / 30
    assert result["machine"]
