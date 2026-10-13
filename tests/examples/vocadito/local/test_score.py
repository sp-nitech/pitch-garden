"""Tests for the scoring script of the Vocadito recipe."""

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
SCRIPT = os.path.join(EXAMPLES_DIR, "vocadito", "local", "score.py")
# The scripts import the shared utilities linked from 'local/utils'.
sys.path.insert(0, os.path.dirname(SCRIPT))
spec = importlib.util.spec_from_file_location("score", SCRIPT)
assert spec is not None and spec.loader is not None
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)

STEP = 256 / 44100  # The time step of the reference in seconds.


def write_reference(path, f0: list[float]) -> None:
    """Write a reference F0 file in the format of Vocadito."""
    times = np.arange(len(f0)) * STEP
    np.savetxt(path, np.stack([times, f0], axis=1), delimiter=",")


def write_estimate(path, f0: list[float], frame_period: float) -> None:
    """Write an estimated F0 file in the NPZ format of pitch-garden."""
    pitch = Pitch(
        f0=np.array(f0), frame_period=frame_period, sample_rate=44100, source="t"
    )
    pitch.tofile(str(path), output_format="linear", file_type="npz")


def test_load_reference(tmp_path) -> None:
    """Test loading a reference file."""
    write_reference(tmp_path / "a_f0.csv", [0.0, 100.0, 200.0])
    times, f0 = score.load_reference(str(tmp_path / "a_f0.csv"))
    np.testing.assert_allclose(times, [0.0, STEP, 2 * STEP])
    np.testing.assert_allclose(f0, [0.0, 100.0, 200.0])


def test_evaluate_tracker_pools_frames(tmp_path) -> None:
    """Test that the frames of all files are pooled, i.e., micro-averaged."""
    ref_dir, est_dir = tmp_path / "ref", tmp_path / "est"
    ref_dir.mkdir()
    est_dir.mkdir()
    refs = {"a": [0.0, 100.0, 100.0], "b": [200.0, 200.0, 0.0, 0.0]}
    ests = {"a": [0.0, 101.0, 300.0], "b": [200.0, 0.0, 0.0, 150.0, 999.0]}
    for name in refs:
        write_reference(ref_dir / f"{name}_f0.csv", refs[name])
        write_estimate(est_dir / f"{name}.npz", ests[name], 1000 * STEP)

    result = score.evaluate_tracker(str(ref_dir), str(est_dir))
    # The extra frame of the estimate of 'b' is ignored.
    expected = evaluate(
        np.array(refs["a"] + refs["b"]), np.array(ests["a"] + ests["b"][:4])
    )
    assert result.keys() == expected.keys()
    for key in expected:
        assert result[key] == pytest.approx(expected[key], nan_ok=True)


def test_evaluate_tracker_resamples_estimates(tmp_path) -> None:
    """Test that the estimates are resampled onto the time steps of the reference."""
    ref_dir, est_dir = tmp_path / "ref", tmp_path / "est"
    ref_dir.mkdir()
    est_dir.mkdir()
    write_reference(ref_dir / "a_f0.csv", [100.0] * 10)
    # The estimate has half the time step of the reference.
    write_estimate(est_dir / "a.npz", [100.0] * 20, 500 * STEP)
    result = score.evaluate_tracker(str(ref_dir), str(est_dir))
    assert result["rpa"] == pytest.approx(1.0)


def test_evaluate_tracker_requires_all_estimates(tmp_path) -> None:
    """Test that a missing estimate is reported."""
    ref_dir, est_dir = tmp_path / "ref", tmp_path / "est"
    ref_dir.mkdir()
    est_dir.mkdir()
    write_reference(ref_dir / "a_f0.csv", [100.0])
    with pytest.raises(FileNotFoundError, match="a.npz"):
        score.evaluate_tracker(str(ref_dir), str(est_dir))


def test_collect_results(tmp_path) -> None:
    """Test that the results are collected for each tracker."""
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    write_reference(ref_dir / "a_f0.csv", [100.0, 100.0])
    for name in ("rapt", "swipe"):
        est_dir = tmp_path / "exp" / "trackers" / name
        est_dir.mkdir(parents=True)
        write_estimate(est_dir / "a.npz", [100.0, 100.0], 1000 * STEP)
    results = score.collect_results(str(ref_dir), str(tmp_path / "exp"))
    assert sorted(results) == ["rapt", "swipe"]
    assert results["rapt"]["rpa"] == pytest.approx(1.0)
