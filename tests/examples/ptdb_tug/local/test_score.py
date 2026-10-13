"""Tests for the scoring script of the PTDB-TUG recipe."""

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
SCRIPT = os.path.join(EXAMPLES_DIR, "ptdb_tug", "local", "score.py")
# The scripts import the shared utilities linked from 'local/utils'.
sys.path.insert(0, os.path.dirname(SCRIPT))
spec = importlib.util.spec_from_file_location("ptdb_score", SCRIPT)
assert spec is not None and spec.loader is not None
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def write_reference(path, f0: list[float]) -> None:
    """Write a reference file in the format of PTDB-TUG, i.e., the output of RAPT."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [[value, float(0 < value), 100.0, 0.5] for value in f0]
    np.savetxt(path, np.array(rows))


def write_estimate(path, f0: list[float], frame_period: float = 2.0) -> None:
    """Write an estimated F0 file in the NPZ format of pitch-garden."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pitch = Pitch(
        f0=np.array(f0), frame_period=frame_period, sample_rate=48000, source="t"
    )
    pitch.tofile(str(path), output_format="linear", file_type="npz")


def test_load_reference(tmp_path) -> None:
    """Test that the F0 is the first column of the reference file."""
    path = tmp_path / "ref_F01_sa1.f0"
    write_reference(str(path), [0.0, 120.0, 0.0])
    np.testing.assert_array_equal(score.load_reference(str(path)), [0.0, 120.0, 0.0])


def test_sample_estimate_at_reference_times(tmp_path) -> None:
    """Test that the estimate is sampled at 16 ms + 10 ms * i without interpolation."""
    path = tmp_path / "est.npz"
    # The frame k of the estimate is at 2 ms * k, whose F0 is 100 + k.
    write_estimate(str(path), [100.0 + k for k in range(20)])
    sampled = score.sample_estimate(str(path), 3)
    # The frames 8, 13, and 18 are at 16, 26, and 36 ms.
    np.testing.assert_array_equal(sampled, [108.0, 113.0, 118.0])


def test_sample_estimate_beyond_end_is_unvoiced(tmp_path) -> None:
    """Test that the reference frames beyond the end of the estimate are unvoiced."""
    path = tmp_path / "est.npz"
    write_estimate(str(path), [150.0] * 10)
    np.testing.assert_array_equal(score.sample_estimate(str(path), 3), [150.0, 0, 0])


def test_sample_empty_estimate(tmp_path) -> None:
    """Test that all reference frames are unvoiced for an empty estimate."""
    path = tmp_path / "est.npz"
    write_estimate(str(path), [])
    np.testing.assert_array_equal(score.sample_estimate(str(path), 2), [0.0, 0.0])


def test_evaluate_tracker_pools_all_files(tmp_path) -> None:
    """Test that the metrics are computed on the frames pooled over the speakers."""
    db = tmp_path / "SPEECH DATA"
    refs = {"F01_sa1": [0.0, 200.0], "M01_sa1": [100.0, 100.0]}
    sexes = {"F": "FEMALE", "M": "MALE"}
    estimates = {"F01_sa1": [0.0, 200.0], "M01_sa1": [100.0, 0.0]}
    for name, f0 in refs.items():
        speaker = name.split("_")[0]
        path = db / sexes[speaker[-3]] / "REF" / speaker / f"ref_{name}.f0"
        write_reference(str(path), f0)
        est = estimates[name]
        # The estimate holds the F0 at the frames 8 and 13, i.e., 16 and 26 ms.
        write_estimate(
            str(tmp_path / "est" / f"{name}.npz"), [est[0]] * 13 + [est[1]] * 2
        )
    result = score.evaluate_tracker(str(db), str(tmp_path / "est"))
    expected = evaluate(
        np.array([0.0, 200.0, 100.0, 100.0]), np.array([0, 200, 100, 0])
    )
    assert result["oa"] == pytest.approx(expected["oa"])
    assert result["vde"] == pytest.approx(expected["vde"])


def test_evaluate_tracker_requires_all_estimates(tmp_path) -> None:
    """Test that a missing estimate is an error."""
    db = tmp_path / "SPEECH DATA"
    write_reference(str(db / "FEMALE" / "REF" / "F01" / "ref_F01_sa1.f0"), [100.0])
    with pytest.raises(FileNotFoundError, match="F01_sa1"):
        score.evaluate_tracker(str(db), str(tmp_path / "est"))
