"""Tests for the extraction script of the PTDB-TUG recipe."""

import importlib.util
import os
import sys

import numpy as np
import soundfile as sf

from ....helpers import generate_pulse_train

EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples"
)
SCRIPT = os.path.join(EXAMPLES_DIR, "ptdb_tug", "local", "extract.py")
# The scripts import the shared utilities linked from 'local/utils'.
sys.path.insert(0, os.path.dirname(SCRIPT))
spec = importlib.util.spec_from_file_location("ptdb_extract", SCRIPT)
assert spec is not None and spec.loader is not None
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def test_list_jobs(tmp_path) -> None:
    """Test that the microphone signals of both sexes are listed with their names."""
    db = tmp_path / "SPEECH DATA"
    for path in ("FEMALE/MIC/F01/mic_F01_sa1.wav", "MALE/MIC/M02/mic_M02_si642.wav"):
        os.makedirs(os.path.dirname(db / path), exist_ok=True)
        (db / path).touch()
    jobs = extract.list_jobs(str(db), str(tmp_path / "out"))
    assert [os.path.basename(save) for _, save in jobs] == [
        "F01_sa1.npz",
        "M02_si642.npz",
    ]
    assert jobs[0][0] == str(db / "FEMALE" / "MIC" / "F01" / "mic_F01_sa1.wav")


def test_extract(tmp_path) -> None:
    """Test that the F0 is extracted with the frame period of 2 ms."""
    db = tmp_path / "SPEECH DATA"
    path = db / "MALE" / "MIC" / "M01" / "mic_M01_sa1.wav"
    os.makedirs(os.path.dirname(path))
    sf.write(path, generate_pulse_train(150.0), 16000)
    extract.extract(str(db), str(tmp_path / "exp"), "rapt")
    saved = tmp_path / "exp" / "trackers" / "rapt" / "M01_sa1.npz"
    with np.load(saved) as data:
        assert float(data["frame_period"]) == 2.0
        voiced = data["f0"][data["f0"] > 0]
    assert abs(1200 * np.log2(np.median(voiced) / 150.0)) < 50


def test_two_pass_with_default_policy(tmp_path, monkeypatch) -> None:
    """Test that the trackers are run with their default policies of the two-pass."""
    calls = []
    original = extract.pg.Pipeline

    class SpyPipeline(original):
        def __init__(self, trackers, *, two_pass=None, **kwargs) -> None:
            calls.append((trackers, two_pass))
            super().__init__(trackers, two_pass=two_pass, **kwargs)

    monkeypatch.setattr(extract.pg, "Pipeline", SpyPipeline)
    path = tmp_path / "mic.wav"
    sf.write(path, generate_pulse_train(150.0), 16000)
    tracker = extract.create_tracker("rapt")
    extract.extract_file(tracker, str(path), str(tmp_path / "out.npz"))
    ((trackers, two_pass),) = calls
    assert trackers == [tracker]
    assert (two_pass.fmin, two_pass.fmax) == extract.F0_RANGE
    # The default policy chosen on pitch-benchmark is kept.
    assert tracker.two_pass_policy.enabled
