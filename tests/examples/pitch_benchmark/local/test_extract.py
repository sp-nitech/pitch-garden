"""Tests for the extraction script of the pitch-benchmark recipe."""

import importlib.util
import os
import sys

import pytest
import soundfile as sf

from ....helpers import generate_tone

EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples"
)
SCRIPT = os.path.join(EXAMPLES_DIR, "pitch_benchmark", "local", "extract.py")
# The scripts import the shared utilities linked from 'local/utils'.
sys.path.insert(0, os.path.dirname(SCRIPT))
spec = importlib.util.spec_from_file_location("pb_extract", SCRIPT)
assert spec is not None and spec.loader is not None
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def test_one_pass_has_no_two_pass() -> None:
    """Test that the one-pass setting does not use the two-pass tracking."""
    assert extract.create_two_pass("one-pass") is None


@pytest.mark.parametrize("voicing", ["1st", "2nd"])
def test_two_pass_reruns_tracker(voicing: str) -> None:
    """Test that the compared settings rerun the tracker with the given voicing."""
    setting = f"two-pass-{voicing}"
    two_pass = extract.create_two_pass(setting)
    assert two_pass is not None
    assert (two_pass.fmin, two_pass.fmax) == extract.F0_RANGE
    policy = extract.create_tracker("swipe", setting).two_pass_policy
    assert (policy.enabled, policy.voicing_from) == (True, voicing)


def test_unknown_setting() -> None:
    """Test that an unknown setting is rejected."""
    with pytest.raises(ValueError, match="setting"):
        extract.create_two_pass("three-pass")


def test_tracker_uses_fixed_range() -> None:
    """Test that the tracker has the fixed range."""
    assert extract.create_tracker("rapt", "one-pass").f0_range == extract.F0_RANGE


def test_tracker_is_created_once(tmp_path, monkeypatch) -> None:
    """Test that a tracker is created once for all files in the main process."""
    audio_dir = tmp_path / "APLAWD" / "test" / "audio" / "identity"
    audio_dir.mkdir(parents=True)
    for name in ("a", "b", "c"):
        sf.write(audio_dir / f"{name}.flac", generate_tone(150.0), 16000)
    calls = []
    original = extract.create_tracker

    def create_tracker(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(extract, "create_tracker", create_tracker)
    extract.extract(str(tmp_path), str(tmp_path / "exp"), "rapt", "one-pass", "test")
    saved = sorted((tmp_path / "exp" / "one-pass" / "trackers" / "rapt").rglob("*.npz"))
    assert [p.stem for p in saved] == ["a", "b", "c"]
    assert len(calls) == 1
