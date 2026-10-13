"""Tests for the batch runner."""

import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from pitch_garden.config import parse_config
from pitch_garden.interfaces.types import LOG_ZERO
from pitch_garden.runner import collect_input_files, group_files, run_batch


def test_collect_input_files(wav_dir, tmp_path) -> None:
    """Test collecting input files from a file, a directory, and a list."""
    files = collect_input_files(wav_dir)
    assert len(files) == 4
    assert collect_input_files(files[0]) == [files[0]]
    scp = tmp_path / "list.scp"
    scp.write_text("\n".join(files[:2]) + "\n\n")
    assert collect_input_files(str(scp)) == files[:2]
    with pytest.raises(ValueError):
        collect_input_files(str(tmp_path / "missing"))


def test_collect_raw_files(tmp_path) -> None:
    """Test collecting raw files by their extensions."""
    for name in ("a.raw", "b.pcm", "c.txt"):
        (tmp_path / name).write_bytes(b"\x00\x00")
    assert collect_input_files(str(tmp_path / "a.raw")) == [str(tmp_path / "a.raw")]
    assert collect_input_files(str(tmp_path)) == [str(tmp_path / "a.raw")]
    files = collect_input_files(str(tmp_path), raw_extensions=[".pcm"])
    assert files == [str(tmp_path / "b.pcm")]


def test_run_batch_raw(tmp_path, constant) -> None:
    """Test the batch processing of raw files."""
    sf.write(tmp_path / "a.wav", np.zeros(1600), 16000)
    (np.zeros(800, dtype="<i2")).tofile(tmp_path / "b.raw")
    config = parse_config(
        {
            "raw": {"sample_rate": 8000, "data_type": "s"},
            "trackers": [{"algorithm": "constant"}],
        }
    )
    assert run_batch(config, str(tmp_path), str(tmp_path / "out"), quiet=True) == 0
    assert len(np.fromfile(tmp_path / "out" / "pitch" / "a.lf0")) == 20
    assert len(np.fromfile(tmp_path / "out" / "pitch" / "b.lf0")) == 20

    # Without the format, raw files fail to be read as audio.
    config.raw = None
    assert run_batch(config, str(tmp_path), str(tmp_path / "out"), quiet=True) == 1


def test_group_files(wav_dir, tmp_path) -> None:
    """Test grouping the files by directory and by a map file."""
    files = collect_input_files(wav_dir)
    config = parse_config({"trackers": [{"algorithm": "swift-f0"}], "two_pass": {}})
    settings = config.two_pass
    assert settings is not None

    settings.scope = "utterance"
    assert len(group_files(files, settings)) == 4
    settings.scope = "global"
    assert len(group_files(files, settings)) == 1
    settings.scope = "group"
    groups = group_files(files, settings)
    assert sorted(len(v) for v in groups.values()) == [2, 2]

    map_file = tmp_path / "spk.map"
    map_file.write_text("spk1_0 A\nspk1_1 B\nspk2_0 A\nspk2_1 B\n")
    settings.group_by.type = "map"
    settings.group_by.path = str(map_file)
    groups = group_files(files, settings)
    assert sorted(groups.keys()) == ["A", "B"]
    map_file.write_text("spk1_0 A\n")
    with pytest.raises(ValueError):
        group_files(files, settings)
    settings.group_by.path = str(tmp_path / "missing.map")
    with pytest.raises(ValueError, match="missing.map"):
        group_files(files, settings)


def test_run_batch_fixed(wav_dir, tmp_path, constant) -> None:
    """Test the batch processing with the fixed range."""
    output_dir = tmp_path / "out"
    config = parse_config(
        {
            "subdir_offset": len(Path(wav_dir).parts) - 2,
            "num_workers": 1,
            "output": {"format": "log", "file_type": "double", "save_each": True},
            "trackers": [
                {"algorithm": "constant", "name": "a", "params": {"f0": 100.0}},
                {"algorithm": "constant", "name": "b", "params": {"f0": 100.0}},
            ],
        }
    )
    assert run_batch(config, wav_dir, str(output_dir), quiet=True) == 0
    final = np.fromfile(output_dir / "pitch" / "wav" / "spk1" / "spk1_0.lf0")
    assert len(final) == 100
    np.testing.assert_allclose(final, np.log(100.0))
    assert (output_dir / "trackers" / "a" / "wav" / "spk2" / "spk2_1.lf0").exists()
    assert not (output_dir / "f0_range.yaml").exists()


@pytest.mark.parametrize("scope", ["utterance", "group", "global"])
def test_run_batch_two_pass(wav_dir, tmp_path, scope: str) -> None:
    """Test the two-pass batch processing with SwiftF0."""
    pytest.importorskip("swift_f0")
    output_dir = tmp_path / "out"
    config = parse_config(
        {
            "num_workers": 1,
            "output": {"format": "linear", "file_type": "npz", "save_each": True},
            "trackers": [{"algorithm": "swift-f0"}],
            "two_pass": {"scope": scope},
        }
    )
    assert run_batch(config, wav_dir, str(output_dir), quiet=True) == 0

    with open(output_dir / "f0_range.yaml") as f:
        ranges = yaml.safe_load(f)
    assert len(ranges) == {"utterance": 4, "group": 2, "global": 1}[scope]
    for r in ranges.values():
        assert 40.0 <= r["fmin"] < r["fmax"] <= 2000.0
        assert set(r["files"]) <= set(collect_input_files(wav_dir))

    with np.load(output_dir / "pitch" / "spk1_0.npz") as data:
        f0 = data["f0"]
    assert np.median(f0[0 < f0]) == pytest.approx(110.0, rel=0.03)
    assert (output_dir / "pass1" / "pitch" / "spk1_0.npz").exists()
    # The output of a single tracker is not duplicated.
    assert not (output_dir / "pass1" / "trackers").exists()
    assert not (output_dir / "trackers").exists()


@pytest.mark.parametrize("scope", ["utterance", "group"])
def test_progress_is_counted_over_all_files(
    wav_dir, tmp_path, constant, caplog, scope: str
) -> None:
    """Test that the progress is numbered over all files, not per unit."""
    config = parse_config(
        {
            "num_workers": 1,
            "trackers": [{"algorithm": "constant"}],
            "two_pass": {"scope": scope},
        }
    )
    with caplog.at_level(logging.INFO, logger="pitch_garden"):
        assert run_batch(config, wav_dir, str(tmp_path / "out")) == 0
    messages = [r.getMessage() for r in caplog.records]
    for name in ("pass 1", "pass 2"):
        progress = [m.split()[0] for m in messages if f"] {name}: " in m]
        assert progress == ["[1/4]", "[2/4]", "[3/4]", "[4/4]"]


def test_run_batch_counts_errors(tmp_path, constant) -> None:
    """Test that broken files are skipped and counted."""
    broken = tmp_path / "broken.wav"
    broken.write_text("not audio")
    scp = tmp_path / "list.scp"
    scp.write_text(f"{broken}\n{tmp_path / 'missing.wav'}\n")
    config = parse_config(
        {
            "trackers": [{"algorithm": "constant"}],
        }
    )
    assert run_batch(config, str(scp), str(tmp_path / "out"), quiet=True) == 2


def test_log_zero_in_output(tmp_path, constant) -> None:
    """Test that unvoiced frames are written as the SPTK magic number."""
    sf.write(tmp_path / "a.wav", np.zeros(1600), 16000)
    config = parse_config(
        {
            "trackers": [{"algorithm": "constant", "params": {"score": 0.0}}],
        }
    )
    source, output_dir = str(tmp_path / "a.wav"), str(tmp_path / "out")
    assert run_batch(config, source, output_dir, quiet=True) == 0
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "out" / "pitch" / "a.lf0"), LOG_ZERO
    )


def test_same_outputs_are_an_error(tmp_path, constant) -> None:
    """Test that the inputs saved to the same output are an error before processing."""
    for speaker in ("spk1", "spk2"):
        (tmp_path / "wav" / speaker).mkdir(parents=True)
        sf.write(tmp_path / "wav" / speaker / "001.wav", np.zeros(1600), 16000)
    config = parse_config({"trackers": [{"algorithm": "constant"}]})
    output_dir = tmp_path / "out"
    with pytest.raises(ValueError, match=r"001\.wav.*subdir_offset"):
        run_batch(config, str(tmp_path / "wav"), str(output_dir), quiet=True)
    assert not output_dir.exists()

    # The subdirectories distinguish the outputs.
    offset = len((tmp_path / "wav").parts) - 1
    config = parse_config(
        {"trackers": [{"algorithm": "constant"}], "subdir_offset": offset}
    )
    assert run_batch(config, str(tmp_path / "wav"), str(output_dir), quiet=True) == 0
    # The outputs of a previous run are overwritten.
    assert run_batch(config, str(tmp_path / "wav"), str(output_dir), quiet=True) == 0
    assert sorted(p.name for p in output_dir.rglob("*.lf0")) == ["001.lf0", "001.lf0"]
