"""Tests for the command-line interface."""

import logging

import numpy as np
import pytest
import yaml

import pitch_garden as pg
from pitch_garden.cli import main as cli


def test_cli(wav_dir, tmp_path, constant) -> None:
    """Test the command-line interface."""
    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": {"format": "inverse", "file_type": "float"},
                "trackers": [{"algorithm": "constant", "params": {"f0": 160.0}}],
            }
        )
    )
    cli([str(config_path), wav_dir, "-o", str(output_dir), "--quiet"])
    data = np.fromfile(output_dir / "pitch" / "spk1_0.pit", dtype=np.float32)
    np.testing.assert_allclose(data, 100.0)  # 16000 / 160

    config_path.write_text(yaml.safe_dump({"frame_period": 5.0}))
    with pytest.raises(SystemExit):
        cli([str(config_path), wav_dir, "-o", str(output_dir), "--quiet"])


def test_cli_invalid_input(tmp_path, constant) -> None:
    """Test that an invalid input is reported without a traceback."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"trackers": [{"algorithm": "constant"}]}))
    with pytest.raises(SystemExit) as e:
        cli([str(config_path), str(tmp_path / "missing"), "--quiet"])
    assert e.value.code == 2


def test_cli_invalid_config_file(tmp_path) -> None:
    """Test that an unreadable configuration is reported without a traceback."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("trackers: [\n")
    for path in (config_path, tmp_path / "missing.yaml"):
        with pytest.raises(SystemExit) as e:
            cli([str(path), str(tmp_path), "--quiet"])
        assert e.value.code == 2


def test_cli_skipped_files_are_counted(tmp_path, constant, caplog) -> None:
    """Test the messages of the skipped files."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"trackers": [{"algorithm": "constant"}]}))
    list_path = tmp_path / "list.txt"
    list_path.write_text(f"{tmp_path / 'missing.wav'}\n")
    with caplog.at_level(logging.INFO, logger="pitch_garden"):
        with pytest.raises(SystemExit) as e:
            cli([str(config_path), str(list_path), "-o", str(tmp_path / "out")])
    assert e.value.code == 1
    messages = [r.getMessage() for r in caplog.records]
    assert "1 file was skipped due to errors." in messages
    assert not any(".." in m for m in messages)
    assert not any("RawFormat" in m for m in messages)


def test_cli_version(capsys) -> None:
    """Test showing the version."""
    with pytest.raises(SystemExit) as e:
        cli(["--version"])
    assert e.value.code == 0
    assert pg.__version__ in capsys.readouterr().out
