"""Tests for the command-line interface."""

import numpy as np
import pytest
import yaml

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
