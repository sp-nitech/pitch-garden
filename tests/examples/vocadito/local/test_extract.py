"""Tests for the extraction script of the Vocadito recipe."""

import importlib.util
import os
import sys

import numpy as np
import soundfile as sf

from ....helpers import generate_pulse_train

EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples"
)
SCRIPT = os.path.join(EXAMPLES_DIR, "vocadito", "local", "extract.py")
# The scripts import the shared utilities linked from 'local/utils'.
sys.path.insert(0, os.path.dirname(SCRIPT))
spec = importlib.util.spec_from_file_location("vocadito_extract", SCRIPT)
assert spec is not None and spec.loader is not None
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def write_audio(database_dir) -> None:
    """Write an excerpt in the layout of Vocadito."""
    os.makedirs(database_dir / "Audio")
    sf.write(
        database_dir / "Audio" / "vocadito_1.wav", generate_pulse_train(150.0), 16000
    )


def test_extract(tmp_path) -> None:
    """Test that the F0 is extracted with the frame period of the reference."""
    write_audio(tmp_path / "db")
    extract.extract(str(tmp_path / "db"), str(tmp_path / "exp"), "rapt")
    saved = tmp_path / "exp" / "trackers" / "rapt" / "vocadito_1.npz"
    with np.load(saved) as data:
        assert float(data["frame_period"]) == extract.FRAME_PERIOD
