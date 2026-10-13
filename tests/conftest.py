"""Shared fixtures for testing."""

import os

import pytest
import soundfile as sf

from pitch_garden.algorithms import ALGORITHM_MAP

from .dummy import ConstantAlgorithm
from .helpers import generate_tone


@pytest.fixture
def wav_dir(tmp_path) -> str:
    """Create audio files of two speakers."""
    for speaker, f0 in (("spk1", 110.0), ("spk2", 220.0)):
        os.makedirs(tmp_path / "wav" / speaker)
        for i in range(2):
            x = generate_tone(f0 * (1 + 0.05 * i), duration_sec=0.5)
            sf.write(tmp_path / "wav" / speaker / f"{speaker}_{i}.wav", x, 16000)
    return str(tmp_path / "wav")


@pytest.fixture
def constant(monkeypatch) -> None:
    """Register the constant algorithm."""
    monkeypatch.setitem(ALGORITHM_MAP, "constant", ConstantAlgorithm)
