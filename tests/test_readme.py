"""Tests for the example code in the README file."""

import os
import re

import pytest
import soundfile as sf

from .helpers import generate_tone

README = os.path.join(os.path.dirname(__file__), "..", "README.md")


def extract_code_blocks() -> list[tuple[str, str]]:
    """Extract the Python code blocks with the titles of their sections."""
    with open(README, encoding="utf-8") as f:
        text = f.read()
    blocks = []
    title = ""
    pattern = r"^(###? [^\n]+)$|^```python\n(.*?)^```$"
    for match in re.finditer(pattern, text, re.M | re.S):
        if match.group(1):
            title = match.group(1)
        else:
            blocks.append((title, match.group(2)))
    return blocks


BLOCKS = extract_code_blocks()


def test_code_blocks_exist() -> None:
    """Test that the README file has code blocks."""
    assert 3 <= len(BLOCKS)


@pytest.mark.parametrize("title, code", BLOCKS, ids=[t for t, _ in BLOCKS])
def test_code_block(tmp_path, monkeypatch, title: str, code: str) -> None:
    """Test that each code block runs by itself on short audio files."""
    pytest.importorskip("swift_f0")
    monkeypatch.chdir(tmp_path)
    x = generate_tone(150.0, duration_sec=0.5)
    sf.write("speech.wav", x, 16000)
    (x * 32768).astype("<i2").tofile("speech.raw")
    exec(compile(code, f"README: {title}", "exec"), {"__name__": "__readme__"})
