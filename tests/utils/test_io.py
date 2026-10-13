"""Tests for the I/O utilities."""

from typing import get_args

import numpy as np
import pytest

from pitch_garden.utils.io import RawDataType, RawFormat, _load_raw, load_audio


def test_load_raw_short(tmp_path) -> None:
    """Test that 16-bit integers are normalized into [-1, 1)."""
    path = tmp_path / "a.raw"
    np.array([-32768, 0, 16384], dtype="<i2").tofile(path)
    audio = _load_raw(str(path), 16000, "s")
    np.testing.assert_allclose(audio.data, [-1.0, 0.0, 0.5])
    assert audio.sample_rate == 16000


def test_load_raw_unsigned_char(tmp_path) -> None:
    """Test that unsigned integers are offset to be centered at zero."""
    path = tmp_path / "a.raw"
    np.array([0, 128, 192], dtype="u1").tofile(path)
    audio = _load_raw(str(path), 8000, "C")
    np.testing.assert_allclose(audio.data, [-1.0, 0.0, 0.5])


def test_load_raw_24bit(tmp_path) -> None:
    """Test that 3-byte integers are decoded."""
    path = tmp_path / "a.raw"
    values = np.array([-(2**23), -1, 0, 2**22], dtype=np.int64)
    data = (values & 0xFFFFFF).astype("<u4").view(np.uint8).reshape(-1, 4)[:, :3]
    data.tofile(path)
    audio = _load_raw(str(path), 16000, "h")
    np.testing.assert_allclose(audio.data, values / 2**23)

    unsigned = np.array([0, 2**23, 3 * 2**22], dtype="<u4")
    unsigned.view(np.uint8).reshape(-1, 4)[:, :3].tofile(path)
    audio = _load_raw(str(path), 16000, "H")
    np.testing.assert_allclose(audio.data, [-1.0, 0.0, 0.5])


@pytest.mark.parametrize("data_type", ["f", "d"])
def test_load_raw_float(tmp_path, data_type: str) -> None:
    """Test that floating-point values are kept as they are."""
    path = tmp_path / "a.raw"
    dtype = {"f": "<f4", "d": "<f8"}[data_type]
    np.array([-100.0, 0.25], dtype=dtype).tofile(path)
    audio = _load_raw(str(path), 16000, data_type)
    np.testing.assert_allclose(audio.data, [-100, 0.25])


@pytest.mark.parametrize("data_type", get_args(RawDataType))
def test_load_raw_all_types(tmp_path, data_type: str) -> None:
    """Test that all data types of SPTK x2x except 'e' and 'a' are readable."""
    path = tmp_path / "a.raw"
    size = {"c": 1, "s": 2, "h": 3, "i": 4, "l": 8, "f": 4, "d": 8}[data_type.lower()]
    np.zeros(4 * size, dtype=np.uint8).tofile(path)
    assert _load_raw(str(path), 16000, data_type).length == 4


def test_load_raw_invalid(tmp_path) -> None:
    """Test that invalid inputs are rejected."""
    path = tmp_path / "a.raw"
    np.zeros(3, dtype=np.uint8).tofile(path)
    with pytest.raises(ValueError, match="multiple"):
        _load_raw(str(path), 16000, "s")
    with pytest.raises(ValueError, match=str(list(get_args(RawDataType)))):
        _load_raw(str(path), 16000, "e")


def test_load_audio_dispatches_raw(tmp_path) -> None:
    """Test that raw files are read with the given format."""
    path = tmp_path / "a.raw"
    np.array([16384], dtype="<i2").tofile(path)
    audio = load_audio(str(path), RawFormat(sample_rate=22050, data_type="s"))
    assert audio.sample_rate == 22050
    # Without the format, the file is not read as raw.
    with pytest.raises(ValueError, match="RawFormat"):
        load_audio(str(path))
    with pytest.raises(TypeError, match="sample_rate"):
        RawFormat()  # type: ignore[call-arg]
    # A missing file is not regarded as raw.
    for missing in (tmp_path / "missing.wav", tmp_path / "missing.raw"):
        with pytest.raises(FileNotFoundError):
            load_audio(str(missing))
    # Other extensions can be regarded as raw.
    other = tmp_path / "a.pcm"
    np.array([16384], dtype="<i2").tofile(other)
    raw = RawFormat(sample_rate=8000, extensions=[".pcm"])
    np.testing.assert_allclose(load_audio(str(other), raw).data, [0.5])


def test_load_audio_is_exposed(tmp_path) -> None:
    """Test that audio files can be loaded and tracked without soundfile."""
    import soundfile as sf

    import pitch_garden as pg

    path = tmp_path / "a.wav"
    sf.write(path, np.zeros((1600, 2)), 16000)
    audio = pg.load_audio(str(path))
    assert isinstance(audio, pg.Audio)
    assert (audio.length, audio.sample_rate) == (1600, 16000)

    raw = tmp_path / "a.raw"
    np.zeros(800, dtype="<i2").tofile(raw)
    audio = pg.load_audio(str(raw), RawFormat(sample_rate=8000))
    assert (audio.length, audio.sample_rate) == (800, 8000)


def test_load_raw_multichannel(tmp_path) -> None:
    """Test that interleaved channels are averaged."""
    path = tmp_path / "a.raw"
    # Two channels: (L, R) = (-0.5, 0.5), (0.25, 0.75)
    np.array([-16384, 16384, 8192, 24576], dtype="<i2").tofile(path)
    audio = _load_raw(str(path), 16000, "s", num_channels=2)
    np.testing.assert_allclose(audio.data, [0.0, 0.5])

    audio = load_audio(str(path), RawFormat(sample_rate=16000, num_channels=2))
    assert audio.length == 2


def test_load_raw_multichannel_invalid(tmp_path) -> None:
    """Test that invalid channels are rejected."""
    path = tmp_path / "a.raw"
    np.zeros(3, dtype="<i2").tofile(path)
    with pytest.raises(ValueError, match="multiple"):
        _load_raw(str(path), 16000, "s", num_channels=2)
    with pytest.raises(ValueError, match="num_channels"):
        _load_raw(str(path), 16000, "s", num_channels=0)
