"""I/O utilities."""

import os
from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np
import soundfile as sf

from ..interfaces.types import Audio

# The data types of headerless raw files, following `x2x` of SPTK. The long double
# ('e') and ASCII ('a') types are not supported.
RawDataType = Literal["c", "C", "s", "S", "h", "H", "i", "I", "l", "L", "f", "d"]

# The number of bytes and the kind of each data type.
_RAW_TYPES: dict[str, tuple[int, str]] = {
    "c": (1, "signed"),
    "C": (1, "unsigned"),
    "s": (2, "signed"),
    "S": (2, "unsigned"),
    "h": (3, "signed"),
    "H": (3, "unsigned"),
    "i": (4, "signed"),
    "I": (4, "unsigned"),
    "l": (8, "signed"),
    "L": (8, "unsigned"),
    "f": (4, "float"),
    "d": (8, "float"),
}


@dataclass
class RawFormat:
    """The format of headerless raw audio files."""

    sample_rate: int  # The sample rate in Hz.
    data_type: RawDataType = "s"  # The data type in the notation of SPTK x2x.
    num_channels: int = 1  # The number of interleaved channels, which are averaged.
    extensions: list[str] = field(default_factory=lambda: [".raw"])

    def is_raw(self, path: str) -> bool:
        """Check whether the file is regarded as raw by its extension.

        Parameters
        ----------
        path : str
            The path to the file.

        Returns
        -------
        out : bool
            True if the file is regarded as raw.

        """
        ext = os.path.splitext(path)[1].lower()
        return ext in [e.lower() for e in self.extensions]


def _load_raw(
    path: str,
    sample_rate: int,
    data_type: RawDataType | str,
    num_channels: int = 1,
) -> Audio:
    """Load a headerless little-endian raw audio file.

    Integers are normalized into [-1, 1), where unsigned integers are offset to be
    centered at zero. Floating-point values are kept as they are. Multiple channels
    are assumed to be interleaved and are averaged.

    Parameters
    ----------
    path : str
        The path to the raw file.

    sample_rate : int
        The sample rate in Hz.

    data_type : ['c', 'C', 's', 'S', 'h', 'H', 'i', 'I', 'l', 'L', 'f', 'd']
        The data type in the notation of SPTK x2x.

    num_channels : int, optional
        The number of interleaved channels.

    Returns
    -------
    out : Audio
        The loaded monaural audio.

    Raises
    ------
    ValueError
        If the data type or the number of channels is invalid, or the file size is
        inconsistent with them.

    """
    if data_type not in _RAW_TYPES:
        raise ValueError(
            f"Unsupported raw data type '{data_type}'. "
            f"Supported types are: {list(get_args(RawDataType))}"
        )
    if num_channels < 1:
        raise ValueError("num_channels must be a positive integer.")
    size, kind = _RAW_TYPES[data_type]
    data = np.fromfile(path, dtype=np.uint8)
    frame_size = size * num_channels
    if len(data) % frame_size != 0:
        raise ValueError(
            f"The size of {path} ({len(data)} bytes) is not a multiple of "
            f"{frame_size} bytes of {num_channels} channel(s) of the data type "
            f"'{data_type}'."
        )

    x = _decode(data, size, kind)
    return Audio(x.reshape(-1, num_channels).mean(axis=1), sample_rate)


def _decode(data: np.ndarray, size: int, kind: str) -> np.ndarray:
    """Decode the bytes into normalized samples."""
    if kind == "float":
        return data.view(f"<f{size}").astype(np.float64)

    bits = 8 * size
    if size == 3:
        b = data.reshape(-1, 3).astype(np.int64)
        u = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    else:
        u = data.view(f"<u{size}").astype(np.float64 if size == 8 else np.int64)
    half = 2.0 ** (bits - 1)
    x = np.where(half <= u, u - 2.0**bits, u) if kind == "signed" else u - half
    return np.asarray(x, dtype=np.float64) / half


def load_audio(path: str, raw: RawFormat | None = None) -> Audio:
    """Load an audio file as monaural audio.

    Parameters
    ----------
    path : str
        The path to the audio file.

    raw : RawFormat | None, optional
        The format of raw files. The files with its extensions are read as raw.
        If None, no file is read as raw.

    Returns
    -------
    out : Audio
        The loaded audio, where multiple channels are averaged.

    Raises
    ------
    ValueError
        If the file cannot be read as audio without the format of raw files.

    """
    if raw is not None and raw.is_raw(path):
        return _load_raw(path, raw.sample_rate, raw.data_type, raw.num_channels)

    try:
        x, sr = sf.read(path, dtype="float64", always_2d=True)
    except (TypeError, sf.LibsndfileError) as e:
        if raw is not None:
            raise
        raise ValueError(
            f"Failed to read {path} as audio. Give a RawFormat (the raw section of "
            "the configuration) to read it as headerless raw audio."
        ) from e
    data = np.asarray(x, dtype=np.float64)
    return Audio(data.mean(axis=1), int(sr))
