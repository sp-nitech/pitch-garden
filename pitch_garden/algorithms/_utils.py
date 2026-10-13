"""A module for the utilities shared by the algorithms."""

from typing import Literal

import numpy as np

from ..utils.units import hz_to_mel, mel_to_hz


def frame_audio(
    data: np.ndarray,
    sample_rate: int,
    frame_period: float,
    frame_length: int,
    pad_mode: Literal["constant", "reflect"] = "constant",
) -> tuple[np.ndarray, int]:
    """Split audio into frames centered at multiples of the hop.

    Parameters
    ----------
    data : np.ndarray
        The waveform with shape (N,).

    sample_rate : int
        The sample rate in Hz.

    frame_period : float
        The frame period in milliseconds, which is rounded to an integer number of
        samples.

    frame_length : int
        The length of a frame in samples.

    pad_mode : Literal["constant", "reflect"], optional
        The padding of both ends by about half of the frame length, i.e., zeros or
        the reflection of the waveform.

    Returns
    -------
    frames : np.ndarray
        The frames with shape (1 + N // hop, frame_length), which is a view of the
        padded waveform.

    hop : int
        The frame shift in samples.

    """
    hop = max(1, round(sample_rate * frame_period / 1000))
    num_frames = 1 + len(data) // hop
    # The right side is padded by one more sample for an odd frame length, so that
    # the last frame centered at the end is available.
    half = frame_length // 2
    padded = np.pad(data, (half, frame_length - half), mode=pad_mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame_length)
    return windows[::hop][:num_frames], hop


def mask_bins_out_of_range(
    salience: np.ndarray,
    frequencies: np.ndarray,
    f0_range: tuple[float, float],
    fill: float = -np.inf,
) -> np.ndarray:
    """Mask the pitch bins out of the search range.

    The bin nearest to the center of the range on the log scale is kept, so that a
    range narrower than the interval of the bins still has a bin.

    Parameters
    ----------
    salience : np.ndarray
        The salience of the pitch bins with shape (T, B).

    frequencies : np.ndarray
        The frequencies of the pitch bins in Hz with shape (B,).

    f0_range : tuple[float, float]
        The F0 search range in Hz.

    fill : float, optional
        The value of the masked bins, e.g., 0 for the salience averaged around the
        peak.

    Returns
    -------
    out : np.ndarray
        The masked copy of the salience.

    """
    outside = (frequencies < f0_range[0]) | (f0_range[1] < frequencies)
    center = np.log(f0_range[0] * f0_range[1]) / 2
    outside[np.argmin(np.abs(np.log(frequencies) - center))] = False
    return np.where(outside, fill, salience)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    frequency_range: tuple[float, float],
    htk: bool = True,
) -> np.ndarray:
    """Calculate the mel filterbank with the Slaney normalization.

    This is identical to `librosa.filters.mel` with the same arguments.

    Parameters
    ----------
    sample_rate : int
        The sample rate in Hz.

    n_fft : int
        The FFT length.

    n_mels : int
        The number of mel bands.

    frequency_range : tuple[float, float]
        The lowest and highest frequencies of the bands in Hz.

    htk : bool, optional
        If True, the HTK formula of mels is used. Otherwise, the Slaney formula is
        used as the default of librosa.

    Returns
    -------
    out : np.ndarray
        The filterbank with shape (n_mels, n_fft // 2 + 1) in single precision.

    """
    fft_frequencies = np.linspace(0, sample_rate / 2, 1 + n_fft // 2)
    lower, upper = hz_to_mel(np.asarray(frequency_range), htk=htk)
    mel_frequencies = mel_to_hz(np.linspace(lower, upper, n_mels + 2), htk=htk)
    differences = np.diff(mel_frequencies)
    ramps = np.subtract.outer(mel_frequencies, fft_frequencies)
    rising = -ramps[:-2] / differences[:-1, None]
    falling = ramps[2:] / differences[1:, None]
    weights = np.maximum(0, np.minimum(rising, falling))
    weights *= 2.0 / (mel_frequencies[2:] - mel_frequencies[:-2])[:, None]
    return weights.astype(np.float32)
