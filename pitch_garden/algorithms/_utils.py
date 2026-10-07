"""A module for the utilities shared by the algorithms."""

from typing import Literal

import numpy as np


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
