"""Utilities for the uniform frame grid."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GridContour:
    """Dataclass to hold a contour resampled onto the uniform frame grid."""

    f0: np.ndarray  # The F0 in Hz (unvoiced: 0).
    score: np.ndarray | None  # The voicing score.


def check_frame_period(frame_period: float) -> None:
    """Check that the frame period is a positive finite number.

    Parameters
    ----------
    frame_period : float
        The frame period in milliseconds.

    Raises
    ------
    ValueError
        If the frame period is not a positive finite number.

    """
    if not (np.isfinite(frame_period) and 0 < frame_period):
        raise ValueError(
            f"frame_period must be a positive finite number, got {frame_period}."
        )


def get_num_frames(num_samples: int, sample_rate: int, frame_period: float) -> int:
    """Calculate the number of frames, i.e., ceil(T / hop), as in SPTK.

    Parameters
    ----------
    num_samples : int
        The number of samples of the audio.

    sample_rate : int
        The sample rate in Hz.

    frame_period : float
        The frame period in milliseconds.

    Returns
    -------
    out : int
        The number of frames.

    """
    hop = sample_rate * frame_period / 1000
    # Subtract a small value so that a floating-point error in an exact multiple of
    # the hop does not add an extra frame.
    return int(np.ceil(num_samples / hop - 1e-9))


def get_frame_times(num_frames: int, frame_period: float) -> np.ndarray:
    """Get the center times of the frames.

    Parameters
    ----------
    num_frames : int
        The number of frames.

    frame_period : float
        The frame period in milliseconds.

    Returns
    -------
    out : np.ndarray
        The center times in seconds.

    """
    return np.arange(num_frames) * frame_period / 1000


def resample(
    times: np.ndarray,
    f0: np.ndarray,
    *,
    num_frames: int,
    frame_period: float,
    score: np.ndarray | None = None,
) -> GridContour:
    """Resample a native contour of a tracker onto the uniform frame grid.

    The voicing decision and the score follow the nearest native frame. The F0 of a
    voiced frame is linearly interpolated in the log domain if both of the
    neighboring native frames are voiced, otherwise the F0 of the nearest native
    frame is used. Note that the F0 candidates of unvoiced frames are unreliable and
    must not be used for the interpolation.

    Parameters
    ----------
    times : np.ndarray
        The center times of the native frames in seconds with shape (M,).

    f0 : np.ndarray
        The native F0 in Hz with shape (M,). Unvoiced frames are 0.

    num_frames : int
        The number of output frames.

    frame_period : float
        The output frame period in milliseconds.

    score : np.ndarray | None, optional
        The native voicing score with shape (M,).

    Returns
    -------
    out : GridContour
        The resampled contour.

    """
    times = np.asarray(times, dtype=np.float64)
    f0 = np.asarray(f0, dtype=np.float64)
    if len(times) == 0:
        return GridContour(
            f0=np.zeros(num_frames),
            score=None if score is None else np.zeros(num_frames),
        )

    t = get_frame_times(num_frames, frame_period)
    right = np.clip(np.searchsorted(times, t), 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    span = times[right] - times[left]
    weight = np.where(0 < span, (t - times[left]) / np.where(0 < span, span, 1), 0)
    weight = np.clip(weight, 0, 1)
    nearest = np.where(weight <= 0.5, left, right)

    def interpolate(x: np.ndarray) -> np.ndarray:
        both = (0 < x[left]) & (0 < x[right])
        safe = np.where(0 < x, x, 1)
        log_x = (1 - weight) * np.log(safe[left]) + weight * np.log(safe[right])
        exact = (weight == 0) | (weight == 1)
        return np.where(both & ~exact, np.exp(log_x), x[nearest])

    voiced = 0 < f0[nearest]

    new_score = None
    if score is not None:
        new_score = np.asarray(score, dtype=np.float64)[nearest]

    new_f0 = np.where(voiced, interpolate(f0), 0.0)
    return GridContour(f0=new_f0, score=new_score)
