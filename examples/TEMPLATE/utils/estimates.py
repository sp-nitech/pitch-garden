"""Utilities for loading the F0 contours estimated in the recipes."""

import numpy as np

import pitch_garden as pg


def load_estimate(path: str, num_frames: int, frame_period: float) -> np.ndarray:
    """Load an estimated F0 file and put it on the frame grid of the reference.

    Parameters
    ----------
    path : str
        The path to the NPZ file saved by pitch-garden in the 'linear' format.

    num_frames : int
        The number of frames of the reference, whose first frame is at 0 seconds.

    frame_period : float
        The frame period of the reference in milliseconds.

    Returns
    -------
    out : np.ndarray
        The estimated F0 in Hz on the frame grid of the reference, where the frames
        beyond the end of the estimate are unvoiced.

    """
    with np.load(path) as data:
        pitch = pg.Pitch(
            f0=data["f0"],
            frame_period=float(data["frame_period"]),
            sample_rate=int(data["sample_rate"]),
            source=str(data["source"]),
        )
    return pitch.to_grid(num_frames, frame_period).f0
