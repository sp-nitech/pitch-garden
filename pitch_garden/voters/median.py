"""A module for the median voter."""

from collections.abc import Mapping

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..interfaces.types import Pitch
from .base import BaseVoter


class MedianVoter(BaseVoter):
    """A voter taking the median of log F0 across trackers and neighboring frames.

    This is compatible with `medfilt -w 1 -magic` of SPTK. For each frame, the
    values of all trackers within the temporal window are collected. The frame is
    unvoiced if the unvoiced values outnumber the voiced ones; otherwise, the median
    of the voiced values in the log domain is taken.

    """

    def __init__(self, filter_length: int = 3) -> None:
        """Initialize the median voter.

        Parameters
        ----------
        filter_length : int, optional
            The length of the temporal window in frames. The window is truncated at
            the edges.

        Raises
        ------
        ValueError
            If the filter length is not positive.

        """
        if filter_length < 1:
            raise ValueError("filter_length must be a positive integer.")
        self.filter_length = filter_length

    def vote(self, pitches: Mapping[str, Pitch]) -> Pitch:
        """Integrate the outputs of the trackers.

        Parameters
        ----------
        pitches : Mapping[str, Pitch]
            The outputs of the trackers keyed by the tracker names.

        Returns
        -------
        out : Pitch
            The integrated F0 contour.

        """
        f0 = np.stack([p.f0 for p in pitches.values()], axis=1)  # (N, K)
        voted = self._vote(f0) if len(f0) else np.zeros(0)
        first = next(iter(pitches.values()))
        return Pitch(
            f0=voted,
            frame_period=first.frame_period,
            sample_rate=first.sample_rate,
            source="median",
        )

    def _vote(self, f0: np.ndarray) -> np.ndarray:
        """Vote the F0 of the trackers with shape (N, K), where N is positive."""
        num_frames, num_trackers = f0.shape
        order = self.filter_length - 1
        future = order // 2
        past = order - future

        # The unvoiced values are NaN, and the padded values out of the contour are
        # excluded from the counts, which truncates the window at the edges.
        log_f0 = np.full_like(f0, np.nan)
        np.log(f0, out=log_f0, where=0 < f0)
        padding = ((past, future), (0, 0))
        values = np.pad(log_f0, padding, constant_values=np.nan)
        inside = np.pad(np.ones_like(f0, dtype=bool), padding)
        shape = (num_frames, num_trackers * self.filter_length)
        values = sliding_window_view(values, self.filter_length, axis=0).reshape(shape)
        inside = sliding_window_view(inside, self.filter_length, axis=0).reshape(shape)

        num_voiced = np.sum(~np.isnan(values), axis=1)
        num_unvoiced = np.sum(inside, axis=1) - num_voiced
        voiced = num_unvoiced <= num_voiced
        voted = np.zeros(num_frames)
        if np.any(voiced):
            voted[voiced] = np.exp(np.nanmedian(values[voiced], axis=1))
        return voted
