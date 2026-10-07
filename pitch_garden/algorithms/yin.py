# This module follows the YIN of librosa (https://github.com/librosa/librosa), which
# is distributed under the following license.
#
# ISC License
#
# Copyright (c) 2013--2026, librosa development team.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""A module for the YIN algorithm.

The F0 is estimated as in `librosa.yin`, and the voicing is decided by the
aperiodicity, i.e., the cumulative mean normalized difference at the chosen period,
as suggested by de Cheveigné and Kawahara (2002).

"""

from dataclasses import dataclass
from typing import cast

import numpy as np

from ..interfaces.types import Audio
from ._utils import frame_audio
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy

SAMPLE_RATE = 16000  # The sample rate of the analysis in Hz.
FRAME_LENGTH = 1024  # The length of an analysis frame in samples, i.e., 64 ms.

# The number of frames analyzed at once to limit the memory usage.
_BATCH_SIZE = 1024


def cumulative_mean_normalized_difference(
    frames: np.ndarray, min_period: int, max_period: int
) -> np.ndarray:
    """Calculate the cumulative mean normalized difference function.

    Parameters
    ----------
    frames : np.ndarray
        The frames with shape (T, frame_length).

    min_period : int
        The shortest period in samples.

    max_period : int
        The longest period in samples.

    Returns
    -------
    out : np.ndarray
        The function of the periods in [min_period, max_period] with shape (T, P).

    """
    n_fft = 2 ** int(np.ceil(np.log2(2 * frames.shape[1] - 1)))
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    acf = np.fft.irfft(np.abs(spectrum) ** 2, n=n_fft, axis=1)[:, : max_period + 1]
    energy = np.cumsum(frames**2, axis=1)
    difference = np.zeros_like(acf)
    difference[:, 1:] = 2 * (acf[:, :1] - acf[:, 1:]) - energy[:, :max_period]
    lags = np.arange(1, max_period + 1)
    cumulative_mean = np.cumsum(difference[:, 1:], axis=1) / lags
    numerator = difference[:, min_period : max_period + 1]
    denominator = cumulative_mean[:, min_period - 1 : max_period]
    # Unlike librosa, silent frames, whose function is 0/0, are regarded as aperiodic.
    positive = 0 < denominator
    return np.where(positive, numerator / np.where(positive, denominator, 1), 1.0)


def parabolic_shifts(x: np.ndarray) -> np.ndarray:
    """Get the positions of the parabola optima relative to the bins.

    The shift is 0 if the optimum is outside the neighbors or at the edges.

    """
    shifts = np.zeros_like(x)
    a = x[:, 2:] + x[:, :-2] - 2 * x[:, 1:-1]
    b = (x[:, 2:] - x[:, :-2]) / 2
    inside = np.abs(b) < np.abs(a)
    safe_a = np.where(inside, a, 1)
    shifts[:, 1:-1] = np.where(inside, -b / safe_a, 0)
    return shifts


def find_troughs(x: np.ndarray) -> np.ndarray:
    """Find the troughs of the cumulative mean normalized difference as in librosa.

    Parameters
    ----------
    x : np.ndarray
        The function with shape (T, P).

    Returns
    -------
    out : np.ndarray
        The boolean array that is True at the local minima, where the first period
        is a trough if it is lower than the second, and the last one is not.

    """
    is_trough = np.zeros_like(x, dtype=bool)
    is_trough[:, 1:-1] = (x[:, 1:-1] < x[:, :-2]) & (x[:, 1:-1] <= x[:, 2:])
    is_trough[:, 0] = x[:, 0] < x[:, 1]
    return is_trough


def yin(
    frames: np.ndarray,
    sample_rate: int,
    f0_range: tuple[float, float],
    trough_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate F0 by YIN as in librosa.

    Parameters
    ----------
    frames : np.ndarray
        The frames with shape (T, frame_length).

    sample_rate : int
        The sample rate in Hz.

    f0_range : tuple[float, float]
        The F0 search range in Hz.

    trough_threshold : float, optional
        The first trough below this threshold is chosen as the period, or the
        global minimum if there is none.

    Returns
    -------
    f0 : np.ndarray
        The F0 candidates in Hz with shape (T,).

    aperiodicity : np.ndarray
        The cumulative mean normalized difference at the chosen period with shape
        (T,), which is close to 0 for periodic frames.

    """
    fmin, fmax = f0_range
    min_period = int(np.floor(sample_rate / fmax))
    max_period = min(int(np.ceil(sample_rate / fmin)), frames.shape[1] - 1)
    cmndf = cumulative_mean_normalized_difference(frames, min_period, max_period)
    shifts = parabolic_shifts(cmndf)

    below = find_troughs(cmndf) & (cmndf < trough_threshold)
    index = np.where(below.any(axis=1), below.argmax(axis=1), cmndf.argmin(axis=1))

    rows = np.arange(len(cmndf))
    period = min_period + index + shifts[rows, index]
    return sample_rate / period, cmndf[rows, index]


@dataclass(frozen=True)
class YinConfig(AlgorithmConfig):
    """Configuration of YIN.

    The default search range covers both speech and singing voice.

    """

    fmin: float | None = 40.0
    fmax: float | None = 800.0

    # The first trough of the cumulative mean normalized difference below this
    # threshold is chosen as the period. The default value is that of librosa.
    trough_threshold: float = 0.1

    # The frames whose score, i.e., 1 - aperiodicity, is greater than or equal to
    # the threshold are regarded as voiced. The default value is chosen on the speech
    # and the singing voice of pitch-benchmark (examples/pitch_benchmark).
    voicing_threshold: float = 0.4


class YinAlgorithm(BaseAlgorithm):
    """YIN: a fundamental frequency estimator (de Cheveigné and Kawahara, 2002)."""

    config_class = YinConfig
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=True, voicing_from="2nd")

    @classmethod
    def get_supported_sample_rates(cls) -> tuple[int, int] | None:
        """Get the sample rate of the analysis.

        Returns
        -------
        out : tuple[int, int] | None
            The sample rate in Hz as the lower and upper limits.

        """
        return SAMPLE_RATE, SAMPLE_RATE

    @classmethod
    def get_f0_limits(cls) -> tuple[float, float]:
        """Get the F0 range where at least one period fits into the frame.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper limits in Hz.

        """
        return SAMPLE_RATE / (FRAME_LENGTH - 2), SAMPLE_RATE / 2

    def load(self) -> None:
        """Load nothing."""

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Estimate the F0 contour.

        Parameters
        ----------
        audio : Audio
            The input audio at 16 kHz.

        f0_range : tuple[float, float]
            The F0 search range in Hz.

        frame_period : float
            The frame period in milliseconds, which is rounded to an integer number of
            samples.

        Returns
        -------
        out : NativeContour
            The estimated contour, where the score is 1 - aperiodicity.

        """
        config = cast(YinConfig, self.config)
        frames, hop = frame_audio(audio.data, SAMPLE_RATE, frame_period, FRAME_LENGTH)
        num_frames = len(frames)
        candidate, aperiodicity = (
            np.concatenate(results)
            for results in zip(
                *(
                    yin(
                        frames[start : start + _BATCH_SIZE],
                        SAMPLE_RATE,
                        f0_range,
                        config.trough_threshold,
                    )
                    for start in range(0, num_frames, _BATCH_SIZE)
                )
            )
        )
        score = np.clip(1 - aperiodicity, 0, 1)
        f0 = np.where(config.voicing_threshold <= score, candidate, 0.0)
        return NativeContour(
            times=np.arange(num_frames) * hop / SAMPLE_RATE,
            f0=f0,
            score=score,
            candidate=candidate,
        )
