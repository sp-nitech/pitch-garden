# This module follows the pYIN of librosa (https://github.com/librosa/librosa), which
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

"""A module for the pYIN algorithm.

The F0 is estimated as in `librosa.pyin` with its default parameters, where the
distributions of scipy are replaced with their closed forms, and the observation
probabilities are computed frame by frame to limit the memory usage.

"""

from dataclasses import dataclass
from math import comb

import numpy as np

from ..interfaces.types import Audio
from ._utils import frame_audio
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy
from .yin import (
    FRAME_LENGTH,
    SAMPLE_RATE,
    cumulative_mean_normalized_difference,
    find_troughs,
    parabolic_shifts,
)

# The parameters of librosa.pyin.
_N_THRESHOLDS = 100
_BETA_PARAMETERS = (2, 18)
_BOLTZMANN_PARAMETER = 2.0
_RESOLUTION = 0.1  # The resolution of the pitch bins in semitones.
_MAX_TRANSITION_RATE = 35.92  # The largest pitch change in octaves per second.
_SWITCH_PROB = 0.01
_NO_TROUGH_PROB = 0.01
_TRANSITION_MIN_PROB = 1e-4

# The smallest positive number added before taking the logarithm as in librosa.
_TINY = np.finfo(np.float64).tiny


def _beta_cdf(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """Calculate the CDF of the beta distribution with integer parameters."""
    n = a + b - 1
    terms = [comb(n, j) * x**j * (1 - x) ** (n - j) for j in range(a, n + 1)]
    return np.sum(terms, axis=0)


def _boltzmann_pmf(k: np.ndarray, lambda_: float, n: np.ndarray) -> np.ndarray:
    """Calculate the PMF of the truncated discrete exponential distribution."""
    support = (0 <= k) & (k < n)
    safe_n = np.where(0 < n, n, 1)
    pmf = (
        (1 - np.exp(-lambda_)) * np.exp(-lambda_ * k) / (1 - np.exp(-lambda_ * safe_n))
    )
    return np.where(support, pmf, 0.0)


def _local_transition(n_states: int, width: int) -> np.ndarray:
    """Create the transition with a triangular window as in librosa."""
    m = np.arange(1, (width + 1) // 2 + 1)
    if width % 2 == 0:
        half = (2 * m - 1) / width
        window = np.r_[half, half[::-1]]
    else:
        half = 2 * m / (width + 1)
        window = np.r_[half, half[-2::-1]]
    offsets = np.arange(n_states)[None, :] - np.arange(n_states)[:, None]
    inside = np.abs(offsets) <= width // 2
    index = np.clip(offsets + width // 2, 0, width - 1)
    transition = np.where(inside, window[index], 0.0)
    return transition / transition.sum(axis=1, keepdims=True)


def _observation(
    cmndf: np.ndarray,
    shifts: np.ndarray,
    is_trough: np.ndarray,
    beta_probs: np.ndarray,
    thresholds: np.ndarray,
    min_period: int,
    fmin: float,
    n_bins: int,
    bins_per_semitone: int,
) -> np.ndarray:
    """Calculate the observation probabilities of the voiced states of a frame."""
    probs = np.zeros(n_bins + 1)
    (troughs,) = np.nonzero(is_trough)
    if len(troughs) == 0:
        return probs[:n_bins]
    heights = cmndf[troughs]
    below = np.less.outer(heights, thresholds[1:])
    positions = np.cumsum(below, axis=0) - 1
    n_troughs = np.count_nonzero(below, axis=0)
    prior = _boltzmann_pmf(positions, _BOLTZMANN_PARAMETER, n_troughs)
    prior[~below] = 0
    trough_probs = prior.dot(beta_probs)
    global_min = np.argmin(heights)
    n_below_min = np.count_nonzero(~below[global_min])
    trough_probs[global_min] += _NO_TROUGH_PROB * np.sum(beta_probs[:n_below_min])

    nonzero = trough_probs != 0
    periods = min_period + troughs[nonzero] + shifts[troughs[nonzero]]
    f0 = SAMPLE_RATE / periods
    bins = np.clip(
        np.round(12 * bins_per_semitone * np.log2(f0 / fmin)), 0, n_bins
    ).astype(int)
    # The last assignment wins for the same bin as in librosa, where the bin beyond
    # the highest one is discarded.
    probs[bins] = trough_probs[nonzero]
    return probs[:n_bins]


def pyin(frames: np.ndarray, hop: int, f0_range: tuple[float, float]) -> np.ndarray:
    """Estimate F0 by pYIN as in librosa.

    Parameters
    ----------
    frames : np.ndarray
        The frames at 16 kHz with shape (T, frame_length).

    hop : int
        The frame shift in samples.

    f0_range : tuple[float, float]
        The F0 search range in Hz.

    Returns
    -------
    out : np.ndarray
        The F0 in Hz with shape (T,), where unvoiced frames are 0.

    """
    fmin, fmax = f0_range
    min_period = int(np.floor(SAMPLE_RATE / fmax))
    max_period = min(int(np.ceil(SAMPLE_RATE / fmin)), frames.shape[1] - 1)
    thresholds = np.linspace(0, 1, _N_THRESHOLDS + 1)
    beta_probs = np.diff(_beta_cdf(thresholds, *_BETA_PARAMETERS))
    bins_per_semitone = int(np.ceil(1.0 / _RESOLUTION))
    n_bins = int(np.floor(12 * bins_per_semitone * np.log2(fmax / fmin))) + 1

    # The transition between the voiced and unvoiced states of the pitch bins.
    max_semitones = round(_MAX_TRANSITION_RATE * 12 * hop / SAMPLE_RATE)
    local = _local_transition(n_bins, max_semitones * bins_per_semitone + 1)
    switch = np.array(
        [[1 - _SWITCH_PROB, _SWITCH_PROB], [_SWITCH_PROB, 1 - _SWITCH_PROB]]
    )
    log_transition = np.log(np.kron(switch, local) + _TINY)
    n_states = 2 * n_bins

    # The predecessors of each state in ascending order, where the transitions below
    # the minimum probability are pruned as in librosa.
    allowed = np.log(_TRANSITION_MIN_PROB + _TINY) <= log_transition
    width = int(allowed.sum(axis=0).max())
    predecessors = np.zeros((n_states, width), dtype=np.int64)
    weights = np.full((n_states, width), -np.inf)
    for j in range(n_states):
        (k,) = np.nonzero(allowed[:, j])
        predecessors[j, : len(k)] = k
        weights[j, : len(k)] = log_transition[k, j]

    num_frames = len(frames)
    pointers = np.zeros((num_frames, n_states), dtype=np.uint16)
    value = np.zeros(n_states)
    states = np.arange(n_states)
    for start in range(0, num_frames, 1024):
        cmndf = cumulative_mean_normalized_difference(
            frames[start : start + 1024], min_period, max_period
        )
        shifts = parabolic_shifts(cmndf)
        is_trough = find_troughs(cmndf)
        for i in range(len(cmndf)):
            voiced = _observation(
                cmndf[i],
                shifts[i],
                is_trough[i],
                beta_probs,
                thresholds,
                min_period,
                fmin,
                n_bins,
                bins_per_semitone,
            )
            unvoiced = (1 - np.clip(voiced.sum(), 0, 1)) / n_bins
            log_prob = np.log(np.r_[voiced, np.full(n_bins, unvoiced)] + _TINY)
            t = start + i
            if t == 0:
                value = log_prob + np.log(1 / n_states + _TINY)
                continue
            scores = value[predecessors] + weights
            best = np.argmax(scores, axis=1)
            pointers[t] = predecessors[states, best]
            value = scores[states, best] + log_prob

    path = np.zeros(num_frames, dtype=np.int64)
    path[-1] = np.argmax(value)
    for t in range(num_frames - 1, 0, -1):
        path[t - 1] = pointers[t, path[t]]
    frequencies = fmin * 2 ** (np.arange(n_bins) / (12 * bins_per_semitone))
    return np.where(path < n_bins, frequencies[path % n_bins], 0.0)


@dataclass(frozen=True)
class PyinConfig(AlgorithmConfig):
    """Configuration of pYIN.

    The default search range covers both speech and singing voice. The voicing is
    decided by Viterbi decoding, and thus no threshold is given.

    """

    fmin: float | None = 40.0
    fmax: float | None = 800.0


class PyinAlgorithm(BaseAlgorithm):
    """pYIN: probabilistic YIN (Mauch and Dixon, 2014)."""

    config_class = PyinConfig
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=False)
    # The pitch can move by the semitones rounded from 35.92 octaves per second in a
    # frame, which is 0 for frame periods shorter than about 1.2 ms. For long frame
    # periods, e.g., 50 ms, the onsets of the decoded F0 rise from lower pitches.
    min_frame_period = 2.0
    max_frame_period = 10.0

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
            The estimated contour, whose voicing is decided by Viterbi decoding.

        """
        frames, hop = frame_audio(audio.data, SAMPLE_RATE, frame_period, FRAME_LENGTH)
        f0 = pyin(frames, hop, f0_range)
        return NativeContour(times=np.arange(len(f0)) * hop / SAMPLE_RATE, f0=f0)
