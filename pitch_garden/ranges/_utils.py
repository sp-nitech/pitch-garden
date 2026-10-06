"""Utility functions for the F0 search range."""

from collections.abc import Sequence

import numpy as np

from ..interfaces.types import Pitch
from .base import BaseRangeEstimator, F0Range


def estimate_f0_range(
    pitches: Pitch | Sequence[Pitch],
    estimator: BaseRangeEstimator,
    default: tuple[float, float],
) -> F0Range:
    """Estimate the F0 search range from the voiced frames of contours.

    Parameters
    ----------
    pitches : Pitch | Sequence[Pitch]
        The contours whose voiced frames are pooled.

    estimator : BaseRangeEstimator
        The estimator.

    default : tuple[float, float]
        The first-pass range, which is used as a fallback and bounds the result.

    Returns
    -------
    out : F0Range
        The estimated F0 search range.

    """
    if isinstance(pitches, Pitch):
        pitches = [pitches]
    values = [p.f0[p.voiced] for p in pitches]
    pooled = np.concatenate(values) if values else np.zeros(0)
    return estimator(pooled, default=default)


def clip_to_range(pitch: Pitch, f0_range: F0Range) -> Pitch:
    """Unvoice the frames out of the F0 range.

    Parameters
    ----------
    pitch : Pitch
        The input contour.

    f0_range : F0Range
        The F0 range.

    Returns
    -------
    out : Pitch
        The clipped contour.

    """
    inside = (f0_range.fmin <= pitch.f0) & (pitch.f0 <= f0_range.fmax)
    return pitch.with_f0(np.where(inside, pitch.f0, 0.0))
