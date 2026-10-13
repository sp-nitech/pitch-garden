"""Utility functions for the evaluation metrics."""

from typing import SupportsFloat

import numpy as np

from ..interfaces.types import Pitch

F0Like = Pitch | np.ndarray


def prepare(
    reference: F0Like, estimate: F0Like, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the F0 arrays for evaluation.

    The longer contour is truncated to the shorter one, and the masked frames are
    removed.

    Parameters
    ----------
    reference : Pitch | np.ndarray
        The reference F0 in Hz (unvoiced: 0).

    estimate : Pitch | np.ndarray
        The estimated F0 in Hz (unvoiced: 0).

    mask : np.ndarray | None, optional
        The boolean array that is True for the frames to be evaluated, e.g., False
        for the frames whose reference is unreliable.

    Returns
    -------
    out : tuple[np.ndarray, np.ndarray]
        The reference and estimated F0.

    Raises
    ------
    ValueError
        If the frame periods are different or the mask is too short.

    """
    if isinstance(reference, Pitch) and isinstance(estimate, Pitch):
        if reference.frame_period != estimate.frame_period:
            raise ValueError(
                f"The frame periods are different: {reference.frame_period} and "
                f"{estimate.frame_period}."
            )
    ref = reference.f0 if isinstance(reference, Pitch) else np.asarray(reference)
    est = estimate.f0 if isinstance(estimate, Pitch) else np.asarray(estimate)
    n = min(len(ref), len(est))
    ref = ref[:n].astype(np.float64)
    est = est[:n].astype(np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if len(mask) < n:
            raise ValueError(
                f"The mask has {len(mask)} frames, fewer than the {n} frames evaluated."
            )
        mask = mask[:n]
        ref, est = ref[mask], est[mask]
    return ref, est


def safe_divide(numerator: SupportsFloat, denominator: SupportsFloat) -> float:
    """Divide and return NaN if the denominator is zero."""
    n, d = float(numerator), float(denominator)
    return n / d if 0 < d else float("nan")
