"""Metrics on the F0 values."""

import numpy as np

from ..utils.units import hz_to_cents
from ._utils import F0Like, prepare, safe_divide


def gpe(
    reference: F0Like,
    estimate: F0Like,
    mask: np.ndarray | None = None,
    threshold: float = 0.2,
) -> float:
    """Calculate GPE (Gross Pitch Error).

    Parameters
    ----------
    reference : Pitch | np.ndarray
        The reference F0 in Hz (unvoiced: 0).

    estimate : Pitch | np.ndarray
        The estimated F0 in Hz (unvoiced: 0).

    mask : np.ndarray | None, optional
        The boolean array that is True for the frames to be evaluated.

    threshold : float, optional
        The relative error regarded as gross.

    Returns
    -------
    out : float
        The ratio of gross errors among the frames voiced in both.

    """
    ref, est = prepare(reference, estimate, mask)
    both = (0 < ref) & (0 < est)
    gross = threshold < np.abs(est[both] / ref[both] - 1)
    return safe_divide(np.sum(gross), np.sum(both))


def fpe(
    reference: F0Like,
    estimate: F0Like,
    mask: np.ndarray | None = None,
    threshold: float = 0.2,
) -> float:
    """Calculate FPE (Fine Pitch Error).

    Parameters
    ----------
    reference : Pitch | np.ndarray
        The reference F0 in Hz (unvoiced: 0).

    estimate : Pitch | np.ndarray
        The estimated F0 in Hz (unvoiced: 0).

    mask : np.ndarray | None, optional
        The boolean array that is True for the frames to be evaluated.

    threshold : float, optional
        The relative error regarded as gross, which is excluded.

    Returns
    -------
    out : float
        The standard deviation of the error in cents among the frames voiced in
        both without gross errors.

    """
    ref, est = prepare(reference, estimate, mask)
    both = (0 < ref) & (0 < est)
    fine = both.copy()
    fine[both] = np.abs(est[both] / ref[both] - 1) <= threshold
    if not np.any(fine):
        return float("nan")
    return float(np.std(hz_to_cents(est[fine], ref[fine])))


def _hits(
    ref: np.ndarray, est: np.ndarray, tolerance: float, chroma: bool
) -> np.ndarray:
    """Find the frames whose estimated F0 is within the tolerance."""
    both = (0 < ref) & (0 < est)
    hits = np.zeros_like(both)
    diff = hz_to_cents(est[both], ref[both])
    if chroma:
        diff = (diff + 600) % 1200 - 600
    hits[both] = np.abs(diff) < tolerance
    return hits


def rpa(
    reference: F0Like,
    estimate: F0Like,
    mask: np.ndarray | None = None,
    tolerance: float = 50.0,
) -> float:
    """Calculate RPA (Raw Pitch Accuracy).

    Parameters
    ----------
    reference : Pitch | np.ndarray
        The reference F0 in Hz (unvoiced: 0).

    estimate : Pitch | np.ndarray
        The estimated F0 in Hz (unvoiced: 0).

    mask : np.ndarray | None, optional
        The boolean array that is True for the frames to be evaluated.

    tolerance : float, optional
        The tolerance in cents.

    Returns
    -------
    out : float
        The ratio of reference voiced frames estimated within the tolerance.

    """
    ref, est = prepare(reference, estimate, mask)
    return safe_divide(np.sum(_hits(ref, est, tolerance, False)), np.sum(0 < ref))


def rca(
    reference: F0Like,
    estimate: F0Like,
    mask: np.ndarray | None = None,
    tolerance: float = 50.0,
) -> float:
    """Calculate RCA (Raw Chroma Accuracy), i.e., RPA ignoring octave errors.

    Parameters
    ----------
    reference : Pitch | np.ndarray
        The reference F0 in Hz (unvoiced: 0).

    estimate : Pitch | np.ndarray
        The estimated F0 in Hz (unvoiced: 0).

    mask : np.ndarray | None, optional
        The boolean array that is True for the frames to be evaluated.

    tolerance : float, optional
        The tolerance in cents.

    Returns
    -------
    out : float
        The ratio of reference voiced frames estimated within the tolerance after
        octave folding.

    """
    ref, est = prepare(reference, estimate, mask)
    return safe_divide(np.sum(_hits(ref, est, tolerance, True)), np.sum(0 < ref))


def oa(
    reference: F0Like,
    estimate: F0Like,
    mask: np.ndarray | None = None,
    tolerance: float = 50.0,
) -> float:
    """Calculate OA (Overall Accuracy).

    Parameters
    ----------
    reference : Pitch | np.ndarray
        The reference F0 in Hz (unvoiced: 0).

    estimate : Pitch | np.ndarray
        The estimated F0 in Hz (unvoiced: 0).

    mask : np.ndarray | None, optional
        The boolean array that is True for the frames to be evaluated.

    tolerance : float, optional
        The tolerance in cents.

    Returns
    -------
    out : float
        The ratio of frames that are correctly unvoiced or voiced in both with the
        estimated F0 within the tolerance.

    """
    ref, est = prepare(reference, estimate, mask)
    unvoiced = (ref <= 0) & (est <= 0)
    correct = unvoiced | _hits(ref, est, tolerance, False)
    return safe_divide(np.sum(correct), len(ref))


def f1_score(
    reference: F0Like,
    estimate: F0Like,
    mask: np.ndarray | None = None,
    tolerance: float = 50.0,
) -> float:
    """Calculate the pitch F1 score.

    A frame is a hit if both are voiced and the error is within the tolerance. The
    precision and recall are the ratios of hits to the estimated and reference
    voiced frames, respectively.

    Parameters
    ----------
    reference : Pitch | np.ndarray
        The reference F0 in Hz (unvoiced: 0).

    estimate : Pitch | np.ndarray
        The estimated F0 in Hz (unvoiced: 0).

    mask : np.ndarray | None, optional
        The boolean array that is True for the frames to be evaluated.

    tolerance : float, optional
        The tolerance in cents.

    Returns
    -------
    out : float
        The pitch F1 score.

    """
    ref, est = prepare(reference, estimate, mask)
    hits = np.sum(_hits(ref, est, tolerance, False))
    precision = safe_divide(hits, np.sum(0 < est))
    recall = safe_divide(hits, np.sum(0 < ref))
    return safe_divide(2 * precision * recall, precision + recall)
