"""Metrics on the voicing decision."""

import numpy as np

from ._utils import F0Like, prepare, safe_divide


def vde(reference: F0Like, estimate: F0Like, mask: np.ndarray | None = None) -> float:
    """Calculate VDE (Voicing Decision Error).

    Parameters
    ----------
    reference : Pitch | np.ndarray
        The reference F0 in Hz (unvoiced: 0).

    estimate : Pitch | np.ndarray
        The estimated F0 in Hz (unvoiced: 0).

    mask : np.ndarray | None, optional
        The boolean array that is True for the frames to be evaluated.

    Returns
    -------
    out : float
        The ratio of frames whose voicing decision is wrong.

    """
    ref, est = prepare(reference, estimate, mask)
    return safe_divide(np.sum((0 < ref) != (0 < est)), len(ref))


def voicing_scores(
    reference: F0Like, estimate: F0Like, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Calculate the precision, recall, and F1 of the voicing decision.

    The voiced frames are regarded as positive.

    Parameters
    ----------
    reference : Pitch | np.ndarray
        The reference F0 in Hz (unvoiced: 0).

    estimate : Pitch | np.ndarray
        The estimated F0 in Hz (unvoiced: 0).

    mask : np.ndarray | None, optional
        The boolean array that is True for the frames to be evaluated.

    Returns
    -------
    out : dict[str, float]
        The 'precision', 'recall', and 'f1'.

    """
    ref, est = prepare(reference, estimate, mask)
    ref_voiced, est_voiced = 0 < ref, 0 < est
    tp = np.sum(ref_voiced & est_voiced)
    precision = safe_divide(tp, np.sum(est_voiced))
    recall = safe_divide(tp, np.sum(ref_voiced))
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
