"""The module for evaluating F0 contours.

The following metrics are provided:

- VDE (Voicing Decision Error): the ratio of frames with wrong voicing decisions.
- Voicing precision, recall, and F1, where voiced frames are regarded as positive.
- GPE (Gross Pitch Error): the ratio of frames whose relative error exceeds 20%
  among the frames voiced in both.
- FPE (Fine Pitch Error): the standard deviation of the error in cents among the
  frames voiced in both without gross errors.
- RPA (Raw Pitch Accuracy): the ratio of reference voiced frames estimated within
  50 cents.
- RCA (Raw Chroma Accuracy): RPA ignoring octave errors.
- OA (Overall Accuracy): the ratio of frames that are correctly unvoiced or voiced
  in both with errors within 50 cents.
- Pitch F1: the harmonic mean of the pitch precision and recall at 50 cents,
  where a hit is a frame voiced in both with an error within 50 cents.

"""

import numpy as np

from ._utils import F0Like
from .pitch import f1_score, fpe, gpe, oa, rca, rpa
from .voicing import vde, voicing_scores


def evaluate(
    reference: F0Like,
    estimate: F0Like,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate all metrics with their default settings.

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
        The metrics keyed by their names.

    """
    voicing = voicing_scores(reference, estimate, mask)
    return {
        "vde": vde(reference, estimate, mask),
        "voicing_precision": voicing["precision"],
        "voicing_recall": voicing["recall"],
        "voicing_f1": voicing["f1"],
        "gpe": gpe(reference, estimate, mask),
        "fpe": fpe(reference, estimate, mask),
        "rpa": rpa(reference, estimate, mask),
        "rca": rca(reference, estimate, mask),
        "oa": oa(reference, estimate, mask),
        "f1": f1_score(reference, estimate, mask),
    }


__all__ = [
    "evaluate",
    "f1_score",
    "fpe",
    "gpe",
    "oa",
    "rca",
    "rpa",
    "vde",
    "voicing_scores",
]
