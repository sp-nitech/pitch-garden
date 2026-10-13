"""A module for the voter aligning the trackers before voting."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..interfaces.types import Pitch
from ..utils.grid import resample
from ..utils.units import cents_to_hz, hz_to_cents
from .base import BaseVoter

_EPS = 1e-9

logger = logging.getLogger("pitch_garden")


@dataclass(frozen=True)
class Alignment:
    """Dataclass to hold the correction of an F0 contour."""

    shift: float  # The shift in frames; a positive value delays the contour.
    bias: float  # The bias in cents subtracted from the voiced frames.
    applied: bool  # Whether the correction is estimated and applied.


NO_ALIGNMENT = Alignment(shift=0.0, bias=0.0, applied=False)


def validate_parameters(
    max_shift: float, shift_step: float, tolerance: float, min_overlap: int
) -> None:
    """Validate the parameters of the alignment.

    Parameters
    ----------
    max_shift : float
        The maximum shift in frames.

    shift_step : float
        The step of the candidate shifts in frames.

    tolerance : float
        The tolerance in cents of the raw pitch accuracy.

    min_overlap : int
        The minimum number of frames voiced in both.

    Raises
    ------
    ValueError
        If the parameters are invalid.

    """
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative.")
    if shift_step <= 0:
        raise ValueError("shift_step must be positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if min_overlap < 1:
        raise ValueError("min_overlap must be a positive integer.")


def shift_contour(pitch: Pitch, shift: float) -> np.ndarray:
    """Shift an F0 contour in time by interpolation.

    Parameters
    ----------
    pitch : Pitch
        The input contour.

    shift : float
        The shift in frames, where a positive value delays the contour.

    Returns
    -------
    out : np.ndarray
        The shifted F0, where the frames shifted in from outside are unvoiced.

    """
    if shift == 0 or pitch.length == 0:
        return pitch.f0.copy()
    period = pitch.frame_period / 1000
    times = (np.arange(pitch.length) + shift) * period
    grid = resample(
        times, pitch.f0, num_frames=pitch.length, frame_period=pitch.frame_period
    )
    t = np.arange(pitch.length) * period
    inside = (times[0] - _EPS <= t) & (t <= times[-1] + _EPS)
    return np.where(inside, grid.f0, 0.0)


def estimate_alignment(
    estimate: Pitch,
    reference: Pitch,
    *,
    max_shift: float = 2.0,
    shift_step: float = 0.1,
    tolerance: float = 50.0,
    min_overlap: int = 50,
) -> Alignment:
    """Estimate the shift and the bias aligning an F0 contour to a reference.

    The shift maximizing the raw pitch accuracy with respect to the reference is
    searched within [-max_shift, max_shift] frames with the step of `shift_step`
    frames. Unlike the original method, the bias, i.e., the median of the cent
    differences in the frames voiced in both, is removed for each candidate shift
    before computing the accuracy, so that the shift can be found even if the bias
    exceeds the tolerance. Among the shifts with the maximum accuracy, the one
    closest to zero is chosen.

    Parameters
    ----------
    estimate : Pitch
        The contour to be aligned.

    reference : Pitch
        The reference contour, e.g., the output of another tracker.

    max_shift : float, optional
        The maximum shift in frames.

    shift_step : float, optional
        The step of the candidate shifts in frames.

    tolerance : float, optional
        The tolerance in cents of the raw pitch accuracy.

    min_overlap : int, optional
        The minimum number of frames voiced in both. If the number is smaller, no
        correction is applied.

    Returns
    -------
    out : Alignment
        The estimated correction.

    Raises
    ------
    ValueError
        If the parameters are invalid or the contours are inconsistent.

    """
    validate_parameters(max_shift, shift_step, tolerance, min_overlap)
    if estimate.length != reference.length:
        raise ValueError("The contours must have the same number of frames.")
    if estimate.frame_period != reference.frame_period:
        raise ValueError("The contours must have the same frame period.")

    num_steps = int(np.floor(max_shift / shift_step + _EPS))
    shifts = np.arange(-num_steps, num_steps + 1) * shift_step
    shifts = shifts[np.argsort(np.abs(shifts), kind="stable")]

    ref = reference.f0
    num_ref_voiced = np.count_nonzero(0 < ref)
    best: tuple[float, float, float] | None = None  # (accuracy, shift, bias)
    for shift in shifts:
        est = shift_contour(estimate, float(shift))
        both = (0 < ref) & (0 < est)
        if np.count_nonzero(both) < min_overlap:
            continue
        cents = hz_to_cents(est[both], ref[both])
        bias = float(np.median(cents))
        accuracy = np.count_nonzero(np.abs(cents - bias) < tolerance) / num_ref_voiced
        if best is None or best[0] + _EPS < accuracy:
            best = (accuracy, float(shift), bias)

    if best is None:
        return NO_ALIGNMENT
    return Alignment(shift=best[1], bias=best[2], applied=True)


def apply_alignment(pitch: Pitch, alignment: Alignment) -> Pitch:
    """Apply a correction to an F0 contour.

    Parameters
    ----------
    pitch : Pitch
        The input contour.

    alignment : Alignment
        The correction.

    Returns
    -------
    out : Pitch
        The corrected contour.

    """
    if not alignment.applied:
        return pitch
    f0 = shift_contour(pitch, alignment.shift)
    f0 = np.where(0 < f0, cents_to_hz(-alignment.bias, reference=f0), 0.0)
    return pitch.with_f0(f0)


class AlignedVoter(BaseVoter):
    """A voter aligning the trackers to a reference tracker before voting.

    For each utterance, the output of each tracker is shifted in time and corrected
    in frequency to match the output of the reference tracker, and the corrected
    outputs are integrated by the inner voter [1]_.

    References
    ----------
    .. [1] J. Koguchi and T. Koriyama, "Voting-based pitch estimation with temporal
       and frequential alignment and correlation aware selection," in Proc.
       ICASSP, 2026. https://arxiv.org/abs/2602.01727

    """

    def __init__(
        self,
        reference: str,
        voter: BaseVoter | str = "median",
        voter_params: Mapping[str, Any] | None = None,
        *,
        max_shift: float = 2.0,
        shift_step: float = 0.1,
        tolerance: float = 50.0,
        min_overlap: int = 50,
    ) -> None:
        """Initialize the voter.

        Parameters
        ----------
        reference : str
            The name of the reference tracker.

        voter : BaseVoter | str, optional
            The inner voter integrating the aligned outputs, or its registered name.

        voter_params : Mapping[str, Any] | None, optional
            The parameters of the inner voter given by its name.

        max_shift : float, optional
            The maximum shift in frames.

        shift_step : float, optional
            The step of the candidate shifts in frames.

        tolerance : float, optional
            The tolerance in cents of the raw pitch accuracy used for the alignment.

        min_overlap : int, optional
            The minimum number of frames voiced in both a tracker and the reference.
            If the number is smaller, the tracker is not corrected.

        Raises
        ------
        ValueError
            If the parameters are invalid.

        """
        validate_parameters(max_shift, shift_step, tolerance, min_overlap)
        if isinstance(voter, str):
            from . import create_voter

            voter = create_voter(voter, **(voter_params or {}))
        self.reference = reference
        self.voter = voter
        self.max_shift = max_shift
        self.shift_step = shift_step
        self.tolerance = tolerance
        self.min_overlap = min_overlap

    def check_trackers(self, names: Sequence[str]) -> None:
        """Check that the reference is one of the trackers.

        Parameters
        ----------
        names : Sequence[str]
            The names of the trackers.

        Raises
        ------
        ValueError
            If the reference is not found.

        """
        if self.reference not in names:
            raise ValueError(
                f"The reference tracker '{self.reference}' is not found in "
                f"{list(names)}."
            )
        self.voter.check_trackers(names)

    def align(self, pitches: Mapping[str, Pitch]) -> dict[str, tuple[Pitch, Alignment]]:
        """Align the outputs of the trackers to the reference.

        Parameters
        ----------
        pitches : Mapping[str, Pitch]
            The outputs of the trackers keyed by the tracker names.

        Returns
        -------
        out : dict[str, tuple[Pitch, Alignment]]
            The aligned outputs and their corrections keyed by the tracker names. If
            the reference is not found, e.g., it failed in the pipeline, the outputs
            are not corrected.

        """
        if self.reference not in pitches:
            logger.warning(
                f"The reference tracker '{self.reference}' is not found, and the "
                "trackers are voted without the alignment."
            )
            return {name: (pitch, NO_ALIGNMENT) for name, pitch in pitches.items()}
        reference = pitches[self.reference]
        aligned = {}
        for name, pitch in pitches.items():
            if name == self.reference:
                aligned[name] = (pitch, NO_ALIGNMENT)
                continue
            alignment = estimate_alignment(
                pitch,
                reference,
                max_shift=self.max_shift,
                shift_step=self.shift_step,
                tolerance=self.tolerance,
                min_overlap=self.min_overlap,
            )
            aligned[name] = (apply_alignment(pitch, alignment), alignment)
        return aligned

    def vote(self, pitches: Mapping[str, Pitch]) -> Pitch:
        """Align and integrate the outputs of the trackers.

        Parameters
        ----------
        pitches : Mapping[str, Pitch]
            The outputs of the trackers keyed by the tracker names.

        Returns
        -------
        out : Pitch
            The integrated F0 contour.

        """
        aligned = {name: p for name, (p, _) in self.align(pitches).items()}
        return self.voter(aligned)
