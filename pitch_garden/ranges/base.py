"""A module defining the base class of F0 range estimators."""

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("pitch_garden")


@dataclass
class F0Range:
    """Dataclass to hold an estimated F0 search range."""

    fmin: float  # The lower bound in Hz.
    fmax: float  # The upper bound in Hz.
    num_voiced_frames: int  # The number of voiced frames used for the estimation.
    fallback: bool = False  # Whether the first-pass range is used as it is.

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization.

        Returns
        -------
        out : dict[str, Any]
            The dictionary of plain Python objects.

        """
        out = asdict(self)
        out["fmin"] = float(self.fmin)
        out["fmax"] = float(self.fmax)
        return out


class BaseRangeEstimator(ABC):
    """An abstract base class for F0 range estimators."""

    def __init__(self, min_octaves: float = 0.5) -> None:
        """Initialize the estimator.

        Parameters
        ----------
        min_octaves : float, optional
            The minimum width of the range in octaves. A narrower range, e.g., of a
            steady tone, is widened around its geometric center.

        Raises
        ------
        ValueError
            If min_octaves is negative.

        """
        if min_octaves < 0:
            raise ValueError("min_octaves must be non-negative.")
        self.min_octaves = min_octaves

    def __call__(self, values: np.ndarray, default: tuple[float, float]) -> F0Range:
        """Estimate the F0 search range from voiced F0 values.

        Parameters
        ----------
        values : np.ndarray
            The voiced F0 values in Hz.

        default : tuple[float, float]
            The first-pass range, which is used as a fallback and bounds the result.

        Returns
        -------
        out : F0Range
            The estimated F0 search range.

        """
        values = np.asarray(values, dtype=np.float64)
        values = values[0 < values]
        if len(values) == 0:
            logger.warning(
                f"No voiced frames are found. The range {default} is used as it is."
            )
            return F0Range(default[0], default[1], len(values), fallback=True)

        fmin, fmax = self.estimate(values)
        if 0 < fmin <= fmax:
            fmin, fmax = self._widen(fmin, fmax)
        fmin = float(np.clip(fmin, default[0], default[1]))
        fmax = float(np.clip(fmax, default[0], default[1]))
        if fmax <= fmin:
            return F0Range(default[0], default[1], len(values), fallback=True)
        return F0Range(fmin, fmax, len(values))

    def _widen(self, fmin: float, fmax: float) -> tuple[float, float]:
        """Widen the range to the minimum width around its geometric center."""
        octaves = np.log2(fmax / fmin)
        if self.min_octaves <= octaves:
            return fmin, fmax
        center = np.sqrt(fmin * fmax)
        half = self.min_octaves / 2
        return float(center * 2**-half), float(center * 2**half)

    @abstractmethod
    def estimate(self, values: np.ndarray) -> tuple[float, float]:
        """Estimate the F0 search range.

        Parameters
        ----------
        values : np.ndarray
            The voiced F0 values in Hz.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper bounds in Hz.

        """
