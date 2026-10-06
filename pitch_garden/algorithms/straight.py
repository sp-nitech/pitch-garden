"""A module for the F0 extraction of STRAIGHT."""

import sys
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..interfaces.types import Audio
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy


def _import_pylstraight() -> Any:
    """Import the pylstraight package lazily.

    Returns
    -------
    out : module
        The pylstraight module.

    Raises
    ------
    ImportError
        If the pylstraight package is not installed.

    """
    try:
        import pylstraight
    except ImportError as e:
        raise ImportError(
            "STRAIGHT requires the pylstraight package. "
            'Install it via `pip install "pitch-garden[lite]"`.'
        ) from e
    return pylstraight


@dataclass(frozen=True)
class StraightConfig(AlgorithmConfig):
    """Configuration of STRAIGHT.

    The default search range covers both speech and singing voice.

    """

    fmin: float | None = 40.0
    fmax: float | None = 800.0

    # The seed of the random noise added inside pylstraight, which makes the
    # results reproducible. If None, the results vary slightly at every call.
    seed: int | None = 0


class StraightAlgorithm(BaseAlgorithm):
    """STRAIGHT: the F0 extraction based on multiple cues (Kawahara et al., 2005).

    This uses pylstraight, a Python port of Legacy STRAIGHT. See
    https://github.com/takenori-y/pylstraight for details. The voicing decision is
    made inside the algorithm, and thus the voicing score is not provided. Note that
    the voicing decision relies on the contrast with silent segments.

    """

    config_class = StraightConfig
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=False)

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize STRAIGHT.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of STRAIGHT.

        """
        super().__init__(config)
        self.pyls: Any = None

    @classmethod
    def get_supported_sample_rates(cls) -> tuple[int, int] | None:
        """Get the sample rates that pylstraight accepts.

        Returns
        -------
        out : tuple[int, int] | None
            The lower and upper limits in Hz, where the upper one is unlimited.

        """
        return int(_import_pylstraight().min_fs), sys.maxsize

    @classmethod
    def get_f0_limits(cls) -> tuple[float, float]:
        """Get the F0 range that pylstraight accepts.

        The upper limit also has to be lower than or equal to the Nyquist frequency.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper limits in Hz.

        """
        return float(_import_pylstraight().min_f0), float("inf")

    def load(self) -> None:
        """Import pylstraight.

        Raises
        ------
        ImportError
            If the pylstraight package is not installed.

        """
        if self.pyls is None:
            self.pyls = _import_pylstraight()

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Estimate the F0 contour.

        Parameters
        ----------
        audio : Audio
            The input audio.

        f0_range : tuple[float, float]
            The F0 search range in Hz.

        frame_period : float
            The frame period in milliseconds.

        Returns
        -------
        out : NativeContour
            The estimated contour.

        Raises
        ------
        ValueError
            If the search range is invalid for pylstraight, e.g., above the Nyquist
            frequency or too narrow.

        """
        self.load()
        config = cast(StraightConfig, self.config)
        f0 = self.pyls.extract_f0(
            audio.data,
            audio.sample_rate,
            frame_shift=frame_period,
            f0_range=f0_range,
            seed=config.seed,
        )
        f0 = np.asarray(f0, dtype=np.float64)
        times = np.arange(len(f0)) * frame_period / 1000
        return NativeContour(times=times, f0=f0)
