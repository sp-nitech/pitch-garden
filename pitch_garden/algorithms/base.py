"""A module defining the base class of pitch tracking algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal, get_args

from ..interfaces.types import Audio, NativeContour


@dataclass(frozen=True)
class AlgorithmConfig:
    """The base configuration shared by all algorithms.

    Each algorithm defines its own configuration by inheriting this class and adding
    algorithm-specific fields.

    """

    fmin: float | None = None  # The lower bound of the F0 search range in Hz.
    fmax: float | None = None  # The upper bound of the F0 search range in Hz.


# The pass giving the voicing decisions in the two-pass tracking.
VoicingSource = Literal["1st", "2nd"]


@dataclass(frozen=True)
class TwoPassPolicy:
    """The settings of the two-pass tracking for an algorithm or a tracker.

    If the voicing is taken from the first pass, the F0 values of the second pass
    are used for the frames voiced in both passes, and those of the first pass for
    the frames voiced only in the first pass.

    """

    enabled: bool = True  # Whether the tracker is rerun in the second pass.
    voicing_from: VoicingSource = "2nd"  # The pass giving the voicing.

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises
        ------
        ValueError
            If the pass giving the voicing is unknown.

        """
        if self.voicing_from not in get_args(VoicingSource):
            raise ValueError(
                f"Unsupported voicing_from '{self.voicing_from}'. "
                f"Supported values are: {list(get_args(VoicingSource))}"
            )


class BaseAlgorithm(ABC):
    """An abstract base class for pitch tracking algorithms."""

    # The configuration class of the algorithm.
    config_class: ClassVar[type[AlgorithmConfig]] = AlgorithmConfig

    # Whether the F0 search range can be specified.
    supports_f0_range: ClassVar[bool] = True

    # Whether the algorithm is based on neural networks.
    is_neural: ClassVar[bool] = False

    # The recommended settings of the two-pass tracking.
    two_pass_policy: ClassVar[TwoPassPolicy] = TwoPassPolicy()

    # The sample rate in Hz to which the audio of unsupported sample rates is
    # resampled before the estimation.
    preferred_sample_rate: ClassVar[int] = 16000

    # The shortest frame period in milliseconds used for the estimation. A shorter
    # frame period is replaced by this, and the result is resampled onto the frame
    # grid. A short frame period increases the memory usage of some algorithms,
    # e.g., STRAIGHT, and is rejected by some, e.g., STRAIGHT accepts 1 ms or longer.
    min_frame_period: ClassVar[float] = 1.0

    # The longest frame period in milliseconds used for the estimation. A longer
    # frame period is replaced by this, and the result is resampled onto the frame
    # grid. A long frame period makes some algorithms fail, e.g., RAPT accepts
    # shorter than 100 ms, or degrades them, e.g., REAPER, DIO, and STRAIGHT.
    max_frame_period: ClassVar[float] = 50.0

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize the algorithm.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of the algorithm. Algorithms running on GPUs define
            the device as a field of their own configuration.

        """
        self.config = config

    @classmethod
    def is_parallel_safe(cls, config: AlgorithmConfig) -> bool:
        """Get whether the algorithm can be run in a worker process.

        Algorithms running on GPUs should return False for non-CPU devices.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of the algorithm.

        Returns
        -------
        out : bool
            True if the algorithm can be run in a worker process.

        """
        return True

    @classmethod
    def get_f0_limits(cls) -> tuple[float, float]:
        """Get the F0 range that the algorithm can handle.

        Algorithms should use the constants published by their upstream packages
        if available, importing the packages lazily.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper limits in Hz.

        """
        return 1.0, 10000.0

    @classmethod
    def get_supported_sample_rates(cls) -> tuple[int, int] | None:
        """Get the range of the sample rates that the algorithm accepts.

        The audio out of the range is resampled to `preferred_sample_rate` before
        the estimation. Algorithms should use the constants published by their
        upstream packages if available, importing the packages lazily.

        Returns
        -------
        out : tuple[int, int] | None
            The lower and upper limits in Hz, or None to accept any sample rate.

        """
        return None

    @classmethod
    def parallel_config(cls, config: AlgorithmConfig) -> AlgorithmConfig:
        """Adjust the configuration for running in a worker process.

        For example, the number of threads should be limited to avoid
        oversubscription. The values explicitly given by users should be kept.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of the algorithm.

        Returns
        -------
        out : AlgorithmConfig
            The adjusted configuration.

        """
        return config

    @abstractmethod
    def load(self) -> None:
        """Load the resources required by the algorithm.

        This is called before the estimation, so that optional dependencies are
        imported lazily and model weights are loaded only once. Algorithms requiring
        no resources implement this with an empty body.

        Raises
        ------
        ImportError
            If the optional dependencies are not installed.

        """

    @abstractmethod
    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Estimate the F0 contour on the native frame grid.

        The search range is given at each call rather than by the configuration, so
        that the range can be changed without reloading the algorithm.

        Parameters
        ----------
        audio : Audio
            The input audio.

        f0_range : tuple[float, float]
            The F0 search range in Hz within the limits of the algorithm. Algorithms
            not supporting the search range ignore it.

        frame_period : float
            The frame period of the output in milliseconds. Algorithms supporting an
            arbitrary frame period use it as the native one, which avoids the
            interpolation onto the output frame grid.

        Returns
        -------
        out : NativeContour
            The estimated contour.

        """
