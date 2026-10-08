"""A module for the SwiftF0 algorithm."""

from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from ..interfaces.types import Audio
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy


def _import_swift_f0() -> Any:
    """Import the swift-f0 package lazily.

    Returns
    -------
    out : module
        The swift_f0 module.

    Raises
    ------
    ImportError
        If the swift-f0 package is not installed.

    """
    try:
        import swift_f0
    except ImportError as e:
        raise ImportError(
            "SwiftF0 requires the swift-f0 package. "
            'Install it via `pip install "pitch-garden[lite]"`.'
        ) from e
    return swift_f0


@dataclass(frozen=True)
class SwiftF0Config(AlgorithmConfig):
    """Configuration of SwiftF0."""

    # The frames whose confidence is greater than or equal to the threshold are
    # regarded as voiced. The default value is that of swift-f0.
    voicing_threshold: float = 0.5

    # The number of CPU threads used by ONNX Runtime for inference. If None, the
    # number of physical cores is used, or 1 when running in parallel.
    num_threads: int | None = None


class SwiftF0Algorithm(BaseAlgorithm):
    """SwiftF0: a fast and accurate monophonic pitch detector.

    See https://github.com/lars76/swift-f0 for details.

    """

    config_class = SwiftF0Config
    is_neural = True
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=False)

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize SwiftF0.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of SwiftF0. The upstream package runs only on CPU.

        """
        super().__init__(config)
        self.model: Any = None

    @classmethod
    def get_supported_sample_rates(cls) -> tuple[int, int] | None:
        """Get the sample rate of the model published by swift-f0.

        Returns
        -------
        out : tuple[int, int] | None
            The sample rate in Hz as the lower and upper limits.

        """
        swift_f0 = _import_swift_f0()
        return int(swift_f0.SAMPLE_RATE), int(swift_f0.SAMPLE_RATE)

    @classmethod
    def get_f0_limits(cls) -> tuple[float, float]:
        """Get the F0 range published by swift-f0.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper limits in Hz.

        """
        swift_f0 = _import_swift_f0()
        return float(swift_f0.FMIN), float(swift_f0.FMAX)

    @classmethod
    def parallel_config(cls, config: AlgorithmConfig) -> AlgorithmConfig:
        """Use a single thread when running in parallel.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of SwiftF0.

        Returns
        -------
        out : AlgorithmConfig
            The adjusted configuration.

        """
        if isinstance(config, SwiftF0Config) and config.num_threads is None:
            return replace(config, num_threads=1)
        return config

    def load(self) -> None:
        """Load the ONNX model.

        Raises
        ------
        ImportError
            If the swift-f0 package is not installed.

        """
        if self.model is not None:
            return
        swift_f0 = _import_swift_f0()
        config = cast(SwiftF0Config, self.config)
        self.model = swift_f0.SwiftF0(threads=config.num_threads)

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
            The frame period of the output, which is ignored because the native frame
            period of SwiftF0 is fixed to 16 ms.

        Returns
        -------
        out : NativeContour
            The estimated contour with 16 ms frame period.

        """
        self.load()
        config = cast(SwiftF0Config, self.config)
        fmin, fmax = f0_range
        result = self.model.detect(audio.data, audio.sample_rate, fmin=fmin, fmax=fmax)
        candidate = np.asarray(result.pitch_hz, dtype=np.float64)
        score = np.asarray(result.confidence, dtype=np.float64)
        f0 = np.where(config.voicing_threshold <= score, candidate, 0.0)
        return NativeContour(
            times=np.asarray(result.timestamps, dtype=np.float64),
            f0=f0,
            score=score,
            candidate=candidate,
        )
