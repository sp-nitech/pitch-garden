"""Dummy algorithms for testing."""

from dataclasses import dataclass, replace
from typing import ClassVar, cast

import numpy as np

from pitch_garden.algorithms.base import (
    AlgorithmConfig,
    BaseAlgorithm,
    NativeContour,
    TwoPassPolicy,
)
from pitch_garden.interfaces.types import Audio
from pitch_garden.ranges.base import BaseRangeEstimator


@dataclass(frozen=True)
class ConstantConfig(AlgorithmConfig):
    """Configuration of the constant algorithm."""

    f0: float = 100.0  # The F0 to be returned in Hz.
    voicing_threshold: float = 0.5  # The threshold on the score.
    score: float = 1.0  # The score of all frames.
    hop: float = 10.0  # The native frame period in milliseconds.
    num_threads: int | None = None  # The number of threads (only for testing).


class ConstantAlgorithm(BaseAlgorithm):
    """An algorithm returning a constant F0 for testing."""

    config_class = ConstantConfig

    @classmethod
    def get_f0_limits(cls) -> tuple[float, float]:
        """Get the F0 limits."""
        return 10.0, 5000.0

    num_calls: ClassVar[int] = 0
    num_instances: ClassVar[int] = 0

    def __init__(self, config: AlgorithmConfig) -> None:
        """Count the instantiations."""
        super().__init__(config)
        type(self).num_instances += 1

    def load(self) -> None:
        """Load nothing."""

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Return a constant F0 clipped by the search range."""
        type(self).num_calls += 1
        config = cast(ConstantConfig, self.config)
        n = max(1, int(audio.duration * 1000 / config.hop))
        times = np.arange(n) * config.hop / 1000
        candidate = np.full(n, config.f0)
        score = np.full(n, config.score)
        fmin, fmax = f0_range
        candidate = np.where((fmin <= candidate) & (candidate <= fmax), candidate, 0)
        f0 = np.where(config.voicing_threshold <= score, candidate, 0.0)
        return NativeContour(times=times, f0=f0, score=score, candidate=candidate)

    @classmethod
    def parallel_config(cls, config: AlgorithmConfig) -> AlgorithmConfig:
        """Use a single thread when running in parallel."""
        if isinstance(config, ConstantConfig) and config.num_threads is None:
            return replace(config, num_threads=1)
        return config


class FramePeriodAlgorithm(ConstantAlgorithm):
    """An algorithm recording the frame period given to the estimation."""

    min_frame_period = 2.0
    max_frame_period = 40.0

    received: ClassVar[float | None] = None

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Record the frame period and return a constant F0."""
        type(self).received = frame_period
        return super().estimate(audio, f0_range, frame_period)


@dataclass(frozen=True)
class RangelessConfig(AlgorithmConfig):
    """Configuration of the rangeless algorithm."""

    f0: float = 100.0  # The F0 to be returned in Hz.


class RangelessAlgorithm(BaseAlgorithm):
    """An algorithm whose search range cannot be specified."""

    config_class = RangelessConfig
    supports_f0_range = False
    num_calls: ClassVar[int] = 0

    def load(self) -> None:
        """Load nothing."""

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Return a constant F0."""
        type(self).num_calls += 1
        config = cast(RangelessConfig, self.config)
        n = max(1, int(audio.duration * 100))
        return NativeContour(times=np.arange(n) * 0.01, f0=np.full(n, config.f0))


class DeviceAlgorithm(BaseAlgorithm):
    """An algorithm that can run on GPUs."""

    is_neural = True
    supports_gpu = True

    # The device used for 'auto', which simulates an available GPU.
    available_device: ClassVar[str] = "cuda"

    @classmethod
    def resolve_device(cls, device: str) -> str:
        """Resolve 'auto' into the available device."""
        return cls.available_device if device == "auto" else device

    def load(self) -> None:
        """Load nothing."""

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Return an unvoiced contour."""
        return NativeContour(times=np.zeros(1), f0=np.zeros(1))


@dataclass(frozen=True)
class AlternatingConfig(AlgorithmConfig):
    """Configuration of the alternating algorithm."""


class AlternatingAlgorithm(BaseAlgorithm):
    """An algorithm alternating between 100 and 200 Hz frame by frame."""

    config_class = AlternatingConfig

    def load(self) -> None:
        """Load nothing."""

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Return an alternating F0 on the 5-ms grid."""
        n = max(1, int(audio.duration * 200))
        f0 = np.where(np.arange(n) % 2 == 0, 100.0, 200.0)
        return NativeContour(times=np.arange(n) * 0.005, f0=f0)


@dataclass(frozen=True)
class RangeSensitiveConfig(AlgorithmConfig):
    """Configuration of the range-sensitive algorithm."""


class RangeSensitiveAlgorithm(BaseAlgorithm):
    """An algorithm whose voicing and F0 depend on the search range.

    With a wide range, all frames are voiced at 400 Hz (an octave error). With a
    narrow range, the F0 is 200 Hz but only the even frames are voiced.

    """

    config_class = RangeSensitiveConfig

    def load(self) -> None:
        """Load nothing."""

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Return the contour depending on the search range."""
        n = max(1, round(audio.duration * 1000 / frame_period))
        if 1000 <= f0_range[1]:
            f0 = np.full(n, 400.0)
        else:
            f0 = np.where(np.arange(n) % 2 == 0, 200.0, 0.0)
        return NativeContour(times=np.arange(n) * frame_period / 1000, f0=f0)


class FirstVoicingAlgorithm(RangeSensitiveAlgorithm):
    """A range-sensitive algorithm recommending the voicing of the first pass."""

    two_pass_policy = TwoPassPolicy(enabled=True, voicing_from="1st")


class NoTwoPassAlgorithm(ConstantAlgorithm):
    """A constant algorithm recommending no two-pass tracking."""

    two_pass_policy = TwoPassPolicy(enabled=False)


class MedianScaleRangeEstimator(BaseRangeEstimator):
    """An estimator giving [lower * median, upper * median] for testing."""

    def __init__(self, lower: float = 0.75, upper: float = 1.5) -> None:
        """Initialize the estimator."""
        super().__init__()
        self.lower = lower
        self.upper = upper

    def estimate(self, values: np.ndarray) -> tuple[float, float]:
        """Scale the median of the voiced F0."""
        median = float(np.median(values))
        return self.lower * median, self.upper * median


class ThreadsAlgorithm(ConstantAlgorithm):
    """An algorithm returning 100 times the number of threads as F0.

    The F0 is 0, i.e., unvoiced, if the number of threads is not given.

    """

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Return the number of threads as F0."""
        config = cast(ConstantConfig, self.config)
        n = max(1, int(audio.duration * 100))
        f0 = np.full(n, 100.0 * (config.num_threads or 0))
        return NativeContour(times=np.arange(n) * 0.01, f0=f0)


class SampleRateAlgorithm(ConstantAlgorithm):
    """A constant algorithm accepting only 8-16 kHz and recording the sample rate."""

    received_sample_rates: ClassVar[list[int]] = []

    @classmethod
    def get_supported_sample_rates(cls) -> tuple[int, int] | None:
        """Accept only 8-16 kHz."""
        return 8000, 16000

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Record the sample rate of the given audio."""
        type(self).received_sample_rates.append(audio.sample_rate)
        return super().estimate(audio, f0_range, frame_period)


class NarrowRangeFailingAlgorithm(ConstantAlgorithm):
    """A constant algorithm failing with a search range narrower than 1 kHz."""

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Fail with a narrow search range."""
        if f0_range[1] < 1000:
            raise RuntimeError("Failed with a narrow range.")
        return super().estimate(audio, f0_range, frame_period)


class FailingAlgorithm(ConstantAlgorithm):
    """An algorithm always failing."""

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Raise an error regardless of the input."""
        raise RuntimeError("Always failed.")
