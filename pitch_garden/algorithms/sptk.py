"""A module for the pitch extraction algorithms of SPTK."""

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np

from ..interfaces.types import Audio
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy

# The scale of 16-bit integers, which is expected by the pitch extraction of SPTK.
_INT16_SCALE = 32768.0

# The smallest lower bound greater than 10 Hz, which SPTK excludes.
_MIN_F0 = float(np.nextafter(10.0, np.inf))


def _import_sptk() -> Any:
    """Import the binding of SPTK lazily.

    Returns
    -------
    out : module
        The pitch_garden_sptk module.

    Raises
    ------
    ImportError
        If the binding is not installed.

    """
    try:
        import pitch_garden_sptk
    except ImportError as e:
        raise ImportError(
            "The algorithms of SPTK require the pitch-garden-sptk package. "
            'Install it via `pip install "pitch-garden[lite]"`.'
        ) from e
    return pitch_garden_sptk


@dataclass(frozen=True)
class SptkConfig(AlgorithmConfig):
    """The base configuration of the algorithms of SPTK.

    The default search range covers both speech and singing voice.

    """

    fmin: float | None = 40.0
    fmax: float | None = 800.0
    voicing_threshold: float = 0.0


@dataclass(frozen=True)
class RaptConfig(SptkConfig):
    """Configuration of RAPT, where a higher threshold gives more voiced frames.

    The recommended range of the threshold is [-0.6, 0.7] as in SPTK, but values
    outside are also accepted.

    """

    voicing_threshold: float = 0.0


@dataclass(frozen=True)
class SwipeConfig(SptkConfig):
    """Configuration of SWIPE', where a lower threshold gives more voiced frames.

    The recommended range of the threshold is [0.2, 0.5] as in SPTK, but values
    outside are also accepted.

    """

    voicing_threshold: float = 0.3


@dataclass(frozen=True)
class ReaperConfig(SptkConfig):
    """Configuration of REAPER, where a higher threshold gives more voiced frames.

    The recommended range of the threshold is [-0.5, 1.6] as in SPTK, but values
    outside are also accepted.

    """

    voicing_threshold: float = 0.9


@dataclass(frozen=True)
class DioConfig(SptkConfig):
    """Configuration of DIO, where a higher threshold gives more voiced frames.

    The recommended range of the threshold is [0.02, 0.2] as in SPTK, but values
    outside are also accepted.

    """

    voicing_threshold: float = 0.1


@dataclass(frozen=True)
class HarvestConfig(SptkConfig):
    """Configuration of Harvest, where a higher threshold gives more voiced frames.

    The recommended range of the threshold is [0.0, 0.2] as in SPTK, but values
    outside are also accepted.

    """

    voicing_threshold: float = 0.01


class SptkAlgorithm(BaseAlgorithm):
    """The base class of the pitch extraction algorithms of SPTK.

    The results are identical to the pitch command of SPTK for 16-bit audio unless
    the frame period is out of [`min_frame_period`, `max_frame_period`] or exceeds a
    quarter of the duration. Since the frame shift of SPTK is an integer number of
    samples, a frame period that is not, e.g., 5 ms at 44.1 kHz, is rounded to the
    nearest integer number of samples, and the result is interpolated onto the frame
    grid. The audio whose sample rate is out of those supported by the binding of
    SPTK is resampled to 16 kHz, where the results are not identical to SPTK. The
    voicing decision is made inside the algorithms, and thus the voicing score is
    not provided.

    """

    config_class = SptkConfig

    # The name of the algorithm in the binding.
    sptk_name: ClassVar[str] = ""

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize the algorithm.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of the algorithm.

        """
        super().__init__(config)
        self.sptk: Any = None

    @classmethod
    def get_supported_sample_rates(cls) -> tuple[int, int] | None:
        """Get the sample rates supported by the binding of SPTK.

        Returns
        -------
        out : tuple[int, int] | None
            The lower and upper limits in Hz.

        """
        sptk = _import_sptk()
        return sptk.MIN_SAMPLE_RATE, sptk.MAX_SAMPLE_RATE

    @classmethod
    def get_f0_limits(cls) -> tuple[float, float]:
        """Get the F0 range that SPTK accepts.

        SPTK requires the lower bound to be greater than 10 Hz, which is a sanity
        check of the arguments rather than a limitation of the algorithms. The upper
        limit also has to be lower than the Nyquist frequency.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper limits in Hz.

        """
        return _MIN_F0, 49000.0

    def load(self) -> None:
        """Import the binding of SPTK.

        Raises
        ------
        ImportError
            If the binding is not installed.

        """
        if self.sptk is None:
            self.sptk = _import_sptk()

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
            The frame period in milliseconds, which is used as the frame shift
            rounded to an integer number of samples.

        Returns
        -------
        out : NativeContour
            The estimated contour.

        Raises
        ------
        ValueError
            If the search range is not lower than the Nyquist frequency.

        RuntimeError
            If SPTK fails to extract the F0, e.g., for too short audio.

        """
        self.load()
        config = cast(SptkConfig, self.config)
        waveform, sample_rate = audio.data, audio.sample_rate
        fmin, fmax = f0_range
        if sample_rate / 2 <= fmax:
            raise ValueError(
                f"The upper bound of the search range ({fmax} Hz) must be lower "
                f"than the Nyquist frequency ({sample_rate / 2} Hz)."
            )
        frame_shift = max(1, round(sample_rate * frame_period / 1000))
        try:
            f0 = self.sptk.extract_f0(
                np.ascontiguousarray(waveform * _INT16_SCALE),
                float(sample_rate),
                frame_shift,
                fmin,
                fmax,
                config.voicing_threshold,
                self.sptk_name,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"{e} The arguments of {self.sptk_name} were: {len(waveform)} samples "
                f"at {sample_rate} Hz, frame shift of {frame_shift} samples, search "
                f"range of {fmin:g}-{fmax:g} Hz, and voicing threshold of "
                f"{config.voicing_threshold:g}."
            ) from e
        times = np.arange(len(f0)) * frame_shift / sample_rate
        return NativeContour(times=times, f0=np.asarray(f0, dtype=np.float64))


class RaptAlgorithm(SptkAlgorithm):
    """RAPT: a robust algorithm for pitch tracking (Talkin, 1995)."""

    config_class = RaptConfig
    sptk_name = "rapt"
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    # Narrowing the search range increases the unvoiced frames, which is avoided by
    # the voicing of the first pass.
    two_pass_policy = TwoPassPolicy(enabled=True, voicing_from="1st")


class SwipeAlgorithm(SptkAlgorithm):
    """SWIPE': a sawtooth waveform inspired pitch estimator (Camacho, 2007)."""

    config_class = SwipeConfig
    sptk_name = "swipe"
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=False)


class ReaperAlgorithm(SptkAlgorithm):
    """REAPER: robust epoch and pitch estimator (Talkin, 2015)."""

    config_class = ReaperConfig
    sptk_name = "reaper"
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=True, voicing_from="1st")


class DioAlgorithm(SptkAlgorithm):
    """DIO: a fast F0 estimator of WORLD (Morise et al., 2009)."""

    config_class = DioConfig
    sptk_name = "dio"
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=True, voicing_from="2nd")


class HarvestAlgorithm(SptkAlgorithm):
    """Harvest: a high-performance F0 estimator of WORLD (Morise, 2017)."""

    config_class = HarvestConfig
    sptk_name = "harvest"
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=True, voicing_from="2nd")
