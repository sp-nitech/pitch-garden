"""A module for the YAAPT algorithm."""

import contextlib
import io
import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..interfaces.types import Audio
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy

# The frame length of YAAPT in milliseconds, which is the default of AMFM_decompy.
_FRAME_LENGTH = 35.0

# The lower limit of F0 in Hz, below which the search range is not tested.
_MIN_F0 = 10.0


def _import_yaapt() -> tuple[Any, Any]:
    """Import the YAAPT modules of AMFM_decompy lazily.

    Returns
    -------
    basic : module
        The amfm_decompy.basic_tools module.

    pyaapt : module
        The amfm_decompy.pYAAPT module.

    Raises
    ------
    ImportError
        If the amfm-decompy package is not installed.

    """
    try:
        import amfm_decompy.basic_tools as basic
        import amfm_decompy.pYAAPT as pyaapt
    except ImportError as e:
        raise ImportError(
            "YAAPT requires the amfm-decompy package. "
            'Install it via `pip install "pitch-garden[lite]"`.'
        ) from e
    return basic, pyaapt


@dataclass(frozen=True)
class YaaptConfig(AlgorithmConfig):
    """Configuration of YAAPT.

    The default search range covers both speech and singing voice.

    """

    fmin: float | None = 40.0
    fmax: float | None = 800.0

    # The threshold of the normalized low-frequency energy ratio (NLFER), i.e.,
    # nlfer_thresh1 of AMFM_decompy, above which the frames are regarded as voiced
    # candidates. The default value is that of AMFM_decompy.
    voicing_threshold: float = 0.75


class YaaptAlgorithm(BaseAlgorithm):
    """YAAPT: yet another algorithm for pitch tracking (Zahorian and Hu, 2008).

    This uses AMFM_decompy, a Python port of YAAPT. See
    https://github.com/bjbschmitt/AMFM_decompy for details. The voicing decision is
    made inside the algorithm by dynamic programming, and thus the voicing score is
    not provided. The warnings and the messages of AMFM_decompy are suppressed.

    """

    config_class = YaaptConfig
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=True, voicing_from="1st")

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize YAAPT.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of YAAPT.

        """
        super().__init__(config)
        self.basic: Any = None
        self.pyaapt: Any = None

    @classmethod
    def get_f0_limits(cls) -> tuple[float, float]:
        """Get the F0 range that YAAPT accepts.

        The upper limit also has to be lower than or equal to the Nyquist frequency.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper limits in Hz.

        """
        return _MIN_F0, float("inf")

    def load(self) -> None:
        """Import AMFM_decompy.

        Raises
        ------
        ImportError
            If the amfm-decompy package is not installed.

        """
        if self.pyaapt is None:
            self.basic, self.pyaapt = _import_yaapt()

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
            The estimated contour, whose frames are centered at the positions given
            by AMFM_decompy.

        """
        self.load()
        config = cast(YaaptConfig, self.config)
        data = np.asarray(audio.data, dtype=np.float64)
        # AMFM_decompy fails on audio giving only a few frames, so short audio is
        # padded with zeros. The frames beyond the audio are ignored by the frame
        # grid of the tracker.
        min_duration = _FRAME_LENGTH + 4 * frame_period + 10.0
        min_length = int(np.ceil(audio.sample_rate * min_duration / 1000))
        data = np.pad(data, (0, max(0, min_length - len(data))))
        signal = self.basic.SignalObj(data, audio.sample_rate)
        # Redirecting the standard output is not thread-safe, but the trackers are
        # run in separate processes.
        with (
            warnings.catch_warnings(),
            np.errstate(all="ignore"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            result = self.pyaapt.yaapt(
                signal,
                frame_length=_FRAME_LENGTH,
                tda_frame_length=_FRAME_LENGTH,
                frame_space=frame_period,
                f0_min=f0_range[0],
                f0_max=f0_range[1],
                nlfer_thresh1=config.voicing_threshold,
            )
        times = np.asarray(result.frames_pos, dtype=np.float64) / audio.sample_rate
        f0 = np.asarray(result.samp_values, dtype=np.float64)
        return NativeContour(times=times, f0=f0)
