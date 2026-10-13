"""A module for tracking pitch with a specified algorithm."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import numpy as np
import soxr

from ..algorithms import ALGORITHM_MAP
from ..algorithms.base import AlgorithmConfig, BaseAlgorithm, TwoPassPolicy
from ..utils.grid import check_frame_period, get_num_frames, resample
from ._utils import create_audio_object
from .types import Audio, NativeContour, Pitch

# The minimum number of native frames in the audio. Some algorithms fail if the
# frame period is long relative to the duration, e.g., RAPT, REAPER, and STRAIGHT.
_MIN_NATIVE_FRAMES = 4


def _clip_range(
    algo_cls: type[BaseAlgorithm], fmin: float | None, fmax: float | None
) -> tuple[float, float]:
    """Clip a search range by the limits of an algorithm.

    Parameters
    ----------
    algo_cls : type[BaseAlgorithm]
        The algorithm class.

    fmin : float | None
        The lower bound in Hz, or None for the lower limit of the algorithm.

    fmax : float | None
        The upper bound in Hz, or None for the upper limit of the algorithm.

    Returns
    -------
    out : tuple[float, float]
        The clipped range in Hz.

    Raises
    ------
    ValueError
        If the clipped range is empty.

    """
    lower, upper = algo_cls.get_f0_limits()
    low = lower if fmin is None else max(lower, fmin)
    high = upper if fmax is None else min(upper, fmax)
    if high <= low:
        raise ValueError(
            f"Invalid F0 search range: fmin={low}, fmax={high}. "
            f"The limits of {algo_cls.__name__} are {(lower, upper)}."
        )
    return low, high


class Tracker:
    """A class for tracking pitch with an algorithm and its own parameters."""

    def __init__(
        self,
        algorithm: str | type[BaseAlgorithm],
        *,
        name: str | None = None,
        two_pass_policy: TwoPassPolicy | None = None,
        **params: Any,
    ) -> None:
        """Initialize the tracker.

        Parameters
        ----------
        algorithm : str | type[BaseAlgorithm]
            The name of a registered algorithm or an algorithm class.

        name : str | None, optional
            The name of the tracker used to identify its output. If None, the name of
            the algorithm is used. Different names allow using the same algorithm
            with different parameters.

        two_pass_policy : TwoPassPolicy | None, optional
            The settings of the two-pass tracking in the pipeline, i.e., whether the
            tracker is rerun and which pass gives the voicing decisions. If None,
            the recommended settings of the algorithm are used.

        **params : Any
            The algorithm-specific parameters defined by its configuration class,
            including the device, e.g., cuda, for algorithms supporting GPUs.

        Raises
        ------
        ValueError
            If the algorithm or the parameters are invalid.

        """
        if isinstance(algorithm, str):
            try:
                cls = ALGORITHM_MAP[algorithm]
            except KeyError as e:
                raise ValueError(
                    f"Unsupported algorithm '{algorithm}'. "
                    f"Supported algorithms are: {list(ALGORITHM_MAP.keys())}"
                ) from e
            default_name = algorithm
        else:
            cls = algorithm
            default_name = cls.__name__

        self.algorithm_class = cls
        self.name = default_name if name is None else name
        self.config = self._create_config(cls, params)
        self._two_pass_policy = two_pass_policy
        self._algorithm: BaseAlgorithm | None = None

    @staticmethod
    def _create_config(
        algo_cls: type[BaseAlgorithm], params: dict[str, Any]
    ) -> AlgorithmConfig:
        """Create and validate the configuration of the algorithm.

        Parameters
        ----------
        algo_cls : type[BaseAlgorithm]
            The algorithm class.

        params : dict[str, Any]
            The algorithm-specific parameters.

        Returns
        -------
        out : AlgorithmConfig
            The validated configuration.

        Raises
        ------
        ValueError
            If the parameters are invalid.

        """
        valid_keys = [f.name for f in fields(algo_cls.config_class)]
        unknown_keys = [k for k in params if k not in valid_keys]
        if unknown_keys:
            raise ValueError(
                f"Unknown parameters {unknown_keys} for {algo_cls.__name__}. "
                f"Valid parameters are: {valid_keys}"
            )
        config = algo_cls.config_class(**params)
        config = replace(config, device=algo_cls.resolve_device(config.device))
        if not algo_cls.supports_gpu and config.device != "cpu":
            raise ValueError(
                f"{algo_cls.__name__} runs only on CPU, got device '{config.device}'."
            )

        if not algo_cls.supports_f0_range and (
            config.fmin is not None or config.fmax is not None
        ):
            raise ValueError(
                f"{algo_cls.__name__} does not support the F0 search range."
            )
        _clip_range(algo_cls, config.fmin, config.fmax)
        return config

    def __getstate__(self) -> dict[str, Any]:
        """Get the state for pickling without the loaded algorithm."""
        state = self.__dict__.copy()
        state["_algorithm"] = None
        return state

    def __repr__(self) -> str:
        """Get the string representation."""
        return f"Tracker(name={self.name!r}, config={self.config!r})"

    @property
    def supports_f0_range(self) -> bool:
        """Get whether the F0 search range can be specified.

        Returns
        -------
        out : bool
            True if the F0 search range can be specified.

        """
        return self.algorithm_class.supports_f0_range

    @property
    def two_pass_policy(self) -> TwoPassPolicy:
        """Get the settings of the two-pass tracking.

        Returns
        -------
        out : TwoPassPolicy
            The given settings, or the recommended settings of the algorithm.

        """
        if self._two_pass_policy is None:
            return self.algorithm_class.two_pass_policy
        return self._two_pass_policy

    @property
    def parallel_safe(self) -> bool:
        """Get whether the tracker can be run in a worker process.

        Returns
        -------
        out : bool
            True if the tracker can be run in a worker process.

        """
        return self.algorithm_class.is_parallel_safe(self.config)

    @property
    def f0_range(self) -> tuple[float, float]:
        """Get the effective F0 search range.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper bounds of the F0 search range in Hz.

        """
        return _clip_range(self.algorithm_class, self.config.fmin, self.config.fmax)

    def _track(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Run the algorithm after resampling the audio if it is unsupported.

        The frame period given to the algorithm is limited by the algorithm and by
        the duration, where the lower limit of the algorithm takes precedence. The
        result is resampled onto the frame grid anyway.

        """
        cls = self.algorithm_class
        upper = min(cls.max_frame_period, audio.duration * 1000 / _MIN_NATIVE_FRAMES)
        frame_period = max(cls.min_frame_period, min(frame_period, upper))
        limits = self.algorithm_class.get_supported_sample_rates()
        if limits is not None and not limits[0] <= audio.sample_rate <= limits[1]:
            target = self.algorithm_class.preferred_sample_rate
            data = soxr.resample(audio.data, audio.sample_rate, target)
            audio = Audio(data, target)
        return self.load().estimate(audio, f0_range, frame_period)

    def _resolve_call_range(
        self, fmin: float | None, fmax: float | None
    ) -> tuple[float, float]:
        """Get the search range used at a call."""
        if fmin is None and fmax is None:
            return self.f0_range
        if not self.supports_f0_range:
            raise ValueError(
                f"{self.algorithm_class.__name__} does not support the F0 search range."
            )
        return _clip_range(
            self.algorithm_class,
            self.config.fmin if fmin is None else fmin,
            self.config.fmax if fmax is None else fmax,
        )

    def replace(self, **params: Any) -> Tracker:
        """Create a new tracker with some parameters replaced.

        Parameters
        ----------
        **params : Any
            The parameters to be replaced, including `two_pass_policy`.

        Returns
        -------
        out : Tracker
            The new tracker with the same name.

        """
        current: dict[str, Any] = {
            f.name: getattr(self.config, f.name) for f in fields(self.config)
        }
        current["two_pass_policy"] = self._two_pass_policy
        current.update(params)
        return Tracker(self.algorithm_class, name=self.name, **current)

    def _for_parallel(self) -> Tracker:
        """Create a new tracker adjusted for running in a worker process.

        Returns
        -------
        out : Tracker
            The adjusted tracker.

        """
        new_tracker = Tracker.__new__(Tracker)
        new_tracker.__dict__.update(self.__getstate__())
        new_tracker.config = self.algorithm_class.parallel_config(self.config)
        return new_tracker

    def load(self) -> BaseAlgorithm:
        """Load the algorithm if it is not loaded yet.

        Returns
        -------
        out : BaseAlgorithm
            The loaded algorithm.

        """
        if self._algorithm is None:
            self._algorithm = self.algorithm_class(self.config)
        self._algorithm.load()
        return self._algorithm

    def __call__(
        self,
        source: np.ndarray | Audio,
        sample_rate: int | None = None,
        *,
        frame_period: float = 5.0,
        fmin: float | None = None,
        fmax: float | None = None,
    ) -> Pitch:
        """Track the pitch of the input audio.

        Parameters
        ----------
        source : np.ndarray | Audio
            The input waveform with shape (T,) or an Audio object.

        sample_rate : int | None, optional
            The sample rate of the input waveform.

        frame_period : float, optional
            The frame period of the output in milliseconds.

        fmin : float | None, optional
            The lower bound of the F0 search range in Hz overriding the one given at
            the creation only for this call. If None, the one given at the creation
            is used. The range is clipped to the limits of the algorithm, and
            changing it does not reload the algorithm.

        fmax : float | None, optional
            The upper bound of the F0 search range in Hz, which is given in the same
            way as `fmin`.

        Returns
        -------
        out : Pitch
            The estimated F0 contour.

        Raises
        ------
        ValueError
            If the frame period is invalid, or the search range is invalid or not
            supported by the algorithm.

        Examples
        --------
        >>> import numpy as np
        >>> import pitch_garden as pg
        >>> sr = 16000
        >>> x = np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
        >>> tracker = pg.Tracker("swift-f0", voicing_threshold=0.5)
        >>> pitch = tracker(x, sr, frame_period=5.0)
        >>> pitch.length
        200

        """
        check_frame_period(frame_period)
        audio = create_audio_object(source, sample_rate)
        call_range = self._resolve_call_range(fmin, fmax)
        if audio.length == 0:
            # Some algorithms fail on empty audio, which has no frames anyway.
            native = NativeContour(times=np.zeros(0), f0=np.zeros(0))
        else:
            native = self._track(audio, call_range, frame_period)
        grid = resample(
            native.times,
            native.f0,
            num_frames=get_num_frames(audio.length, audio.sample_rate, frame_period),
            frame_period=frame_period,
            score=native.score,
        )
        return Pitch(
            f0=grid.f0,
            frame_period=frame_period,
            sample_rate=audio.sample_rate,
            source=self.name,
            score=grid.score,
            native=native,
        )
