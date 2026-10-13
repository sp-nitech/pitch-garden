"""A module for the pipeline of tracking, voting, and two-pass tracking."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np

from ..algorithms.base import VoicingSource
from ..ranges._utils import clip_to_range, estimate_f0_range
from ..ranges.base import BaseRangeEstimator, F0Range
from ..ranges.log_normal import LogNormalRangeEstimator
from ..utils.grid import check_frame_period
from ..voters import create_voter
from ..voters.base import BaseVoter
from ._utils import create_audio_object
from .tracker import Tracker
from .types import Audio, Pitch

logger = logging.getLogger("pitch_garden")

OnError = Literal["skip", "raise"]

# The trackers loaded in a worker process, keyed by their identities.
_WORKER_CACHE: dict[str, Tracker] = {}


def _tracker_key(tracker: Tracker) -> str:
    """Get the identity of a tracker."""
    cls = tracker.algorithm_class
    return f"{cls.__module__}.{cls.__qualname__}|{tracker.name}|{tracker.config!r}"


def _run_in_worker(
    tracker: Tracker,
    audio: Audio,
    frame_period: float,
    f0_range: tuple[float, float] | None,
) -> Pitch:
    """Run a tracker in a worker process while reusing the loaded algorithm.

    The search range is given at the call, so that the cache does not grow with the
    estimated ranges.

    """
    key = _tracker_key(tracker)
    cached = _WORKER_CACHE.get(key)
    if cached is None:
        cached = _WORKER_CACHE[key] = tracker
    return cached(audio, frame_period=frame_period, f0_range=f0_range)


@dataclass
class TwoPass:
    """Settings of the two-pass tracking with an estimated F0 search range.

    In the first pass, the trackers supporting the search range are run with the
    wide range [fmin, fmax]. The range is then estimated from the distribution of
    voiced F0 of `range_from`. In the second pass, the trackers supporting the
    search range are rerun with the estimated range if their `two_pass_policy`
    enables it, and the outputs of the other trackers are reused. Whether each
    tracker is rerun and which pass gives its voicing decisions are given by
    `Tracker.two_pass_policy`. The out-of-range frames of the trackers not
    supporting the search range, i.e., `supports_f0_range=False`, are unvoiced if
    `clip_out_of_range`, which does not affect the built-in algorithms since all of
    them support the range.

    """

    estimator: BaseRangeEstimator = field(default_factory=LogNormalRangeEstimator)
    range_from: str | None = None  # The tracker giving the range (None: the final).
    fmin: float = 40.0  # The lower bound of the first-pass range in Hz.
    fmax: float = 2000.0  # The upper bound of the first-pass range in Hz.
    clip_out_of_range: bool = True  # Only for `supports_f0_range=False`.

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises
        ------
        ValueError
            If the settings are invalid.

        """
        if not 0 < self.fmin < self.fmax:
            raise ValueError("0 < fmin < fmax must be satisfied.")

    def check_trackers(self, names: Sequence[str], has_voter: bool) -> None:
        """Check that the settings can be used with the trackers.

        Parameters
        ----------
        names : Sequence[str]
            The names of the trackers.

        has_voter : bool
            Whether a voter is used.

        Raises
        ------
        ValueError
            If the contour giving the range is unavailable.

        """
        if self.range_from is None:
            if not has_voter and 1 < len(names):
                raise ValueError(
                    "A voter or range_from is required to estimate the F0 range from "
                    "multiple trackers."
                )
        elif self.range_from not in names:
            raise ValueError(
                f"Unknown tracker '{self.range_from}' in range_from. "
                f"Choose one of {list(names)}."
            )


@dataclass
class PipelineResult:
    """Dataclass to hold the result of the pipeline."""

    # The final contour: the output of the tracker if only one tracker is used, or
    # the voted contour otherwise, which is None if no voter is used.
    pitch: Pitch | None
    each: dict[str, Pitch]  # The outputs of the trackers keyed by their names.
    f0_range: F0Range | None = None  # The estimated range of the two-pass tracking.
    pass1: PipelineResult | None = None  # The result of the first pass.
    # The error messages of the failed trackers keyed by their names. The trackers
    # failed in the first pass are excluded from each, and the ones failed in the
    # second pass keep the outputs of the first pass.
    errors: dict[str, str] = field(default_factory=dict)


class Pipeline:
    """A class for the pipeline of tracking, voting, and two-pass tracking.

    The pipeline runs one or more trackers, votes their outputs if a voter is given,
    and reruns them with an estimated F0 search range if the two-pass tracking is
    enabled. The trackers that can be run in worker processes (CPU-only ones) are
    run in parallel, and the others are run sequentially in the main process.

    """

    def __init__(
        self,
        trackers: Sequence[Tracker],
        voter: BaseVoter | str | None = "median",
        *,
        num_workers: int | None = None,
        two_pass: TwoPass | None = None,
        on_error: OnError = "skip",
        mp_context: str = "spawn",
    ) -> None:
        """Initialize the pipeline.

        Parameters
        ----------
        trackers : Sequence[Tracker]
            The trackers, each of which holds its own parameters.

        voter : BaseVoter | str | None, optional
            The voter instance, the registered name of a voter with its default
            parameters, or None to skip voting.

        num_workers : int | None, optional
            The number of worker processes. If None, it is the smaller of the number
            of parallelizable trackers and the number of CPUs. If 1, all trackers are
            run sequentially in the main process.

        two_pass : TwoPass | None, optional
            The settings of the two-pass tracking. If None, the trackers are run
            once with their own search ranges.

        on_error : OnError, optional
            The behavior on a failure of a tracker. If 'skip', the failed tracker is
            excluded from voting with a warning, or keeps the output of the first
            pass if it fails in the second pass. If 'raise', the error is raised.
            Note that the error is always raised if all trackers fail.

        mp_context : str, optional
            The start method of the worker processes.

        Raises
        ------
        ValueError
            If the trackers are invalid.

        """
        if len(trackers) == 0:
            raise ValueError("At least one tracker is required.")
        names = [t.name for t in trackers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate tracker names {duplicates}. "
                "Give different names to the trackers using the same algorithm."
            )
        if num_workers is not None and num_workers < 1:
            raise ValueError("num_workers must be a positive integer.")
        if on_error not in get_args(OnError):
            raise ValueError(
                f"Invalid on_error '{on_error}'. "
                f"Valid options are: {list(get_args(OnError))}"
            )

        self.trackers = list(trackers)
        self.voter = create_voter(voter) if isinstance(voter, str) else voter
        if self.voter is not None:
            self.voter.check_trackers(names)
        if two_pass is not None:
            two_pass.check_trackers(names, has_voter=self.voter is not None)
        self.two_pass = two_pass
        self.num_workers = num_workers
        self.on_error = on_error
        self.mp_context = mp_context
        self._executor: ProcessPoolExecutor | None = None
        self._warned = False

    def __enter__(self) -> Pipeline:
        """Enter the context."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context and shut down the worker processes."""
        self.close()

    def close(self) -> None:
        """Shut down the worker processes."""
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def _get_executor(self, num_parallel: int) -> ProcessPoolExecutor | None:
        """Get the process pool, creating it if needed."""
        num_workers = self.num_workers
        if num_workers is None:
            num_workers = min(num_parallel, os.cpu_count() or 1)
        if num_workers <= 1 or num_parallel <= 1:
            return None
        if not self._can_start_workers():
            if not self._warned:
                logger.warning(
                    "The worker processes cannot be started because the main script "
                    "is not a file, e.g., given via stdin. The trackers are run "
                    "sequentially."
                )
                self._warned = True
            return None
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=num_workers, mp_context=mp.get_context(self.mp_context)
            )
        return self._executor

    def _can_start_workers(self) -> bool:
        """Check whether the worker processes can be started.

        The 'spawn' and 'forkserver' start methods re-import the main script in the
        worker processes, which fails and hangs if the script is not a file.

        """
        if self.mp_context == "fork":
            return True
        path = getattr(sys.modules.get("__main__"), "__file__", None)
        return path is None or os.path.isfile(path)

    def track(
        self,
        audio: Audio,
        *,
        frame_period: float = 5.0,
        trackers: Sequence[Tracker] | None = None,
        f0_ranges: Mapping[str, tuple[float, float]] | None = None,
    ) -> dict[str, Pitch]:
        """Run the trackers.

        Parameters
        ----------
        audio : Audio
            The input audio.

        frame_period : float, optional
            The frame period of the outputs in milliseconds.

        trackers : Sequence[Tracker] | None, optional
            The trackers to be run. If None, all trackers of the pipeline are run.

        f0_ranges : Mapping[str, tuple[float, float]] | None, optional
            The F0 search ranges in Hz keyed by the tracker names. The trackers not
            included or not supporting the search range use their own ranges.

        Returns
        -------
        out : dict[str, Pitch]
            The outputs of the trackers keyed by their names, in the given order.

        Raises
        ------
        Exception
            The error of the first failed tracker, regardless of on_error.

        """
        outputs, errors = self._track_each(
            audio, frame_period=frame_period, trackers=trackers, f0_ranges=f0_ranges
        )
        if errors:
            raise next(iter(errors.values()))
        return outputs

    def _track_each(
        self,
        audio: Audio,
        *,
        frame_period: float,
        trackers: Sequence[Tracker] | None,
        f0_ranges: Mapping[str, tuple[float, float]] | None,
    ) -> tuple[dict[str, Pitch], dict[str, Exception]]:
        """Run the trackers and collect their outputs and errors separately."""
        # An invalid frame period is an error of the input, not of the trackers.
        check_frame_period(frame_period)
        trackers = self.trackers if trackers is None else list(trackers)
        parallel = [t for t in trackers if t.parallel_safe]
        executor = self._get_executor(len(parallel))

        f0_ranges = {} if f0_ranges is None else f0_ranges
        ranges = {t.name: self._range_for(t, f0_ranges.get(t.name)) for t in trackers}
        futures: dict[str, Future[Pitch]] = {}
        if executor is not None:
            for t in parallel:
                args = (t._for_parallel(), audio, frame_period, ranges[t.name])
                futures[t.name] = executor.submit(_run_in_worker, *args)

        outputs: dict[str, Pitch] = {}
        errors: dict[str, Exception] = {}
        for t in trackers:
            if t.name not in futures:
                try:
                    outputs[t.name] = t(
                        audio, frame_period=frame_period, f0_range=ranges[t.name]
                    )
                except Exception as e:
                    errors[t.name] = e
        for name, future in futures.items():
            try:
                outputs[name] = future.result()
            except Exception as e:
                errors[name] = e
        order = [t.name for t in trackers]
        return (
            {n: outputs[n] for n in order if n in outputs},
            {n: errors[n] for n in order if n in errors},
        )

    def _handle_errors(self, errors: dict[str, Exception], stage: str) -> None:
        """Raise the first error or warn about all errors depending on on_error."""
        if not errors:
            return
        if self.on_error == "raise":
            raise next(iter(errors.values()))
        for name, error in errors.items():
            logger.warning(f"Tracker '{name}' failed in the {stage}: {error}")

    def _finalize(self, each: dict[str, Pitch]) -> Pitch | None:
        """Get the final contour from the outputs of the trackers.

        The output of a single tracker is used as it is without voting.

        Parameters
        ----------
        each : dict[str, Pitch]
            The outputs of the trackers keyed by their names.

        Returns
        -------
        out : Pitch | None
            The final contour, which is None if multiple trackers are used without
            a voter.

        """
        if len(each) == 1:
            return next(iter(each.values()))
        return None if self.voter is None else self.voter(each)

    def __call__(
        self,
        source: np.ndarray | Audio,
        sample_rate: int | None = None,
        *,
        frame_period: float = 5.0,
    ) -> PipelineResult:
        """Run the trackers and vote their outputs.

        Parameters
        ----------
        source : np.ndarray | Audio
            The input waveform with shape (T,) or an Audio object.

        sample_rate : int | None, optional
            The sample rate of the input waveform.

        frame_period : float, optional
            The frame period of the outputs in milliseconds.

        Returns
        -------
        out : PipelineResult
            The final contour and the outputs of the trackers.

        """
        audio = create_audio_object(source, sample_rate)
        first = self.first_pass(audio, frame_period=frame_period)
        if self.two_pass is None:
            return first
        f0_range = self.estimate_f0_range([first])
        return self.second_pass(audio, first, f0_range, frame_period=frame_period)

    def first_pass(self, audio: Audio, *, frame_period: float = 5.0) -> PipelineResult:
        """Run the trackers with the first-pass search range and vote their outputs.

        If the two-pass tracking is disabled, the trackers are run with their own
        search ranges.

        Parameters
        ----------
        audio : Audio
            The input audio.

        frame_period : float, optional
            The frame period of the outputs in milliseconds.

        Returns
        -------
        out : PipelineResult
            The result of the first pass.

        """
        f0_ranges = None
        if self.two_pass is not None:
            # All trackers use the wide range regardless of whether they are rerun.
            wide = (self.two_pass.fmin, self.two_pass.fmax)
            f0_ranges = {t.name: wide for t in self.trackers}
        each, errors = self._track_each(
            audio, frame_period=frame_period, trackers=None, f0_ranges=f0_ranges
        )
        stage = "tracking" if self.two_pass is None else "first pass"
        self._handle_errors(errors, stage)
        if not each:
            raise RuntimeError(
                f"All trackers failed: {_format_errors(errors)}"
            ) from next(iter(errors.values()))
        return PipelineResult(
            pitch=self._finalize(each), each=each, errors=_format_errors(errors)
        )

    def estimate_f0_range(self, results: Sequence[PipelineResult]) -> F0Range:
        """Estimate the F0 search range from the results of the first pass.

        Parameters
        ----------
        results : Sequence[PipelineResult]
            The results of the first pass, e.g., of all utterances of a speaker.

        Returns
        -------
        out : F0Range
            The estimated F0 search range.

        Raises
        ------
        RuntimeError
            If the two-pass tracking is disabled.

        ValueError
            If the contour giving the range is not found in the results.

        """
        if self.two_pass is None:
            raise RuntimeError("The two-pass tracking is disabled.")
        name = self.two_pass.range_from
        pitches = []
        for r in results:
            pitch = r.pitch if name is None else r.each.get(name)
            if pitch is None:
                if name is not None and name in r.errors:
                    # The failed tracker is skipped, which may lead to the fallback.
                    continue
                target = "The final contour" if name is None else f"Tracker '{name}'"
                raise ValueError(f"{target} is not found in the results.")
            pitches.append(pitch)
        return estimate_f0_range(
            pitches,
            self.two_pass.estimator,
            default=(self.two_pass.fmin, self.two_pass.fmax),
        )

    def second_pass(
        self,
        audio: Audio,
        first: PipelineResult,
        f0_range: F0Range,
        *,
        frame_period: float = 5.0,
    ) -> PipelineResult:
        """Rerun the trackers with the estimated search range and vote the outputs.

        Parameters
        ----------
        audio : Audio
            The input audio.

        first : PipelineResult
            The result of the first pass of the same audio.

        f0_range : F0Range
            The estimated F0 search range.

        frame_period : float, optional
            The frame period of the outputs in milliseconds.

        Returns
        -------
        out : PipelineResult
            The result of the second pass with the estimated range and the result of
            the first pass.

        Raises
        ------
        RuntimeError
            If the two-pass tracking is disabled.

        """
        if self.two_pass is None:
            raise RuntimeError("The two-pass tracking is disabled.")
        # The trackers need not be rerun if the first-pass range is kept.
        rerun_each: dict[str, Pitch] = {}
        errors: dict[str, Exception] = {}
        if not f0_range.fallback:
            # The trackers failed in the first pass are not rerun.
            rerun = [t for t in self._rerun_trackers() if t.name in first.each]
            estimated = (f0_range.fmin, f0_range.fmax)
            rerun_each, errors = self._track_each(
                audio,
                frame_period=frame_period,
                trackers=rerun,
                f0_ranges={t.name: estimated for t in rerun},
            )
            self._handle_errors(errors, "second pass")

        each: dict[str, Pitch] = {}
        for t in self.trackers:
            if t.name not in first.each:
                continue
            if t.name in rerun_each:
                second = rerun_each[t.name]
                voicing = t.two_pass_policy.voicing_from
                each[t.name] = self._merge_voicing(first.each[t.name], second, voicing)
            elif t.supports_f0_range:
                each[t.name] = first.each[t.name]
            elif self.two_pass.clip_out_of_range:
                each[t.name] = clip_to_range(first.each[t.name], f0_range)
            else:
                each[t.name] = first.each[t.name]
        return PipelineResult(
            pitch=self._finalize(each),
            each=each,
            f0_range=f0_range,
            pass1=first,
            errors={**first.errors, **_format_errors(errors)},
        )

    def _rerun_trackers(self) -> list[Tracker]:
        """Get the trackers rerun in the second pass."""
        return [
            t
            for t in self.trackers
            if t.supports_f0_range and t.two_pass_policy.enabled
        ]

    @staticmethod
    def _merge_voicing(first: Pitch, second: Pitch, voicing: VoicingSource) -> Pitch:
        """Take the voicing decisions from the first pass if requested."""
        if voicing == "2nd":
            return second
        f0 = np.where(0 < second.f0, second.f0, first.f0)
        return second.with_f0(np.where(0 < first.f0, f0, 0.0))

    @staticmethod
    def _range_for(
        tracker: Tracker, f0_range: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Get the search range given to a tracker at a call.

        None is returned, i.e., the tracker uses its own range, if the tracker does
        not support the search range or the range is out of its limits.

        """
        if f0_range is None or not tracker.supports_f0_range:
            return None
        lower, upper = tracker.algorithm_class.get_f0_limits()
        if min(f0_range[1], upper) <= max(f0_range[0], lower):
            return None
        return f0_range


def _format_errors(errors: Mapping[str, Exception]) -> dict[str, str]:
    """Format the errors of the trackers as messages."""
    return {name: f"{type(e).__name__}: {e}" for name, e in errors.items()}
