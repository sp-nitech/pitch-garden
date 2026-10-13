"""A module defining data types."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, cast, get_args

import numpy as np

from ..utils.grid import check_frame_period, get_frame_times, resample

LOG_ZERO = -1e10  # The magic number for unvoiced frames in the log domain (SPTK).

OutputFormat = Literal["linear", "log", "inverse"]
FileType = Literal["npz", "float", "double"]


@dataclass
class Audio:
    """Dataclass to hold monaural audio data."""

    data: np.ndarray  # The waveform with shape (T,).
    sample_rate: int  # The sample rate in Hz.

    def __post_init__(self) -> None:
        """Validate the input data."""
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 1:
            raise ValueError(f"data must be a 1-D array, got shape {self.data.shape}.")
        if not isinstance(self.sample_rate, (int, np.integer)) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        self.sample_rate = int(self.sample_rate)

    @property
    def length(self) -> int:
        """Get the number of samples.

        Returns
        -------
        out : int
            The number of samples.

        """
        return len(self.data)

    @property
    def duration(self) -> float:
        """Get the duration in seconds.

        Returns
        -------
        out : float
            The duration in seconds.

        """
        return self.length / self.sample_rate


@dataclass
class NativeContour:
    """Dataclass to hold a contour on the native frame grid of an algorithm."""

    times: np.ndarray  # The center times of the frames in seconds with shape (M,).
    f0: np.ndarray  # The F0 in Hz with shape (M,). Unvoiced frames are 0.
    score: np.ndarray | None = None  # The voicing score with shape (M,).
    candidate: np.ndarray | None = None  # The F0 candidate in all frames (M,).

    def __post_init__(self) -> None:
        """Validate the input data."""
        self.times = np.asarray(self.times, dtype=np.float64)
        self.f0 = np.asarray(self.f0, dtype=np.float64)
        for name in ("score", "candidate"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=np.float64))


@dataclass
class Pitch:
    """Dataclass to hold an F0 contour on a uniform frame grid.

    The center of the n-th frame is located at ``n * frame_period`` milliseconds.

    """

    f0: np.ndarray  # The F0 in Hz with shape (N,). Unvoiced frames are 0.
    frame_period: float  # The frame period in milliseconds.
    sample_rate: int  # The sample rate of the source audio in Hz.
    source: str  # The name of the tracker or voter producing the contour.
    score: np.ndarray | None = None  # The voicing score with shape (N,).
    native: NativeContour | None = None  # The contour on the native frame grid.

    def __post_init__(self) -> None:
        """Validate the input data."""
        self.f0 = np.asarray(self.f0, dtype=np.float64)
        if self.f0.ndim != 1:
            raise ValueError(f"f0 must be a 1-D array, got shape {self.f0.shape}.")
        if np.any(self.f0 < 0) or not np.all(np.isfinite(self.f0)):
            raise ValueError("f0 must be finite and non-negative.")
        check_frame_period(self.frame_period)
        if self.score is not None:
            self.score = np.asarray(self.score, dtype=np.float64)
            if self.score.shape != self.f0.shape:
                raise ValueError("score must have the same shape as f0.")

    def __repr__(self) -> str:
        """Get a concise representation without the arrays."""
        return (
            f"Pitch(source={self.source!r}, num_frames={self.length}, "
            f"num_voiced_frames={int(self.voiced.sum())}, "
            f"frame_period={self.frame_period}, sample_rate={self.sample_rate})"
        )

    @property
    def length(self) -> int:
        """Get the number of frames.

        Returns
        -------
        out : int
            The number of frames.

        """
        return len(self.f0)

    @property
    def voiced(self) -> np.ndarray:
        """Get the voiced flags.

        Returns
        -------
        out : np.ndarray
            The boolean array that is True for voiced frames.

        """
        return 0 < self.f0

    def to(self, output_format: OutputFormat | str) -> np.ndarray:
        """Convert the F0 contour to the specified format.

        Parameters
        ----------
        output_format : ['linear', 'log', 'inverse']
            The output format. 'linear' is F0 in Hz (unvoiced: 0), 'log' is log F0
            (unvoiced: -1e10), and 'inverse' is the pitch period in samples, i.e.,
            Fs/F0 (unvoiced: 0), where Fs is the sample rate of the source audio.
            These are compatible with SPTK.

        Returns
        -------
        out : np.ndarray
            The converted contour.

        Raises
        ------
        ValueError
            If the output format is not supported.

        """
        voiced = self.voiced
        if output_format == "linear":
            return self.f0.copy()
        if output_format == "log":
            out = np.full_like(self.f0, LOG_ZERO)
            out[voiced] = np.log(self.f0[voiced])
            return out
        if output_format == "inverse":
            out = np.zeros_like(self.f0)
            out[voiced] = self.sample_rate / self.f0[voiced]
            return out
        raise ValueError(
            f"Unsupported output format '{output_format}'. "
            f"Supported formats are: {list(get_args(OutputFormat))}"
        )

    def _can_rethreshold(self) -> bool:
        """Check whether the voicing score is available for re-thresholding."""
        native = self.native
        return (
            native is not None
            and native.score is not None
            and native.candidate is not None
        )

    def with_threshold(self, threshold: float) -> Pitch:
        """Redo the voicing decision with a new threshold on the voicing score.

        The threshold is applied on the native frame grid of the tracker, and the
        result is resampled onto the frame grid, so that the result is identical to
        re-extraction with the threshold.

        This is available only for the trackers providing the voicing score, e.g.,
        the confidence of neural networks. The voicing decision of the other
        trackers is made inside their algorithms, and thus requires re-extraction
        with different parameters.

        Parameters
        ----------
        threshold : float
            The frames whose score is greater than or equal to the threshold are
            regarded as voiced.

        Returns
        -------
        out : Pitch
            The new pitch object.

        Raises
        ------
        ValueError
            If the voicing score is not available.

        """
        if not self._can_rethreshold():
            raise ValueError(
                f"The voicing threshold of '{self.source}' cannot be changed without "
                "re-extraction because the voicing score is not available. Extract "
                "again with a tracker whose parameters are changed by "
                "`tracker.replace(...)`."
            )
        native = cast(NativeContour, self.native)
        score = cast(np.ndarray, native.score)
        candidate = cast(np.ndarray, native.candidate)
        native = replace(native, f0=np.where(threshold <= score, candidate, 0.0))
        grid = resample(
            native.times,
            native.f0,
            num_frames=self.length,
            frame_period=self.frame_period,
            score=score,
        )
        return replace(self, f0=grid.f0, score=grid.score, native=native)

    def with_f0(self, f0: np.ndarray, source: str | None = None) -> Pitch:
        """Create a new pitch object with the given F0 contour.

        The native contour is dropped because it is no longer consistent.

        Parameters
        ----------
        f0 : np.ndarray
            The new F0 contour in Hz.

        source : str | None, optional
            The new source name. If None, the current one is kept.

        Returns
        -------
        out : Pitch
            The new pitch object.

        """
        return replace(
            self,
            f0=f0,
            source=self.source if source is None else source,
            native=None,
        )

    def to_grid(self, num_frames: int, frame_period: float) -> Pitch:
        """Put the F0 contour on another uniform frame grid.

        The F0 and the score are resampled in the same way as the native contours of
        trackers, and the frames beyond the last frame of the contour are unvoiced.
        The native contour is dropped because it is no longer consistent.

        Parameters
        ----------
        num_frames : int
            The number of frames of the new grid.

        frame_period : float
            The frame period of the new grid in milliseconds.

        Returns
        -------
        out : Pitch
            The new pitch object.

        """
        check_frame_period(frame_period)
        times = get_frame_times(self.length, self.frame_period)
        grid = resample(
            times,
            self.f0,
            num_frames=num_frames,
            frame_period=frame_period,
            score=self.score,
        )
        f0 = grid.f0
        if 0 < self.length:
            beyond = times[-1] + 1e-9 < get_frame_times(num_frames, frame_period)
            f0 = np.where(beyond, 0.0, f0)
        return replace(
            self, f0=f0, frame_period=frame_period, score=grid.score, native=None
        )

    def tofile(
        self,
        path: str,
        output_format: OutputFormat | str = "linear",
        file_type: FileType | str = "npz",
    ) -> None:
        """Save the F0 contour to a file.

        Parameters
        ----------
        path : str
            The path to the output file.

        output_format : ['linear', 'log', 'inverse']
            The output format.

        file_type : ['npz', 'float', 'double']
            The file type. 'float' and 'double' are headerless binary files that can
            be directly processed by SPTK.

        Raises
        ------
        ValueError
            If the output format or the file type is not supported.

        """
        data = self.to(output_format)
        if file_type == "npz":
            np.savez_compressed(
                path,
                f0=data,
                format=output_format,
                frame_period=self.frame_period,
                sample_rate=self.sample_rate,
                source=self.source,
            )
        elif file_type == "float":
            data.astype(np.float32).tofile(path)
        elif file_type == "double":
            data.astype(np.float64).tofile(path)
        else:
            raise ValueError(
                f"Unsupported file type '{file_type}'. "
                f"Supported types are: {list(get_args(FileType))}"
            )
