# This module is ported from torchcrepe (https://github.com/maxrmorrison/torchcrepe),
# which is distributed under the following license.
#
# MIT License
#
# Copyright (c) 2020 Max Morrison
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""A module for the CREPE algorithm.

The model and the weights are those of torchcrepe, which are ported here to avoid
its dependencies, e.g., librosa and resampy.

"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Literal, cast, get_args

import numpy as np

from ..interfaces.types import Audio
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy

SAMPLE_RATE = 16000  # The sample rate of the model in Hz.
WINDOW_SIZE = 1024  # The number of samples of an input frame.
NUM_BINS = 360  # The number of pitch bins.
CENTS_PER_BIN = 20  # The interval of the pitch bins in cents.
MAX_FMAX = 2006.0  # The upper limit of F0 in Hz given by torchcrepe.

# The cents of the lowest pitch bin relative to 10 Hz.
_CENTS_OFFSET = 1997.3794084376191

# The pitch bins around the decoded one used for the weighted average.
_AVERAGE_RADIUS = 4

# The largest jump of the pitch bins between adjacent frames in Viterbi decoding.
_MAX_JUMP = 11

# The constants of the A-weighted loudness of torchcrepe in dB.
_MIN_DB = -100.0
_REF_DB = 20.0
_TOP_DB = 80.0  # The dynamic range of the spectrum as in librosa.

# The weights of torchcrepe at a fixed commit and their SHA-256 hashes.
_WEIGHTS_URL = (
    "https://github.com/maxrmorrison/torchcrepe/raw/"
    "19e2ec3d494c0797a5ff2a11408ec5838fba6681/torchcrepe/assets/{}.pth"
)
_WEIGHTS_SHA256 = {
    "full": "133225604dedd2e4005f8bbd1bd0a2ec073ba8b7a6cd31ff6d5edbbfa3539986",
    "tiny": "d4993eea36ed1a0ad9ac549c740dae5265b049ce72004f00c2f59e01c0be8432",
}

ModelCapacity = Literal["full", "tiny"]
Decoder = Literal["viterbi", "weighted_argmax"]


def _import_torch() -> Any:
    """Import PyTorch lazily.

    Returns
    -------
    out : module
        The torch module.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "CREPE requires PyTorch. "
            'Install it via `pip install "pitch-garden[torch]"`.'
        ) from e
    return torch


def bins_to_cents(bins: np.ndarray | float) -> np.ndarray:
    """Convert pitch bins to cents relative to 10 Hz.

    Parameters
    ----------
    bins : np.ndarray | float
        The pitch bins, which may be fractional.

    Returns
    -------
    out : np.ndarray
        The cents.

    """
    return CENTS_PER_BIN * np.asarray(bins, dtype=np.float64) + _CENTS_OFFSET


def _frequency_to_bin(frequency: float) -> float:
    """Convert a frequency in Hz to a fractional pitch bin."""
    return (1200 * np.log2(frequency / 10) - _CENTS_OFFSET) / CENTS_PER_BIN


def _average_cents(probabilities: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Average the cents around the given bins weighted by the probabilities."""
    num_frames = len(probabilities)
    offsets = np.arange(-_AVERAGE_RADIUS, _AVERAGE_RADIUS + 1)
    window = bins[:, None] + offsets[None, :]
    valid = (0 <= window) & (window < NUM_BINS)
    window = np.clip(window, 0, NUM_BINS - 1)
    weights = probabilities[np.arange(num_frames)[:, None], window]
    weights = np.where(valid & np.isfinite(weights), weights, 0.0)
    total = weights.sum(axis=1)
    cents = (weights * bins_to_cents(window)).sum(axis=1)
    safe_total = np.where(0 < total, total, 1)
    return np.where(0 < total, cents / safe_total, bins_to_cents(bins))


def decode_weighted_argmax(
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the pitch by the weighted average around the argmax.

    This follows the original CREPE, where the probabilities are used as the weights
    and no dither is added.

    Parameters
    ----------
    probabilities : np.ndarray
        The probabilities of the pitch bins with shape (T, 360), where the bins out
        of the search range are -inf.

    Returns
    -------
    cents : np.ndarray
        The decoded pitch in cents relative to 10 Hz with shape (T,).

    bins : np.ndarray
        The decoded pitch bins with shape (T,).

    """
    bins = np.argmax(probabilities, axis=1)
    return _average_cents(probabilities, bins), bins


def decode_viterbi(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode the pitch by Viterbi decoding followed by the weighted average.

    The path is decoded as in torchcrepe, i.e., the emission is the softmax of the
    probabilities, and the transition decreases linearly with the jump of the bins.
    Unlike torchcrepe, all rows of the transition are normalized as the inner ones,
    because the rows of the edge bins normalized by their smaller sums make the path
    stick to the edge bins. The cents are then averaged around the path as in the
    original CREPE.

    Parameters
    ----------
    probabilities : np.ndarray
        The probabilities of the pitch bins with shape (T, 360), where the bins out
        of the search range are -inf.

    Returns
    -------
    cents : np.ndarray
        The decoded pitch in cents relative to 10 Hz with shape (T,).

    bins : np.ndarray
        The decoded pitch bins with shape (T,).

    """
    tiny = np.finfo(np.float64).tiny
    num_frames = len(probabilities)
    emission = np.exp(probabilities - probabilities.max(axis=1, keepdims=True))
    emission /= emission.sum(axis=1, keepdims=True)
    log_emission = np.log(emission + tiny)

    # The candidate predecessors i = j - jump of each bin j in ascending order of i,
    # so that the first maximum is chosen as in the decoding with the full matrix.
    jumps = np.arange(_MAX_JUMP, -_MAX_JUMP - 1, -1)
    states = np.arange(NUM_BINS)
    predecessors = states[None, :] - jumps[:, None]
    valid = (0 <= predecessors) & (predecessors < NUM_BINS)
    predecessors = np.clip(predecessors, 0, NUM_BINS - 1)
    weights = (_MAX_JUMP + 1 - np.abs(jumps)).astype(np.float64)
    log_transition = np.where(
        valid, np.log(weights / weights.sum() + tiny)[:, None], -np.inf
    )

    value = log_emission[0] + np.log(1 / NUM_BINS + tiny)
    pointers = np.zeros((num_frames, NUM_BINS), dtype=np.int64)
    for t in range(1, num_frames):
        scores = value[predecessors] + log_transition
        best = np.argmax(scores, axis=0)
        pointers[t] = predecessors[best, states]
        value = scores[best, states] + log_emission[t]

    bins = np.zeros(num_frames, dtype=np.int64)
    if 0 < num_frames:
        bins[-1] = np.argmax(value)
    for t in range(num_frames - 1, 0, -1):
        bins[t - 1] = pointers[t, bins[t]]
    return _average_cents(probabilities, bins), bins


DECODERS = {"viterbi": decode_viterbi, "weighted_argmax": decode_weighted_argmax}


def filter_window(filter_length: float, frame_period: float) -> int:
    """Convert the length of the median filter into an odd number of frames.

    Parameters
    ----------
    filter_length : float
        The length of the filter in milliseconds.

    frame_period : float
        The frame period in milliseconds.

    Returns
    -------
    out : int
        The number of frames, which is the largest odd number not exceeding one
        plus the length in frames.

    """
    return 2 * int(filter_length / frame_period / 2) + 1


def median_filter(x: np.ndarray, window: int) -> np.ndarray:
    """Apply the median filter as in torchcrepe.

    The window is truncated at the edges, and the lower median is taken for an even
    number of values.

    Parameters
    ----------
    x : np.ndarray
        The input sequence with shape (T,).

    window : int
        The odd number of frames of the window.

    Returns
    -------
    out : np.ndarray
        The filtered sequence with shape (T,).

    """
    half = window // 2
    if half == 0 or len(x) == 0:
        return x.copy()
    padded = np.pad(x.astype(np.float64), half, constant_values=np.inf)
    values = np.sort(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)
    count = np.isfinite(values).sum(axis=1)
    return values[np.arange(len(x)), (count - 1) // 2]


def _a_weighting(frequencies: np.ndarray) -> np.ndarray:
    """Calculate the A-weighting in dB as in librosa, whose minimum is -80 dB."""
    c = np.array([12194.217, 20.598997, 107.65265, 737.86223]) ** 2
    f2 = frequencies**2
    with np.errstate(divide="ignore"):
        weights = 2.0 + 20.0 * (
            np.log10(c[0])
            + 2 * np.log10(f2)
            - np.log10(f2 + c[0])
            - np.log10(f2 + c[1])
            - 0.5 * np.log10(f2 + c[2])
            - 0.5 * np.log10(f2 + c[3])
        )
    return np.maximum(-80.0, weights)


def a_weighted_loudness(frames: np.ndarray, batch_size: int) -> np.ndarray:
    """Calculate the A-weighted loudness of frames as in torchcrepe.

    Parameters
    ----------
    frames : np.ndarray
        The frames of the audio with shape (T, 1024).

    batch_size : int
        The number of frames processed at once.

    Returns
    -------
    out : np.ndarray
        The loudness in dB with shape (T,).

    """
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WINDOW_SIZE) / WINDOW_SIZE)
    frequencies = np.fft.rfftfreq(WINDOW_SIZE, 1 / SAMPLE_RATE)
    weights = _a_weighting(frequencies) - _REF_DB

    def spectrum(start: int) -> np.ndarray:
        batch = frames[start : start + batch_size] * window
        amplitude = np.abs(np.fft.rfft(batch, axis=1))
        return 20 * np.log10(np.maximum(1e-5, amplitude))

    starts = range(0, len(frames), batch_size)
    peak = max((spectrum(start).max() for start in starts), default=0.0)
    loudness = []
    for start in starts:
        db = np.maximum(spectrum(start), peak - _TOP_DB) + weights
        loudness.append(np.maximum(_MIN_DB, db).mean(axis=1))
    return np.concatenate(loudness) if loudness else np.zeros(0)


@dataclass(frozen=True)
class CrepeConfig(AlgorithmConfig):
    """Configuration of CREPE."""

    model: ModelCapacity = "full"  # The model capacity: full or tiny.
    decoder: Decoder = "viterbi"  # The decoding: viterbi or weighted_argmax.

    # The frames whose periodicity is greater than or equal to the threshold are
    # regarded as voiced. The default value works for clean speech in torchcrepe.
    voicing_threshold: float = 0.21

    # The length in milliseconds of the median filter applied to the periodicity,
    # which suppresses the noise of the voicing decision. The default value works
    # for clean speech in torchcrepe.
    filter_length: float = 15.0

    # The frames whose A-weighted loudness is lower than the threshold in dB are
    # regarded as silent and given zero periodicity, because CREPE was not trained
    # on silent audio. If None, the loudness is not used. This is disabled by
    # default because the absolute loudness unvoices quiet speech, e.g., -60 dB of
    # torchcrepe degrades the accuracy on pitch-benchmark (examples/pitch_benchmark).
    silence_threshold: float | None = None

    batch_size: int = 512  # The number of frames in a forward pass.
    device: str = "cpu"  # The device of PyTorch, e.g., cpu or cuda.

    # The number of CPU threads used by PyTorch. If None, the default of PyTorch is
    # used, or 1 when running in parallel.
    num_threads: int | None = None

    # The path to the weights for offline use. If None, the weights are downloaded
    # to the cache directory of PyTorch, i.e., TORCH_HOME, at the first use.
    weights_path: str | None = None

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises
        ------
        ValueError
            If the model, the decoder, the filter length, or the batch size is
            invalid.

        """
        if self.model not in get_args(ModelCapacity):
            raise ValueError(
                f"Unsupported model '{self.model}'. "
                f"Supported models are: {list(get_args(ModelCapacity))}"
            )
        if self.decoder not in get_args(Decoder):
            raise ValueError(
                f"Unsupported decoder '{self.decoder}'. "
                f"Supported decoders are: {list(get_args(Decoder))}"
            )
        if self.filter_length < 0:
            raise ValueError("filter_length must be non-negative.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")


def _create_model(torch: Any, capacity: str) -> Any:
    """Create the CREPE model of torchcrepe.

    Parameters
    ----------
    torch : module
        The torch module.

    capacity : str
        The model capacity.

    Returns
    -------
    out : torch.nn.Module
        The model, whose parameter names match the weights of torchcrepe.

    """
    nn = torch.nn
    functional = torch.nn.functional
    if capacity == "full":
        channels = [1, 1024, 128, 128, 128, 256, 512]
    else:
        channels = [1, 128, 16, 16, 16, 32, 64]

    class Crepe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for k in range(6):
                kernel_size = (512, 1) if k == 0 else (64, 1)
                stride = (4, 1) if k == 0 else (1, 1)
                conv = nn.Conv2d(channels[k], channels[k + 1], kernel_size, stride)
                # The batch normalization converted from Keras by torchcrepe.
                norm = nn.BatchNorm2d(
                    channels[k + 1], eps=0.0010000000474974513, momentum=0.0
                )
                setattr(self, f"conv{k + 1}", conv)
                setattr(self, f"conv{k + 1}_BN", norm)
            self.in_features = channels[-1] * 4
            self.classifier = nn.Linear(self.in_features, NUM_BINS)

        def forward(self, x: Any) -> Any:
            x = x[:, None, :, None]
            for k in range(6):
                padding = (0, 0, 254, 254) if k == 0 else (0, 0, 31, 32)
                x = functional.pad(x, padding)
                x = getattr(self, f"conv{k + 1}")(x)
                x = functional.relu(x)
                x = getattr(self, f"conv{k + 1}_BN")(x)
                x = functional.max_pool2d(x, (2, 1), (2, 1))
            x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)
            return torch.sigmoid(self.classifier(x))

    return Crepe()


class CrepeAlgorithm(BaseAlgorithm):
    """CREPE: a convolutional representation for pitch estimation (Kim et al., 2018).

    The model and the weights are those of torchcrepe. Unlike torchcrepe, no dither
    is added to the pitch, so the results are reproducible, and the audio is
    resampled by soxr instead of resampy.

    """

    config_class = CrepeConfig
    is_neural = True
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=False)

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize CREPE.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of CREPE.

        """
        super().__init__(config)
        self.torch: Any = None
        self.model: Any = None

    @classmethod
    def is_parallel_safe(cls, config: AlgorithmConfig) -> bool:
        """Get whether CREPE can be run in a worker process.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of CREPE.

        Returns
        -------
        out : bool
            True if CREPE runs on CPU.

        """
        return cast(CrepeConfig, config).device == "cpu"

    @classmethod
    def get_supported_sample_rates(cls) -> tuple[int, int] | None:
        """Get the sample rate of the model.

        Returns
        -------
        out : tuple[int, int] | None
            The sample rate in Hz as the lower and upper limits.

        """
        return SAMPLE_RATE, SAMPLE_RATE

    @classmethod
    def get_f0_limits(cls) -> tuple[float, float]:
        """Get the F0 range of the pitch bins.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper limits in Hz.

        """
        return float(10 * 2 ** (_CENTS_OFFSET / 1200)), MAX_FMAX

    @classmethod
    def parallel_config(cls, config: AlgorithmConfig) -> AlgorithmConfig:
        """Use a single thread when running in parallel.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of CREPE.

        Returns
        -------
        out : AlgorithmConfig
            The adjusted configuration.

        """
        if isinstance(config, CrepeConfig) and config.num_threads is None:
            return replace(config, num_threads=1)
        return config

    def _weights_path(self) -> str:
        """Get the path to the weights, downloading them if necessary."""
        config = cast(CrepeConfig, self.config)
        if config.weights_path is not None:
            return config.weights_path
        sha256 = _WEIGHTS_SHA256[config.model]
        directory = os.path.join(self.torch.hub.get_dir(), "checkpoints")
        path = os.path.join(directory, f"crepe-{config.model}-{sha256[:8]}.pth")
        if not os.path.exists(path):
            os.makedirs(directory, exist_ok=True)
            self.torch.hub.download_url_to_file(
                _WEIGHTS_URL.format(config.model), path, hash_prefix=sha256
            )
        return path

    def load(self) -> None:
        """Load the model.

        Raises
        ------
        ImportError
            If PyTorch is not installed.

        """
        if self.model is not None:
            return
        self.torch = _import_torch()
        config = cast(CrepeConfig, self.config)
        if config.num_threads is not None:
            self.torch.set_num_threads(config.num_threads)
        model = _create_model(self.torch, config.model)
        state = self.torch.load(
            self._weights_path(), map_location=config.device, weights_only=True
        )
        model.load_state_dict(state)
        self.model = model.to(config.device).eval()

    def _infer(self, frames: np.ndarray) -> np.ndarray:
        """Compute the probabilities of the pitch bins of normalized frames."""
        config = cast(CrepeConfig, self.config)
        outputs = []
        with self.torch.no_grad():
            for start in range(0, len(frames), config.batch_size):
                batch = frames[start : start + config.batch_size]
                # Normalize each frame as in torchcrepe.
                batch = batch - batch.mean(axis=1, keepdims=True)
                batch /= np.maximum(1e-10, batch.std(axis=1, ddof=1, keepdims=True))
                inputs = self.torch.from_numpy(batch.astype(np.float32))
                outputs.append(self.model(inputs.to(config.device)).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)

    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Estimate the F0 contour.

        Parameters
        ----------
        audio : Audio
            The input audio at 16 kHz.

        f0_range : tuple[float, float]
            The F0 search range in Hz.

        frame_period : float
            The frame period in milliseconds, which is rounded to an integer number of
            samples.

        Returns
        -------
        out : NativeContour
            The estimated contour, where the score is the periodicity.

        """
        self.load()
        config = cast(CrepeConfig, self.config)
        hop = max(1, round(SAMPLE_RATE * frame_period / 1000))
        num_frames = 1 + len(audio.data) // hop
        padded = np.pad(audio.data, WINDOW_SIZE // 2)
        windows = np.lib.stride_tricks.sliding_window_view(padded, WINDOW_SIZE)
        frames = windows[::hop][:num_frames]
        probabilities = self._infer(frames)

        fmin, fmax = f0_range
        lower = int(np.floor(_frequency_to_bin(fmin)))
        upper = int(np.ceil(_frequency_to_bin(fmax)))
        probabilities[:, : max(0, lower)] = -np.inf
        probabilities[:, max(0, upper) :] = -np.inf

        cents, bins = DECODERS[config.decoder](probabilities)
        score = probabilities[np.arange(num_frames), bins]
        window = filter_window(config.filter_length, hop * 1000 / SAMPLE_RATE)
        score = median_filter(score, window)
        if config.silence_threshold is not None:
            loudness = a_weighted_loudness(frames, config.batch_size)
            score = np.where(loudness < config.silence_threshold, 0.0, score)
        candidate = 10 * 2 ** (cents / 1200)
        f0 = np.where(config.voicing_threshold <= score, candidate, 0.0)
        return NativeContour(
            times=np.arange(num_frames) * hop / SAMPLE_RATE,
            f0=f0,
            score=score,
            candidate=candidate,
        )
