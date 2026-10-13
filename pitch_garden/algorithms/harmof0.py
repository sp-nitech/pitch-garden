# This module is ported from HarmoF0 (https://github.com/WX-Wei/HarmoF0), which is
# distributed under the following license.
#
# The MIT License (MIT)
#
# Copyright (c) 2022 Weiweixing
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

"""A module for the HarmoF0 algorithm.

The models and the weights are those of HarmoF0, which are ported here to avoid its
dependencies, e.g., librosa and torchaudio.

"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any, Literal, cast, get_args

import numpy as np

from ..interfaces.types import Audio
from ._torch import (
    download_weights,
    import_torch,
    resolve_torch_device,
    with_num_threads,
)
from ._utils import frame_audio, mask_bins_out_of_range
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy

SAMPLE_RATE = 16000  # The sample rate of the models in Hz.
WINDOW_SIZE = 1024  # The number of samples of an input frame.
# The frame shift in samples, i.e., about 2.9 ms, which is that of the annotations
# of MDB-stem-synth, 128 samples at 44.1 kHz.
HOP_SIZE = 46
NUM_BINS = 352  # The number of pitch bins.
BINS_PER_OCTAVE = 48  # The number of pitch bins per octave, i.e., 25 cents.
LOWEST_FREQUENCY = 27.5  # The frequency of the lowest pitch bin in Hz.
HIGHEST_FREQUENCY = LOWEST_FREQUENCY * 2 ** ((NUM_BINS - 1) / BINS_PER_OCTAVE)

# The dynamic range of the log spectrogram in dB.
_TOP_DB = 80.0

# The number of the frames on each side that affect the output of a frame through
# the four blocks of the convolution with three frames.
_CONTEXT = 4

Model = Literal["mdb-stem-synth", "mir-1k"]

# The weights of HarmoF0 at a fixed commit and their SHA-256 hashes.
_WEIGHTS_URL = (
    "https://github.com/WX-Wei/HarmoF0/raw/"
    "3b2223649953f7786b602c8b7d608bce1562da38/harmof0/checkpoints/{}.pth"
)
_WEIGHTS_SHA256 = {
    "mdb-stem-synth": (
        "1d47ce083858782cf3de78b1a0f61d98e5836b5a81745a3e47f2ad8ddafec99e"
    ),
    "mir-1k": "f4d224e47871594360cedddeab3cc49048997eaa68a6fd6f5a631113a943cce6",
}


@functools.cache
def _bin_frequencies() -> np.ndarray:
    """Get the frequencies of the pitch bins in Hz.

    Returns
    -------
    out : np.ndarray
        The read-only frequencies with shape (352,), which are shared by the calls.

    """
    frequencies = LOWEST_FREQUENCY * 2 ** (np.arange(NUM_BINS) / BINS_PER_OCTAVE)
    frequencies.setflags(write=False)
    return frequencies


@dataclass(frozen=True)
class HarmoF0Config(AlgorithmConfig):
    """Configuration of HarmoF0."""

    # The model named after its training data: mdb-stem-synth or mir-1k.
    model: Model = "mdb-stem-synth"

    # The frames whose confidence, i.e., the maximum activation, is greater than or
    # equal to the threshold are regarded as voiced. The default value is the high
    # threshold of the post-processing of HarmoF0.
    voicing_threshold: float = 0.8

    batch_size: int = 1000  # The number of frames in a forward pass.
    device: str = "auto"  # The device of PyTorch, where 'auto' prefers a GPU.

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
            If the model or the batch size is invalid.

        """
        if self.model not in get_args(Model):
            raise ValueError(
                f"Unsupported model '{self.model}'. "
                f"Supported models are: {list(get_args(Model))}"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")


def _create_spectrogram(torch: Any) -> Any:
    """Create the log-frequency power spectrogram of HarmoF0 before the dB scaling.

    Parameters
    ----------
    torch : module
        The torch module.

    Returns
    -------
    out : torch.nn.Module
        The module taking frames with shape (B, 1024) and giving the power with
        shape (B, 352).

    """
    nn = torch.nn

    class Spectrogram(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("window", torch.hann_window(WINDOW_SIZE))
            indices = torch.arange(0, NUM_BINS)
            positions = (
                LOWEST_FREQUENCY
                * (2 ** (indices / BINS_PER_OCTAVE))
                / (SAMPLE_RATE / WINDOW_SIZE)
            )
            floor = torch.floor(positions).long()
            ceil = torch.ceil(positions).long()
            self.register_buffer("floor", floor)
            self.register_buffer("ceil", ceil)
            # The weights of the linear interpolation are swapped as in HarmoF0,
            # with which the models are trained.
            self.register_buffer("floor_weight", positions - floor)
            self.register_buffer("ceil_weight", ceil - positions)

        def forward(self, x: Any) -> Any:
            spectrum = torch.fft.fft(x * self.window)
            power = torch.abs(spectrum[:, : WINDOW_SIZE // 2 + 1]) ** 2
            return (
                power[:, self.floor] * self.floor_weight
                + power[:, self.ceil] * self.ceil_weight
            )

    return Spectrogram()


def _create_network(torch: Any) -> Any:
    """Create the network of HarmoF0 after the log spectrogram.

    Parameters
    ----------
    torch : module
        The torch module.

    Returns
    -------
    out : torch.nn.Module
        The network taking the log spectrogram with shape (T, 352) and giving the
        activations with shape (T, 352), whose parameter names match the weights of
        HarmoF0.

    """
    nn = torch.nn

    class MultipleRateDilatedConv(nn.Module):
        """The convolution summing the bins at the harmonics of each bin."""

        def __init__(self, channels: int, dilations: list[int]) -> None:
            super().__init__()
            self.dilations = dilations
            self.conv_list = nn.ModuleList(
                nn.Conv2d(channels, channels, kernel_size=[1, 1]) for _ in dilations
            )

        def forward(self, x: Any) -> Any:
            y = self.conv_list[0](x)
            y = nn.functional.pad(y, pad=[0, self.dilations[0]])
            y = y[:, :, :, self.dilations[0] :]
            for conv, dilation in zip(self.conv_list[1:], self.dilations[1:]):
                z = conv(x)[:, :, :, dilation:]
                y[:, :, :, : z.size(3)] += z
            return y

    def block(in_channels: int, out_channels: int, dilated: Any) -> Any:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=[3, 3], padding=[1, 1])
        norm = nn.BatchNorm2d(out_channels)
        return nn.Sequential(conv, nn.ReLU(), dilated, nn.ReLU(), norm)

    def fixed(channels: int) -> Any:
        return nn.Conv2d(
            channels, channels, kernel_size=[1, 3], padding=[0, 48], dilation=[1, 48]
        )

    # The dilations at the first 12 harmonics in the bins.
    harmonics = np.log(np.arange(1, 13)) / np.log(2 ** (1 / BINS_PER_OCTAVE))
    dilations = [int(d) for d in harmonics.round()]

    class HarmoF0(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_1 = block(1, 32, MultipleRateDilatedConv(32, dilations))
            self.block_2 = block(32, 64, fixed(64))
            self.block_3 = block(64, 128, fixed(128))
            self.block_4 = block(128, 128, fixed(128))
            self.conv_5 = nn.Conv2d(128, 64, kernel_size=[1, 1])
            self.conv_6 = nn.Conv2d(64, 1, kernel_size=[1, 1])

        def forward(self, x: Any) -> Any:
            x = x[None, None]
            for layer in (self.block_1, self.block_2, self.block_3, self.block_4):
                x = layer(x)
            x = torch.relu(self.conv_5(x))
            return torch.sigmoid(self.conv_6(x))[0, 0]

    return HarmoF0()


class HarmoF0Algorithm(BaseAlgorithm):
    """HarmoF0: logarithmic scale dilated convolution (Wei et al., 2022).

    The models and the weights are those of HarmoF0. The frames are analyzed every
    46 samples, i.e., about 2.9 milliseconds, for any frame period, because the
    convolution over the frames is trained on such dense frames, e.g., a hop of 10
    milliseconds degrades the accuracy considerably. Unlike HarmoF0, the frames are
    centered at multiples of the hop, the dynamic range of the log spectrogram is
    limited relative to the maximum of the whole audio instead of every 1000 frames,
    and the batches of frames overlap so that the results are independent of the
    batch size.

    """

    config_class = HarmoF0Config
    is_neural = True
    supports_gpu = True
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=True, voicing_from="2nd")

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize HarmoF0.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of HarmoF0.

        """
        super().__init__(config)
        self.torch: Any = None
        self.spectrogram: Any = None
        self.network: Any = None

    @classmethod
    def get_supported_sample_rates(cls) -> tuple[int, int] | None:
        """Get the sample rate of the models.

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
        return LOWEST_FREQUENCY, HIGHEST_FREQUENCY

    @classmethod
    def parallel_config(cls, config: AlgorithmConfig) -> AlgorithmConfig:
        """Use a single thread when running in parallel.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of HarmoF0.

        Returns
        -------
        out : AlgorithmConfig
            The adjusted configuration.

        """
        if isinstance(config, HarmoF0Config) and config.num_threads is None:
            return replace(config, num_threads=1)
        return config

    @classmethod
    def resolve_device(cls, device: str) -> str:
        """Resolve the device 'auto' into cuda if a GPU is available.

        Parameters
        ----------
        device : str
            The device given by users.

        Returns
        -------
        out : str
            The actual device.

        """
        return resolve_torch_device(device)

    def _weights_path(self) -> str:
        """Get the path to the weights, downloading them if necessary."""
        config = cast(HarmoF0Config, self.config)
        if config.weights_path is not None:
            return config.weights_path
        return download_weights(
            self.torch,
            _WEIGHTS_URL.format(config.model),
            _WEIGHTS_SHA256[config.model],
            f"harmof0-{config.model}",
            ".pth",
        )

    def load(self) -> None:
        """Load the model.

        Raises
        ------
        ImportError
            If PyTorch is not installed.

        """
        if self.network is not None:
            return
        self.torch = import_torch("HarmoF0")
        config = cast(HarmoF0Config, self.config)
        network = _create_network(self.torch)
        state = self.torch.load(
            self._weights_path(), map_location=config.device, weights_only=True
        )
        # The window of an unused spectrogram of torchaudio is also saved in HarmoF0.
        prefix = "waveform_to_logspecgram."
        network.load_state_dict(
            {k: v for k, v in state.items() if not k.startswith(prefix)}
        )
        self.spectrogram = _create_spectrogram(self.torch).to(config.device)
        self.network = network.to(config.device).eval()

    def _infer(self, frames: np.ndarray) -> np.ndarray:
        """Compute the activations of the pitch bins."""
        torch = self.torch
        config = cast(HarmoF0Config, self.config)
        with torch.inference_mode():
            power = torch.cat(
                [
                    self.spectrogram(
                        torch.from_numpy(batch.astype(np.float32)).to(config.device)
                    )
                    for batch in np.array_split(
                        frames, range(config.batch_size, len(frames), config.batch_size)
                    )
                ]
            )
            log_power = 10 * torch.log10(torch.clamp(power, min=1e-10))
            log_power = torch.maximum(log_power, log_power.max() - _TOP_DB)
            outputs = []
            for start in range(0, len(frames), config.batch_size):
                end = min(start + config.batch_size, len(frames))
                # The frames on both sides are added to the batch as the context.
                first = max(0, start - _CONTEXT)
                last = min(len(frames), end + _CONTEXT)
                activation = self.network(log_power[first:last])
                outputs.append(activation[start - first : end - first].cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)

    @with_num_threads
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
            The frame period in milliseconds, which is unused because the frames are
            analyzed every 46 samples.

        Returns
        -------
        out : NativeContour
            The estimated contour, where the score is the confidence.

        """
        self.load()
        config = cast(HarmoF0Config, self.config)
        frames, hop = frame_audio(
            audio.data, SAMPLE_RATE, HOP_SIZE * 1000 / SAMPLE_RATE, WINDOW_SIZE
        )
        activation = self._infer(frames)
        frequencies = _bin_frequencies()
        masked = mask_bins_out_of_range(activation, frequencies, f0_range)
        candidate = frequencies[np.argmax(masked, axis=1)]
        score = masked.max(axis=1)
        f0 = np.where(config.voicing_threshold <= score, candidate, 0.0)
        return NativeContour(
            times=np.arange(len(frames)) * hop / SAMPLE_RATE,
            f0=f0,
            score=score,
            candidate=candidate,
        )
