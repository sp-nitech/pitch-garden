# This module is ported from PiENet (https://github.com/mairaksi/PiENet) by Manu
# Airaksinen and Lauri Juvela with modifications, e.g., the framing and the
# selection of the search range. PiENet is distributed under the following license.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module for the PiENet algorithm.

The model and the weights are those of PiENet, which are ported here because PiENet
is not distributed on PyPI.

"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any, cast

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

SAMPLE_RATE = 16000  # The sample rate of the model in Hz.
WINDOW_SIZE = 512  # The number of samples of an input frame.
HOP_SIZE = 160  # The frame shift of the model in samples, i.e., 10 ms.
NUM_BINS = 350  # The number of pitch bins, which are followed by the unvoiced class.
LOWEST_FREQUENCY = 50.0  # The frequency of the lowest pitch bin in Hz.
HIGHEST_FREQUENCY = 500.0  # The frequency of the highest pitch bin in Hz.

# The number of the frames on each side that affect the output of a frame through
# the dilated convolution and the convolution of the output layers.
_CONTEXT = 64

# The weights of PiENet at a fixed commit and their SHA-256 hash.
_WEIGHTS_URL = (
    "https://github.com/mairaksi/PiENet/raw/"
    "2c5e7ebb11792b9c1491383a5234d83e1489b461/src/pienet/assets/gtaug.pt"
)
_WEIGHTS_SHA256 = "0088d1d502c4ed8845e4d15ae813e60029caada84514a23a55c84b4c6bdbd0a1"


@functools.cache
def _bin_frequencies() -> np.ndarray:
    """Get the frequencies of the pitch bins in Hz.

    Returns
    -------
    out : np.ndarray
        The read-only frequencies evenly spaced on the log scale with shape (350,),
        which are shared by the calls.

    """
    frequencies = np.exp(
        np.linspace(np.log(LOWEST_FREQUENCY), np.log(HIGHEST_FREQUENCY), NUM_BINS)
    )
    frequencies.setflags(write=False)
    return frequencies


@dataclass(frozen=True)
class PiENetConfig(AlgorithmConfig):
    """Configuration of PiENet."""

    # The frames whose voicing probability, i.e., 1 - the probability of the
    # unvoiced class, is greater than or equal to the threshold are regarded as
    # voiced. The default value is chosen on the speech of pitch-benchmark
    # (examples/pitch_benchmark).
    voicing_threshold: float = 0.9

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
            If the batch size is invalid.

        """
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")


def _create_model(torch: Any) -> Any:
    """Create the PiENet model.

    Parameters
    ----------
    torch : module
        The torch module.

    Returns
    -------
    out : torch.nn.Module
        The model taking frames with shape (T, 512) and giving the logits of the
        pitch bins and the unvoiced class with shape (T, 351), whose parameter names
        match the weights of PiENet.

    """
    nn = torch.nn

    class GatedResidualBlock(nn.Module):
        """The gated dilated convolution with the residual and skip outputs."""

        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.filter_gate = nn.Conv1d(
                channels, 2 * channels, 5, dilation=dilation, padding=2 * dilation
            )
            self.skip = nn.Conv1d(channels, channels, 1)
            self.output = nn.Conv1d(channels, channels, 1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            filt, gate = self.filter_gate(x).chunk(2, dim=1)
            y = torch.tanh(filt) * torch.sigmoid(gate)
            return self.output(y) + x, self.skip(y)

    class PiENet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_conv = nn.Conv1d(WINDOW_SIZE, 128, 1)
            self.blocks = nn.ModuleList(
                GatedResidualBlock(128, d) for d in (1, 2, 4, 8, 1, 2, 4, 8)
            )
            self.postnet_conv1 = nn.Conv1d(128, 256, 5, padding=2)
            self.postnet_conv2 = nn.Conv1d(256, NUM_BINS + 1, 5, padding=2)

        def forward(self, frames: Any) -> Any:
            x = torch.tanh(self.input_conv(frames.T[None]))
            skips = 0
            for block in self.blocks:
                x, skip = block(x)
                skips = skips + skip
            y = torch.relu(self.postnet_conv1(skips))
            return self.postnet_conv2(y)[0].T

    return PiENet()


class PiENetAlgorithm(BaseAlgorithm):
    """PiENet: the pitch estimation network (Airaksinen et al., 2019).

    The model and the weights are those of GTE-AUG in PiENet, which is trained with
    noise augmentation for robustness. The frames are analyzed every 10
    milliseconds, with which the convolution over the frames is trained, for any
    frame period, and the batches of frames overlap so that the results are
    independent of the batch size.

    """

    config_class = PiENetConfig
    is_neural = True
    supports_gpu = True
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=False)

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize PiENet.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of PiENet.

        """
        super().__init__(config)
        self.torch: Any = None
        self.model: Any = None

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
        return LOWEST_FREQUENCY, HIGHEST_FREQUENCY

    @classmethod
    def parallel_config(cls, config: AlgorithmConfig) -> AlgorithmConfig:
        """Use a single thread when running in parallel.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of PiENet.

        Returns
        -------
        out : AlgorithmConfig
            The adjusted configuration.

        """
        if isinstance(config, PiENetConfig) and config.num_threads is None:
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
        config = cast(PiENetConfig, self.config)
        if config.weights_path is not None:
            return config.weights_path
        return download_weights(
            self.torch, _WEIGHTS_URL, _WEIGHTS_SHA256, "pienet-gtaug", ".pt"
        )

    def load(self) -> None:
        """Load the model.

        Raises
        ------
        ImportError
            If PyTorch is not installed.

        """
        if self.model is not None:
            return
        self.torch = import_torch("PiENet")
        config = cast(PiENetConfig, self.config)
        model = _create_model(self.torch)
        checkpoint = self.torch.load(
            self._weights_path(), map_location=config.device, weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"])
        self.model = model.to(config.device).eval()

    def _infer(self, frames: np.ndarray) -> np.ndarray:
        """Compute the probabilities of the pitch bins and the unvoiced class."""
        torch = self.torch
        config = cast(PiENetConfig, self.config)
        inputs = torch.from_numpy(frames.astype(np.float32)).to(config.device)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(frames), config.batch_size):
                end = min(start + config.batch_size, len(frames))
                # The frames on both sides are added to the batch as the context.
                first = max(0, start - _CONTEXT)
                last = min(len(frames), end + _CONTEXT)
                logits = self.model(inputs[first:last])
                probabilities = torch.softmax(logits, dim=1)
                outputs.append(probabilities[start - first : end - first].cpu().numpy())
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
            analyzed every 10 milliseconds.

        Returns
        -------
        out : NativeContour
            The estimated contour, where the score is the voicing probability.

        """
        self.load()
        config = cast(PiENetConfig, self.config)
        frames, hop = frame_audio(
            audio.data, SAMPLE_RATE, HOP_SIZE * 1000 / SAMPLE_RATE, WINDOW_SIZE
        )
        probabilities = self._infer(frames)
        frequencies = _bin_frequencies()
        pitch = mask_bins_out_of_range(
            probabilities[:, :NUM_BINS], frequencies, f0_range
        )
        candidate = frequencies[np.argmax(pitch, axis=1)]
        score = 1 - probabilities[:, NUM_BINS]
        f0 = np.where(config.voicing_threshold <= score, candidate, 0.0)
        return NativeContour(
            times=np.arange(len(frames)) * hop / SAMPLE_RATE,
            f0=f0,
            score=score,
            candidate=candidate,
        )
