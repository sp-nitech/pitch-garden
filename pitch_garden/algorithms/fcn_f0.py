# This module is ported from FCN-f0 (https://github.com/ardaillon/FCN-f0), which is
# distributed under the following license.
#
# MIT License
#
# Copyright (c) 2019 Luc Ardaillon
# Copyright (c) 2018 Jong Wook Kim
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

"""A module for the FCN-f0 algorithm.

The models and the weights are those of FCN-f0, whose Keras models are ported here
to PyTorch. The weights in HDF5 are read by h5py.

"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Literal, cast, get_args

import numpy as np

from ..interfaces.types import Audio
from ..utils.units import cents_to_hz, hz_to_cents
from ._torch import import_torch, resolve_torch_device
from ._utils import frame_audio
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy

SAMPLE_RATE = 8000  # The sample rate of the models in Hz.
NUM_BINS = 486  # The number of pitch bins.
LOWEST_FREQUENCY = 30.0  # The frequency of the lowest pitch bin in Hz.
HIGHEST_FREQUENCY = 1000.0  # The frequency of the highest pitch bin in Hz.
REFERENCE = 10.0  # The reference frequency of the cents in Hz.

# The cents of the pitch bins, which are evenly spaced by about 12.5 cents.
_BIN_CENTS = np.linspace(
    hz_to_cents(LOWEST_FREQUENCY, REFERENCE),
    hz_to_cents(HIGHEST_FREQUENCY, REFERENCE),
    NUM_BINS,
)

# The number of the pitch bins averaged on each side of the argmax.
_AVERAGE_RADIUS = 4

ModelSize = Literal["929", "993", "1953"]

# The numbers of filters, the kernel widths, and the number of the pooled layers of
# the models, whose names are their input sizes in samples.
_ARCHITECTURES: dict[str, tuple[list[int], list[int], int]] = {
    "929": ([256, 32, 128, 256, 512], [32, 64, 64, 64, 64], 2),
    "993": ([256, 32, 32, 128, 256, 512], [32, 32, 32, 32, 32, 32], 3),
    "1953": ([256, 32, 32, 128, 256, 512], [32, 64, 64, 64, 64, 64], 3),
}

# The weights of FCN-f0 at a fixed commit and their SHA-256 hashes.
_WEIGHTS_URL = (
    "https://github.com/ardaillon/FCN-f0/raw/"
    "8a2b530af821319b6badca93c8a0ed1f14bfee3c/models/FCN_{}/weights.h5"
)
_WEIGHTS_SHA256 = {
    "929": "34bc883250f68c28272c0667e18a730cc92ef7074d7497d101f25adae218fc4e",
    "993": "1ef2ec2762399e74809c46b7484f637e2a9d6e9b1a4daef06b3613f731011222",
    "1953": "62497ec0f8d150f71968168f7e219e9746b578dce151cdd3f4bf6c54f81d3d19",
}


def mask_out_of_range(
    salience: np.ndarray, f0_range: tuple[float, float]
) -> np.ndarray:
    """Zero the salience of the pitch bins out of the search range.

    Parameters
    ----------
    salience : np.ndarray
        The salience of the pitch bins with shape (T, 486).

    f0_range : tuple[float, float]
        The F0 search range in Hz.

    Returns
    -------
    out : np.ndarray
        The salience, where the bins out of the search range are zero.

    """
    lower, upper = hz_to_cents(np.asarray(f0_range), REFERENCE)
    outside = (_BIN_CENTS < lower) | (upper < _BIN_CENTS)
    return np.where(outside, 0.0, salience)


def local_average_cents(salience: np.ndarray) -> np.ndarray:
    """Average the cents around the argmax with the salience as in FCN-f0.

    Parameters
    ----------
    salience : np.ndarray
        The salience of the pitch bins with shape (T, 486).

    Returns
    -------
    out : np.ndarray
        The pitch in cents relative to 10 Hz with shape (T,).

    """
    bins = np.argmax(salience, axis=1)
    indices = bins[:, None] + np.arange(-_AVERAGE_RADIUS, _AVERAGE_RADIUS + 1)
    inside = (0 <= indices) & (indices < NUM_BINS)
    indices = np.clip(indices, 0, NUM_BINS - 1)
    weights = np.where(inside, np.take_along_axis(salience, indices, axis=1), 0.0)
    total = weights.sum(axis=1)
    cents = (weights * _BIN_CENTS[indices]).sum(axis=1)
    # The pitch of the frames without salience falls back to the argmax.
    safe_total = np.where(0 < total, total, 1)
    return np.where(0 < total, cents / safe_total, _BIN_CENTS[bins])


@dataclass(frozen=True)
class FcnF0Config(AlgorithmConfig):
    """Configuration of FCN-f0."""

    # The model named after its input size in samples: 929, 993, or 1953.
    model: ModelSize = "993"

    # The frames whose confidence, i.e., the maximum salience, is greater than or
    # equal to the threshold are regarded as voiced. The default value is chosen on
    # the speech of pitch-benchmark (examples/pitch_benchmark).
    voicing_threshold: float = 0.4

    batch_size: int = 512  # The number of frames in a forward pass.
    device: str = "auto"  # The device of PyTorch, where 'auto' prefers a GPU.

    # The number of CPU threads used by PyTorch. If None, the default of PyTorch is
    # used, or 1 when running in parallel.
    num_threads: int | None = None

    # The path to the weights in HDF5 for offline use. If None, the weights are
    # downloaded to the cache directory of PyTorch, i.e., TORCH_HOME, at the first
    # use.
    weights_path: str | None = None

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises
        ------
        ValueError
            If the model or the batch size is invalid.

        """
        if self.model not in get_args(ModelSize):
            raise ValueError(
                f"Unsupported model '{self.model}'. "
                f"Supported models are: {list(get_args(ModelSize))}"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")


def _create_model(torch: Any, size: str, weights: dict[str, np.ndarray]) -> Any:
    """Create the FCN-f0 model with the weights of the Keras model.

    Parameters
    ----------
    torch : module
        The torch module.

    size : str
        The name of the model.

    weights : dict[str, np.ndarray]
        The weights of the Keras model keyed by their layer and variable names, e.g.,
        'conv1/kernel'.

    Returns
    -------
    out : torch.nn.Module
        The model taking frames with shape (B, size) and giving the salience with
        shape (B, 486).

    """
    nn = torch.nn
    filters, widths, num_pooled = _ARCHITECTURES[size]

    def tensor(array: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    def conv(name: str, in_channels: int, out_channels: int, width: int) -> Any:
        layer = nn.Conv1d(in_channels, out_channels, width)
        # The Keras kernel with shape (width, 1, in, out) is transposed.
        layer.weight.data = tensor(weights[f"{name}/kernel"][:, 0].transpose(2, 1, 0))
        layer.bias.data = tensor(weights[f"{name}/bias"])
        return layer

    layers = []
    in_channels = 1
    for k, (out_channels, width) in enumerate(zip(filters, widths), start=1):
        layers += [conv(f"conv{k}", in_channels, out_channels, width), nn.ReLU()]
        if k <= num_pooled:
            layers.append(nn.MaxPool1d(2))
        # The default epsilon of the batch normalization of Keras.
        norm = nn.BatchNorm1d(out_channels, eps=1e-3)
        norm.weight.data = tensor(weights[f"conv{k}-BN/gamma"])
        norm.bias.data = tensor(weights[f"conv{k}-BN/beta"])
        norm.running_mean.data = tensor(weights[f"conv{k}-BN/moving_mean"])
        norm.running_var.data = tensor(weights[f"conv{k}-BN/moving_variance"])
        layers.append(norm)
        in_channels = out_channels
    layers += [conv("classifier", in_channels, NUM_BINS, 4), nn.Sigmoid()]

    class FcnF0(nn.Sequential):
        def forward(self, x: Any) -> Any:
            return super().forward(x[:, None, :])[:, :, 0]

    return FcnF0(*layers)


def _read_weights(path: str) -> dict[str, np.ndarray]:
    """Read the weights of the Keras model in HDF5.

    Parameters
    ----------
    path : str
        The path to the weights.

    Returns
    -------
    out : dict[str, np.ndarray]
        The weights keyed by their layer and variable names, e.g., 'conv1/kernel'.

    Raises
    ------
    ImportError
        If h5py is not installed.

    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            'FCN-f0 requires h5py. Install it via `pip install "pitch-garden[torch]"`.'
        ) from e
    weights = {}

    def visit(name: str, item: Any) -> None:
        # The variables are stored as, e.g., 'conv1/conv1/kernel:0'.
        if isinstance(item, h5py.Dataset):
            layer, _, variable = name.split("/")
            weights[f"{layer}/{variable.removesuffix(':0')}"] = item[()]

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return weights


class FcnF0Algorithm(BaseAlgorithm):
    """FCN-f0: a fully-convolutional network for pitch estimation of speech signals.

    This is proposed by Ardaillon and Roebel (2019). The frames are normalized and
    analyzed one by one as `FULLCONV=False` of FCN-f0, unlike the default of FCN-f0
    analyzing the whole audio at once with the normalization by sliding windows,
    so that any frame period is available.

    """

    config_class = FcnF0Config
    is_neural = True
    supports_gpu = True
    preferred_sample_rate = SAMPLE_RATE
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=True, voicing_from="2nd")

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize FCN-f0.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of FCN-f0.

        """
        super().__init__(config)
        self.torch: Any = None
        self.model: Any = None

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
            The configuration of FCN-f0.

        Returns
        -------
        out : AlgorithmConfig
            The adjusted configuration.

        """
        if isinstance(config, FcnF0Config) and config.num_threads is None:
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
        config = cast(FcnF0Config, self.config)
        if config.weights_path is not None:
            return config.weights_path
        sha256 = _WEIGHTS_SHA256[config.model]
        directory = os.path.join(self.torch.hub.get_dir(), "checkpoints")
        path = os.path.join(directory, f"fcn-f0-{config.model}-{sha256[:8]}.h5")
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
            If PyTorch or h5py is not installed.

        """
        if self.model is not None:
            return
        self.torch = import_torch("FCN-f0")
        config = cast(FcnF0Config, self.config)
        if config.num_threads is not None:
            self.torch.set_num_threads(config.num_threads)
        weights = _read_weights(self._weights_path())
        model = _create_model(self.torch, config.model, weights)
        self.model = model.to(config.device).eval()

    def _infer(self, frames: np.ndarray) -> np.ndarray:
        """Compute the salience of the pitch bins of normalized frames."""
        config = cast(FcnF0Config, self.config)
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(frames), config.batch_size):
                batch = frames[start : start + config.batch_size]
                # Normalize each frame as in FCN-f0.
                batch = batch - batch.mean(axis=1, keepdims=True)
                batch /= np.maximum(1e-10, batch.std(axis=1, keepdims=True))
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
            The input audio at 8 kHz.

        f0_range : tuple[float, float]
            The F0 search range in Hz.

        frame_period : float
            The frame period in milliseconds, which is rounded to an integer number of
            samples.

        Returns
        -------
        out : NativeContour
            The estimated contour, where the score is the confidence.

        """
        self.load()
        config = cast(FcnF0Config, self.config)
        frames, hop = frame_audio(
            audio.data, SAMPLE_RATE, frame_period, int(config.model)
        )
        salience = mask_out_of_range(self._infer(frames), f0_range)
        candidate = cents_to_hz(local_average_cents(salience), REFERENCE)
        score = salience.max(axis=1)
        f0 = np.where(config.voicing_threshold <= score, candidate, 0.0)
        return NativeContour(
            times=np.arange(len(frames)) * hop / SAMPLE_RATE,
            f0=f0,
            score=score,
            candidate=candidate,
        )
