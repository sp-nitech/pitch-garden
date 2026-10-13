# This module is ported from FCPE (https://github.com/CNChTu/FCPE), which is
# distributed under the following license.
#
# MIT License
#
# Copyright (c) 2023 CN_ChiTu
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

"""A module for the FCPE algorithm.

The model and the weights are those of FCPE in torchfcpe, which are ported here to
avoid its dependencies, e.g., local-attention, which the bundled model does not use.

"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from ..interfaces.types import Audio
from ..utils.units import cents_to_hz, hz_to_cents
from ._torch import (
    download_weights,
    import_torch,
    resolve_torch_device,
    with_num_threads,
)
from ._utils import mask_bins_out_of_range, mel_filterbank
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy

SAMPLE_RATE = 16000  # The sample rate of the model in Hz.
WINDOW_SIZE = 1024  # The window length and the FFT length in samples.
HOP_SIZE = 160  # The frame shift of the model in samples, i.e., 10 ms.
NUM_BINS = 360  # The number of pitch bins.
LOWEST_FREQUENCY = 32.7  # The frequency of the lowest pitch bin in Hz.
HIGHEST_FREQUENCY = 1975.5  # The frequency of the highest pitch bin in Hz.
REFERENCE = 10.0  # The reference frequency of the cents in Hz.

# The settings of the log mel spectrogram.
_N_MELS = 128
_MEL_RANGE = (0.0, 8000.0)
_CLAMP = 1e-5

# The padding on the left side, which centers the windows 80 samples after the hops.
_PAD = (WINDOW_SIZE - HOP_SIZE) // 2

# The number of the pitch bins averaged on each side of the argmax.
_AVERAGE_RADIUS = 4

# The weights of FCPE at a fixed commit and their SHA-256 hash.
_WEIGHTS_URL = (
    "https://github.com/CNChTu/FCPE/raw/"
    "6a149c1afb1c7e7821b71869dfb31ad50c95b516/torchfcpe/assets/fcpe_c_v001.pt"
)
_WEIGHTS_SHA256 = "b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015"


@functools.cache
def _bin_cents() -> np.ndarray:
    """Get the cents of the pitch bins relative to 10 Hz.

    Returns
    -------
    out : np.ndarray
        The read-only cents evenly spaced with shape (360,), which are shared by the
        calls.

    """
    lower, upper = hz_to_cents(
        np.array([LOWEST_FREQUENCY, HIGHEST_FREQUENCY]), REFERENCE
    )
    cents = np.linspace(lower, upper, NUM_BINS)
    cents.setflags(write=False)
    return cents


def local_average_cents(salience: np.ndarray) -> np.ndarray:
    """Average the cents around the argmax with the salience as in FCPE.

    The indices out of the bins are clipped to the edges, so the edge bins are
    counted multiple times as in FCPE.

    Parameters
    ----------
    salience : np.ndarray
        The salience of the pitch bins with shape (T, 360), where the bins out of
        the search range are -inf.

    Returns
    -------
    out : np.ndarray
        The pitch in cents relative to 10 Hz with shape (T,).

    """
    bins = np.argmax(salience, axis=1)
    offsets = np.arange(-_AVERAGE_RADIUS, _AVERAGE_RADIUS + 1)
    indices = np.clip(bins[:, None] + offsets, 0, NUM_BINS - 1)
    weights = np.take_along_axis(salience, indices, axis=1)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    cents = _bin_cents()
    total = weights.sum(axis=1)
    # The pitch of the frames without salience falls back to the argmax.
    safe_total = np.where(0 < total, total, 1)
    average = (weights * cents[indices]).sum(axis=1) / safe_total
    return np.where(0 < total, average, cents[bins])


@dataclass(frozen=True)
class FcpeConfig(AlgorithmConfig):
    """Configuration of FCPE."""

    # The frames whose confidence, i.e., the maximum salience, is greater than or
    # equal to the threshold are regarded as voiced. The default value is that of
    # torchfcpe.
    voicing_threshold: float = 0.006

    device: str = "auto"  # The device of PyTorch, where 'auto' prefers a GPU.

    # The number of CPU threads used by PyTorch. If None, the default of PyTorch is
    # used, or 1 when running in parallel.
    num_threads: int | None = None

    # The path to the weights for offline use. If None, the weights are downloaded
    # to the cache directory of PyTorch, i.e., TORCH_HOME, at the first use.
    weights_path: str | None = None


def _create_model(torch: Any) -> Any:
    """Create the FCPE model of the bundled weights.

    Parameters
    ----------
    torch : module
        The torch module.

    Returns
    -------
    out : torch.nn.Module
        The model taking the log mel spectrogram with shape (T, 128) and giving the
        salience with shape (T, 360), whose parameter names match the weights of
        FCPE except for the weight normalization of the output layer.

    """
    nn = torch.nn

    class DepthwiseConv(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv1d(channels, channels, 31, padding=15, groups=channels)

        def forward(self, x: Any) -> Any:
            return self.conv(x)

    class ConvModule(nn.Module):
        """The convolution module of the Conformer, which has no attention."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            # The layers are indexed as the sequential module of FCPE, where the
            # transpositions and the activations have no parameters.
            self.net = nn.ModuleDict(
                {
                    "0": nn.LayerNorm(dim),
                    "2": nn.Conv1d(dim, 4 * dim, 1),
                    "4": DepthwiseConv(2 * dim),
                    "6": nn.Conv1d(2 * dim, dim, 1),
                }
            )

        def forward(self, x: Any) -> Any:
            y = self.net["0"](x).transpose(1, 2)
            y = nn.functional.glu(self.net["2"](y), dim=1)
            y = nn.functional.silu(self.net["4"](y))
            return self.net["6"](y).transpose(1, 2)

    class EncoderLayer(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.conformer = ConvModule(dim)

        def forward(self, x: Any) -> Any:
            return x + self.conformer(x)

    class Encoder(nn.Module):
        def __init__(self, dim: int, num_layers: int) -> None:
            super().__init__()
            self.encoder_layers = nn.ModuleList(
                EncoderLayer(dim) for _ in range(num_layers)
            )

        def forward(self, x: Any) -> Any:
            for layer in self.encoder_layers:
                x = layer(x)
            return x

    class Fcpe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_stack = nn.Sequential(
                nn.Conv1d(_N_MELS, 512, 3, 1, 1),
                nn.GroupNorm(4, 512),
                nn.LeakyReLU(),
                nn.Conv1d(512, 512, 3, 1, 1),
            )
            self.net = Encoder(512, 6)
            self.norm = nn.LayerNorm(512)
            self.output_proj = nn.Linear(512, NUM_BINS)

        def forward(self, mel: Any) -> Any:
            x = self.input_stack(mel.T[None]).transpose(1, 2)
            x = self.norm(self.net(x))
            return torch.sigmoid(self.output_proj(x))[0]

    return Fcpe()


def _convert_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert the weights of FCPE into those of the ported model.

    The weight normalization of the output layer is folded into its weight, and the
    unused buffers and the normalization of the omitted attention are removed.

    Parameters
    ----------
    state : dict[str, Any]
        The state dictionary of FCPE.

    Returns
    -------
    out : dict[str, Any]
        The state dictionary of the ported model.

    """
    unused = ("cent_table", "gaussian_blurred_cent_mask")
    converted = {
        k: v
        for k, v in state.items()
        if k not in unused
        and not k.startswith("output_proj.weight_")
        and not (k.startswith("net.encoder_layers.") and ".norm." in k)
    }
    g, v = state["output_proj.weight_g"], state["output_proj.weight_v"]
    converted["output_proj.weight"] = g * v / v.norm(dim=1, keepdim=True)
    return converted


class FcpeAlgorithm(BaseAlgorithm):
    """FCPE: fast context-based pitch estimation (Luo et al., 2025).

    The model and the weights are those bundled in torchfcpe. The frames are
    analyzed every 10 milliseconds for any frame period. The outputs are at the hops
    as trained in FCPE, although the windows are centered 80 samples after the hops.
    The whole audio is analyzed at once because the group normalization of the model
    uses the statistics over all the frames.

    """

    config_class = FcpeConfig
    is_neural = True
    supports_gpu = True
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=False)

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize FCPE.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of FCPE.

        """
        super().__init__(config)
        self.torch: Any = None
        self.model: Any = None
        self.mel_basis: Any = None

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
            The configuration of FCPE.

        Returns
        -------
        out : AlgorithmConfig
            The adjusted configuration.

        """
        if isinstance(config, FcpeConfig) and config.num_threads is None:
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
        config = cast(FcpeConfig, self.config)
        if config.weights_path is not None:
            return config.weights_path
        return download_weights(
            self.torch, _WEIGHTS_URL, _WEIGHTS_SHA256, "fcpe-c-v001", ".pt"
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
        self.torch = import_torch("FCPE")
        config = cast(FcpeConfig, self.config)
        model = _create_model(self.torch)
        checkpoint = self.torch.load(
            self._weights_path(), map_location=config.device, weights_only=True
        )
        model.load_state_dict(_convert_state(checkpoint["model"]))
        # The settings of FCPE, i.e., the Slaney formula of librosa.
        mel_basis = mel_filterbank(
            SAMPLE_RATE, WINDOW_SIZE, _N_MELS, _MEL_RANGE, htk=False
        )
        self.mel_basis = self.torch.from_numpy(mel_basis).to(config.device)
        self.model = model.to(config.device).eval()

    def _log_mel(self, data: np.ndarray) -> Any:
        """Calculate the log mel spectrogram as in FCPE."""
        torch = self.torch
        config = cast(FcpeConfig, self.config)
        y = torch.from_numpy(data.astype(np.float32)).to(config.device)[None, None]
        pad_right = max(_PAD, WINDOW_SIZE - len(data) - _PAD)
        mode = "reflect" if pad_right < len(data) else "constant"
        y = torch.nn.functional.pad(y, (_PAD, pad_right), mode=mode)[0]
        window = torch.hann_window(WINDOW_SIZE, device=config.device)
        spectrum = torch.stft(
            y,
            WINDOW_SIZE,
            hop_length=HOP_SIZE,
            win_length=WINDOW_SIZE,
            window=window,
            center=False,
            return_complex=True,
        )[0]
        magnitude = torch.sqrt(spectrum.real**2 + spectrum.imag**2 + 1e-9)
        mel = torch.log(torch.clamp(self.mel_basis @ magnitude, min=_CLAMP)).T
        # The number of frames is adjusted by repeating the last frame as in FCPE.
        num_frames = len(data) // HOP_SIZE + 1
        if len(mel) < num_frames:
            mel = torch.cat([mel, mel[-1:].expand(num_frames - len(mel), -1)])
        return mel[:num_frames]

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
            The estimated contour, where the score is the confidence.

        """
        self.load()
        config = cast(FcpeConfig, self.config)
        with self.torch.inference_mode():
            mel = self._log_mel(np.asarray(audio.data))
            salience = self.model(mel).cpu().numpy().astype(np.float64)
        frequencies = cents_to_hz(_bin_cents(), REFERENCE)
        masked = mask_bins_out_of_range(salience, frequencies, f0_range)
        candidate = cents_to_hz(local_average_cents(masked), REFERENCE)
        score = masked.max(axis=1)
        f0 = np.where(config.voicing_threshold <= score, candidate, 0.0)
        times = np.arange(len(f0)) * HOP_SIZE / SAMPLE_RATE
        return NativeContour(times=times, f0=f0, score=score, candidate=candidate)
