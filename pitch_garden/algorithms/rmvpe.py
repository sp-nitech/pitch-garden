# This module is ported from RVC (https://github.com/RVC-Project/Retrieval-based-
# Voice-Conversion-WebUI), whose implementation of RMVPE is based on the original one
# (https://github.com/Dream-High/RMVPE), and which is distributed under the following
# license.
#
# MIT License
#
# Copyright (c) 2023 liujing04
# Copyright (c) 2023 源文雨
# Copyright (c) 2023 Ftps
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

"""A module for the RMVPE algorithm.

The model is ported from RVC, which uses the weights of RMVPE distributed by RVC, to
avoid its dependencies, e.g., librosa.

"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from ..interfaces.types import Audio
from ..utils.units import cents_to_hz
from ._salience import (
    LOWEST_FREQUENCY,
    NUM_BINS,
    REFERENCE,
    bins_to_cents,
    decode_weighted_argmax,
    mask_out_of_range,
)
from ._torch import import_torch, resolve_torch_device
from ._utils import mel_filterbank
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy

SAMPLE_RATE = 16000  # The sample rate of the model in Hz.
HOP_SIZE = 160  # The frame shift of the model in samples, i.e., 10 ms.

# The settings of the log mel spectrogram used by RVC.
_N_FFT = 1024
_N_MELS = 128
_MEL_FMIN = 30.0
_MEL_FMAX = 8000.0
_CLAMP = 1e-5

# The number of frames of the input of the model must be a multiple of this.
_FRAME_UNIT = 32

# The weights distributed by RVC at a fixed revision and their SHA-256 hash.
_WEIGHTS_URL = (
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/"
    "e6d0c1a17da07c33557852f9dfa2bd44cc75737d/rmvpe.pt"
)
_WEIGHTS_SHA256 = "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"


def _create_model(torch: Any) -> Any:
    """Create the RMVPE model of RVC, i.e., E2E(4, 1, (2, 2)).

    Parameters
    ----------
    torch : module
        The torch module.

    Returns
    -------
    out : torch.nn.Module
        The model, whose parameter names match the weights distributed by RVC.

    """
    nn = torch.nn

    class ConvBlockRes(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, (3, 3), padding=(1, 1), bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels, out_channels, (3, 3), padding=(1, 1), bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            if in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

        def forward(self, x: Any) -> Any:
            if hasattr(self, "shortcut"):
                return self.conv(x) + self.shortcut(x)
            return self.conv(x) + x

    class ResEncoderBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: tuple[int, int] | None,
            n_blocks: int,
        ) -> None:
            super().__init__()
            self.conv = nn.ModuleList(
                [ConvBlockRes(in_channels, out_channels)]
                + [
                    ConvBlockRes(out_channels, out_channels)
                    for _ in range(n_blocks - 1)
                ]
            )
            self.kernel_size = kernel_size
            if kernel_size is not None:
                self.pool = nn.AvgPool2d(kernel_size=kernel_size)

        def forward(self, x: Any) -> Any:
            for conv in self.conv:
                x = conv(x)
            if self.kernel_size is not None:
                return x, self.pool(x)
            return x

    class Encoder(nn.Module):
        def __init__(self, n_encoders: int, n_blocks: int) -> None:
            super().__init__()
            self.bn = nn.BatchNorm2d(1)
            self.layers = nn.ModuleList()
            in_channels, out_channels = 1, 16
            for _ in range(n_encoders):
                self.layers.append(
                    ResEncoderBlock(in_channels, out_channels, (2, 2), n_blocks)
                )
                in_channels, out_channels = out_channels, out_channels * 2
            self.out_channel = out_channels

        def forward(self, x: Any) -> tuple[Any, list[Any]]:
            concat_tensors = []
            x = self.bn(x)
            for layer in self.layers:
                t, x = layer(x)
                concat_tensors.append(t)
            return x, concat_tensors

    class Intermediate(nn.Module):
        def __init__(
            self, in_channels: int, out_channels: int, n_inters: int, n_blocks: int
        ) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [ResEncoderBlock(in_channels, out_channels, None, n_blocks)]
                + [
                    ResEncoderBlock(out_channels, out_channels, None, n_blocks)
                    for _ in range(n_inters - 1)
                ]
            )

        def forward(self, x: Any) -> Any:
            for layer in self.layers:
                x = layer(x)
            return x

    class ResDecoderBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, n_blocks: int) -> None:
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    (3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    output_padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            self.conv2 = nn.ModuleList(
                [ConvBlockRes(out_channels * 2, out_channels)]
                + [
                    ConvBlockRes(out_channels, out_channels)
                    for _ in range(n_blocks - 1)
                ]
            )

        def forward(self, x: Any, concat_tensor: Any) -> Any:
            x = torch.cat((self.conv1(x), concat_tensor), dim=1)
            for conv in self.conv2:
                x = conv(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, in_channels: int, n_decoders: int, n_blocks: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList()
            for _ in range(n_decoders):
                self.layers.append(
                    ResDecoderBlock(in_channels, in_channels // 2, n_blocks)
                )
                in_channels //= 2

        def forward(self, x: Any, concat_tensors: list[Any]) -> Any:
            for i, layer in enumerate(self.layers):
                x = layer(x, concat_tensors[-1 - i])
            return x

    class DeepUnet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder(n_encoders=5, n_blocks=4)
            channels = self.encoder.out_channel
            self.intermediate = Intermediate(channels // 2, channels, 4, 4)
            self.decoder = Decoder(channels, n_decoders=5, n_blocks=4)

        def forward(self, x: Any) -> Any:
            x, concat_tensors = self.encoder(x)
            return self.decoder(self.intermediate(x), concat_tensors)

    class BiGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(3 * _N_MELS, 256, batch_first=True, bidirectional=True)

        def forward(self, x: Any) -> Any:
            return self.gru(x)[0]

    class E2E(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unet = DeepUnet()
            self.cnn = nn.Conv2d(16, 3, (3, 3), padding=(1, 1))
            self.fc = nn.Sequential(
                BiGRU(), nn.Linear(512, NUM_BINS), nn.Dropout(0.25), nn.Sigmoid()
            )

        def forward(self, mel: Any) -> Any:
            x = self.unet(mel.transpose(-1, -2).unsqueeze(1))
            return self.fc(self.cnn(x).transpose(1, 2).flatten(-2))

    return E2E()


@dataclass(frozen=True)
class RmvpeConfig(AlgorithmConfig):
    """Configuration of RMVPE."""

    # The frames whose salience is greater than or equal to the threshold are
    # regarded as voiced. The default value is that of RVC.
    voicing_threshold: float = 0.03

    device: str = "auto"  # The device of PyTorch, where 'auto' prefers a GPU.

    # The number of CPU threads used by PyTorch. If None, the default of PyTorch is
    # used, or 1 when running in parallel.
    num_threads: int | None = None

    # The path to the weights for offline use. If None, the weights are downloaded
    # to the cache directory of PyTorch, i.e., TORCH_HOME, at the first use.
    weights_path: str | None = None


class RmvpeAlgorithm(BaseAlgorithm):
    """RMVPE: a robust model for vocal pitch estimation (Wei et al., 2023).

    The model and the weights are those of RVC. The frame period of the model is
    fixed to 10 ms.

    """

    config_class = RmvpeConfig
    is_neural = True
    supports_gpu = True
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=False)

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize RMVPE.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of RMVPE.

        """
        super().__init__(config)
        self.torch: Any = None
        self.model: Any = None
        self.mel_basis: Any = None
        self.window: Any = None

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
        highest = cents_to_hz(bins_to_cents(NUM_BINS - 1), REFERENCE)
        return LOWEST_FREQUENCY, float(highest)

    @classmethod
    def parallel_config(cls, config: AlgorithmConfig) -> AlgorithmConfig:
        """Use a single thread when running in parallel.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of RMVPE.

        Returns
        -------
        out : AlgorithmConfig
            The adjusted configuration.

        """
        if isinstance(config, RmvpeConfig) and config.num_threads is None:
            return replace(config, num_threads=1)
        return config

    def _weights_path(self) -> str:
        """Get the path to the weights, downloading them if necessary."""
        config = cast(RmvpeConfig, self.config)
        if config.weights_path is not None:
            return config.weights_path
        directory = os.path.join(self.torch.hub.get_dir(), "checkpoints")
        path = os.path.join(directory, f"rmvpe-{_WEIGHTS_SHA256[:8]}.pth")
        if not os.path.exists(path):
            os.makedirs(directory, exist_ok=True)
            self.torch.hub.download_url_to_file(
                _WEIGHTS_URL, path, hash_prefix=_WEIGHTS_SHA256
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
        self.torch = import_torch("RMVPE")
        config = cast(RmvpeConfig, self.config)
        if config.num_threads is not None:
            self.torch.set_num_threads(config.num_threads)
        model = _create_model(self.torch)
        state = self.torch.load(
            self._weights_path(), map_location=config.device, weights_only=True
        )
        model.load_state_dict(state)
        self.model = model.to(config.device).eval()
        # The settings of RVC, i.e., the HTK formula with the Slaney normalization.
        mel_basis = mel_filterbank(
            SAMPLE_RATE, _N_FFT, _N_MELS, (_MEL_FMIN, _MEL_FMAX), htk=True
        )
        self.mel_basis = self.torch.from_numpy(mel_basis).to(config.device)
        self.window = self.torch.hann_window(_N_FFT, device=config.device)

    def _salience(self, data: np.ndarray) -> np.ndarray:
        """Compute the salience of the pitch bins of 16 kHz audio."""
        torch = self.torch
        config = cast(RmvpeConfig, self.config)
        audio = torch.from_numpy(data.astype(np.float32)).to(config.device)
        with torch.inference_mode():
            spectrum = torch.stft(
                audio[None],
                n_fft=_N_FFT,
                hop_length=HOP_SIZE,
                window=self.window,
                center=True,
                return_complex=True,
            ).abs()
            mel = torch.log(torch.clamp(self.mel_basis @ spectrum, min=_CLAMP))
            num_frames = mel.shape[-1]
            padding = -num_frames % _FRAME_UNIT
            mel = torch.nn.functional.pad(mel, (0, padding))
            salience = self.model(mel)[0, :num_frames]
        return salience.cpu().numpy().astype(np.float64)

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
            The frame period of the output, which is ignored because the native frame
            period of RMVPE is fixed to 10 ms.

        Returns
        -------
        out : NativeContour
            The estimated contour, where the score is the salience.

        """
        self.load()
        config = cast(RmvpeConfig, self.config)
        salience = mask_out_of_range(self._salience(audio.data), f0_range)
        cents, bins = decode_weighted_argmax(salience)
        score = salience[np.arange(len(bins)), bins]
        candidate = cents_to_hz(cents, REFERENCE)
        f0 = np.where(config.voicing_threshold <= score, candidate, 0.0)
        return NativeContour(
            times=np.arange(len(f0)) * HOP_SIZE / SAMPLE_RATE,
            f0=f0,
            score=score,
            candidate=candidate,
        )
