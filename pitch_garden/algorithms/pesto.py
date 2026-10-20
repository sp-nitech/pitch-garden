"""A module for the PESTO algorithm.

The pesto-pitch package is used as a dependency, which is distributed under the
LGPLv3, and thus its code is not ported here unlike CREPE. The package is installed
only on Python 3.14 or earlier because it requires torchaudio, which does not
support Python 3.15.

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from ..interfaces.types import Audio
from ..utils.units import cents_to_hz, midi_to_hz
from ._torch import import_torch, resolve_torch_device, with_num_threads
from .base import AlgorithmConfig, BaseAlgorithm, NativeContour, TwoPassPolicy

# The model published by pesto-pitch, whose HCQT covers 251 bins of a third of a
# semitone from 27.5 Hz.
_MODEL = "mir-1k_g7"
_FMIN = 27.5
_FMAX = float(cents_to_hz(251 / 3 * 100, _FMIN))

# The reduction of the activations into the pitch, which is the default of PESTO.
_REDUCTION = "alwa"


def _import_pesto() -> Any:
    """Import the pesto-pitch package lazily.

    Returns
    -------
    out : module
        The pesto module.

    Raises
    ------
    ImportError
        If the pesto-pitch package is not installed.

    """
    try:
        import pesto
    except ImportError as e:
        raise ImportError(
            "PESTO requires the pesto-pitch package, which is available on "
            "Python 3.14 or earlier because it depends on torchaudio. "
            'Install it via `pip install "pitch-garden[torch]"`.'
        ) from e
    return pesto


@dataclass(frozen=True)
class PestoConfig(AlgorithmConfig):
    """Configuration of PESTO."""

    # The frames whose confidence is greater than or equal to the threshold are
    # regarded as voiced. The threshold is lower than the middle of the confidence
    # because PESTO, which is trained on the singing voice of MIR-1K, gives low
    # confidence to the voiced frames of speech, e.g., 0.1-0.3, and a higher
    # threshold unvoices many of them. The value is chosen on the speech and the
    # singing voice of pitch-benchmark (examples/pitch_benchmark), where 0.15 is
    # the best on the valid split but 0.1 is stable on both splits.
    voicing_threshold: float = 0.1

    device: str = "auto"  # The device of PyTorch, where 'auto' prefers a GPU.

    # The number of CPU threads used by PyTorch. If None, the default of PyTorch is
    # used, or 1 when running in parallel.
    num_threads: int | None = None


class PestoAlgorithm(BaseAlgorithm):
    """PESTO: a self-supervised pitch estimator (Riou et al., 2023).

    PESTO stands for pitch estimation with self-supervised transposition-equivariant
    objective. See https://github.com/SonyCSLParis/pesto for details.

    """

    config_class = PestoConfig
    is_neural = True
    supports_gpu = True
    # The setting chosen on the speech of pitch-benchmark (examples/pitch_benchmark).
    two_pass_policy = TwoPassPolicy(enabled=True, voicing_from="2nd")

    def __init__(self, config: AlgorithmConfig) -> None:
        """Initialize PESTO.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of PESTO.

        """
        super().__init__(config)
        self.torch: Any = None
        self.pesto: Any = None
        self.model: Any = None
        self.model_frame_period: float | None = None

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
    def get_f0_limits(cls) -> tuple[float, float]:
        """Get the F0 range of the input of the model.

        Returns
        -------
        out : tuple[float, float]
            The lower and upper limits in Hz.

        """
        return _FMIN, _FMAX

    @classmethod
    def parallel_config(cls, config: AlgorithmConfig) -> AlgorithmConfig:
        """Use a single thread when running in parallel.

        Parameters
        ----------
        config : AlgorithmConfig
            The configuration of PESTO.

        Returns
        -------
        out : AlgorithmConfig
            The adjusted configuration.

        """
        if isinstance(config, PestoConfig) and config.num_threads is None:
            return replace(config, num_threads=1)
        return config

    def load(self) -> None:
        """Import PyTorch and pesto-pitch.

        The model itself is loaded at the estimation, because its frame period is
        fixed when it is loaded.

        Raises
        ------
        ImportError
            If PyTorch or pesto-pitch is not installed.

        """
        if self.pesto is not None:
            return
        self.torch = import_torch("PESTO")
        self.pesto = _import_pesto()

    def _get_model(self, frame_period: float) -> Any:
        """Get the model of the frame period, loading it if the period changes."""
        if self.model is None or self.model_frame_period != frame_period:
            model = self.pesto.load_model(_MODEL, step_size=frame_period)
            self.model = model.to(cast(PestoConfig, self.config).device)
            self.model_frame_period = frame_period
        return self.model

    @with_num_threads
    def estimate(
        self, audio: Audio, f0_range: tuple[float, float], frame_period: float
    ) -> NativeContour:
        """Estimate the F0 contour.

        Parameters
        ----------
        audio : Audio
            The input audio at any sample rate.

        f0_range : tuple[float, float]
            The F0 search range in Hz, out of which the activations are ignored.

        frame_period : float
            The frame period in milliseconds, which is rounded to an integer number of
            samples.

        Returns
        -------
        out : NativeContour
            The estimated contour, where the score is the confidence.

        """
        self.load()
        torch = self.torch
        config = cast(PestoConfig, self.config)
        model = self._get_model(frame_period)
        x = torch.from_numpy(audio.data.astype(np.float32)).to(config.device)
        with torch.inference_mode():
            _, confidence, _, activations = model(
                x,
                sr=audio.sample_rate,
                convert_to_freq=False,
                return_activations=True,
            )
            # The activations are on the bins of a third of a semitone from MIDI 0.
            bins = np.arange(activations.shape[-1]) / model.bins_per_semitone
            hz = midi_to_hz(bins)
            fmin, fmax = f0_range
            band = torch.from_numpy((fmin <= hz) & (hz <= fmax))
            pitch = self.pesto.utils.reduce_activations(
                activations * band.to(activations.device), reduction=_REDUCTION
            )
        candidate = midi_to_hz(pitch.cpu().numpy().astype(np.float64))
        candidate = np.where(np.isfinite(candidate), candidate, 0.0)
        score = confidence.cpu().numpy().astype(np.float64)
        voiced = (config.voicing_threshold <= score) & (0 < candidate)
        f0 = np.where(voiced, candidate, 0.0)
        hop = int(frame_period * audio.sample_rate / 1000 + 0.5)
        return NativeContour(
            times=np.arange(len(f0)) * hop / audio.sample_rate,
            f0=f0,
            score=score,
            candidate=candidate,
        )
