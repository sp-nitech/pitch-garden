"""Utility functions for testing."""

import numpy as np

from pitch_garden import Tracker
from pitch_garden.utils.units import hz_to_cents


def generate_tone(
    f0: float | np.ndarray,
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    num_harmonics: int = 10,
) -> np.ndarray:
    """Generate a harmonic tone with a constant or time-varying F0.

    Parameters
    ----------
    f0 : float | np.ndarray
        The F0 in Hz, or the sample-wise F0 contour.

    duration_sec : float
        The duration in seconds.

    sample_rate : int
        The sample rate in Hz.

    num_harmonics : int
        The number of harmonics.

    Returns
    -------
    out : np.ndarray
        The generated waveform.

    """
    num_samples = int(duration_sec * sample_rate)
    f0 = np.broadcast_to(np.asarray(f0, dtype=np.float64), (num_samples,))
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    x = np.zeros(num_samples)
    for k in range(1, num_harmonics + 1):
        x += np.where(k * f0 < sample_rate / 2, np.sin(k * phase), 0) / k
    return 0.3 * x / np.max(np.abs(x))


def cent_error(estimated: np.ndarray, reference: float) -> np.ndarray:
    """Calculate the error in cents."""
    return np.abs(hz_to_cents(estimated, reference))


def generate_pulse_train(
    f0: float, duration_sec: float = 1.0, sample_rate: int = 16000
) -> np.ndarray:
    """Generate a pulse train with a decaying resonance, which resembles voice.

    Unlike a sum of sinusoids, this has clear excitation epochs, which are required
    by epoch-based algorithms such as REAPER.

    Parameters
    ----------
    f0 : float
        The F0 in Hz.

    duration_sec : float
        The duration in seconds.

    sample_rate : int
        The sample rate in Hz.

    Returns
    -------
    out : np.ndarray
        The generated waveform.

    """
    num_samples = int(duration_sec * sample_rate)
    excitation = np.zeros(num_samples)
    excitation[:: int(round(sample_rate / f0))] = 1.0
    t = np.arange(int(0.0125 * sample_rate)) / sample_rate
    resonance = np.exp(-t / 0.002) * np.sin(2 * np.pi * 700 * t)
    x = np.convolve(excitation, resonance)[:num_samples]
    return 0.3 * x / np.max(np.abs(x))


def create_tracker(algorithm: str) -> Tracker:
    """Create a tracker with the default parameters, where CREPE uses the tiny model.

    The tiny model of CREPE is much faster than the full model, which is the default.

    """
    if algorithm == "crepe":
        return Tracker(algorithm, model="tiny")
    return Tracker(algorithm)
