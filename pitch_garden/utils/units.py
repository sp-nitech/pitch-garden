"""Conversions of the units of frequency.

The conversions are purely mathematical, and thus the unvoiced frames, i.e., 0 Hz,
must be excluded by the callers.

"""

import numpy as np
from numpy.typing import ArrayLike


def hz_to_cents(frequency: ArrayLike, reference: ArrayLike) -> np.ndarray:
    """Convert frequencies in Hz into cents relative to a reference.

    Parameters
    ----------
    frequency : ArrayLike
        The frequencies in Hz.

    reference : ArrayLike
        The reference frequencies in Hz.

    Returns
    -------
    out : np.ndarray
        The cents, i.e., 1200 * log2(frequency / reference).

    """
    return 1200 * np.log2(np.asarray(frequency) / np.asarray(reference))


def cents_to_hz(cents: ArrayLike, reference: ArrayLike) -> np.ndarray:
    """Convert cents relative to a reference into frequencies in Hz.

    Parameters
    ----------
    cents : ArrayLike
        The cents.

    reference : ArrayLike
        The reference frequencies in Hz.

    Returns
    -------
    out : np.ndarray
        The frequencies in Hz, i.e., reference * 2 ** (cents / 1200).

    """
    return np.asarray(reference) * 2 ** (np.asarray(cents) / 1200)


def hz_to_midi(frequency: ArrayLike) -> np.ndarray:
    """Convert frequencies in Hz into MIDI note numbers, where A4 (440 Hz) is 69.

    Parameters
    ----------
    frequency : ArrayLike
        The frequencies in Hz.

    Returns
    -------
    out : np.ndarray
        The fractional MIDI note numbers.

    """
    return 69 + hz_to_cents(frequency, reference=440.0) / 100


def midi_to_hz(midi: ArrayLike) -> np.ndarray:
    """Convert MIDI note numbers into frequencies in Hz, where A4 (440 Hz) is 69.

    Parameters
    ----------
    midi : ArrayLike
        The fractional MIDI note numbers.

    Returns
    -------
    out : np.ndarray
        The frequencies in Hz.

    """
    return cents_to_hz(100 * (np.asarray(midi) - 69), reference=440.0)


# The constants of the Slaney formula, which is linear below 1 kHz and logarithmic
# above it as in the Auditory Toolbox and librosa.
_SLANEY_HZ_PER_MEL = 200.0 / 3
_SLANEY_MIN_LOG_HZ = 1000.0
_SLANEY_MIN_LOG_MEL = _SLANEY_MIN_LOG_HZ / _SLANEY_HZ_PER_MEL
_SLANEY_LOG_STEP = np.log(6.4) / 27.0


def hz_to_mel(frequency: ArrayLike, htk: bool = True) -> np.ndarray:
    """Convert frequencies in Hz into mels.

    Parameters
    ----------
    frequency : ArrayLike
        The frequencies in Hz.

    htk : bool, optional
        If True, the HTK formula, i.e., 2595 * log10(1 + frequency / 700), is used.
        Otherwise, the Slaney formula is used as the default of librosa.

    Returns
    -------
    out : np.ndarray
        The mels.

    """
    frequency = np.asarray(frequency, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + frequency / 700.0)
    linear = frequency / _SLANEY_HZ_PER_MEL
    safe = np.maximum(frequency, _SLANEY_MIN_LOG_HZ)
    log = _SLANEY_MIN_LOG_MEL + np.log(safe / _SLANEY_MIN_LOG_HZ) / _SLANEY_LOG_STEP
    return np.where(frequency < _SLANEY_MIN_LOG_HZ, linear, log)


def mel_to_hz(mel: ArrayLike, htk: bool = True) -> np.ndarray:
    """Convert mels into frequencies in Hz.

    Parameters
    ----------
    mel : ArrayLike
        The mels.

    htk : bool, optional
        If True, the HTK formula, i.e., 700 * (10 ** (mel / 2595) - 1), is used.
        Otherwise, the Slaney formula is used as the default of librosa.

    Returns
    -------
    out : np.ndarray
        The frequencies in Hz.

    """
    mel = np.asarray(mel, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    linear = mel * _SLANEY_HZ_PER_MEL
    log = _SLANEY_MIN_LOG_HZ * np.exp(_SLANEY_LOG_STEP * (mel - _SLANEY_MIN_LOG_MEL))
    return np.where(mel < _SLANEY_MIN_LOG_MEL, linear, log)
