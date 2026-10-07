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


def hz_to_mel(frequency: ArrayLike) -> np.ndarray:
    """Convert frequencies in Hz into mels in the HTK formula.

    Parameters
    ----------
    frequency : ArrayLike
        The frequencies in Hz.

    Returns
    -------
    out : np.ndarray
        The mels, i.e., 2595 * log10(1 + frequency / 700).

    """
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def mel_to_hz(mel: ArrayLike) -> np.ndarray:
    """Convert mels in the HTK formula into frequencies in Hz.

    Parameters
    ----------
    mel : ArrayLike
        The mels.

    Returns
    -------
    out : np.ndarray
        The frequencies in Hz, i.e., 700 * (10 ** (mel / 2595) - 1).

    """
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)
