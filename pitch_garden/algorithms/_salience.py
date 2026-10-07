"""Utilities for the salience of the pitch bins shared by CREPE and RMVPE.

The pitch bins are spaced by 20 cents from about 31.7 Hz, as in the original CREPE.

"""

import numpy as np

from ..utils.units import cents_to_hz, hz_to_cents

NUM_BINS = 360  # The number of pitch bins.
CENTS_PER_BIN = 20  # The interval of the pitch bins in cents.

# The reference frequency of the cents in Hz and the cents of the lowest pitch bin.
REFERENCE = 10.0
CENTS_OFFSET = 1997.3794084376191

# The pitch bins around the decoded one used for the weighted average.
_AVERAGE_RADIUS = 4

# The frequency of the lowest pitch bin in Hz.
LOWEST_FREQUENCY = float(cents_to_hz(CENTS_OFFSET, REFERENCE))


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
    return CENTS_PER_BIN * np.asarray(bins, dtype=np.float64) + CENTS_OFFSET


def mask_out_of_range(
    salience: np.ndarray, f0_range: tuple[float, float]
) -> np.ndarray:
    """Exclude the pitch bins out of a search range as in torchcrepe.

    Parameters
    ----------
    salience : np.ndarray
        The salience of the pitch bins with shape (T, 360).

    f0_range : tuple[float, float]
        The search range in Hz.

    Returns
    -------
    out : np.ndarray
        The salience, where the bins out of the range are -inf.

    """

    def to_bin(frequency: float) -> float:
        return float(hz_to_cents(frequency, REFERENCE) - CENTS_OFFSET) / CENTS_PER_BIN

    lower = int(np.floor(to_bin(f0_range[0])))
    upper = int(np.ceil(to_bin(f0_range[1])))
    salience = salience.copy()
    salience[:, : max(0, lower)] = -np.inf
    salience[:, max(0, upper) :] = -np.inf
    return salience


def average_cents(salience: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Average the cents around the given bins weighted by the salience.

    Parameters
    ----------
    salience : np.ndarray
        The salience of the pitch bins with shape (T, 360), where the excluded bins
        are -inf.

    bins : np.ndarray
        The center bins with shape (T,).

    Returns
    -------
    out : np.ndarray
        The cents relative to 10 Hz with shape (T,).

    """
    num_frames = len(salience)
    offsets = np.arange(-_AVERAGE_RADIUS, _AVERAGE_RADIUS + 1)
    window = bins[:, None] + offsets[None, :]
    valid = (0 <= window) & (window < NUM_BINS)
    window = np.clip(window, 0, NUM_BINS - 1)
    weights = salience[np.arange(num_frames)[:, None], window]
    weights = np.where(valid & np.isfinite(weights), weights, 0.0)
    total = weights.sum(axis=1)
    cents = (weights * bins_to_cents(window)).sum(axis=1)
    safe_total = np.where(0 < total, total, 1)
    return np.where(0 < total, cents / safe_total, bins_to_cents(bins))


def decode_weighted_argmax(salience: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode the pitch by the weighted average around the argmax.

    This follows the original CREPE, where the salience is used as the weights and
    no dither is added.

    Parameters
    ----------
    salience : np.ndarray
        The salience of the pitch bins with shape (T, 360), where the bins out of
        the search range are -inf.

    Returns
    -------
    cents : np.ndarray
        The decoded pitch in cents relative to 10 Hz with shape (T,).

    bins : np.ndarray
        The decoded pitch bins with shape (T,).

    """
    bins = np.argmax(salience, axis=1)
    return average_cents(salience, bins), bins
