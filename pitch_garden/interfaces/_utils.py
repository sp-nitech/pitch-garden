"""Utility functions for interfaces."""

import numpy as np

from .types import Audio


def create_audio_object(source: np.ndarray | Audio, sample_rate: int | None) -> Audio:
    """Create an Audio object from the input source.

    Parameters
    ----------
    source : np.ndarray | Audio
        The input waveform with shape (T,) or an Audio object.

    sample_rate : int | None
        The sample rate of the input waveform.

    Returns
    -------
    out : Audio
        The created Audio object.

    Raises
    ------
    ValueError
        If the sample_rate is invalid for the given source.

    """
    if isinstance(source, Audio):
        if sample_rate is not None and source.sample_rate != sample_rate:
            raise ValueError(
                "sample_rate must be None when source is an Audio object "
                "or must match the sample rate of the Audio object."
            )
        return source
    if sample_rate is None:
        raise ValueError("sample_rate must be provided when source is not an Audio.")
    return Audio(source, sample_rate)
