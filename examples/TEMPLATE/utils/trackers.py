"""Utilities for creating trackers in the recipes."""

import dataclasses
from typing import Any

import pitch_garden as pg
from pitch_garden.algorithms import ALGORITHM_MAP


def create_tracker(
    algorithm: str,
    *,
    device: str | None = None,
    num_threads: int | None = None,
    **params: Any,
) -> pg.Tracker:
    """Create a tracker with the options given only to the algorithms accepting them.

    Parameters
    ----------
    algorithm : str
        The name of the algorithm.

    device : str | None, optional
        The device, e.g., cuda, given only to the algorithms supporting GPUs.

    num_threads : int | None, optional
        The number of threads given only to the algorithms having the parameter.

    **params : Any
        The other parameters of the tracker, e.g., the search range.

    Returns
    -------
    out : Tracker
        The tracker.

    """
    cls = ALGORITHM_MAP[algorithm]
    if device is not None and cls.supports_gpu:
        params["device"] = device
    fields = {f.name for f in dataclasses.fields(cls.config_class)}
    if num_threads is not None and "num_threads" in fields:
        params["num_threads"] = num_threads
    return pg.Tracker(algorithm, **params)
