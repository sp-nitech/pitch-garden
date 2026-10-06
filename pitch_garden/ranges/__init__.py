"""The module for estimating the F0 search range."""

from typing import Any

from .base import BaseRangeEstimator
from .log_normal import LogNormalRangeEstimator

RANGE_ESTIMATOR_MAP: dict[str, type[BaseRangeEstimator]] = {
    "log-normal": LogNormalRangeEstimator,
}


def create_range_estimator(estimator_type: str, **params: Any) -> BaseRangeEstimator:
    """Create an F0 range estimator by its registered name.

    Parameters
    ----------
    estimator_type : str
        The registered name of the estimator.

    **params : Any
        The parameters of the estimator.

    Returns
    -------
    out : BaseRangeEstimator
        The created estimator.

    Raises
    ------
    ValueError
        If the estimator type is not supported.

    """
    try:
        cls = RANGE_ESTIMATOR_MAP[estimator_type]
    except KeyError as e:
        raise ValueError(
            f"Unsupported F0 range estimator '{estimator_type}'. "
            f"Supported estimators are: {list(RANGE_ESTIMATOR_MAP.keys())}"
        ) from e
    return cls(**params)


__all__ = [
    "RANGE_ESTIMATOR_MAP",
    "create_range_estimator",
]
