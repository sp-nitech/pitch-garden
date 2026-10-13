"""The module for voters integrating the outputs of trackers."""

from typing import Any

from .aligned import AlignedVoter
from .base import BaseVoter
from .median import MedianVoter

VOTER_MAP: dict[str, type[BaseVoter]] = {
    "aligned": AlignedVoter,
    "median": MedianVoter,
}


def create_voter(voter_type: str, **params: Any) -> BaseVoter:
    """Create a voter by its registered name.

    Parameters
    ----------
    voter_type : str
        The registered name of the voter.

    **params : Any
        The parameters of the voter.

    Returns
    -------
    out : BaseVoter
        The created voter.

    Raises
    ------
    ValueError
        If the voter type is not supported.

    """
    try:
        cls = VOTER_MAP[voter_type]
    except KeyError as e:
        raise ValueError(
            f"Unsupported voter '{voter_type}'. "
            f"Supported voters are: {list(VOTER_MAP.keys())}"
        ) from e
    return cls(**params)


__all__ = [
    "VOTER_MAP",
    "create_voter",
]
