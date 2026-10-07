"""Utilities for the algorithms based on PyTorch."""

from typing import Any


def import_torch(algorithm: str) -> Any:
    """Import PyTorch lazily.

    Parameters
    ----------
    algorithm : str
        The name of the algorithm requiring PyTorch, which is used in the message.

    Returns
    -------
    out : module
        The torch module.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            f"{algorithm} requires PyTorch. "
            'Install it via `pip install "pitch-garden[torch]"`.'
        ) from e
    return torch


def resolve_torch_device(device: str) -> str:
    """Resolve the device 'auto' into cuda if a GPU is available.

    Parameters
    ----------
    device : str
        The device given by users.

    Returns
    -------
    out : str
        The actual device, which is cpu for 'auto' if PyTorch is not installed, so
        that the error is raised when the model is loaded.

    """
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
