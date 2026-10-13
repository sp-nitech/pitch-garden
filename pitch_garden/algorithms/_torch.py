"""Utilities for the algorithms based on PyTorch."""

import functools
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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


@contextmanager
def limit_threads(torch: Any, num_threads: int | None) -> Iterator[None]:
    """Limit the number of threads of PyTorch temporarily.

    The number of threads is a setting of the process, and thus it is restored on
    exit so that the other trackers in the process are not affected.

    Parameters
    ----------
    torch : module
        The torch module.

    num_threads : int | None
        The number of threads, or None to keep the current one.

    """
    if num_threads is None:
        yield
        return
    original = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def with_num_threads[F: Callable[..., Any]](estimate: F) -> F:
    """Decorate the estimation of an algorithm to run with its number of threads.

    The algorithm is loaded before the estimation, and its configuration has to have
    `num_threads`.

    Parameters
    ----------
    estimate : Callable
        The `estimate` method of an algorithm based on PyTorch.

    Returns
    -------
    out : Callable
        The decorated method.

    """

    @functools.wraps(estimate)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        self.load()
        with limit_threads(self.torch, self.config.num_threads):
            return estimate(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


def download_weights(
    torch: Any, url: str, sha256: str, prefix: str, suffix: str
) -> str:
    """Get the path to the weights in the cache of PyTorch, downloading if necessary.

    The weights are saved as '<prefix>-<the first 8 digits of sha256><suffix>' in
    the 'checkpoints' directory of PyTorch Hub, and verified by the hash when they
    are downloaded.

    Parameters
    ----------
    torch : module
        The torch module.

    url : str
        The URL of the weights.

    sha256 : str
        The SHA-256 hash of the weights.

    prefix : str
        The prefix of the file name, e.g., the name of the model.

    suffix : str
        The extension of the file name, e.g., .pth.

    Returns
    -------
    out : str
        The path to the weights.

    """
    directory = os.path.join(torch.hub.get_dir(), "checkpoints")
    path = os.path.join(directory, f"{prefix}-{sha256[:8]}{suffix}")
    if not os.path.exists(path):
        os.makedirs(directory, exist_ok=True)
        torch.hub.download_url_to_file(url, path, hash_prefix=sha256)
    return path
