"""Tests for the utilities for the algorithms based on PyTorch."""

import pytest

from pitch_garden import Tracker
from pitch_garden.algorithms import ALGORITHM_MAP

from ..helpers import generate_pulse_train

TORCH_ALGORITHMS = sorted(
    name
    for name, cls in ALGORITHM_MAP.items()
    if "num_threads" in cls.config_class.__dataclass_fields__ and cls.supports_gpu
)


@pytest.mark.parametrize("algorithm", TORCH_ALGORITHMS)
def test_num_threads_is_local(algorithm: str, monkeypatch) -> None:
    """Test that the threads of a tracker are limited only during its estimation."""
    import torch

    original = torch.get_num_threads()
    target = 1 if original != 1 else 2
    used = []
    call = torch.nn.Module.__call__

    def spy(self, *args, **kwargs):
        used.append(torch.get_num_threads())
        return call(self, *args, **kwargs)

    tracker = Tracker(algorithm, num_threads=target, device="cpu")
    try:
        tracker.load()
    except ImportError:
        pytest.skip(f"{algorithm} is not installed.")
    assert torch.get_num_threads() == original
    monkeypatch.setattr(torch.nn.Module, "__call__", spy)
    tracker(generate_pulse_train(150.0)[:4000], 16000)
    assert used and set(used) == {target}
    assert torch.get_num_threads() == original


def test_download_weights(tmp_path, monkeypatch) -> None:
    """Test that the weights are downloaded once with the hash into the cache."""
    import torch

    from pitch_garden.algorithms._torch import download_weights

    calls = []

    def fake_download(url, path, hash_prefix):
        calls.append((url, hash_prefix))
        with open(path, "w") as f:
            f.write("weights")

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)
    sha256 = "0123456789abcdef"
    paths = [
        download_weights(torch, "https://a/b", sha256, "model", ".pth")
        for _ in range(2)
    ]
    assert paths == [str(tmp_path / "checkpoints" / "model-01234567.pth")] * 2
    assert calls == [("https://a/b", sha256)]
