"""Tests for the utilities for creating trackers in the recipes."""

from dataclasses import replace

from .loader import load_module

trackers = load_module("trackers")


def test_device_is_given_only_if_supported() -> None:
    """Test that the device is given to the algorithms supporting GPUs."""
    tracker = trackers.create_tracker("crepe", device="cuda")
    assert tracker.config == replace(tracker.config, device="cuda")
    assert not tracker.parallel_safe
    assert trackers.create_tracker("rapt", device="cuda").parallel_safe


def test_num_threads_is_given_only_if_supported() -> None:
    """Test that the number of threads is given to the algorithms having it."""
    tracker = trackers.create_tracker("swift-f0", num_threads=1)
    assert tracker.config == replace(tracker.config, num_threads=1)
    trackers.create_tracker("rapt", num_threads=1)


def test_other_parameters() -> None:
    """Test that the other parameters are given as they are."""
    assert trackers.create_tracker("rapt", fmin=60.0, fmax=400.0).f0_range == (
        60.0,
        400.0,
    )
