"""Tests for the tracker."""

import pickle
from typing import cast

import numpy as np
import pytest

from pitch_garden import Tracker
from pitch_garden.algorithms import ALGORITHM_MAP
from pitch_garden.algorithms.swift_f0 import SwiftF0Config

from ..dummy import (
    ConstantAlgorithm,
    ConstantConfig,
    DeviceAlgorithm,
    FailingAlgorithm,
    FramePeriodAlgorithm,
    RangelessAlgorithm,
)
from ..helpers import cent_error, create_tracker, generate_pulse_train


def test_tracker_with_class() -> None:
    """Test that a tracker can be created from an algorithm class."""
    tracker = Tracker(ConstantAlgorithm, f0=200.0)
    pitch = tracker(np.zeros(16000), 16000, frame_period=5.0)
    assert pitch.length == 200
    assert pitch.source == "ConstantAlgorithm"
    np.testing.assert_allclose(pitch.f0, 200.0)
    assert pitch.score is not None
    assert pitch.native is not None


def test_tracker_with_name() -> None:
    """Test that a tracker can be created from a registered name."""
    tracker = Tracker("swift-f0", voicing_threshold=0.8)
    assert tracker.name == "swift-f0"
    assert cast(SwiftF0Config, tracker.config).voicing_threshold == 0.8


def test_tracker_custom_name() -> None:
    """Test that the same algorithm can be used with different names."""
    a = Tracker(ConstantAlgorithm, name="a", f0=100.0)
    b = Tracker(ConstantAlgorithm, name="b", f0=200.0)
    assert a.name != b.name
    assert a(np.zeros(1600), 16000).source == "a"


def test_tracker_unknown_algorithm() -> None:
    """Test that an unknown algorithm is rejected."""
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        Tracker("unknown")


def test_tracker_unknown_parameter() -> None:
    """Test that an unknown parameter is rejected with the valid keys."""
    with pytest.raises(ValueError, match="voicing_threshold"):
        Tracker(ConstantAlgorithm, foo=1)


def test_tracker_device() -> None:
    """Test that all algorithms accept the device, where GPUs need the support."""
    assert Tracker(ConstantAlgorithm, device="cpu").parallel_safe
    with pytest.raises(ValueError, match="device"):
        Tracker(ConstantAlgorithm, device="cuda")
    assert Tracker(DeviceAlgorithm).parallel_safe
    assert not Tracker(DeviceAlgorithm, device="cuda").parallel_safe


def test_tracker_auto_device() -> None:
    """Test that the automatic device is resolved when the tracker is created."""
    assert Tracker(ConstantAlgorithm, device="auto").config.device == "cpu"
    tracker = Tracker(DeviceAlgorithm, device="auto")
    assert tracker.config.device == DeviceAlgorithm.available_device
    assert not tracker.parallel_safe


def test_tracker_invalid_range() -> None:
    """Test that an invalid search range is rejected."""
    with pytest.raises(ValueError):
        Tracker(ConstantAlgorithm, fmin=300.0, fmax=200.0)


def test_tracker_range_is_clipped_to_limits() -> None:
    """Test that the search range is clipped to the limits of the algorithm."""
    tracker = Tracker(ConstantAlgorithm, fmin=1.0, fmax=1e5)
    assert tracker.f0_range == (10.0, 5000.0)


def test_tracker_range_for_rangeless_algorithm() -> None:
    """Test that the search range cannot be given to rangeless algorithms."""
    assert not Tracker(RangelessAlgorithm).supports_f0_range
    with pytest.raises(ValueError):
        Tracker(RangelessAlgorithm, fmin=100.0)


def test_tracker_range_at_call() -> None:
    """Test that the search range can be given at call without reloading."""
    tracker = Tracker(ConstantAlgorithm, f0=200.0)
    ConstantAlgorithm.num_instances = 0
    x = np.zeros(1600)
    assert np.all(tracker(x, 16000).voiced)
    assert not np.any(tracker(x, 16000, f0_range=(300.0, 400.0)).voiced)
    # The range is clipped to the limits of the algorithm.
    assert np.all(tracker(x, 16000, f0_range=(1.0, 1e5)).voiced)
    assert ConstantAlgorithm.num_instances == 1
    with pytest.raises(ValueError):
        tracker(x, 16000, f0_range=(300.0, 200.0))
    with pytest.raises(ValueError):
        Tracker(RangelessAlgorithm)(x, 16000, f0_range=(100.0, 200.0))


def test_tracker_replace() -> None:
    """Test that the parameters can be replaced."""
    tracker = Tracker(ConstantAlgorithm, name="a", f0=100.0)
    new_tracker = tracker.replace(fmin=50.0, fmax=150.0)
    assert new_tracker.name == "a"
    assert new_tracker.f0_range == (50.0, 150.0)
    assert isinstance(new_tracker.config, ConstantConfig)
    assert new_tracker.config.f0 == 100.0


def test_tracker_is_picklable() -> None:
    """Test that the tracker can be pickled."""
    tracker = Tracker(ConstantAlgorithm, f0=150.0)
    tracker(np.zeros(1600), 16000)  # Load the algorithm.
    restored = pickle.loads(pickle.dumps(tracker))
    np.testing.assert_allclose(restored(np.zeros(1600), 16000).f0, 150.0)


@pytest.mark.parametrize(
    "sample_rate, received", [(8000, 8000), (16000, 16000), (44100, 16000)]
)
def test_resampled_only_if_unsupported(sample_rate: int, received: int) -> None:
    """Test that the audio is resampled only for unsupported sample rates."""
    from ..dummy import SampleRateAlgorithm

    SampleRateAlgorithm.received_sample_rates = []
    x = np.zeros(int(1.234 * sample_rate))
    pitch = Tracker(SampleRateAlgorithm)(x, sample_rate, frame_period=5.0)
    assert SampleRateAlgorithm.received_sample_rates == [received]
    # The number of frames is determined by the original audio.
    assert pitch.length == int(np.ceil(len(x) / (sample_rate * 0.005)))
    assert pitch.sample_rate == sample_rate


@pytest.mark.parametrize("algorithm", sorted(ALGORITHM_MAP))
def test_preferred_sample_rate_is_supported(algorithm: str) -> None:
    """Test that unsupported audio is resampled to a supported sample rate."""
    cls = ALGORITHM_MAP[algorithm]
    limits = cls.get_supported_sample_rates()
    if limits is not None:
        assert limits[0] <= cls.preferred_sample_rate <= limits[1]


def test_two_pass_policy() -> None:
    """Test that the two-pass policy follows the algorithm unless it is given."""
    from pitch_garden import TwoPassPolicy

    from ..dummy import NoTwoPassAlgorithm

    default = Tracker(NoTwoPassAlgorithm).two_pass_policy
    assert default == NoTwoPassAlgorithm.two_pass_policy
    policy = TwoPassPolicy(enabled=True, voicing_from="1st")
    tracker = Tracker(NoTwoPassAlgorithm, two_pass_policy=policy)
    assert tracker.two_pass_policy == policy
    assert tracker.replace(f0=150.0).two_pass_policy == policy


@pytest.mark.parametrize("frame_period", [0.0, -5.0, np.nan, np.inf])
def test_tracker_invalid_frame_period(frame_period: float) -> None:
    """Test that an invalid frame period is rejected before the tracking."""
    ConstantAlgorithm.num_calls = 0
    with pytest.raises(ValueError, match="frame_period"):
        Tracker(ConstantAlgorithm)(np.zeros(16000), 16000, frame_period=frame_period)
    assert ConstantAlgorithm.num_calls == 0


@pytest.mark.parametrize(
    "frame_period, duration, expected",
    [
        (5.0, 1.0, 5.0),  # Within the limits.
        (80.0, 1.0, 40.0),  # The limit of the algorithm.
        (80.0, 0.1, 25.0),  # A quarter of the duration.
        (0.5, 1.0, 2.0),  # The lower limit of the algorithm.
        (5.0, 0.004, 2.0),  # The lower limit precedes a quarter of the duration.
    ],
)
def test_native_frame_period_is_limited(
    frame_period: float, duration: float, expected: float
) -> None:
    """Test that the frame period for the estimation is limited."""
    x = np.zeros(int(16000 * duration))
    pitch = Tracker(FramePeriodAlgorithm)(x, 16000, frame_period=frame_period)
    assert FramePeriodAlgorithm.received == pytest.approx(expected)
    assert pitch.frame_period == frame_period


def test_empty_audio_gives_empty_pitch() -> None:
    """Test that empty audio gives an empty contour without calling the algorithm."""
    assert Tracker(FailingAlgorithm)(np.zeros(0), 16000).length == 0


@pytest.mark.parametrize("algorithm", sorted(ALGORITHM_MAP))
def test_long_frame_period(algorithm: str) -> None:
    """Test that a frame period longer than the algorithms accept still works."""
    tracker = create_tracker(algorithm)
    try:
        tracker.load()
    except ImportError:
        pytest.skip(f"{algorithm} is not installed.")
    # Some algorithms, e.g., STRAIGHT, need silence to detect voiced frames.
    silence = np.zeros(8000)
    x = np.concatenate([silence, generate_pulse_train(150.0), silence])
    pitch = tracker(x, 16000, frame_period=200.0)
    assert pitch.length == 10
    assert np.all(cent_error(pitch.f0[3:8], 150.0) < 50)  # 0.6-1.4 seconds
    assert tracker(x, 16000, frame_period=2000.0).length == 1


@pytest.mark.parametrize("algorithm", sorted(ALGORITHM_MAP))
def test_short_frame_period(algorithm: str) -> None:
    """Test that a frame period shorter than the algorithms accept still works."""
    tracker = create_tracker(algorithm)
    try:
        tracker.load()
    except ImportError:
        pytest.skip(f"{algorithm} is not installed.")
    silence = np.zeros(8000)
    x = np.concatenate([silence, generate_pulse_train(150.0), silence])
    pitch = tracker(x, 16000, frame_period=0.5)
    assert pitch.length == 4000
    assert np.all(cent_error(pitch.f0[1200:2800], 150.0) < 50)  # 0.6-1.4 seconds
