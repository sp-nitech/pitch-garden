"""Tests for the SwiftF0 algorithm."""

import subprocess
import sys
from typing import cast

import numpy as np

from pitch_garden import Tracker
from pitch_garden.algorithms.base import AlgorithmConfig
from pitch_garden.algorithms.swift_f0 import SwiftF0Algorithm, SwiftF0Config

from ..helpers import cent_error, generate_tone


def test_tone() -> None:
    """Test that the F0 of a harmonic tone is estimated within 50 cents."""
    x = generate_tone(220.0)
    pitch = Tracker("swift-f0")(x, 16000, frame_period=5.0)
    assert pitch.length == 200
    inner = pitch.f0[20:-20]
    assert np.all(0 < inner)
    assert np.all(cent_error(inner, 220.0) < 50)


def test_swift_f0_is_imported_lazily() -> None:
    """Test that importing the package does not import swift-f0."""
    code = "import sys, pitch_garden; print('swift_f0' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_num_threads() -> None:
    """Test that the number of threads is configurable and limited in parallel."""

    def num_threads(config: AlgorithmConfig) -> int | None:
        return cast(SwiftF0Config, config).num_threads

    tracker = Tracker("swift-f0", num_threads=2)
    assert num_threads(tracker.config) == 2
    parallel_config = SwiftF0Algorithm.parallel_config
    assert num_threads(parallel_config(Tracker("swift-f0").config)) == 1
    assert num_threads(parallel_config(tracker.config)) == 2


def test_search_range() -> None:
    """Test that the search range is reflected."""
    x = generate_tone(220.0)
    pitch = Tracker("swift-f0", fmin=300.0, fmax=1000.0)(x, 16000)
    voiced_f0 = pitch.f0[pitch.voiced]
    assert np.all((300.0 <= voiced_f0) & (voiced_f0 <= 1000.0))
    assert not np.any(cent_error(voiced_f0, 220.0) < 50)


def test_voicing_threshold() -> None:
    """Test that a higher threshold gives fewer voiced frames."""
    rng = np.random.default_rng(0)
    x = generate_tone(220.0) + 0.05 * rng.standard_normal(16000)
    x[8000:] = 0.02 * rng.standard_normal(8000)
    counts = []
    for threshold in (0.1, 0.5, 0.9, 1.01):
        pitch = Tracker("swift-f0", voicing_threshold=threshold)(x, 16000)
        counts.append(int(np.sum(pitch.voiced)))
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 0


def test_with_threshold_equals_reextraction() -> None:
    """Test that re-thresholding is identical to re-extraction."""
    x = generate_tone(220.0)
    x[8000:] = 0
    base = Tracker("swift-f0", voicing_threshold=0.5)(x, 16000)
    high = Tracker("swift-f0", voicing_threshold=0.9)(x, 16000)
    np.testing.assert_allclose(base.with_threshold(0.9).f0, high.f0)
