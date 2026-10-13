"""Tests for the pipeline, including the two-pass tracking."""

import logging
import sys

import numpy as np
import pytest

from pitch_garden import Pipeline, Tracker, TwoPass, TwoPassPolicy
from pitch_garden.algorithms import ALGORITHM_MAP
from pitch_garden.algorithms.base import VoicingSource
from pitch_garden.voters.median import MedianVoter

from ..dummy import (
    AlternatingAlgorithm,
    ConstantAlgorithm,
    FailingAlgorithm,
    FirstVoicingAlgorithm,
    MedianScaleRangeEstimator,
    NarrowRangeFailingAlgorithm,
    NoTwoPassAlgorithm,
    RangelessAlgorithm,
    RangeSensitiveAlgorithm,
    ThreadsAlgorithm,
)
from ..helpers import create_tracker


def make_trackers() -> list[Tracker]:
    """Create dummy trackers."""
    return [
        Tracker(ConstantAlgorithm, name="a", f0=100.0),
        Tracker(ConstantAlgorithm, name="b", f0=200.0, hop=16.0),
        Tracker(RangelessAlgorithm, name="c", f0=400.0),
    ]


def test_voting() -> None:
    """Test that the outputs of the trackers are voted."""
    with Pipeline(make_trackers(), num_workers=1) as pipeline:
        result = pipeline(np.zeros(16000), 16000, frame_period=5.0)
    assert list(result.each.keys()) == ["a", "b", "c"]
    assert result.pitch is not None
    np.testing.assert_allclose(result.pitch.f0, 200.0)


def test_parallel_equals_sequential() -> None:
    """Test that the parallel execution gives the same results."""
    x = np.zeros(8000)
    with Pipeline(make_trackers(), num_workers=1) as pipeline:
        sequential = pipeline(x, 16000)
    with Pipeline(make_trackers(), num_workers=3) as pipeline:
        parallel = pipeline(x, 16000)
        # The pool is reused for the second call.
        parallel = pipeline(x, 16000)
    assert sequential.pitch is not None and parallel.pitch is not None
    np.testing.assert_array_equal(sequential.pitch.f0, parallel.pitch.f0)
    for name in sequential.each:
        np.testing.assert_array_equal(sequential.each[name].f0, parallel.each[name].f0)


def test_parallel_config() -> None:
    """Test that the number of threads is limited only in worker processes."""
    trackers = [
        Tracker(ThreadsAlgorithm, name="default"),
        Tracker(ThreadsAlgorithm, name="explicit", num_threads=4),
    ]
    x = np.zeros(1600)
    with Pipeline(trackers, voter=None, num_workers=2) as pipeline:
        parallel = pipeline(x, 16000)
    with Pipeline(trackers, voter=None, num_workers=1) as pipeline:
        sequential = pipeline(x, 16000)
    np.testing.assert_allclose(parallel.each["default"].f0, 100.0)
    np.testing.assert_allclose(parallel.each["explicit"].f0, 400.0)
    np.testing.assert_allclose(sequential.each["default"].f0, 0.0)
    np.testing.assert_allclose(sequential.each["explicit"].f0, 400.0)


def test_sequential_if_workers_cannot_be_spawned(monkeypatch, caplog) -> None:
    """Test that the trackers run sequentially if the main script is not a file."""
    monkeypatch.setattr(sys.modules["__main__"], "__file__", "<stdin>", raising=False)
    with caplog.at_level(logging.WARNING, logger="pitch_garden"):
        with Pipeline(make_trackers(), num_workers=3) as pipeline:
            result = pipeline(np.zeros(1600), 16000)
            assert pipeline._executor is None
    assert result.pitch is not None
    assert any("sequentially" in r.getMessage() for r in caplog.records)


def test_single_tracker_is_not_voted() -> None:
    """Test that the output of a single tracker is the final result as it is."""
    tracker = Tracker(AlternatingAlgorithm, name="alt")
    with Pipeline([tracker], voter="median") as pipeline:
        result = pipeline(np.zeros(1600), 16000, frame_period=5.0)
    assert result.pitch is not None and result.pitch is result.each["alt"]
    assert result.pitch.source == "alt"
    np.testing.assert_array_equal(result.pitch.f0[:4], [100, 200, 100, 200])
    assert result.f0_range is None and result.pass1 is None


def test_without_voter() -> None:
    """Test that only the individual outputs are returned without a voter."""
    with Pipeline(make_trackers(), voter=None, num_workers=1) as pipeline:
        result = pipeline(np.zeros(1600), 16000)
    assert result.pitch is None
    assert len(result.each) == 3


def test_duplicate_names() -> None:
    """Test that duplicate tracker names are rejected."""
    with pytest.raises(ValueError, match="Duplicate"):
        Pipeline([Tracker(ConstantAlgorithm), Tracker(ConstantAlgorithm)])


def test_empty_trackers() -> None:
    """Test that at least one tracker is required."""
    with pytest.raises(ValueError):
        Pipeline([])


def test_invalid_frame_period_is_not_a_tracker_failure() -> None:
    """Test that an invalid frame period is raised as it is, not as failures."""
    with Pipeline([Tracker(ConstantAlgorithm)], num_workers=1) as pipeline:
        with pytest.raises(ValueError, match="frame_period"):
            pipeline(np.zeros(16000), 16000, frame_period=0.0)


def test_empty_audio_in_pipeline() -> None:
    """Test that the pipeline handles empty audio including the two-pass tracking."""
    trackers = [Tracker(ConstantAlgorithm, name=n) for n in ("a", "b", "c")]
    with Pipeline(trackers, two_pass=TwoPass(), num_workers=1) as pipeline:
        result = pipeline(np.zeros(0), 16000)
    assert result.pitch is not None and result.pitch.length == 0
    assert result.f0_range is not None and result.f0_range.fallback
    assert MedianVoter()(result.each).length == 0


def test_second_pass_failure_falls_back_to_first_pass(caplog) -> None:
    """Test that a failure in the second pass keeps the result of the first pass."""
    two_pass = TwoPass(
        estimator=MedianScaleRangeEstimator(),
        fmin=40.0,
        fmax=2000.0,
    )
    tracker = Tracker(NarrowRangeFailingAlgorithm, name="a", f0=200.0)
    with caplog.at_level(logging.WARNING, logger="pitch_garden"):
        with Pipeline([tracker], two_pass=two_pass) as pipeline:
            result = pipeline(np.zeros(16000), 16000)
    assert result.pass1 is not None
    np.testing.assert_array_equal(result.each["a"].f0, result.pass1.each["a"].f0)
    assert "a" in result.errors
    assert any("second pass" in r.getMessage() for r in caplog.records)


def test_failed_tracker_is_skipped_by_default(caplog) -> None:
    """Test that a failed tracker is excluded from voting by default."""
    trackers = [
        Tracker(ConstantAlgorithm, name="a", f0=100.0),
        Tracker(ConstantAlgorithm, name="b", f0=100.0),
        Tracker(FailingAlgorithm, name="c"),
    ]
    with caplog.at_level(logging.WARNING, logger="pitch_garden"):
        with Pipeline(trackers, num_workers=1) as pipeline:
            result = pipeline(np.zeros(1600), 16000)
    assert list(result.each) == ["a", "b"]
    assert "Always failed" in result.errors["c"]
    assert result.pitch is not None
    np.testing.assert_allclose(result.pitch.f0, 100.0)
    assert any("'c'" in r.getMessage() for r in caplog.records)


def test_failure_is_raised_if_requested() -> None:
    """Test that a failure is raised with on_error='raise'."""
    trackers = [Tracker(ConstantAlgorithm, name="a"), Tracker(FailingAlgorithm)]
    with Pipeline(trackers, num_workers=1, on_error="raise") as pipeline:
        with pytest.raises(RuntimeError, match="Always failed"):
            pipeline(np.zeros(1600), 16000)


def test_all_failures_are_raised() -> None:
    """Test that the failure of all trackers is raised even with on_error='skip'."""
    with Pipeline([Tracker(FailingAlgorithm)], num_workers=1) as pipeline:
        with pytest.raises(RuntimeError, match="All trackers failed"):
            pipeline(np.zeros(1600), 16000)


def test_failure_in_worker_is_skipped() -> None:
    """Test that a failure in a worker process is also skipped."""
    trackers = [
        Tracker(ConstantAlgorithm, name="a", f0=100.0),
        Tracker(FailingAlgorithm, name="c"),
    ]
    with Pipeline(trackers, num_workers=2) as pipeline:
        result = pipeline(np.zeros(1600), 16000)
    assert list(result.each) == ["a"]
    assert "c" in result.errors


def test_invalid_on_error() -> None:
    """Test that an unknown option is rejected."""
    with pytest.raises(ValueError, match="on_error"):
        Pipeline([Tracker(ConstantAlgorithm)], on_error="ignore")  # type: ignore


def test_too_short_audio_in_pipeline() -> None:
    """Test that the trackers failing on too short audio are skipped."""
    pytest.importorskip("swift_f0")
    pytest.importorskip("pitch_garden_sptk")
    names = sorted(ALGORITHM_MAP)
    with Pipeline([create_tracker(n) for n in names], num_workers=1) as pipeline:
        result = pipeline(np.random.default_rng(0).normal(0, 0.1, 10), 16000)
    assert result.pitch is not None and result.pitch.length == 1
    assert sorted([*result.each, *result.errors]) == names
    assert "rapt" in result.errors


def test_two_pass_reruns_only_range_capable_trackers() -> None:
    """Test that only the trackers supporting the search range are rerun."""
    trackers = [
        Tracker(ConstantAlgorithm, name="a", f0=200.0),
        Tracker(RangelessAlgorithm, name="b", f0=200.0),
        Tracker(RangelessAlgorithm, name="c", f0=1000.0),
    ]
    two_pass = TwoPass(
        estimator=MedianScaleRangeEstimator(),
        fmin=40.0,
        fmax=2000.0,
    )
    ConstantAlgorithm.num_calls = 0
    RangelessAlgorithm.num_calls = 0
    with Pipeline(trackers, num_workers=1, two_pass=two_pass) as pipeline:
        result = pipeline(np.zeros(16000), 16000)
    assert ConstantAlgorithm.num_calls == 2
    assert RangelessAlgorithm.num_calls == 2  # Two trackers in the first pass only.

    f0_range, first = result.f0_range, result.pass1
    assert f0_range is not None and first is not None
    assert (f0_range.fmin, f0_range.fmax) == pytest.approx((150.0, 300.0))
    # The out-of-range contour is clipped.
    assert not np.any(result.each["c"].voiced)
    assert np.all(first.each["c"].voiced)
    assert result.pitch is not None
    np.testing.assert_allclose(result.pitch.f0, 200.0)


def test_two_pass_fallback_skips_rerun() -> None:
    """Test that the trackers are not rerun if the first-pass range is kept."""
    trackers = [
        Tracker(ConstantAlgorithm, name="a", score=0.0),
        Tracker(RangelessAlgorithm, name="c", f0=3000.0),
    ]
    two_pass = TwoPass(
        estimator=MedianScaleRangeEstimator(),
        fmin=40.0,
        fmax=2000.0,
    )
    ConstantAlgorithm.num_calls = 0
    with Pipeline(trackers, num_workers=1, two_pass=two_pass) as pipeline:
        result = pipeline(np.zeros(16000), 16000)
    assert ConstantAlgorithm.num_calls == 1
    first = result.pass1
    assert result.f0_range is not None and result.f0_range.fallback
    assert first is not None
    np.testing.assert_array_equal(result.each["a"].f0, first.each["a"].f0)
    # The rangeless tracker is still clipped to the first-pass range.
    assert np.all(first.each["c"].voiced)
    assert not np.any(result.each["c"].voiced)


def test_two_pass_does_not_reload_algorithms() -> None:
    """Test that the algorithms are loaded once over multiple calls."""
    trackers = [Tracker(ConstantAlgorithm, name="a", f0=200.0)]
    two_pass = TwoPass(estimator=MedianScaleRangeEstimator())
    ConstantAlgorithm.num_instances = 0
    with Pipeline(trackers, num_workers=1, two_pass=two_pass) as pipeline:
        for _ in range(3):
            pipeline(np.zeros(16000), 16000)
    assert ConstantAlgorithm.num_instances == 1


def test_two_pass_without_clipping() -> None:
    """Test that the out-of-range contour is kept without clipping."""
    trackers = [
        Tracker(ConstantAlgorithm, name="a", f0=200.0),
        Tracker(RangelessAlgorithm, name="c", f0=1000.0),
    ]
    two_pass = TwoPass(
        estimator=MedianScaleRangeEstimator(),
        range_from="a",
        clip_out_of_range=False,
    )
    with Pipeline(trackers, num_workers=1, two_pass=two_pass) as pipeline:
        result = pipeline(np.zeros(16000), 16000)
    assert np.all(result.each["c"].voiced)


def test_two_pass_invalid_range_from() -> None:
    """Test that an unknown tracker in range_from is rejected."""
    with pytest.raises(ValueError):
        Pipeline([Tracker(ConstantAlgorithm)], two_pass=TwoPass(range_from="unknown"))
    trackers = [Tracker(ConstantAlgorithm, name="a"), Tracker(ConstantAlgorithm)]
    with pytest.raises(ValueError):
        Pipeline(trackers, voter=None, two_pass=TwoPass())


def test_two_pass_range_from_tracker_named_pitch() -> None:
    """Test that a tracker named 'pitch' is distinguished from the final contour."""
    trackers = [
        Tracker(ConstantAlgorithm, name="pitch", f0=200.0),
        Tracker(ConstantAlgorithm, name="b", f0=800.0),
    ]
    two_pass = TwoPass(estimator=MedianScaleRangeEstimator(), range_from="pitch")
    with Pipeline(trackers, num_workers=1, two_pass=two_pass) as pipeline:
        result = pipeline(np.zeros(16000), 16000)
    assert result.f0_range is not None
    assert (result.f0_range.fmin, result.f0_range.fmax) == pytest.approx((150, 300))


def test_two_pass_with_single_tracker_without_voter() -> None:
    """Test that a single tracker can be run in two passes without a voter."""
    two_pass = TwoPass(estimator=MedianScaleRangeEstimator())
    tracker = Tracker(ConstantAlgorithm, name="a", f0=200.0)
    with Pipeline([tracker], voter=None, two_pass=two_pass) as pipeline:
        result = pipeline(np.zeros(16000), 16000)
    assert result.f0_range is not None
    assert (result.f0_range.fmin, result.f0_range.fmax) == pytest.approx((150, 300))
    assert result.pitch is result.each["a"]


@pytest.mark.parametrize(
    "voicing_from, expected",
    [("1st", [200.0, 400.0, 200.0, 400.0]), ("2nd", [200.0, 0.0, 200.0, 0.0])],
)
def test_two_pass_voicing_from(
    voicing_from: VoicingSource, expected: list[float]
) -> None:
    """Test that the voicing decision can be taken from the first pass."""
    two_pass = TwoPass(
        estimator=MedianScaleRangeEstimator(),
        fmin=40.0,
        fmax=2000.0,
    )
    policy = TwoPassPolicy(enabled=True, voicing_from=voicing_from)
    tracker = Tracker(RangeSensitiveAlgorithm, name="a", two_pass_policy=policy)
    with Pipeline([tracker], two_pass=two_pass) as pipeline:
        result = pipeline(np.zeros(16000), 16000, frame_period=5.0)
    assert result.pitch is not None
    np.testing.assert_allclose(result.pitch.f0[:4], expected)


def test_invalid_two_pass_policy() -> None:
    """Test that an unknown pass of the voicing is rejected."""
    with pytest.raises(ValueError, match="voicing_from"):
        TwoPassPolicy(voicing_from="both")  # type: ignore


def test_recommended_voicing() -> None:
    """Test that the recommended voicing of each algorithm is used by default."""
    two_pass = TwoPass(estimator=MedianScaleRangeEstimator())
    tracker = Tracker(FirstVoicingAlgorithm, name="a")
    with Pipeline([tracker], two_pass=two_pass) as pipeline:
        result = pipeline(np.zeros(16000), 16000, frame_period=5.0)
    assert result.pitch is not None
    np.testing.assert_allclose(result.pitch.f0[:4], [200.0, 400.0, 200.0, 400.0])


def test_policy_per_tracker() -> None:
    """Test that the policy can be given for each tracker."""
    trackers = [
        Tracker(RangeSensitiveAlgorithm, name="a"),
        Tracker(
            RangeSensitiveAlgorithm,
            name="b",
            two_pass_policy=TwoPassPolicy(enabled=True, voicing_from="1st"),
        ),
    ]
    two_pass = TwoPass(estimator=MedianScaleRangeEstimator(), range_from="a")
    with Pipeline(trackers, voter=None, two_pass=two_pass) as pipeline:
        result = pipeline(np.zeros(16000), 16000, frame_period=5.0)
    np.testing.assert_allclose(result.each["a"].f0[:4], [200.0, 0.0, 200.0, 0.0])
    np.testing.assert_allclose(result.each["b"].f0[:4], [200.0, 400.0, 200.0, 400.0])


def test_no_rerun_if_not_recommended() -> None:
    """Test that the trackers not recommending the two-pass tracking are kept."""
    two_pass = TwoPass(estimator=MedianScaleRangeEstimator())
    tracker = Tracker(NoTwoPassAlgorithm, name="a", f0=200.0)
    NoTwoPassAlgorithm.num_calls = 0
    with Pipeline([tracker], two_pass=two_pass) as pipeline:
        result = pipeline(np.zeros(16000), 16000)
    assert NoTwoPassAlgorithm.num_calls == 1
    assert result.pass1 is not None
    np.testing.assert_array_equal(result.each["a"].f0, result.pass1.each["a"].f0)

    tracker = tracker.replace(two_pass_policy=TwoPassPolicy(enabled=True))
    NoTwoPassAlgorithm.num_calls = 0
    with Pipeline([tracker], two_pass=two_pass) as pipeline:
        pipeline(np.zeros(16000), 16000)
    assert NoTwoPassAlgorithm.num_calls == 2


def test_trackers_not_rerun_use_first_pass_range() -> None:
    """Test that the first-pass range is given also to the trackers not rerun."""
    two_pass = TwoPass(
        estimator=MedianScaleRangeEstimator(),
        fmin=40.0,
        fmax=2000.0,
    )
    # 100 Hz is out of the own range of the tracker but in the first-pass range.
    tracker = Tracker(NoTwoPassAlgorithm, name="a", f0=100.0, fmin=150.0)
    with Pipeline([tracker], two_pass=two_pass) as pipeline:
        result = pipeline(np.zeros(16000), 16000)
    assert result.pass1 is not None
    assert np.all(result.pass1.each["a"].voiced)
    assert np.all(result.each["a"].voiced)
