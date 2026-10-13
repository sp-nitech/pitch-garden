"""Tests for the utilities for the salience of the pitch bins."""

import numpy as np
import pytest

from pitch_garden.algorithms._salience import (
    NUM_BINS,
    bins_to_cents,
    decode_weighted_argmax,
    viterbi_path,
)


def test_weighted_argmax_averages_neighbors() -> None:
    """Test that the cents are averaged around the argmax with the probabilities."""
    probabilities = np.zeros((2, NUM_BINS))
    probabilities[0, 99:102] = [0.5, 1.0, 0.5]  # Symmetric around bin 100.
    probabilities[1, 100:102] = [1.0, 1.0]  # Between bins 100 and 101.
    cents, bins = decode_weighted_argmax(probabilities)
    np.testing.assert_array_equal(bins, [100, 100])
    np.testing.assert_allclose(cents, [bins_to_cents(100), bins_to_cents(100.5)])


def full_viterbi(log_prob: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """Decode the most likely path with the full transition matrix."""
    num_frames, num_states = log_prob.shape
    value = log_prob[0] - np.log(num_states)
    pointer = np.zeros((num_frames, num_states), dtype=np.int64)
    for t in range(1, num_frames):
        scores = value[:, None] + log_trans
        pointer[t] = np.argmax(scores, axis=0)
        value = scores[pointer[t], np.arange(num_states)] + log_prob[t]
    path = np.zeros(num_frames, dtype=np.int64)
    path[-1] = np.argmax(value)
    for t in range(num_frames - 1, 0, -1):
        path[t - 1] = pointer[t, path[t]]
    return path


@pytest.mark.parametrize(("num_bins", "max_jump"), [(360, 11), (100, 3)])
def test_viterbi_matches_full_transition(num_bins: int, max_jump: int) -> None:
    """Test that the banded Viterbi decoding matches the full one."""
    rng = np.random.default_rng(0)
    scores = rng.random((50, num_bins))
    scores[:, :30] = -np.inf  # Out of the search range.
    path = viterbi_path(scores, max_jump)

    tiny = np.finfo(np.float64).tiny
    emission = np.exp(scores - scores.max(axis=1, keepdims=True))
    emission /= emission.sum(axis=1, keepdims=True)
    i, j = np.meshgrid(np.arange(num_bins), np.arange(num_bins), indexing="ij")
    transition = np.maximum(max_jump + 1 - np.abs(i - j), 0).astype(np.float64)
    # All rows are normalized as the inner ones not to favor the edge bins.
    transition /= transition[num_bins // 2].sum()
    expected = full_viterbi(np.log(emission + tiny), np.log(transition + tiny))
    np.testing.assert_array_equal(path, expected)
