"""Tests for the registry of the voters."""

import pytest

from pitch_garden.voters import VOTER_MAP, create_voter
from pitch_garden.voters.median import MedianVoter


def test_registered() -> None:
    """Test that the median voter is registered."""
    assert VOTER_MAP["median"] is MedianVoter
    assert isinstance(create_voter("median", filter_length=1), MedianVoter)
    with pytest.raises(ValueError):
        create_voter("unknown")
