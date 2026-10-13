"""Tests for the conversions of the units of frequency."""

import numpy as np

from pitch_garden.utils.units import (
    cents_to_hz,
    hz_to_cents,
    hz_to_mel,
    hz_to_midi,
    mel_to_hz,
    midi_to_hz,
)


def test_cents() -> None:
    """Test the conversion between Hz and cents relative to a reference."""
    np.testing.assert_allclose(
        hz_to_cents(np.array([10.0, 20.0, 5.0]), reference=10.0), [0, 1200, -1200]
    )
    np.testing.assert_allclose(hz_to_cents(440.0, reference=220.0), 1200)
    np.testing.assert_allclose(
        cents_to_hz(hz_to_cents(123.4, reference=10.0), reference=10.0), 123.4
    )
    np.testing.assert_allclose(cents_to_hz(-1200, reference=220.0), 110.0)


def test_midi() -> None:
    """Test the conversion between Hz and MIDI note numbers."""
    np.testing.assert_allclose(hz_to_midi(np.array([440.0, 880.0])), [69, 81])
    np.testing.assert_allclose(midi_to_hz(60), 261.6255653005986)
    np.testing.assert_allclose(midi_to_hz(hz_to_midi(123.4)), 123.4)


def test_mel() -> None:
    """Test the conversion between Hz and mels in the HTK formula."""
    np.testing.assert_allclose(
        hz_to_mel(np.array([0.0, 700.0])), [0, 2595 * np.log10(2)]
    )
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(123.4)), 123.4)


def test_mel_slaney() -> None:
    """Test the conversion between Hz and mels in the Slaney formula.

    The expected values are given by librosa 1.0.0 with `librosa.hz_to_mel`.

    """
    np.testing.assert_allclose(
        hz_to_mel(np.array([0.0, 500.0, 1000.0, 6400.0]), htk=False),
        [0.0, 7.5, 15.0, 42.0],
    )
    np.testing.assert_allclose(
        mel_to_hz(hz_to_mel(np.array([123.4, 4321.0]), htk=False), htk=False),
        [123.4, 4321.0],
    )
