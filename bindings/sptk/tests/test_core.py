"""Tests for the binding of the pitch extraction of SPTK."""

import os
import shutil
import subprocess

import numpy as np
import pytest
from pitch_garden_sptk import ALGORITHMS, extract_f0

# The SPTK commands built by `make sptk`.
SPTK_BIN = os.environ.get(
    "SPTK_BIN", os.path.join(os.path.dirname(__file__), "..", "build", "sptk")
)


def generate_tone(f0: float, sample_rate: int = 16000, duration: float = 1.0):
    """Generate a harmonic tone in the 16-bit scale."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 11))
    return 10000 * x / np.max(np.abs(x))


def test_invalid_arguments() -> None:
    """Test that invalid arguments are rejected."""
    x = generate_tone(220.0)
    with pytest.raises(ValueError, match="algorithm"):
        extract_f0(x, 16000, 80, 60.0, 400.0, 0.0, "yin")
    with pytest.raises(ValueError):
        extract_f0(x, 16000, 80, 400.0, 60.0, 0.0, "rapt")
    with pytest.raises(ValueError):
        extract_f0(x, 16000, 80, 60.0, 9000.0, 0.0, "rapt")


@pytest.mark.parametrize("index, algorithm", list(enumerate(ALGORITHMS)))
def test_compatibility_with_sptk(index: int, algorithm: str) -> None:
    """Test that the result is identical to the pitch command of SPTK."""
    pitch = os.path.join(SPTK_BIN, "pitch")
    if shutil.which(pitch) is None:
        pytest.skip("SPTK is not found.")

    rng = np.random.default_rng(0)
    x = np.round(generate_tone(150.0) + 100 * rng.standard_normal(16000))
    threshold = {"rapt": 0.0, "swipe": 0.3, "reaper": 0.9, "dio": 0.1, "harvest": 0.01}
    t = threshold[algorithm]
    command = [pitch, "-s", "16", "-p", "80", "-L", "60", "-H", "400"]
    command += ["-a", str(index), f"-t{index}", str(t), "-o", "1"]
    result = subprocess.run(
        command, input=x.astype(np.float64).tobytes(), capture_output=True, check=True
    )
    expected = np.frombuffer(result.stdout, dtype=np.float64)
    actual = extract_f0(x, 16000, 80, 60.0, 400.0, t, algorithm)
    np.testing.assert_array_equal(actual, expected)
