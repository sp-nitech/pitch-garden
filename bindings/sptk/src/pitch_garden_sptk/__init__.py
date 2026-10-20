"""A Python binding of the pitch extraction of SPTK."""

from ._core import extract_f0  # pyright: ignore[reportMissingModuleSource]

# The algorithms in the order of the -a option of the pitch command of SPTK.
ALGORITHMS = ("rapt", "swipe", "reaper", "dio", "harvest")

# The supported sample rates in Hz, which are round numbers within those accepted
# by the pitch command of SPTK, i.e., 6 kHz < sample rate <= 98 kHz.
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 96000

__all__ = ["ALGORITHMS", "MAX_SAMPLE_RATE", "MIN_SAMPLE_RATE", "extract_f0"]
