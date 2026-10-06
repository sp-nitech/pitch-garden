"""Type stubs of the binding of the pitch extraction of SPTK."""

import numpy as np
import numpy.typing as npt

def extract_f0(
    waveform: npt.NDArray[np.float64],
    sample_rate: float,
    frame_shift: int,
    lower_f0: float,
    upper_f0: float,
    voicing_threshold: float,
    algorithm: str,
) -> npt.NDArray[np.float64]:
    """Extract F0 in Hz, where unvoiced frames are 0."""
