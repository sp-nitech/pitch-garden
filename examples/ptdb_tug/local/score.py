"""Score the F0 contours extracted from PTDB-TUG against the reference."""

import argparse
import glob
import os

import numpy as np

from pitch_garden.metrics import evaluate

from utils.report import format_table, update_readme

# The time of the first reference frame and the frame period in milliseconds, where
# the offset is chosen empirically by comparing the reference with the trackers.
REFERENCE_OFFSET = 16.0
REFERENCE_PERIOD = 10.0


def load_reference(path: str) -> np.ndarray:
    """Load a reference file of PTDB-TUG.

    Parameters
    ----------
    path : str
        The path to the output of RAPT, whose first column is the F0.

    Returns
    -------
    out : np.ndarray
        The F0 values in Hz, where unvoiced frames are 0.

    """
    return np.loadtxt(path, ndmin=2)[:, 0]


def sample_estimate(path: str, num_frames: int) -> np.ndarray:
    """Sample an estimated F0 file at the times of the reference frames.

    Parameters
    ----------
    path : str
        The path to the NPZ file saved by pitch-garden in the 'linear' format.

    num_frames : int
        The number of frames of the reference.

    Returns
    -------
    out : np.ndarray
        The estimated F0 in Hz at the nearest frames, where the frames beyond the
        end of the estimate are unvoiced.

    """
    with np.load(path) as data:
        f0 = np.asarray(data["f0"], dtype=np.float64)
        frame_period = float(data["frame_period"])
    times = REFERENCE_OFFSET + REFERENCE_PERIOD * np.arange(num_frames)
    indices = np.round(times / frame_period).astype(int)
    sampled = np.zeros(num_frames)
    inside = indices < len(f0)
    sampled[inside] = f0[indices[inside]]
    return sampled


def evaluate_tracker(database_dir: str, estimate_dir: str) -> dict[str, float]:
    """Evaluate a tracker over all files with the frames pooled.

    Parameters
    ----------
    database_dir : str
        The directory of 'SPEECH DATA' of PTDB-TUG.

    estimate_dir : str
        The directory of the estimated files named '<speaker>_<sentence>.npz'.

    Returns
    -------
    out : dict[str, float]
        The metrics computed on the pooled frames of all files.

    Raises
    ------
    FileNotFoundError
        If an estimated file is missing.

    """
    references, estimates = [], []
    pattern = os.path.join(database_dir, "*", "REF", "*", "ref_*.f0")
    for path in sorted(glob.glob(pattern)):
        name = os.path.splitext(os.path.basename(path))[0].removeprefix("ref_")
        est_path = os.path.join(estimate_dir, f"{name}.npz")
        if not os.path.isfile(est_path):
            raise FileNotFoundError(f"The estimate is not found: {est_path}")
        reference = load_reference(path)
        references.append(reference)
        estimates.append(sample_estimate(est_path, len(reference)))
    return evaluate(np.concatenate(references), np.concatenate(estimates))


def collect_results(database_dir: str, exp_dir: str) -> dict[str, dict[str, float]]:
    """Evaluate all trackers.

    Parameters
    ----------
    database_dir : str
        The directory of 'SPEECH DATA' of PTDB-TUG.

    exp_dir : str
        The directory of 'trackers/<tracker>/*.npz'.

    Returns
    -------
    out : dict[str, dict[str, float]]
        The metrics keyed by the tracker names.

    """
    results = {}
    for tracker_dir in sorted(glob.glob(os.path.join(exp_dir, "trackers", "*"))):
        name = os.path.basename(tracker_dir)
        results[name] = evaluate_tracker(database_dir, tracker_dir)
    return results


def main() -> None:
    """Score all trackers and update the README file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_dir", help="The directory of 'SPEECH DATA'.")
    parser.add_argument("exp_dir", help="The directory of 'trackers/<tracker>/*.npz'.")
    parser.add_argument("--readme", help="The README file to be updated.")
    args = parser.parse_args()

    results = collect_results(args.database_dir, args.exp_dir)
    table = format_table(results)
    print(table)
    if args.readme is not None:
        update_readme(args.readme, "results", table)


if __name__ == "__main__":
    main()
