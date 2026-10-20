"""Score the F0 contours extracted from Vocadito against the reference."""

import argparse
import glob
import os

import numpy as np

from pitch_garden.metrics import evaluate

from utils.estimates import load_estimate
from utils.report import format_table, update_readme


def load_reference(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a reference F0 file of Vocadito.

    Parameters
    ----------
    path : str
        The path to the CSV file of time stamps and F0 values (unvoiced: 0).

    Returns
    -------
    out : tuple[np.ndarray, np.ndarray]
        The time stamps in seconds and the F0 values in Hz.

    """
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    return data[:, 0], data[:, 1]


def evaluate_tracker(reference_dir: str, estimate_dir: str) -> dict[str, float]:
    """Evaluate a tracker over all files with the frames pooled.

    Parameters
    ----------
    reference_dir : str
        The directory of the reference files named '<name>_f0.csv'.

    estimate_dir : str
        The directory of the estimated files named '<name>.npz'.

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
    for path in sorted(glob.glob(os.path.join(reference_dir, "*_f0.csv"))):
        name = os.path.basename(path)[: -len("_f0.csv")]
        est_path = os.path.join(estimate_dir, f"{name}.npz")
        if not os.path.isfile(est_path):
            raise FileNotFoundError(f"The estimate is not found: {est_path}")
        times, f0 = load_reference(path)
        references.append(f0)
        step = times[1] - times[0] if 1 < len(times) else 1.0
        estimates.append(load_estimate(est_path, len(times), 1000 * step))
    return evaluate(np.concatenate(references), np.concatenate(estimates))


def collect_results(reference_dir: str, exp_dir: str) -> dict[str, dict[str, float]]:
    """Evaluate all trackers.

    Parameters
    ----------
    reference_dir : str
        The directory of the reference files.

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
        results[name] = evaluate_tracker(reference_dir, tracker_dir)
    return results


def main() -> None:
    """Score all trackers and update the README file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", help="The directory of the reference files.")
    parser.add_argument("exp_dir", help="The directory of 'trackers/<tracker>/*.npz'.")
    parser.add_argument("--readme", help="The README file to be updated.")
    args = parser.parse_args()

    results = collect_results(args.reference_dir, args.exp_dir)
    table = format_table(results, sort_by="oa")
    print(table)
    if args.readme is not None:
        update_readme(args.readme, "results", table)


if __name__ == "__main__":
    main()
