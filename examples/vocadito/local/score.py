"""Score the F0 contours extracted from Vocadito against the reference."""

import argparse
import glob
import os
import re

import numpy as np

from pitch_garden.metrics import evaluate
from pitch_garden.utils.grid import resample

BEGIN_MARKER = "<!-- results:begin -->"
END_MARKER = "<!-- results:end -->"

# The columns of the results table: (key, header, scale, number of decimals).
COLUMNS = [
    ("rpa", "RPA [%]", 100, 1),
    ("rca", "RCA [%]", 100, 1),
    ("vde", "VDE [%]", 100, 1),
    ("oa", "OA [%]", 100, 1),
]


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


def load_estimate(path: str, times: np.ndarray) -> np.ndarray:
    """Load an estimated F0 file and resample it onto the reference time stamps.

    Parameters
    ----------
    path : str
        The path to the NPZ file saved by pitch-garden in the 'linear' format.

    times : np.ndarray
        The evenly spaced time stamps of the reference in seconds.

    Returns
    -------
    out : np.ndarray
        The estimated F0 in Hz on the time stamps of the reference.

    """
    with np.load(path) as data:
        f0 = data["f0"]
        frame_period = float(data["frame_period"])
    est_times = np.arange(len(f0)) * frame_period / 1000
    step = times[1] - times[0] if 1 < len(times) else frame_period / 1000
    grid = resample(est_times, f0, num_frames=len(times), frame_period=1000 * step)
    # The frames beyond the end of the estimate are regarded as unvoiced.
    return np.where(times <= est_times[-1] + 1e-9, grid.f0, 0.0)


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
        estimates.append(load_estimate(est_path, times))
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


def format_table(results: dict[str, dict[str, float]]) -> str:
    """Format the results as a Markdown table.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        The metrics keyed by the tracker names.

    Returns
    -------
    out : str
        The Markdown table, where the trackers are sorted by name.

    """
    header = "| Tracker | " + " | ".join(c[1] for c in COLUMNS) + " |"
    align = "| :--- | " + " | ".join("---:" for _ in COLUMNS) + " |"
    rows = [header, align]
    for name in sorted(results):
        metrics = results[name]
        values = [
            f"{metrics[key] * scale:.{digits}f}" for key, _, scale, digits in COLUMNS
        ]
        rows.append(f"| `{name}` | " + " | ".join(values) + " |")
    return "\n".join(rows)


def update_readme(path: str, table: str) -> None:
    """Replace the results section of a README file.

    Parameters
    ----------
    path : str
        The path to the README file.

    table : str
        The new contents of the results section.

    Raises
    ------
    ValueError
        If the markers of the results section are not found.

    """
    with open(path) as f:
        text = f.read()
    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r"\n.*?" + re.escape(END_MARKER), re.S
    )
    if pattern.search(text) is None:
        raise ValueError(f"{BEGIN_MARKER} and {END_MARKER} are not found in {path}.")
    # The table is surrounded by blank lines to be stable under mdformat.
    replacement = f"{BEGIN_MARKER}\n\n{table}\n\n{END_MARKER}"
    with open(path, "w") as f:
        f.write(pattern.sub(lambda _: replacement, text))


def main() -> None:
    """Score all trackers and update the README file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", help="The directory of the reference files.")
    parser.add_argument("exp_dir", help="The directory of 'trackers/<tracker>/*.npz'.")
    parser.add_argument("--readme", help="The README file to be updated.")
    args = parser.parse_args()

    results = collect_results(args.reference_dir, args.exp_dir)
    table = format_table(results)
    print(table)
    if args.readme is not None:
        update_readme(args.readme, table)


if __name__ == "__main__":
    main()
