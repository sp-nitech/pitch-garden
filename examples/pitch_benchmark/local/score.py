"""Score the F0 contours extracted from the eval set of pitch-benchmark.

The metrics are computed on the trusted frames pooled within each corpus, and then
averaged over the corpora of the speech, optionally with those of the singing voice.
"""

import argparse
import glob
import os
import re
from collections.abc import Sequence

import numpy as np

from pitch_garden.metrics import evaluate
from pitch_garden.utils.grid import resample

# The corpora of the speech and the singing voice, where those of musical
# instruments are excluded.
SPEECH_CORPORA = ("APLAWD", "AVID", "FDA", "KEELE", "OSFGlottis", "SVD", "SpeechSynth")
SINGING_CORPORA = ("Vocadito",)
CORPORA = SPEECH_CORPORA + SINGING_CORPORA

# The frame period of the labels, i.e., 256 samples at 16 kHz.
LABEL_FRAME_PERIOD = 16.0

# The columns of the results table: (key, header, scale, number of decimals).
COLUMNS = [
    ("rpa", "RPA [%]", 100, 1),
    ("rca", "RCA [%]", 100, 1),
    ("vde", "VDE [%]", 100, 1),
    ("oa", "OA [%]", 100, 1),
]


def load_label(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a label file of pitch-benchmark.

    Parameters
    ----------
    path : str
        The path to the NPZ file of the F0 in Hz (unvoiced: 0) and the trusted mask.

    Returns
    -------
    out : tuple[np.ndarray, np.ndarray]
        The F0 values in Hz and the boolean mask of the trusted frames.

    """
    with np.load(path) as data:
        return data["pitch"].astype(np.float64), data["trusted"].astype(bool)


def load_estimate(path: str, num_frames: int) -> np.ndarray:
    """Load an estimated F0 file and put it on the frame grid of the labels.

    Parameters
    ----------
    path : str
        The path to the NPZ file saved by pitch-garden in the 'linear' format.

    num_frames : int
        The number of frames of the label.

    Returns
    -------
    out : np.ndarray
        The estimated F0 in Hz on the frame grid of the label.

    """
    with np.load(path) as data:
        f0 = data["f0"]
        frame_period = float(data["frame_period"])
    est_times = np.arange(len(f0)) * frame_period / 1000
    grid = resample(
        est_times, f0, num_frames=num_frames, frame_period=LABEL_FRAME_PERIOD
    )
    # The frames beyond the end of the estimate are regarded as unvoiced.
    times = np.arange(num_frames) * LABEL_FRAME_PERIOD / 1000
    return np.where(times <= est_times[-1] + 1e-9, grid.f0, 0.0)


def evaluate_corpus(
    database_dir: str, corpus: str, split: str, estimate_dir: str
) -> dict[str, float]:
    """Evaluate a tracker on a corpus.

    Parameters
    ----------
    database_dir : str
        The directory of the eval set containing '<corpus>/<split>/labels'.

    corpus : str
        The name of the corpus.

    split : str
        The name of the split.

    estimate_dir : str
        The directory of the estimated files named '<corpus>/<clip>.npz'.

    Returns
    -------
    out : dict[str, float]
        The metrics computed on the trusted frames pooled within the corpus.

    Raises
    ------
    FileNotFoundError
        If an estimated file is missing.

    """
    references, estimates, masks = [], [], []
    pattern = os.path.join(database_dir, corpus, split, "labels", "*.npz")
    for path in sorted(glob.glob(pattern)):
        name = os.path.splitext(os.path.basename(path))[0]
        est_path = os.path.join(estimate_dir, corpus, f"{name}.npz")
        if not os.path.isfile(est_path):
            raise FileNotFoundError(f"The estimate is not found: {est_path}")
        ref, trusted = load_label(path)
        est = load_estimate(est_path, len(ref))
        references.append(ref)
        estimates.append(est)
        masks.append(trusted)
    return evaluate(
        np.concatenate(references),
        np.concatenate(estimates),
        mask=np.concatenate(masks),
    )


def macro_average(results: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average the metrics over corpora.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        The metrics keyed by the corpus names.

    Returns
    -------
    out : dict[str, float]
        The metrics averaged over the corpora.

    """
    keys = next(iter(results.values())).keys()
    return {k: float(np.mean([r[k] for r in results.values()])) for k in keys}


def evaluate_tracker(
    database_dir: str,
    split: str,
    estimate_dir: str,
    corpora: Sequence[str] = CORPORA,
) -> dict[str, float]:
    """Evaluate a tracker on the corpora.

    Parameters
    ----------
    database_dir : str
        The directory of the eval set.

    split : str
        The name of the split.

    estimate_dir : str
        The directory of the estimated files named '<corpus>/<clip>.npz'.

    corpora : Sequence[str], optional
        The corpora to be evaluated, where those without the split are skipped.

    Returns
    -------
    out : dict[str, float]
        The metrics averaged over the corpora.

    """
    results = {}
    for corpus in corpora:
        if not glob.glob(os.path.join(database_dir, corpus, split, "labels", "*")):
            continue
        results[corpus] = evaluate_corpus(database_dir, corpus, split, estimate_dir)
    return macro_average(results)


def _format_values(metrics: dict[str, float]) -> str:
    """Format the metrics as the cells of a Markdown table."""
    return " | ".join(
        f"{metrics[key] * scale:.{digits}f}" for key, _, scale, digits in COLUMNS
    )


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
        rows.append(f"| `{name}` | " + _format_values(results[name]) + " |")
    return "\n".join(rows)


def format_comparison_table(results: dict[tuple[str, str], dict[str, float]]) -> str:
    """Format the comparison of the settings as a Markdown table.

    Parameters
    ----------
    results : dict[tuple[str, str], dict[str, float]]
        The metrics keyed by the pairs of the tracker name and the setting.

    Returns
    -------
    out : str
        The Markdown table in the given order.

    """
    header = "| Tracker | Setting | " + " | ".join(c[1] for c in COLUMNS) + " |"
    align = "| :--- | :--- | " + " | ".join("---:" for _ in COLUMNS) + " |"
    rows = [header, align]
    for (name, setting), metrics in results.items():
        rows.append(f"| `{name}` | {setting} | " + _format_values(metrics) + " |")
    return "\n".join(rows)


def update_readme(path: str, name: str, table: str) -> None:
    """Replace a named section of a README file.

    Parameters
    ----------
    path : str
        The path to the README file.

    name : str
        The name of the section between '<!-- name:begin -->' and '<!-- name:end -->'.

    table : str
        The new contents of the section.

    Raises
    ------
    ValueError
        If the markers of the section are not found.

    """
    begin, end = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    with open(path) as f:
        text = f.read()
    pattern = re.compile(re.escape(begin) + r"\n.*?" + re.escape(end), re.S)
    if pattern.search(text) is None:
        raise ValueError(f"{begin} and {end} are not found in {path}.")
    # The table is surrounded by blank lines to be stable under mdformat.
    replacement = f"{begin}\n\n{table}\n\n{end}"
    with open(path, "w") as f:
        f.write(pattern.sub(lambda _: replacement, text))


def main() -> None:
    """Score the trackers under the settings and update the README file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_dir", help="The directory of the eval set.")
    parser.add_argument(
        "exp_dir", help="The directory of '<setting>/trackers/<tracker>/<corpus>'."
    )
    parser.add_argument("--split", required=True, help="The split to be scored.")
    parser.add_argument(
        "--settings", nargs="+", required=True, help="The settings to be scored."
    )
    parser.add_argument(
        "--speech_only", action="store_true", help="Exclude the singing voice."
    )
    parser.add_argument("--readme", help="The README file to be updated.")
    parser.add_argument("--section", help="The name of the section of the README.")
    args = parser.parse_args()

    corpora = SPEECH_CORPORA if args.speech_only else CORPORA
    results = {}
    for setting in args.settings:
        pattern = os.path.join(args.exp_dir, setting, "trackers", "*")
        for tracker_dir in sorted(glob.glob(pattern)):
            name = os.path.basename(tracker_dir)
            results[(name, setting)] = evaluate_tracker(
                args.database_dir, args.split, tracker_dir, corpora
            )

    if len(args.settings) == 1:
        table = format_table({name: r for (name, _), r in results.items()})
    else:
        order = sorted(results, key=lambda k: (k[0], args.settings.index(k[1])))
        table = format_comparison_table({k: results[k] for k in order})
    print(table)
    if args.readme is not None and args.section is not None:
        update_readme(args.readme, args.section, table)


if __name__ == "__main__":
    main()
