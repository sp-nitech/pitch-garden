"""Summarize the best setting of each tracker averaged over the splits."""

import argparse
import glob
import os

import numpy as np
from score import evaluate_tracker, format_comparison_table

from utils.report import update_readme


def best_settings(
    results: dict[str, dict[tuple[str, str], dict[str, float]]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Choose the best setting of each tracker by OA averaged over the splits.

    Parameters
    ----------
    results : dict[str, dict[tuple[str, str], dict[str, float]]]
        The metrics keyed by the splits and then by the pairs of the tracker name
        and the setting.

    Returns
    -------
    out : dict[tuple[str, str], dict[str, float]]
        The metrics averaged over the splits of the best setting of each tracker,
        which are sorted in descending order of OA and then by name.

    """
    pairs = sorted({pair for split in results.values() for pair in split})
    averaged = {
        pair: {
            key: float(np.mean([split[pair][key] for split in results.values()]))
            for key in next(iter(results.values()))[pair]
        }
        for pair in pairs
    }
    best: dict[str, tuple[str, str]] = {}
    for name, setting in pairs:
        current = best.get(name)
        if current is None or averaged[current]["oa"] < averaged[(name, setting)]["oa"]:
            best[name] = (name, setting)
    order = sorted(best.values(), key=lambda pair: (-averaged[pair]["oa"], pair[0]))
    return {pair: averaged[pair] for pair in order}


def main() -> None:
    """Summarize the trackers and update the README file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database_dir", help="The directory of the eval set.")
    parser.add_argument(
        "exp_dir", help="The directory of '<split>/<setting>/trackers/<tracker>'."
    )
    parser.add_argument(
        "--splits", nargs="+", required=True, help="The splits to be averaged."
    )
    parser.add_argument(
        "--settings", nargs="+", required=True, help="The settings to be compared."
    )
    parser.add_argument("--readme", help="The README file to be updated.")
    parser.add_argument("--section", help="The name of the section of the README.")
    args = parser.parse_args()

    results: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for split in args.splits:
        results[split] = {}
        for setting in args.settings:
            pattern = os.path.join(args.exp_dir, split, setting, "trackers", "*")
            for tracker_dir in sorted(glob.glob(pattern)):
                name = os.path.basename(tracker_dir)
                results[split][(name, setting)] = evaluate_tracker(
                    args.database_dir, split, tracker_dir
                )

    table = format_comparison_table(best_settings(results))
    print(table)
    if args.readme is not None and args.section is not None:
        update_readme(args.readme, args.section, table)


if __name__ == "__main__":
    main()
