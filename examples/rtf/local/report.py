"""Tabulate the real-time factors (RTFs) and update the README file."""

import argparse
import glob
import json
import os

from utils.report import update_readme

DEVICES = (("cpu", "CPU"), ("cuda", "GPU"))


def _load_all(exp_dir: str) -> list[dict]:
    """Load all results saved as '<device>/<algorithm>.json'."""
    results = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "*", "*.json"))):
        with open(path) as f:
            results.append(json.load(f))
    return results


def collect_results(exp_dir: str) -> dict[str, dict[str, float]]:
    """Collect the RTFs.

    Parameters
    ----------
    exp_dir : str
        The directory of '<device>/<algorithm>.json'.

    Returns
    -------
    out : dict[str, dict[str, float]]
        The RTFs keyed by the tracker names and then by the devices.

    """
    results: dict[str, dict[str, float]] = {}
    for result in _load_all(exp_dir):
        results.setdefault(result["algorithm"], {})[result["device"]] = result["rtf"]
    return results


def collect_machines(exp_dir: str) -> dict[str, list[str]]:
    """Collect the processors of the measurements.

    Parameters
    ----------
    exp_dir : str
        The directory of '<device>/<algorithm>.json'.

    Returns
    -------
    out : dict[str, list[str]]
        The sorted names of the processors keyed by the devices.

    """
    machines: dict[str, set[str]] = {}
    for result in _load_all(exp_dir):
        machines.setdefault(result["device"], set()).add(result["machine"])
    return {device: sorted(names) for device, names in machines.items()}


def format_table(results: dict[str, dict[str, float]]) -> str:
    """Format the RTFs as a Markdown table.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        The RTFs keyed by the tracker names and then by the devices.

    Returns
    -------
    out : str
        The Markdown table, where the trackers are sorted by name and the missing
        RTFs are dashes.

    """
    header = "| Tracker | " + " | ".join(f"RTF ({h})" for _, h in DEVICES) + " |"
    align = "| :--- | " + " | ".join("---:" for _ in DEVICES) + " |"
    rows = [header, align]
    for name in sorted(results):
        cells = [
            f"{results[name][d]:.4f}" if d in results[name] else "—" for d, _ in DEVICES
        ]
        rows.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    """Tabulate the RTFs and update the README file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", help="The directory of '<device>/*.json'.")
    parser.add_argument("--readme", help="The README file to be updated.")
    args = parser.parse_args()

    table = format_table(collect_results(args.exp_dir))
    machines = collect_machines(args.exp_dir)
    notes = [
        f"- {header}: {', '.join(machines[device])}"
        for device, header in DEVICES
        if device in machines
    ]
    text = table + ("\n\n" + "\n".join(notes) if notes else "")
    print(text)
    if args.readme is not None:
        update_readme(args.readme, "results", text)


if __name__ == "__main__":
    main()
