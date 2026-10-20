"""Utilities for reporting the results in the README files of the recipes."""

import re

# The columns of the results tables: (key, header, scale, number of decimals).
COLUMNS = [
    ("rpa", "RPA [%]", 100, 1),
    ("rca", "RCA [%]", 100, 1),
    ("vde", "VDE [%]", 100, 1),
    ("oa", "OA [%]", 100, 1),
]


def format_values(metrics: dict[str, float]) -> str:
    """Format the metrics as the cells of a Markdown table.

    Parameters
    ----------
    metrics : dict[str, float]
        The metrics keyed by their names.

    Returns
    -------
    out : str
        The cells separated by vertical bars.

    """
    return " | ".join(
        f"{metrics[key] * scale:.{digits}f}" for key, _, scale, digits in COLUMNS
    )


def format_table(
    results: dict[str, dict[str, float]], sort_by: str | None = None
) -> str:
    """Format the results as a Markdown table.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        The metrics keyed by the tracker names.

    sort_by : str or None
        The metric by which the trackers are sorted in descending order, where the
        ties are sorted by name. If None, the trackers are sorted by name.

    Returns
    -------
    out : str
        The Markdown table.

    """
    header = "| Tracker | " + " | ".join(c[1] for c in COLUMNS) + " |"
    align = "| :--- | " + " | ".join("---:" for _ in COLUMNS) + " |"
    rows = [header, align]
    names = sorted(results)
    if sort_by is not None:
        names.sort(key=lambda name: -results[name][sort_by])
    for name in names:
        rows.append(f"| `{name}` | " + format_values(results[name]) + " |")
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
