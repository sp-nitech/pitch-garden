"""A module for processing multiple audio files according to a configuration."""

from __future__ import annotations

import logging
import mimetypes
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

import yaml

from .config import Config, TwoPassSettings, build_pipeline
from .interfaces.pipeline import PipelineResult
from .interfaces.types import Pitch
from .ranges.base import F0Range
from .utils.io import load_audio

logger = logging.getLogger("pitch_garden")

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif")
FORMAT_EXTENSIONS: dict[str, str] = {"linear": "f0", "log": "lf0", "inverse": "pit"}


def collect_input_files(
    source: str, raw_extensions: Sequence[str] = (".raw",)
) -> list[str]:
    """Collect the input audio files.

    Parameters
    ----------
    source : str
        An audio file, a directory searched recursively, or a text file listing
        audio files line by line.

    raw_extensions : Sequence[str], optional
        The extensions of headerless raw audio files.

    Returns
    -------
    out : list[str]
        The paths to the audio files.

    Raises
    ------
    ValueError
        If the source does not exist.

    """
    extensions = AUDIO_EXTENSIONS + tuple(e.lower() for e in raw_extensions)
    if os.path.isfile(source):
        mime_type, _ = mimetypes.guess_type(source)
        is_audio = mime_type is not None and mime_type.startswith("audio/")
        if is_audio or source.lower().endswith(extensions):
            return [source]
        with open(source) as f:
            return [line.strip() for line in f if line.strip()]
    if os.path.isdir(source):
        files = []
        for root, _, names in os.walk(source):
            for name in names:
                if name.lower().endswith(extensions):
                    files.append(os.path.join(root, name))
        return sorted(files)
    raise ValueError(f"Invalid source: {source}")


def _basename(path: str) -> str:
    """Get the basename without the extension."""
    return os.path.splitext(os.path.basename(path))[0]


def group_files(files: list[str], settings: TwoPassSettings) -> dict[str, list[str]]:
    """Group the files into the units of the F0 range estimation.

    Parameters
    ----------
    files : list[str]
        The input files.

    settings : TwoPassSettings
        The settings of the two-pass tracking.

    Returns
    -------
    out : dict[str, list[str]]
        The files keyed by the unit names, in the order of appearance.

    Raises
    ------
    ValueError
        If the map file is not given or a file is not found in the map file.

    """
    if settings.scope == "utterance":
        return {f: [f] for f in files}
    if settings.scope == "global":
        return {"global": list(files)}

    group_by = settings.group_by
    mapping: dict[str, str] = {}
    if group_by.type == "map":
        if group_by.path is None:
            raise ValueError("The map file is required to group files by 'map'.")
        with open(group_by.path) as f:
            for line in f:
                items = line.split()
                if len(items) >= 2:
                    mapping[items[0]] = items[1]

    groups: dict[str, list[str]] = {}
    for file in files:
        if group_by.type == "map":
            try:
                key = mapping[_basename(file)]
            except KeyError as e:
                raise ValueError(
                    f"'{_basename(file)}' is not found in {group_by.path}."
                ) from e
        else:
            parents = Path(file).parents
            key = str(parents[min(group_by.level, len(parents)) - 1])
        groups.setdefault(key, []).append(file)
    return groups


class _Writer:
    """A helper class for writing the results."""

    def __init__(self, config: Config, output_dir: str) -> None:
        self.config = config
        self.output_dir = output_dir
        output = config.output
        if output.file_type == "npz":
            self.ext = "npz"
        else:
            self.ext = FORMAT_EXTENSIONS[output.format]

    def _subdir(self, input_file: str) -> str:
        offset = self.config.subdir_offset
        if offset is None:
            return ""
        path = Path(input_file).parent
        dirs = path.relative_to(path.anchor).parts
        if offset > len(dirs):
            raise ValueError(
                f"Subdir offset {offset} is too large for file: {input_file}."
            )
        return str(Path(*dirs[offset:])) if dirs[offset:] else ""

    def _relative_path(self, input_file: str) -> str:
        return os.path.join(
            self._subdir(input_file), f"{_basename(input_file)}.{self.ext}"
        )

    def check_outputs(self, input_files: Sequence[str]) -> None:
        """Check that the input files are saved to different outputs.

        Raises
        ------
        ValueError
            If multiple input files are saved to the same output.

        """
        seen: dict[str, str] = {}
        for file in input_files:
            path = self._relative_path(file)
            if path in seen:
                raise ValueError(
                    f"{seen[path]} and {file} are saved to the same output '{path}'. "
                    "Give subdir_offset to keep their subdirectories."
                )
            seen[path] = file

    def _path(self, prefix: str, input_file: str) -> str:
        path = os.path.join(self.output_dir, prefix, self._relative_path(input_file))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _save(self, pitch: Pitch, path: str) -> None:
        output = self.config.output
        pitch.tofile(
            path,
            output_format=output.format,
            file_type=output.file_type,
        )

    def write(self, input_file: str, result: PipelineResult, prefix: str = "") -> None:
        output = self.config.output
        if result.pitch is not None:
            path = self._path(os.path.join(prefix, "pitch"), input_file)
            self._save(result.pitch, path)
        # The output of a single tracker is identical to the final contour.
        if len(result.each) > 1 and (output.save_each or result.pitch is None):
            for name, pitch in result.each.items():
                path = self._path(os.path.join(prefix, "trackers", name), input_file)
                self._save(pitch, path)


def _iterate(
    files: Iterable[str], index: dict[str, int], quiet: bool, label: str = ""
) -> Iterable[str]:
    """Iterate over the files while logging the progress over all files."""
    prefix = f"{label}: " if label else ""
    for file in files:
        if not quiet:
            logger.info(f"[{index[file] + 1}/{len(index)}] {prefix}{file}")
        yield file


def run_batch(
    config: Config, source: str, output_dir: str, *, quiet: bool = False
) -> int:
    """Process all input files according to the configuration.

    Parameters
    ----------
    config : Config
        The configuration.

    source : str
        An audio file, a directory searched recursively, or a text file listing
        audio files line by line.

    output_dir : str
        The directory where the results are saved.

    quiet : bool, optional
        If True, suppress the progress output.

    Returns
    -------
    out : int
        The number of files skipped due to errors.

    Raises
    ------
    ValueError
        If the source is invalid or multiple input files are saved to the same
        output.

    """
    raw = config.raw
    # Without the format, raw files are collected and fail to be read as audio.
    files = (
        collect_input_files(source)
        if raw is None
        else collect_input_files(source, raw.extensions)
    )
    logger.info(f"Found {len(files)} audio files to process.")
    index = {file: i for i, file in enumerate(files)}
    writer = _Writer(config, output_dir)
    writer.check_outputs(files)
    frame_period = config.frame_period
    num_errors = 0

    def process(file: str, func) -> PipelineResult | None:
        nonlocal num_errors
        try:
            return func(load_audio(file, raw))
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}. Skipping.")
            num_errors += 1
            return None

    with build_pipeline(config) as pipeline:
        if pipeline.two_pass is None:
            for file in _iterate(files, index, quiet):
                result = process(file, lambda a: pipeline(a, frame_period=frame_period))
                if result is not None:
                    writer.write(file, result)
            return num_errors

        ranges: dict[str, F0Range] = {}
        groups = group_files(files, cast(TwoPassSettings, config.two_pass))
        for key, group in groups.items():
            # The first pass for all files in the unit.
            first: dict[str, PipelineResult] = {}
            for file in _iterate(group, index, quiet, "pass 1"):
                result = process(
                    file, lambda a: pipeline.first_pass(a, frame_period=frame_period)
                )
                if result is not None:
                    first[file] = result
            if not first:
                continue

            f0_range = pipeline.estimate_f0_range(list(first.values()))
            ranges[key] = f0_range
            status = " (fallback)" if f0_range.fallback else ""
            logger.info(
                f"F0 range of {key}: {f0_range.fmin:.1f}-{f0_range.fmax:.1f} Hz{status}"
            )

            # The second pass with the estimated range.
            for file in _iterate(first, index, quiet, "pass 2"):
                first_result = first[file]
                result = process(
                    file,
                    lambda a, r=first_result: pipeline.second_pass(
                        a, r, f0_range, frame_period=frame_period
                    ),
                )
                if result is not None:
                    writer.write(file, result)
                    if config.output.save_each:
                        writer.write(file, first_result, prefix="pass1")

        os.makedirs(output_dir, exist_ok=True)
        summary = {
            key: {**r.to_dict(), "files": groups[key]} for key, r in ranges.items()
        }
        with open(os.path.join(output_dir, "f0_range.yaml"), "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)

    return num_errors
