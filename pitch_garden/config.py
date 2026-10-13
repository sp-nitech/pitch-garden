"""Loading and validating the YAML configuration."""

from __future__ import annotations

import dataclasses
import json
import types
from dataclasses import dataclass, field
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

import yaml

from .algorithms.base import TwoPassPolicy, VoicingSource
from .interfaces.pipeline import OnError, Pipeline, TwoPass
from .interfaces.tracker import Tracker
from .interfaces.types import FileType, OutputFormat
from .ranges import create_range_estimator
from .utils.grid import check_frame_period
from .utils.io import RawFormat
from .voters import create_voter
from .voters.base import BaseVoter


class ConfigError(ValueError):
    """An error in the configuration."""


@dataclass
class OutputSettings:
    """Settings of the output files."""

    format: OutputFormat = "log"
    file_type: FileType = "double"
    sample_rate: int | None = None  # The sample rate for the 'inverse' format.
    save_each: bool = False  # Whether to save the outputs of the trackers.


@dataclass
class TwoPassPolicySettings:
    """Settings of the two-pass tracking for a tracker."""

    enabled: bool = True  # Whether the tracker is rerun in the second pass.
    voicing_from: str = "2nd"  # '1st' or '2nd': the pass giving the voicing.


@dataclass
class TrackerSettings:
    """Settings of a tracker."""

    algorithm: str
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    # None follows the recommended settings of the algorithm.
    two_pass: TwoPassPolicySettings | None = None


@dataclass
class VoterSettings:
    """Settings of the voter."""

    type: str = "median"
    params: dict[str, Any] = field(default_factory=dict)
    state: str | None = None  # The JSON file of the learned parameters.


@dataclass
class EstimatorSettings:
    """Settings of the F0 range estimator."""

    type: str = "log-normal"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupBySettings:
    """Settings of grouping the input files for the 'group' scope."""

    type: Literal["dirname", "map"] = "dirname"
    level: int = 1  # 1: the parent directory, 2: the grandparent directory, ...
    path: str | None = None  # The file of '<basename> <group>' lines for 'map'.


@dataclass
class TwoPassSettings:
    """Settings of the two-pass tracking."""

    fmin: float = 40.0  # The first-pass range in Hz.
    fmax: float = 2000.0
    estimator: EstimatorSettings = field(default_factory=EstimatorSettings)
    source: str = "pitch"  # 'pitch' (the final contour) or the name of a tracker.
    scope: Literal["utterance", "group", "global"] = "utterance"
    group_by: GroupBySettings = field(default_factory=GroupBySettings)
    clip_out_of_range: bool = True


@dataclass
class Config:
    """The whole configuration."""

    trackers: list[TrackerSettings]
    subdir_offset: int | None = None
    frame_period: float = 5.0  # The frame period in milliseconds.
    num_workers: int | None = None
    on_error: OnError = "skip"  # The behavior on a failure of a tracker.
    raw: RawFormat | None = None  # The format of raw files. None reads no raw.
    output: OutputSettings = field(default_factory=OutputSettings)
    voter: VoterSettings | None = field(default_factory=VoterSettings)
    two_pass: TwoPassSettings | None = None  # None disables the two-pass tracking.


def _type_name(tp: Any) -> str:
    """Get a readable name of a type."""
    return getattr(tp, "__name__", str(tp))


def _convert(value: Any, tp: Any, path: str) -> Any:
    """Convert a value to the given type with validation."""
    origin = get_origin(tp)

    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: must not be null.")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _convert(value, arg, path)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(errors[0])

    if origin is Literal:
        choices = get_args(tp)
        if value not in choices:
            raise ConfigError(f"{path}: must be one of {list(choices)}, got {value!r}.")
        return value

    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: must be a list.")
        (item_type,) = get_args(tp)
        return [_convert(v, item_type, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is dict:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: must be a mapping.")
        return dict(value)

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _parse_dataclass(value, tp, path)

    if tp is Any:
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: must be a boolean, got {value!r}.")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: must be an integer, got {value!r}.")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: must be a number, got {value!r}.")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: must be a string, got {value!r}.")
        return value
    raise ConfigError(f"{path}: unsupported type {_type_name(tp)}.")  # pragma: no cover


def _parse_dataclass(data: Any, cls: type, path: str) -> Any:
    """Parse a mapping into a dataclass with validation."""
    prefix = f"{path}." if path else ""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'root'}: must be a mapping.")

    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = [k for k in data if k not in fields]
    if unknown:
        raise ConfigError(
            f"{prefix}{unknown[0]}: unknown key. Valid keys are: {list(fields)}"
        )

    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _convert(data[name], hints[name], f"{prefix}{name}")
        elif (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ):
            raise ConfigError(f"{prefix}{name}: required key is missing.")
    return cls(**kwargs)


def _build_trackers(config: Config) -> list[Tracker]:
    """Build the trackers, reporting the errors with their paths."""
    trackers = []
    for i, t in enumerate(config.trackers):
        policy = None
        if t.two_pass is not None:
            try:
                policy = TwoPassPolicy(
                    enabled=t.two_pass.enabled,
                    voicing_from=cast(VoicingSource, t.two_pass.voicing_from),
                )
            except ValueError as e:
                raise ConfigError(f"trackers[{i}].two_pass: {e}") from e
        try:
            trackers.append(
                Tracker(t.algorithm, name=t.name, two_pass_policy=policy, **t.params)
            )
        except (TypeError, ValueError) as e:
            key = "algorithm" if "Unsupported algorithm" in str(e) else "params"
            raise ConfigError(f"trackers[{i}].{key}: {e}") from e
    return trackers


def _build_voter(config: Config, names: list[str]) -> BaseVoter | None:
    """Build the voter, reporting the errors with its path."""
    if config.voter is None:
        return None
    try:
        voter = create_voter(config.voter.type, **config.voter.params)
        voter.check_trackers(names)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"voter: {e}") from e
    if config.voter.state is not None:
        try:
            with open(config.voter.state) as f:
                voter.load_state_dict(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            raise ConfigError(f"voter.state: {e}") from e
    return voter


def _build_two_pass(config: Config, names: list[str]) -> TwoPass | None:
    """Build the settings of the two-pass tracking, reporting the errors."""
    r = config.two_pass
    if r is None:
        return None
    try:
        estimator = create_range_estimator(r.estimator.type, **r.estimator.params)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"two_pass.estimator: {e}") from e
    try:
        two_pass = TwoPass(
            estimator=estimator,
            source=r.source,
            fmin=r.fmin,
            fmax=r.fmax,
            clip_out_of_range=r.clip_out_of_range,
        )
        two_pass.check_trackers(names, has_voter=config.voter is not None)
    except ValueError as e:
        raise ConfigError(f"two_pass: {e}") from e
    return two_pass


def build_pipeline(config: Config) -> Pipeline:
    """Build the pipeline from the configuration.

    The objects are validated by themselves, and their errors are reported with the
    paths in the configuration.

    Parameters
    ----------
    config : Config
        The configuration.

    Returns
    -------
    out : Pipeline
        The pipeline.

    Raises
    ------
    ConfigError
        If the configuration is invalid.

    """
    trackers = _build_trackers(config)
    names = [t.name for t in trackers]
    voter = _build_voter(config, names)
    two_pass = _build_two_pass(config, names)
    try:
        return Pipeline(
            trackers,
            voter=voter,
            num_workers=config.num_workers,
            two_pass=two_pass,
            on_error=config.on_error,
        )
    except ValueError as e:
        raise ConfigError(f"trackers: {e}") from e


def _validate(config: Config) -> None:
    """Validate the configuration, including the objects built from it."""
    try:
        check_frame_period(config.frame_period)
    except ValueError as e:
        raise ConfigError(f"frame_period: {e}") from e
    if config.num_workers is not None and config.num_workers < 1:
        raise ConfigError("num_workers: must be a positive integer.")
    if config.raw is not None:
        if config.raw.sample_rate < 1:
            raise ConfigError("raw.sample_rate: must be a positive integer.")
        if config.raw.num_channels < 1:
            raise ConfigError("raw.num_channels: must be a positive integer.")
    r = config.two_pass
    if r is not None:
        if r.scope == "group" and r.group_by.type == "map" and not r.group_by.path:
            raise ConfigError("two_pass.group_by.path: required for 'map'.")
        if r.group_by.level < 1:
            raise ConfigError("two_pass.group_by.level: must be positive.")
    build_pipeline(config).close()


def parse_config(data: Any) -> Config:
    """Parse and validate the configuration.

    Parameters
    ----------
    data : Any
        The configuration loaded from YAML.

    Returns
    -------
    out : Config
        The validated configuration.

    Raises
    ------
    ConfigError
        If the configuration is invalid.

    """
    config = _parse_dataclass(data, Config, "")
    _validate(config)
    return config


def load_config(path: str) -> Config:
    """Load and validate the configuration from a YAML file.

    Parameters
    ----------
    path : str
        The path to the YAML file.

    Returns
    -------
    out : Config
        The validated configuration.

    Raises
    ------
    ConfigError
        If the configuration is invalid.

    """
    with open(path) as f:
        data = yaml.safe_load(f)
    try:
        return parse_config(data)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from e
