"""Tests for the YAML configuration."""

import glob

import pytest
import yaml

from pitch_garden.config import Config, ConfigError, load_config, parse_config


def minimal() -> dict:
    """Create a minimal configuration."""
    return {"trackers": [{"algorithm": "swift-f0"}]}


def test_defaults() -> None:
    """Test that the defaults are filled."""
    config = parse_config(minimal())
    assert isinstance(config, Config)
    assert config.frame_period == 5.0
    assert config.output.format == "log"
    assert config.output.file_type == "double"
    assert config.voter is not None and config.voter.type == "median"
    assert config.two_pass is None
    assert config.trackers[0].params == {}


def test_configs_are_valid() -> None:
    """Test that all configuration files are valid."""
    paths = sorted(glob.glob("configs/*.yaml"))
    assert len(paths) >= 2
    for path in paths:
        load_config(path)


def test_input_is_not_in_config() -> None:
    """Test that the input is given by the command line, not by the file."""
    data = minimal()
    data["input"] = "a.wav"
    with pytest.raises(ConfigError, match="input"):
        parse_config(data)


def test_output_dir_is_not_in_config() -> None:
    """Test that the output directory is given by the command line."""
    data = minimal()
    data["output_dir"] = "exp"
    with pytest.raises(ConfigError, match="output_dir"):
        parse_config(data)


def test_unknown_key_with_path() -> None:
    """Test that an unknown key is reported with its path."""
    data = minimal()
    data["output"] = {"fromat": "linear"}
    with pytest.raises(ConfigError, match=r"output\.fromat"):
        parse_config(data)
    # The pitch period is always in samples of the input.
    data["output"] = {"format": "inverse", "sample_rate": 16000}
    with pytest.raises(ConfigError, match=r"output\.sample_rate"):
        parse_config(data)


def test_unknown_algorithm_parameter_with_path() -> None:
    """Test that an unknown algorithm parameter is reported with its path."""
    data = minimal()
    data["trackers"].append({"algorithm": "swift-f0", "name": "b", "params": {"x": 1}})
    with pytest.raises(ConfigError, match=r"trackers\[1\]\.params"):
        parse_config(data)


def test_device_is_not_global() -> None:
    """Test that the device is given as an algorithm parameter, not globally."""
    data = minimal()
    data["device"] = "cuda"
    with pytest.raises(ConfigError, match="device"):
        parse_config(data)
    data = minimal()
    data["trackers"][0]["device"] = "cuda"
    with pytest.raises(ConfigError, match=r"trackers\[0\]\.device"):
        parse_config(data)


def test_raw_settings() -> None:
    """Test the settings of raw files."""
    # Raw files are not read without the section.
    assert parse_config(minimal()).raw is None

    data = minimal()
    data["raw"] = {"sample_rate": 48000, "data_type": "d", "extensions": [".d"]}
    config = parse_config(data)
    assert config.raw is not None
    assert (config.raw.sample_rate, config.raw.data_type) == (48000, "d")
    assert config.raw.extensions == [".d"]

    data["raw"] = {"sample_rate": 16000}
    config = parse_config(data)
    assert config.raw is not None
    assert (config.raw.data_type, config.raw.num_channels) == ("s", 1)
    assert config.raw.extensions == [".raw"]

    data["raw"] = {"sample_rate": 16000, "data_type": "e"}
    with pytest.raises(ConfigError, match=r"raw\.data_type"):
        parse_config(data)

    data["raw"] = {"sample_rate": 16000, "num_channels": 2}
    assert parse_config(data).raw.num_channels == 2  # type: ignore[union-attr]
    data["raw"] = {"sample_rate": 16000, "num_channels": 0}
    with pytest.raises(ConfigError, match=r"raw\.num_channels"):
        parse_config(data)


@pytest.mark.parametrize("raw", [{}, {"sample_rate": None}, {"sample_rate": 0}])
def test_invalid_raw_sample_rate(raw: dict) -> None:
    """Test that the sample rate of raw files is required to be positive."""
    data = minimal()
    data["raw"] = raw
    with pytest.raises(ConfigError, match=r"raw\.sample_rate"):
        parse_config(data)


def test_type_error_with_path() -> None:
    """Test that a type error is reported with its path."""
    data = minimal()
    data["frame_period"] = "five"
    with pytest.raises(ConfigError, match="frame_period"):
        parse_config(data)


@pytest.mark.parametrize("frame_period", [0.0, float("nan"), float("inf")])
def test_invalid_frame_period(frame_period: float) -> None:
    """Test that the frame period must be a positive finite number."""
    data = minimal()
    data["frame_period"] = frame_period
    with pytest.raises(ConfigError, match="frame_period"):
        parse_config(data)


def test_choice_error() -> None:
    """Test that an invalid choice is reported."""
    data = minimal()
    data["output"] = {"format": "cent"}
    with pytest.raises(ConfigError, match=r"output\.format"):
        parse_config(data)


def test_missing_required_key() -> None:
    """Test that a missing required key is reported."""
    with pytest.raises(ConfigError, match="trackers"):
        parse_config({})


def test_int_is_accepted_as_float() -> None:
    """Test that an integer is accepted for a float field."""
    data = minimal()
    data["frame_period"] = 10
    assert parse_config(data).frame_period == 10.0


def test_voter_can_be_disabled() -> None:
    """Test that voting can be disabled."""
    data = minimal()
    data["voter"] = None
    assert parse_config(data).voter is None


@pytest.mark.parametrize("content", [None, "{"])
def test_invalid_voter_state(tmp_path, content: str | None) -> None:
    """Test that a missing or broken state file is an error of the configuration."""
    path = tmp_path / "state.json"
    if content is not None:
        path.write_text(content)
    data = minimal()
    data["voter"] = {"type": "median", "state": str(path)}
    with pytest.raises(ConfigError, match=r"voter\.state"):
        parse_config(data)


def test_two_pass_settings() -> None:
    """Test the settings of the two-pass tracking."""
    data = minimal()
    data["two_pass"] = {
        "estimator": {"type": "log-normal", "params": {"gamma": 2.0}},
        "scope": "group",
        "group_by": {"type": "dirname", "level": 2},
    }
    config = parse_config(data)
    assert config.two_pass is not None
    assert config.two_pass.estimator.type == "log-normal"
    assert config.two_pass.group_by.level == 2


def test_invalid_estimator_parameter() -> None:
    """Test that an invalid estimator parameter is reported."""
    data = minimal()
    data["two_pass"] = {"estimator": {"params": {"foo": 1}}}
    with pytest.raises(ConfigError, match=r"two_pass\.estimator"):
        parse_config(data)


def test_load_from_file(tmp_path) -> None:
    """Test loading from a YAML file."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(minimal()))
    assert load_config(str(path)).trackers[0].algorithm == "swift-f0"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError):
        load_config(str(path))
    path.write_text("trackers: [\n")
    with pytest.raises(ConfigError, match="config.yaml"):
        load_config(str(path))
    with pytest.raises(ConfigError, match="missing.yaml"):
        load_config(str(tmp_path / "missing.yaml"))


def test_two_pass_policy_of_tracker() -> None:
    """Test the setting of the two-pass policy of a tracker."""
    from pitch_garden import TwoPassPolicy
    from pitch_garden.config import build_pipeline

    data = minimal()
    data["trackers"][0]["two_pass"] = {"enabled": True, "voicing_from": "1st"}
    data["two_pass"] = {}
    with build_pipeline(parse_config(data)) as pipeline:
        expected = TwoPassPolicy(enabled=True, voicing_from="1st")
        assert pipeline.trackers[0].two_pass_policy == expected

    data["trackers"][0]["two_pass"] = {"voicing_from": "both"}
    with pytest.raises(ConfigError, match=r"trackers\[0\]\.two_pass"):
        parse_config(data)


def test_build_pipeline() -> None:
    """Test building the pipeline from the configuration."""
    from pitch_garden import Pipeline, TwoPass
    from pitch_garden.config import build_pipeline

    data = minimal()
    data["trackers"].append({"algorithm": "swift-f0", "name": "b"})
    data["two_pass"] = {"fmin": 50.0, "range_from": "b"}
    with build_pipeline(parse_config(data)) as pipeline:
        assert isinstance(pipeline, Pipeline)
        assert [t.name for t in pipeline.trackers] == ["swift-f0", "b"]
        assert isinstance(pipeline.two_pass, TwoPass)
        assert pipeline.two_pass.fmin == 50.0


def test_errors_of_objects_are_reported_with_paths() -> None:
    """Test that the errors of the built objects are reported with the paths."""
    data = minimal()
    data["two_pass"] = {"range_from": "unknown"}
    with pytest.raises(ConfigError, match=r"two_pass: .*unknown"):
        parse_config(data)
    data = minimal()
    data["trackers"].append({"algorithm": "swift-f0"})
    with pytest.raises(ConfigError, match=r"trackers: .*Duplicate"):
        parse_config(data)
    data = minimal()
    data["voter"] = {"type": "aligned", "params": {"reference": "rapt"}}
    with pytest.raises(ConfigError, match=r"voter: .*reference"):
        parse_config(data)


def test_on_error() -> None:
    """Test the setting of the behavior on failures of trackers."""
    from pitch_garden.config import build_pipeline

    assert parse_config(minimal()).on_error == "skip"
    data = minimal()
    data["on_error"] = "raise"
    with build_pipeline(parse_config(data)) as pipeline:
        assert pipeline.on_error == "raise"
    data["on_error"] = "ignore"
    with pytest.raises(ConfigError, match="on_error"):
        parse_config(data)
