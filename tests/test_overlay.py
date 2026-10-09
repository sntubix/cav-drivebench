import re
from dataclasses import fields
from pathlib import Path

import pytest

from metadrive_starter.config import config_from_dict
from metadrive_starter.overlay import (
    ASSIGNMENT_1,
    ASSIGNMENT_2,
    Allowlist,
    Direction,
    OverlayError,
    PermittedKey,
    apply_overlay,
)
from metadrive_starter.safety import CommandValidationSettings
from metadrive_starter.simulation import run_simulation
from metadrive_starter.submission import apply_submission_overlay
from metadrive_starter.types import ControlTick


def test_permitted_key_changes_the_configuration_and_leaves_the_base_alone() -> None:
    base = config_from_dict({})

    tuned = apply_overlay(base, {"controller": {"steering_pid": {"kd": 0.1}}}, ASSIGNMENT_1)

    assert tuned.controller.steering_pid.kd == 0.1
    assert base.controller.steering_pid.kd == 0.0


def test_key_outside_the_allowlist_is_rejected_by_name() -> None:
    with pytest.raises(OverlayError, match=r"simulator\.map is not permitted in assignment 1"):
        apply_overlay(config_from_dict({}), {"simulator": {"map": "S"}}, ASSIGNMENT_1)


@pytest.mark.parametrize(
    ("overlay", "message"),
    [
        (
            {"controller": {"speed_pid": {"kp": -0.1}}},
            r"controller\.speed_pid\.kp must be at least 0; got -0\.1",
        ),
        (
            {"controller": {"speed_pid": {"output_max": 1.5}}},
            r"controller\.speed_pid\.output_max must be at most 1; got 1\.5",
        ),
    ],
    ids=["below-minimum", "above-maximum"],
)
def test_bounded_key_outside_its_bounds_is_rejected(
    overlay: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(OverlayError, match=message):
        apply_overlay(config_from_dict({}), overlay, ASSIGNMENT_1)


def test_bounded_key_inside_its_bounds_applies() -> None:
    overlay = {"controller": {"speed_pid": {"output_max": 0.8}}}

    tuned = apply_overlay(config_from_dict({}), overlay, ASSIGNMENT_1)

    assert tuned.controller.speed_pid.output_max == 0.8


# Assignment 2's confidence floor: teams may tighten it, never loosen it.
CONFIDENCE_FLOOR = Allowlist(
    "confidence test",
    (PermittedKey("command_validation.minimum_confidence", only=Direction.RAISE),),
)


@pytest.mark.parametrize(
    ("allowlist", "overlay", "message"),
    [
        (
            ASSIGNMENT_1,
            {"controller": {"target_speed_mps": 12.0}},
            r"controller\.target_speed_mps may only be lowered from 9\.72222; got 12",
        ),
        (
            CONFIDENCE_FLOOR,
            {"command_validation": {"minimum_confidence": 0.3}},
            r"command_validation\.minimum_confidence may only be raised from 0\.5; got 0\.3",
        ),
    ],
    ids=["lower-only-raised", "raise-only-lowered"],
)
def test_one_directional_key_rejects_the_loosening_direction(
    allowlist: Allowlist,
    overlay: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(OverlayError, match=message):
        apply_overlay(config_from_dict({}), overlay, allowlist)


def test_one_directional_keys_accept_the_tightening_direction() -> None:
    base = config_from_dict({})

    slower = apply_overlay(base, {"controller": {"target_speed_mps": 8.0}}, ASSIGNMENT_1)
    stricter = apply_overlay(
        base,
        {"command_validation": {"minimum_confidence": 0.7}},
        CONFIDENCE_FLOOR,
    )

    assert slower.controller.target_speed_mps == 8.0
    assert stricter.command_validation.minimum_confidence == 0.7


@pytest.mark.parametrize("allowlist", [ASSIGNMENT_1, ASSIGNMENT_2], ids=lambda a: a.name)
def test_endpoint_settings_cannot_be_set_from_an_overlay(allowlist: Allowlist) -> None:
    overlay = {"vla": {"http": {"endpoint_url": "http://10.0.0.5:8000/v1/chat/completions"}}}

    with pytest.raises(OverlayError, match=r"vla\.http\.endpoint_url is not permitted"):
        apply_overlay(config_from_dict({}), overlay, allowlist)


@pytest.mark.parametrize(
    "path",
    ["vla.http.endpoint_url", "vla.http.model_id", "vla.vertex.model_id"],
)
def test_no_allowlist_can_permit_an_endpoint_or_model_pin(path: str) -> None:
    with pytest.raises(ValueError, match=re.escape(f"{path} is instructor-owned")):
        Allowlist("leaky", (PermittedKey(path),))


@pytest.mark.parametrize(
    ("overlay", "message"),
    [
        (
            {"controller": {"speed_pid": {"kp": "fast"}}},
            r"controller\.speed_pid\.kp must be a finite number; got 'fast'",
        ),
        (
            {"controller": {"speed_pid": {"kp": True}}},
            r"controller\.speed_pid\.kp must be a finite number; got True",
        ),
        (
            {"controller": {"speed_pid": {"kp": float("nan")}}},
            r"controller\.speed_pid\.kp must be a finite number; got nan",
        ),
        (
            {"controller": {"speed_pid": {"anti_windup": "yes"}}},
            r"controller\.speed_pid\.anti_windup must be true or false; got 'yes'",
        ),
    ],
    ids=["string", "boolean-for-number", "nan", "string-for-boolean"],
)
def test_value_of_the_wrong_type_is_rejected(
    overlay: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(OverlayError, match=message):
        apply_overlay(config_from_dict({}), overlay, ASSIGNMENT_1)


def test_integer_is_accepted_for_a_real_valued_key() -> None:
    tuned = apply_overlay(
        config_from_dict({}),
        {"controller": {"speed_pid": {"kp": 1}}},
        ASSIGNMENT_1,
    )

    assert tuned.controller.speed_pid.kp == 1.0
    assert isinstance(tuned.controller.speed_pid.kp, float)


def test_every_problem_is_reported_at_once() -> None:
    overlay = {
        "simulator": {"map": "S"},
        "controller": {"target_speed_mps": 20.0, "steering_pid": {"kd": -1.0}},
    }

    with pytest.raises(OverlayError) as raised:
        apply_overlay(config_from_dict({}), overlay, ASSIGNMENT_1)

    message = str(raised.value)
    assert "simulator.map is not permitted" in message
    assert "controller.target_speed_mps may only be lowered" in message
    assert "controller.steering_pid.kd must be at least 0" in message


def test_values_that_combine_into_an_invalid_configuration_are_rejected() -> None:
    overlay = {"controller": {"speed_pid": {"output_min": 0.5, "output_max": 0.2}}}

    with pytest.raises(OverlayError, match="output_min must be below output_max"):
        apply_overlay(config_from_dict({}), overlay, ASSIGNMENT_1)


def test_overlay_must_be_a_mapping_of_settings() -> None:
    with pytest.raises(OverlayError, match="must be a mapping of settings; got a list"):
        apply_overlay(config_from_dict({}), ["kp", 0.3], ASSIGNMENT_1)  # type: ignore[arg-type]


def test_every_assignment_1_key_names_a_number_or_boolean_in_the_configuration() -> None:
    config = config_from_dict({})

    for key in ASSIGNMENT_1.keys:
        value: object = config
        for part in key.path.split("."):
            value = getattr(value, part)
        assert isinstance(value, (bool, int, float)), key.path


def test_assignment_1_permits_controller_and_tracking_values_only() -> None:
    paths = {key.path for key in ASSIGNMENT_1.keys}
    lower_only = {key.path for key in ASSIGNMENT_1.keys if key.only is Direction.LOWER}

    assert {path.split(".")[0] for path in paths} == {"controller", "planner"}
    assert {path for path in paths if path.startswith("planner.")} == {
        "planner.curvature_preview_m",
        "planner.maximum_lateral_acceleration_mps2",
        "planner.minimum_curve_speed_mps",
    }
    assert lower_only == {
        "controller.target_speed_mps",
        "planner.maximum_lateral_acceleration_mps2",
        "planner.minimum_curve_speed_mps",
    }


class _RecordingController:
    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.ticks: list[ControlTick] = []

    def update(self, tick: ControlTick) -> tuple[float, float]:
        self.ticks.append(tick)
        return (0.0, 0.0)

    def reset_speed_control(self) -> None:
        pass


def test_agent_overlay_reaches_the_controller_at_runtime(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text(
        "controller:\n  target_speed_mps: 6.0\n  steering_pid:\n    kp: 0.9\n"
    )
    built: list[_RecordingController] = []

    def factory(settings: object) -> _RecordingController:
        built.append(_RecordingController(settings))
        return built[-1]

    config = apply_submission_overlay(config_from_dict({}), tmp_path)
    run_simulation(config, dry_run=True, controller_factory=factory)

    [controller] = built
    assert controller.settings.steering_pid.kp == 0.9  # type: ignore[attr-defined]
    assert controller.ticks[0].target_speed_mps == 6.0


@pytest.mark.instructor
def test_lowering_target_speed_in_agent_yaml_slows_a_route(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text("controller:\n  target_speed_mps: 6.0\n")
    config = config_from_dict(
        {
            "simulator": {
                "map": "S",
                "traffic_density": 0.0,
                "obstacle_probability": 0.0,
                "horizon": 1000,
                "headless": True,
            }
        }
    )

    baseline = run_simulation(config)
    slower = run_simulation(apply_submission_overlay(config, tmp_path))

    assert baseline.arrived and slower.arrived
    assert slower.steps > baseline.steps


def test_misspelt_key_suggests_the_permitted_one() -> None:
    overlay = {"controller": {"steering_pid": {"Kd": 0.1}}}

    with pytest.raises(
        OverlayError,
        match=(
            r"controller\.steering_pid\.Kd is not permitted in assignment 1; "
            r"did you mean controller\.steering_pid\.kd\?"
        ),
    ):
        apply_overlay(config_from_dict({}), overlay, ASSIGNMENT_1)


def test_assignment_2_adds_the_observation_and_floor_thresholds_to_assignment_1() -> None:
    added = {key.path for key in ASSIGNMENT_2.keys} - {key.path for key in ASSIGNMENT_1.keys}

    assert set(ASSIGNMENT_1.keys) <= set(ASSIGNMENT_2.keys)
    assert added == {
        "vla.prompt_policy",
        "observation.driving_context",
        "observation.scene_context",
        "observation.scene_fields.lanes",
        "observation.scene_fields.traffic_controls",
        "observation.scene_fields.objects",
        *(
            f"command_validation.{field.name}"
            for field in fields(CommandValidationSettings)
        ),
    }


@pytest.mark.parametrize("path", ["vla.enabled", "vla.provider", "vla.max_scene_objects"])
def test_assignment_2_leaves_the_model_run_itself_instructor_owned(path: str) -> None:
    overlay: dict[str, object] = {}
    parts = path.split(".")
    level = overlay
    for part in parts[:-1]:
        level = level.setdefault(part, {})  # type: ignore[assignment]
    level[parts[-1]] = 1

    with pytest.raises(OverlayError, match=rf"{re.escape(path)} is not permitted in assignment 2"):
        apply_overlay(config_from_dict({}), overlay, ASSIGNMENT_2)


def test_prompt_text_and_observation_channels_apply_from_an_overlay() -> None:
    overlay = {
        "vla": {"prompt_policy": "  Prefer the cruise speed on a clear lane.\n"},
        "observation": {"driving_context": False, "scene_fields": {"objects": False}},
    }

    tuned = apply_overlay(config_from_dict({}), overlay, ASSIGNMENT_2)

    assert tuned.vla.prompt_policy == "Prefer the cruise speed on a clear lane."
    assert not tuned.observation.driving_context
    assert tuned.observation.scene_context
    assert not tuned.observation.scene_fields.objects
    assert tuned.observation.scene_fields.lanes


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (42, r"vla\.prompt_policy must be text; got 42"),
        ("x" * 4097, r"invalid configuration: vla\.prompt_policy must contain at most 4096"),
    ],
    ids=["not-text", "too-long"],
)
def test_prompt_text_must_be_bounded_text(policy: object, message: str) -> None:
    with pytest.raises(OverlayError, match=message):
        apply_overlay(config_from_dict({}), {"vla": {"prompt_policy": policy}}, ASSIGNMENT_2)
