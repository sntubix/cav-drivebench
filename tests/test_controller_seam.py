from pathlib import Path

import pytest

from metadrive_starter import submission
from metadrive_starter.config import config_from_dict
from metadrive_starter.controllers import ControllerOutputError, build_vehicle_controller
from metadrive_starter.simulation import run_simulation
from metadrive_starter.submission import SubmissionError, load_controller
from metadrive_starter.types import ControlTick


class _ControllerWithoutReset:
    def update(self, tick: ControlTick) -> tuple[float, float]:
        return (0.0, 0.0)


def test_factory_product_without_the_controller_protocol_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"update\(tick\).*reset_speed_control\(\)"):
        build_vehicle_controller(
            config_from_dict({}).controller,
            factory=lambda settings: _ControllerWithoutReset(),
        )


class _RecordingController:
    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.ticks: list[ControlTick] = []

    def update(self, tick: ControlTick) -> tuple[float, float]:
        self.ticks.append(tick)
        return (0.0, 0.0)

    def reset_speed_control(self) -> None:
        pass


def test_dry_run_drives_the_injected_controller_with_configured_values() -> None:
    built: list[_RecordingController] = []

    def factory(settings: object) -> _RecordingController:
        built.append(_RecordingController(settings))
        return built[-1]

    config = config_from_dict(
        {
            "simulator": {"decision_repeat": 4, "physics_world_step_size": 0.025},
            "controller": {"target_speed_mps": 7.5},
        }
    )

    run_simulation(config, dry_run=True, controller_factory=factory)

    [controller] = built
    [tick] = controller.ticks
    assert controller.settings == config.controller
    assert tick.target_speed_mps == 7.5
    assert tick.dt_s == pytest.approx(0.1)


class _FixedActionController:
    def __init__(self, action: object) -> None:
        self.action = action

    def update(self, tick: ControlTick) -> object:
        return self.action

    def reset_speed_control(self) -> None:
        pass


@pytest.mark.parametrize(
    ("action", "message"),
    [
        ((0.0, float("nan")), r"throttle_brake must be a finite number"),
        ((float("inf"), 0.0), r"steering must be a finite number"),
        ((0.0, 0.2, 0.1), r"must return \(steering, throttle_brake\)"),
        (0.4, r"must return \(steering, throttle_brake\)"),
        (("left", 0.2), r"steering must be a number"),
        ((0.1, True), r"throttle_brake must be a number"),
    ],
)
def test_run_rejects_a_controller_action_outside_the_contract(
    action: object,
    message: str,
) -> None:
    with pytest.raises(ControllerOutputError, match=message):
        run_simulation(
            config_from_dict({}),
            dry_run=True,
            controller_factory=lambda settings: _FixedActionController(action),
        )


NEUTRAL_CONTROLLER = (
    "class Controller:\n"
    "    def __init__(self, settings):\n"
    "        pass\n"
    "    def update(self, tick):\n"
    "        return (0.0, 0.0)\n"
    "    def reset_speed_control(self):\n"
    "        pass\n"
)


@pytest.mark.instructor
def test_run_summary_names_the_controller_that_drove(tmp_path: Path) -> None:
    (tmp_path / "controller.py").write_text(NEUTRAL_CONTROLLER)
    config = config_from_dict({})

    default = run_simulation(config, dry_run=True)
    submitted = run_simulation(
        config,
        dry_run=True,
        controller_factory=load_controller(tmp_path),
    )

    assert default.controller == "reference.controller.Controller"
    assert submitted.controller == "submission.controller.Controller"


def test_manual_control_needs_no_controller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(submission, "INSTRUCTOR_DIR", tmp_path / "absent")
    config = config_from_dict({"simulator": {"manual_control": True}})

    assert run_simulation(config, dry_run=True).controller is None


def test_run_without_a_submission_or_reference_asks_for_a_submission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(submission, "INSTRUCTOR_DIR", tmp_path / "absent")

    with pytest.raises(SubmissionError, match="run with --submission submission"):
        run_simulation(config_from_dict({}), dry_run=True)


def test_run_clamps_an_out_of_range_action_and_counts_it() -> None:
    summary = run_simulation(
        config_from_dict({}),
        dry_run=True,
        controller_factory=lambda settings: _FixedActionController((1.5, -2.0)),
    )

    assert summary.controller_outputs_clamped == 1


def test_run_counts_no_clamping_for_an_action_within_range() -> None:
    summary = run_simulation(
        config_from_dict({}),
        dry_run=True,
        controller_factory=lambda settings: _FixedActionController((1.0, -1.0)),
    )

    assert summary.controller_outputs_clamped == 0
