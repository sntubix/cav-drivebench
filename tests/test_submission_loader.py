import textwrap
from pathlib import Path

import pytest

from metadrive_starter.config import config_from_dict
from metadrive_starter.submission import (
    SubmissionError,
    apply_submission_overlay,
    load_controller,
)
from metadrive_starter.types import ControlTick


PROPORTIONAL_CONTROLLER = """
    class Controller:
        def __init__(self, settings):
            self.speed_kp = settings.speed_pid.kp
            self.steering_kp = settings.steering_pid.kp

        def update(self, tick):
            error = tick.target_speed_mps - tick.speed_mps
            return (self.steering_kp * tick.heading_error_rad, self.speed_kp * error)

        def reset_speed_control(self):
            pass
"""


def _submission(directory: Path, controller_source: str) -> Path:
    (directory / "controller.py").write_text(textwrap.dedent(controller_source))
    return directory


def test_submitted_controller_is_built_from_run_settings_and_drives(
    tmp_path: Path,
) -> None:
    factory = load_controller(_submission(tmp_path, PROPORTIONAL_CONTROLLER))
    settings = config_from_dict(
        {"controller": {"speed_pid": {"kp": 0.5}, "steering_pid": {"kp": 2.0}}}
    ).controller

    controller = factory(settings)

    assert controller.update(ControlTick(10.0, 9.0, 0.1, 0.0, 0.1)) == pytest.approx(
        (0.2, 0.5)
    )


def test_missing_entry_point_names_what_was_expected_and_what_was_found(
    tmp_path: Path,
) -> None:
    submission = _submission(
        tmp_path,
        """
        from metadrive_starter.types import ControlTick

        class MyController:
            pass

        def helper():
            pass
        """,
    )

    with pytest.raises(
        SubmissionError,
        match=(
            r"controller\.py must define Controller\(settings\) returning a "
            r"VehicleController with update\(tick\) and reset_speed_control\(\); "
            r"it defines MyController, helper"
        ),
    ):
        load_controller(submission)


@pytest.mark.parametrize(
    ("source", "found"),
    [
        (
            """
            class Controller:
                def __init__(self):
                    pass
            """,
            r"its signature is Controller\(\)",
        ),
        (
            """
            class Controller:
                def __init__(self, kp: float, ki: float, kd: float):
                    pass
            """,
            r"its signature is Controller\(kp, ki, kd\)",
        ),
        ("Controller = 3", r"found 3"),
    ],
    ids=["no-settings", "gains-instead-of-settings", "not-callable"],
)
def test_entry_point_with_wrong_signature_names_the_expected_call(
    tmp_path: Path,
    source: str,
    found: str,
) -> None:
    submission = _submission(tmp_path, source)

    with pytest.raises(
        SubmissionError,
        match=(
            r"controller\.py: Controller must be callable as "
            r"Controller\(settings\); " + found
        ),
    ):
        load_controller(submission)


@pytest.mark.parametrize(
    ("source", "problem"),
    [
        (
            """
            class Controller:
                def __init__(self, settings):
                    pass

                def update(self, tick):
                    return (0.0, 0.0)
            """,
            r"Controller\(settings\) returned a Controller without "
            r"reset_speed_control\(\); expected Controller\(settings\) returning a "
            r"VehicleController with update\(tick\) and reset_speed_control\(\)",
        ),
        (
            """
            class Controller:
                def __init__(self, settings):
                    pass

                def update(self):
                    return (0.0, 0.0)

                def reset_speed_control(self):
                    pass
            """,
            r"Controller\.update must be callable as update\(tick\); "
            r"its signature is update\(\)",
        ),
        (
            """
            def Controller(settings):
                return None
            """,
            r"Controller\(settings\) returned a NoneType without update\(tick\)",
        ),
    ],
    ids=["missing-method", "update-without-tick", "factory-returns-nothing"],
)
def test_controller_without_the_protocol_names_the_expected_method(
    tmp_path: Path,
    source: str,
    problem: str,
) -> None:
    factory = load_controller(_submission(tmp_path, source))

    with pytest.raises(SubmissionError, match=r"controller\.py: " + problem):
        factory(config_from_dict({}).controller)


def test_missing_controller_file_names_the_file_and_its_entry_point(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        SubmissionError,
        match=(
            r"controller\.py not found; expected a file defining "
            r"Controller\(settings\) returning a VehicleController"
        ),
    ):
        load_controller(tmp_path)


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("import definitely_not_an_installed_module", ModuleNotFoundError),
        (
            """
            class Controller:
                def update(self, tick) return (0.0, 0.0)
            """,
            SyntaxError,
        ),
    ],
    ids=["missing-import", "syntax-error"],
)
def test_controller_that_fails_to_import_reports_the_original_error(
    tmp_path: Path,
    source: str,
    error: type[Exception],
) -> None:
    submission = _submission(tmp_path, source)

    with pytest.raises(
        SubmissionError,
        match=rf"controller\.py failed to import: {error.__name__}",
    ) as raised:
        load_controller(submission)
    assert isinstance(raised.value.__cause__, error)


def test_controller_written_with_dataclasses_and_postponed_annotations_loads(
    tmp_path: Path,
) -> None:
    submission = _submission(
        tmp_path,
        """
        from __future__ import annotations

        from dataclasses import dataclass

        @dataclass
        class Loop:
            kp: float

            def update(self, error: float) -> float:
                return self.kp * error

        class Controller:
            def __init__(self, settings) -> None:
                self.speed = Loop(settings.speed_pid.kp)

            def update(self, tick) -> tuple[float, float]:
                error = tick.target_speed_mps - tick.speed_mps
                return (0.0, self.speed.update(error))

            def reset_speed_control(self) -> None:
                pass
        """,
    )
    settings = config_from_dict({"controller": {"speed_pid": {"kp": 0.25}}}).controller

    controller = load_controller(submission)(settings)

    assert controller.update(ControlTick(10.0, 8.0, 0.0, 0.0, 0.1)) == (0.0, 0.5)


def test_controller_that_fails_to_construct_reports_the_original_error(
    tmp_path: Path,
) -> None:
    factory = load_controller(
        _submission(
            tmp_path,
            """
            class Controller:
                def __init__(self, settings):
                    self.kp = settings.speed_gain
            """,
        )
    )

    with pytest.raises(
        SubmissionError,
        match=r"controller\.py: Controller\(settings\) raised AttributeError: .*speed_gain",
    ) as raised:
        factory(config_from_dict({}).controller)
    assert isinstance(raised.value.__cause__, AttributeError)


def test_loading_a_submission_writes_nothing_into_its_directory(
    tmp_path: Path,
) -> None:
    submission = _submission(tmp_path, PROPORTIONAL_CONTROLLER)

    load_controller(submission)(config_from_dict({}).controller)

    assert [path.name for path in submission.iterdir()] == ["controller.py"]


def test_submitted_agent_overlay_tunes_the_configuration(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text("controller:\n  steering_pid:\n    kd: 0.1\n")

    config = apply_submission_overlay(config_from_dict({}), tmp_path)

    assert config.controller.steering_pid.kd == 0.1


@pytest.mark.parametrize(
    "agent_yaml",
    [None, "# Tune permitted values here.\n"],
    ids=["absent", "comments-only"],
)
def test_submission_without_overlay_values_keeps_the_instructor_configuration(
    tmp_path: Path,
    agent_yaml: str | None,
) -> None:
    if agent_yaml is not None:
        (tmp_path / "agent.yaml").write_text(agent_yaml)
    base = config_from_dict({})

    config = apply_submission_overlay(base, tmp_path)

    assert config.to_dict() == base.to_dict()


@pytest.mark.parametrize(
    ("agent_yaml", "message"),
    [
        ("simulator:\n  map: S\n", r"agent\.yaml: simulator\.map is not permitted in assignment 2"),
        ("controller: [kp\n", r"agent\.yaml is not valid YAML"),
        ("- kp\n- 0.3\n", r"agent\.yaml: the overlay must be a mapping of settings"),
    ],
    ids=["not-permitted", "malformed", "not-a-mapping"],
)
def test_invalid_agent_overlay_names_the_file(
    tmp_path: Path,
    agent_yaml: str,
    message: str,
) -> None:
    (tmp_path / "agent.yaml").write_text(agent_yaml)

    with pytest.raises(SubmissionError, match=message):
        apply_submission_overlay(config_from_dict({}), tmp_path)


def test_submission_overlay_returns_a_configuration_independent_of_the_base(
    tmp_path: Path,
) -> None:
    base = config_from_dict({})

    config = apply_submission_overlay(base, tmp_path)
    config.simulator.map = "S"

    assert base.simulator.map == "XTOC"
