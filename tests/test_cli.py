import json
from pathlib import Path

import pytest

from metadrive_starter import submission
from metadrive_starter.cli import main
from metadrive_starter.cloud_preflight import PreflightCheck, VertexPreflightReport
from metadrive_starter.config import config_from_dict
from metadrive_starter.events import EventLogger
from metadrive_starter.evaluation import EvaluationReport
from metadrive_starter.episode_replay import EpisodeReplaySummary, RaceReplaySummary
from metadrive_starter.race import RaceReport
from metadrive_starter.provider_fixture_capture import (
    ProviderFixtureConversion,
    ProviderFixtureConversionReport,
)
from metadrive_starter.simulation import _format_display_speed, run_simulation
from metadrive_starter.vla_probe import ProbeRunSummary, ProbeScenarioResult
from metadrive_starter.vertex_capture import VertexCaptureReport


@pytest.mark.instructor
def test_cli_smoke_dry_run(capsys) -> None:
    assert main(["smoke", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert '"dry_run": true' in output


def _write_controller(directory: Path, source: str) -> Path:
    (directory / "controller.py").write_text(source)
    return directory


def test_cli_smoke_counts_clamped_outputs_of_a_submitted_controller(
    capsys,
    tmp_path,
) -> None:
    submission = _write_controller(
        tmp_path,
        "class Controller:\n"
        "    def __init__(self, settings):\n"
        "        pass\n"
        "    def update(self, tick):\n"
        "        return (2.0, 0.0)\n"
        "    def reset_speed_control(self):\n"
        "        pass\n",
    )

    assert main(["smoke", "--submission", str(submission)]) == 0

    assert '"controller_outputs_clamped": 1' in capsys.readouterr().out


def test_cli_reports_a_submission_that_does_not_load(capsys, tmp_path) -> None:
    submission = _write_controller(tmp_path, "class MyController:\n    pass\n")

    assert main(["run", "--dry-run", "--submission", str(submission)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "must define Controller(settings)" in captured.err


def test_cli_shows_where_a_submission_failed_to_import(capsys, tmp_path) -> None:
    submission = _write_controller(
        tmp_path,
        "import math\n\nSPEED_GAIN = math.tau / undefined_constant\n",
    )

    assert main(["smoke", "--submission", str(submission)]) == 1

    error = capsys.readouterr().err
    assert "failed to import: NameError" in error
    assert "SPEED_GAIN = math.tau / undefined_constant" in error


def test_cli_asks_for_a_submission_when_there_is_no_reference(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.setattr(submission, "INSTRUCTOR_DIR", tmp_path / "absent")

    assert main(["smoke"]) == 1

    assert "run with --submission submission" in capsys.readouterr().err


NEUTRAL_CONTROLLER = (
    "class Controller:\n"
    "    def __init__(self, settings):\n"
    "        pass\n"
    "    def update(self, tick):\n"
    "        return (0.0, 0.0)\n"
    "    def reset_speed_control(self):\n"
    "        pass\n"
)


def test_cli_config_shows_the_submitted_agent_overlay(capsys, tmp_path) -> None:
    (tmp_path / "agent.yaml").write_text("controller:\n  steering_pid:\n    kd: 0.1\n")

    assert main(["config", "--submission", str(tmp_path)]) == 0

    config = json.loads(capsys.readouterr().out)
    assert config["controller"]["steering_pid"]["kd"] == 0.1


def test_cli_rejects_an_agent_overlay_key_outside_the_allowlist(capsys, tmp_path) -> None:
    submission = _write_controller(tmp_path, NEUTRAL_CONTROLLER)
    (submission / "agent.yaml").write_text("simulator:\n  map: S\n")

    assert main(["run", "--dry-run", "--submission", str(submission)]) == 1

    error = capsys.readouterr().err
    assert "agent.yaml: simulator.map is not permitted in assignment 2" in error
    assert "Traceback" not in error


STRAIGHT_GATE_PLAN = (
    "version: 1\nid: test-gates\nvisibility: public\nscenarios:\n"
    "  - {id: straight, map: S, seed: 0, horizon: 600, par_steps: 134}\n"
)
SATURATED_CONTROLLER = (
    "class Controller:\n"
    "    def __init__(self, settings):\n"
    "        self.kp = settings.speed_pid.kp\n"
    "        self.heading_kp = settings.steering_pid.kp\n"
    "        self.lateral_kp = settings.lateral_pid.kp\n"
    "    def update(self, tick):\n"
    "        throttle = self.kp * (tick.target_speed_mps - tick.speed_mps)\n"
    "        steering = (self.heading_kp * tick.heading_error_rad\n"
    "                    + self.lateral_kp * tick.lateral_error_m)\n"
    "        return (max(-1.0, min(1.0, steering)), max(-1.0, min(1.0, throttle)))\n"
    "    def reset_speed_control(self):\n"
    "        pass\n"
)


def test_cli_gates_pass_a_well_behaved_submission(capsys, tmp_path) -> None:
    submission = _write_controller(tmp_path, SATURATED_CONTROLLER)
    plan = tmp_path / "gates.yaml"
    plan.write_text(STRAIGHT_GATE_PLAN)

    assert main(["gates", "--submission", str(submission), "--manifest", str(plan)]) == 0

    output = capsys.readouterr().out
    assert "passed   submission loads" in output
    assert "passed   no run stops on an error: 1 route driven without an error" in output
    assert "\nPASSED\n" in output
    # Proportional-only on the starting gains: 11 steps behind par, and no
    # implementation check passes.
    assert "Score 26.8 of 100, a quality measure that never decides a gate" in output
    assert "  implementation checks 0.0: 0 of 6 passed" in output
    assert "    failed   speed loop integral term: with 0.1 s ticks" in output
    assert "  driving 53.5" in output
    assert "     53.5  straight: 145 steps against par 134, speed error" in output


def test_cli_gates_report_a_submission_that_does_not_load(capsys, tmp_path) -> None:
    submission = _write_controller(tmp_path, "class Controller(:\n")
    plan = tmp_path / "gates.yaml"
    plan.write_text(STRAIGHT_GATE_PLAN)

    assert main(
        ["gates", "--submission", str(submission), "--manifest", str(plan), "--json"]
    ) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["results"][0]["gate"] == "submission loads"
    assert report["results"][0]["status"] == "failed"
    assert report["score"] is None


def test_cli_gates_mark_a_run_of_selected_scenarios_partial(capsys, tmp_path) -> None:
    submission = _write_controller(tmp_path, SATURATED_CONTROLLER)
    plan = tmp_path / "gates.yaml"
    plan.write_text(
        STRAIGHT_GATE_PLAN
        + "  - {id: curve, map: C, seed: 0, horizon: 900, par_steps: 257}\n"
    )

    assert main(
        [
            "gates",
            "--submission",
            str(submission),
            "--manifest",
            str(plan),
            "--scenario",
            "straight",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert (
        "  PARTIAL RUN: straight only; a full run drives every scenario, as grading does"
        in output
    )
    assert "passed   no run stops on an error: 1 route driven without an error" in output


def test_cli_gates_refuse_an_unknown_scenario_before_driving(capsys, tmp_path) -> None:
    plan = tmp_path / "gates.yaml"
    plan.write_text(STRAIGHT_GATE_PLAN)

    with pytest.raises(SystemExit) as exited:
        main(["gates", "--manifest", str(plan), "--scenario", "curve"])

    assert exited.value.code == 2
    assert "unknown gate scenario id: curve; test-gates has straight" in (
        capsys.readouterr().err
    )


def test_cli_exposes_demo_overrides(capsys) -> None:
    assert main(
        ["config", "--continue-off-road", "--continue-after-crash", "--obstacle-probability", "0.25"]
    ) == 0

    output = capsys.readouterr().out
    assert '"map": "XTOC"' in output
    assert '"obstacle_probability": 0.25' in output
    assert '"out_of_road_done": false' in output
    assert '"crash_vehicle_done": false' in output
    assert '"crash_object_done": false' in output


def test_cli_overrides_rendered_speed_unit(capsys) -> None:
    assert main(["config", "--speed-unit", "kph"]) == 0

    output = capsys.readouterr().out
    assert '"speed_unit": "kph"' in output


def test_cli_overrides_realtime_pacing(capsys) -> None:
    assert main(["config", "--realtime"]) == 0
    assert '"realtime": true' in capsys.readouterr().out

    assert main(["config", "--realtime-factor", "0.5"]) == 0
    output = capsys.readouterr().out
    assert '"realtime": true' in output
    assert '"realtime_factor": 0.5' in output


def test_cli_overrides_headway_speed_cap_mode(capsys) -> None:
    assert main(["config", "--headway-speed-cap", "enforce"]) == 0
    assert '"headway_speed_cap_mode": "enforce"' in capsys.readouterr().out

    assert main(
        [
            "config",
            "--config",
            "configs/demo-vla-qwen3-vl-2b-mlx.yaml",
            "--unpaced",
        ]
    ) == 0
    assert '"realtime": false' in capsys.readouterr().out


def test_cli_overrides_action_speed_policy_mode(capsys) -> None:
    assert main(["config", "--action-speed-policy", "shadow"]) == 0
    assert '"action_speed_policy_mode": "shadow"' in capsys.readouterr().out


def test_cli_seeds_every_http_model_request_for_one_run(capsys) -> None:
    assert main(["config", "--config", "configs/vla-probe-smolvlm-mlx.yaml", "--seed", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["vla"]["http"]["seed"] == 3

    assert main(["config", "--config", "configs/vla-probe-smolvlm-mlx.yaml"]) == 0
    assert json.loads(capsys.readouterr().out)["vla"]["http"]["seed"] is None


def test_cli_refuses_a_seed_that_no_request_would_carry(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["config", "--seed", "-1"])
    with pytest.raises(SystemExit):
        main(["run", "--config", "configs/demo-vla-fixture.yaml", "--seed", "1", "--dry-run"])
    assert "uses the fixture provider" in capsys.readouterr().err


def test_cli_probe_asks_the_model_under_the_given_seed(monkeypatch, capsys, tmp_path) -> None:
    received: dict[str, object] = {}
    monkeypatch.setattr(
        "metadrive_starter.cli.build_vla_provider",
        lambda settings: received.update(settings=settings) or object(),
    )
    monkeypatch.setattr(
        "metadrive_starter.cli.replay_vla_probe_artifacts",
        lambda config, source_dir, output_dir, **kwargs: ProbeRunSummary(
            output_dir=str(output_dir),
            inference_attempted=True,
            scenarios=(ProbeScenarioResult("clear-road", "success", str(output_dir)),),
        ),
    )

    assert main(
        [
            "probe",
            "--config",
            "configs/vla-probe-qwen3-vl-2b-llama.yaml",
            "--infer",
            "--replay-from",
            str(tmp_path / "source"),
            "--output-dir",
            str(tmp_path / "output"),
            "--seed",
            "2",
        ]
    ) == 0

    assert received["settings"].http.seed == 2
    assert received["settings"].http.temperature == 0.7


def test_cli_rejects_invalid_or_conflicting_realtime_factor() -> None:
    with pytest.raises(SystemExit):
        main(["config", "--realtime-factor", "0"])
    with pytest.raises(SystemExit):
        main(["config", "--unpaced", "--realtime-factor", "0.5"])


def test_speed_display_converts_only_when_kph_is_selected() -> None:
    assert _format_display_speed(10.0, "mps") == "10.0 m/s"
    assert _format_display_speed(10.0, "kph") == "36.0 km/h"


def test_explicit_autopilot_overrides_manual_config(capsys) -> None:
    assert main(["config", "--config", "configs/demo-manual.yaml", "--autopilot"]) == 0

    output = capsys.readouterr().out
    assert '"manual_control": false' in output


def test_cli_probe_runs_capture_only_selected_scenario(monkeypatch, capsys, tmp_path) -> None:
    received: dict[str, object] = {}

    monkeypatch.setattr(
        "metadrive_starter.cli.load_probe_scenarios",
        lambda path: received.update(manifest=path) or ("scenario",),
    )

    def run(config, scenarios, output_dir, **kwargs):
        received.update(config=config, scenarios=scenarios, output_dir=output_dir, kwargs=kwargs)
        return ProbeRunSummary(
            output_dir=str(output_dir),
            inference_attempted=False,
            scenarios=(
                ProbeScenarioResult("clear-straight", "captured", str(output_dir)),
            ),
        )

    monkeypatch.setattr("metadrive_starter.cli.run_vla_probe_catalog", run)
    output_dir = tmp_path / "probe"

    assert main(
        [
            "probe",
            "--manifest",
            "fixtures.yaml",
            "--output-dir",
            str(output_dir),
            "--scenario",
            "clear-straight",
        ]
    ) == 0

    assert received["scenarios"] == ("scenario",)
    assert received["output_dir"] == output_dir
    assert received["kwargs"] == {
        "scenario_ids": ["clear-straight"],
        "infer": False,
        "provider": None,
        # Without --submission the probe builds DriveBench's original observation.
        "observation_builder": None,
    }
    assert '"successful": true' in capsys.readouterr().out


def test_cli_event_log_override_enables_structured_logging(capsys, tmp_path) -> None:
    path = tmp_path / "events.jsonl"

    assert main(["config", "--event-log", str(path)]) == 0

    output = capsys.readouterr().out
    assert '"event_log"' in output
    assert '"enabled": true' in output
    assert str(path) in output


def test_cli_replay_reports_no_replayable_events(monkeypatch, capsys, tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    with EventLogger(path, run_id="run-1") as logger:
        logger.write(
            "run_started",
            sim_time_s=0.0,
            payload={"config": config_from_dict({}).to_dict()},
        )

    assert main(["replay", "--event-log", str(path)]) == 1
    assert '"validation_events": 0' in capsys.readouterr().out


@pytest.mark.instructor
def test_cli_evaluate_delegates_manifest_and_output_without_launching_direct_run(
    monkeypatch, capsys, tmp_path
) -> None:
    received: dict[str, object] = {}
    plan = object()
    monkeypatch.setattr(
        "metadrive_starter.cli.load_evaluation_plan",
        lambda path: received.update(manifest=path) or plan,
    )

    def evaluate(config, selected_plan, output_dir, **kwargs):
        received.update(
            config=config,
            plan=selected_plan,
            output_dir=output_dir,
            kwargs=kwargs,
        )
        return EvaluationReport(
            output_dir=str(output_dir),
            runs=(),
            aggregate={"runs": 0},
            by_scenario={},
        )

    monkeypatch.setattr("metadrive_starter.cli.run_evaluation", evaluate)
    output = tmp_path / "evaluation"

    assert main(
        [
            "evaluate",
            "--manifest",
            "matrix.yaml",
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    ) == 0

    assert received["manifest"] == Path("matrix.yaml")
    assert received["plan"] is plan
    assert received["output_dir"] == output
    runner = received["kwargs"].pop("runner")
    assert received["kwargs"] == {"dry_run": True}
    # Without --submission, every run drives the instructor reference.
    assert runner.func is run_simulation
    assert runner.keywords["controller_factory"] is not None
    assert '"successful": true' in capsys.readouterr().out


@pytest.mark.instructor
def test_cli_race_delegates_locked_manifest_and_output(monkeypatch, capsys, tmp_path) -> None:
    received: dict[str, object] = {}
    plan = object()
    monkeypatch.setattr(
        "metadrive_starter.cli.load_race_plan",
        lambda path: received.update(manifest=path) or plan,
    )

    def race(config, selected_plan, output_dir, **kwargs):
        received.update(
            config=config,
            plan=selected_plan,
            output_dir=output_dir,
            kwargs=kwargs,
        )
        return RaceReport(
            suite_id="public-v1",
            visibility="public",
            rendered=True,
            manifest_sha256="a" * 64,
            base_config_sha256="b" * 64,
            output_dir=str(output_dir),
            runs=(),
            aggregate={"runs": 0},
        )

    monkeypatch.setattr("metadrive_starter.cli.run_race", race)
    output = tmp_path / "race"

    assert main(
        [
            "race",
            "--manifest",
            "race.yaml",
            "--output-dir",
            str(output),
            "--dry-run",
            "--render",
        ]
    ) == 0

    assert received["manifest"] == Path("race.yaml")
    assert received["plan"] is plan
    assert received["output_dir"] == output
    runner = received["kwargs"].pop("runner")
    assert received["kwargs"] == {"dry_run": True, "render": True}
    assert runner.func is run_simulation
    assert runner.keywords["controller_factory"] is not None
    assert '"suite_id": "public-v1"' in capsys.readouterr().out


@pytest.mark.parametrize("command", ["evaluate", "race"])
def test_cli_suite_drives_a_submission_where_there_is_no_reference(
    command,
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.setattr(submission, "INSTRUCTOR_DIR", tmp_path / "absent")
    team = tmp_path / "team"
    team.mkdir()
    # Every run must build this controller from the configuration agent.yaml tunes.
    _write_controller(
        team,
        "class Controller:\n"
        "    def __init__(self, settings):\n"
        "        assert settings.target_speed_mps == 5.0, 'agent.yaml not applied'\n"
        "    def update(self, tick):\n"
        "        return (0.0, 0.0)\n"
        "    def reset_speed_control(self):\n"
        "        pass\n",
    )
    (team / "agent.yaml").write_text("controller:\n  target_speed_mps: 5.0\n")

    assert main(
        [command, "--submission", str(team), "--output-dir", str(tmp_path / "out"), "--dry-run"]
    ) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["runs"]
    assert {run["summary"]["controller"] for run in report["runs"]} == {
        "submission.controller.Controller"
    }


@pytest.mark.parametrize("command", ["evaluate", "race"])
def test_cli_suite_asks_for_a_submission_when_there_is_no_reference(
    command,
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.setattr(submission, "INSTRUCTOR_DIR", tmp_path / "absent")
    output = tmp_path / "out"

    assert main([command, "--output-dir", str(output), "--dry-run"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "run with --submission submission" in captured.err
    assert not output.exists()


@pytest.mark.parametrize("command", ["evaluate", "race"])
def test_cli_suite_stops_before_running_a_submission_that_does_not_load(
    command,
    capsys,
    tmp_path,
) -> None:
    team = _write_controller(tmp_path, NEUTRAL_CONTROLLER)
    (team / "agent.yaml").write_text("simulator:\n  map: S\n")
    output = tmp_path / "out"

    assert main([command, "--submission", str(team), "--output-dir", str(output), "--dry-run"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "agent.yaml: simulator.map is not permitted in assignment 2" in captured.err
    assert not output.exists()


def test_cli_race_replay_requires_trust_and_delegates(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}

    def replay(path, **kwargs):
        received.update(path=path, kwargs=kwargs)
        return EpisodeReplaySummary(
            artifact_dir=str(path),
            payload_sha256="a" * 64,
            run_id="run-1",
            scenario_id="scenario-1",
            recorded_steps=10,
            replayed_steps=10,
            verified_control_frames=11,
            rendered=True,
            replay_done=True,
        )

    monkeypatch.setattr("metadrive_starter.cli.replay_episode_artifact", replay)

    with pytest.raises(SystemExit):
        main(["race-replay", "--artifact", "episode", "--render"])

    assert main(
        [
            "race-replay",
            "--artifact",
            "episode",
            "--render",
            "--unpaced",
            "--trust-artifact",
        ]
    ) == 0

    assert received == {
        "path": Path("episode"),
        "kwargs": {"trusted": True, "render": True, "paced": False},
    }
    assert '"verified_control_frames": 11' in capsys.readouterr().out


def test_cli_race_replay_delegates_complete_race_output(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}

    def replay(path, **kwargs):
        received.update(path=path, kwargs=kwargs)
        episode = EpisodeReplaySummary(
            artifact_dir="episode",
            payload_sha256="a" * 64,
            run_id="run-1",
            scenario_id="scenario-1",
            recorded_steps=10,
            replayed_steps=10,
            verified_control_frames=11,
            rendered=False,
            replay_done=True,
        )
        return RaceReplaySummary(
            race_output_dir=str(path),
            replays=(episode,),
        )

    monkeypatch.setattr("metadrive_starter.cli.replay_race_output", replay)

    assert main(
        [
            "race-replay",
            "--race-output",
            "race-results",
            "--unpaced",
            "--trust-artifact",
        ]
    ) == 0

    assert received == {
        "path": Path("race-results"),
        "kwargs": {"trusted": True, "render": False, "paced": False},
    }
    assert '"successful": true' in capsys.readouterr().out


def test_cli_cloud_check_delegates_without_launching_simulator(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}

    def run(settings, **kwargs):
        received.update(settings=settings, kwargs=kwargs)
        return VertexPreflightReport(
            project_id="project-123",
            location=settings.location,
            model_id=settings.model_id,
            checks=(PreflightCheck("sdk", True, "available"),),
        )

    monkeypatch.setattr("metadrive_starter.cli.run_vertex_preflight", run)

    assert main(
        [
            "cloud-check",
            "--project",
            "project-123",
            "--location",
            "europe-west1",
            "--model",
            "course-model",
            "--samples",
            "2",
        ]
    ) == 0

    settings = received["settings"]
    assert settings.location == "europe-west1"
    assert settings.model_id == "course-model"
    kwargs = received["kwargs"]
    assert kwargs["prompt_policy"].startswith("On a clear lane")
    assert {key: value for key, value in kwargs.items() if key != "prompt_policy"} == {
        "project_id": "project-123",
        "samples": 2,
        "request_timeout_s": 10.0,
    }
    assert '"successful": true' in capsys.readouterr().out


def test_cli_probe_replays_saved_artifacts_without_loading_manifest(
    monkeypatch, capsys, tmp_path
) -> None:
    received: dict[str, object] = {}
    provider = object()
    monkeypatch.setattr(
        "metadrive_starter.cli.build_vla_provider",
        lambda settings: provider,
    )
    monkeypatch.setattr(
        "metadrive_starter.cli.load_probe_scenarios",
        lambda path: (_ for _ in ()).throw(AssertionError("manifest should not load")),
    )

    def replay(config, source_dir, output_dir, **kwargs):
        received.update(
            config=config,
            source_dir=source_dir,
            output_dir=output_dir,
            kwargs=kwargs,
        )
        return ProbeRunSummary(
            output_dir=str(output_dir),
            inference_attempted=True,
            scenarios=(ProbeScenarioResult("clear-road", "success", str(output_dir)),),
        )

    monkeypatch.setattr("metadrive_starter.cli.replay_vla_probe_artifacts", replay)
    source = tmp_path / "source"
    output = tmp_path / "output"

    assert main(
        [
            "probe",
            "--config",
            "configs/vla-probe-qwen3-vl-2b-mlx.yaml",
            "--infer",
            "--replay-from",
            str(source),
            "--output-dir",
            str(output),
            "--scenario",
            "clear-road",
        ]
    ) == 0

    assert received["source_dir"] == source
    assert received["output_dir"] == output
    assert received["kwargs"] == {
        "scenario_ids": ["clear-road"],
        "provider": provider,
    }
    assert '"successful": true' in capsys.readouterr().out


def test_cli_vertex_capture_requires_confirmation_and_delegates(
    monkeypatch, capsys, tmp_path
) -> None:
    received: dict[str, object] = {}

    def capture(config, source_dir, output_dir, **kwargs):
        received.update(
            config=config,
            source_dir=source_dir,
            output_dir=output_dir,
            kwargs=kwargs,
        )
        preflight = VertexPreflightReport(
            project_id="project-123",
            location=kwargs["vertex_settings"].location,
            model_id=kwargs["vertex_settings"].model_id,
            checks=(PreflightCheck("fake", True, "passed"),),
        )
        probes = ProbeRunSummary(
            output_dir=str(output_dir),
            inference_attempted=True,
            scenarios=(
                ProbeScenarioResult("clear-straight", "success", str(output_dir)),
            ),
        )
        return VertexCaptureReport(
            output_dir=str(output_dir),
            source_dir=str(source_dir),
            scenario_ids=("clear-straight",),
            request_cap=5,
            planned_requests=5,
            used_requests=5,
            request_labels=("one", "two", "three", "four", "five"),
            preflight=preflight,
            probes=probes,
        )

    monkeypatch.setattr("metadrive_starter.cli.run_vertex_capture", capture)
    source = tmp_path / "inputs"
    output = tmp_path / "evidence"

    with pytest.raises(SystemExit):
        main(["vertex-capture", "--replay-from", str(source)])

    assert main(
        [
            "vertex-capture",
            "--replay-from",
            str(source),
            "--output-dir",
            str(output),
            "--project",
            "project-123",
            "--location",
            "global",
            "--model",
            "course-model",
            "--execute",
        ]
    ) == 0

    assert received["source_dir"] == source
    assert received["output_dir"] == output
    assert received["kwargs"]["project_id"] == "project-123"
    assert received["kwargs"]["maximum_requests"] == 5
    assert received["kwargs"]["vertex_settings"].model_id == "course-model"
    assert '"hard_cap": 5' in capsys.readouterr().out


def test_cli_provider_fixtures_requires_review_and_delegates(
    monkeypatch, capsys, tmp_path
) -> None:
    received: dict[str, object] = {}

    def convert(source_dir, output_dir, **kwargs):
        received.update(
            source_dir=source_dir,
            output_dir=output_dir,
            kwargs=kwargs,
        )
        return ProviderFixtureConversionReport(
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            response_ids_retained=False,
            fixtures=(
                ProviderFixtureConversion(
                    scenario_id="clear-straight",
                    fixture_id="vertex-clear-straight",
                    path=str(output_dir / "01-clear-straight.json"),
                    artifact_sha256="a" * 64,
                    semantic_passed=True,
                ),
            ),
        )

    monkeypatch.setattr(
        "metadrive_starter.cli.convert_probe_capture_to_provider_fixtures",
        convert,
    )
    source = tmp_path / "capture"
    output = tmp_path / "fixtures"

    with pytest.raises(SystemExit):
        main(
            [
                "provider-fixtures",
                "--source-dir",
                str(source),
                "--output-dir",
                str(output),
                "--fixture-prefix",
                "vertex",
            ]
        )

    assert main(
        [
            "provider-fixtures",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
            "--fixture-prefix",
            "vertex",
            "--scenario",
            "clear-straight",
            "--reviewed",
        ]
    ) == 0

    assert received == {
        "source_dir": source,
        "output_dir": output,
        "kwargs": {
            "fixture_prefix": "vertex",
            "scenario_ids": ["clear-straight"],
            "retain_response_ids": False,
        },
    }
    assert '"successful": true' in capsys.readouterr().out
