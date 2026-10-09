from __future__ import annotations

import json
from pathlib import Path

import pytest

from metadrive_starter.config import SimulatorSettings, config_from_dict
from metadrive_starter.env import make_env
from metadrive_starter.episode_replay import (
    EpisodeArtifactError,
    replay_episode_artifact,
    replay_race_output,
)
from metadrive_starter.race import (
    build_race_config,
    load_race_plan,
    race_environment_sha256,
    race_manifest_sha256,
    run_race,
)
from metadrive_starter.simulation import RunSummary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.instructor
def test_checked_in_public_and_hidden_suites_use_same_locked_contract() -> None:
    public = load_race_plan(PROJECT_ROOT / "configs/race-public-v1.yaml")
    hidden = load_race_plan(
        PROJECT_ROOT / "course/instructor/race-hidden-v1.yaml"
    )

    assert public.visibility == "public"
    assert hidden.visibility == "hidden"
    assert len(public.scenarios) == 5
    assert len(hidden.scenarios) == 5
    assert public.runtime == hidden.runtime
    assert public.runtime.realtime_factor == 1.0
    assert {scenario.seed for scenario in public.scenarios}.isdisjoint(
        scenario.seed for scenario in hidden.scenarios
    )
    assert any(
        scenario.stopped_vehicle_ahead_m is not None
        for scenario in public.scenarios
    )
    assert any(
        scenario.stopped_vehicle_ahead_m is not None
        for scenario in hidden.scenarios
    )


def test_race_config_locks_world_and_retains_agent_tuning(tmp_path) -> None:
    plan = load_race_plan(PROJECT_ROOT / "configs/race-public-v1.yaml")
    scenario = plan.scenarios[0]
    base = config_from_dict(
        {
            "simulator": {
                "map": "R",
                "traffic_density": 0.9,
                "random_traffic": True,
                "traffic_mode": "respawn",
                "obstacle_probability": 0.9,
                "num_scenarios": 99,
                "start_seed": 99,
                "decision_repeat": 2,
                "physics_world_step_size": 0.01,
                "horizon": 2,
                "headless": False,
                "manual_control": True,
                "realtime": False,
                "realtime_factor": 0.25,
                "out_of_road_done": False,
                "crash_vehicle_done": False,
                "crash_object_done": False,
            },
            "controller": {"target_speed_mps": 7.25},
            "scenario": {
                "stopped_vehicle_ahead_m": 5.0,
                "traffic_light": {"enabled": True},
            },
            "faults": [
                {
                    "fault_id": "old-fault",
                    "kind": "camera_dropout",
                    "start_step": 0,
                    "duration_steps": 1,
                }
            ],
        }
    )

    config = build_race_config(
        base,
        plan,
        scenario,
        event_log_path=tmp_path / "events.jsonl",
    )

    assert config.controller.target_speed_mps == 7.25
    assert config.simulator.map == scenario.map
    assert config.simulator.traffic_density == scenario.traffic_density
    assert config.simulator.random_traffic is False
    assert config.simulator.traffic_mode == "trigger"
    assert config.simulator.obstacle_probability == scenario.obstacle_probability
    assert config.simulator.num_scenarios == 1
    assert config.simulator.start_seed == scenario.seed
    assert config.simulator.decision_repeat == plan.runtime.decision_repeat
    assert (
        config.simulator.physics_world_step_size
        == plan.runtime.physics_world_step_size
    )
    assert config.simulator.horizon == scenario.horizon
    assert config.simulator.headless is True
    assert config.simulator.manual_control is False
    assert config.simulator.realtime is True
    assert config.simulator.realtime_factor == 1.0
    assert config.simulator.out_of_road_done is True
    assert config.simulator.crash_vehicle_done is True
    assert config.simulator.crash_object_done is True
    assert config.scenario.stopped_vehicle_ahead_m == scenario.stopped_vehicle_ahead_m
    assert config.scenario.traffic_light.enabled is False
    assert config.scenario.lane_change_hazard.enabled is False
    assert config.faults == ()
    assert config.event_log.enabled is True
    assert config.event_log.scenario_id == scenario.scenario_id
    assert config.episode_recording.enabled is True
    assert config.episode_recording.path == str(
        (tmp_path / "replay").resolve()
    )

    rendered = build_race_config(
        base,
        plan,
        scenario,
        event_log_path=tmp_path / "rendered-events.jsonl",
        render=True,
    )
    assert rendered.simulator.headless is False
    assert race_environment_sha256(plan, scenario, render=True) != (
        race_environment_sha256(plan, scenario)
    )


def test_race_runs_each_locked_case_once_and_records_fingerprints(tmp_path) -> None:
    manifest = tmp_path / "race.yaml"
    manifest.write_text(
        """
version: 1
id: test-race
visibility: public
runtime:
  decision_repeat: 5
  physics_world_step_size: 0.02
  realtime_factor: 1.0
scenarios:
  - id: first
    seed: 10
    map: S
    traffic_density: 0.0
    obstacle_probability: 0.0
    horizon: 20
  - id: second
    seed: 11
    map: C
    traffic_density: 0.1
    obstacle_probability: 0.0
    horizon: 30
""",
        encoding="utf-8",
    )
    plan = load_race_plan(manifest)
    received = []

    def runner(config, *, dry_run: bool):
        received.append((config, dry_run))
        if config.simulator.map == "C":
            raise RuntimeError("injected failure")
        return RunSummary(
            steps=config.simulator.horizon,
            total_reward=3.0,
            control_mode="autopilot",
            route_completion=0.75,
            arrived=True,
        )

    output = tmp_path / "results"
    report = run_race(
        config_from_dict({}),
        plan,
        output,
        dry_run=True,
        runner=runner,
    )

    assert len(received) == 2
    assert all(dry_run for _, dry_run in received)
    assert [config.simulator.start_seed for config, _ in received] == [10, 11]
    assert report.successful is False
    assert report.rendered is False
    assert report.manifest_sha256 == race_manifest_sha256(plan)
    assert report.runs[0].environment_sha256 == race_environment_sha256(
        plan, plan.scenarios[0]
    )
    assert report.runs[0].environment_sha256 != report.runs[1].environment_sha256
    assert report.aggregate["runs"] == 2
    assert report.aggregate["completed"] == 1
    assert report.aggregate["failed"] == 1
    assert report.aggregate["arrival_rate"] == 1.0
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["manifest_sha256"] == report.manifest_sha256
    assert saved["runs"][1]["error_category"] == "RuntimeError"


@pytest.mark.parametrize(
    "content",
    [
        "version: 2\nid: race\nvisibility: public\nruntime: {}\nscenarios: []\n",
        "version: 1\nid: race\nvisibility: leaked\nruntime: {decision_repeat: 5, physics_world_step_size: 0.02, realtime_factor: 1.0}\nscenarios: [{id: s, seed: 0, map: S, traffic_density: 0, obstacle_probability: 0, horizon: 1}]\n",
        "version: 1\nid: race\nvisibility: public\nruntime: {decision_repeat: 5, physics_world_step_size: 0.02, realtime_factor: 0.5}\nscenarios: [{id: s, seed: 0, map: S, traffic_density: 0, obstacle_probability: 0, horizon: 1}]\n",
        "version: 1\nid: race\nvisibility: public\nruntime: {decision_repeat: 5, physics_world_step_size: 0.02, realtime_factor: 1.0}\nscenarios: [{id: s, seed: 0, map: S, traffic_density: 0, obstacle_probability: 0, horizon: 1, extra: true}]\n",
    ],
)
def test_race_manifest_rejects_invalid_or_ambiguous_inputs(
    tmp_path,
    content: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_race_plan(path)


def test_race_refuses_to_overwrite_artifact_root(tmp_path) -> None:
    plan = load_race_plan(PROJECT_ROOT / "configs/race-public-v1.yaml")
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        run_race(config_from_dict({}), plan, output, dry_run=True)


@pytest.mark.instructor
def test_real_headless_race_produces_replayable_episode_artifact(tmp_path) -> None:
    manifest = tmp_path / "race.yaml"
    manifest.write_text(
        """
version: 1
id: replay-race
visibility: public
runtime:
  decision_repeat: 5
  physics_world_step_size: 0.02
  realtime_factor: 1.0
scenarios:
  - id: short-run
    seed: 77
    map: S
    traffic_density: 0.0
    obstacle_probability: 0.0
    horizon: 3
    stopped_vehicle_ahead_m: 20.0
""",
        encoding="utf-8",
    )
    output = tmp_path / "results"

    report = run_race(
        config_from_dict({}),
        load_race_plan(manifest),
        output,
    )

    assert report.successful is True
    result = report.runs[0]
    assert result.replay_artifact_path is not None
    assert result.replay_artifact_sha256 is not None
    replay = replay_episode_artifact(
        result.replay_artifact_path,
        trusted=True,
        render=False,
        paced=False,
    )
    assert replay.successful is True
    assert replay.payload_sha256 == result.replay_artifact_sha256
    assert replay.recorded_steps == 3

    race_replay = replay_race_output(
        output,
        trusted=True,
        render=False,
        paced=False,
    )
    assert race_replay.successful is True
    assert tuple(item.scenario_id for item in race_replay.replays) == ("short-run",)


def test_race_replay_refuses_silent_partial_suite(tmp_path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "scenario_id": "missing-artifact",
                        "status": "completed",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EpisodeArtifactError, match="has no replay artifact"):
        replay_race_output(output, trusted=True, render=False, paced=False)


def test_seeded_non_random_traffic_repeats_initial_trajectory() -> None:
    settings = SimulatorSettings(
        map="XTOC",
        traffic_density=0.2,
        random_traffic=False,
        traffic_mode="trigger",
        obstacle_probability=0.0,
        num_scenarios=1,
        start_seed=1105,
        horizon=120,
        headless=True,
    )

    def trace() -> tuple[tuple[tuple[float, float], ...], ...]:
        env = make_env(settings)
        samples = []
        try:
            env.reset()
            for _ in range(100):
                samples.append(
                    tuple(
                        sorted(
                            (
                                round(float(vehicle.position[0]), 6),
                                round(float(vehicle.position[1]), 6),
                            )
                            for vehicle in env.engine.traffic_manager.traffic_vehicles
                        )
                    )
                )
                env.step([0.0, 1.0])
        finally:
            env.close()
        return tuple(samples)

    first = trace()
    second = trace()

    assert first
    assert any(sample for sample in first)
    assert second == first
