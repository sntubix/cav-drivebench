from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter.config import (
    EventLogSettings,
    FixtureProviderSettings,
    ScenarioVehicleSettings,
    config_from_dict,
    load_config,
)
from metadrive_starter.events import read_event_log
from metadrive_starter.faults import FaultSpec
from metadrive_starter.replay import replay_event_log
from metadrive_starter.simulation import run_simulation
from metadrive_starter.vla import (
    ModelRequest,
    ModelResponse,
    VLA_PROMPT_CONTRACT_VERSION,
    VLAInferencePipeline,
    VLAInferenceScheduler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChangeLaneRightProvider:
    def __init__(self) -> None:
        self._sent_lane_change = False

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        del timeout_s
        action = "KEEP_LANE" if self._sent_lane_change else "CHANGE_LANE_RIGHT"
        self._sent_lane_change = True
        return ModelResponse(
            request_id=request.request_id,
            text=json.dumps(
                {
                    "scene_summary": "Stopped vehicle ahead; right lane clear.",
                    "relevant_hazards": [
                        {
                            "type": "vehicle",
                            "relative_location": "front",
                            "risk": "medium",
                        }
                    ],
                    "meta_action": action,
                    "target_speed_mps": 4.0,
                    "confidence": 0.95,
                    "brief_justification": "Use clear adjacent lane.",
                }
            ),
            model_id="integration-scripted-vlm",
            latency_s=0.0,
        )


class FollowProvider:
    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        del timeout_s
        return ModelResponse(
            request_id=request.request_id,
            text=json.dumps(
                {
                    "scene_summary": "Slower vehicle ahead.",
                    "relevant_hazards": [
                        {
                            "type": "vehicle",
                            "relative_location": "front",
                            "risk": "medium",
                        }
                    ],
                    "meta_action": "FOLLOW",
                    "target_speed_mps": 12.0,
                    "confidence": 0.95,
                    "brief_justification": "Maintain local time headway.",
                }
            ),
            model_id="integration-scripted-vlm",
            latency_s=0.0,
        )


@pytest.mark.instructor
def test_runtime_spawns_configured_relative_scenario_vehicle(
    tmp_path: Path,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "lane-change-fixture.yaml")
    config.simulator.headless = True
    config.simulator.horizon = 1
    config.simulator.spawn_longitude_m = 30.0
    config.vla = replace(config.vla, enabled=False)
    config.scenario.vehicles = (
        ScenarioVehicleSettings(
            vehicle_id="fast-rear-right",
            longitudinal_offset_m=-18.0,
            lane_offset=1,
            speed_mps=12.0,
        ),
    )
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "relative-vehicle-events.jsonl"),
        scenario_id="relative-vehicle",
    )

    run_simulation(config)
    records = read_event_log(config.event_log.path)
    spawn = next(
        record for record in records if record.event_type == "scenario_vehicle_spawned"
    )

    assert spawn.payload["vehicle_id"] == "fast-rear-right"
    assert spawn.payload["longitudinal_offset_m"] == -18.0
    assert spawn.payload["lane_offset"] == 1
    assert spawn.payload["speed_mps"] == 12.0


@pytest.mark.instructor
def test_headless_runtime_executes_validated_command_and_replays_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pace_calls: dict[str, object] = {"reset": None, "wait": []}

    class RecordingPacer:
        def __init__(self, *, realtime_factor: float = 1.0) -> None:
            pace_calls["factor"] = realtime_factor

        def reset(self, simulation_time_s: float = 0.0) -> None:
            pace_calls["reset"] = simulation_time_s

        def wait(self, simulation_time_s: float) -> float:
            pace_calls["wait"].append(simulation_time_s)  # type: ignore[union-attr]
            return 0.0

    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.headless = True
    config.simulator.realtime = True
    config.simulator.realtime_factor = 0.5
    config.simulator.horizon = 12
    config.camera = replace(config.camera, enabled=True, width=160, height=90)
    config.vla = replace(
        config.vla,
        enabled=True,
        provider="fixture",
        minimum_interval_s=0.0,
        fixture=FixtureProviderSettings(
            path=str(
                PROJECT_ROOT
                / "fixtures"
                / "vla"
                / "providers"
                / "synthetic-keep-lane.json"
            ),
            repeat_last=True,
        ),
    )
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "events.jsonl"),
        scenario_id="vla-integration",
    )
    monkeypatch.setattr(
        "metadrive_starter.simulation.RealTimePacer",
        RecordingPacer,
    )

    summary = run_simulation(config)

    records = read_event_log(tmp_path / "events.jsonl")
    event_types = [record.event_type for record in records]
    submitted_ids = {
        record.payload["request_id"]
        for record in records
        if record.event_type == "inference_submitted"
    }
    terminal_records = [
        record
        for record in records
        if record.event_type in {"inference_completed", "inference_failed"}
    ]
    terminal_ids = {record.payload["request_id"] for record in terminal_records}
    submitted_identities = {
        (
            record.payload["request_id"],
            record.payload["episode_id"],
            record.payload["generation_id"],
        )
        for record in records
        if record.event_type == "inference_submitted"
    }
    terminal_identities = {
        (
            record.payload["request_id"],
            record.payload["episode_id"],
            record.payload["generation_id"],
        )
        for record in terminal_records
    }
    shutdown_records = [
        record
        for record in terminal_records
        if record.payload.get("disposition") == "discarded_simulation_ended"
    ]
    replay = replay_event_log(tmp_path / "events.jsonl")
    cap_records = [
        record
        for record in records
        if record.event_type == "control_applied"
        and record.payload.get("headway_speed_cap") is not None
    ]
    assert summary.control_mode == "vla_autopilot"
    assert summary.run_id is not None
    assert summary.event_log_path == str((tmp_path / "events.jsonl").resolve())
    assert "inference_submitted" in event_types
    assert "inference_completed" in event_types
    assert "command_validation" in event_types
    assert "command_execution" in event_types
    assert "control_applied" in event_types
    assert "speed_pid_reset" in event_types
    assert event_types[-1] == "run_ended"
    assert terminal_ids == submitted_ids
    assert terminal_identities == submitted_identities
    assert len(shutdown_records) == 1
    assert replay.validation_events > 0
    assert replay.execution_events > 0
    assert replay.safety_events == summary.steps
    assert replay.headway_cap_events == summary.headway_cap_evaluations
    assert len(cap_records) == summary.headway_cap_evaluations
    assert summary.headway_cap_would_intervene == 0
    assert summary.headway_cap_applied == 0
    metrics = summary.vla_metrics
    assert metrics.enabled is True
    assert metrics.prompt_contract_version == VLA_PROMPT_CONTRACT_VERSION
    assert metrics.requests_started > 0
    assert (
        metrics.responses_succeeded
        + metrics.responses_failed
        + metrics.responses_discarded
        == metrics.requests_started
    )
    assert metrics.responses_failed == 0
    assert metrics.authority_steps + metrics.fallback_steps == summary.steps
    assert metrics.authority_steps > 0
    assert metrics.validations_accepted > 0
    assert metrics.latency_p50_s == 0.125
    assert metrics.latency_p95_s == 0.125
    assert metrics.providers == ("synthetic_fixture",)
    assert metrics.model_ids == ("synthetic-fixture-model",)
    assert metrics.model_versions == ("synthetic-v1",)
    assert metrics.fixture_ids == ("synthetic-keep-lane",)
    assert metrics.fixture_sha256s == (
        "f1bcab570fbd88a919366cd602898e567d39b58294cf3a17ed302579ed5084f5",
    )
    assert metrics.responses_with_token_usage == (
        metrics.responses_succeeded + metrics.responses_discarded
    )
    assert metrics.total_tokens == metrics.responses_with_token_usage * 176
    completed_records = [
        record for record in records if record.event_type == "inference_completed"
    ]
    assert all(
        record.payload["prompt_contract_version"]
        == VLA_PROMPT_CONTRACT_VERSION
        for record in completed_records
    )
    assert all(
        record.payload["provider_metadata"]["model_version"]
        == "synthetic-v1"
        for record in completed_records
    )
    assert all(
        record.payload["headway_speed_cap"]["mode"] == "off"
        for record in cap_records
    )
    assert all(
        record.payload["control_dt_s"] == pytest.approx(0.1)
        for record in records
        if record.event_type == "control_applied"
    )
    assert replay.successful is True
    assert pace_calls["reset"] == 0.0
    assert pace_calls["factor"] == 0.5
    assert len(pace_calls["wait"]) == summary.steps  # type: ignore[arg-type]
    assert pace_calls["wait"][-1] == summary.simulation_time_s  # type: ignore[index]


@pytest.mark.parametrize("map_code", ["XTOC", "S", "C", "X", "O"])
@pytest.mark.instructor
def test_lane_change_completes_and_replays_across_map_shapes(
    map_code: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.map = map_code
    config.simulator.headless = True
    config.simulator.horizon = 120
    config.simulator.out_of_road_done = False
    config.camera = replace(config.camera, enabled=True, width=160, height=90)
    config.vla = replace(
        config.vla,
        enabled=True,
        minimum_interval_s=0.0,
        action_speed_policy_mode="enforce",
    )
    config.planner.lane_change_enabled = True
    config.planner.lane_change_transition_m = 18.0
    config.planner.lane_change_timeout_s = 8.0
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / f"lane-change-{map_code}-events.jsonl"),
        scenario_id=f"lane-change-{map_code}",
    )
    scheduler = VLAInferenceScheduler(
            VLAInferencePipeline(ChangeLaneRightProvider()),
        minimum_interval_s=0.0,
    )
    monkeypatch.setattr(
        "metadrive_starter.simulation.build_vla_scheduler",
        lambda settings, **_: scheduler,
    )

    summary = run_simulation(config)
    records = read_event_log(config.event_log.path)
    replay = replay_event_log(config.event_log.path)
    transitions = [
        record.payload["decision"]["phase"]
        for record in records
        if record.event_type == "lane_change_transition"
    ]

    assert summary.lane_changes_started == 1
    assert summary.lane_changes_completed == 1
    assert summary.lane_changes_aborted == 0
    assert transitions[:2] == ["started", "completed"]
    assert summary.went_off_road is False
    assert replay.lane_change_events == summary.steps
    assert replay.successful is True


@pytest.mark.instructor
def test_follow_forces_live_headway_enforcement_when_global_mode_is_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.headless = True
    config.simulator.horizon = 24
    config.camera = replace(config.camera, enabled=True, width=160, height=90)
    config.scenario = replace(config.scenario, stopped_vehicle_ahead_m=12.0)
    config.safety = replace(config.safety, headway_speed_cap_mode="off")
    config.vla = replace(
        config.vla,
        enabled=True,
        minimum_interval_s=0.0,
        action_speed_policy_mode="enforce",
    )
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "follow-events.jsonl"),
        scenario_id="follow-integration",
    )
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(FollowProvider()),
        minimum_interval_s=0.0,
    )
    monkeypatch.setattr(
        "metadrive_starter.simulation.build_vla_scheduler",
        lambda settings, **_: scheduler,
    )

    summary = run_simulation(config)
    records = read_event_log(config.event_log.path)
    follow_controls = [
        record
        for record in records
        if record.event_type == "control_applied"
        and record.payload.get("headway_speed_cap_trigger") == "follow_action"
    ]
    replay = replay_event_log(config.event_log.path)

    assert follow_controls
    assert all(
        record.payload["headway_speed_cap"]["mode"] == "enforce"
        for record in follow_controls
    )
    assert any(
        record.payload["headway_speed_cap"]["applied"] is True
        for record in follow_controls
    )
    assert summary.headway_cap_applied > 0
    assert replay.successful is True


@pytest.mark.instructor
def test_target_lane_hazard_aborts_manoeuvre_and_triggers_emergency_brake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.headless = True
    config.simulator.horizon = 80
    config.simulator.out_of_road_done = False
    config.simulator.crash_vehicle_done = False
    config.camera = replace(config.camera, enabled=True, width=160, height=90)
    config.vla = replace(
        config.vla,
        enabled=True,
        minimum_interval_s=0.0,
        action_speed_policy_mode="enforce",
    )
    config.planner.lane_change_enabled = True
    config.scenario.stopped_vehicle_ahead_m = None
    config.scenario.lane_change_hazard = replace(
        config.scenario.lane_change_hazard,
        enabled=True,
        distance_ahead_m=6.0,
    )
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "lane-hazard-events.jsonl"),
        scenario_id="lane-hazard-integration",
    )
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(ChangeLaneRightProvider()),
        minimum_interval_s=0.0,
    )
    monkeypatch.setattr(
        "metadrive_starter.simulation.build_vla_scheduler",
        lambda settings, **_: scheduler,
    )

    summary = run_simulation(config)
    records = read_event_log(config.event_log.path)
    spawn = next(
        record
        for record in records
        if record.event_type == "lane_change_hazard_spawned"
    )
    aborted_control = next(
        record
        for record in records
        if record.event_type == "control_applied"
        and record.payload["lane_change"]["phase"] == "aborted"
    )
    replay = replay_event_log(config.event_log.path)

    assert summary.lane_changes_started == 1
    assert summary.lane_changes_completed == 0
    assert summary.lane_changes_aborted == 1
    assert summary.lane_change_hazards_spawned == 1
    assert summary.emergency_brakes > 0
    assert spawn.payload["lane_offset"] == 1
    assert aborted_control.payload["lane_change_context"]["clearance"]["clear"] is False
    assert (
        aborted_control.payload["lane_change_context"]["clearance"]["object_id"]
        == spawn.payload["object_id"]
    )
    assert aborted_control.payload["safety_decision"]["level"] == "emergency"
    assert aborted_control.payload["applied_command"]["brake"] == 1.0
    assert replay.successful is True


@pytest.mark.instructor
def test_scheduled_lidar_dropout_is_invalidated_logged_and_replayable(
    tmp_path: Path,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.headless = True
    config.simulator.horizon = 4
    config.perception.safety_source = "lidar"
    config.faults = (
        FaultSpec("lidar-loss-1", "lidar_dropout", start_step=1, duration_steps=2),
    )
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "lidar-fault-events.jsonl"),
        scenario_id="lidar-fault-integration",
    )

    summary = run_simulation(config)

    records = read_event_log(config.event_log.path)
    control_records = [
        record for record in records if record.event_type == "control_applied"
    ]
    invalid_records = [
        record for record in control_records if record.payload["scene"]["valid"] is False
    ]
    assert summary.steps == 4
    assert summary.fault_activations == 1
    assert sum(record.event_type == "fault_activated" for record in records) == 1
    assert sum(record.event_type == "fault_cleared" for record in records) == 1
    assert len(invalid_records) == 2
    assert all(record.payload["scene"]["objects"] == [] for record in invalid_records)
    assert all(
        record.payload["safety_decision"]["level"] == "degraded"
        for record in invalid_records
    )
    assert replay_event_log(config.event_log.path).successful is True


@pytest.mark.parametrize(
    ("camera_enabled", "safety_enabled", "message"),
    [
        (False, True, "camera.enabled"),
        (True, False, "safety.enabled"),
    ],
)
def test_runtime_rejects_vla_without_required_local_boundaries(
    camera_enabled: bool,
    safety_enabled: bool,
    message: str,
) -> None:
    config = config_from_dict(
        {
            "camera": {"enabled": camera_enabled},
            "vla": {"enabled": True},
            "safety": {"enabled": safety_enabled},
        }
    )

    with pytest.raises(ValueError, match=message):
        run_simulation(config, dry_run=True)


def test_runtime_rejects_vla_authority_in_manual_mode() -> None:
    config = config_from_dict(
        {
            "simulator": {"manual_control": True},
            "camera": {"enabled": True},
            "vla": {"enabled": True},
            "safety": {"enabled": True},
        }
    )

    with pytest.raises(ValueError, match="manual control"):
        run_simulation(config, dry_run=True)
