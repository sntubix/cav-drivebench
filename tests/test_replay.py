from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from metadrive_starter.config import config_from_dict
from metadrive_starter.events import EventLogger
from metadrive_starter.perception import (
    LaneRelation,
    LocalScene,
    TrackedObject,
    TrafficLightObservation,
    TrafficLightState,
)
from metadrive_starter.planning.command_executor import (
    CommandExecutionDecision,
    CommandExecutionSource,
    VLACommandExecutor,
)
from metadrive_starter.planning.lane_change import LaneChangeCoordinator
from metadrive_starter.replay import replay_event_log
from metadrive_starter.safety import (
    EmergencyBrakingSupervisor,
    TimeHeadwaySpeedCap,
    VLACommandValidator,
)
from metadrive_starter.types import ControlCommand
from metadrive_starter.vla import HighLevelAction, VLACommand


def _scene() -> LocalScene:
    return LocalScene(
        timestamp_s=1.0,
        ego_speed_mps=8.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        objects=(
            TrackedObject(
                object_id="lead",
                kind="vehicle",
                relative_position_m=(18.0, 0.0),
                relative_velocity_mps=(-2.0, 0.0),
                length_m=4.5,
                width_m=1.8,
                lane_relation=LaneRelation.SAME,
                in_path=True,
                path_distance_m=18.0,
            ),
        ),
        traffic_lights=(
            TrafficLightObservation(
                light_id="signal",
                state=TrafficLightState.RED,
                relative_position_m=(30.0, 0.0),
                in_path=True,
                path_distance_m=30.0,
            ),
        ),
        left_lane_available=True,
        right_lane_available=False,
    )


def _write_replayable_log(path: Path) -> None:
    config = config_from_dict(
        {"safety": {"enabled": True, "headway_speed_cap_mode": "enforce"}}
    )
    command = VLACommand(
        action=HighLevelAction.SLOW_DOWN,
        target_speed_mps=35.0,
        issued_at_s=1.0,
        action_horizon_s=2.0,
        confidence=0.9,
        command_id="model-1",
    )
    scene = _scene()
    decision = VLACommandValidator(config.command_validation).validate(
        command, scene, now_s=1.0
    )
    execution = VLACommandExecutor(config.controller.target_speed_mps).execute(
        decision, now_s=1.0
    )
    proposed = ControlCommand(steering=0.1, throttle=0.5, brake=0.0)
    safety_decision = EmergencyBrakingSupervisor(config.safety).evaluate(
        scene, proposed, now_s=1.0
    )
    headway_decision = TimeHeadwaySpeedCap(
        mode=config.safety.headway_speed_cap_mode,
        minimum_gap_m=config.safety.minimum_gap_m,
        time_headway_s=config.safety.minimum_headway_s,
        scene_stale_after_s=config.safety.stale_after_s,
    ).evaluate(scene, execution.target_speed_mps, now_s=1.0)
    with EventLogger(path, run_id="run-1") as logger:
        logger.write(
            "run_started",
            sim_time_s=0.0,
            payload={"config": config.to_dict()},
        )
        logger.write(
            "command_validation",
            sim_time_s=1.0,
            payload={
                "now_s": 1.0,
                "requested_command": command,
                "scene": scene,
                "decision": decision,
            },
        )
        logger.write(
            "command_execution",
            sim_time_s=1.0,
            payload={
                "now_s": 1.0,
                "validation_decision": decision,
                "execution": execution,
            },
        )
        logger.write(
            "control_applied",
            sim_time_s=1.0,
            payload={
                "now_s": 1.0,
                "scene": scene,
                "raw_target_speed_mps": execution.target_speed_mps,
                "headway_speed_cap": headway_decision,
                "proposed_command": proposed,
                "safety_decision": safety_decision,
            },
        )


def _write_follow_log(path: Path) -> None:
    config = config_from_dict(
        {"safety": {"enabled": True, "headway_speed_cap_mode": "off"}}
    )
    scene = _scene()
    command = VLACommand(
        action=HighLevelAction.FOLLOW,
        target_speed_mps=12.0,
        issued_at_s=1.0,
        action_horizon_s=2.0,
        confidence=0.9,
        command_id="follow-1",
    )
    validation = VLACommandValidator(config.command_validation).validate(
        command,
        scene,
        now_s=1.0,
    )
    execution = VLACommandExecutor(config.controller.target_speed_mps).execute(
        validation,
        now_s=1.0,
    )
    headway = TimeHeadwaySpeedCap(
        mode="enforce",
        minimum_gap_m=config.safety.minimum_gap_m,
        time_headway_s=config.safety.minimum_headway_s,
        scene_stale_after_s=config.safety.stale_after_s,
    ).evaluate(scene, execution.target_speed_mps, now_s=1.0)
    with EventLogger(path, run_id="follow-run") as logger:
        logger.write(
            "run_started",
            sim_time_s=0.0,
            payload={"config": config.to_dict()},
        )
        logger.write(
            "control_applied",
            sim_time_s=1.0,
            payload={
                "now_s": 1.0,
                "scene": scene,
                "execution": execution,
                "raw_target_speed_mps": execution.target_speed_mps,
                "headway_speed_cap_trigger": "follow_action",
                "headway_speed_cap": headway,
                "lane_change": None,
                "lane_change_context": None,
                "safety_decision": None,
            },
        )


def _write_lane_change_log(path: Path) -> None:
    config = config_from_dict(
        {
            "planner": {"lane_change_enabled": True},
            "safety": {"enabled": True},
        }
    )
    coordinator = LaneChangeCoordinator(
        timeout_s=config.planner.lane_change_timeout_s,
        completion_tolerance_m=config.planner.lane_change_completion_tolerance_m,
    )
    validator = VLACommandValidator(config.command_validation)
    execution = CommandExecutionDecision(
        source=CommandExecutionSource.VLA,
        action=HighLevelAction.CHANGE_LANE_RIGHT,
        target_speed_mps=5.0,
        command_id="lane-1",
        reason="test lane request",
        valid_until_s=5.0,
    )
    clear_scene = replace(
        _scene(),
        objects=(),
        left_lane_available=False,
        right_lane_available=True,
    )
    blocked_scene = replace(
        clear_scene,
        timestamp_s=1.2,
        objects=(
            TrackedObject(
                object_id="right-front",
                kind="vehicle",
                relative_position_m=(8.0, -3.5),
                relative_velocity_mps=(0.0, 0.0),
                length_m=4.5,
                width_m=1.8,
                lane_relation=LaneRelation.RIGHT,
            ),
        ),
    )
    ticks = (
        (1.0, clear_scene, execution),
        (1.1, replace(clear_scene, timestamp_s=1.1), execution),
        (1.2, blocked_scene, execution),
    )
    with EventLogger(path, run_id="lane-run") as logger:
        logger.write(
            "run_started",
            sim_time_s=0.0,
            payload={"config": config.to_dict()},
        )
        for now_s, scene, tick_execution in ticks:
            target_lane_index = coordinator.target_lane_index
            clearance = None
            target_lane_clear = None
            clearance_reason = None
            if target_lane_index is not None:
                clearance = validator.lane_change_clearance(
                    scene,
                    LaneRelation.RIGHT,
                    now_s=now_s,
                )
                target_lane_clear = clearance.clear
                clearance_reason = clearance.reason
            decision = coordinator.update(
                tick_execution,
                now_s=now_s,
                current_lane_index=0,
                lane_count=2,
                target_lane_offset_m=None,
                target_lane_clear=target_lane_clear,
                clearance_reason=clearance_reason,
            )
            logger.write(
                "control_applied",
                sim_time_s=now_s,
                payload={
                    "now_s": now_s,
                    "scene": scene,
                    "execution": tick_execution,
                    "headway_speed_cap": None,
                    "lane_change": decision,
                    "lane_change_context": {
                        "current_lane_index": 0,
                        "lane_count": 2,
                        "target_lane_offset_m": None,
                        "clearance": clearance,
                        "forced_abort_reason": None,
                    },
                    "safety_decision": None,
                },
            )


def test_replay_reproduces_recorded_command_validation(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_replayable_log(path)

    summary = replay_event_log(path)

    assert summary.successful is True
    assert summary.validation_events == 1
    assert summary.execution_events == 1
    assert summary.safety_events == 1
    assert summary.headway_cap_events == 1
    assert summary.matched_events == 4
    assert summary.mismatches == ()


def test_replay_reports_a_tampered_validation_decision(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_replayable_log(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[1])
    event["payload"]["decision"]["effective_command"]["target_speed_mps"] = 19.0
    lines[1] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = replay_event_log(path)

    assert summary.successful is False
    assert summary.matched_events == 3
    assert summary.mismatches[0].reason == "replayed validation decision differs"


def test_replay_reports_a_tampered_headway_cap_decision(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_replayable_log(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[3])
    event["payload"]["headway_speed_cap"]["calculated_cap_mps"] = 99.0
    lines[3] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = replay_event_log(path)

    assert summary.successful is False
    assert summary.matched_events == 3
    assert summary.mismatches[0].reason == "replayed headway cap decision differs"


def test_follow_forces_enforced_headway_even_when_global_cap_is_off(
    tmp_path: Path,
) -> None:
    path = tmp_path / "follow-events.jsonl"
    _write_follow_log(path)

    summary = replay_event_log(path)
    control = json.loads(path.read_text(encoding="utf-8").splitlines()[1])

    assert control["payload"]["headway_speed_cap"]["mode"] == "enforce"
    assert control["payload"]["headway_speed_cap"]["applied"] is True
    assert summary.headway_cap_events == 1
    assert summary.matched_events == 1
    assert summary.successful is True


def test_replay_reproduces_lane_change_state_and_dynamic_abort(tmp_path: Path) -> None:
    path = tmp_path / "lane-events.jsonl"
    _write_lane_change_log(path)

    summary = replay_event_log(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert [record["payload"]["lane_change"]["phase"] for record in records[1:]] == [
        "started",
        "active",
        "aborted",
    ]
    assert summary.lane_change_events == 3
    assert summary.matched_events == 3
    assert summary.successful is True


def test_replay_reports_tampered_lane_change_clearance(tmp_path: Path) -> None:
    path = tmp_path / "lane-events.jsonl"
    _write_lane_change_log(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[2])
    event["payload"]["lane_change_context"]["clearance"]["clear"] = False
    lines[2] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = replay_event_log(path)

    assert summary.successful is False
    assert summary.lane_change_events == 3
    assert any(
        "replayed lane-change clearance decision differs" in mismatch.reason
        for mismatch in summary.mismatches
    )


def test_replay_reports_removed_lane_change_state(tmp_path: Path) -> None:
    path = tmp_path / "lane-events.jsonl"
    _write_lane_change_log(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[1])
    event["payload"]["lane_change"] = None
    event["payload"]["lane_change_context"] = None
    lines[1] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = replay_event_log(path)

    assert summary.successful is False
    assert summary.lane_change_events == 3
    assert any(
        "must both be present" in mismatch.reason
        for mismatch in summary.mismatches
    )
