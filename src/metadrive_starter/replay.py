from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from metadrive_starter.config import AppConfig, config_from_dict
from metadrive_starter.events import EventLogError, EventRecord, read_event_log, to_json_value
from metadrive_starter.perception import (
    LaneRelation,
    LocalScene,
    TrackedObject,
    TrafficLightObservation,
    TrafficLightState,
)
from metadrive_starter.planning import ActionSpeedPolicy
from metadrive_starter.planning.command_executor import (
    CommandExecutionDecision,
    CommandExecutionSource,
    VLACommandExecutor,
)
from metadrive_starter.planning.lane_change import LaneChangeCoordinator
from metadrive_starter.safety import (
    CommandDisposition,
    CommandValidationDecision,
    EmergencyBrakingSupervisor,
    TimeHeadwaySpeedCap,
    VLACommandValidator,
)
from metadrive_starter.types import ControlCommand
from metadrive_starter.vla import HighLevelAction, VLACommand


@dataclass(frozen=True)
class ReplayMismatch:
    run_id: str
    sequence: int
    event_type: str
    reason: str


@dataclass(frozen=True)
class ReplaySummary:
    event_log_path: str
    total_events: int
    runs: int
    validation_events: int
    execution_events: int
    safety_events: int
    headway_cap_events: int
    lane_change_events: int
    matched_events: int
    mismatches: tuple[ReplayMismatch, ...]

    @property
    def successful(self) -> bool:
        replayable_events = (
            self.validation_events
            + self.execution_events
            + self.safety_events
            + self.headway_cap_events
            + self.lane_change_events
        )
        return replayable_events > 0 and not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_log_path": self.event_log_path,
            "total_events": self.total_events,
            "runs": self.runs,
            "validation_events": self.validation_events,
            "execution_events": self.execution_events,
            "safety_events": self.safety_events,
            "headway_cap_events": self.headway_cap_events,
            "lane_change_events": self.lane_change_events,
            "matched_events": self.matched_events,
            "successful": self.successful,
            "mismatches": [to_json_value(mismatch) for mismatch in self.mismatches],
        }


def replay_event_log(path: Path | str) -> ReplaySummary:
    """Re-run deterministic validation, execution, and safety decisions."""
    records = read_event_log(path)
    configs: dict[str, AppConfig] = {}
    mismatches: list[ReplayMismatch] = []
    validation_events = 0
    execution_events = 0
    safety_events = 0
    headway_cap_events = 0
    lane_change_events = 0
    matched_events = 0
    executors: dict[str, VLACommandExecutor] = {}
    supervisors: dict[str, EmergencyBrakingSupervisor] = {}
    headway_caps: dict[tuple[str, str], TimeHeadwaySpeedCap] = {}
    lane_change_coordinators: dict[str, LaneChangeCoordinator] = {}
    lane_change_validators: dict[str, VLACommandValidator] = {}

    for record in records:
        if record.event_type == "run_started":
            try:
                configs[record.run_id] = _config_from_event(record)
            except (TypeError, ValueError) as exc:
                mismatches.append(_mismatch(record, f"invalid run configuration: {exc}"))
            continue
        config = configs.get(record.run_id)
        if record.event_type == "command_validation":
            validation_events += 1
            if config is None:
                mismatches.append(_mismatch(record, "run_started configuration is unavailable"))
                continue
            try:
                requested = _command(record.payload.get("requested_command"))
                if requested is None:
                    raise EventLogError("requested_command must not be null")
                scene = scene_from_payload(record.payload.get("scene"))
                now_s = _number(record.payload, "now_s")
                expected = _mapping(record.payload.get("decision"), "decision")
                actual = VLACommandValidator(config.command_validation).validate(
                    requested,
                    scene,
                    now_s=now_s,
                )
            except (KeyError, TypeError, ValueError) as exc:
                mismatches.append(_mismatch(record, f"cannot replay validation: {exc}"))
                continue
            if to_json_value(actual) != dict(expected):
                mismatches.append(_mismatch(record, "replayed validation decision differs"))
                continue
            matched_events += 1
            continue

        if record.event_type == "command_execution":
            execution_events += 1
            if config is None:
                mismatches.append(_mismatch(record, "run_started configuration is unavailable"))
                continue
            try:
                decision = _validation_decision(
                    record.payload.get("validation_decision")
                )
                now_s = _number(record.payload, "now_s")
                expected = _mapping(record.payload.get("execution"), "execution")
                executor = executors.setdefault(
                    record.run_id,
                    VLACommandExecutor(
                        config.controller.target_speed_mps,
                        action_speed_policy=ActionSpeedPolicy(
                            config.vla.action_speed_policy_mode,
                            cruise_speed_mps=config.controller.target_speed_mps,
                            slow_down_speed_mps=(
                                config.command_validation.slow_down_speed_mps
                            ),
                            yield_speed_mps=config.command_validation.yield_speed_mps,
                        ),
                        lane_change_enabled=config.planner.lane_change_enabled,
                    ),
                )
                actual = executor.execute(decision, now_s=now_s)
            except (KeyError, TypeError, ValueError) as exc:
                mismatches.append(_mismatch(record, f"cannot replay execution: {exc}"))
                continue
            if to_json_value(actual) != dict(expected):
                mismatches.append(_mismatch(record, "replayed execution decision differs"))
                continue
            matched_events += 1
            continue

        if record.event_type != "control_applied":
            continue
        if record.payload.get("headway_speed_cap") is not None:
            headway_cap_events += 1
            if config is None:
                mismatches.append(
                    _mismatch(record, "run_started configuration is unavailable")
                )
            else:
                try:
                    scene = scene_from_payload(record.payload.get("scene"))
                    now_s = _number(record.payload, "now_s")
                    raw_target_speed_mps = _number(
                        record.payload,
                        "raw_target_speed_mps",
                    )
                    expected = _mapping(
                        record.payload.get("headway_speed_cap"),
                        "headway_speed_cap",
                    )
                    trigger = _headway_speed_cap_trigger(record.payload)
                    cap = headway_caps.setdefault(
                        (record.run_id, trigger),
                        TimeHeadwaySpeedCap(
                            mode=(
                                "enforce"
                                if trigger == "follow_action"
                                else config.safety.headway_speed_cap_mode
                            ),
                            minimum_gap_m=config.safety.minimum_gap_m,
                            time_headway_s=config.safety.minimum_headway_s,
                            scene_stale_after_s=config.safety.stale_after_s,
                        ),
                    )
                    actual = cap.evaluate(
                        scene,
                        raw_target_speed_mps,
                        now_s=now_s,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    mismatches.append(
                        _mismatch(record, f"cannot replay headway cap: {exc}")
                    )
                else:
                    if to_json_value(actual) != dict(expected):
                        mismatches.append(
                            _mismatch(record, "replayed headway cap decision differs")
                        )
                    else:
                        matched_events += 1

        lane_value = record.payload.get("lane_change")
        lane_context_value = record.payload.get("lane_change_context")
        lane_replay_required = (
            config is not None and config.planner.lane_change_enabled
        )
        if lane_replay_required or lane_value is not None or lane_context_value is not None:
            lane_change_events += 1
            if config is None:
                mismatches.append(
                    _mismatch(record, "run_started configuration is unavailable")
                )
            elif not config.planner.lane_change_enabled:
                mismatches.append(
                    _mismatch(record, "lane-change event recorded while feature is disabled")
                )
            else:
                try:
                    if lane_value is None or lane_context_value is None:
                        raise EventLogError(
                            "lane_change and lane_change_context must both be present"
                        )
                    expected_lane = _mapping(lane_value, "lane_change")
                    context = _mapping(lane_context_value, "lane_change_context")
                    now_s = _number(record.payload, "now_s")
                    current_lane_index = _integer(context, "current_lane_index")
                    lane_count = _integer(context, "lane_count")
                    target_lane_offset_m = _optional_number(
                        context,
                        "target_lane_offset_m",
                    )
                    execution = _execution_decision(record.payload.get("execution"))
                    coordinator = lane_change_coordinators.setdefault(
                        record.run_id,
                        LaneChangeCoordinator(
                            timeout_s=config.planner.lane_change_timeout_s,
                            completion_tolerance_m=(
                                config.planner.lane_change_completion_tolerance_m
                            ),
                        ),
                    )
                    target_lane_clear: bool | None = None
                    clearance_reason: str | None = None
                    actual_clearance = None
                    target_lane_index = coordinator.target_lane_index
                    if target_lane_index is not None:
                        relation = (
                            LaneRelation.SAME
                            if target_lane_index == current_lane_index
                            else LaneRelation.LEFT
                            if target_lane_index < current_lane_index
                            else LaneRelation.RIGHT
                        )
                        scene_value = record.payload.get("scene")
                        if scene_value is None:
                            target_lane_clear = False
                            clearance_reason = (
                                "local scene is unavailable during lane change"
                            )
                        else:
                            validator = lane_change_validators.setdefault(
                                record.run_id,
                                VLACommandValidator(config.command_validation),
                            )
                            actual_clearance = validator.lane_change_clearance(
                                scene_from_payload(scene_value),
                                relation,
                                now_s=now_s,
                                include_planned_path=True,
                            )
                            target_lane_clear = actual_clearance.clear
                            clearance_reason = actual_clearance.reason

                    expected_clearance = context.get("clearance")
                    if to_json_value(actual_clearance) != expected_clearance:
                        raise EventLogError(
                            "replayed lane-change clearance decision differs"
                        )
                    actual_lane = coordinator.update(
                        execution,
                        now_s=now_s,
                        current_lane_index=current_lane_index,
                        lane_count=lane_count,
                        target_lane_offset_m=target_lane_offset_m,
                        target_lane_clear=target_lane_clear,
                        clearance_reason=clearance_reason,
                    )
                    forced_abort_reason = _optional_string(
                        context,
                        "forced_abort_reason",
                    )
                    if forced_abort_reason is not None:
                        actual_lane = coordinator.abort(
                            now_s=now_s,
                            reason=forced_abort_reason,
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    mismatches.append(
                        _mismatch(record, f"cannot replay lane change: {exc}")
                    )
                else:
                    if to_json_value(actual_lane) != dict(expected_lane):
                        mismatches.append(
                            _mismatch(
                                record,
                                "replayed lane-change decision differs",
                            )
                        )
                    else:
                        matched_events += 1

        if record.payload.get("safety_decision") is None:
            continue
        safety_events += 1
        if config is None:
            mismatches.append(_mismatch(record, "run_started configuration is unavailable"))
            continue
        try:
            scene = scene_from_payload(record.payload.get("scene"))
            proposed = _control_command(record.payload.get("proposed_command"))
            now_s = _number(record.payload, "now_s")
            expected = _mapping(record.payload.get("safety_decision"), "safety_decision")
            supervisor = supervisors.setdefault(
                record.run_id,
                EmergencyBrakingSupervisor(config.safety),
            )
            actual = supervisor.evaluate(scene, proposed, now_s=now_s)
        except (KeyError, TypeError, ValueError) as exc:
            mismatches.append(_mismatch(record, f"cannot replay safety decision: {exc}"))
            continue
        if to_json_value(actual) != dict(expected):
            mismatches.append(_mismatch(record, "replayed safety decision differs"))
            continue
        matched_events += 1

    return ReplaySummary(
        event_log_path=str(Path(path).resolve()),
        total_events=len(records),
        runs=len({record.run_id for record in records}),
        validation_events=validation_events,
        execution_events=execution_events,
        safety_events=safety_events,
        headway_cap_events=headway_cap_events,
        lane_change_events=lane_change_events,
        matched_events=matched_events,
        mismatches=tuple(mismatches),
    )


def _config_from_event(record: EventRecord) -> AppConfig:
    value = _mapping(record.payload.get("config"), "config")
    return config_from_dict(dict(value))


def _command(value: object) -> VLACommand | None:
    if value is None:
        return None
    return VLACommand.from_payload(_mapping(value, "VLA command"))


def _validation_decision(value: object) -> CommandValidationDecision | None:
    if value is None:
        return None
    data = _mapping(value, "validation decision")
    reasons = data.get("reasons")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        raise EventLogError("validation decision reasons must be a list of strings")
    requested = _command(data.get("requested_command"))
    effective = _command(data.get("effective_command"))
    if effective is None:
        raise EventLogError("validation decision effective_command must not be null")
    return CommandValidationDecision(
        disposition=CommandDisposition(_string(data, "disposition")),
        requested_command=requested,
        effective_command=effective,
        reasons=tuple(reasons),
    )


def _execution_decision(value: object) -> CommandExecutionDecision | None:
    if value is None:
        return None
    data = _mapping(value, "execution")
    return CommandExecutionDecision(
        source=CommandExecutionSource(_string(data, "source")),
        action=HighLevelAction(_string(data, "action")),
        target_speed_mps=_number(data, "target_speed_mps"),
        command_id=_string(data, "command_id"),
        reason=_string(data, "reason"),
        valid_until_s=_optional_number(data, "valid_until_s"),
    )


def _headway_speed_cap_trigger(payload: Mapping[str, Any]) -> str:
    value = payload.get("headway_speed_cap_trigger")
    if value is None:
        return "configured_mode"
    if value not in {"configured_mode", "follow_action"}:
        raise EventLogError(
            "headway_speed_cap_trigger must be 'configured_mode' or 'follow_action'"
        )
    execution = _execution_decision(payload.get("execution"))
    is_follow = (
        execution is not None
        and execution.source is CommandExecutionSource.VLA
        and execution.action is HighLevelAction.FOLLOW
    )
    if (value == "follow_action") != is_follow:
        raise EventLogError("headway speed-cap trigger disagrees with execution")
    return value


def _control_command(value: object) -> ControlCommand:
    data = _mapping(value, "control command")
    return ControlCommand(
        steering=_number(data, "steering"),
        throttle=_number(data, "throttle"),
        brake=_number(data, "brake"),
    )


def scene_from_payload(value: object) -> LocalScene:
    """Decode a logged ``LocalScene``; older logs without traffic lights decode too."""
    data = _mapping(value, "scene")
    objects_value = data.get("objects", [])
    if not isinstance(objects_value, list):
        raise EventLogError("scene.objects must be a list")
    objects = tuple(_tracked_object(item) for item in objects_value)
    traffic_lights_value = data.get("traffic_lights", [])
    if not isinstance(traffic_lights_value, list):
        raise EventLogError("scene.traffic_lights must be a list")
    traffic_lights = tuple(_traffic_light(item) for item in traffic_lights_value)
    return LocalScene(
        timestamp_s=_number(data, "timestamp_s"),
        ego_speed_mps=_number(data, "ego_speed_mps"),
        ego_length_m=_number(data, "ego_length_m"),
        ego_width_m=_number(data, "ego_width_m"),
        lane_offset_m=_number(data, "lane_offset_m"),
        heading_error_rad=_number(data, "heading_error_rad"),
        objects=objects,
        traffic_lights=traffic_lights,
        distance_to_left_boundary_m=_optional_number(data, "distance_to_left_boundary_m"),
        distance_to_right_boundary_m=_optional_number(data, "distance_to_right_boundary_m"),
        left_lane_available=_optional_boolean(data, "left_lane_available"),
        right_lane_available=_optional_boolean(data, "right_lane_available"),
        valid=_boolean(data, "valid"),
    )


def _traffic_light(value: object) -> TrafficLightObservation:
    data = _mapping(value, "traffic light")
    return TrafficLightObservation(
        light_id=_string(data, "light_id"),
        state=TrafficLightState(_string(data, "state")),
        relative_position_m=_vector(data, "relative_position_m"),
        in_path=_boolean(data, "in_path"),
        path_distance_m=_optional_number(data, "path_distance_m"),
        confidence=_number(data, "confidence"),
    )


def _tracked_object(value: object) -> TrackedObject:
    data = _mapping(value, "tracked object")
    relative_position = _vector(data, "relative_position_m")
    relative_velocity = _vector(data, "relative_velocity_mps")
    return TrackedObject(
        object_id=_string(data, "object_id"),
        kind=_string(data, "kind"),
        relative_position_m=relative_position,
        relative_velocity_mps=relative_velocity,
        length_m=_number(data, "length_m"),
        width_m=_number(data, "width_m"),
        lane_relation=LaneRelation(_string(data, "lane_relation")),
        in_path=_boolean(data, "in_path"),
        path_distance_m=_optional_number(data, "path_distance_m"),
        path_relative_velocity_mps=_optional_number(
            data, "path_relative_velocity_mps"
        ),
        path_cross_track_m=_optional_number(data, "path_cross_track_m"),
        confidence=_number(data, "confidence"),
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EventLogError(f"{name} must be a JSON object")
    return value


def _string(data: Mapping[str, Any], name: str) -> str:
    value = data[name]
    if not isinstance(value, str):
        raise EventLogError(f"{name} must be a string")
    return value


def _number(data: Mapping[str, Any], name: str) -> float:
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventLogError(f"{name} must be numeric")
    return float(value)


def _integer(data: Mapping[str, Any], name: str) -> int:
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise EventLogError(f"{name} must be an integer")
    return value


def _optional_number(data: Mapping[str, Any], name: str) -> float | None:
    value = data.get(name)
    return None if value is None else _number(data, name)


def _optional_string(data: Mapping[str, Any], name: str) -> str | None:
    value = data.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise EventLogError(f"{name} must be a non-empty string or null")
    return value


def _boolean(data: Mapping[str, Any], name: str) -> bool:
    value = data[name]
    if not isinstance(value, bool):
        raise EventLogError(f"{name} must be a boolean")
    return value


def _optional_boolean(data: Mapping[str, Any], name: str) -> bool | None:
    value = data.get(name)
    if value is None:
        return None
    return _boolean(data, name)


def _vector(data: Mapping[str, Any], name: str) -> tuple[float, float]:
    value = data[name]
    if not isinstance(value, list) or len(value) != 2:
        raise EventLogError(f"{name} must contain two numbers")
    vector_data = {"x": value[0], "y": value[1]}
    return _number(vector_data, "x"), _number(vector_data, "y")


def _mismatch(record: EventRecord, reason: str) -> ReplayMismatch:
    return ReplayMismatch(record.run_id, record.sequence, record.event_type, reason)
