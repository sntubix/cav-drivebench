from metadrive_starter.perception import LaneRelation, LocalScene, TrackedObject
from metadrive_starter.safety import EmergencyBrakingSupervisor, SafetyLevel, SafetySettings
from metadrive_starter.types import ControlCommand


CRUISE = ControlCommand(steering=0.2, throttle=0.6, brake=0.0)


def _object(
    *,
    longitudinal_m: float,
    relative_speed_mps: float,
    in_path: bool = True,
    lane_relation: LaneRelation = LaneRelation.SAME,
    path_distance_m: float | None = None,
    path_relative_velocity_mps: float | None = None,
) -> TrackedObject:
    return TrackedObject(
        object_id="vehicle-1",
        kind="vehicle",
        relative_position_m=(longitudinal_m, 0.0),
        relative_velocity_mps=(relative_speed_mps, 0.0),
        length_m=4.5,
        width_m=1.8,
        lane_relation=lane_relation,
        in_path=in_path,
        path_distance_m=path_distance_m,
        path_relative_velocity_mps=path_relative_velocity_mps,
    )


def _scene(*objects: TrackedObject, timestamp_s: float = 10.0, valid: bool = True) -> LocalScene:
    return LocalScene(
        timestamp_s=timestamp_s,
        ego_speed_mps=10.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        objects=objects,
        valid=valid,
    )


def test_clear_road_preserves_proposed_command() -> None:
    decision = EmergencyBrakingSupervisor().evaluate(_scene(), CRUISE, now_s=10.0)

    assert decision.level is SafetyLevel.CLEAR
    assert decision.command == CRUISE
    assert decision.intervened is False


def test_stopped_vehicle_in_same_lane_triggers_emergency_braking() -> None:
    obstacle = _object(longitudinal_m=10.0, relative_speed_mps=-10.0)

    decision = EmergencyBrakingSupervisor().evaluate(_scene(obstacle), CRUISE, now_s=10.0)

    assert decision.level is SafetyLevel.EMERGENCY
    assert decision.command.throttle == 0.0
    assert decision.command.brake == 1.0
    assert decision.intervened is True
    assert decision.minimum_ttc_s is not None
    assert decision.minimum_ttc_s < 1.0


def test_slower_lead_vehicle_triggers_controlled_braking() -> None:
    lead_vehicle = _object(longitudinal_m=10.0, relative_speed_mps=-2.0)

    decision = EmergencyBrakingSupervisor().evaluate(_scene(lead_vehicle), CRUISE, now_s=10.0)

    assert decision.level is SafetyLevel.BRAKE
    assert decision.command.throttle == 0.0
    assert 0.0 < decision.command.brake < 1.0


def test_close_vehicle_at_equal_speed_removes_throttle_without_braking() -> None:
    close_vehicle = _object(longitudinal_m=14.0, relative_speed_mps=0.0)

    decision = EmergencyBrakingSupervisor().evaluate(_scene(close_vehicle), CRUISE, now_s=10.0)

    assert decision.level is SafetyLevel.CAUTION
    assert decision.command.throttle == 0.0
    assert decision.command.brake == 0.0


def test_adjacent_vehicle_does_not_trigger_braking() -> None:
    adjacent_vehicle = _object(
        longitudinal_m=5.0,
        relative_speed_mps=-10.0,
        in_path=False,
        lane_relation=LaneRelation.LEFT,
    )

    decision = EmergencyBrakingSupervisor().evaluate(_scene(adjacent_vehicle), CRUISE, now_s=10.0)

    assert decision.level is SafetyLevel.CLEAR
    assert decision.command == CRUISE


def test_receding_vehicle_does_not_trigger_braking() -> None:
    receding_vehicle = _object(longitudinal_m=30.0, relative_speed_mps=2.0)

    decision = EmergencyBrakingSupervisor().evaluate(_scene(receding_vehicle), CRUISE, now_s=10.0)

    assert decision.level is SafetyLevel.CLEAR


def test_crossing_object_in_predicted_path_triggers_braking() -> None:
    crossing_vehicle = _object(
        longitudinal_m=10.0,
        relative_speed_mps=-2.0,
        lane_relation=LaneRelation.CROSSING,
    )

    decision = EmergencyBrakingSupervisor().evaluate(_scene(crossing_vehicle), CRUISE, now_s=10.0)

    assert decision.level is SafetyLevel.BRAKE


def test_object_around_a_curve_uses_path_distance_and_closing_speed() -> None:
    curved_path_obstacle = _object(
        longitudinal_m=-5.0,
        relative_speed_mps=2.0,
        path_distance_m=10.0,
        path_relative_velocity_mps=-10.0,
    )

    decision = EmergencyBrakingSupervisor().evaluate(
        _scene(curved_path_obstacle),
        CRUISE,
        now_s=10.0,
    )

    assert decision.level is SafetyLevel.EMERGENCY
    assert decision.command.brake == 1.0


def test_stale_scene_triggers_degraded_emergency_stop() -> None:
    settings = SafetySettings(stale_after_s=0.2)

    decision = EmergencyBrakingSupervisor(settings).evaluate(_scene(timestamp_s=9.0), CRUISE, now_s=10.0)

    assert decision.level is SafetyLevel.DEGRADED
    assert decision.command.throttle == 0.0
    assert decision.command.brake == 1.0
    assert "stale" in decision.reason


def test_invalid_scene_triggers_degraded_emergency_stop() -> None:
    decision = EmergencyBrakingSupervisor().evaluate(_scene(valid=False), CRUISE, now_s=10.0)

    assert decision.level is SafetyLevel.DEGRADED
    assert decision.command.brake == 1.0


def test_braking_is_held_briefly_after_hazard_disappears() -> None:
    supervisor = EmergencyBrakingSupervisor(SafetySettings(release_hold_s=0.3))
    obstacle = _object(longitudinal_m=10.0, relative_speed_mps=-10.0)

    supervisor.evaluate(_scene(obstacle), CRUISE, now_s=10.0)
    held = supervisor.evaluate(_scene(timestamp_s=10.1), CRUISE, now_s=10.1)
    released = supervisor.evaluate(_scene(timestamp_s=10.4), CRUISE, now_s=10.4)

    assert held.level is SafetyLevel.BRAKE
    assert held.command.throttle == 0.0
    assert held.command.brake > 0.0
    assert released.level is SafetyLevel.CLEAR
    assert released.command == CRUISE


def test_control_command_rejects_simultaneous_throttle_and_brake() -> None:
    try:
        ControlCommand(steering=0.0, throttle=0.5, brake=0.5)
    except ValueError as exc:
        assert "cannot both be positive" in str(exc)
    else:
        raise AssertionError("ControlCommand accepted simultaneous throttle and brake")
