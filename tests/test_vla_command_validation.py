from __future__ import annotations

from dataclasses import replace

import pytest

from metadrive_starter.perception import LaneRelation, LocalScene, TrackedObject
from metadrive_starter.safety import CommandDisposition, VLACommandValidator
from metadrive_starter.vla import HighLevelAction, VLACommand


def _command(
    action: HighLevelAction = HighLevelAction.KEEP_LANE,
    *,
    target_speed_mps: float = 30.0 / 3.6,
    issued_at_s: float = 9.5,
    action_horizon_s: float = 2.0,
    confidence: float = 0.9,
) -> VLACommand:
    return VLACommand(
        action=action,
        target_speed_mps=target_speed_mps,
        issued_at_s=issued_at_s,
        action_horizon_s=action_horizon_s,
        confidence=confidence,
        command_id="command-1",
    )


def _object(
    object_id: str,
    *,
    longitudinal_m: float,
    relative_speed_mps: float = 0.0,
    lane_relation: LaneRelation,
    in_path: bool = False,
) -> TrackedObject:
    return TrackedObject(
        object_id=object_id,
        kind="vehicle",
        relative_position_m=(longitudinal_m, 3.5 if lane_relation is LaneRelation.LEFT else -3.5),
        relative_velocity_mps=(relative_speed_mps, 0.0),
        length_m=4.5,
        width_m=1.8,
        lane_relation=lane_relation,
        in_path=in_path,
    )


def _ahead(
    object_id: str = "truck",
    *,
    gap_m: float,
    lateral_m: float = 0.0,
    closing_mps: float = 0.0,
) -> TrackedObject:
    """A stopped-truck-sized object on the ego's path, ``gap_m`` bumper to bumper."""
    return TrackedObject(
        object_id=object_id,
        kind="vehicle",
        relative_position_m=(gap_m + (4.5 + 5.74) / 2.0, lateral_m),
        relative_velocity_mps=(-closing_mps, 0.0),
        length_m=5.74,
        width_m=2.3,
        lane_relation=LaneRelation.SAME,
        in_path=True,
        path_distance_m=gap_m + (4.5 + 5.74) / 2.0,
        path_cross_track_m=lateral_m,
    )


def _scene(
    *objects: TrackedObject,
    timestamp_s: float = 10.0,
    valid: bool = True,
    left_lane_available: bool | None = True,
    right_lane_available: bool | None = True,
) -> LocalScene:
    return LocalScene(
        timestamp_s=timestamp_s,
        ego_speed_mps=10.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        objects=objects,
        left_lane_available=left_lane_available,
        right_lane_available=right_lane_available,
        valid=valid,
    )


def test_valid_keep_lane_command_is_accepted_unchanged() -> None:
    command = _command()

    decision = VLACommandValidator().validate(command, _scene(), now_s=10.0)

    assert decision.disposition is CommandDisposition.ACCEPTED
    assert decision.effective_command == command
    assert decision.intervened is False


def test_payload_parser_rejects_missing_fields_into_fallback() -> None:
    decision = VLACommandValidator().validate_payload(
        {"action": "KEEP_LANE"},
        _scene(),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.FALLBACK
    assert decision.requested_command is None
    assert decision.effective_command.action is HighLevelAction.REQUEST_FALLBACK
    assert "missing" in decision.reasons[0]


def test_payload_parser_rejects_unknown_fields_into_fallback() -> None:
    decision = VLACommandValidator().validate_payload(
        {
            "action": "KEEP_LANE",
            "target_speed_mps": 20.0,
            "issued_at_s": 10.0,
            "action_horizon_s": 1.0,
            "confidence": 0.9,
            "steering": 1.0,
        },
        _scene(),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.FALLBACK
    assert "unknown" in decision.reasons[0]


def test_payload_parser_rejects_oversized_numeric_values_into_fallback() -> None:
    decision = VLACommandValidator().validate_payload(
        {
            "action": "KEEP_LANE",
            "target_speed_mps": 10**10000,
            "issued_at_s": 10.0,
            "action_horizon_s": 1.0,
            "confidence": 0.9,
        },
        _scene(),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.FALLBACK
    assert "numeric" in decision.reasons[0]


def test_target_speed_is_limited_locally() -> None:
    decision = VLACommandValidator().validate(
        _command(target_speed_mps=90.0 / 3.6),
        _scene(),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.MODIFIED
    assert decision.effective_command.target_speed_mps == pytest.approx(50.0 / 3.6)


@pytest.mark.parametrize(
    ("action", "requested_speed", "expected_speed"),
    [
        (HighLevelAction.STOP, 30.0 / 3.6, 0.0),
        (HighLevelAction.SLOW_DOWN, 30.0 / 3.6, 20.0 / 3.6),
        (HighLevelAction.YIELD, 30.0 / 3.6, 10.0 / 3.6),
    ],
)
def test_semantic_actions_apply_local_speed_limits(
    action: HighLevelAction,
    requested_speed: float,
    expected_speed: float,
) -> None:
    decision = VLACommandValidator().validate(
        _command(action, target_speed_mps=requested_speed),
        _scene(),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.MODIFIED
    assert decision.effective_command.target_speed_mps == pytest.approx(expected_speed)


@pytest.mark.parametrize(
    "command",
    [
        _command(issued_at_s=7.0, action_horizon_s=10.0),
        _command(issued_at_s=9.0, action_horizon_s=0.5),
        _command(issued_at_s=11.0),
        _command(confidence=0.2),
        _command(HighLevelAction.REQUEST_FALLBACK),
    ],
    ids=["stale", "expired-horizon", "future", "low-confidence", "requested-fallback"],
)
def test_unusable_command_requests_fallback(command: VLACommand) -> None:
    decision = VLACommandValidator().validate(command, _scene(), now_s=10.0)

    assert decision.disposition is CommandDisposition.FALLBACK
    assert decision.effective_command.action is HighLevelAction.REQUEST_FALLBACK


def test_action_horizon_boundary_tolerates_only_float_noise() -> None:
    validator = VLACommandValidator()
    command = _command(issued_at_s=0.0, action_horizon_s=1.5)

    boundary = validator.validate(
        command,
        _scene(timestamp_s=1.5000000000000004),
        now_s=1.5000000000000004,
    )
    late = validator.validate(
        command,
        _scene(timestamp_s=1.6),
        now_s=1.6,
    )

    assert boundary.disposition is CommandDisposition.ACCEPTED
    assert late.disposition is CommandDisposition.FALLBACK
    assert "expired" in late.reasons[0]


@pytest.mark.parametrize(
    "scene",
    [_scene(valid=False), _scene(timestamp_s=9.0)],
    ids=["invalid", "stale"],
)
def test_unusable_scene_requests_fallback(scene: LocalScene) -> None:
    decision = VLACommandValidator().validate(_command(), scene, now_s=10.0)

    assert decision.disposition is CommandDisposition.FALLBACK


def test_clear_available_lane_change_is_accepted() -> None:
    decision = VLACommandValidator().validate(
        _command(HighLevelAction.CHANGE_LANE_LEFT),
        _scene(left_lane_available=True),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.ACCEPTED
    assert decision.effective_command.action is HighLevelAction.CHANGE_LANE_LEFT


@pytest.mark.parametrize("lane_available", [False, None], ids=["unavailable", "unknown"])
def test_lane_change_fails_closed_when_lane_is_not_known_available(
    lane_available: bool | None,
) -> None:
    decision = VLACommandValidator().validate(
        _command(HighLevelAction.CHANGE_LANE_LEFT),
        _scene(left_lane_available=lane_available),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.REJECTED
    assert decision.effective_command.action is HighLevelAction.KEEP_LANE


def test_close_adjacent_vehicle_rejects_lane_change() -> None:
    adjacent = _object(
        "left-front",
        longitudinal_m=8.0,
        lane_relation=LaneRelation.LEFT,
    )

    decision = VLACommandValidator().validate(
        _command(HighLevelAction.OVERTAKE),
        _scene(adjacent),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.REJECTED
    assert decision.effective_command.action is HighLevelAction.KEEP_LANE
    assert "gap" in decision.reasons[0]


def test_fast_rear_vehicle_rejects_lane_change_by_ttc() -> None:
    rear = _object(
        "left-rear",
        longitudinal_m=-20.0,
        relative_speed_mps=10.0,
        lane_relation=LaneRelation.LEFT,
    )

    decision = VLACommandValidator().validate(
        _command(HighLevelAction.CHANGE_LANE_LEFT),
        _scene(rear),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.REJECTED
    assert "TTC" in decision.reasons[0]


def test_distant_adjacent_vehicle_allows_lane_change() -> None:
    adjacent = _object(
        "left-far",
        longitudinal_m=40.0,
        lane_relation=LaneRelation.LEFT,
    )

    decision = VLACommandValidator().validate(
        _command(HighLevelAction.CHANGE_LANE_LEFT),
        _scene(adjacent),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.ACCEPTED


def test_in_flight_clearance_rechecks_lane_which_is_now_current() -> None:
    blocking = _object(
        "target-front",
        longitudinal_m=8.0,
        lane_relation=LaneRelation.SAME,
    )

    clearance = VLACommandValidator().lane_change_clearance(
        _scene(blocking),
        LaneRelation.SAME,
        now_s=10.0,
    )

    assert clearance.clear is False
    assert clearance.object_id == "target-front"
    assert clearance.gap_m == pytest.approx(3.5)
    assert "gap" in clearance.reason


def test_in_flight_clearance_includes_projected_path_hazard_without_lane_id() -> None:
    crossing = _object(
        "projected-hazard",
        longitudinal_m=8.0,
        lane_relation=LaneRelation.CROSSING,
        in_path=True,
    )
    validator = VLACommandValidator()

    admission = validator.lane_change_clearance(
        _scene(crossing),
        LaneRelation.LEFT,
        now_s=10.0,
    )
    in_flight = validator.lane_change_clearance(
        _scene(crossing),
        LaneRelation.LEFT,
        now_s=10.0,
        include_planned_path=True,
    )

    assert admission.clear is True
    assert in_flight.clear is False
    assert in_flight.object_id == "projected-hazard"


def test_lane_change_stopped_behind_a_truck_is_declined_rather_than_started() -> None:
    # Stopped 3 m behind a truck, the car can neither swing past it on the planner's
    # route nor leave its lane: the validator used to accept this, and the running
    # check to abort it a tick later (ADR-0009).
    decision = VLACommandValidator().validate(
        _command(HighLevelAction.CHANGE_LANE_LEFT),
        _scene(_ahead(gap_m=3.0)),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.REJECTED
    assert decision.effective_command.action is HighLevelAction.KEEP_LANE
    assert "truck violates lane-change gap" in decision.reasons[0]


@pytest.mark.parametrize(
    ("gap_m", "closing_mps", "clear"),
    [
        (11.9, 0.0, False),
        (12.0, 0.0, True),
        (12.0, 4.0, True),
        (12.0, 4.5, False),  # 2.67 s
        (16.0, 5.56, False),  # 2.88 s
        (16.7, 5.56, True),
        (28.0, 9.72, False),  # 2.88 s
        (29.2, 9.72, True),
    ],
)
def test_lane_change_zone_is_the_front_gap_and_the_time_to_collision(
    gap_m: float,
    closing_mps: float,
    clear: bool,
) -> None:
    decision = VLACommandValidator().validate(
        _command(HighLevelAction.CHANGE_LANE_RIGHT),
        _scene(_ahead(gap_m=gap_m, closing_mps=closing_mps)),
        now_s=10.0,
    )

    expected = CommandDisposition.ACCEPTED if clear else CommandDisposition.REJECTED
    assert decision.disposition is expected


@pytest.mark.parametrize(
    "relation",
    [LaneRelation.LEFT, LaneRelation.RIGHT],
    ids=["left", "right"],
)
@pytest.mark.parametrize(
    "tracked",
    [
        _ahead(gap_m=3.0),
        _ahead(gap_m=11.0),
        _ahead(gap_m=20.0),
        _ahead(gap_m=11.0, lateral_m=2.3),
        _ahead(gap_m=11.0, lateral_m=-2.3),
        _object("left-front", longitudinal_m=8.0, lane_relation=LaneRelation.LEFT),
        _object("right-rear", longitudinal_m=-6.0, lane_relation=LaneRelation.RIGHT),
    ],
    ids=["gap-3", "gap-11", "gap-20", "left-of-path", "right-of-path", "left-lane", "right-lane"],
)
def test_validation_and_the_running_check_give_one_answer(
    relation: LaneRelation,
    tracked: TrackedObject,
) -> None:
    validator = VLACommandValidator()
    action = (
        HighLevelAction.CHANGE_LANE_LEFT
        if relation is LaneRelation.LEFT
        else HighLevelAction.CHANGE_LANE_RIGHT
    )

    decision = validator.validate(_command(action), _scene(tracked), now_s=10.0)
    running = validator.lane_change_clearance(
        _scene(tracked),
        relation,
        now_s=10.0,
        include_planned_path=True,
    )

    assert (decision.disposition is CommandDisposition.ACCEPTED) is running.clear


@pytest.mark.parametrize(
    ("lateral_m", "action", "disposition"),
    [
        # 2.3 m to the left clears the ego's width: moving right leaves it behind,
        # moving left runs into it.
        (2.3, HighLevelAction.CHANGE_LANE_RIGHT, CommandDisposition.ACCEPTED),
        (2.3, HighLevelAction.CHANGE_LANE_LEFT, CommandDisposition.REJECTED),
        (-2.3, HighLevelAction.CHANGE_LANE_LEFT, CommandDisposition.ACCEPTED),
        (-2.3, HighLevelAction.CHANGE_LANE_RIGHT, CommandDisposition.REJECTED),
        # Overlapping the ego's width it is on the path, whichever way it moves.
        (1.5, HighLevelAction.CHANGE_LANE_RIGHT, CommandDisposition.REJECTED),
    ],
)
def test_lane_change_may_start_away_from_an_object_it_does_not_overlap(
    lateral_m: float,
    action: HighLevelAction,
    disposition: CommandDisposition,
) -> None:
    decision = VLACommandValidator().validate(
        _command(action),
        _scene(_ahead(gap_m=8.0, lateral_m=lateral_m)),
        now_s=10.0,
    )

    assert decision.disposition is disposition


def test_object_behind_in_the_lane_being_left_does_not_hold_a_lane_change() -> None:
    behind = replace(
        _ahead(gap_m=0.0),
        relative_position_m=(-8.0, 0.0),
        relative_velocity_mps=(3.0, 0.0),
        path_distance_m=None,
        path_cross_track_m=None,
    )

    decision = VLACommandValidator().validate(
        _command(HighLevelAction.CHANGE_LANE_LEFT),
        _scene(behind),
        now_s=10.0,
    )

    assert decision.disposition is CommandDisposition.ACCEPTED


@pytest.mark.parametrize(
    ("longitudinal_m", "lane_relation"),
    [(8.0, LaneRelation.LEFT), (0.0, LaneRelation.LEFT), (-6.0, LaneRelation.LEFT)],
    ids=["ahead", "alongside", "behind"],
)
def test_nothing_left_behind_excuses_an_object_in_the_target_lane(
    longitudinal_m: float,
    lane_relation: LaneRelation,
) -> None:
    target = _object("target", longitudinal_m=longitudinal_m, lane_relation=lane_relation)

    for running in (False, True):
        clearance = VLACommandValidator().lane_change_clearance(
            _scene(target),
            LaneRelation.LEFT,
            now_s=10.0,
            include_planned_path=running,
        )
        assert clearance.clear is False
        assert clearance.object_id == "target"


def test_object_on_the_side_being_entered_stays_on_the_path_behind_the_ego() -> None:
    # Measured without a lane, close enough to be in the corridor mid-manoeuvre.
    entered = replace(
        _ahead(gap_m=0.0),
        relative_position_m=(-6.0, 2.2),
        lane_relation=LaneRelation.UNKNOWN,
        path_distance_m=None,
        path_cross_track_m=None,
    )

    clearance = VLACommandValidator().lane_change_clearance(
        _scene(entered),
        LaneRelation.LEFT,
        now_s=10.0,
        include_planned_path=True,
    )

    assert clearance.clear is False


def test_abeam_object_on_the_side_being_left_does_not_abort_a_running_change() -> None:
    abeam = replace(
        _ahead(gap_m=0.0),
        relative_position_m=(2.0, -2.4),
        path_distance_m=None,
        path_cross_track_m=None,
    )

    clearance = VLACommandValidator().lane_change_clearance(
        _scene(abeam),
        LaneRelation.LEFT,
        now_s=10.0,
        include_planned_path=True,
    )

    assert clearance.clear is True


@pytest.mark.parametrize(
    ("scene", "reason"),
    [
        (_scene(valid=False), "invalid"),
        (_scene(timestamp_s=9.0), "stale"),
        (_scene(timestamp_s=11.0), "future"),
    ],
)
def test_in_flight_clearance_fails_closed_on_unusable_scene(
    scene: LocalScene,
    reason: str,
) -> None:
    clearance = VLACommandValidator().lane_change_clearance(
        scene,
        LaneRelation.LEFT,
        now_s=10.0,
    )

    assert clearance.clear is False
    assert reason in clearance.reason
