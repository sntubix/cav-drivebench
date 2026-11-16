"""Only the lane directly beside the ego reads as LEFT or RIGHT.

A vehicle two lanes over once read as in the lane beside, so the measured lane
clearance and the safety floor's lane-change check both held that lane occupied. On a
road of stopped cars every lane but one, that closed the one lane that was open.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from metadrive_starter.perception import LaneRelation, OracleSceneAdapter, TrackedObject
from metadrive_starter.perception.object_scene import _lane_relation
from metadrive_starter.safety import CommandDisposition, VLACommandValidator
from metadrive_starter.vla import HighLevelAction, VLACommand
from metadrive_starter.vla.prompting import build_vla_scene_context

LANE_WIDTH_M = 3.5


class _StraightLane:
    """The ego's lane, along world +x."""

    def local_coordinates(self, position: object) -> tuple[float, float]:
        return float(position[0]), float(position[1])  # type: ignore[index]

    def heading_theta_at(self, longitude: float) -> float:
        del longitude
        return 0.0


def _vehicle(object_id: str, *, x_m: float, lane_number: int) -> SimpleNamespace:
    # MetaDrive numbers a road's lanes from its left: the ego drives lane 2 of three, so
    # lane 1 is beside it on the left and lane 0 is two lanes over.
    return SimpleNamespace(
        id=object_id,
        position=(x_m, (2 - lane_number) * LANE_WIDTH_M),
        velocity=(0.0, 0.0),
        heading_theta=0.0,
        lane_index=("a", "b", lane_number),
        LENGTH=4.5,
        WIDTH=1.852,
        metadrive_type="VEHICLE",
    )


def _measured_scene(*others: SimpleNamespace, timestamp_s: float = 10.0):
    ego = _vehicle("ego", x_m=0.0, lane_number=2)
    ego.speed = 5.0
    ego.velocity = (5.0, 0.0)
    ego.lane = _StraightLane()
    objects = {"ego": ego, **{other.id: other for other in others}}
    env = SimpleNamespace(
        agent=ego,
        engine=SimpleNamespace(get_objects=lambda: objects),
        current_map=SimpleNamespace(
            road_network=SimpleNamespace(graph={"a": {"b": [object(), object(), object()]}})
        ),
    )
    return OracleSceneAdapter().observe(env, timestamp_s=timestamp_s)


def _change_lane_left(now_s: float = 10.0) -> VLACommand:
    return VLACommand(
        action=HighLevelAction.CHANGE_LANE_LEFT,
        target_speed_mps=5.0,
        issued_at_s=now_s,
        action_horizon_s=2.0,
        confidence=0.9,
        command_id="change-left",
    )


@pytest.mark.parametrize(
    ("lane_number", "relation"),
    [(2, LaneRelation.SAME), (1, LaneRelation.LEFT), (0, LaneRelation.UNKNOWN)],
)
def test_only_the_lane_beside_reads_left(lane_number: int, relation: LaneRelation) -> None:
    scene = _measured_scene(_vehicle("other", x_m=20.0, lane_number=lane_number))

    assert scene.left_lane_available is True
    assert scene.objects[0].lane_relation is relation


@pytest.mark.parametrize(
    ("ego_lane", "object_lane", "lateral_m", "in_path", "relation"),
    [
        (("a", "b", 1), ("a", "b", 2), -3.5, False, LaneRelation.RIGHT),
        (("a", "b", 0), ("a", "b", 2), -7.0, False, LaneRelation.UNKNOWN),
        (("a", "b", 3), ("a", "b", 0), 10.5, False, LaneRelation.UNKNOWN),
        # Further over yet on the path, as round a bend: related by its path alone.
        (("a", "b", 0), ("a", "b", 2), -0.5, True, LaneRelation.CROSSING),
        # Another road block: no lane relation, as before.
        (("a", "b", 1), ("b", "c", 0), 3.5, False, LaneRelation.UNKNOWN),
        (("a", "b", 1), ("b", "c", 1), 0.0, True, LaneRelation.CROSSING),
    ],
)
def test_lane_relation_by_lanes_apart(
    ego_lane: tuple[str, str, int],
    object_lane: tuple[str, str, int],
    lateral_m: float,
    in_path: bool,
    relation: LaneRelation,
) -> None:
    assert _lane_relation(ego_lane, object_lane, lateral_m, in_path) is relation


def test_a_car_two_lanes_over_leaves_the_lane_beside_clear() -> None:
    scene = _measured_scene(_vehicle("two-over", x_m=2.0, lane_number=0))

    context = build_vla_scene_context(scene, max_scene_objects=8)

    assert context["lanes"]["left"] == {"available": True, "observed_clear": True}
    # The prompt still says where it is.
    assert context["objects"][0]["relative_location"] == "front_left"


def test_a_car_beside_still_occupies_the_lane() -> None:
    scene = _measured_scene(_vehicle("beside", x_m=2.0, lane_number=1))

    context = build_vla_scene_context(scene, max_scene_objects=8)

    assert context["lanes"]["left"] == {"available": True, "observed_clear": False}


def test_the_floor_admits_a_lane_change_past_a_car_two_lanes_over() -> None:
    scene = _measured_scene(_vehicle("two-over", x_m=2.0, lane_number=0))
    validator = VLACommandValidator()

    decision = validator.validate(_change_lane_left(), scene, now_s=10.0)
    in_flight = validator.lane_change_clearance(
        scene,
        LaneRelation.LEFT,
        now_s=10.0,
        include_planned_path=True,
    )

    assert decision.disposition is CommandDisposition.ACCEPTED
    assert decision.effective_command.action is HighLevelAction.CHANGE_LANE_LEFT
    assert in_flight.clear is True


def test_the_floor_still_refuses_a_lane_change_into_a_car_beside() -> None:
    scene = _measured_scene(_vehicle("beside", x_m=2.0, lane_number=1))

    decision = VLACommandValidator().validate(_change_lane_left(), scene, now_s=10.0)

    assert decision.disposition is CommandDisposition.REJECTED
    assert decision.reasons[0] == "object beside violates lane-change gap"


@pytest.mark.parametrize(
    ("lateral_m", "width_m", "observed_clear"),
    [
        (3.5, 0.2, False),  # a cone in the lane beside
        (7.0, 0.2, True),  # a cone two lanes over
        (7.0, 1.852, True),  # a car two lanes over, on another road block
        (7.0, 2.5, True),  # a truck there, 5.75 m off at its near side
        (6.1, 1.852, False),  # straddling the far line: 5.17 m off at its near side
        (5.25, 0.2, False),  # on the far line
    ],
)
def test_an_object_without_a_lane_relation_is_beside_if_it_reaches_the_lane(
    lateral_m: float,
    width_m: float,
    observed_clear: bool,
) -> None:
    scene = _measured_scene()
    unrelated = TrackedObject(
        object_id="unrelated",
        kind="traffic_cone" if width_m < 1.0 else "vehicle",
        relative_position_m=(10.0, lateral_m),
        relative_velocity_mps=(0.0, 0.0),
        length_m=4.5,
        width_m=width_m,
        lane_relation=LaneRelation.UNKNOWN,
        in_path=False,
    )
    scene = replace(scene, objects=(unrelated,))

    context = build_vla_scene_context(scene, max_scene_objects=8)

    assert context["lanes"]["left"]["observed_clear"] is observed_clear
