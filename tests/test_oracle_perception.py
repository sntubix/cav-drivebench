import math
from types import SimpleNamespace

import pytest

from metadrive_starter.perception import (
    LaneRelation,
    OracleSceneAdapter,
    TrafficLightState,
)
from metadrive_starter.planning import PolylineFuturePath


class _Lane:
    def local_coordinates(self, position: object) -> tuple[float, float]:
        return float(position[1]), float(-position[0])  # type: ignore[index]

    def heading_theta_at(self, longitude: float) -> float:
        del longitude
        return math.pi / 2


def _vehicle(
    object_id: str,
    *,
    position: tuple[float, float],
    velocity: tuple[float, float],
    lane_index: tuple[str, str, int],
) -> SimpleNamespace:
    return SimpleNamespace(
        id=object_id,
        position=position,
        velocity=velocity,
        lane_index=lane_index,
        LENGTH=4.5,
        WIDTH=1.8,
        metadrive_type="VEHICLE",
    )


def test_oracle_adapter_converts_world_state_to_ego_frame() -> None:
    ego = _vehicle("ego", position=(0.0, 0.0), velocity=(0.0, 5.0), lane_index=("a", "b", 0))
    ego.heading_theta = math.pi / 2
    ego.speed = 5.0
    ego.lane = _Lane()
    ego.dist_to_left_side = 2.0
    ego.dist_to_right_side = 5.0
    lead = _vehicle("lead", position=(0.0, 20.0), velocity=(0.0, 2.0), lane_index=("a", "b", 0))
    engine = SimpleNamespace(get_objects=lambda: {"ego": ego, "lead": lead})
    env = SimpleNamespace(agent=ego, engine=engine)

    scene = OracleSceneAdapter().observe(env, timestamp_s=1.25)

    assert scene.timestamp_s == 1.25
    assert scene.ego_speed_mps == 5.0
    assert scene.lane_offset_m == 0.0
    assert scene.distance_to_left_boundary_m == 2.0
    assert len(scene.objects) == 1
    tracked = scene.objects[0]
    assert tracked.object_id == "lead"
    assert tracked.relative_position_m == pytest.approx((20.0, 0.0))
    assert tracked.relative_velocity_mps == pytest.approx((-3.0, 0.0))
    assert tracked.lane_relation is LaneRelation.SAME
    assert tracked.in_path is True


def test_oracle_adapter_marks_adjacent_vehicle_outside_corridor() -> None:
    ego = _vehicle("ego", position=(0.0, 0.0), velocity=(5.0, 0.0), lane_index=("a", "b", 0))
    ego.heading_theta = 0.0
    ego.speed = 5.0
    ego.lane = _Lane()
    adjacent = _vehicle("left", position=(10.0, 3.5), velocity=(5.0, 0.0), lane_index=("a", "b", 1))
    engine = SimpleNamespace(get_objects=lambda: {"ego": ego, "left": adjacent})
    env = SimpleNamespace(agent=ego, engine=engine)

    tracked = OracleSceneAdapter(corridor_margin_m=0.5).observe(env, timestamp_s=0.0).objects[0]

    assert tracked.lane_relation is LaneRelation.LEFT
    assert tracked.in_path is False


def test_oracle_adapter_ignores_objects_outside_detection_radius() -> None:
    ego = _vehicle("ego", position=(0.0, 0.0), velocity=(0.0, 0.0), lane_index=("a", "b", 0))
    ego.heading_theta = 0.0
    ego.speed = 0.0
    ego.lane = _Lane()
    distant = _vehicle("far", position=(60.0, 0.0), velocity=(0.0, 0.0), lane_index=("a", "b", 0))
    engine = SimpleNamespace(get_objects=lambda: {"ego": ego, "far": distant})
    env = SimpleNamespace(agent=ego, engine=engine)

    scene = OracleSceneAdapter(detection_radius_m=50.0).observe(env, timestamp_s=0.0)

    assert scene.objects == ()


def test_oracle_adapter_separates_traffic_light_from_collision_objects() -> None:
    ego = _vehicle("ego", position=(0.0, 0.0), velocity=(5.0, 0.0), lane_index=("a", "b", 0))
    ego.heading_theta = 0.0
    ego.speed = 5.0
    ego.lane = _Lane()
    light = SimpleNamespace(
        id="signal",
        position=(20.0, 0.0),
        WIDTH=3.5,
        metadrive_type="TRAFFIC_LIGHT",
        status="TRAFFIC_LIGHT_RED",
    )
    engine = SimpleNamespace(get_objects=lambda: {"ego": ego, "signal": light})
    env = SimpleNamespace(agent=ego, engine=engine)
    adapter = OracleSceneAdapter(
        future_path=PolylineFuturePath([(0.0, 0.0), (50.0, 0.0)])
    )

    scene = adapter.observe(env, timestamp_s=1.0)

    assert scene.objects == ()
    assert len(scene.traffic_lights) == 1
    observed = scene.traffic_lights[0]
    assert observed.state is TrafficLightState.RED
    assert observed.in_path is True
    assert observed.path_distance_m == pytest.approx(20.0)


def test_oracle_adapter_tracks_an_object_around_a_curve_by_path_distance() -> None:
    ego = _vehicle("ego", position=(0.0, 0.0), velocity=(5.0, 0.0), lane_index=("a", "b", 0))
    ego.heading_theta = 0.0
    ego.speed = 5.0
    ego.lane = _Lane()
    stopped = _vehicle(
        "stopped",
        position=(0.0, 10.0),
        velocity=(0.0, 0.0),
        lane_index=("c", "d", 0),
    )
    stopped.heading_theta = math.pi
    engine = SimpleNamespace(get_objects=lambda: {"ego": ego, "stopped": stopped})
    env = SimpleNamespace(agent=ego, engine=engine)
    future_path = PolylineFuturePath(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    )

    tracked = OracleSceneAdapter(future_path=future_path).observe(env, timestamp_s=0.0).objects[0]

    assert tracked.relative_position_m == pytest.approx((0.0, 10.0))
    assert tracked.path_distance_m == pytest.approx(30.0)
    assert tracked.path_relative_velocity_mps == pytest.approx(-5.0)
    assert tracked.path_cross_track_m == pytest.approx(0.0)
    assert tracked.in_path is True
    assert tracked.lane_relation is LaneRelation.CROSSING


@pytest.mark.parametrize(
    ("lane_number", "left_available", "right_available"),
    [(0, False, True), (1, True, False)],
)
def test_oracle_scene_reports_adjacent_lane_availability(
    lane_number: int,
    left_available: bool,
    right_available: bool,
) -> None:
    ego = _vehicle(
        "ego",
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        lane_index=("a", "b", lane_number),
    )
    ego.heading_theta = 0.0
    ego.speed = 0.0
    ego.lane = _Lane()
    engine = SimpleNamespace(get_objects=lambda: {"ego": ego})
    road_network = SimpleNamespace(graph={"a": {"b": [object(), object()]}})
    env = SimpleNamespace(
        agent=ego,
        engine=engine,
        current_map=SimpleNamespace(road_network=road_network),
    )

    scene = OracleSceneAdapter().observe(env, timestamp_s=0.0)

    assert scene.left_lane_available is left_available
    assert scene.right_lane_available is right_available
