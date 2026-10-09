from __future__ import annotations

from types import SimpleNamespace

import pytest

from metadrive_starter.perception import LidarSceneAdapter
from metadrive_starter.planning import PolylineFuturePath


class _Lane:
    def local_coordinates(self, position: object) -> tuple[float, float]:
        return float(position[0]), float(position[1])  # type: ignore[index]

    def heading_theta_at(self, longitude: float) -> float:
        del longitude
        return 0.0


class _LidarSensor:
    def __init__(self, detected_objects: list[object]):
        self.detected_objects = detected_objects
        self.call: dict[str, object] | None = None

    def perceive(self, ego: object, **kwargs: object) -> tuple[list[float], list[object]]:
        self.call = {"ego": ego, **kwargs}
        return [1.0] * int(kwargs["num_lasers"]), self.detected_objects


def _vehicle(
    object_id: str,
    *,
    position: tuple[float, float],
    velocity: tuple[float, float],
) -> SimpleNamespace:
    return SimpleNamespace(
        id=object_id,
        position=position,
        velocity=velocity,
        heading_theta=0.0,
        lane_index=("a", "b", 0),
        LENGTH=4.5,
        WIDTH=1.8,
        metadrive_type="VEHICLE",
    )


def _environment(sensor: _LidarSensor, *, num_lasers: int = 120) -> SimpleNamespace:
    ego = _vehicle("ego", position=(0.0, 0.0), velocity=(5.0, 0.0))
    ego.speed = 5.0
    ego.lane = _Lane()
    ego.config = {
        "lidar": {"num_lasers": num_lasers, "distance": 50.0},
        "show_lidar": False,
    }
    physics_world = SimpleNamespace(dynamic_world=object())
    engine = SimpleNamespace(
        physics_world=physics_world,
        get_sensor=lambda name: sensor if name == "lidar" else None,
    )
    return SimpleNamespace(agent=ego, engine=engine)


def test_lidar_adapter_builds_scene_only_from_sensor_detections() -> None:
    lead = _vehicle("lead", position=(20.0, 0.0), velocity=(2.0, 0.0))
    sensor = _LidarSensor([lead])
    env = _environment(sensor)
    adapter = LidarSceneAdapter(
        detection_radius_m=40.0,
        future_path=PolylineFuturePath([(0.0, 0.0), (50.0, 0.0)]),
    )

    scene = adapter.observe(env, timestamp_s=1.5)

    assert scene.valid is True
    assert len(scene.objects) == 1
    tracked = scene.objects[0]
    assert tracked.object_id == "lead"
    assert tracked.relative_position_m == pytest.approx((20.0, 0.0))
    assert tracked.path_distance_m == pytest.approx(20.0)
    assert tracked.path_relative_velocity_mps == pytest.approx(-3.0)
    assert tracked.in_path is True
    assert sensor.call is not None
    assert sensor.call["num_lasers"] == 120
    assert sensor.call["distance"] == 40.0


def test_disabled_lidar_produces_invalid_fail_safe_scene() -> None:
    sensor = _LidarSensor([])
    env = _environment(sensor, num_lasers=0)

    scene = LidarSceneAdapter().observe(env, timestamp_s=2.0)

    assert scene.valid is False
    assert scene.objects == ()
    assert sensor.call is None
