from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter.config import load_config
from metadrive_starter.env import (
    _route_position_ahead,
    make_env,
    spawn_static_vehicle_ahead,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_static_truck_stands_on_the_road_where_it_was_placed() -> None:
    from metadrive.component.vehicle.vehicle_type import XLVehicle

    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    env = make_env(
        replace(config.simulator, map="S", headless=True),
        replace(config.camera, enabled=False),
    )
    try:
        env.reset()
        lane, longitudinal = _route_position_ahead(env, 25.0, lane_offset=1)
        placed = lane.position(longitudinal, 0.5)
        truck = spawn_static_vehicle_ahead(
            env, 25.0, lane_offset=1, lateral_m=0.5, vehicle_kind="truck", name="static-truck"
        )
        car = spawn_static_vehicle_ahead(env, 15.0, vehicle_kind="car", name="static-car")
        twin_lane, twin_longitudinal = _route_position_ahead(env, 60.0, lane_offset=1)
        twin = env.engine.spawn_object(
            XLVehicle,
            force_spawn=True,
            name="settling-truck",
            vehicle_config={
                "spawn_lane_index": twin_lane.index,
                "spawn_longitude": float(twin_longitudinal),
                "spawn_lateral": 0.0,
            },
        )
        for _ in range(40):
            env.step((0.0, 0.0))
            twin.set_velocity([0.0, 0.0])

        # A moving truck's suspension settles it; the static one hangs its wheels
        # just above the road, no more than 10 cm higher (it was 0.77 m).
        settled_z = twin.origin.getZ()
        assert 0.0 < truck.origin.getZ() - settled_z < 0.1
        # The measured scene does not move: same place, and no velocity.
        assert truck.position == pytest.approx(tuple(placed))
        assert tuple(truck.velocity) == (0.0, 0.0)
        # Static cars keep MetaDrive's own spawn height.
        assert car.origin.getZ() == pytest.approx(car.HEIGHT / 2)
    finally:
        env.close()
