from pathlib import Path

import pytest

from metadrive_starter.config import load_config
from metadrive_starter.env import (
    make_env,
    navigation_waypoints,
    simulation_time_s,
    spawn_stopped_vehicle_ahead,
)
from metadrive_starter.perception import DetectionStatus, DualSceneObserver


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_oracle_and_lidar_observe_the_same_metadrive_state() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.headless = True
    env = make_env(config.simulator)
    try:
        env.reset()
        obstacle = spawn_stopped_vehicle_ahead(
            env,
            config.scenario.stopped_vehicle_ahead_m or 25.0,
        )
        env.step((0.0, 0.0))
        route = navigation_waypoints(env, config.planner.waypoint_spacing_m)
        observer = DualSceneObserver.for_route(
            route,
            detection_radius_m=config.perception.detection_radius_m,
            corridor_margin_m=config.perception.corridor_margin_m,
        )

        comparison = observer.observe(
            env,
            timestamp_s=simulation_time_s(env),
        ).comparison

        obstacle_comparison = next(
            item for item in comparison.objects if item.object_id == str(obstacle.id)
        )
        assert obstacle_comparison.status is DetectionStatus.MATCHED
        assert obstacle_comparison.distance_error_m == pytest.approx(0.0)
        assert obstacle_comparison.closing_speed_error_mps == pytest.approx(0.0)
        assert obstacle_comparison.in_path_agrees is True
    finally:
        env.close()
