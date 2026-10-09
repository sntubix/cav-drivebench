from pathlib import Path

import pytest

from metadrive_starter.config import ScenarioVehicleSettings, load_config
from metadrive_starter.env import (
    lane_change_waypoints,
    lane_topology,
    make_env,
    navigation_waypoints,
    simulation_time_s,
    spawn_scenario_vehicle,
)
from metadrive_starter.perception import OracleSceneAdapter
from metadrive_starter.planning import PolylineFuturePath
from metadrive_starter.safety import CommandDisposition, VLACommandValidator
from metadrive_starter.vla import HighLevelAction, VLACommand


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_validator_rejects_lane_change_beyond_metadrive_road_edge() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.headless = True
    env = make_env(config.simulator)
    try:
        env.reset()
        route = navigation_waypoints(env, config.planner.waypoint_spacing_m)
        now_s = simulation_time_s(env)
        scene = OracleSceneAdapter(
            detection_radius_m=config.perception.detection_radius_m,
            corridor_margin_m=config.perception.corridor_margin_m,
            future_path=PolylineFuturePath(route),
        ).observe(env, timestamp_s=now_s)
        command = VLACommand(
            action=HighLevelAction.CHANGE_LANE_LEFT,
            target_speed_mps=20.0,
            issued_at_s=now_s,
            action_horizon_s=1.0,
            confidence=0.9,
        )

        decision = VLACommandValidator().validate(command, scene, now_s=now_s)

        assert scene.left_lane_available is False
        assert decision.disposition is CommandDisposition.REJECTED
        assert decision.effective_command.action is HighLevelAction.KEEP_LANE
    finally:
        env.close()


def test_validator_rejects_right_change_for_fast_rear_scenario_vehicle() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "lane-change-fixture.yaml")
    config.simulator.headless = True
    config.simulator.spawn_longitude_m = 30.0
    env = make_env(config.simulator)
    try:
        env.reset()
        rear = spawn_scenario_vehicle(
            env,
            ScenarioVehicleSettings(
                vehicle_id="fast-rear-right",
                longitudinal_offset_m=-18.0,
                lane_offset=1,
                speed_mps=12.0,
            ),
        )
        route = navigation_waypoints(env, config.planner.waypoint_spacing_m)
        now_s = simulation_time_s(env)
        scene = OracleSceneAdapter(
            detection_radius_m=config.perception.detection_radius_m,
            corridor_margin_m=config.perception.corridor_margin_m,
            future_path=PolylineFuturePath(route),
        ).observe(env, timestamp_s=now_s)
        command = VLACommand(
            action=HighLevelAction.CHANGE_LANE_RIGHT,
            target_speed_mps=4.0,
            issued_at_s=now_s,
            action_horizon_s=2.0,
            confidence=0.95,
        )

        decision = VLACommandValidator().validate(command, scene, now_s=now_s)

        assert any(item.object_id == str(rear.id) for item in scene.objects)
        assert decision.disposition is CommandDisposition.REJECTED
        assert any("violates lane-change TTC" in reason for reason in decision.reasons)
    finally:
        env.close()


def test_metadrive_lane_change_route_ends_in_adjacent_lane() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.headless = True
    env = make_env(config.simulator)
    try:
        env.reset()
        current_lane, lane_count, _ = lane_topology(env)
        lane_offset = 1 if current_lane + 1 < lane_count else -1
        target_lane = current_lane + lane_offset

        route = lane_change_waypoints(
            env,
            2.0,
            lane_offset=lane_offset,
            transition_distance_m=18.0,
        )
        start, end, _ = env.agent.lane_index
        target = env.current_map.road_network.graph[start][end][target_lane]
        transition_lateral_m = target.local_coordinates(route[9])[1]

        assert route[0] == tuple(float(value) for value in env.agent.position)
        assert len(route) > 10
        assert abs(transition_lateral_m) < 1e-6
    finally:
        env.close()


@pytest.mark.parametrize("map_code", ["S", "C", "X", "O"])
def test_lane_change_route_is_continuous_across_map_shapes(map_code: str) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.map = map_code
    config.simulator.headless = True
    env = make_env(config.simulator)
    try:
        env.reset()
        current_lane, lane_count, _ = lane_topology(env)
        lane_offset = 1 if current_lane + 1 < lane_count else -1
        target_lane_index = current_lane + lane_offset

        route = lane_change_waypoints(
            env,
            2.0,
            lane_offset=lane_offset,
            transition_distance_m=18.0,
        )
        segment_lengths = [
            ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
            for start, end in zip(route, route[1:])
        ]
        start, end, _ = env.agent.lane_index
        target_lane = env.current_map.road_network.graph[start][end][
            target_lane_index
        ]

        assert route[0] == tuple(float(value) for value in env.agent.position)
        assert min(segment_lengths) > 0.0
        assert max(segment_lengths) <= 2.1
        assert abs(target_lane.local_coordinates(route[9])[1]) < 1e-6
    finally:
        env.close()
