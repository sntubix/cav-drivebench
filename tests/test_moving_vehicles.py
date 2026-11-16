"""Vehicles placed along the ego route that drive on in their lane at a set speed."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from metadrive_starter.config import (
    EventLogSettings,
    MovingVehicleSettings,
    config_from_dict,
    load_config,
)
from metadrive_starter.env import (
    make_env,
    spawn_moving_vehicle_ahead,
    spawn_static_vehicle_ahead,
)
from metadrive_starter.episode_replay import replay_episode_artifact
from metadrive_starter.events import read_event_log, to_json_value
from metadrive_starter.perception import OracleSceneAdapter
from metadrive_starter.simulation import run_simulation
from metadrive_starter.types import ControlTick

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_moving_vehicles_parse_and_survive_the_event_log() -> None:
    config = config_from_dict(
        {
            "scenario": {
                "moving_vehicles": [
                    {"vehicle_id": "lead", "distance_m": 20, "speed_mps": 3},
                    {
                        "vehicle_id": "truck",
                        "distance_m": 40.0,
                        "speed_mps": 4.5,
                        "lane_offset": 1,
                        "kind": "truck",
                    },
                ]
            }
        }
    )

    logged = json.loads(json.dumps(to_json_value(config.to_dict())))

    assert config.scenario.moving_vehicles == (
        MovingVehicleSettings("lead", 20.0, 3.0),
        MovingVehicleSettings("truck", 40.0, 4.5, lane_offset=1, kind="truck"),
    )
    assert config_from_dict(logged).scenario == config.scenario


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"vehicle_id": "lead", "distance_m": 0.0, "speed_mps": 3.0}, "distance_m must be finite and positive"),
        ({"vehicle_id": "lead", "distance_m": 20.0, "speed_mps": 0.0}, "speed_mps must be finite and positive"),
        ({"vehicle_id": "lead", "distance_m": 20.0}, "missing 1 required positional argument: 'speed_mps'"),
        ({"vehicle_id": "lead", "distance_m": 20.0, "speed_mps": 3.0, "kind": "bus"}, "kind must be 'car' or 'truck'"),
        ({"vehicle_id": "lead", "distance_m": 20.0, "speed_mps": 3.0, "lane_offset": 1.5}, "lane_offset must be an integer"),
        ({"vehicle_id": "lead", "distance_m": 20.0, "speed_mps": 3.0, "lateral_m": 0.5}, "unexpected keyword argument 'lateral_m'"),
        ({"vehicle_id": "a lead", "distance_m": 20.0, "speed_mps": 3.0}, "vehicle_id must contain only"),
    ],
)
def test_moving_vehicles_reject_what_cannot_be_placed(entry: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        config_from_dict({"scenario": {"moving_vehicles": [entry]}})


def test_moving_vehicle_ids_are_unique_among_all_vehicles_and_entries_are_mappings() -> None:
    with pytest.raises(ValueError, match="vehicle ids must be unique"):
        config_from_dict(
            {
                "scenario": {
                    "stopped_vehicles": [{"vehicle_id": "car", "distance_m": 40.0}],
                    "moving_vehicles": [{"vehicle_id": "car", "distance_m": 20.0, "speed_mps": 3.0}],
                }
            }
        )
    with pytest.raises(ValueError, match="moving_vehicles must be a list of mappings"):
        config_from_dict({"scenario": {"moving_vehicles": ["lead"]}})


def _drive(
    map_name: str,
    steps: int,
    place: Callable[[Any], Any],
) -> tuple[list[tuple[tuple[float, float], float, tuple, float]], Any]:
    """Hold the ego still and record the placed vehicle's position, speed, lane and offset."""
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    env = make_env(
        replace(
            config.simulator,
            map=map_name,
            headless=True,
            traffic_density=0.0,
            obstacle_probability=0.0,
        ),
        replace(config.camera, enabled=False),
    )
    try:
        env.reset()
        vehicle = place(env)
        trace = []
        for _ in range(steps):
            env.step((0.0, 0.0))
            _, lateral = vehicle.lane.local_coordinates(vehicle.position)
            trace.append(
                (tuple(vehicle.position), float(vehicle.speed), vehicle.lane_index, float(lateral))
            )
        scene = OracleSceneAdapter(detection_radius_m=80.0).observe(env, timestamp_s=1.0)
        return trace, scene
    finally:
        env.close()


def test_a_moving_car_keeps_its_speed_and_lane_through_a_curve_the_same_way_twice() -> None:
    def place(env: Any) -> Any:
        with pytest.raises(ValueError, match="speed_mps must be finite and positive"):
            spawn_moving_vehicle_ahead(env, 20.0, 0.0)
        return spawn_moving_vehicle_ahead(env, 20.0, 5.0, name="scenario-lead")

    trace, _ = _drive("C", 200, place)

    speeds = [speed for _, speed, _, _ in trace]
    assert min(speeds) == pytest.approx(5.0, abs=0.1)
    assert max(speeds) == pytest.approx(5.0, abs=0.01)
    assert max(abs(lateral) for *_, lateral in trace) < 0.3
    roads = [lane_index[:2] for _, _, lane_index, _ in trace]
    # The lane index follows it from the straight into the curve.
    assert list(dict.fromkeys(roads)) == [(">>", ">>>"), (">>>", "1C0_0_")]
    assert all(lane_index[-1] == 0 for _, _, lane_index, _ in trace)

    assert _drive("C", 200, lambda env: spawn_moving_vehicle_ahead(env, 20.0, 5.0, name="scenario-lead"))[0] == trace


def test_a_moving_truck_in_the_next_lane_reaches_the_measured_scene_at_its_speed() -> None:
    trace, scene = _drive(
        "S",
        30,
        lambda env: spawn_moving_vehicle_ahead(
            env, 25.0, 4.0, lane_offset=1, vehicle_kind="truck", name="scenario-truck"
        ),
    )

    assert min(speed for _, speed, _, _ in trace) == pytest.approx(4.0, abs=0.1)
    assert trace[-1][2][-1] == 1
    measured = {item.object_id: item for item in scene.objects}["scenario-truck"]
    # The ego stands still, so the truck's relative velocity is its own.
    assert measured.relative_velocity_mps[0] == pytest.approx(4.0, abs=0.05)
    assert measured.relative_position_m[0] == pytest.approx(25.0 + 4.0 * 3.0, abs=1.0)
    assert measured.relative_position_m[1] < 0.0


def test_a_moving_car_stops_behind_a_stopped_one() -> None:
    def place(env: Any) -> Any:
        spawn_static_vehicle_ahead(env, 60.0, name="scenario-stopped")
        return spawn_moving_vehicle_ahead(env, 20.0, 5.0, name="scenario-lead")

    trace, scene = _drive("S", 200, place)

    assert trace[-1][1] < 0.5
    measured = {item.object_id: item for item in scene.objects}
    gap_m = (
        measured["scenario-stopped"].relative_position_m[0]
        - measured["scenario-lead"].relative_position_m[0]
    )
    assert 6.0 < gap_m < 20.0


class _Coasting:
    def update(self, tick: ControlTick) -> tuple[float, float]:
        return (0.0, 0.0)

    def reset_speed_control(self) -> None:
        pass


def test_moving_vehicles_are_logged_like_stopped_ones(tmp_path: Path) -> None:
    config = config_from_dict(
        {
            "simulator": {"map": "S", "horizon": 2, "headless": True},
            "scenario": {
                "stopped_vehicles": [{"vehicle_id": "truck", "distance_m": 60.0, "kind": "truck"}],
                "moving_vehicles": [
                    {"vehicle_id": "lead", "distance_m": 20.0, "speed_mps": 3.0, "lane_offset": 1}
                ],
            },
        }
    )
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "events.jsonl"),
        scenario_id="moving-vehicles",
    )

    run_simulation(config, controller_factory=lambda settings: _Coasting())

    spawned = [
        record.payload
        for record in read_event_log(config.event_log.path)
        if record.event_type == "scenario_vehicle_spawned"
    ]
    assert spawned[0].keys() == spawned[1].keys()
    assert {key: spawned[1][key] for key in ("vehicle_id", "distance_m", "lane_offset", "speed_mps", "kind")} == {
        "vehicle_id": "lead",
        "distance_m": 20.0,
        "lane_offset": 1,
        "speed_mps": 3.0,
        "kind": "car",
    }


def test_a_recorded_run_replays_the_moving_vehicle_where_it_drove(tmp_path: Path) -> None:
    config = config_from_dict(
        {
            "simulator": {"map": "S", "horizon": 20, "headless": True, "realtime": False},
            "scenario": {
                "moving_vehicles": [{"vehicle_id": "lead", "distance_m": 20.0, "speed_mps": 4.0}],
            },
            "episode_recording": {"enabled": True, "path": str(tmp_path / "recorded")},
        }
    )

    run = run_simulation(config, controller_factory=lambda settings: _Coasting())
    # The artifact is this test's own, so trusting its pickle is safe.
    replay = replay_episode_artifact(tmp_path / "recorded", trusted=True, render=False, paced=False)

    assert replay.successful is True
    assert replay.replayed_steps == run.steps == 20
    assert replay.verified_control_frames == run.steps + 1
