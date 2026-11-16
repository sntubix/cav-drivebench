"""A car the safety floor stops behind an obstacle stays put, and an ego that leaves its
route ends its drive as an outcome, not an error. MetaDrive drives; no model runs.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter.config import ControllerSettings, EventLogSettings, load_config
from metadrive_starter.env import (
    EgoLeftRoute,
    control_timestep_s,
    hold_ego_at_standstill,
    lane_change_waypoints,
    make_env,
    navigation_waypoints,
)
from metadrive_starter.events import read_event_log
from metadrive_starter.replay import replay_event_log
from metadrive_starter.simulation import run_simulation
from metadrive_starter.types import Action, ControlTick
from metadrive_starter.vla import (
    ModelRequest,
    ModelResponse,
    VLAInferencePipeline,
    VLAInferenceScheduler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LocksItsWheelsOnceStopped:
    """Asks for throttle throughout, and turns to full lock once the car has stopped.

    A steering loop on a growing heading error ends there on a car that creeps, so
    this is the worst case the safety floor's brake has to hold.
    """

    def __init__(self, settings: ControllerSettings) -> None:
        del settings
        self._has_moved = False

    def update(self, tick: ControlTick) -> Action:
        self._has_moved = self._has_moved or tick.speed_mps > 2.0
        stopped = self._has_moved and tick.speed_mps < 0.5
        return (1.0 if stopped else 0.0, 0.3)

    def reset_speed_control(self) -> None:
        pass


class EasesOffGently:
    """Reaches 1 m/s, then brakes as lightly as a speed loop easing towards a low target."""

    def __init__(self, settings: ControllerSettings) -> None:
        del settings
        self._easing = False

    def update(self, tick: ControlTick) -> Action:
        self._easing = self._easing or tick.speed_mps > 1.0
        return (0.0, -0.005 if self._easing else 0.3)

    def reset_speed_control(self) -> None:
        pass


class KeepsLane:
    def __init__(self, settings: ControllerSettings) -> None:
        del settings

    def update(self, tick: ControlTick) -> Action:
        return (2.0 * tick.heading_error_rad, 0.3 if tick.speed_mps < 4.0 else 0.0)

    def reset_speed_control(self) -> None:
        pass


class ChangesLaneRightOnce:
    def __init__(self) -> None:
        self._asked = False

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        del timeout_s
        action = "KEEP_LANE" if self._asked else "CHANGE_LANE_RIGHT"
        self._asked = True
        return ModelResponse(
            request_id=request.request_id,
            text=json.dumps(
                {
                    "scene_summary": "Stopped vehicle ahead; right lane clear.",
                    "relevant_hazards": [
                        {"type": "vehicle", "relative_location": "front", "risk": "medium"}
                    ],
                    "meta_action": action,
                    "target_speed_mps": 4.0,
                    "confidence": 0.95,
                    "brief_justification": "Use clear adjacent lane.",
                }
            ),
            model_id="integration-scripted-vlm",
            latency_s=0.0,
        )


def _turn_onto_the_opposite_carriageway(env: object) -> None:
    """Put the ego where a car that pivoted round ends up: facing back, across the road."""
    ego = env.agent  # type: ignore[attr-defined]
    network = env.current_map.road_network  # type: ignore[attr-defined]
    start, end, _ = ego.lane_index
    own_lane = network.graph[start][end][0]
    longitudinal, _ = own_lane.local_coordinates(ego.position)
    # MetaDrive names the carriageway that runs back along a road with a leading '-'.
    opposite = network.graph[f"-{end}"][f"-{start}"][0]
    along = float(opposite.length) - float(longitudinal)
    ego.set_position(opposite.position(along, 0.0))
    ego.set_heading_theta(opposite.heading_theta_at(along))
    ego.navigation.update_localization(ego)


def _stop_then_brake_on_full_lock(*, hold: bool, seconds: float) -> tuple[float, float]:
    """Brake a car to a stop, then keep full brake on full lock; return its drift and turn."""
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    env = make_env(replace(config.simulator, map="S", headless=True, use_lidar=False))
    try:
        env.reset()
        ego = env.agent
        while ego.speed < 5.0:
            env.step((0.0, 0.3))
        for _ in range(30):
            env.step((0.0, -1.0))
        position, heading = tuple(ego.position), float(ego.heading_theta)
        for _ in range(round(seconds / control_timestep_s(config.simulator))):
            if hold:
                hold_ego_at_standstill(env)
            env.step((1.0, -1.0))
        return (
            math.dist(position, tuple(ego.position)),
            abs(math.degrees(float(ego.heading_theta) - heading)),
        )
    finally:
        env.close()


def test_metadrive_brake_alone_lets_a_stopped_car_creep_and_pivot() -> None:
    # Why the runtime holds a stopped car: the brake's capped friction impulse
    # limit-cycles at a standstill. If MetaDrive ever holds the car itself, the
    # hold is no longer needed.
    drift_m, turn_deg = _stop_then_brake_on_full_lock(hold=False, seconds=30.0)

    assert drift_m > 0.5
    assert turn_deg > 5.0


def test_the_standstill_hold_keeps_a_stopped_car_where_it_stopped() -> None:
    drift_m, turn_deg = _stop_then_brake_on_full_lock(hold=True, seconds=30.0)

    assert drift_m < 0.01
    assert turn_deg < 0.1


def test_a_car_the_floor_stops_behind_a_vehicle_stays_where_it_stopped(
    tmp_path: Path,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.headless = True
    config.simulator.use_lidar = False
    config.simulator.horizon = 600
    config.event_log = EventLogSettings(
        enabled=True, path=str(tmp_path / "events.jsonl"), scenario_id="stuck-ego"
    )

    summary = run_simulation(config, controller_factory=LocksItsWheelsOnceStopped)

    ticks = [
        record.payload
        for record in read_event_log(config.event_log.path)
        if record.event_type == "control_applied"
    ]
    # The car inches up to the floor's minimum gap, which then holds it at full
    # brake to the end of the run. Without the hold the brake's creep reopened the
    # gap, so the floor kept releasing it and the car wandered back and turned.
    held_from = next(
        (
            index
            for index in range(len(ticks))
            if all(tick["applied_command"]["brake"] == 1.0 for tick in ticks[index:])
        ),
        None,
    )
    assert held_from is not None, "the floor never held the stopped car to the end"
    held = ticks[held_from:]
    assert len(held) * control_timestep_s(config.simulator) > 30.0
    assert all(abs(tick["applied_command"]["steering"]) == 1.0 for tick in held)
    assert all(tick["standstill_hold"] for tick in held)
    first, last = held[0]["ego"], held[-1]["ego"]
    assert math.dist(first["position"], last["position"]) < 0.02
    assert abs(math.degrees(last["heading_rad"] - first["heading_rad"])) < 0.2
    assert summary.final_speed_mps < 0.05
    assert not summary.crashed
    assert not summary.left_route


def test_a_car_slowing_on_a_light_brake_is_left_to_slow_by_itself(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config.simulator.map = "S"
    config.simulator.headless = True
    config.simulator.use_lidar = False
    config.simulator.horizon = 300
    config.event_log = EventLogSettings(
        enabled=True, path=str(tmp_path / "events.jsonl"), scenario_id="light-brake"
    )

    run_simulation(config, controller_factory=EasesOffGently)

    ticks = [
        record.payload
        for record in read_event_log(config.event_log.path)
        if record.event_type == "control_applied"
    ]
    slow_and_braking = [
        tick
        for tick in ticks
        if tick["applied_command"]["brake"] > 0.0 and tick["ego"]["speed_mps"] < 0.3
    ]
    # A brake this light is weaker than lifting off; held, the car would stop dead
    # from 0.3 m/s in one step instead of rolling to a stop over a few seconds.
    assert len(slow_and_braking) > 10
    assert not any(tick["standstill_hold"] for tick in ticks)
    assert any(0.05 < tick["ego"]["speed_mps"] < 0.25 for tick in slow_and_braking)


def test_an_ego_on_the_opposite_carriageway_has_left_its_route() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    env = make_env(replace(config.simulator, map="S", headless=True, use_lidar=False))
    try:
        env.reset()
        for _ in range(10):
            env.step((0.0, 0.5))
        _turn_onto_the_opposite_carriageway(env)

        with pytest.raises(EgoLeftRoute, match="not part of the selected navigation route"):
            navigation_waypoints(env, 2.0)
        with pytest.raises(EgoLeftRoute):
            lane_change_waypoints(env, 2.0, lane_offset=1, transition_distance_m=18.0)
    finally:
        env.close()


def test_a_drive_whose_ego_leaves_its_route_ends_there_without_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.map = "S"
    config.simulator.headless = True
    config.simulator.use_lidar = False
    config.simulator.horizon = 80
    config.camera = replace(config.camera, enabled=True, width=160, height=90)
    config.vla = replace(
        config.vla,
        enabled=True,
        minimum_interval_s=0.0,
        action_speed_policy_mode="enforce",
    )
    config.planner.lane_change_enabled = True
    config.event_log = EventLogSettings(
        enabled=True, path=str(tmp_path / "events.jsonl"), scenario_id="left-route"
    )
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(ChangesLaneRightOnce()),
        minimum_interval_s=0.0,
    )
    monkeypatch.setattr(
        "metadrive_starter.simulation.build_vla_scheduler",
        lambda settings, **_: scheduler,
    )

    # The lane change starts on a car that has turned round: its lane is off the
    # route, so neither the lane change nor the route back can be sampled.
    def turned_round_first(env: object, *args: object, **kwargs: object) -> object:
        _turn_onto_the_opposite_carriageway(env)
        return lane_change_waypoints(env, *args, **kwargs)

    monkeypatch.setattr(
        "metadrive_starter.simulation.lane_change_waypoints",
        turned_round_first,
    )

    summary = run_simulation(config, controller_factory=KeepsLane)

    records = read_event_log(config.event_log.path)
    event_types = [record.event_type for record in records]
    [left] = [record for record in records if record.event_type == "ego_left_route"]
    phases = [
        record.payload["decision"]["phase"]
        for record in records
        if record.event_type == "lane_change_transition"
    ]
    assert summary.left_route is True
    assert summary.steps < config.simulator.horizon
    assert summary.arrived is False
    assert left.payload["reason"] == "ego lane is not part of the selected navigation route"
    assert left.payload["lane_index"][0].startswith("-")
    assert phases == ["started", "aborted"]
    assert summary.lane_changes_aborted == 1
    assert "run_failed" not in event_types
    assert event_types[-1] == "run_ended"
    assert replay_event_log(config.event_log.path).successful is True
