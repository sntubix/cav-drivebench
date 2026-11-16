"""A scene's arrival point: a drive arrives at a set distance along its route, short of the road's end."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from metadrive_starter.config import EventLogSettings, config_from_dict
from metadrive_starter.events import read_event_log, to_json_value
from metadrive_starter.simulation import run_simulation
from metadrive_starter.types import ControlTick


def test_the_arrival_point_parses_and_survives_the_event_log() -> None:
    config = config_from_dict({"scenario": {"arrival_distance_m": 180}})

    logged = json.loads(json.dumps(to_json_value(config.to_dict())))

    assert config.scenario.arrival_distance_m == 180.0
    assert config_from_dict(logged).scenario == config.scenario
    assert config_from_dict({}).scenario.arrival_distance_m is None


@pytest.mark.parametrize("value", [0.0, -5.0, math.inf, math.nan, True, "180"])
def test_an_arrival_point_must_be_a_positive_distance(value: object) -> None:
    with pytest.raises(ValueError, match="arrival_distance_m must be finite and positive"):
        config_from_dict({"scenario": {"arrival_distance_m": value}})


class _StraightAhead:
    def update(self, tick: ControlTick) -> tuple[float, float]:
        return (0.0, 0.6)

    def reset_speed_control(self) -> None:
        pass


def _config(tmp_path: Path, arrival_distance_m: float | None, horizon: int = 300):
    config = config_from_dict(
        {
            "simulator": {"map": "SS", "horizon": horizon, "headless": True, "realtime": False},
            "camera": {"enabled": False},
            "scenario": {"arrival_distance_m": arrival_distance_m},
        }
    )
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "events.jsonl"),
        scenario_id="arrival-point",
    )
    return config


def test_a_drive_arrives_at_its_arrival_point_and_ends_there(tmp_path: Path) -> None:
    run = run_simulation(_config(tmp_path, 40.0), controller_factory=lambda settings: _StraightAhead())

    assert run.arrived is True
    assert run.steps < 300
    assert not run.crashed and not run.went_off_road
    # The ego starts 5 m along the route, so it ends 45 m along a road of about 196 m.
    assert 0.2 < run.route_completion < 0.25
    ended = [r.payload for r in read_event_log(tmp_path / "events.jsonl") if r.event_type == "run_ended"]
    assert ended[0]["summary"]["arrived"] is True


def test_without_an_arrival_point_the_same_drive_has_not_arrived(tmp_path: Path) -> None:
    arrived = run_simulation(_config(tmp_path, 40.0), controller_factory=lambda settings: _StraightAhead())

    run = run_simulation(
        _config(tmp_path / "unset", None, horizon=arrived.steps),
        controller_factory=lambda settings: _StraightAhead(),
    )

    assert run.steps == arrived.steps
    assert run.arrived is False
    assert run.route_completion == pytest.approx(arrived.route_completion, abs=1e-6)


def test_an_arrival_point_beyond_the_road_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="arrival_distance_m=400.0 lies beyond"):
        run_simulation(_config(tmp_path, 400.0), controller_factory=lambda settings: _StraightAhead())
