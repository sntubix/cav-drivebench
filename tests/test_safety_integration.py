from pathlib import Path

import pytest

from metadrive_starter.config import EventLogSettings, load_config
from metadrive_starter.events import read_event_log
from metadrive_starter.simulation import RunSummary, run_simulation


pytestmark = pytest.mark.instructor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_profile(filename: str) -> RunSummary:
    config = load_config(PROJECT_ROOT / "configs" / filename)
    config.simulator.headless = True
    return run_simulation(config)


def _assert_safe_stop(summary: RunSummary) -> None:
    assert summary.safety_interventions > 0
    assert summary.emergency_brakes > 0
    assert summary.minimum_safety_gap_m is not None
    assert summary.minimum_safety_gap_m >= 2.5
    assert summary.final_speed_mps < 2.0
    assert not summary.crashed
    assert not summary.went_off_road


def test_oracle_supervisor_stops_before_a_vehicle_on_a_straight(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.headless = True
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "events.jsonl"),
        scenario_id="pid-safety-override",
    )

    summary = run_simulation(config)

    _assert_safe_stop(summary)
    reset_events = [
        record
        for record in read_event_log(tmp_path / "events.jsonl")
        if record.event_type == "speed_pid_reset"
    ]
    assert any(
        "local safety overrode PID output" in record.payload["reasons"]
        for record in reset_events
    )


def test_oracle_supervisor_stops_before_a_vehicle_on_a_curve() -> None:
    summary = _run_profile("demo-emergency-braking-curve.yaml")

    _assert_safe_stop(summary)
    assert summary.route_completion > 0.3


def test_lidar_supervisor_stops_before_a_vehicle_on_a_straight() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-emergency-braking.yaml")
    config.simulator.headless = True
    config.perception.safety_source = "lidar"

    _assert_safe_stop(run_simulation(config))
