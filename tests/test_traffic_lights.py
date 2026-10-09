from types import SimpleNamespace

import pytest

from metadrive_starter.traffic_lights import (
    TrafficLightController,
    TrafficLightCycle,
    TrafficLightState,
)


def test_cycle_starts_red_then_changes_green_yellow_and_repeats() -> None:
    cycle = TrafficLightCycle(
        red_duration_s=12.0,
        green_duration_s=10.0,
        yellow_duration_s=2.0,
    )

    assert cycle.state_at(0.0) is TrafficLightState.RED
    assert cycle.state_at(11.999) is TrafficLightState.RED
    assert cycle.state_at(12.0) is TrafficLightState.GREEN
    assert cycle.state_at(22.0) is TrafficLightState.YELLOW
    assert cycle.state_at(24.0) is TrafficLightState.RED


def test_controller_only_applies_state_on_transition() -> None:
    calls: list[str] = []
    light = SimpleNamespace(
        set_red=lambda: calls.append("red"),
        set_green=lambda: calls.append("green"),
        set_yellow=lambda: calls.append("yellow"),
    )
    controller = TrafficLightController(
        light,
        TrafficLightCycle(red_duration_s=2.0, green_duration_s=3.0, yellow_duration_s=1.0),
    )

    assert controller.update(0.0) is TrafficLightState.RED
    assert controller.update(1.0) is None
    assert controller.update(2.0) is TrafficLightState.GREEN
    assert calls == ["red", "green"]


@pytest.mark.parametrize("value", [-1.0, float("nan"), True])
def test_cycle_rejects_invalid_simulation_time(value: object) -> None:
    with pytest.raises(ValueError, match="simulation_time_s"):
        TrafficLightCycle().state_at(value)  # type: ignore[arg-type]
