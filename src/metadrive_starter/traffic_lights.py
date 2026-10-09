from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TrafficLightState(str, Enum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TrafficLightCycle:
    initial_state: TrafficLightState = TrafficLightState.RED
    red_duration_s: float = 12.0
    green_duration_s: float = 10.0
    yellow_duration_s: float = 2.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.initial_state, TrafficLightState)
            or self.initial_state is TrafficLightState.UNKNOWN
        ):
            raise ValueError("initial_state must be a TrafficLightState")
        for name, value in {
            "red_duration_s": self.red_duration_s,
            "green_duration_s": self.green_duration_s,
            "yellow_duration_s": self.yellow_duration_s,
        }.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")

    def state_at(self, simulation_time_s: float) -> TrafficLightState:
        if (
            not isinstance(simulation_time_s, (int, float))
            or isinstance(simulation_time_s, bool)
            or not math.isfinite(simulation_time_s)
            or simulation_time_s < 0.0
        ):
            raise ValueError("simulation_time_s must be finite and non-negative")

        order = (
            TrafficLightState.RED,
            TrafficLightState.GREEN,
            TrafficLightState.YELLOW,
        )
        initial_index = order.index(self.initial_state)
        phases = order[initial_index:] + order[:initial_index]
        durations = {
            TrafficLightState.RED: self.red_duration_s,
            TrafficLightState.GREEN: self.green_duration_s,
            TrafficLightState.YELLOW: self.yellow_duration_s,
        }
        phase_time_s = simulation_time_s % sum(durations.values())
        for state in phases:
            duration_s = durations[state]
            if phase_time_s < duration_s:
                return state
            phase_time_s -= duration_s
        return phases[-1]


class TrafficLightController:
    """Apply a deterministic simulation-time cycle to one MetaDrive light object."""

    def __init__(self, light: Any | tuple[Any, ...], cycle: TrafficLightCycle):
        self.lights = light if isinstance(light, tuple) else (light,)
        if not self.lights:
            raise ValueError("at least one traffic light is required")
        self.light = self.lights[0]
        self.cycle = cycle
        self.current_state: TrafficLightState | None = None

    def update(self, simulation_time_s: float) -> TrafficLightState | None:
        state = self.cycle.state_at(simulation_time_s)
        if state is self.current_state:
            return None
        for light in self.lights:
            set_traffic_light_state(light, state)
        self.current_state = state
        return state


def set_traffic_light_state(light: Any, state: TrafficLightState) -> None:
    if not isinstance(state, TrafficLightState) or state is TrafficLightState.UNKNOWN:
        raise ValueError("state must be red, green, or yellow")
    {
        TrafficLightState.RED: light.set_red,
        TrafficLightState.GREEN: light.set_green,
        TrafficLightState.YELLOW: light.set_yellow,
    }[state]()
    visual_scale = float(getattr(light, "drivebench_visual_scale", 1.0))
    if visual_scale != 1.0 and getattr(light, "current_light", None) is not None:
        light.current_light.setScale(visual_scale)
