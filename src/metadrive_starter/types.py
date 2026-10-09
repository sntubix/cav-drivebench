from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


Point2D = tuple[float, float]
Action = tuple[float, float]


@dataclass(frozen=True)
class ControlCommand:
    """Simulator-independent continuous command with explicit braking."""

    steering: float
    throttle: float
    brake: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.steering <= 1.0:
            raise ValueError("steering must be between -1 and 1")
        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("throttle must be between 0 and 1")
        if not 0.0 <= self.brake <= 1.0:
            raise ValueError("brake must be between 0 and 1")
        if self.throttle > 0.0 and self.brake > 0.0:
            raise ValueError("throttle and brake cannot both be positive")


@dataclass(frozen=True)
class EgoState:
    position: Point2D
    heading_rad: float
    speed_mps: float


@dataclass(frozen=True)
class PerceptionFrame:
    ego: EgoState
    lidar: Sequence[float] | None = None
    raw_observation_type: str = "unknown"


@dataclass(frozen=True)
class Plan:
    target: Point2D
    heading_error_rad: float
    distance_m: float
    # Positive means ego is right of route direction. Keeping this separate
    # from heading error lets lateral control remove accumulated lane offset.
    lateral_error_m: float = 0.0
    route_speed_cap_mps: float | None = None


@dataclass(frozen=True)
class ControlTick:
    """Everything a vehicle controller sees on one control tick.

    Speeds are in m/s and time is simulation time. Both tracking errors are
    signed so that a positive error calls for positive (left) steering.
    """

    target_speed_mps: float
    speed_mps: float
    # Route heading minus ego heading, wrapped to [-pi, pi].
    heading_error_rad: float
    # Positive means ego is right of route direction.
    lateral_error_m: float
    dt_s: float


class Planner(Protocol):
    def plan(self, frame: PerceptionFrame) -> Plan:
        """Return the next local navigation target for the ego vehicle."""


class Perception(Protocol):
    def observe(self, observation: object, info: dict[str, object] | None = None) -> PerceptionFrame:
        """Convert simulator observations into a small, stable perception frame."""
