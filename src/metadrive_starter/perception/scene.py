from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from metadrive_starter.traffic_lights import TrafficLightState


Vector2D = tuple[float, float]


class LaneRelation(str, Enum):
    SAME = "same"
    LEFT = "left"
    RIGHT = "right"
    CROSSING = "crossing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TrackedObject:
    """Object state in the ego frame: forward is +x and left is +y."""

    object_id: str
    kind: str
    relative_position_m: Vector2D
    relative_velocity_mps: Vector2D
    length_m: float
    width_m: float
    lane_relation: LaneRelation = LaneRelation.UNKNOWN
    in_path: bool = False
    path_distance_m: float | None = None
    path_relative_velocity_mps: float | None = None
    path_cross_track_m: float | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.length_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("object dimensions must be positive")
        if self.path_distance_m is not None and self.path_distance_m < 0.0:
            raise ValueError("path_distance_m must not be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class TrafficLightObservation:
    """Traffic-light ground truth kept separate from collision hazards."""

    light_id: str
    state: TrafficLightState
    relative_position_m: Vector2D
    in_path: bool = False
    path_distance_m: float | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.light_id, str) or not self.light_id:
            raise ValueError("traffic light id must not be empty")
        if not isinstance(self.state, TrafficLightState):
            raise ValueError("traffic light state must be a TrafficLightState")
        if len(self.relative_position_m) != 2 or not all(
            math.isfinite(value) for value in self.relative_position_m
        ):
            raise ValueError("traffic light relative position must contain two finite values")
        if self.path_distance_m is not None and self.path_distance_m < 0.0:
            raise ValueError("traffic light path_distance_m must not be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("traffic light confidence must be between 0 and 1")


@dataclass(frozen=True)
class LocalScene:
    """One local, timestamped perception snapshot used by safety logic."""

    timestamp_s: float
    ego_speed_mps: float
    ego_length_m: float
    ego_width_m: float
    lane_offset_m: float
    heading_error_rad: float
    objects: tuple[TrackedObject, ...] = ()
    traffic_lights: tuple[TrafficLightObservation, ...] = ()
    distance_to_left_boundary_m: float | None = None
    distance_to_right_boundary_m: float | None = None
    left_lane_available: bool | None = None
    right_lane_available: bool | None = None
    valid: bool = True

    def __post_init__(self) -> None:
        if self.ego_speed_mps < 0.0:
            raise ValueError("ego_speed_mps must not be negative")
        if self.ego_length_m <= 0.0 or self.ego_width_m <= 0.0:
            raise ValueError("ego dimensions must be positive")
