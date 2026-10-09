from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from metadrive_starter.perception.scene import (
    LaneRelation,
    LocalScene,
    TrackedObject,
    TrafficLightObservation,
    TrafficLightState,
)
from metadrive_starter.planning import FuturePathProjector


@dataclass(frozen=True)
class ObjectSceneBuilder:
    """Build the shared scene contract from an adapter-provided object set."""

    detection_radius_m: float = 50.0
    corridor_margin_m: float = 0.5
    future_path: FuturePathProjector | None = None

    def __post_init__(self) -> None:
        if self.detection_radius_m <= 0.0:
            raise ValueError("detection_radius_m must be positive")
        if self.corridor_margin_m < 0.0:
            raise ValueError("corridor_margin_m must not be negative")

    def observe(
        self,
        env: Any,
        candidates: Iterable[tuple[str, Any]],
        *,
        timestamp_s: float,
        valid: bool = True,
    ) -> LocalScene:
        ego = env.agent
        ego_position = _vector2(ego.position)
        ego_velocity = _vector2(ego.velocity)
        heading = float(ego.heading_theta)
        forward = (math.cos(heading), math.sin(heading))
        left = (-forward[1], forward[0])

        tracked_objects: list[TrackedObject] = []
        traffic_lights: list[TrafficLightObservation] = []
        for object_id, candidate in candidates:
            if _is_traffic_light(candidate):
                traffic_light = self._traffic_light(
                    object_id=str(object_id),
                    candidate=candidate,
                    ego=ego,
                    ego_position=ego_position,
                    forward=forward,
                    left=left,
                )
                if traffic_light is not None:
                    traffic_lights.append(traffic_light)
                continue
            tracked = self._tracked_object(
                object_id=str(object_id),
                candidate=candidate,
                ego=ego,
                ego_position=ego_position,
                ego_velocity=ego_velocity,
                forward=forward,
                left=left,
            )
            if tracked is not None:
                tracked_objects.append(tracked)

        lane = ego.lane
        longitude, lane_offset = lane.local_coordinates(ego.position)
        lane_heading = float(lane.heading_theta_at(longitude))
        heading_error = _wrap_angle(lane_heading - heading)
        left_lane_available, right_lane_available = _lane_availability(env, ego)

        return LocalScene(
            timestamp_s=timestamp_s,
            ego_speed_mps=float(ego.speed),
            ego_length_m=float(ego.LENGTH),
            ego_width_m=float(ego.WIDTH),
            lane_offset_m=float(lane_offset),
            heading_error_rad=heading_error,
            objects=tuple(tracked_objects),
            traffic_lights=tuple(traffic_lights),
            distance_to_left_boundary_m=_optional_float(getattr(ego, "dist_to_left_side", None)),
            distance_to_right_boundary_m=_optional_float(getattr(ego, "dist_to_right_side", None)),
            left_lane_available=left_lane_available,
            right_lane_available=right_lane_available,
            valid=valid,
        )

    def _traffic_light(
        self,
        *,
        object_id: str,
        candidate: Any,
        ego: Any,
        ego_position: tuple[float, float],
        forward: tuple[float, float],
        left: tuple[float, float],
    ) -> TrafficLightObservation | None:
        try:
            position = _vector2(candidate.position)
        except (AttributeError, TypeError, ValueError):
            return None
        displacement = (position[0] - ego_position[0], position[1] - ego_position[1])
        if math.hypot(*displacement) > self.detection_radius_m:
            return None

        relative_position = (_dot(displacement, forward), _dot(displacement, left))
        projection = (
            self.future_path.project(
                ego_position,
                position,
                max_distance_m=self.detection_radius_m,
            )
            if self.future_path is not None
            else None
        )
        if projection is not None:
            path_distance_m = projection.distance_along_path_m
            width_m = float(getattr(candidate, "WIDTH", 0.0))
            in_path = abs(projection.cross_track_m) <= (
                float(ego.WIDTH) / 2.0 + width_m / 2.0 + self.corridor_margin_m
            )
        else:
            path_distance_m = None
            width_m = float(getattr(candidate, "WIDTH", 0.0))
            in_path = abs(relative_position[1]) <= (
                float(ego.WIDTH) / 2.0 + width_m / 2.0 + self.corridor_margin_m
            )

        return TrafficLightObservation(
            light_id=object_id,
            state=_traffic_light_state(getattr(candidate, "status", None)),
            relative_position_m=relative_position,
            in_path=in_path,
            path_distance_m=path_distance_m,
        )

    def _tracked_object(
        self,
        *,
        object_id: str,
        candidate: Any,
        ego: Any,
        ego_position: tuple[float, float],
        ego_velocity: tuple[float, float],
        forward: tuple[float, float],
        left: tuple[float, float],
    ) -> TrackedObject | None:
        if candidate is ego:
            return None

        try:
            position = _vector2(candidate.position)
            length_m = float(candidate.LENGTH)
            width_m = float(candidate.WIDTH)
        except (AttributeError, TypeError, ValueError):
            return None

        displacement = (position[0] - ego_position[0], position[1] - ego_position[1])
        if math.hypot(*displacement) > self.detection_radius_m:
            return None

        try:
            velocity = _vector2(candidate.velocity)
        except (AttributeError, TypeError, ValueError):
            velocity = (0.0, 0.0)
        relative_velocity = (velocity[0] - ego_velocity[0], velocity[1] - ego_velocity[1])
        relative_position_ego = (_dot(displacement, forward), _dot(displacement, left))
        relative_velocity_ego = (_dot(relative_velocity, forward), _dot(relative_velocity, left))

        path_distance_m: float | None = None
        path_relative_velocity_mps: float | None = None
        path_cross_track_m: float | None = None
        projection = (
            self.future_path.project(
                ego_position,
                position,
                max_distance_m=self.detection_radius_m,
            )
            if self.future_path is not None
            else None
        )

        if projection is not None:
            object_heading = float(
                getattr(
                    candidate,
                    "heading_theta",
                    math.atan2(projection.tangent_world[1], projection.tangent_world[0]),
                )
            )
            path_heading = math.atan2(
                projection.tangent_world[1],
                projection.tangent_world[0],
            )
            heading_difference = _wrap_angle(object_heading - path_heading)
            object_lateral_extent = (
                abs(math.sin(heading_difference)) * length_m / 2.0
                + abs(math.cos(heading_difference)) * width_m / 2.0
            )
            corridor_half_width = (
                float(ego.WIDTH) / 2.0 + object_lateral_extent + self.corridor_margin_m
            )
            path_distance_m = projection.distance_along_path_m
            path_cross_track_m = projection.cross_track_m
            path_relative_velocity_mps = _dot(velocity, projection.tangent_world) - _dot(
                ego_velocity,
                projection.ego_tangent_world,
            )
            in_path = abs(projection.cross_track_m) <= corridor_half_width
        else:
            corridor_half_width = (float(ego.WIDTH) + width_m) / 2.0 + self.corridor_margin_m
            in_path = abs(relative_position_ego[1]) <= corridor_half_width

        lane_relation = _lane_relation(
            getattr(ego, "lane_index", None),
            getattr(candidate, "lane_index", None),
            relative_position_ego[1],
            in_path,
        )
        return TrackedObject(
            object_id=object_id,
            kind=str(getattr(candidate, "metadrive_type", candidate.__class__.__name__)).lower(),
            relative_position_m=relative_position_ego,
            relative_velocity_mps=relative_velocity_ego,
            length_m=length_m,
            width_m=width_m,
            lane_relation=lane_relation,
            in_path=in_path,
            path_distance_m=path_distance_m,
            path_relative_velocity_mps=path_relative_velocity_mps,
            path_cross_track_m=path_cross_track_m,
        )


def object_identifier(candidate: Any, fallback: object) -> str:
    """Return the stable simulator ID shared by oracle and sensor adapters."""
    return str(getattr(candidate, "id", fallback))


def _is_traffic_light(candidate: Any) -> bool:
    return str(getattr(candidate, "metadrive_type", "")).upper() == "TRAFFIC_LIGHT"


def _traffic_light_state(status: object) -> TrafficLightState:
    normalized = str(status).upper()
    for state in (
        TrafficLightState.RED,
        TrafficLightState.GREEN,
        TrafficLightState.YELLOW,
    ):
        if normalized.endswith(f"_{state.value.upper()}"):
            return state
    return TrafficLightState.UNKNOWN


def _lane_availability(env: Any, ego: Any) -> tuple[bool | None, bool | None]:
    try:
        start, end, lane_number = ego.lane_index
        lanes = env.current_map.road_network.graph[start][end]
        lane_number = int(lane_number)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None, None
    # MetaDrive orders lanes from physical left to right for the current road.
    # Therefore decreasing the lane index moves left and increasing it moves right.
    return lane_number > 0, lane_number + 1 < len(lanes)


def _lane_relation(
    ego_lane_index: object,
    object_lane_index: object,
    lateral_m: float,
    in_path: bool,
) -> LaneRelation:
    if isinstance(ego_lane_index, tuple) and isinstance(object_lane_index, tuple):
        if ego_lane_index[:2] == object_lane_index[:2]:
            if ego_lane_index[-1] == object_lane_index[-1]:
                return LaneRelation.SAME
            return LaneRelation.LEFT if lateral_m >= 0.0 else LaneRelation.RIGHT
    if in_path:
        return LaneRelation.CROSSING
    return LaneRelation.UNKNOWN


def _vector2(value: object) -> tuple[float, float]:
    return float(value[0]), float(value[1])  # type: ignore[index]


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _wrap_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad
