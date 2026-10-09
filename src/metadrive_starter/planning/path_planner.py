from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from metadrive_starter.types import PerceptionFrame, Plan, Point2D


@dataclass
class WaypointPathPlanner:
    route: Sequence[Point2D]
    lookahead_m: float = 15.0
    curvature_preview_m: float = 40.0
    maximum_lateral_acceleration_mps2: float = 2.0
    minimum_curve_speed_mps: float = 4.0
    _progress_m: float = field(init=False, default=0.0)
    _segments: list[_Segment] = field(init=False, default_factory=list)
    _curvatures: list[tuple[float, float]] = field(init=False, default_factory=list)
    _total_length_m: float = field(init=False, default=0.0)

    _SEARCH_WINDOW = 50

    def __post_init__(self) -> None:
        if not self.route:
            raise ValueError("route must contain at least one waypoint")
        if self.lookahead_m <= 0:
            raise ValueError("lookahead_m must be positive")
        if self.curvature_preview_m <= 0:
            raise ValueError("curvature_preview_m must be positive")
        if self.maximum_lateral_acceleration_mps2 <= 0:
            raise ValueError("maximum_lateral_acceleration_mps2 must be positive")
        if self.minimum_curve_speed_mps < 0:
            raise ValueError("minimum_curve_speed_mps must be non-negative")
        self._build_segments()

    def reset(self, route: Sequence[Point2D] | None = None) -> None:
        if route is not None:
            if not route:
                raise ValueError("route must contain at least one waypoint")
            self.route = route
        self._progress_m = 0.0
        self._build_segments()

    def plan(self, frame: PerceptionFrame) -> Plan:
        position = frame.ego.position
        if not self._segments:
            target = self.route[0]
            dx = target[0] - position[0]
            dy = target[1] - position[1]
            target_heading = math.atan2(dy, dx)
            return Plan(
                target=target,
                heading_error_rad=_wrap_angle(target_heading - frame.ego.heading_rad),
                distance_m=math.hypot(dx, dy),
            )

        projection, segment = self._project(position)
        target, _ = self._point_at(
            min(self._progress_m + self.lookahead_m, self._total_length_m)
        )
        _, heading_segment = self._point_at(
            min(self._progress_m + min(1.0, self.lookahead_m), self._total_length_m)
        )
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        target_heading = math.atan2(heading_segment.dy, heading_segment.dx)
        heading_error = _wrap_angle(target_heading - frame.ego.heading_rad)
        offset_x = position[0] - projection[0]
        offset_y = position[1] - projection[1]
        lateral_error = (segment.dy * offset_x - segment.dx * offset_y) / segment.length_m
        return Plan(
            target=target,
            heading_error_rad=heading_error,
            distance_m=math.hypot(dx, dy),
            lateral_error_m=lateral_error,
            route_speed_cap_mps=self._route_speed_cap(),
        )

    def _build_segments(self) -> None:
        self._segments = []
        self._curvatures = []
        distance_m = 0.0
        for start, end in zip(self.route, self.route[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length_m = math.hypot(dx, dy)
            if length_m <= 1e-9:
                continue
            self._segments.append(
                _Segment(start, end, dx, dy, length_m, distance_m)
            )
            distance_m += length_m
        self._total_length_m = distance_m
        for previous, current in zip(self._segments, self._segments[1:]):
            heading_change = _wrap_angle(
                math.atan2(current.dy, current.dx)
                - math.atan2(previous.dy, previous.dx)
            )
            sample_distance_m = (previous.length_m + current.length_m) / 2.0
            self._curvatures.append(
                (current.start_distance_m, abs(heading_change) / sample_distance_m)
            )

    def _route_speed_cap(self) -> float | None:
        preview_end_m = min(
            self._progress_m + self.curvature_preview_m,
            self._total_length_m,
        )
        maximum_curvature = max(
            (
                curvature
                for distance_m, curvature in self._curvatures
                if self._progress_m <= distance_m <= preview_end_m
            ),
            default=0.0,
        )
        if maximum_curvature <= 1e-4:
            return None
        return max(
            self.minimum_curve_speed_mps,
            math.sqrt(self.maximum_lateral_acceleration_mps2 / maximum_curvature),
        )

    def _project(self, position: Point2D) -> tuple[Point2D, _Segment]:
        start_index = self._segment_index_at(self._progress_m)
        search_end = min(len(self._segments), start_index + self._SEARCH_WINDOW)
        best: tuple[float, float, Point2D, _Segment] | None = None
        for segment in self._segments[start_index:search_end]:
            relative_x = position[0] - segment.start[0]
            relative_y = position[1] - segment.start[1]
            fraction = max(
                0.0,
                min(
                    1.0,
                    (relative_x * segment.dx + relative_y * segment.dy)
                    / (segment.length_m * segment.length_m),
                ),
            )
            progress_m = segment.start_distance_m + fraction * segment.length_m
            progress_m = max(self._progress_m, progress_m)
            projected, projected_segment = self._point_at(progress_m)
            distance_squared = (
                (position[0] - projected[0]) ** 2
                + (position[1] - projected[1]) ** 2
            )
            candidate = (distance_squared, progress_m, projected, projected_segment)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        assert best is not None
        _, self._progress_m, projected, segment = best
        return projected, segment

    def _point_at(self, distance_m: float) -> tuple[Point2D, _Segment]:
        segment = self._segments[self._segment_index_at(distance_m)]
        fraction = max(
            0.0,
            min(1.0, (distance_m - segment.start_distance_m) / segment.length_m),
        )
        return (
            (
                segment.start[0] + fraction * segment.dx,
                segment.start[1] + fraction * segment.dy,
            ),
            segment,
        )

    def _segment_index_at(self, distance_m: float) -> int:
        for index, segment in enumerate(self._segments):
            if distance_m < segment.start_distance_m + segment.length_m:
                return index
        return len(self._segments) - 1


@dataclass(frozen=True)
class _Segment:
    start: Point2D
    end: Point2D
    dx: float
    dy: float
    length_m: float
    start_distance_m: float


def _wrap_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad
