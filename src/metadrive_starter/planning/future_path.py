from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from metadrive_starter.types import Point2D


Vector2D = tuple[float, float]


@dataclass(frozen=True)
class PathProjection:
    """A world-space point projected onto the ego vehicle's future path."""

    distance_along_path_m: float
    cross_track_m: float
    tangent_world: Vector2D
    ego_tangent_world: Vector2D


class FuturePathProjector(Protocol):
    """Replaceable contract for predicting where the ego vehicle will travel."""

    def project(
        self,
        ego_position: Point2D,
        object_position: Point2D,
        *,
        max_distance_m: float,
    ) -> PathProjection | None:
        """Project an object onto the future path, or return None when it is not ahead."""


@dataclass
class PolylineFuturePath:
    """Project objects onto a sampled route while tracking forward progress."""

    route: Sequence[Point2D]
    search_window_segments: int = 20
    _progress_segment: int = field(init=False, default=0)
    _progress_t: float = field(init=False, default=0.0)
    _points: tuple[Point2D, ...] = field(init=False)
    _segments: tuple[tuple[Point2D, Point2D, float, Vector2D], ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.search_window_segments <= 0:
            raise ValueError("search_window_segments must be positive")
        self.reset(self.route)

    def reset(self, route: Sequence[Point2D]) -> None:
        points = tuple((float(point[0]), float(point[1])) for point in route)
        if len(points) < 2:
            raise ValueError("future path requires at least two waypoints")

        segments: list[tuple[Point2D, Point2D, float, Vector2D]] = []
        for start, end in zip(points[:-1], points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            if length > 1e-9:
                segments.append((start, end, length, (dx / length, dy / length)))
        if not segments:
            raise ValueError("future path requires at least two distinct waypoints")

        self.route = points
        self._points = points
        self._segments = tuple(segments)
        self._progress_segment = 0
        self._progress_t = 0.0

    def project(
        self,
        ego_position: Point2D,
        object_position: Point2D,
        *,
        max_distance_m: float,
    ) -> PathProjection | None:
        if max_distance_m <= 0.0:
            raise ValueError("max_distance_m must be positive")

        ego_segment, ego_t = self._locate_ego(ego_position)
        ego_tangent = self._segments[ego_segment][3]
        distance_to_segment = 0.0
        best: PathProjection | None = None

        for index in range(ego_segment, len(self._segments)):
            start, _, length, tangent = self._segments[index]
            minimum_t = ego_t if index == ego_segment else 0.0
            raw_t = _projection_parameter(object_position, start, tangent, length)
            if raw_t < minimum_t or raw_t > 1.0:
                distance_to_segment += (1.0 - minimum_t) * length
                if distance_to_segment > max_distance_m:
                    break
                continue

            distance_along = distance_to_segment + (raw_t - minimum_t) * length
            if distance_along > max_distance_m:
                break
            projected = (
                start[0] + tangent[0] * raw_t * length,
                start[1] + tangent[1] * raw_t * length,
            )
            offset = (
                object_position[0] - projected[0],
                object_position[1] - projected[1],
            )
            cross_track = tangent[0] * offset[1] - tangent[1] * offset[0]
            candidate = PathProjection(
                distance_along_path_m=distance_along,
                cross_track_m=cross_track,
                tangent_world=tangent,
                ego_tangent_world=ego_tangent,
            )
            if best is None or (abs(candidate.cross_track_m), candidate.distance_along_path_m) < (
                abs(best.cross_track_m),
                best.distance_along_path_m,
            ):
                best = candidate

            distance_to_segment += (1.0 - minimum_t) * length
            if distance_to_segment > max_distance_m:
                break

        return best

    def _locate_ego(self, ego_position: Point2D) -> tuple[int, float]:
        search_end = min(len(self._segments), self._progress_segment + self.search_window_segments)
        candidates: list[tuple[float, int, float]] = []
        for index in range(self._progress_segment, search_end):
            start, _, length, tangent = self._segments[index]
            raw_t = _projection_parameter(ego_position, start, tangent, length)
            minimum_t = self._progress_t if index == self._progress_segment else 0.0
            t = min(1.0, max(minimum_t, raw_t))
            projected = (
                start[0] + tangent[0] * t * length,
                start[1] + tangent[1] * t * length,
            )
            distance_squared = (ego_position[0] - projected[0]) ** 2 + (
                ego_position[1] - projected[1]
            ) ** 2
            candidates.append((distance_squared, index, t))

        _, segment, t = min(candidates)
        if t >= 1.0 - 1e-9 and segment + 1 < len(self._segments):
            segment += 1
            t = 0.0
        self._progress_segment = segment
        self._progress_t = t
        return segment, t


def _projection_parameter(
    point: Point2D,
    segment_start: Point2D,
    tangent: Vector2D,
    segment_length: float,
) -> float:
    displacement = (point[0] - segment_start[0], point[1] - segment_start[1])
    return (displacement[0] * tangent[0] + displacement[1] * tangent[1]) / segment_length
