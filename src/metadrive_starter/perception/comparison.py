from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

from metadrive_starter.perception.lidar import LidarSceneAdapter
from metadrive_starter.perception.oracle import OracleSceneAdapter
from metadrive_starter.perception.scene import LocalScene, TrackedObject
from metadrive_starter.planning import PolylineFuturePath
from metadrive_starter.types import Point2D


class DetectionStatus(str, Enum):
    MATCHED = "matched"
    MISSED = "missed"
    EXTRA = "extra"


@dataclass(frozen=True)
class ObjectComparison:
    object_id: str
    status: DetectionStatus
    reference_distance_m: float | None
    candidate_distance_m: float | None
    distance_error_m: float | None
    reference_closing_speed_mps: float | None
    candidate_closing_speed_mps: float | None
    closing_speed_error_mps: float | None
    reference_ttc_s: float | None
    candidate_ttc_s: float | None
    ttc_error_s: float | None
    reference_in_path: bool | None
    candidate_in_path: bool | None
    in_path_agrees: bool | None


@dataclass(frozen=True)
class SceneComparison:
    timestamp_s: float
    reference_valid: bool
    candidate_valid: bool
    objects: tuple[ObjectComparison, ...]

    @property
    def matched_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.objects if item.status is DetectionStatus.MATCHED)

    @property
    def missed_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.objects if item.status is DetectionStatus.MISSED)

    @property
    def extra_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.objects if item.status is DetectionStatus.EXTRA)

    @property
    def in_path_disagreement_ids(self) -> tuple[str, ...]:
        return tuple(
            item.object_id
            for item in self.objects
            if item.in_path_agrees is False
        )


@dataclass(frozen=True)
class DualSceneObservation:
    oracle_scene: LocalScene
    lidar_scene: LocalScene
    comparison: SceneComparison


class SceneAdapter(Protocol):
    def observe(self, env: Any, *, timestamp_s: float) -> LocalScene:
        """Produce a local scene from one simulator state."""


@dataclass(frozen=True)
class DualSceneObserver:
    """Observe one simulator state through oracle and LiDAR paths."""

    oracle: SceneAdapter
    lidar: SceneAdapter

    @classmethod
    def for_route(
        cls,
        route: Sequence[Point2D],
        *,
        detection_radius_m: float = 50.0,
        corridor_margin_m: float = 0.5,
    ) -> DualSceneObserver:
        return cls(
            oracle=OracleSceneAdapter(
                detection_radius_m=detection_radius_m,
                corridor_margin_m=corridor_margin_m,
                future_path=PolylineFuturePath(route),
            ),
            lidar=LidarSceneAdapter(
                detection_radius_m=detection_radius_m,
                corridor_margin_m=corridor_margin_m,
                future_path=PolylineFuturePath(route),
            ),
        )

    def observe(self, env: Any, *, timestamp_s: float) -> DualSceneObservation:
        oracle_scene = self.oracle.observe(env, timestamp_s=timestamp_s)
        lidar_scene = self.lidar.observe(env, timestamp_s=timestamp_s)
        return DualSceneObservation(
            oracle_scene=oracle_scene,
            lidar_scene=lidar_scene,
            comparison=compare_scenes(oracle_scene, lidar_scene),
        )


def compare_scenes(reference: LocalScene, candidate: LocalScene) -> SceneComparison:
    reference_objects = {item.object_id: item for item in reference.objects}
    candidate_objects = {item.object_id: item for item in candidate.objects}
    comparisons = tuple(
        _compare_object(
            object_id,
            reference,
            reference_objects.get(object_id),
            candidate,
            candidate_objects.get(object_id),
        )
        for object_id in sorted(reference_objects.keys() | candidate_objects.keys())
    )
    return SceneComparison(
        timestamp_s=max(reference.timestamp_s, candidate.timestamp_s),
        reference_valid=reference.valid,
        candidate_valid=candidate.valid,
        objects=comparisons,
    )


def _compare_object(
    object_id: str,
    reference_scene: LocalScene,
    reference: TrackedObject | None,
    candidate_scene: LocalScene,
    candidate: TrackedObject | None,
) -> ObjectComparison:
    if reference is None:
        status = DetectionStatus.EXTRA
    elif candidate is None:
        status = DetectionStatus.MISSED
    else:
        status = DetectionStatus.MATCHED

    reference_distance = _distance(reference)
    candidate_distance = _distance(candidate)
    reference_closing_speed = _closing_speed(reference)
    candidate_closing_speed = _closing_speed(candidate)
    reference_ttc = _ttc(reference_scene, reference)
    candidate_ttc = _ttc(candidate_scene, candidate)
    return ObjectComparison(
        object_id=object_id,
        status=status,
        reference_distance_m=reference_distance,
        candidate_distance_m=candidate_distance,
        distance_error_m=_difference(candidate_distance, reference_distance),
        reference_closing_speed_mps=reference_closing_speed,
        candidate_closing_speed_mps=candidate_closing_speed,
        closing_speed_error_mps=_difference(candidate_closing_speed, reference_closing_speed),
        reference_ttc_s=reference_ttc,
        candidate_ttc_s=candidate_ttc,
        ttc_error_s=_difference(candidate_ttc, reference_ttc),
        reference_in_path=reference.in_path if reference is not None else None,
        candidate_in_path=candidate.in_path if candidate is not None else None,
        in_path_agrees=(
            reference.in_path == candidate.in_path
            if reference is not None and candidate is not None
            else None
        ),
    )


def _distance(tracked: TrackedObject | None) -> float | None:
    if tracked is None:
        return None
    if tracked.path_distance_m is not None:
        return tracked.path_distance_m
    return tracked.relative_position_m[0]


def _closing_speed(tracked: TrackedObject | None) -> float | None:
    if tracked is None:
        return None
    relative_velocity = (
        tracked.path_relative_velocity_mps
        if tracked.path_relative_velocity_mps is not None
        else tracked.relative_velocity_mps[0]
    )
    return max(0.0, -relative_velocity)


def _ttc(scene: LocalScene, tracked: TrackedObject | None) -> float | None:
    distance = _distance(tracked)
    closing_speed = _closing_speed(tracked)
    if tracked is None or distance is None or closing_speed is None or closing_speed <= 1e-6:
        return None
    gap = max(0.0, distance - (scene.ego_length_m + tracked.length_m) / 2.0)
    return gap / closing_speed


def _difference(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None:
        return None
    return candidate - reference
