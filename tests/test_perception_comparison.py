from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from metadrive_starter.perception import DetectionStatus, DualSceneObserver


class _Lane:
    def local_coordinates(self, position: object) -> tuple[float, float]:
        return float(position[0]), float(position[1])  # type: ignore[index]

    def heading_theta_at(self, longitude: float) -> float:
        del longitude
        return 0.0


class _LidarSensor:
    def __init__(self, detected_objects: list[object]):
        self.detected_objects = detected_objects

    def perceive(self, ego: object, **kwargs: object) -> tuple[list[float], list[object]]:
        del ego
        return [1.0] * int(kwargs["num_lasers"]), self.detected_objects


@dataclass(frozen=True)
class _MatchedScenario:
    name: str
    route: list[tuple[float, float]]
    position: tuple[float, float]
    heading_rad: float
    lane_index: tuple[str, str, int]
    expected_path_distance_m: float
    expected_in_path: bool


MATCHED_SCENARIOS = (
    _MatchedScenario(
        "straight lead vehicle",
        [(0.0, 0.0), (50.0, 0.0)],
        (20.0, 0.0),
        0.0,
        ("a", "b", 0),
        20.0,
        True,
    ),
    _MatchedScenario(
        "vehicle around a curve",
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        (0.0, 10.0),
        math.pi,
        ("c", "d", 0),
        30.0,
        True,
    ),
    _MatchedScenario(
        "adjacent-lane vehicle",
        [(0.0, 0.0), (50.0, 0.0)],
        (15.0, 3.5),
        0.0,
        ("a", "b", 1),
        15.0,
        False,
    ),
    _MatchedScenario(
        "crossing vehicle",
        [(0.0, 0.0), (50.0, 0.0)],
        (10.0, 0.0),
        math.pi / 2.0,
        ("c", "d", 0),
        10.0,
        True,
    ),
)


def _vehicle(
    object_id: str,
    *,
    position: tuple[float, float],
    velocity: tuple[float, float] = (0.0, 0.0),
    heading_rad: float = 0.0,
    lane_index: tuple[str, str, int] = ("a", "b", 0),
) -> SimpleNamespace:
    return SimpleNamespace(
        id=object_id,
        position=position,
        velocity=velocity,
        heading_theta=heading_rad,
        lane_index=lane_index,
        LENGTH=4.5,
        WIDTH=1.8,
        metadrive_type="VEHICLE",
    )


def _environment(
    truth_objects: list[object],
    lidar_objects: list[object],
    *,
    num_lasers: int = 120,
) -> SimpleNamespace:
    ego = _vehicle("ego", position=(0.0, 0.0), velocity=(5.0, 0.0))
    ego.speed = 5.0
    ego.lane = _Lane()
    ego.config = {
        "lidar": {"num_lasers": num_lasers, "distance": 50.0},
        "show_lidar": False,
    }
    sensor = _LidarSensor(lidar_objects)
    objects = {"ego": ego, **{item.id: item for item in truth_objects}}
    engine = SimpleNamespace(
        get_objects=lambda: objects,
        get_sensor=lambda name: sensor if name == "lidar" else None,
        physics_world=SimpleNamespace(dynamic_world=object()),
    )
    return SimpleNamespace(agent=ego, engine=engine)


@pytest.mark.parametrize("scenario", MATCHED_SCENARIOS, ids=lambda item: item.name)
def test_adapters_compare_on_deterministic_geometry(scenario: _MatchedScenario) -> None:
    tracked = _vehicle(
        "target",
        position=scenario.position,
        heading_rad=scenario.heading_rad,
        lane_index=scenario.lane_index,
    )
    observer = DualSceneObserver.for_route(scenario.route)

    observation = observer.observe(
        _environment([tracked], [tracked]),
        timestamp_s=4.0,
    )

    comparison = observation.comparison.objects[0]
    assert comparison.status is DetectionStatus.MATCHED
    assert comparison.reference_distance_m == pytest.approx(scenario.expected_path_distance_m)
    assert comparison.distance_error_m == pytest.approx(0.0)
    assert comparison.closing_speed_error_mps == pytest.approx(0.0)
    assert comparison.ttc_error_s == pytest.approx(0.0)
    assert comparison.reference_in_path is scenario.expected_in_path
    assert comparison.in_path_agrees is True


def test_comparison_records_lidar_object_dropout_as_missed() -> None:
    lead = _vehicle("lead", position=(20.0, 0.0))
    observer = DualSceneObserver.for_route([(0.0, 0.0), (50.0, 0.0)])

    comparison = observer.observe(
        _environment([lead], []),
        timestamp_s=5.0,
    ).comparison

    assert comparison.missed_ids == ("lead",)
    assert comparison.extra_ids == ()
    assert comparison.objects[0].status is DetectionStatus.MISSED


def test_comparison_records_complete_sensor_dropout_as_invalid_and_missed() -> None:
    lead = _vehicle("lead", position=(20.0, 0.0))
    observer = DualSceneObserver.for_route([(0.0, 0.0), (50.0, 0.0)])

    comparison = observer.observe(
        _environment([lead], [], num_lasers=0),
        timestamp_s=5.5,
    ).comparison

    assert comparison.reference_valid is True
    assert comparison.candidate_valid is False
    assert comparison.missed_ids == ("lead",)


def test_comparison_records_noisy_detection_errors_without_judging_quality() -> None:
    truth = _vehicle("lead", position=(20.0, 0.0))
    noisy_detection = _vehicle("lead", position=(18.5, 0.0), velocity=(0.5, 0.0))
    observer = DualSceneObserver.for_route([(0.0, 0.0), (50.0, 0.0)])

    comparison = observer.observe(
        _environment([truth], [noisy_detection]),
        timestamp_s=6.0,
    ).comparison.objects[0]

    assert comparison.status is DetectionStatus.MATCHED
    assert comparison.distance_error_m == pytest.approx(-1.5)
    assert comparison.closing_speed_error_mps == pytest.approx(-0.5)
    assert comparison.ttc_error_s is not None
    assert comparison.in_path_agrees is True


def test_comparison_records_noisy_in_path_disagreement() -> None:
    truth = _vehicle("adjacent", position=(20.0, 3.5), lane_index=("a", "b", 1))
    noisy_detection = _vehicle("adjacent", position=(20.0, 2.0), lane_index=("a", "b", 1))
    observer = DualSceneObserver.for_route([(0.0, 0.0), (50.0, 0.0)])

    comparison = observer.observe(
        _environment([truth], [noisy_detection]),
        timestamp_s=6.5,
    ).comparison

    assert comparison.in_path_disagreement_ids == ("adjacent",)
    assert comparison.objects[0].reference_in_path is False
    assert comparison.objects[0].candidate_in_path is True


def test_comparison_records_spurious_lidar_object_as_extra() -> None:
    ghost = _vehicle("ghost", position=(20.0, 0.0))
    observer = DualSceneObserver.for_route([(0.0, 0.0), (50.0, 0.0)])

    comparison = observer.observe(
        _environment([], [ghost]),
        timestamp_s=7.0,
    ).comparison

    assert comparison.missed_ids == ()
    assert comparison.extra_ids == ("ghost",)
    assert comparison.objects[0].status is DetectionStatus.EXTRA
