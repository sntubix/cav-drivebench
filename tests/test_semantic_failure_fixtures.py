import json
from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter.config import (
    EventLogSettings,
    StoppedVehicleSettings,
    config_from_dict,
    load_config,
)
from metadrive_starter.events import read_event_log, to_json_value
from metadrive_starter.hazard_agreement import score_hazards
from metadrive_starter.perception import LaneRelation, LocalScene, TrackedObject
from metadrive_starter.simulation import run_simulation
from metadrive_starter.types import ControlTick
from metadrive_starter.vla import (
    HazardType,
    HighLevelAction,
    RelativeLocation,
    decode_vla_assessment,
    load_provider_fixture,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "fixtures" / "vla" / "providers" / "semantic-failures"
STOPPED_VEHICLE = "stopped-vehicle-keep-lane"
PARTIAL_OCCLUSION = "partial-occlusion-missed-vehicle"


def _assessment(fixture_id: str):
    fixture = load_provider_fixture(FIXTURES / f"{fixture_id}.json")
    assert fixture.response is not None
    return fixture, decode_vla_assessment(fixture.response.text)


@pytest.mark.needs("fixtures")
@pytest.mark.parametrize("fixture_id", [STOPPED_VEHICLE, PARTIAL_OCCLUSION])
def test_semantic_failures_are_unmatched_so_they_replay_on_any_gpu(fixture_id: str) -> None:
    fixture, assessment = _assessment(fixture_id)

    assert fixture.source == "synthetic"
    assert fixture.fixture_id == f"semantic-{fixture_id}"
    assert (fixture.prompt_sha256, fixture.rgb_sha256) == (None, None)
    # Wrong in substance, never in form: each decodes and reaches arbitration.
    assert assessment.proposed_action is HighLevelAction.KEEP_LANE
    assert assessment.confidence >= 0.9


def _vehicle(object_id: str, distance_m: float) -> TrackedObject:
    return TrackedObject(
        object_id=object_id,
        kind="vehicle",
        relative_position_m=(distance_m, 0.0),
        relative_velocity_mps=(-8.0, 0.0),
        length_m=4.5,
        width_m=1.8,
        lane_relation=LaneRelation.SAME,
        in_path=True,
        path_distance_m=distance_m,
        path_relative_velocity_mps=-8.0,
    )


def _scene(*objects: TrackedObject) -> LocalScene:
    return LocalScene(
        timestamp_s=1.0,
        ego_speed_mps=8.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        objects=objects,
    )


@pytest.mark.needs("fixtures")
def test_the_stopped_vehicle_failure_agrees_with_the_scene_and_contradicts_itself() -> None:
    _, assessment = _assessment(STOPPED_VEHICLE)

    agreement = score_hazards(assessment.relevant_hazards, _scene(_vehicle("lead", 30.0)))

    assert agreement.agrees
    [hazard] = assessment.relevant_hazards
    assert (hazard.hazard_type, hazard.relative_location, hazard.risk.value) == (
        HazardType.VEHICLE,
        RelativeLocation.FRONT,
        "high",
    )
    # A high-risk vehicle in front, yet the cruise speed of 35 km/h.
    assert assessment.proposed_target_speed_mps == pytest.approx(9.72)


@pytest.mark.needs("fixtures")
def test_the_partial_occlusion_failure_misses_a_vehicle_in_the_path() -> None:
    _, assessment = _assessment(PARTIAL_OCCLUSION)

    agreement = score_hazards(
        assessment.relevant_hazards,
        _scene(_vehicle("occluding-truck", 20.0), _vehicle("hidden-car", 30.0)),
    )

    assert not agreement.agrees
    assert [hazard.source_id for hazard in agreement.missed] == ["hidden-car"]
    assert not agreement.false_claims


@pytest.mark.parametrize(
    ("scene", "fixture_id"),
    [("stopped-vehicle", STOPPED_VEHICLE), ("partial-occlusion", PARTIAL_OCCLUSION)],
)
def test_each_demo_replays_its_failure(scene: str, fixture_id: str) -> None:
    config = load_config(PROJECT_ROOT / "configs" / f"demo-vla-semantic-{scene}.yaml")

    assert config.vla.provider == "fixture"
    assert config.vla.fixture.repeat_last
    assert config.vla.fixture.path == f"fixtures/vla/providers/semantic-failures/{fixture_id}.json"


def test_stopped_vehicles_parse_and_survive_the_event_log() -> None:
    config = config_from_dict(
        {
            "scenario": {
                "stopped_vehicles": [
                    {"vehicle_id": "truck", "distance_m": 20, "lateral_m": -0.5, "kind": "truck"},
                    {"vehicle_id": "car", "distance_m": 30.0},
                ]
            }
        }
    )

    logged = json.loads(json.dumps(to_json_value(config.to_dict())))

    assert config.scenario.stopped_vehicles == (
        StoppedVehicleSettings("truck", 20.0, lateral_m=-0.5, kind="truck"),
        StoppedVehicleSettings("car", 30.0),
    )
    assert config_from_dict(logged).scenario == config.scenario


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"vehicle_id": "car", "distance_m": 0.0}, "distance_m must be finite and positive"),
        ({"vehicle_id": "car", "distance_m": 10.0, "kind": "bus"}, "kind must be 'car' or 'truck'"),
        ({"vehicle_id": "car", "distance": 10.0}, "unexpected keyword argument 'distance'"),
    ],
)
def test_stopped_vehicles_reject_invalid_entries(entry: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        config_from_dict({"scenario": {"stopped_vehicles": [entry]}})


class _Coasting:
    def update(self, tick: ControlTick) -> tuple[float, float]:
        return (0.0, 0.0)

    def reset_speed_control(self) -> None:
        pass


def test_stopped_vehicles_are_placed_along_the_route(tmp_path: Path) -> None:
    config = config_from_dict(
        {
            "simulator": {"map": "S", "horizon": 2, "headless": True},
            "scenario": {
                "stopped_vehicles": [
                    {"vehicle_id": "truck", "distance_m": 20.0, "kind": "truck"},
                    {"vehicle_id": "car", "distance_m": 30.0, "lateral_m": 1.2},
                ]
            },
        }
    )
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "events.jsonl"),
        scenario_id="stopped-vehicles",
    )

    run_simulation(config, controller_factory=lambda settings: _Coasting())

    spawned = [
        record.payload
        for record in read_event_log(config.event_log.path)
        if record.event_type == "scenario_vehicle_spawned"
    ]
    assert [(item["vehicle_id"], item["distance_m"], item["kind"]) for item in spawned] == [
        ("truck", 20.0, "truck"),
        ("car", 30.0, "car"),
    ]
