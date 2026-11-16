"""Static cones and barriers placed along the ego route, as live scenes place vehicles."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter.config import ScenarioObjectSettings, config_from_dict, load_config
from metadrive_starter.env import _route_position_ahead, make_env, spawn_static_object_ahead
from metadrive_starter.events import to_json_value
from metadrive_starter.perception import LidarSceneAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scenario_objects_parse_and_survive_the_event_log() -> None:
    config = config_from_dict(
        {
            "scenario": {
                "objects": [
                    {"object_id": "cone-1", "distance_m": 20, "lateral_m": -0.5},
                    {"object_id": "wall", "distance_m": 30.0, "kind": "barrier", "lane_offset": 1},
                ]
            }
        }
    )

    logged = json.loads(json.dumps(to_json_value(config.to_dict())))

    assert config.scenario.objects == (
        ScenarioObjectSettings("cone-1", 20.0, lateral_m=-0.5),
        ScenarioObjectSettings("wall", 30.0, lane_offset=1, kind="barrier"),
    )
    assert config_from_dict(logged).scenario == config.scenario


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"object_id": "cone", "distance_m": 0.0}, "distance_m must be finite and positive"),
        ({"object_id": "cone", "distance_m": 10.0, "kind": "pedestrian"}, "kind must be 'cone' or 'barrier'"),
        ({"object_id": "cone", "distance_m": 10.0, "lane_offset": 1.5}, "lane_offset must be an integer"),
        ({"object_id": "a cone", "distance_m": 10.0}, "object_id must contain only"),
    ],
)
def test_scenario_objects_reject_what_cannot_be_placed(entry: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_dict({"scenario": {"objects": [entry]}})


def test_scenario_object_ids_are_unique_and_entries_are_mappings() -> None:
    with pytest.raises(ValueError, match="object ids must be unique"):
        config_from_dict(
            {
                "scenario": {
                    "objects": [
                        {"object_id": "cone", "distance_m": 10.0},
                        {"object_id": "cone", "distance_m": 20.0},
                    ]
                }
            }
        )
    with pytest.raises(ValueError, match="objects must be a list of mappings"):
        config_from_dict({"scenario": {"objects": ["cone"]}})


def test_placed_cone_and_barrier_stand_where_placed_and_reach_the_measured_scene() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    env = make_env(
        replace(config.simulator, map="S", headless=True),
        replace(config.camera, enabled=False),
    )
    try:
        env.reset()
        lane, longitudinal = _route_position_ahead(env, 20.0)
        cone_placed = lane.position(longitudinal, 0.0)
        cone = spawn_static_object_ahead(env, 20.0, kind="cone", name="scenario-cone")
        barrier = spawn_static_object_ahead(
            env, 30.0, lane_offset=1, lateral_m=0.3, kind="barrier", name="scenario-wall"
        )
        with pytest.raises(ValueError, match="kind must be"):
            spawn_static_object_ahead(env, 10.0, kind="pedestrian")
        for _ in range(10):
            env.step((0.0, 0.0))

        assert cone.position == pytest.approx(tuple(cone_placed), abs=0.05)
        assert tuple(cone.velocity) == pytest.approx((0.0, 0.0))
        assert tuple(barrier.velocity) == pytest.approx((0.0, 0.0))

        scene = LidarSceneAdapter(detection_radius_m=50.0).observe(env, timestamp_s=1.0)
        kinds = {item.object_id: item.kind for item in scene.objects}
        assert any("cone" in kind for kind in kinds.values()), kinds
        assert any("barrier" in kind for kind in kinds.values()), kinds
    finally:
        env.close()
