"""Assignment 3's practice scenes: public drives on Vertex at the hidden drives' terms."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from metadrive_starter.config import load_config
from metadrive_starter.submission import PROJECT_ROOT

PRACTICE = PROJECT_ROOT / "configs" / "practice-assignment-3"
SCENES = sorted(PRACTICE.glob("*.yaml"))
# A faulted drive is its scene again, with one fault.
FAULTED = {
    "slow-truck-outage.yaml": ("slow-truck.yaml", "provider_outage"),
    "road-closed-quota.yaml": ("road-closed.yaml", "provider_quota"),
}


def test_the_set_is_ten_scenes_and_two_faulted_drives() -> None:
    names = [path.name for path in SCENES]

    assert len(names) == 12
    assert sorted(set(names) & set(FAULTED)) == sorted(FAULTED)


@pytest.mark.parametrize("path", SCENES, ids=lambda path: path.stem)
def test_a_practice_scene_runs_at_the_hidden_drives_terms(path: Path) -> None:
    config = load_config(path)

    assert config.vla.enabled and config.vla.provider == "vertex"
    # What the brief's "Timing" says of the hidden drives.
    assert config.vla.action_horizon_s == 4.0
    assert config.command_validation.maximum_command_age_s == 2.5
    assert config.vla.minimum_interval_s == 0.5
    assert config.vla.request_timeout_s == 6.0
    assert config.simulator.realtime and config.simulator.realtime_factor == 1.0
    # The model finds the objects in the image, as in grading (ADR-0008).
    assert config.vla.request_scene_fields.withheld == frozenset({"objects", "traffic_controls"})
    assert config.vla.request_scene_fields.lanes and config.vla.request_scene_fields.lane_clearance
    assert config.vla.maximum_requests_per_run <= 200


@pytest.mark.parametrize("path", SCENES, ids=lambda path: path.stem)
def test_a_practice_scene_names_the_command_that_drives_it(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    assert f"--config configs/practice-assignment-3/{path.name}" in text
    assert "--submission submission" in text


@pytest.mark.parametrize("name", sorted(FAULTED))
def test_a_faulted_drive_is_its_scene_with_one_fault(name: str) -> None:
    base_name, kind = FAULTED[name]
    faulted, base = load_config(PRACTICE / name), load_config(PRACTICE / base_name)

    assert [fault.kind for fault in faulted.faults] == [kind]
    assert base.faults == ()
    assert replace(faulted, faults=(), event_log=base.event_log) == base


@pytest.mark.instructor
def test_no_practice_scene_drives_a_hidden_scenes_road() -> None:
    hidden = set()
    for path in (PROJECT_ROOT / "course" / "instructor").glob("live-scene-*.yaml"):
        simulator = yaml.safe_load(path.read_text(encoding="utf-8"))["simulator"]
        hidden.add((simulator["map"], simulator["start_seed"]))

    assert hidden
    for path in SCENES:
        simulator = load_config(path).simulator
        assert (simulator.map, simulator.start_seed) not in hidden, path.name
