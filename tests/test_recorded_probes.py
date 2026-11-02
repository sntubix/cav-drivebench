from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter.config import load_config
from metadrive_starter.hazard_agreement import scene_hazards, score_hazards
from metadrive_starter.recorded_probes import (
    INTERPRETATION_CATALOG,
    LOCAL_SCENE_FILE,
    RECORDED_CATALOG,
    load_recorded_probes,
    observation_request,
    record_probes,
    recorded_capture,
)
from metadrive_starter.replay import scene_from_payload
from metadrive_starter.submission import PROJECT_ROOT
from metadrive_starter.vla.assessment import RiskLevel, VLAHazard
from metadrive_starter.vla.prompting import DefaultObservationBuilder
from metadrive_starter.vla_probe import ProbeCapture, load_probe_scenarios

# The recorded catalog is under fixtures/.
pytestmark = pytest.mark.needs("fixtures")


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
INTERPRETATION_CONFIG = PROJECT_ROOT / "configs" / "vla-probe-qwen3-vl-2b-llama-interpretation.yaml"
PROBE_MANIFEST = PROJECT_ROOT / "configs" / "vla-probe-scenarios.yaml"


def test_recorded_catalog_holds_every_public_probe_scene_as_the_manifest_defines_it() -> None:
    recorded = {probe.scenario_id: probe.scenario for probe in load_recorded_probes()}

    assert recorded == {
        scenario.scenario_id: scenario for scenario in load_probe_scenarios(PROBE_MANIFEST)
    }


def test_recorded_scenes_rebuild_the_observation_the_probe_saved() -> None:
    config = load_config(DEFAULT_CONFIG)

    for probe in load_recorded_probes():
        observation = DefaultObservationBuilder().build(observation_request(probe, config))

        assert observation.prompt == probe.prompt, probe.scenario_id
        assert observation.frame == probe.frame


def test_recorded_scenes_carry_what_the_notebooks_teach_with() -> None:
    probes = {probe.scenario_id: probe for probe in load_recorded_probes()}

    assert [tracked.in_path for tracked in probes["partial-occlusion"].scene.objects] == [
        True,
        True,
    ]
    assert {light.state.value for light in probes["red-light-with-traffic"].scene.traffic_lights} == {
        "red"
    }
    assert probes["clear-straight"].scene.objects == ()
    # Only the lead is in the ego path; the lanes to its right are occupied.
    assert [tracked.in_path for tracked in probes["all-lanes-blocked"].scene.objects] == [
        True,
        False,
        False,
    ]


def _truth(probe) -> tuple[list[dict], object]:
    truth = json.loads((probe.source_dir / "truth.json").read_text())
    scene = json.loads((probe.source_dir / "truth_scene.json").read_text())
    return truth["objects"], scene_from_payload(scene)


def test_public_interpretation_scenes_hold_what_the_probe_cannot_place() -> None:
    probes = load_recorded_probes(INTERPRETATION_CATALOG)

    # Half of scene interpretation's objects are pedestrians, cones, and barriers.
    assert 6 <= len(probes) <= 8
    placed = [item for probe in probes for item in _truth(probe)[0]]
    assert {item["spec"]["kind"] for item in placed} == {"pedestrian", "cone", "barrier", "car"}
    assert {item["spec"]["lane_offset"] for item in placed} == {-1, 0, 1}
    assert len({item["distance_m"] for item in placed}) >= 6
    for kind in ("pedestrian", "cone", "barrier"):
        assert len({item["spec"]["lane_offset"] for item in placed if item["spec"]["kind"] == kind}) > 1


def test_public_interpretation_scenes_carry_truth_score_hazards_can_score_against() -> None:
    for probe in load_recorded_probes(INTERPRETATION_CATALOG):
        objects, truth = _truth(probe)

        assert objects and all(item["in_view"] and item["measured"] for item in objects)
        assert {item["object_id"] for item in objects} == {
            tracked.object_id for tracked in truth.objects
        }
        # Claiming each object where its truth locates it finds them all.
        claims = [
            VLAHazard(hazard.hazard_type, hazard.relative_location, RiskLevel.MEDIUM)
            for hazard in scene_hazards(truth)
        ]
        scored = score_hazards(claims, truth)
        assert (len(scored.matched), scored.false_claims) == (len(objects), ()), probe.scenario_id


def test_public_interpretation_scenes_were_saved_as_the_interpretation_profile_sends_them() -> None:
    config = load_config(INTERPRETATION_CONFIG)

    for probe in load_recorded_probes(INTERPRETATION_CATALOG):
        observation = DefaultObservationBuilder().build(observation_request(probe, config))

        assert observation.prompt == probe.prompt, probe.scenario_id
        # Replayed bare, a scene still tells the model no object and no clearance.
        assert '"observed_clear"' not in probe.prompt
        assert '"objects"' not in probe.prompt


def test_a_recorded_capture_hands_back_the_scene_as_it_was_measured() -> None:
    probes = load_recorded_probes(INTERPRETATION_CATALOG)
    capture = recorded_capture(probes)

    captured = capture(load_config(INTERPRETATION_CONFIG), probes[0].scenario)

    assert captured.frame == probes[0].frame
    assert captured.scene == probes[0].scene
    assert captured.ego_speed_mps == probes[0].ego_speed_mps


def test_loading_refuses_a_scene_without_its_measured_local_scene(tmp_path: Path) -> None:
    shutil.copytree(RECORDED_CATALOG / "stopped-vehicle", tmp_path / "stopped-vehicle")
    (tmp_path / "stopped-vehicle" / LOCAL_SCENE_FILE).unlink()

    with pytest.raises(FileNotFoundError, match=LOCAL_SCENE_FILE):
        load_recorded_probes(tmp_path)


def test_recording_keeps_each_measured_scene_beside_the_probe_output(tmp_path: Path) -> None:
    source = load_recorded_probes(scenario_ids=["partial-occlusion"])[0]

    def capture(_config, _scenario) -> ProbeCapture:
        return ProbeCapture(
            frame=source.frame,
            timestamp_s=source.frame.timestamp_s,
            ego_speed_mps=source.ego_speed_mps,
            scene=source.scene,
        )

    record_probes(
        load_config(DEFAULT_CONFIG),
        load_probe_scenarios(PROBE_MANIFEST),
        tmp_path / "recorded",
        scenario_ids=["partial-occlusion"],
        capture_scenario=capture,
    )

    (recorded,) = load_recorded_probes(tmp_path / "recorded")
    assert recorded.scene == source.scene
    assert recorded.prompt == source.prompt


def test_recording_refuses_a_capture_that_measured_no_scene(tmp_path: Path) -> None:
    source = load_recorded_probes(scenario_ids=["clear-straight"])[0]

    def capture(_config, _scenario) -> ProbeCapture:
        return ProbeCapture(frame=source.frame, timestamp_s=source.frame.timestamp_s)

    with pytest.raises(RuntimeError, match="clear-straight"):
        record_probes(
            load_config(DEFAULT_CONFIG),
            load_probe_scenarios(PROBE_MANIFEST),
            tmp_path / "recorded",
            scenario_ids=["clear-straight"],
            capture_scenario=capture,
        )


def test_a_request_follows_the_team_configuration() -> None:
    probe = load_recorded_probes(scenario_ids=["stopped-vehicle"])[0]
    config = load_config(DEFAULT_CONFIG)
    config = replace(
        config,
        vla=replace(config.vla, prompt_policy="Stop for anything.", max_scene_objects=2),
        controller=replace(config.controller, target_speed_mps=7.0),
    )

    request = observation_request(probe, config)

    assert request.prompt_policy == "Stop for anything."
    assert request.max_scene_objects == 2
    assert request.cruise_speed_mps == 7.0
    assert request.model_id == config.vla.requested_model_id
    assert request.scene == probe.scene
    assert request.frame.timestamp_s == request.now_s


def test_a_request_withholds_what_the_configuration_withholds() -> None:
    probe = load_recorded_probes(scenario_ids=["stopped-vehicle"])[0]
    config = load_config(PROJECT_ROOT / "configs" / "vla-probe-qwen3-vl-2b-llama-structured.yaml")

    request = observation_request(probe, config)

    assert probe.scene.objects
    assert request.withheld == {"objects", "traffic_controls"}
    assert request.scene is not None and request.scene.objects == ()
    assert request.scene.ego_speed_mps == probe.scene.ego_speed_mps
