from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter.config import load_config
from metadrive_starter.recorded_probes import (
    LOCAL_SCENE_FILE,
    RECORDED_CATALOG,
    load_recorded_probes,
    observation_request,
    record_probes,
)
from metadrive_starter.submission import PROJECT_ROOT
from metadrive_starter.vla.prompting import DefaultObservationBuilder
from metadrive_starter.vla_probe import ProbeCapture, load_probe_scenarios

# The recorded catalog is under fixtures/.
pytestmark = pytest.mark.needs("fixtures")


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
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
    assert request.scene == probe.scene
    assert request.frame.timestamp_s == request.now_s
