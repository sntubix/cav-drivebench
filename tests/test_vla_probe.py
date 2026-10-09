from __future__ import annotations

import json
from pathlib import Path

import pytest

from metadrive_starter import vla_probe
from metadrive_starter.cli import main
from metadrive_starter.config import config_from_dict
from metadrive_starter.perception import LaneRelation, LocalScene, TrackedObject
from metadrive_starter.vla import (
    HazardType,
    HighLevelAction,
    Observation,
    ObservationRequest,
    RGBFrame,
    RelativeLocation,
    ScriptedModelProvider,
    VLA_PROMPT_CONTRACT_VERSION,
    encode_rgb_frame_png,
)
from metadrive_starter.vla_probe import (
    ProbeCapture,
    ProbeEvaluationSpec,
    ProbeHazardExpectation,
    ProbeScenario,
    load_probe_scenarios,
    replay_vla_probe_artifacts,
    run_vla_probe_catalog,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _scenario(
    scenario_id: str = "clear-road",
    evaluation: ProbeEvaluationSpec | None = None,
) -> ProbeScenario:
    return ProbeScenario(
        scenario_id=scenario_id,
        description="Clear deterministic road.",
        expected_visual="An unobstructed road is visible.",
        map_name="S",
        evaluation=evaluation,
    )


def _capture(_config, _scenario) -> ProbeCapture:
    frame = RGBFrame(
        timestamp_s=1.0,
        width=2,
        height=1,
        rgb_bytes=b"\xff\x00\x00\x00\xff\x00",
    )
    scene = LocalScene(
        timestamp_s=1.0,
        ego_speed_mps=0.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        objects=(
            TrackedObject(
                object_id="lead",
                kind="vehicle",
                relative_position_m=(20.0, 0.0),
                relative_velocity_mps=(0.0, 0.0),
                length_m=4.5,
                width_m=1.8,
                lane_relation=LaneRelation.SAME,
                in_path=True,
                path_distance_m=20.0,
                path_relative_velocity_mps=0.0,
            ),
        ),
        left_lane_available=False,
        right_lane_available=True,
    )
    return ProbeCapture(frame=frame, timestamp_s=1.0, scene=scene)


def test_checked_in_probe_manifest_defines_fixed_visual_catalog() -> None:
    scenarios = load_probe_scenarios(PROJECT_ROOT / "configs/vla-probe-scenarios.yaml")

    assert [scenario.scenario_id for scenario in scenarios] == [
        "clear-straight",
        "stopped-vehicle",
        "curved-road",
        "intersection",
        "red-light-with-traffic",
        "adjacent-traffic",
        "partial-occlusion",
        "all-lanes-blocked",
    ]
    assert all(
        scenario.traffic_density == 0.0
        for scenario in scenarios
        if scenario.scenario_id != "red-light-with-traffic"
    )
    assert all(scenario.evaluation is not None for scenario in scenarios)
    assert scenarios[0].evaluation is not None
    assert scenarios[0].evaluation.acceptable_actions == (HighLevelAction.KEEP_LANE,)
    adjacent = next(scenario for scenario in scenarios if scenario.scenario_id == "adjacent-traffic")
    assert adjacent.vehicles[0].lane_offset == 1
    occlusion = next(
        scenario for scenario in scenarios if scenario.scenario_id == "partial-occlusion"
    )
    assert [vehicle.kind for vehicle in occlusion.vehicles] == ["truck", "car"]
    signal = next(
        scenario for scenario in scenarios if scenario.scenario_id == "red-light-with-traffic"
    )
    assert signal.traffic_density == 0.15
    assert signal.traffic_light is not None
    assert signal.traffic_light.state.value == "red"


def test_probe_catalog_never_accepts_slowing_down_for_a_stopped_obstacle() -> None:
    scenarios = {
        scenario.scenario_id: scenario
        for scenario in load_probe_scenarios(PROJECT_ROOT / "configs/vla-probe-scenarios.yaml")
    }

    # A clear adjacent lane: change into it.
    for scenario_id in ("stopped-vehicle", "partial-occlusion"):
        evaluation = scenarios[scenario_id].evaluation
        assert evaluation is not None
        assert evaluation.acceptable_actions == (HighLevelAction.CHANGE_LANE_RIGHT,)
    # Every lane blocked: wait behind the lead, by stopping or following it.
    blocked = scenarios["all-lanes-blocked"]
    assert [vehicle.lane_offset for vehicle in blocked.vehicles] == [0, 1, 2]
    assert blocked.evaluation is not None
    assert blocked.evaluation.acceptable_actions == (
        HighLevelAction.FOLLOW,
        HighLevelAction.STOP,
    )
    signal = scenarios["red-light-with-traffic"].evaluation
    assert signal is not None
    assert signal.acceptable_actions == (HighLevelAction.STOP,)


def test_lane_change_probe_manifest_defines_clear_and_blocked_right_choices() -> None:
    scenarios = load_probe_scenarios(
        PROJECT_ROOT / "configs" / "vla-probe-lane-change-scenarios.yaml"
    )

    assert [scenario.scenario_id for scenario in scenarios] == [
        "lane-change-clear-right",
        "lane-change-right-blocked",
    ]
    clear, blocked = scenarios
    assert [vehicle.lane_offset for vehicle in clear.vehicles] == [0]
    assert clear.evaluation is not None
    assert clear.evaluation.acceptable_actions == (
        HighLevelAction.CHANGE_LANE_RIGHT,
    )
    assert clear.evaluation.required_hazards == (
        ProbeHazardExpectation(HazardType.VEHICLE, RelativeLocation.FRONT),
    )
    assert clear.evaluation.raw_speed_required is False
    assert [vehicle.lane_offset for vehicle in blocked.vehicles] == [0, 1]
    assert blocked.evaluation is not None
    assert blocked.evaluation.acceptable_actions == (
        HighLevelAction.FOLLOW,
        HighLevelAction.STOP,
    )


def test_capture_only_probe_writes_exact_frame_prompt_and_metadata(tmp_path: Path) -> None:
    config = config_from_dict(
        {
            "vla": {
                "action_horizon_s": 1.5,
                "prompt_policy": "Prefer safe forward progress.",
            }
        }
    )
    scenario = _scenario()

    summary = run_vla_probe_catalog(
        config,
        [scenario],
        tmp_path / "probe",
        capture=_capture,
    )

    artifact_dir = tmp_path / "probe" / scenario.scenario_id
    expected_frame = _capture(config, scenario).frame
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    saved_summary = json.loads((tmp_path / "probe" / "summary.json").read_text())
    assert summary.successful is True
    assert saved_summary == summary.to_dict()
    assert summary.scenarios[0].status == "captured"
    assert (artifact_dir / "frame.png").read_bytes() == encode_rgb_frame_png(expected_frame)
    assert "Return exactly one JSON object" in (artifact_dir / "prompt.txt").read_text()
    assert "CONFIGURED_POLICY:\nPrefer safe forward progress." in (
        artifact_dir / "prompt.txt"
    ).read_text()
    assert metadata["mode"] == "capture_only"
    assert metadata["outcome"] == {"status": "captured"}
    assert metadata["schema_version"] == 4
    assert metadata["benchmark_version"] == 2
    assert metadata["capture"]["observation_builder"] == (
        "metadrive_starter.vla.prompting.DefaultObservationBuilder"
    )
    assert metadata["capture"]["scene_context_included"] is True
    assert metadata["capture"]["scene_file"] == "scene.json"
    assert len(metadata["capture"]["scene_sha256"]) == 64
    assert metadata["capture"]["ego_speed_mps"] == 0.0
    assert metadata["capture"]["cruise_speed_mps"] == pytest.approx(35.0 / 3.6)
    assert len(metadata["capture"]["prompt_sha256"]) == 64
    assert metadata["scenario"]["id"] == scenario.scenario_id
    scene = json.loads((artifact_dir / "scene.json").read_text())
    assert scene["lanes"]["right"]["observed_clear"] is True
    assert scene["objects"][0]["id"] == "lead"
    assert scene["objects"][0]["relative_location"] == "front"
    assert json.dumps(scene, sort_keys=True, separators=(",", ":")) in (
        artifact_dir / "prompt.txt"
    ).read_text()


def test_inference_probe_records_raw_and_parsed_model_output(tmp_path: Path) -> None:
    response = json.dumps(
        {
            "scene_summary": "Stopped vehicle ahead.",
            "relevant_hazards": [
                {"type": "vehicle", "relative_location": "front", "risk": "high"}
            ],
            "meta_action": "STOP",
            "target_speed_mps": 0.0,
            "confidence": 0.9,
            "brief_justification": "Stop for the vehicle.",
        }
    )
    provider = ScriptedModelProvider([response], model_id="probe-model")

    summary = run_vla_probe_catalog(
        config_from_dict({}),
        [_scenario()],
        tmp_path / "probe",
        infer=True,
        provider=provider,
        capture=_capture,
    )

    artifact_dir = tmp_path / "probe" / "clear-road"
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    assert summary.successful is True
    assert summary.scenarios[0].status == "success"
    assert (artifact_dir / "raw_response.txt").read_text() == response
    assert metadata["outcome"]["model_id"] == "probe-model"
    assert metadata["request"]["prompt_contract_version"] == VLA_PROMPT_CONTRACT_VERSION
    assert metadata["outcome"]["prompt_contract_version"] == VLA_PROMPT_CONTRACT_VERSION
    assert metadata["outcome"]["provider_metadata"]["provider"] == "scripted"
    assert metadata["outcome"]["parsed_command"]["action"] == "STOP"
    assert provider.requests[0].prompt == (artifact_dir / "prompt.txt").read_text()
    assert provider.requests[0].frame.rgb_bytes == _capture(None, None).frame.rgb_bytes


def test_inference_probe_keeps_raw_response_when_command_decoding_fails(tmp_path: Path) -> None:
    provider = ScriptedModelProvider(["not JSON"], model_id="noncompliant-model")

    summary = run_vla_probe_catalog(
        config_from_dict({}),
        [_scenario()],
        tmp_path / "probe",
        infer=True,
        provider=provider,
        capture=_capture,
    )

    artifact_dir = tmp_path / "probe" / "clear-road"
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    assert summary.successful is False
    assert summary.scenarios[0].status == "failure"
    assert (artifact_dir / "raw_response.txt").read_text() == "not JSON"
    assert metadata["outcome"]["error_category"] == "VLAResponseFormatError"
    assert metadata["outcome"]["model_id"] == "noncompliant-model"
    assert metadata["outcome"]["latency_s"] == 0.0


@pytest.mark.parametrize(
    ("target_speed_mps", "expected"),
    [(8.333333333333334, True), (0.0, False)],
)
def test_inference_probe_reports_semantic_rubric_separately(
    tmp_path: Path,
    target_speed_mps: float,
    expected: bool,
) -> None:
    evaluation = ProbeEvaluationSpec(
        acceptable_actions=(HighLevelAction.KEEP_LANE,),
        minimum_target_speed_mps=6.944444444444445,
        maximum_target_speed_mps=9.722222222222221,
    )
    response = json.dumps(
        {
            "scene_summary": "Clear road.",
            "relevant_hazards": [],
            "meta_action": "KEEP_LANE",
            "target_speed_mps": target_speed_mps,
            "confidence": 0.9,
            "brief_justification": "Continue safely.",
        }
    )

    summary = run_vla_probe_catalog(
        config_from_dict({}),
        [_scenario(evaluation=evaluation)],
        tmp_path / "probe",
        infer=True,
        provider=ScriptedModelProvider([response]),
        capture=_capture,
    )

    metadata = json.loads(
        (tmp_path / "probe" / "clear-road" / "metadata.json").read_text()
    )
    semantic = metadata["outcome"]["semantic_evaluation"]
    assert summary.successful is True
    assert summary.semantically_successful is expected
    assert summary.scenarios[0].semantic_passed is expected
    assert semantic["passed"] is expected
    assert semantic["action_passed"] is True
    assert semantic["target_speed_passed"] is expected


def test_v2_rubric_separates_perception_action_and_raw_speed(
    tmp_path: Path,
) -> None:
    evaluation = ProbeEvaluationSpec(
        acceptable_actions=(HighLevelAction.CHANGE_LANE_RIGHT,),
        minimum_target_speed_mps=2.0,
        maximum_target_speed_mps=5.0,
        required_hazards=(
            ProbeHazardExpectation(HazardType.VEHICLE, RelativeLocation.FRONT),
        ),
        raw_speed_required=False,
    )
    response = json.dumps(
        {
            "scene_summary": "A stationary lead blocks the current lane.",
            "relevant_hazards": [
                {"type": "vehicle", "relative_location": "front", "risk": "medium"}
            ],
            "meta_action": "CHANGE_LANE_RIGHT",
            "target_speed_mps": 9.0,
            "confidence": 0.9,
            "brief_justification": "The right lane is clear.",
        }
    )

    summary = run_vla_probe_catalog(
        config_from_dict({}),
        [_scenario(evaluation=evaluation)],
        tmp_path / "probe",
        infer=True,
        provider=ScriptedModelProvider([response]),
        capture=_capture,
    )

    result = summary.scenarios[0]
    metadata = json.loads(
        (tmp_path / "probe" / "clear-road" / "metadata.json").read_text()
    )
    semantic = metadata["outcome"]["semantic_evaluation"]
    assert result.semantic_passed is True
    assert result.perception_passed is True
    assert result.tactical_action_passed is True
    assert result.raw_speed_passed is False
    assert summary.perception_successful is True
    assert summary.tactically_successful is True
    assert summary.raw_speed_successful is False
    assert semantic["passed"] is True
    assert semantic["perception_passed"] is True
    assert semantic["tactical_action_passed"] is True
    assert semantic["raw_speed_passed"] is False
    assert semantic["raw_speed_required"] is False


def test_v2_required_hazard_artifact_replays_with_canonical_type_field(
    tmp_path: Path,
) -> None:
    evaluation = ProbeEvaluationSpec(
        acceptable_actions=(HighLevelAction.CHANGE_LANE_RIGHT,),
        minimum_target_speed_mps=2.0,
        maximum_target_speed_mps=5.0,
        required_hazards=(
            ProbeHazardExpectation(HazardType.VEHICLE, RelativeLocation.FRONT),
        ),
        raw_speed_required=False,
    )
    response = json.dumps(
        {
            "scene_summary": "A stationary lead blocks the current lane.",
            "relevant_hazards": [
                {"type": "vehicle", "relative_location": "front", "risk": "high"}
            ],
            "meta_action": "CHANGE_LANE_RIGHT",
            "target_speed_mps": 3.0,
            "confidence": 0.9,
            "brief_justification": "The right lane is clear.",
        }
    )
    config = config_from_dict({})
    source = tmp_path / "source"
    run_vla_probe_catalog(
        config,
        [_scenario(evaluation=evaluation)],
        source,
        capture=_capture,
    )

    metadata = json.loads((source / "clear-road" / "metadata.json").read_text())
    saved_hazard = metadata["scenario"]["evaluation"]["required_hazards"][0]
    assert saved_hazard == {"type": "vehicle", "relative_location": "front"}

    provider = ScriptedModelProvider([response])
    summary = replay_vla_probe_artifacts(
        config,
        source,
        tmp_path / "replay",
        provider=provider,
    )

    assert summary.successful is True
    assert summary.semantically_successful is True
    assert len(provider.requests) == 1


def test_probe_preflight_rejects_existing_artifact_directory(tmp_path: Path) -> None:
    existing = tmp_path / "probe" / "clear-road"
    existing.mkdir(parents=True)
    called = False

    def capture(_config, _scenario):
        nonlocal called
        called = True
        return _capture(_config, _scenario)

    with pytest.raises(FileExistsError, match="already exists"):
        run_vla_probe_catalog(
            config_from_dict({}),
            [_scenario()],
            tmp_path / "probe",
            capture=capture,
        )

    assert called is False


def test_probe_rejects_unknown_scenario_selection_before_capture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown probe scenario"):
        run_vla_probe_catalog(
            config_from_dict({}),
            [_scenario()],
            tmp_path / "probe",
            scenario_ids=["missing"],
            capture=_capture,
        )


def test_probe_manifest_rejects_path_traversal_id(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
version: 1
scenarios:
  - id: ../escape
    description: Invalid id.
    expected_visual: Nothing.
    map: S
""".strip()
    )

    with pytest.raises(ValueError, match="probe scenario id"):
        load_probe_scenarios(manifest)


def test_probe_replay_reuses_exact_saved_prompt_and_png(tmp_path: Path) -> None:
    capture_config = config_from_dict(
        {
            "vla": {
                "request_timeout_s": 2.0,
                "prompt_policy": "Original artifact policy.",
            }
        }
    )
    source = tmp_path / "source"
    run_vla_probe_catalog(capture_config, [_scenario()], source, capture=_capture)
    replay_config = config_from_dict(
        {
            "vla": {
                "request_timeout_s": 2.0,
                "prompt_policy": "Changed current policy.",
            }
        }
    )
    response = json.dumps(
        {
            "scene_summary": "Clear road.",
            "relevant_hazards": [],
            "meta_action": "KEEP_LANE",
            "target_speed_mps": 30.0,
            "confidence": 0.9,
            "brief_justification": "Continue safely.",
        }
    )
    provider = ScriptedModelProvider([response], model_id="replay-model")

    summary = replay_vla_probe_artifacts(
        replay_config,
        source,
        tmp_path / "replay",
        provider=provider,
    )

    source_artifact = source / "clear-road"
    replay_artifact = tmp_path / "replay" / "clear-road"
    request = provider.requests[0]
    assert summary.successful is True
    assert request.prompt == (source_artifact / "prompt.txt").read_text()
    assert "Original artifact policy." in request.prompt
    assert "Changed current policy." not in request.prompt
    assert request.encoded_png_bytes == (source_artifact / "frame.png").read_bytes()
    assert (replay_artifact / "frame.png").read_bytes() == request.encoded_png_bytes
    assert (replay_artifact / "prompt.txt").read_text() == request.prompt
    metadata = json.loads((replay_artifact / "metadata.json").read_text())
    assert metadata["mode"] == "artifact_replay"
    assert metadata["replay"]["exact_prompt"] is True
    assert metadata["replay"]["exact_png_bytes"] is True
    assert metadata["replay"]["exact_scene_context"] is True
    assert request.prompt_contract_version == VLA_PROMPT_CONTRACT_VERSION


def test_probe_replay_keeps_provider_evidence_when_response_is_malformed(
    tmp_path: Path,
) -> None:
    config = config_from_dict({})
    source = tmp_path / "source"
    run_vla_probe_catalog(config, [_scenario()], source, capture=_capture)
    provider = ScriptedModelProvider(["not JSON"], model_id="malformed-model")

    summary = replay_vla_probe_artifacts(
        config,
        source,
        tmp_path / "replay",
        provider=provider,
    )

    metadata = json.loads(
        (tmp_path / "replay" / "clear-road" / "metadata.json").read_text()
    )
    outcome = metadata["outcome"]
    assert summary.successful is False
    assert outcome["error_category"] == "VLAResponseFormatError"
    assert outcome["model_id"] == "malformed-model"
    assert outcome["latency_s"] == 0.0
    assert outcome["provider_metadata"]["provider"] == "scripted"
    assert metadata["capture"]["scene_context_included"] is True
    assert metadata["replay"]["exact_scene_context"] is True


def test_probe_replay_rejects_tampered_saved_png_before_inference(tmp_path: Path) -> None:
    config = config_from_dict({})
    source = tmp_path / "source"
    run_vla_probe_catalog(config, [_scenario()], source, capture=_capture)
    frame_path = source / "clear-road" / "frame.png"
    frame_path.write_bytes(frame_path.read_bytes()[:-1] + b"x")
    provider = ScriptedModelProvider([])

    summary = replay_vla_probe_artifacts(
        config,
        source,
        tmp_path / "replay",
        provider=provider,
    )

    assert summary.successful is False
    assert provider.requests == ()
    assert summary.scenarios[0].error_category == "ValueError"


def test_probe_replay_rejects_tampered_saved_prompt_before_inference(
    tmp_path: Path,
) -> None:
    config = config_from_dict({})
    source = tmp_path / "source"
    run_vla_probe_catalog(config, [_scenario()], source, capture=_capture)
    prompt_path = source / "clear-road" / "prompt.txt"
    prompt_path.write_text(prompt_path.read_text() + "\ntampered")
    provider = ScriptedModelProvider([])

    summary = replay_vla_probe_artifacts(
        config,
        source,
        tmp_path / "replay",
        provider=provider,
    )

    assert summary.successful is False
    assert provider.requests == ()
    assert summary.scenarios[0].error_category == "ValueError"


def test_probe_replay_rejects_tampered_saved_scene_before_inference(
    tmp_path: Path,
) -> None:
    config = config_from_dict({})
    source = tmp_path / "source"
    run_vla_probe_catalog(config, [_scenario()], source, capture=_capture)
    scene_path = source / "clear-road" / "scene.json"
    scene_path.write_text(scene_path.read_text() + " ")
    provider = ScriptedModelProvider([])

    summary = replay_vla_probe_artifacts(
        config,
        source,
        tmp_path / "replay",
        provider=provider,
    )

    assert summary.successful is False
    assert provider.requests == ()
    assert summary.scenarios[0].error_category == "ValueError"


# Probing a team's observation: what is built is what is saved, sent, and replayed.

STOP_RESPONSE = json.dumps(
    {
        "scene_summary": "Stopped vehicle ahead.",
        "relevant_hazards": [{"type": "vehicle", "relative_location": "front", "risk": "high"}],
        "meta_action": "STOP",
        "target_speed_mps": 0.0,
        "confidence": 0.9,
        "brief_justification": "Stop for the vehicle.",
    }
)


class _TeamObservation:
    """Draws over the frame and writes its own prompt, without the scene context."""

    def build(self, request: ObservationRequest) -> Observation:
        frame = RGBFrame(request.frame.timestamp_s, 1, 1, b"\x80\x80\x80")
        prompt = "\n".join(["Team observation.", *request.contract.lines])
        return Observation(frame=frame, prompt=prompt)


def test_probe_saves_and_sends_the_team_observation(tmp_path: Path) -> None:
    provider = ScriptedModelProvider([STOP_RESPONSE])

    run_vla_probe_catalog(
        config_from_dict({}),
        [_scenario()],
        tmp_path / "probe",
        infer=True,
        provider=provider,
        capture=_capture,
        observation_builder=_TeamObservation(),
    )

    artifact_dir = tmp_path / "probe" / "clear-road"
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    [request] = provider.requests
    assert request.prompt == (artifact_dir / "prompt.txt").read_text()
    assert request.prompt.startswith("Team observation.\n")
    assert request.frame == RGBFrame(1.0, 1, 1, b"\x80\x80\x80")
    assert (artifact_dir / "frame.png").read_bytes() == encode_rgb_frame_png(request.frame)
    assert metadata["capture"]["observation_builder"] == f"{__name__}._TeamObservation"
    assert (metadata["capture"]["width"], metadata["capture"]["height"]) == (1, 1)
    # The measured scene is kept beside the observation even when the prompt omits it.
    assert json.loads((artifact_dir / "scene.json").read_text())["objects"][0]["id"] == "lead"


def test_replay_sends_a_saved_team_observation_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run_vla_probe_catalog(
        config_from_dict({}),
        [_scenario()],
        source,
        capture=_capture,
        observation_builder=_TeamObservation(),
    )
    provider = ScriptedModelProvider([STOP_RESPONSE])

    summary = replay_vla_probe_artifacts(
        config_from_dict({}),
        source,
        tmp_path / "replay",
        provider=provider,
    )

    [request] = provider.requests
    assert summary.successful is True
    assert request.prompt == (source / "clear-road" / "prompt.txt").read_text()
    assert request.frame.rgb_bytes == b"\x80\x80\x80"


def test_probe_without_a_submission_refuses_changed_observation_settings(
    tmp_path: Path,
) -> None:
    config = config_from_dict({"observation": {"scene_context": False}})

    with pytest.raises(ValueError, match="read by a submitted observation.py"):
        run_vla_probe_catalog(config, [_scenario()], tmp_path / "probe", capture=_capture)


def test_probe_summary_scores_each_dimension_over_the_scenes_it_scored(
    tmp_path: Path,
) -> None:
    stopped = ProbeEvaluationSpec(
        acceptable_actions=(HighLevelAction.STOP,),
        minimum_target_speed_mps=0.0,
        maximum_target_speed_mps=1.0,
        required_hazards=(ProbeHazardExpectation(HazardType.VEHICLE, RelativeLocation.FRONT),),
    )
    clear = ProbeEvaluationSpec(
        acceptable_actions=(HighLevelAction.KEEP_LANE,),
        minimum_target_speed_mps=5.0,
        maximum_target_speed_mps=10.0,
    )

    summary = run_vla_probe_catalog(
        config_from_dict({}),
        [_scenario("stopped", stopped), _scenario("clear", clear), _scenario("unscored")],
        tmp_path / "probe",
        infer=True,
        provider=ScriptedModelProvider([STOP_RESPONSE] * 3),
        capture=_capture,
    )

    assert summary.to_dict()["scores"] == {
        "semantic": {"passed": 1, "scored": 2},
        "perception": {"passed": 1, "scored": 1},
        "tactical_action": {"passed": 1, "scored": 2},
        "raw_speed": {"passed": 1, "scored": 2},
    }


TEAM_OBSERVATION = '''
from metadrive_starter.vla.camera import RGBFrame
from metadrive_starter.vla.observation import Observation

class ObservationBuilder:
    def __init__(self, settings):
        self.settings = settings

    def build(self, request):
        frame = RGBFrame(request.frame.timestamp_s, 1, 1, bytes([128, 128, 128]))
        lines = ["Team observation.", *request.contract.lines]
        if request.prompt_policy:
            lines.append(request.prompt_policy)
        return Observation(frame=frame, prompt="\\n".join(lines))
'''


def test_cli_probe_builds_observations_with_the_submission(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(vla_probe, "capture_probe_scenario", _capture)
    team = tmp_path / "team"
    team.mkdir()
    (team / "observation.py").write_text(TEAM_OBSERVATION)
    (team / "agent.yaml").write_text("vla:\n  prompt_policy: Team policy.\n")
    manifest = tmp_path / "probe.yaml"
    manifest.write_text(
        "version: 2\nscenarios:\n"
        "  - id: clear-road\n    description: Clear.\n    expected_visual: Road.\n    map: S\n"
    )

    status = main(
        [
            "probe",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "probe"),
            "--submission",
            str(team),
        ]
    )

    assert status == 0, capsys.readouterr().err
    prompt = (tmp_path / "probe" / "clear-road" / "prompt.txt").read_text()
    assert prompt.startswith("Team observation.\n")
    assert prompt.endswith("\nTeam policy.")


def test_cli_probe_replay_refuses_a_submission(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "probe",
                "--infer",
                "--replay-from",
                str(tmp_path),
                "--submission",
                str(tmp_path),
            ]
        )

    assert "--submission applies only when capturing" in capsys.readouterr().err
