from __future__ import annotations

import json
from pathlib import Path

import pytest

from metadrive_starter.config import config_from_dict
from metadrive_starter.events import EventLogger
from metadrive_starter.hazard_agreement import (
    EvidenceClass,
    ScoredAssessment,
    calibrate,
    main,
    scan_event_logs,
    score_hazards,
    summarize,
)
from metadrive_starter.perception import (
    LaneRelation,
    LocalScene,
    TrackedObject,
    TrafficLightObservation,
    TrafficLightState,
)
from metadrive_starter.vla import HighLevelAction
from metadrive_starter.vla.assessment import (
    HazardType,
    RelativeLocation,
    RiskLevel,
    VLAAssessment,
    VLAHazard,
)
from metadrive_starter.vla.provider import ModelResponseMetadata


def _hazard(hazard_type: str, location: str) -> VLAHazard:
    return VLAHazard(HazardType(hazard_type), RelativeLocation(location), RiskLevel.MEDIUM)


def _assessment(*hazards: VLAHazard, confidence: float = 0.9) -> VLAAssessment:
    return VLAAssessment(
        scene_summary="A road.",
        relevant_hazards=hazards,
        proposed_action=HighLevelAction.KEEP_LANE,
        proposed_target_speed_mps=5.0,
        confidence=confidence,
        brief_justification="Test assessment.",
    )


def _object(
    object_id: str = "lead",
    *,
    kind: str = "vehicle",
    position: tuple[float, float] = (18.0, 0.0),
    lane_relation: LaneRelation = LaneRelation.SAME,
    in_path: bool = True,
) -> TrackedObject:
    return TrackedObject(
        object_id=object_id,
        kind=kind,
        relative_position_m=position,
        relative_velocity_mps=(0.0, 0.0),
        length_m=4.5,
        width_m=1.8,
        lane_relation=lane_relation,
        in_path=in_path,
        path_distance_m=position[0] if in_path else None,
    )


def _left_lane_car() -> TrackedObject:
    return _object(
        "left-car",
        position=(25.0, 3.5),
        lane_relation=LaneRelation.LEFT,
        in_path=False,
    )


def _light(state: TrafficLightState) -> TrafficLightObservation:
    return TrafficLightObservation(
        light_id="signal",
        state=state,
        relative_position_m=(30.0, 4.0),
        in_path=True,
        path_distance_m=30.0,
    )


def _scene(
    *objects: TrackedObject,
    lights: tuple[TrafficLightObservation, ...] = (),
    valid: bool = True,
) -> LocalScene:
    return LocalScene(
        timestamp_s=2.0,
        ego_speed_mps=5.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        objects=objects,
        traffic_lights=lights,
        valid=valid,
    )


def _write_run(
    path: Path,
    steps: list[tuple[VLAAssessment | None, LocalScene]],
    *,
    provider: str,
    model_id: str,
    response_provider: str,
    fixture_id: str | None = None,
    run_id: str = "run-1",
) -> None:
    """Log one run the way the simulator does, one validation per step."""
    config = config_from_dict({"vla": {"enabled": True, "provider": provider}})
    with EventLogger(path, run_id=run_id) as logger:
        logger.write("run_started", sim_time_s=0.0, payload={"config": config.to_dict()})
        for index, (assessment, scene) in enumerate(steps, start=1):
            request_id = f"{run_id}-vla-{index}"
            command = (assessment or _assessment()).to_command(
                command_id=request_id,
                issued_at_s=scene.timestamp_s - 1.0,
                action_horizon_s=3.0,
            )
            if assessment is not None:
                logger.write(
                    "inference_completed",
                    sim_time_s=scene.timestamp_s,
                    payload={
                        "request_id": request_id,
                        "model_id": model_id,
                        "provider_metadata": ModelResponseMetadata(
                            provider=response_provider, fixture_id=fixture_id
                        ),
                        "assessment": assessment,
                        "requested_command": command,
                    },
                )
            payload: dict[str, object] = {
                "now_s": scene.timestamp_s,
                "requested_command": command,
                "scene": scene,
            }
            if assessment is not None:
                payload["assessment"] = assessment
            logger.write("command_validation", sim_time_s=scene.timestamp_s, payload=payload)


def _scored(confidence: float, *, agrees: bool) -> ScoredAssessment:
    claims = () if agrees else (_hazard("vehicle", "front"),)
    return ScoredAssessment(
        source="events.jsonl",
        run_id="run-1",
        sim_time_s=2.0,
        evidence=EvidenceClass.LIVE,
        profile="http:model",
        confidence=confidence,
        observation_age_s=1.0,
        agreement=score_hazards(claims, _scene()),
    )


def test_reporting_the_lead_vehicle_agrees_with_the_scene() -> None:
    agreement = score_hazards([_hazard("vehicle", "front")], _scene(_object()))

    assert agreement.agrees
    assert [hazard.source_id for hazard in agreement.matched] == ["lead"]
    assert agreement.false_claims == ()
    assert agreement.missed == ()


def test_missing_an_in_path_hazard_disagrees_but_missing_a_bystander_does_not() -> None:
    missed_lead = score_hazards([], _scene(_object()))
    missed_bystander = score_hazards(
        [_hazard("vehicle", "front")], _scene(_object(), _left_lane_car())
    )

    assert not missed_lead.agrees
    assert [hazard.source_id for hazard in missed_lead.missed] == ["lead"]
    assert missed_bystander.agrees
    assert [hazard.source_id for hazard in missed_bystander.missed] == ["left-car"]


def test_a_hazard_the_scene_lacks_is_false_and_road_features_are_not_checkable() -> None:
    invented = score_hazards([_hazard("vehicle", "front")], _scene())
    road_feature = score_hazards([_hazard("road_feature", "front")], _scene())

    assert not invented.agrees
    assert invented.false_claims == (_hazard("vehicle", "front"),)
    assert road_feature.agrees
    assert road_feature.claims == ()
    assert road_feature.unverifiable_claims == 1


def test_locations_match_within_one_sector_of_the_prompt_description() -> None:
    lead = _scene(_object())
    left_car = _scene(_left_lane_car())

    assert score_hazards([_hazard("vehicle", "front_left")], lead).agrees
    assert not score_hazards([_hazard("vehicle", "rear")], lead).agrees
    # The prompt describes a car ahead in the left lane as front_left.
    assert score_hazards([_hazard("vehicle", "left")], left_car).matched
    assert not score_hazards([_hazard("vehicle", "front_right")], left_car).matched


def test_each_scene_hazard_supports_one_claim_and_exact_locations_pair_first() -> None:
    agreement = score_hazards(
        [_hazard("vehicle", "front_left"), _hazard("vehicle", "front")],
        _scene(_object()),
    )

    assert len(agreement.matched) == 1
    assert agreement.false_claims == (_hazard("vehicle", "front_left"),)


def test_a_green_light_supports_a_claim_but_need_not_be_reported() -> None:
    green = _scene(lights=(_light(TrafficLightState.GREEN),))
    red = _scene(lights=(_light(TrafficLightState.RED),))

    assert score_hazards([], green).agrees
    assert score_hazards([_hazard("traffic_control", "front")], green).agrees
    assert not score_hazards([], red).agrees
    assert score_hazards([_hazard("traffic_control", "front_right")], red).agrees


def test_object_kinds_map_to_the_hazard_types_a_model_may_report() -> None:
    scene = _scene(
        _object("walker", kind="pedestrian"),
        _object("rider", kind="cyclist", position=(22.0, 0.0)),
        _object("cone", kind="traffic_cone", position=(26.0, 0.0)),
    )

    agreement = score_hazards(
        [
            _hazard("pedestrian", "front"),
            _hazard("pedestrian", "front"),
            _hazard("obstacle", "front"),
        ],
        scene,
    )

    assert agreement.agrees
    assert len(agreement.matched) == 3


def test_live_and_synthetic_evidence_are_never_pooled(tmp_path: Path) -> None:
    lead = _scene(_object())
    seen = _assessment(_hazard("vehicle", "front"))
    _write_run(
        tmp_path / "live-vertex.jsonl",
        [(seen, lead)],
        provider="vertex",
        model_id="gemini",
        response_provider="vertex",
    )
    _write_run(
        tmp_path / "replayed-capture.jsonl",
        [(seen, lead), (seen, lead)],
        provider="fixture",
        model_id="gemini",
        response_provider="vertex",
        fixture_id="vertex-capture",
    )
    _write_run(
        tmp_path / "synthetic.jsonl",
        [(_assessment(), lead)],
        provider="fixture",
        model_id="synthetic-model",
        response_provider="synthetic_fixture",
        fixture_id="synthetic-keep-lane",
    )

    report = summarize(scan_event_logs([tmp_path]))

    by_class = {
        (summary.evidence, summary.profile): summary.assessments
        for summary in report.profiles
    }
    assert by_class == {
        (EvidenceClass.LIVE, "vertex:gemini"): 1,
        (EvidenceClass.SYNTHETIC, "fixture:gemini"): 2,
        (EvidenceClass.SYNTHETIC, "fixture:synthetic-model"): 1,
    }
    assert report.profiles[0].evidence is EvidenceClass.LIVE


def test_summary_counts_precision_recall_and_agreement(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    _write_run(
        log,
        [
            (_assessment(_hazard("vehicle", "front"), confidence=0.9), _scene(_object())),
            (
                _assessment(_hazard("vehicle", "rear"), confidence=0.6),
                _scene(_object(), _left_lane_car()),
            ),
        ],
        provider="http",
        model_id="qwen",
        response_provider="openai_compatible_http",
    )

    (summary,) = summarize(scan_event_logs([log])).profiles

    assert summary.evidence is EvidenceClass.LIVE
    assert summary.profile == "http:qwen"
    assert (summary.assessments, summary.runs) == (2, 1)
    assert summary.scenes_with_in_path_hazards == 2
    assert summary.precision == pytest.approx(0.5)
    assert summary.recall == pytest.approx(1 / 3)
    assert summary.in_path_recall == pytest.approx(0.5)
    assert summary.agreement_rate == pytest.approx(0.5)
    assert summary.median_observation_age_s == pytest.approx(1.0)
    assert summary.calibration.auroc == pytest.approx(1.0)
    assert [item.sim_time_s for item in summary.disagreements] == [2.0]


def test_events_without_a_usable_assessment_are_skipped_with_a_reason(
    tmp_path: Path,
) -> None:
    log = tmp_path / "events.jsonl"
    _write_run(
        log,
        [
            (None, _scene(_object())),
            (_assessment(), _scene(valid=False)),
            (_assessment(), _scene()),
        ],
        provider="http",
        model_id="qwen",
        response_provider="openai_compatible_http",
    )

    scan = scan_event_logs([log])

    assert len(scan.assessments) == 1
    assert [event.reason for event in scan.skipped_events] == [
        "no assessment recorded",
        "scene invalid",
    ]
    assert all(event.evidence is EvidenceClass.LIVE for event in scan.skipped_events)


def test_a_directory_scan_skips_jsonl_that_is_not_an_event_log(tmp_path: Path) -> None:
    _write_run(
        tmp_path / "run" / "events.jsonl",
        [(_assessment(), _scene())],
        provider="http",
        model_id="qwen",
        response_provider="openai_compatible_http",
    )
    (tmp_path / "notes.jsonl").write_text('{"note": "not an event"}\n', encoding="utf-8")

    scan = scan_event_logs([tmp_path])

    assert scan.logs_read == 1
    assert [item.source for item in scan.skipped_files] == ["notes.jsonl"]
    assert scan.assessments[0].source == str(Path("run") / "events.jsonl")
    with pytest.raises(FileNotFoundError):
        scan_event_logs([tmp_path / "missing.jsonl"])


def test_constant_confidence_cannot_separate_agreement() -> None:
    calibration = calibrate(
        [_scored(0.95, agrees=True)] * 9 + [_scored(0.95, agrees=False)]
    )

    assert calibration.distinct_confidences == 1
    assert calibration.auroc == pytest.approx(0.5)
    (only_bin,) = calibration.bins
    assert (only_bin.lower, only_bin.assessments) == (pytest.approx(0.9), 10)
    assert only_bin.agreement_rate == pytest.approx(0.9)
    assert calibration.expected_calibration_error == pytest.approx(0.05)


def test_calibration_bins_confidence_and_scores_its_ranking() -> None:
    calibration = calibrate([_scored(0.9, agrees=True), _scored(0.2, agrees=False)])

    assert [item.lower for item in calibration.bins] == [
        pytest.approx(0.2),
        pytest.approx(0.9),
    ]
    assert calibration.auroc == pytest.approx(1.0)
    assert calibration.expected_calibration_error == pytest.approx(0.15)
    assert calibrate([_scored(1.0, agrees=True)]).auroc is None


def test_main_reports_both_evidence_classes_as_text_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_run(
        tmp_path / "events.jsonl",
        [(_assessment(_hazard("vehicle", "front")), _scene())],
        provider="http",
        model_id="qwen",
        response_provider="openai_compatible_http",
    )

    assert main([str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert main([str(tmp_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert "Live model evidence: 1 assessment, 1 profile, 1 run" in text
    assert "Synthetic evidence" in text
    assert "no scene support for vehicle front medium" in text
    assert report["profiles"][0]["evidence"] == "live"
    assert report["profiles"][0]["precision"] == 0.0
