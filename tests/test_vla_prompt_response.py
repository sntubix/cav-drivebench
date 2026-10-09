from __future__ import annotations

import json
from dataclasses import replace

import pytest

from metadrive_starter.perception import (
    LaneRelation,
    LocalScene,
    TrackedObject,
    TrafficLightObservation,
    TrafficLightState,
)
from metadrive_starter.vla import (
    HighLevelAction,
    MODEL_ACTIONS,
    HazardType,
    RiskLevel,
    VLA_PROMPT_CONTRACT_VERSION,
)
from metadrive_starter.vla.prompting import PromptBuildError, build_vla_prompt
from metadrive_starter.vla.response import (
    VLAAssessmentPayloadError,
    VLACommandPayloadError,
    VLAResponseFormatError,
    decode_vla_assessment,
    decode_vla_response,
)


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "action": "SLOW_DOWN",
        "target_speed_mps": 12.0,
        "issued_at_s": 10.0,
        "action_horizon_s": 2.0,
        "confidence": 0.8,
        "command_id": "model-1",
        "justification": "vehicle ahead",
    }
    payload.update(overrides)
    return payload


def _tracked(object_id: str, distance_m: float) -> TrackedObject:
    return TrackedObject(
        object_id=object_id,
        kind="vehicle",
        relative_position_m=(distance_m, 0.0),
        relative_velocity_mps=(-2.0, 0.0),
        length_m=4.5,
        width_m=1.8,
        lane_relation=LaneRelation.SAME,
        in_path=True,
        path_distance_m=distance_m,
        path_relative_velocity_mps=-2.0,
    )


def _scene() -> LocalScene:
    return LocalScene(
        timestamp_s=9.9,
        ego_speed_mps=8.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.1,
        heading_error_rad=0.02,
        objects=(_tracked("far", 30.0), _tracked("near", 8.0)),
        traffic_lights=(
            TrafficLightObservation(
                light_id="signal",
                state=TrafficLightState.RED,
                relative_position_m=(20.0, 0.0),
                in_path=True,
                path_distance_m=20.0,
            ),
        ),
        left_lane_available=True,
        right_lane_available=False,
    )


def test_prompt_is_deterministic_and_constrains_the_output() -> None:
    first = build_vla_prompt(now_s=10.0, action_horizon_s=2.0)
    second = build_vla_prompt(now_s=10.0, action_horizon_s=2.0)

    assert first == second
    assert "front-camera image is the PRIMARY evidence" in first
    assert "LOCAL_SCENE_CONTEXT contains measured tactical facts" in first
    assert "follow it when visual interpretation is ambiguous" in first
    assert "with no Markdown fences or prose" in first
    assert "runtime owns request IDs, timestamps, and command validity" in first
    assert "do not return those fields" in first
    assert "at most 4 relevant hazards" in first
    assert "at most 256 characters" in first
    assert "Mention each observed object at most once" in first
    assert f"Prompt contract: {VLA_PROMPT_CONTRACT_VERSION}" in first
    for action in MODEL_ACTIONS:
        assert action.value in first
    assert "OVERTAKE" not in first and "PULL_OVER" not in first
    for field in (
        "scene_summary",
        "relevant_hazards",
        "meta_action",
        "target_speed_mps",
        "confidence",
        "brief_justification",
    ):
        assert field in first


def test_prompt_includes_yaml_policy_as_a_distinct_bounded_section() -> None:
    policy = (
        "Maintain safe forward progress.\n"
        "Never propose a lane change into an occupied lane."
    )

    prompt = build_vla_prompt(
        now_s=10.0,
        action_horizon_s=2.0,
        prompt_policy=policy,
    )

    assert f"CONFIGURED_POLICY:\n{policy}" in prompt


@pytest.mark.parametrize("policy", [42, "x" * 4097, "bad\x00policy"])
def test_prompt_rejects_invalid_configured_policy(policy: object) -> None:
    with pytest.raises(PromptBuildError, match="prompt_policy"):
        build_vla_prompt(
            now_s=10.0,
            action_horizon_s=2.0,
            prompt_policy=policy,  # type: ignore[arg-type]
        )


def test_prompt_distinguishes_current_speed_from_clear_road_cruise_speed() -> None:
    prompt = build_vla_prompt(
        now_s=10.0,
        action_horizon_s=2.0,
        ego_speed_mps=0.0,
        cruise_speed_mps=35.0,
    )
    context = json.loads(
        prompt.split("DRIVING_CONTEXT:\n", maxsplit=1)[1].split(
            "\nLOCAL_SCENE_CONTEXT:", maxsplit=1
        )[0]
    )

    assert context == {"cruise_speed_mps": 35.0, "ego_speed_mps": 0.0}
    assert "not a copy of the current ego speed" in prompt


def test_prompt_includes_bounded_scene_context_sorted_by_distance() -> None:
    prompt = build_vla_prompt(
        now_s=10.0,
        action_horizon_s=2.0,
        scene=_scene(),
        max_scene_objects=1,
    )
    context = json.loads(prompt.split("LOCAL_SCENE_CONTEXT:\n", maxsplit=1)[1])

    assert context["valid"] is True
    assert context["lanes"]["right"] == {
        "available": False,
        "observed_clear": False,
    }
    assert context["traffic_controls"] == [
        {
            "confidence": 1.0,
            "id": "signal",
            "in_path": True,
            "path_distance_m": 20.0,
            "relative_position_m": [20.0, 0.0],
            "state": "red",
        }
    ]
    assert [item["id"] for item in context["objects"]] == ["near"]
    assert context["objects"][0]["relative_location"] == "front"
    assert context["objects"][0]["distance_m"] == 8.0
    assert context["objects"][0]["estimated_speed_mps"] == 6.0


def test_prompt_normalizes_crossing_and_unknown_objects_into_tactical_locations() -> None:
    lead = replace(_tracked("lead", 22.0), lane_relation=LaneRelation.CROSSING)
    right_blocker = TrackedObject(
        object_id="right-blocker",
        kind="vehicle",
        relative_position_m=(16.0, -3.5),
        relative_velocity_mps=(0.0, 0.0),
        length_m=4.5,
        width_m=1.8,
        lane_relation=LaneRelation.UNKNOWN,
        in_path=False,
        path_distance_m=16.0,
        path_relative_velocity_mps=0.0,
    )
    scene = replace(
        _scene(),
        objects=(lead, right_blocker),
        traffic_lights=(),
        right_lane_available=True,
    )

    context = json.loads(
        build_vla_prompt(
            now_s=10.0,
            action_horizon_s=2.0,
            scene=scene,
        ).split("LOCAL_SCENE_CONTEXT:\n", maxsplit=1)[1]
    )

    assert [item["relative_location"] for item in context["objects"]] == [
        "front_right",
        "front",
    ]
    assert context["lanes"]["left"]["observed_clear"] is True
    assert context["lanes"]["right"]["observed_clear"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"now_s": float("nan"), "action_horizon_s": 2.0}, "now_s"),
        ({"now_s": 10.0, "action_horizon_s": 0.0}, "action_horizon_s"),
        ({"now_s": 10.0, "action_horizon_s": 2.0, "max_scene_objects": -1}, "max_scene_objects"),
        ({"now_s": 10.0, "action_horizon_s": 2.0, "ego_speed_mps": -1.0}, "ego_speed"),
        (
            {"now_s": 10.0, "action_horizon_s": 2.0, "cruise_speed_mps": float("nan")},
            "cruise_speed",
        ),
    ],
)
def test_prompt_rejects_invalid_bounds(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(PromptBuildError, match=message):
        build_vla_prompt(**kwargs)  # type: ignore[arg-type]


def test_decoder_accepts_direct_json_object() -> None:
    command = decode_vla_response(json.dumps(_payload()))

    assert command.action is HighLevelAction.SLOW_DOWN
    assert command.target_speed_mps == 12.0
    assert command.justification == "vehicle ahead"


def test_decoder_rejects_kmh_command_speed() -> None:
    payload = _payload()
    payload.pop("target_speed_mps")
    payload["target_speed_kmh"] = 36.0

    with pytest.raises(VLACommandPayloadError, match="target_speed_mps"):
        decode_vla_response(json.dumps(payload))


def test_decoder_accepts_one_json_fence() -> None:
    response = f"```json\n{json.dumps(_payload(action='STOP'))}\n```"

    command = decode_vla_response(response)

    assert command.action is HighLevelAction.STOP


def test_decoder_accepts_single_command_wrapper() -> None:
    command = decode_vla_response(json.dumps({"command": _payload(action="KEEP_LANE")}))

    assert command.action is HighLevelAction.KEEP_LANE


@pytest.mark.parametrize(
    "response",
    [
        "",
        "I recommend stopping. " + json.dumps(_payload(action="STOP")),
        json.dumps(_payload()) + json.dumps(_payload(action="STOP")),
        f"before\n```json\n{json.dumps(_payload())}\n```\nafter",
        f"```json\n{json.dumps(_payload())}\n```\n```json\n{json.dumps(_payload())}\n```",
        "{not json}",
        '{"action":"STOP","action":"KEEP_LANE"}',
        '{"action":"STOP","target_speed_mps":0,"issued_at_s":10,"action_horizon_s":2,"confidence":NaN}',
        json.dumps([_payload()]),
        json.dumps({"command": _payload(), "reasoning": "extra wrapper field"}),
        json.dumps({"command": "STOP"}),
    ],
)
def test_decoder_rejects_ambiguous_or_non_object_responses(response: str) -> None:
    with pytest.raises(VLAResponseFormatError):
        decode_vla_response(response)


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "STOP"},
        _payload(action="TURN_AROUND"),
        _payload(confidence=2.0),
    ],
)
def test_decoder_reports_command_schema_errors(payload: dict[str, object]) -> None:
    with pytest.raises(VLACommandPayloadError, match="invalid VLA command payload"):
        decode_vla_response(json.dumps(payload))


def _assessment_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "scene_summary": "A vehicle is slowing ahead.",
        "relevant_hazards": [
            {"type": "vehicle", "relative_location": "front", "risk": "medium"}
        ],
        "meta_action": "FOLLOW",
        "target_speed_mps": 8.0,
        "confidence": 0.75,
        "brief_justification": "Maintain a safe following gap.",
    }
    payload.update(overrides)
    return payload


def test_assessment_decoder_returns_typed_non_authoritative_proposal() -> None:
    assessment = decode_vla_assessment(json.dumps(_assessment_payload()))

    assert assessment.proposed_action is HighLevelAction.FOLLOW
    assert assessment.relevant_hazards[0].hazard_type is HazardType.VEHICLE
    assert assessment.relevant_hazards[0].risk is RiskLevel.MEDIUM
    assert assessment.uncertainty == pytest.approx(0.25)


@pytest.mark.parametrize(
    "payload",
    [
        _assessment_payload(scene_summary="x" * 257),
        _assessment_payload(brief_justification="x" * 257),
        _assessment_payload(
            relevant_hazards=[
                {"type": "vehicle", "relative_location": "front", "risk": "low"}
            ]
            * 5
        ),
    ],
)
def test_assessment_decoder_enforces_compact_output_bounds(
    payload: dict[str, object],
) -> None:
    with pytest.raises(VLAAssessmentPayloadError):
        decode_vla_assessment(json.dumps(payload))


@pytest.mark.parametrize(
    "payload",
    [
        _assessment_payload(issued_at_s=10.0),
        _assessment_payload(action_horizon_s=2.0),
        _assessment_payload(meta_action="TURN_AROUND"),
        _assessment_payload(relevant_hazards="vehicle"),
        _assessment_payload(
            relevant_hazards=[
                {"type": "dragon", "relative_location": "front", "risk": "high"}
            ]
        ),
    ],
)
def test_assessment_decoder_rejects_invalid_or_authoritative_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(VLAAssessmentPayloadError):
        decode_vla_assessment(json.dumps(payload))
    decode_vla_assessment,
