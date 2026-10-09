from __future__ import annotations

import json

import pytest

from metadrive_starter.vla import (
    HighLevelAction,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    RGBFrame,
    ScriptedModelProvider,
    VLAAssessmentPayloadError,
    VLAInferenceError,
    VLAInferencePipeline,
    VLAResponseFormatError,
    VLA_PROMPT_CONTRACT_VERSION,
)


def _frame(timestamp_s: float = 10.0) -> RGBFrame:
    return RGBFrame(timestamp_s=timestamp_s, width=1, height=1, rgb_bytes=b"\x01\x02\x03")


def _response(**overrides: object) -> str:
    payload: dict[str, object] = {
        "scene_summary": "Clear road ahead.",
        "relevant_hazards": [],
        "meta_action": "KEEP_LANE",
        "target_speed_mps": 25.0,
        "confidence": 0.9,
        "brief_justification": "Lane is clear.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_pipeline_builds_request_and_decodes_typed_command() -> None:
    provider = ScriptedModelProvider([_response()], model_id="demo-vlm")
    pipeline = VLAInferencePipeline(
        provider,
        timeout_s=3.0,
        action_horizon_s=2.0,
        prompt_policy="Prefer the clear adjacent lane.",
    )
    frame = _frame()

    result = pipeline.infer(
        frame,
        now_s=10.0,
        request_id="inference-1",
        ego_speed_mps=12.0,
        cruise_speed_mps=35.0,
    )

    assert result.requested_command.action is HighLevelAction.KEEP_LANE
    assert result.assessment.scene_summary == "Clear road ahead."
    assert result.requested_command.command_id == "inference-1"
    assert result.requested_command.issued_at_s == 10.0
    assert result.requested_command.action_horizon_s == 2.0
    assert result.response.model_id == "demo-vlm"
    assert provider.requests == (result.request,)
    assert provider.requests[0].frame is frame
    assert (
        provider.requests[0].prompt_contract_version
        == VLA_PROMPT_CONTRACT_VERSION
    )
    assert "front-camera image is the PRIMARY evidence" in provider.requests[0].prompt
    assert '"cruise_speed_mps":35.0' in provider.requests[0].prompt
    assert '"ego_speed_mps":12.0' in provider.requests[0].prompt
    assert "CONFIGURED_POLICY:\nPrefer the clear adjacent lane." in provider.requests[0].prompt


@pytest.mark.parametrize(
    "field",
    [
        "issued_at_s",
        "action_horizon_s",
    ],
)
def test_pipeline_rejects_model_owned_execution_metadata(field: str) -> None:
    pipeline = VLAInferencePipeline(
        ScriptedModelProvider([_response(**{field: 99.0})]),
        action_horizon_s=2.0,
    )

    with pytest.raises(VLAAssessmentPayloadError, match="unknown VLA assessment"):
        pipeline.infer(_frame(), now_s=10.0, request_id="inference-1")


def test_pipeline_stamps_exact_local_time_and_horizon() -> None:
    now_s = 11.200000000000001
    pipeline = VLAInferencePipeline(
        ScriptedModelProvider([_response()]),
        action_horizon_s=2.0,
    )

    result = pipeline.infer(
        _frame(now_s),
        now_s=now_s,
        request_id="inference-1",
    )

    assert result.requested_command.issued_at_s == now_s
    assert result.requested_command.action_horizon_s == 2.0


@pytest.mark.parametrize(
    ("frame_timestamp_s", "message"),
    [(9.4, "stale"), (10.1, "future")],
)
def test_pipeline_rejects_stale_or_future_camera_frames(
    frame_timestamp_s: float,
    message: str,
) -> None:
    pipeline = VLAInferencePipeline(
        ScriptedModelProvider([_response()]),
        maximum_frame_age_s=0.5,
        maximum_clock_skew_s=0.05,
    )

    with pytest.raises(VLAInferenceError, match=message):
        pipeline.infer(
            _frame(frame_timestamp_s),
            now_s=10.0,
            request_id="inference-1",
        )


def test_pipeline_rejects_mismatched_provider_response() -> None:
    class MismatchedProvider:
        def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
            return ModelResponse(
                request_id="different-request",
                text=_response(),
                model_id="broken",
                latency_s=0.0,
            )

    pipeline = VLAInferencePipeline(MismatchedProvider())

    with pytest.raises(VLAInferenceError, match="request_id"):
        pipeline.infer(_frame(), now_s=10.0, request_id="inference-1")


def test_pipeline_rejects_invalid_provider_response_type() -> None:
    class InvalidProvider:
        def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
            return object()  # type: ignore[return-value]

    with pytest.raises(ModelProviderError, match="invalid response type"):
        VLAInferencePipeline(InvalidProvider()).infer(
            _frame(),
            now_s=10.0,
            request_id="inference-1",
        )


def test_pipeline_preserves_response_format_errors() -> None:
    pipeline = VLAInferencePipeline(ScriptedModelProvider(["not JSON"]))

    with pytest.raises(VLAResponseFormatError):
        pipeline.infer(_frame(), now_s=10.0, request_id="inference-1")


@pytest.mark.parametrize("value", [0.0, -1.0, True, float("nan")])
def test_pipeline_rejects_invalid_timeouts(value: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        VLAInferencePipeline(ScriptedModelProvider([]), timeout_s=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_frame_age_s", 0.0),
        ("maximum_frame_age_s", float("inf")),
        ("maximum_clock_skew_s", -0.1),
        ("maximum_clock_skew_s", True),
    ],
)
def test_pipeline_rejects_invalid_frame_time_bounds(field: str, value: object) -> None:
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        VLAInferencePipeline(ScriptedModelProvider([]), **kwargs)  # type: ignore[arg-type]
