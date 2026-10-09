from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from metadrive_starter.vla.camera import RGBFrame
from metadrive_starter.vla.provider import (
    ModelFailureCategory,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelResponseMetadata,
    ModelTimeoutError,
    RequestBudget,
    RequestBudgetError,
    RequestCappedModelProvider,
    ScriptedModelProvider,
)


def _frame() -> RGBFrame:
    return RGBFrame(
        timestamp_s=10.0,
        width=1,
        height=1,
        rgb_bytes=b"\x01\x02\x03",
    )


def _request(request_id: str = "request-1") -> ModelRequest:
    return ModelRequest(
        request_id=request_id,
        prompt="Choose the next safe driving command.",
        frame=_frame(),
        created_at_s=10.1,
    )


def test_request_and_response_are_immutable() -> None:
    request = _request()
    response = ModelResponse(
        request_id=request.request_id,
        text='{"action":"KEEP_LANE"}',
        model_id="test-model",
        latency_s=0.2,
    )

    with pytest.raises(FrozenInstanceError):
        request.prompt = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        response.text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", ""),
        ("request_id", 3),
        ("request_id", "request\nInjected: value"),
        ("episode_id", ""),
        ("episode_id", "bad\nepisode"),
        ("generation_id", 0),
        ("generation_id", True),
        ("prompt", "  "),
        ("prompt", None),
        ("frame", b"not-an-rgb-frame"),
        ("created_at_s", -0.1),
        ("created_at_s", True),
        ("created_at_s", float("nan")),
        ("encoded_png_bytes", b""),
        ("encoded_png_bytes", b"not-a-png"),
        ("prompt_contract_version", ""),
        ("prompt_contract_version", "bad\nversion"),
    ],
)
def test_request_rejects_invalid_contract_fields(field: str, value: object) -> None:
    values: dict[str, object] = {
        "request_id": "request-1",
        "prompt": "Choose a command.",
        "frame": _frame(),
        "created_at_s": 10.1,
    }
    values[field] = value

    with pytest.raises(ValueError):
        ModelRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", ""),
        ("request_id", None),
        ("text", None),
        ("model_id", "  "),
        ("model_id", 4),
        ("latency_s", -0.1),
        ("latency_s", True),
        ("latency_s", float("inf")),
        ("metadata", {}),
    ],
)
def test_response_rejects_invalid_contract_fields(field: str, value: object) -> None:
    values: dict[str, object] = {
        "request_id": "request-1",
        "text": "{}",
        "model_id": "test-model",
        "latency_s": 0.1,
    }
    values[field] = value

    with pytest.raises(ValueError):
        ModelResponse(**values)  # type: ignore[arg-type]


def test_scripted_provider_returns_results_in_order_and_records_requests() -> None:
    provider = ScriptedModelProvider(
        ['{"action":"KEEP_LANE"}', '{"action":"STOP"}'],
        model_id="demo-model",
    )
    first = _request("first")
    second = _request("second")

    first_response = provider.generate(first, timeout_s=1.0)
    second_response = provider.generate(second, timeout_s=1.0)

    assert first_response == ModelResponse(
        request_id="first",
        text='{"action":"KEEP_LANE"}',
        model_id="demo-model",
        latency_s=0.0,
        metadata=ModelResponseMetadata(provider="scripted"),
    )
    assert second_response.request_id == "second"
    assert second_response.text == '{"action":"STOP"}'
    assert provider.requests == (first, second)


def test_scripted_provider_raises_scripted_timeout() -> None:
    provider = ScriptedModelProvider([ModelTimeoutError("model timed out")])

    with pytest.raises(ModelTimeoutError, match="timed out"):
        provider.generate(_request(), timeout_s=0.5)


def test_scripted_provider_reports_exhaustion() -> None:
    provider = ScriptedModelProvider([])

    with pytest.raises(ModelProviderError, match="exhausted"):
        provider.generate(_request(), timeout_s=1.0)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, True, float("nan")])
def test_scripted_provider_rejects_invalid_timeout(timeout_s: float) -> None:
    provider = ScriptedModelProvider(["{}"])

    with pytest.raises(ValueError, match="timeout_s"):
        provider.generate(_request(), timeout_s=timeout_s)


def test_model_provider_protocol_is_structural_at_runtime() -> None:
    class InProcessProvider:
        def generate(
            self,
            request: ModelRequest,
            *,
            timeout_s: float,
        ) -> ModelResponse:
            return ModelResponse(
                request_id=request.request_id,
                text="{}",
                model_id="in-process",
                latency_s=0.0,
            )

    assert isinstance(InProcessProvider(), ModelProvider)
    assert isinstance(ScriptedModelProvider(["{}"]), ModelProvider)


def test_timeout_error_is_catchable_as_provider_and_builtin_timeout() -> None:
    error = ModelTimeoutError("deadline exceeded")

    assert isinstance(error, ModelProviderError)
    assert isinstance(error, TimeoutError)
    assert error.category is ModelFailureCategory.TIMEOUT


def test_response_metadata_is_typed_and_immutable() -> None:
    metadata = ModelResponseMetadata(
        provider="vertex",
        response_id="response-1",
        model_version="model-2026-08",
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
    )
    response = ModelResponse("request-1", "{}", "configured-model", 0.2, metadata)

    assert response.metadata.total_tokens == 120
    with pytest.raises(FrozenInstanceError):
        metadata.total_tokens = 121  # type: ignore[misc]


@pytest.mark.parametrize(
    "values",
    [
        {"provider": ""},
        {"response_id": "bad\nid"},
        {"input_tokens": -1},
        {"output_tokens": True},
        {"total_tokens": 1.5},
        {"fixture_id": ""},
        {"fixture_sha256": "not-a-hash"},
    ],
)
def test_response_metadata_rejects_invalid_values(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelResponseMetadata(**values)  # type: ignore[arg-type]


def test_request_budget_caps_attempts_before_delegating() -> None:
    underlying = ScriptedModelProvider(["{}", "{}"])
    budget = RequestBudget(1)
    provider = RequestCappedModelProvider(underlying, budget)

    provider.generate(_request("first"), timeout_s=1.0)
    with pytest.raises(RequestBudgetError, match="hard request cap"):
        provider.generate(_request("second"), timeout_s=1.0)

    assert [request.request_id for request in underlying.requests] == ["first"]
    assert budget.used_requests == 1
    assert budget.remaining_requests == 0
    assert budget.request_labels == ("first",)


def test_request_budget_rejects_plan_that_cannot_fit() -> None:
    budget = RequestBudget(4)

    with pytest.raises(RequestBudgetError, match="needs 5"):
        budget.require_capacity(5)

    assert budget.used_requests == 0
