from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from metadrive_starter.vla import (
    ModelFailureCategory,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelTimeoutError,
    RGBFrame,
    VertexModelProvider,
)


class Value:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)


class Part:
    @staticmethod
    def from_bytes(*, data: bytes, mime_type: str) -> Value:
        return Value(data=data, mime_type=mime_type)

    @staticmethod
    def from_text(*, text: str) -> Value:
        return Value(text=text)


class FakeTypes:
    Part = Part
    Content = Value
    GenerateContentConfig = Value
    AutomaticFunctionCallingConfig = Value
    ThinkingConfig = Value
    HttpOptions = Value
    HttpRetryOptions = Value


class RecordingModels:
    def __init__(self, response_text: str | None = None, error: Exception | None = None) -> None:
        self.response_text = response_text
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **values: object) -> object:
        self.calls.append(values)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.response_text)


def command_text() -> str:
    return json.dumps(
        {
            "scene_summary": "Clear road.",
            "relevant_hazards": [],
            "meta_action": "KEEP_LANE",
            "target_speed_mps": 8.0,
            "confidence": 0.9,
            "brief_justification": "No hazards visible.",
        }
    )


def request() -> ModelRequest:
    return ModelRequest(
        request_id="request-1",
        prompt="Return one safe command.",
        frame=RGBFrame(10.0, 1, 1, b"\x01\x02\x03"),
        created_at_s=10.0,
    )


def test_vertex_provider_implements_model_boundary_and_sends_png_schema() -> None:
    models = RecordingModels(command_text())
    clock_values = iter((20.0, 20.25))
    provider = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        client=SimpleNamespace(models=models),
        types_module=FakeTypes,
        clock=lambda: next(clock_values),
    )

    response = provider.generate(request(), timeout_s=5.0)

    assert isinstance(provider, ModelProvider)
    assert response.request_id == "request-1"
    assert response.text == command_text()
    assert response.model_id == "course-model"
    assert response.latency_s == 0.25
    assert response.metadata.provider == "vertex"
    call = models.calls[0]
    assert call["model"] == "course-model"
    content = call["contents"][0]  # type: ignore[index]
    assert content.role == "user"
    assert content.parts[0].mime_type == "image/png"
    assert content.parts[0].data.startswith(b"\x89PNG\r\n\x1a\n")
    assert content.parts[1].text == request().prompt
    config = call["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema["required"] == [
        "scene_summary",
        "relevant_hazards",
        "meta_action",
        "target_speed_mps",
        "confidence",
        "brief_justification",
    ]
    assert config.automatic_function_calling.disable is True


def test_vertex_provider_builds_sdk_client_with_requested_timeout() -> None:
    models = RecordingModels(command_text())
    created: list[dict[str, object]] = []

    class FakeGenAI:
        @staticmethod
        def Client(**values: object) -> object:
            created.append(values)
            return SimpleNamespace(models=models)

    provider = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        sdk_loader=lambda: (FakeGenAI, FakeTypes),
    )

    provider.generate(request(), timeout_s=7.5)

    assert created[0]["vertexai"] is True
    assert created[0]["project"] == "project-123"
    assert created[0]["location"] == "global"
    assert created[0]["http_options"].timeout == 7500  # type: ignore[union-attr]
    assert created[0]["http_options"].retry_options.attempts == 1  # type: ignore[union-attr]


def test_vertex_provider_maps_timeout_without_retrying() -> None:
    models = RecordingModels(error=TimeoutError("network deadline"))
    provider = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        client=SimpleNamespace(models=models),
        types_module=FakeTypes,
    )

    with pytest.raises(ModelTimeoutError, match="timed out"):
        provider.generate(request(), timeout_s=5.0)

    assert len(models.calls) == 1


def test_vertex_provider_rejects_empty_response() -> None:
    provider = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        client=SimpleNamespace(models=RecordingModels("")),
        types_module=FakeTypes,
    )

    with pytest.raises(ModelProviderError, match="empty response"):
        provider.generate(request(), timeout_s=5.0)


def test_vertex_provider_preserves_response_identity_version_and_usage() -> None:
    response = SimpleNamespace(
        text=command_text(),
        response_id="vertex-response-1",
        model_version="gemini-effective-001",
        usage_metadata=SimpleNamespace(
            prompt_token_count=120,
            candidates_token_count=30,
            total_token_count=150,
        ),
    )

    class MetadataModels:
        @staticmethod
        def generate_content(**values: object) -> object:
            del values
            return response

    provider = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        client=SimpleNamespace(models=MetadataModels()),
        types_module=FakeTypes,
    )

    result = provider.generate(request(), timeout_s=5.0)

    assert result.metadata.response_id == "vertex-response-1"
    assert result.metadata.model_version == "gemini-effective-001"
    assert result.metadata.input_tokens == 120
    assert result.metadata.output_tokens == 30
    assert result.metadata.total_tokens == 150


def test_vertex_provider_normalizes_quota_error() -> None:
    error = RuntimeError("resource exhausted")
    error.status_code = 429  # type: ignore[attr-defined]
    provider = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        client=SimpleNamespace(models=RecordingModels(error=error)),
        types_module=FakeTypes,
    )

    with pytest.raises(ModelProviderError) as raised:
        provider.generate(request(), timeout_s=5.0)

    assert raised.value.category is ModelFailureCategory.QUOTA


def test_vertex_provider_reports_missing_optional_sdk() -> None:
    def unavailable() -> tuple[object, object]:
        raise ImportError("missing")

    provider = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        sdk_loader=unavailable,
    )

    with pytest.raises(ModelProviderError, match="uv sync --extra cloud"):
        provider.generate(request(), timeout_s=5.0)
