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


class _Credentials:
    def __init__(self, valid: bool) -> None:
        self.valid = valid
        self.refreshed = 0

    def refresh(self, request: object) -> None:
        self.refreshed += 1
        self.valid = True


def _fake_sdk(created: list[dict[str, object]], models: RecordingModels | None = None):
    class FakeGenAI:
        @staticmethod
        def Client(**values: object) -> object:
            created.append(values)
            return SimpleNamespace(models=models or RecordingModels(command_text()))

    return lambda: (FakeGenAI, FakeTypes)


def test_prepare_builds_the_client_and_refreshes_expired_credentials_without_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from metadrive_starter.vla import vertex_provider

    monkeypatch.setattr(vertex_provider, "_credentials_request", lambda: object())
    created: list[dict[str, object]] = []
    models = RecordingModels(command_text())
    credentials = _Credentials(valid=False)
    loads = 0

    def load_credentials() -> _Credentials:
        nonlocal loads
        loads += 1
        return credentials

    provider = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        sdk_loader=_fake_sdk(created, models),
        credentials_loader=load_credentials,
    )

    elapsed = provider.prepare(timeout_s=6.0)

    assert elapsed >= 0.0
    assert len(created) == 1
    assert created[0]["credentials"] is credentials
    assert created[0]["http_options"].timeout == 6000  # type: ignore[union-attr]
    assert credentials.refreshed == 1
    assert models.calls == []
    # The request then reuses the prepared client and credentials.
    provider.generate(request(), timeout_s=6.0)
    assert len(created) == 1
    assert loads == 1
    assert len(models.calls) == 1


def test_prepare_leaves_valid_credentials_alone_and_an_injected_client_untouched() -> None:
    created: list[dict[str, object]] = []
    credentials = _Credentials(valid=True)
    provider = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        sdk_loader=_fake_sdk(created),
        credentials_loader=lambda: credentials,
    )
    assert provider.prepare(timeout_s=6.0) >= 0.0
    assert credentials.refreshed == 0

    injected = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        client=SimpleNamespace(models=RecordingModels(command_text())),
        types_module=FakeTypes,
    )
    assert injected.prepare(timeout_s=6.0) == 0.0


def test_prepare_reports_a_client_it_cannot_build_as_a_provider_error() -> None:
    def unavailable() -> tuple[object, object]:
        raise ImportError("no google-genai")

    provider = VertexModelProvider(
        "project-123", "global", "course-model", sdk_loader=unavailable
    )

    with pytest.raises(ModelProviderError, match="cloud extra"):
        provider.prepare(timeout_s=6.0)

    def failing_credentials() -> object:
        raise RuntimeError("no ADC")

    created: list[dict[str, object]] = []
    provider = VertexModelProvider(
        "project-123",
        "global",
        "course-model",
        sdk_loader=_fake_sdk(created),
        credentials_loader=failing_credentials,
    )
    with pytest.raises(ModelProviderError, match="could not be prepared"):
        provider.prepare(timeout_s=6.0)
    with pytest.raises(ValueError, match="timeout_s"):
        provider.prepare(timeout_s=0.0)


# A demo student project: no refusal message may repeat it, its number, or the
# troubleshooter id that encodes it.
STUDENT_PROJECT = "student-demo-4821"
STUDENT_PROJECT_NUMBER = "481516234200"
ENCODED_PROJECT = "c3R1ZGVudC1kZW1vLTQ4MjE="


class ClientError(Exception):
    """Shaped as ``google.genai.errors.APIError``: ``code``, ``status``, ``details``."""

    def __init__(self, code: int, response_json: dict[str, object]) -> None:
        error = response_json.get("error", response_json)
        assert isinstance(error, dict)
        self.code = code
        self.status = error.get("status")
        self.message = error.get("message")
        self.details = response_json
        super().__init__(f"{code} {self.status}. {response_json}")


def iam_refusal() -> ClientError:
    resource = (
        f"projects/{STUDENT_PROJECT}/locations/global/publishers/google/models/course-model"
    )
    return ClientError(
        403,
        {
            "error": {
                "code": 403,
                "message": (
                    "Permission 'aiplatform.endpoints.predict' denied on resource "
                    f"'//aiplatform.googleapis.com/{resource}' (or it may not exist)."
                ),
                "status": "PERMISSION_DENIED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "IAM_PERMISSION_DENIED",
                        "domain": "aiplatform.googleapis.com",
                        "metadata": {
                            "permission": "aiplatform.endpoints.predict",
                            "resource": resource,
                            "error_info_id": ENCODED_PROJECT,
                        },
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.Help",
                        "links": [
                            {
                                "description": "Troubleshoot",
                                "url": (
                                    "https://console.cloud.google.com/iam-admin/"
                                    f"troubleshooter;errorId={ENCODED_PROJECT}"
                                ),
                            }
                        ],
                    },
                ],
            }
        },
    )


def service_disabled_refusal() -> ClientError:
    return ClientError(
        403,
        {
            "error": {
                "code": 403,
                "message": (
                    f"Vertex AI API has not been used in project {STUDENT_PROJECT_NUMBER} "
                    "before or it is disabled."
                ),
                "status": "PERMISSION_DENIED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "SERVICE_DISABLED",
                        "domain": "googleapis.com",
                        "metadata": {
                            "service": "aiplatform.googleapis.com",
                            "consumer": f"projects/{STUDENT_PROJECT_NUMBER}",
                            "activationUrl": (
                                "https://console.developers.google.com/apis/api/"
                                "aiplatform.googleapis.com/overview?project="
                                f"{STUDENT_PROJECT_NUMBER}"
                            ),
                        },
                    }
                ],
            }
        },
    )


def quota_refusal() -> ClientError:
    return ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": f"Resource exhausted in {STUDENT_PROJECT}. Try again later.",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )


def unwrapped_iam_refusal() -> ClientError:
    """The SDK's replay path keeps ``details`` as the inner ``error`` object."""

    wrapped = iam_refusal().details
    assert isinstance(wrapped, dict)
    return ClientError(403, wrapped["error"])


def refusal_with_untrusted_fields() -> ClientError:
    return ClientError(
        403,
        {
            "error": {
                "code": 403,
                "status": f"denied for {STUDENT_PROJECT}",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": f"projects/{STUDENT_PROJECT}",
                        "metadata": {"permission": f"{STUDENT_PROJECT} predict"},
                    }
                ],
            }
        },
    )


REFUSALS = [
    pytest.param(
        iam_refusal,
        "403 PERMISSION_DENIED, IAM_PERMISSION_DENIED, permission aiplatform.endpoints.predict",
        ModelFailureCategory.AUTHENTICATION,
        id="missing-iam-grant",
    ),
    pytest.param(
        service_disabled_refusal,
        "403 PERMISSION_DENIED, SERVICE_DISABLED",
        ModelFailureCategory.AUTHENTICATION,
        id="api-disabled",
    ),
    pytest.param(
        quota_refusal,
        "429 RESOURCE_EXHAUSTED",
        ModelFailureCategory.QUOTA,
        id="quota",
    ),
    pytest.param(
        unwrapped_iam_refusal,
        "403 PERMISSION_DENIED, IAM_PERMISSION_DENIED, permission aiplatform.endpoints.predict",
        ModelFailureCategory.AUTHENTICATION,
        id="details-without-error-wrapper",
    ),
    pytest.param(
        refusal_with_untrusted_fields,
        "403",
        ModelFailureCategory.AUTHENTICATION,
        id="fields-that-are-not-tokens-are-dropped",
    ),
    pytest.param(
        lambda: RuntimeError(f"connection reset talking to {STUDENT_PROJECT}"),
        "RuntimeError",
        ModelFailureCategory.TRANSPORT,
        id="not-an-api-error",
    ),
]


@pytest.mark.parametrize(("make_error", "reason", "category"), REFUSALS)
def test_vertex_provider_says_why_vertex_refused_without_naming_the_project(
    make_error: object,
    reason: str,
    category: ModelFailureCategory,
) -> None:
    error = make_error()  # type: ignore[operator]
    provider = VertexModelProvider(
        STUDENT_PROJECT,
        "global",
        "course-model",
        client=SimpleNamespace(models=RecordingModels(error=error)),
        types_module=FakeTypes,
    )

    with pytest.raises(ModelProviderError) as raised:
        provider.generate(request(), timeout_s=5.0)

    message = str(raised.value)
    assert message == f"Vertex model request failed ({reason})"
    assert raised.value.category is category
    assert raised.value.__cause__ is error
    for secret in (STUDENT_PROJECT, STUDENT_PROJECT_NUMBER, ENCODED_PROJECT):
        assert secret not in message
