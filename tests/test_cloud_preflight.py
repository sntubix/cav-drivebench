from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from metadrive_starter.cloud_preflight import run_vertex_preflight
from metadrive_starter.config import VertexProviderSettings
from metadrive_starter.vla import RequestBudget


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


class PreflightModels:
    def __init__(self) -> None:
        self.calls = 0
        self.configs: list[object] = []
        self.contents: list[object] = []

    def generate_content(self, *, contents: object, **kwargs: object) -> object:
        self.calls += 1
        self.configs.append(kwargs.get("config"))
        self.contents.append(contents)
        if isinstance(contents, str):
            return SimpleNamespace(text="ready")
        return SimpleNamespace(
            text=json.dumps(
                {
                    "scene_summary": "Clear synthetic road.",
                    "relevant_hazards": [],
                    "meta_action": "KEEP_LANE",
                    "target_speed_mps": 8.0,
                    "confidence": 0.9,
                    "brief_justification": "No hazard visible.",
                }
            )
        )


def test_preflight_checks_credentials_text_image_schema_and_latency() -> None:
    models = PreflightModels()

    class FakeGenAI:
        @staticmethod
        def Client(**kwargs: object) -> object:
            del kwargs
            return SimpleNamespace(models=models)

    report = run_vertex_preflight(
        VertexProviderSettings(model_id="course-model"),
        samples=2,
        environment={"GOOGLE_CLOUD_PROJECT": "project-123"},
        sdk_loader=lambda: (FakeGenAI, FakeTypes),
        credentials_loader=lambda: (object(), None),
        prompt_policy="Prefer conservative progress.",
    )

    assert report.successful is True
    assert report.project_id == "project-123"
    assert [check.name for check in report.checks] == [
        "sdk",
        "credentials",
        "configuration",
        "text_round_trip",
        "image_schema_round_trip",
        "latency",
    ]
    assert len(report.latency_samples_s) == 2
    assert report.suggested_minimum_interval_s is not None
    assert models.calls == 3
    image_request = models.contents[1]
    assert "CONFIGURED_POLICY:\nPrefer conservative progress." in (
        image_request[0].parts[1].text  # type: ignore[index,union-attr]
    )
    assert all(
        config.automatic_function_calling.disable is True  # type: ignore[union-attr]
        for config in models.configs
    )


def test_preflight_fails_cleanly_when_cloud_sdk_is_missing() -> None:
    def unavailable() -> tuple[object, object]:
        raise ImportError("missing")

    report = run_vertex_preflight(
        VertexProviderSettings(),
        sdk_loader=unavailable,
    )

    assert report.successful is False
    assert len(report.checks) == 1
    assert report.checks[0].name == "sdk"
    assert "uv sync --extra cloud" in report.checks[0].detail


def test_preflight_counts_each_outbound_attempt_against_shared_budget() -> None:
    models = PreflightModels()
    created: list[dict[str, object]] = []

    class FakeGenAI:
        @staticmethod
        def Client(**kwargs: object) -> object:
            created.append(kwargs)
            return SimpleNamespace(models=models)

    budget = RequestBudget(2)
    report = run_vertex_preflight(
        VertexProviderSettings(),
        samples=1,
        environment={"GOOGLE_CLOUD_PROJECT": "project-123"},
        sdk_loader=lambda: (FakeGenAI, FakeTypes),
        credentials_loader=lambda: (object(), None),
        request_budget=budget,
    )

    assert report.successful is True
    assert budget.used_requests == 2
    assert budget.request_labels == (
        "vertex-preflight-text",
        "vertex-preflight-image-1",
    )
    assert created[0]["http_options"].retry_options.attempts == 1  # type: ignore[union-attr]


STUDENT_PROJECT = "student-demo-4821"


class ClientError(Exception):
    """Shaped as ``google.genai.errors.APIError``: ``code``, ``status``, ``details``."""

    def __init__(self, code: int, response_json: dict[str, object]) -> None:
        error = response_json["error"]
        assert isinstance(error, dict)
        self.code = code
        self.status = error.get("status")
        self.message = error.get("message")
        self.details = response_json
        super().__init__(f"{code} {self.status}. {response_json}")


def refusal(code: int, status: str, *details: dict[str, object]) -> ClientError:
    return ClientError(
        code,
        {
            "error": {
                "code": code,
                "message": f"Refused for project {STUDENT_PROJECT}.",
                "status": status,
                "details": list(details),
            }
        },
    )


def error_info(reason: str, **metadata: str) -> dict[str, object]:
    return {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": reason,
        "domain": "aiplatform.googleapis.com",
        "metadata": {"consumer": f"projects/{STUDENT_PROJECT}", **metadata},
    }


PREFLIGHT_REFUSALS = [
    pytest.param(
        lambda: refusal(
            403,
            "PERMISSION_DENIED",
            error_info("IAM_PERMISSION_DENIED", permission="aiplatform.endpoints.predict"),
        ),
        "403 PERMISSION_DENIED, IAM_PERMISSION_DENIED, permission aiplatform.endpoints.predict",
        id="missing-iam-grant",
    ),
    pytest.param(
        lambda: refusal(
            403,
            "PERMISSION_DENIED",
            error_info("SERVICE_DISABLED", service="aiplatform.googleapis.com"),
        ),
        "403 PERMISSION_DENIED, SERVICE_DISABLED",
        id="api-disabled",
    ),
    pytest.param(
        lambda: refusal(429, "RESOURCE_EXHAUSTED"),
        "429 RESOURCE_EXHAUSTED",
        id="quota",
    ),
    pytest.param(
        lambda: RuntimeError(f"connection reset talking to {STUDENT_PROJECT}"),
        "RuntimeError",
        id="not-an-api-error",
    ),
]


class RefusingModels(PreflightModels):
    def __init__(self, error: Exception, *, refuse_text: bool) -> None:
        super().__init__()
        self.error = error
        self.refuse_text = refuse_text

    def generate_content(self, *, contents: object, **kwargs: object) -> object:
        if isinstance(contents, str) == self.refuse_text:
            self.calls += 1
            raise self.error
        return super().generate_content(contents=contents, **kwargs)


def preflight_with(models: PreflightModels) -> object:
    class FakeGenAI:
        @staticmethod
        def Client(**kwargs: object) -> object:
            del kwargs
            return SimpleNamespace(models=models)

    return run_vertex_preflight(
        VertexProviderSettings(model_id="course-model"),
        samples=1,
        environment={"GOOGLE_CLOUD_PROJECT": STUDENT_PROJECT},
        sdk_loader=lambda: (FakeGenAI, FakeTypes),
        credentials_loader=lambda: (object(), None),
    )


@pytest.mark.parametrize(("make_error", "reason"), PREFLIGHT_REFUSALS)
def test_preflight_says_why_vertex_refused_the_text_request(
    make_error: object,
    reason: str,
) -> None:
    report = preflight_with(RefusingModels(make_error(), refuse_text=True))  # type: ignore[operator]

    assert report.successful is False  # type: ignore[attr-defined]
    failed = report.checks[-1]  # type: ignore[attr-defined]
    assert failed.name == "text_round_trip"
    assert failed.passed is False
    assert failed.detail == f"Vertex text request failed ({reason})"
    assert STUDENT_PROJECT not in failed.detail


@pytest.mark.parametrize(("make_error", "reason"), PREFLIGHT_REFUSALS[:3])
def test_preflight_says_why_vertex_refused_the_image_request(
    make_error: object,
    reason: str,
) -> None:
    report = preflight_with(RefusingModels(make_error(), refuse_text=False))  # type: ignore[operator]

    assert report.successful is False  # type: ignore[attr-defined]
    failed = report.checks[-1]  # type: ignore[attr-defined]
    assert failed.name == "image_schema_round_trip"
    assert failed.passed is False
    assert failed.detail == f"Vertex image/schema request failed ({reason})"
    assert STUDENT_PROJECT not in failed.detail


def test_preflight_keeps_naming_an_image_failure_that_is_not_a_refusal() -> None:
    report = preflight_with(
        RefusingModels(RuntimeError(f"reset by {STUDENT_PROJECT}"), refuse_text=False)
    )

    assert report.checks[-1].detail == (  # type: ignore[attr-defined]
        "Vertex image/schema request failed (ModelProviderError)"
    )
