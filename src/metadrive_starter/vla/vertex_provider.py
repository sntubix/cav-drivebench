from __future__ import annotations

import math
import time
from collections.abc import Callable
from threading import Lock
from typing import Any

from metadrive_starter.vla.assessment import vertex_vla_assessment_schema
from metadrive_starter.vla.http_provider import encode_rgb_frame_png
from metadrive_starter.vla.provider import (
    ModelFailureCategory,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelResponseMetadata,
    ModelTimeoutError,
)


SDKLoader = Callable[[], tuple[Any, Any]]


class VertexModelProvider:
    """Gemini on Vertex AI behind DriveBench's transport-neutral model boundary."""

    def __init__(
        self,
        project_id: str,
        location: str,
        model_id: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        thinking_budget: int | None = 0,
        client: Any | None = None,
        types_module: Any | None = None,
        sdk_loader: SDKLoader | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in {
            "project_id": project_id,
            "location": location,
            "model_id": model_id,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0.0 <= temperature <= 2.0
        ):
            raise ValueError("temperature must be finite and between 0 and 2")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if thinking_budget is not None and (
            isinstance(thinking_budget, bool)
            or not isinstance(thinking_budget, int)
            or thinking_budget < 0
        ):
            raise ValueError("thinking_budget must be a non-negative integer or None")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if (client is None) != (types_module is None):
            raise ValueError("client and types_module must be supplied together")

        self.project_id = project_id.strip()
        self.location = location.strip()
        self.model_id = model_id.strip()
        self.temperature = float(temperature)
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self._client = client
        self._types = types_module
        self._sdk_loader = sdk_loader or _load_vertex_sdk
        self._clock = clock
        self._client_timeout_s: float | None = None
        self._client_lock = Lock()
        self._injected_client = client is not None

    def generate(
        self,
        request: ModelRequest,
        *,
        timeout_s: float,
    ) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise ValueError("request must be a ModelRequest")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
        ):
            raise ValueError("timeout_s must be finite and positive")

        try:
            client, types = self._client_and_types(float(timeout_s))
            png_bytes = request.encoded_png_bytes or encode_rgb_frame_png(request.frame)
            parts = [
                types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                types.Part.from_text(text=request.prompt),
            ]
            config_values: dict[str, Any] = {
                "response_mime_type": "application/json",
                "response_schema": vertex_vla_response_schema(),
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            }
            if self.thinking_budget is not None:
                config_values["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
            config = types.GenerateContentConfig(**config_values)
            contents = [types.Content(role="user", parts=parts)]

            started_at = self._clock()
            response = client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )
            latency_s = self._clock() - started_at
        except ModelProviderError:
            raise
        except Exception as exc:
            if _is_timeout_error(exc):
                raise ModelTimeoutError("Vertex model request timed out") from exc
            raise ModelProviderError(
                f"Vertex model request failed ({type(exc).__name__})",
                category=_vertex_failure_category(exc),
            ) from exc

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ModelProviderError(
                "Vertex model returned an empty response",
                category=ModelFailureCategory.INVALID_RESPONSE,
            )
        return ModelResponse(
            request_id=request.request_id,
            text=text.strip(),
            model_id=self.model_id,
            latency_s=max(0.0, latency_s),
            metadata=_vertex_response_metadata(response),
        )

    def _client_and_types(self, timeout_s: float) -> tuple[Any, Any]:
        with self._client_lock:
            if self._injected_client:
                assert self._client is not None
                assert self._types is not None
                return self._client, self._types
            if self._client is not None and self._client_timeout_s == timeout_s:
                assert self._types is not None
                return self._client, self._types

            try:
                genai, types = self._sdk_loader()
            except ImportError as exc:
                raise ModelProviderError(
                    "Vertex support requires the cloud extra; run `uv sync --extra cloud`"
                ) from exc
            self._types = types
            self._client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
                http_options=types.HttpOptions(
                    timeout=int(timeout_s * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            self._client_timeout_s = timeout_s
            return self._client, types


def vertex_vla_response_schema() -> dict[str, object]:
    """Vertex schema for DriveBench's model-facing assessment."""

    return vertex_vla_assessment_schema()


def _load_vertex_sdk() -> tuple[Any, Any]:
    from google import genai
    from google.genai import types

    return genai, types


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or "deadline" in name:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if callable(status):
        try:
            status = status()
        except Exception:
            status = None
    return status in {408, 504, "408", "504", "DEADLINE_EXCEEDED"}


def _vertex_failure_category(exc: Exception) -> ModelFailureCategory:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if callable(status):
        try:
            status = status()
        except Exception:
            status = None
    normalized = str(status).upper()
    if status in {401, 403, "401", "403"} or normalized in {
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
    }:
        return ModelFailureCategory.AUTHENTICATION
    if status in {404, "404"} or normalized == "NOT_FOUND":
        return ModelFailureCategory.MODEL_UNAVAILABLE
    if status in {429, "429"} or normalized == "RESOURCE_EXHAUSTED":
        return ModelFailureCategory.QUOTA
    return ModelFailureCategory.TRANSPORT


def _vertex_response_metadata(response: object) -> ModelResponseMetadata:
    usage = getattr(response, "usage_metadata", None)
    return ModelResponseMetadata(
        provider="vertex",
        response_id=_optional_text(getattr(response, "response_id", None)),
        model_version=_optional_text(getattr(response, "model_version", None)),
        input_tokens=_optional_token_count(usage, "prompt_token_count"),
        output_tokens=_optional_token_count(usage, "candidates_token_count"),
        total_tokens=_optional_token_count(usage, "total_token_count"),
    )


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_token_count(usage: object, name: str) -> int | None:
    value = getattr(usage, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
