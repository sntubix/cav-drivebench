from __future__ import annotations

import math
import os
import statistics
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from metadrive_starter.config import VertexProviderSettings
from metadrive_starter.vla import (
    ModelRequest,
    RGBFrame,
    RequestBudget,
    VertexModelProvider,
    build_vla_prompt,
    decode_vla_assessment,
)


SDKLoader = Callable[[], tuple[Any, Any]]
CredentialsLoader = Callable[[], tuple[Any, str | None]]


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class VertexPreflightReport:
    project_id: str | None
    location: str
    model_id: str
    checks: tuple[PreflightCheck, ...]
    latency_samples_s: tuple[float, ...] = ()
    suggested_minimum_interval_s: float | None = None
    suggested_maximum_command_age_s: float | None = None

    @property
    def successful(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "successful": self.successful,
            "project_id": self.project_id,
            "location": self.location,
            "model_id": self.model_id,
            "checks": [asdict(check) for check in self.checks],
            "latency_samples_s": list(self.latency_samples_s),
            "suggested_minimum_interval_s": self.suggested_minimum_interval_s,
            "suggested_maximum_command_age_s": self.suggested_maximum_command_age_s,
        }


def run_vertex_preflight(
    settings: VertexProviderSettings,
    *,
    project_id: str | None = None,
    location: str | None = None,
    model_id: str | None = None,
    samples: int = 3,
    request_timeout_s: float = 15.0,
    environment: Mapping[str, str] | None = None,
    sdk_loader: SDKLoader | None = None,
    credentials_loader: CredentialsLoader | None = None,
    request_budget: RequestBudget | None = None,
    prompt_policy: str = "",
) -> VertexPreflightReport:
    """Check Vertex credentials, connectivity, image input, schema, and latency."""
    if not isinstance(settings, VertexProviderSettings):
        raise TypeError("settings must be VertexProviderSettings")
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 20:
        raise ValueError("samples must be an integer between 1 and 20")
    if (
        isinstance(request_timeout_s, bool)
        or not isinstance(request_timeout_s, (int, float))
        or not math.isfinite(request_timeout_s)
        or request_timeout_s <= 0.0
    ):
        raise ValueError("request_timeout_s must be finite and positive")
    if request_budget is not None and not isinstance(request_budget, RequestBudget):
        raise TypeError("request_budget must be a RequestBudget or None")

    effective_location = location or settings.location
    effective_model = model_id or settings.model_id
    source = os.environ if environment is None else environment
    checks: list[PreflightCheck] = []
    load_sdk = sdk_loader or _load_vertex_sdk
    load_credentials = credentials_loader or _load_default_credentials

    try:
        genai, types = load_sdk()
    except Exception:
        checks.append(
            PreflightCheck(
                "sdk",
                False,
                "google-genai is unavailable; run `uv sync --extra cloud`",
            )
        )
        return _report(None, effective_location, effective_model, checks)
    checks.append(PreflightCheck("sdk", True, "google-genai import succeeded"))

    try:
        _, detected_project = load_credentials()
    except Exception:
        checks.append(
            PreflightCheck(
                "credentials",
                False,
                "Application Default Credentials unavailable; run "
                "`gcloud auth application-default login`",
            )
        )
        return _report(None, effective_location, effective_model, checks)
    checks.append(
        PreflightCheck("credentials", True, "Application Default Credentials found")
    )

    effective_project = (
        project_id or source.get(settings.project_env) or detected_project
    )
    if not isinstance(effective_project, str) or not effective_project.strip():
        checks.append(
            PreflightCheck(
                "configuration",
                False,
                f"project missing; pass --project or set {settings.project_env}",
            )
        )
        return _report(None, effective_location, effective_model, checks)
    effective_project = effective_project.strip()
    checks.append(
        PreflightCheck(
            "configuration",
            True,
            f"project={effective_project} location={effective_location} model={effective_model}",
        )
    )

    try:
        client = genai.Client(
            vertexai=True,
            project=effective_project,
            location=effective_location,
            http_options=types.HttpOptions(
                timeout=int(request_timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        if request_budget is not None:
            request_budget.consume("vertex-preflight-text")
        response = client.models.generate_content(
            model=effective_model,
            contents="Reply with the single word: ready",
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            ),
        )
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("model returned no text")
    except Exception as exc:
        checks.append(
            PreflightCheck(
                "text_round_trip",
                False,
                f"Vertex text request failed ({type(exc).__name__})",
            )
        )
        return _report(effective_project, effective_location, effective_model, checks)
    checks.append(PreflightCheck("text_round_trip", True, "Vertex text request succeeded"))

    provider = VertexModelProvider(
        effective_project,
        effective_location,
        effective_model,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        thinking_budget=settings.thinking_budget,
        client=client,
        types_module=types,
    )
    prompt = build_vla_prompt(
        now_s=0.0,
        action_horizon_s=2.0,
        ego_speed_mps=9.0,
        cruise_speed_mps=10.0,
        prompt_policy=prompt_policy,
    )
    frame = _synthetic_road_frame()
    latencies: list[float] = []
    try:
        for sample in range(samples):
            if request_budget is not None:
                request_budget.consume(f"vertex-preflight-image-{sample + 1}")
            response = provider.generate(
                ModelRequest(
                    request_id=f"vertex-preflight-{sample + 1}",
                    prompt=prompt,
                    frame=frame,
                    created_at_s=0.0,
                ),
                timeout_s=request_timeout_s,
            )
            decode_vla_assessment(response.text)
            latencies.append(response.latency_s)
    except Exception as exc:
        checks.append(
            PreflightCheck(
                "image_schema_round_trip",
                False,
                f"Vertex image/schema request failed ({type(exc).__name__})",
            )
        )
        return _report(effective_project, effective_location, effective_model, checks)
    checks.append(
        PreflightCheck(
            "image_schema_round_trip",
            True,
            f"{samples} image/schema request(s) decoded",
        )
    )

    median_s = statistics.median(latencies)
    maximum_s = max(latencies)
    minimum_interval_s = round(max(0.5, median_s * 1.5), 3)
    maximum_command_age_s = round(minimum_interval_s + 2.0 * median_s + 1.0, 3)
    checks.append(
        PreflightCheck(
            "latency",
            True,
            f"median={median_s:.3f}s maximum={maximum_s:.3f}s",
        )
    )
    return VertexPreflightReport(
        project_id=effective_project,
        location=effective_location,
        model_id=effective_model,
        checks=tuple(checks),
        latency_samples_s=tuple(round(value, 6) for value in latencies),
        suggested_minimum_interval_s=minimum_interval_s,
        suggested_maximum_command_age_s=maximum_command_age_s,
    )


def _report(
    project_id: str | None,
    location: str,
    model_id: str,
    checks: list[PreflightCheck],
) -> VertexPreflightReport:
    return VertexPreflightReport(project_id, location, model_id, tuple(checks))


def _synthetic_road_frame() -> RGBFrame:
    width, height = 64, 36
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            if y < height // 2:
                pixel = (135, 206, 235)
            elif width // 2 - 1 <= x <= width // 2 + 1:
                pixel = (255, 255, 255)
            else:
                pixel = (70, 70, 70)
            pixels.extend(pixel)
    return RGBFrame(0.0, width, height, bytes(pixels))


def _load_vertex_sdk() -> tuple[Any, Any]:
    from google import genai
    from google.genai import types

    return genai, types


def _load_default_credentials() -> tuple[Any, str | None]:
    import google.auth

    return google.auth.default()
