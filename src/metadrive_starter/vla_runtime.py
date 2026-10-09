from __future__ import annotations

import os
from collections.abc import Mapping

from metadrive_starter.config import VLASettings
from metadrive_starter.faults import FaultInjectingModelProvider
from metadrive_starter.vla import (
    HTTPTransport,
    ModelProvider,
    ObservationBuilder,
    OpenAICompatibleHTTPProvider,
    RecordedModelProvider,
    RequestBudget,
    RequestCappedModelProvider,
    VLAInferencePipeline,
    VLAInferenceScheduler,
    VertexModelProvider,
)


def build_vla_scheduler(
    settings: VLASettings,
    *,
    environment: Mapping[str, str] | None = None,
    transport: HTTPTransport | None = None,
    vertex_client: object | None = None,
    vertex_types_module: object | None = None,
    inject_faults: bool = False,
    observation_builder: ObservationBuilder | None = None,
) -> VLAInferenceScheduler:
    """Compose the selected model provider, inference pipeline, and scheduler.

    Without an observation builder the pipeline builds DriveBench's original
    observation.
    """
    provider = build_vla_provider(
        settings,
        environment=environment,
        transport=transport,
        vertex_client=vertex_client,
        vertex_types_module=vertex_types_module,
    )
    provider, request_budget = apply_vla_request_cap(settings, provider)
    if inject_faults:
        provider = FaultInjectingModelProvider(provider)
    return _build_scheduler(
        settings,
        provider,
        request_budget=request_budget,
        observation_builder=observation_builder,
    )


def build_vla_provider(
    settings: VLASettings,
    *,
    environment: Mapping[str, str] | None = None,
    transport: HTTPTransport | None = None,
    vertex_client: object | None = None,
    vertex_types_module: object | None = None,
) -> ModelProvider:
    """Build the provider selected by ``vla.provider``."""
    if not isinstance(settings, VLASettings):
        raise TypeError("settings must be VLASettings")
    if settings.provider == "http":
        if vertex_client is not None or vertex_types_module is not None:
            raise ValueError("Vertex client overrides require vla.provider='vertex'")
        return build_http_vla_provider(
            settings,
            environment=environment,
            transport=transport,
        )
    if settings.provider == "fixture":
        if transport is not None:
            raise ValueError("HTTP transport override requires vla.provider='http'")
        if vertex_client is not None or vertex_types_module is not None:
            raise ValueError("Vertex client overrides require vla.provider='vertex'")
        return build_fixture_vla_provider(settings)
    if transport is not None:
        raise ValueError("HTTP transport override requires vla.provider='http'")
    return build_vertex_vla_provider(
        settings,
        environment=environment,
        client=vertex_client,
        types_module=vertex_types_module,
    )


def build_fixture_vla_provider(settings: VLASettings) -> RecordedModelProvider:
    """Build a deterministic offline provider from integrity-checked fixtures."""

    if not isinstance(settings, VLASettings):
        raise TypeError("settings must be VLASettings")
    if not settings.enabled:
        raise ValueError("vla.enabled must be true before building the VLA provider")
    if settings.provider != "fixture":
        raise ValueError(
            "vla.provider must be 'fixture' before building a fixture provider"
        )
    return RecordedModelProvider.from_path(
        settings.fixture.path,
        repeat_last=settings.fixture.repeat_last,
    )


def build_http_vla_scheduler(
    settings: VLASettings,
    *,
    environment: Mapping[str, str] | None = None,
    transport: HTTPTransport | None = None,
) -> VLAInferenceScheduler:
    """Compose the configured HTTP provider, inference pipeline, and scheduler.

    Secrets are resolved from the named environment variable at composition
    time. They are never stored in YAML or copied into the typed settings.
    """
    provider = build_http_vla_provider(
        settings,
        environment=environment,
        transport=transport,
    )
    provider, request_budget = apply_vla_request_cap(settings, provider)

    return _build_scheduler(
        settings,
        provider,
        request_budget=request_budget,
    )


def build_http_vla_provider(
    settings: VLASettings,
    *,
    environment: Mapping[str, str] | None = None,
    transport: HTTPTransport | None = None,
) -> OpenAICompatibleHTTPProvider:
    """Build the configured HTTP provider and resolve its optional API key."""
    if not isinstance(settings, VLASettings):
        raise TypeError("settings must be VLASettings")
    if not settings.enabled:
        raise ValueError("vla.enabled must be true before building the VLA provider")

    source = os.environ if environment is None else environment
    api_key: str | None = None
    if settings.http.api_key_env is not None:
        api_key = source.get(settings.http.api_key_env)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                f"environment variable {settings.http.api_key_env!r} "
                "must contain the model API key"
            )

    return OpenAICompatibleHTTPProvider(
        settings.http.endpoint_url,
        settings.http.model_id,
        api_key=api_key,
        max_response_bytes=settings.http.max_response_bytes,
        max_tokens=settings.http.max_tokens,
        allow_insecure_http=settings.http.allow_insecure_http,
        structured_output=settings.http.structured_output,
        temperature=settings.http.temperature,
        seed=settings.http.seed,
        transport=transport,
    )


def build_vertex_vla_provider(
    settings: VLASettings,
    *,
    environment: Mapping[str, str] | None = None,
    client: object | None = None,
    types_module: object | None = None,
) -> VertexModelProvider:
    """Build a Vertex provider while resolving the project only from the environment."""
    if not isinstance(settings, VLASettings):
        raise TypeError("settings must be VLASettings")
    if not settings.enabled:
        raise ValueError("vla.enabled must be true before building the VLA provider")
    if settings.provider != "vertex":
        raise ValueError("vla.provider must be 'vertex' before building a Vertex provider")

    source = os.environ if environment is None else environment
    project_id = source.get(settings.vertex.project_env)
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError(
            f"environment variable {settings.vertex.project_env!r} "
            "must contain the GCP project ID"
        )
    return VertexModelProvider(
        project_id,
        settings.vertex.location,
        settings.vertex.model_id,
        temperature=settings.vertex.temperature,
        max_output_tokens=settings.vertex.max_output_tokens,
        thinking_budget=settings.vertex.thinking_budget,
        client=client,
        types_module=types_module,
    )


def _build_scheduler(
    settings: VLASettings,
    provider: ModelProvider,
    *,
    request_budget: RequestBudget | None = None,
    observation_builder: ObservationBuilder | None = None,
) -> VLAInferenceScheduler:
    pipeline = VLAInferencePipeline(
        provider,
        timeout_s=settings.request_timeout_s,
        action_horizon_s=settings.action_horizon_s,
        max_scene_objects=settings.max_scene_objects,
        maximum_frame_age_s=settings.maximum_frame_age_s,
        maximum_clock_skew_s=settings.maximum_clock_skew_s,
        prompt_policy=settings.prompt_policy,
        observation_builder=observation_builder,
    )
    return VLAInferenceScheduler(
        pipeline,
        minimum_interval_s=settings.minimum_interval_s,
        request_budget=request_budget,
    )


def apply_vla_request_cap(
    settings: VLASettings,
    provider: ModelProvider,
) -> tuple[ModelProvider, RequestBudget | None]:
    """Apply the configured hard request budget to any provider workflow."""

    if not isinstance(settings, VLASettings):
        raise TypeError("settings must be VLASettings")
    maximum_requests = settings.maximum_requests_per_run
    if maximum_requests is None:
        return provider, None
    if not isinstance(provider, ModelProvider):
        raise TypeError("provider must implement ModelProvider.generate()")
    budget = RequestBudget(maximum_requests)
    return RequestCappedModelProvider(provider, budget), budget
