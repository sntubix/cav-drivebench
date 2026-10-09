from __future__ import annotations

from collections.abc import Mapping

import pytest

from metadrive_starter.config import config_from_dict
from metadrive_starter.faults import FaultInjectingModelProvider
from metadrive_starter.vla import (
    HTTPTransportResponse,
    OpenAICompatibleHTTPProvider,
    RecordedModelProvider,
    RequestCappedModelProvider,
    VLAInferenceScheduler,
    VertexModelProvider,
)
from metadrive_starter.vla_runtime import (
    apply_vla_request_cap,
    build_fixture_vla_provider,
    build_http_vla_provider,
    build_http_vla_scheduler,
    build_vertex_vla_provider,
    build_vla_provider,
    build_vla_scheduler,
)


class RecordingTransport:
    def __init__(self) -> None:
        self.headers: Mapping[str, str] | None = None

    def post(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_s: float,
        max_response_bytes: int,
    ) -> HTTPTransportResponse:
        del url, body, timeout_s, max_response_bytes
        self.headers = dict(headers)
        return HTTPTransportResponse(status_code=200, body=b"{}")


def _settings(
    *,
    enabled: bool = True,
    api_key_env: str | None = None,
    maximum_requests_per_run: int | None = None,
):
    return config_from_dict(
        {
            "vla": {
                "enabled": enabled,
                "maximum_requests_per_run": maximum_requests_per_run,
                "request_timeout_s": 4.0,
                "minimum_interval_s": 0.25,
                "http": {
                    "endpoint_url": "https://models.example/v1/chat/completions",
                    "model_id": "demo-vlm",
                    "api_key_env": api_key_env,
                    "max_response_bytes": 2048,
                    "max_tokens": 64,
                    "structured_output": True,
                },
            }
        }
    ).vla


def test_build_http_vla_scheduler_composes_all_configured_bounds() -> None:
    transport = RecordingTransport()
    scheduler = build_http_vla_scheduler(
        _settings(api_key_env="DEMO_KEY"),
        environment={"DEMO_KEY": "secret-value"},
        transport=transport,
    )
    try:
        assert isinstance(scheduler, VLAInferenceScheduler)
        assert scheduler.minimum_interval_s == 0.25
        assert scheduler.pipeline.timeout_s == 4.0
        assert scheduler.pipeline.provider.model_id == "demo-vlm"
        assert scheduler.pipeline.provider.max_response_bytes == 2048
        assert scheduler.pipeline.provider.max_tokens == 64
        assert scheduler.pipeline.provider.structured_output is True
        assert "secret-value" not in repr(scheduler.pipeline.provider)
    finally:
        scheduler.close()


def test_generic_scheduler_can_wrap_provider_for_request_scoped_faults() -> None:
    scheduler = build_vla_scheduler(
        _settings(),
        transport=RecordingTransport(),
        inject_faults=True,
    )
    try:
        assert isinstance(scheduler.pipeline.provider, FaultInjectingModelProvider)
        assert isinstance(
            scheduler.pipeline.provider.provider, OpenAICompatibleHTTPProvider
        )
    finally:
        scheduler.close()


def test_scheduler_composes_one_shared_per_run_request_cap() -> None:
    scheduler = build_vla_scheduler(
        _settings(maximum_requests_per_run=2),
        transport=RecordingTransport(),
        inject_faults=True,
    )
    try:
        assert scheduler.request_budget is not None
        assert scheduler.request_budget.maximum_requests == 2
        assert isinstance(scheduler.pipeline.provider, FaultInjectingModelProvider)
        assert isinstance(
            scheduler.pipeline.provider.provider,
            RequestCappedModelProvider,
        )
        assert (
            scheduler.pipeline.provider.provider.budget
            is scheduler.request_budget
        )
    finally:
        scheduler.close()


def test_open_loop_provider_can_apply_same_configured_request_cap() -> None:
    underlying = build_http_vla_provider(
        _settings(maximum_requests_per_run=2),
        transport=RecordingTransport(),
    )

    provider, budget = apply_vla_request_cap(
        _settings(maximum_requests_per_run=2),
        underlying,
    )

    assert isinstance(provider, RequestCappedModelProvider)
    assert budget is not None
    assert budget.maximum_requests == 2
    assert provider.budget is budget


def test_build_http_vla_provider_resolves_secret_without_storing_it_in_settings() -> None:
    provider = build_http_vla_provider(
        _settings(api_key_env="DEMO_KEY"),
        environment={"DEMO_KEY": "secret-value"},
        transport=RecordingTransport(),
    )

    assert isinstance(provider, OpenAICompatibleHTTPProvider)
    assert provider.model_id == "demo-vlm"
    assert provider.structured_output is True
    assert "secret-value" not in repr(provider)


def test_build_http_vla_scheduler_requires_enabled_configuration() -> None:
    with pytest.raises(ValueError, match="enabled"):
        build_http_vla_scheduler(_settings(enabled=False), transport=RecordingTransport())


def test_build_http_vla_scheduler_requires_named_secret() -> None:
    with pytest.raises(ValueError, match="DEMO_KEY"):
        build_http_vla_scheduler(
            _settings(api_key_env="DEMO_KEY"),
            environment={},
            transport=RecordingTransport(),
        )


def test_build_vertex_provider_resolves_project_from_named_environment() -> None:
    settings = config_from_dict(
        {
            "vla": {
                "enabled": True,
                "provider": "vertex",
                "vertex": {
                    "project_env": "COURSE_PROJECT",
                    "location": "global",
                    "model_id": "course-model",
                },
            }
        }
    ).vla
    client = object()
    types_module = object()

    provider = build_vertex_vla_provider(
        settings,
        environment={"COURSE_PROJECT": "project-123"},
        client=client,
        types_module=types_module,
    )

    assert isinstance(provider, VertexModelProvider)
    assert provider.project_id == "project-123"
    assert provider.location == "global"
    assert provider.model_id == "course-model"


def test_generic_provider_builder_dispatches_to_vertex() -> None:
    settings = config_from_dict(
        {"vla": {"enabled": True, "provider": "vertex"}}
    ).vla

    provider = build_vla_provider(
        settings,
        environment={"GOOGLE_CLOUD_PROJECT": "project-123"},
        vertex_client=object(),
        vertex_types_module=object(),
    )

    assert isinstance(provider, VertexModelProvider)


@pytest.mark.needs("fixtures")
def test_generic_provider_builder_dispatches_to_offline_fixture() -> None:
    settings = config_from_dict(
        {
            "vla": {
                "enabled": True,
                "provider": "fixture",
                "fixture": {
                    "path": "fixtures/vla/providers/synthetic-keep-lane.json",
                    "repeat_last": True,
                },
            }
        }
    ).vla

    provider = build_vla_provider(settings, environment={})

    assert isinstance(provider, RecordedModelProvider)
    assert provider.fixture_ids == ("synthetic-keep-lane",)


def test_fixture_provider_rejects_wrong_provider_selection() -> None:
    with pytest.raises(ValueError, match="fixture"):
        build_fixture_vla_provider(_settings())


def test_vertex_provider_requires_named_project_environment() -> None:
    settings = config_from_dict(
        {"vla": {"enabled": True, "provider": "vertex"}}
    ).vla

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        build_vertex_vla_provider(settings, environment={})
