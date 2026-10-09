from __future__ import annotations

import pytest

from metadrive_starter.faults import (
    FaultInjectingModelProvider,
    FaultKind,
    FaultSchedule,
    FaultSpec,
)
from metadrive_starter.vla import (
    ModelProviderError,
    ModelRequest,
    ModelTimeoutError,
    RGBFrame,
    ScriptedModelProvider,
)


def _request(request_id: str = "request-1") -> ModelRequest:
    return ModelRequest(
        request_id=request_id,
        prompt="Describe the scene",
        frame=RGBFrame(
            width=1,
            height=1,
            rgb_bytes=b"\x00\x00\x00",
            timestamp_s=0.0,
        ),
        created_at_s=0.0,
    )


def test_fault_schedule_reports_half_open_intervals_and_edges() -> None:
    fault = FaultSpec("camera-1", "camera_dropout", start_step=2, duration_steps=2)
    schedule = FaultSchedule((fault,))

    assert schedule.advance(0) == ((), ())
    active, transitions = schedule.advance(2)
    assert active == (fault,)
    assert transitions[0].fault == fault
    assert transitions[0].active is True
    assert schedule.advance(3) == ((fault,), ())
    active, transitions = schedule.advance(4)
    assert active == ()
    assert transitions[0].active is False


def test_fault_schedule_rejects_duplicate_ids_and_non_monotonic_steps() -> None:
    fault = FaultSpec("duplicate", "camera_dropout", 0, 1)
    with pytest.raises(ValueError, match="unique"):
        FaultSchedule((fault, fault))

    schedule = FaultSchedule()
    schedule.advance(1)
    with pytest.raises(ValueError, match="increase strictly"):
        schedule.advance(1)


@pytest.mark.parametrize(
    ("kind", "error_type"),
    [
        (FaultKind.PROVIDER_TIMEOUT, ModelTimeoutError),
        (FaultKind.PROVIDER_ERROR, ModelProviderError),
    ],
)
def test_provider_adapter_injects_failures_for_registered_request_only(
    kind: FaultKind,
    error_type: type[Exception],
) -> None:
    provider = ScriptedModelProvider(("good",))
    adapter = FaultInjectingModelProvider(provider)
    request = _request()
    adapter.register(request.request_id, (FaultSpec("provider-1", kind, 0, 1),))

    with pytest.raises(error_type, match="injected"):
        adapter.generate(request, timeout_s=1.0)

    response = adapter.generate(request, timeout_s=1.0)
    assert response.text == "good"


def test_provider_adapter_injects_latency_and_malformed_output() -> None:
    sleeps: list[float] = []
    provider = ScriptedModelProvider(("valid JSON",))
    adapter = FaultInjectingModelProvider(provider, sleeper=sleeps.append)
    request = _request()
    adapter.register(
        request.request_id,
        (
            FaultSpec("latency-1", "provider_latency", 0, 1, latency_s=0.2),
            FaultSpec("malformed-1", "malformed_output", 0, 1),
        ),
    )

    response = adapter.generate(request, timeout_s=1.0)

    assert sleeps == [0.2]
    assert response.text == "{injected malformed output"
    assert response.latency_s == pytest.approx(0.2)


def test_provider_latency_cannot_exceed_timeout_silently() -> None:
    sleeps: list[float] = []
    adapter = FaultInjectingModelProvider(
        ScriptedModelProvider(("unused",)), sleeper=sleeps.append
    )
    request = _request()
    adapter.register(
        request.request_id,
        (FaultSpec("latency-1", "provider_latency", 0, 1, latency_s=2.0),),
    )

    with pytest.raises(ModelTimeoutError, match="latency"):
        adapter.generate(request, timeout_s=0.5)

    assert sleeps == [0.5]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fault_id": "bad id", "kind": "camera_dropout", "start_step": 0, "duration_steps": 1},
        {"fault_id": "ok", "kind": "unknown", "start_step": 0, "duration_steps": 1},
        {"fault_id": "ok", "kind": "camera_dropout", "start_step": -1, "duration_steps": 1},
        {"fault_id": "ok", "kind": "camera_dropout", "start_step": 0, "duration_steps": 0},
        {"fault_id": "ok", "kind": "provider_latency", "start_step": 0, "duration_steps": 1},
    ],
)
def test_fault_spec_rejects_invalid_contracts(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FaultSpec(**kwargs)  # type: ignore[arg-type]
