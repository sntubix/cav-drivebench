from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, replace
from enum import Enum
from threading import Lock

from metadrive_starter.vla.provider import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelTimeoutError,
)


class FaultKind(str, Enum):
    """Faults that can be scheduled by simulation step."""

    CAMERA_DROPOUT = "camera_dropout"
    LIDAR_DROPOUT = "lidar_dropout"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_LATENCY = "provider_latency"
    MALFORMED_OUTPUT = "malformed_output"


_PROVIDER_FAULTS = {
    FaultKind.PROVIDER_TIMEOUT,
    FaultKind.PROVIDER_ERROR,
    FaultKind.PROVIDER_LATENCY,
    FaultKind.MALFORMED_OUTPUT,
}


@dataclass(frozen=True)
class FaultSpec:
    """One deterministic half-open fault interval: [start, start + duration)."""

    fault_id: str
    kind: FaultKind | str
    start_step: int
    duration_steps: int
    latency_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.fault_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.fault_id
        ) is None:
            raise ValueError(
                "fault_id must contain only letters, numbers, '.', '_', or '-'"
            )
        try:
            kind = FaultKind(self.kind)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in FaultKind)
            raise ValueError(f"fault kind must be one of: {allowed}") from exc
        object.__setattr__(self, "kind", kind)
        if (
            isinstance(self.start_step, bool)
            or not isinstance(self.start_step, int)
            or self.start_step < 0
        ):
            raise ValueError("fault start_step must be a non-negative integer")
        if (
            isinstance(self.duration_steps, bool)
            or not isinstance(self.duration_steps, int)
            or self.duration_steps <= 0
        ):
            raise ValueError("fault duration_steps must be a positive integer")
        if (
            isinstance(self.latency_s, bool)
            or not isinstance(self.latency_s, (int, float))
            or not math.isfinite(self.latency_s)
            or self.latency_s < 0.0
        ):
            raise ValueError("fault latency_s must be finite and non-negative")
        if kind is FaultKind.PROVIDER_LATENCY and self.latency_s <= 0.0:
            raise ValueError("provider_latency requires a positive latency_s")
        if kind is not FaultKind.PROVIDER_LATENCY and self.latency_s != 0.0:
            raise ValueError("latency_s is only valid for provider_latency")

    @property
    def end_step(self) -> int:
        return self.start_step + self.duration_steps

    def active_at(self, step: int) -> bool:
        return self.start_step <= step < self.end_step


@dataclass(frozen=True)
class FaultTransition:
    fault: FaultSpec
    active: bool


class FaultSchedule:
    """Stateful schedule that reports active faults and edge transitions."""

    def __init__(self, faults: tuple[FaultSpec, ...] | list[FaultSpec] = ()) -> None:
        if any(not isinstance(fault, FaultSpec) for fault in faults):
            raise TypeError("fault schedule entries must be FaultSpec values")
        ids = [fault.fault_id for fault in faults]
        if len(set(ids)) != len(ids):
            raise ValueError("fault_id values must be unique within a schedule")
        self.faults = tuple(
            sorted(faults, key=lambda fault: (fault.start_step, fault.fault_id))
        )
        self._active_ids: frozenset[str] = frozenset()
        self._last_step: int | None = None

    def advance(
        self, step: int
    ) -> tuple[tuple[FaultSpec, ...], tuple[FaultTransition, ...]]:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("fault schedule step must be a non-negative integer")
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("fault schedule steps must increase strictly")
        active = tuple(fault for fault in self.faults if fault.active_at(step))
        active_ids = frozenset(fault.fault_id for fault in active)
        transitions = tuple(
            FaultTransition(fault, fault.fault_id in active_ids)
            for fault in self.faults
            if (fault.fault_id in active_ids) != (fault.fault_id in self._active_ids)
        )
        self._active_ids = active_ids
        self._last_step = step
        return active, transitions


class FaultInjectingModelProvider:
    """Per-request provider fault adapter used by the non-blocking scheduler."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        sleeper=time.sleep,
    ) -> None:
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must implement ModelProvider.generate()")
        if not callable(sleeper):
            raise TypeError("sleeper must be callable")
        self.provider = provider
        self._sleeper = sleeper
        self._faults_by_request: dict[str, tuple[FaultSpec, ...]] = {}
        self._lock = Lock()

    def register(self, request_id: str, faults: tuple[FaultSpec, ...]) -> None:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must not be empty")
        if not isinstance(faults, tuple) or any(
            not isinstance(fault, FaultSpec) for fault in faults
        ):
            raise TypeError("faults must be a tuple of FaultSpec values")
        selected = tuple(fault for fault in faults if fault.kind in _PROVIDER_FAULTS)
        with self._lock:
            if request_id in self._faults_by_request:
                raise ValueError(f"faults already registered for request {request_id!r}")
            self._faults_by_request[request_id] = selected

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._faults_by_request.pop(request_id, None)

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
        with self._lock:
            faults = self._faults_by_request.pop(request.request_id, ())

        kinds = {fault.kind for fault in faults}
        if FaultKind.PROVIDER_TIMEOUT in kinds:
            raise ModelTimeoutError("injected provider timeout")
        if FaultKind.PROVIDER_ERROR in kinds:
            raise ModelProviderError("injected provider error")

        injected_latency_s = sum(
            fault.latency_s
            for fault in faults
            if fault.kind is FaultKind.PROVIDER_LATENCY
        )
        remaining_timeout_s = timeout_s
        if injected_latency_s:
            self._sleeper(min(injected_latency_s, timeout_s))
            if injected_latency_s >= timeout_s:
                raise ModelTimeoutError("injected provider latency exceeded timeout")
            remaining_timeout_s -= injected_latency_s

        response = self.provider.generate(request, timeout_s=remaining_timeout_s)
        if FaultKind.MALFORMED_OUTPUT in kinds:
            response = replace(response, text="{injected malformed output")
        if injected_latency_s:
            response = replace(
                response,
                latency_s=response.latency_s + injected_latency_s,
            )
        return response


def has_fault(faults: tuple[FaultSpec, ...], kind: FaultKind) -> bool:
    return any(fault.kind is kind for fault in faults)


def has_provider_faults(faults: tuple[FaultSpec, ...]) -> bool:
    return any(fault.kind in _PROVIDER_FAULTS for fault in faults)
