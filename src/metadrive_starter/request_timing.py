"""A request policy driven on a scripted road, in simulated time.

The Assignment 3 timing notebook asks what a request policy does over a drive:
when it asks, how long each answer's command holds, and what it does through an
outage or quota refusals. A drive in MetaDrive answers that only at the
simulator's pace and, on a cloud model, at a cost per request. This drive
answers it in about a second of wall-clock time, with no simulator and no
request to any model:

- a script gives the measured scene on every control tick;
- the configuration's recorded answers stand in for the model, through its
  fixture provider, and each reaches the car a set time after its request, as
  a cloud model's answer would;
- everything between the policy and the model is DriveBench's own: the runtime
  that asks the policy, its request floor (one request in flight, the minimum
  interval, the budget), the safety floor that validates each answer, and the
  fault wrapper that refuses every request a fault window covers, inside the
  budget, before it would leave the car.

The car does not drive. The script says where it is and how fast it goes,
whatever the answers say, so a timing drive shows when a policy asks, never how
the car would drive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import Event, Lock
from typing import Callable, Sequence

from metadrive_starter.config import AppConfig
from metadrive_starter.env import control_timestep_s
from metadrive_starter.faults import FaultInjectingModelProvider, FaultKind, FaultSpec
from metadrive_starter.perception.scene import LocalScene
from metadrive_starter.planning.command_executor import VLACommandExecutor
from metadrive_starter.safety.policy import build_high_level_safety_policy
from metadrive_starter.vla import (
    InferenceDiscarded,
    InferenceFailure,
    InferenceSubmitDisposition,
    InferenceSuccess,
    ModelFailureCategory,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelTimeoutError,
    RequestCappedModelProvider,
    RequestPolicy,
    RGBFrame,
    model_failure_category,
)
from metadrive_starter.vla_control import VLACommandRuntime
from metadrive_starter.vla_runtime import build_vla_scheduler

# The provider faults a timing drive can schedule: the two that Assignment 3's
# hidden drives add.
FAULT_KINDS = (FaultKind.PROVIDER_OUTAGE, FaultKind.PROVIDER_QUOTA)
# How long the drive waits, in wall-clock time, for DriveBench's inference
# worker. The worker has nothing to wait for but this drive, so it is reached
# only if something is broken.
_WORKER_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class FaultWindow:
    """Every request sent from ``start_s`` until ``end_s`` is refused, in one way."""

    kind: FaultKind
    start_s: float
    # None: until the drive ends.
    end_s: float | None = None

    def __post_init__(self) -> None:
        kind = FaultKind(self.kind)
        if kind not in FAULT_KINDS:
            allowed = ", ".join(item.value for item in FAULT_KINDS)
            raise ValueError(f"a timing drive's fault must be one of: {allowed}")
        object.__setattr__(self, "kind", kind)
        if not _non_negative_finite(self.start_s):
            raise ValueError("start_s must be finite and non-negative")
        if self.end_s is not None and (
            not _non_negative_finite(self.end_s) or self.end_s <= self.start_s
        ):
            raise ValueError("end_s must be None or later than start_s")

    def covers(self, now_s: float) -> bool:
        return self.start_s <= now_s and (self.end_s is None or now_s < self.end_s)


@dataclass(frozen=True)
class TimedRequest:
    """One request the provider received, and what came of it."""

    request_id: str
    sent_s: float
    # The scene measured on the tick it was sent.
    scene: LocalScene
    # The fault windows that refused it; empty when none covered it.
    faults: tuple[FaultKind, ...]
    # When its answer or failure reached the car; None when the drive ended first.
    collected_s: float | None
    # Why it produced no answer, as the policy saw it; None for an answer.
    failure: ModelFailureCategory | None
    # When the command from its answer held authority, from the tick it was
    # accepted until the tick it no longer did: its horizon ended, a newer
    # command replaced it, or the review ended it. The end is the drive's end
    # when it still held then. None when it never held authority.
    authority: tuple[float, float] | None


@dataclass(frozen=True)
class TimedTick:
    """One control tick as the policy and the request floor decided it."""

    now_s: float
    # True when the policy asked and its answer stood.
    asked: bool
    # Why the runtime held the request back for the policy; None when its answer
    # stood.
    problem: str | None
    # Wall-clock time the policy took to answer.
    elapsed_s: float
    # What the request floor did with the tick: started, held, busy,
    # rate_limited, or request_cap_exhausted; None after the budget is spent.
    disposition: str | None
    # Requests left in the budget after the tick; None when unbounded.
    requests_remaining: int | None
    # The request whose command held authority after the tick; None while the
    # local fallback drove.
    authority: str | None


@dataclass(frozen=True)
class TimedDrive:
    """A timing drive's record: every tick, and every request."""

    control_step_s: float
    duration_s: float
    action_horizon_s: float
    minimum_interval_s: float
    request_budget: int | None
    answer_latency_s: float
    faults: tuple[FaultWindow, ...]
    ticks: tuple[TimedTick, ...]
    requests: tuple[TimedRequest, ...]


def drive_request_policy(
    policy: RequestPolicy,
    config: AppConfig,
    scene_at: Callable[[float], LocalScene],
    *,
    duration_s: float,
    answer_latency_s: float,
    faults: Sequence[FaultWindow] = (),
) -> TimedDrive:
    """Drive ``policy`` for ``duration_s`` of simulated time on the road
    ``scene_at`` describes, and record when it asked and what each answer held.

    ``config`` sets the timing, the request limits, and the safety floor, as a
    drive's configuration does after ``agent.yaml``; its provider must be the
    fixture provider, whose recorded answers stand in for the model. Each
    answer reaches the car ``answer_latency_s`` after its request, or fails as
    a timeout at ``vla.request_timeout_s`` when that is later. A request a fault
    window covers is refused on the next tick. ``scene_at(now_s)`` returns the
    scene measured at ``now_s``.

    ``begin_episode()`` is called once, before the first tick, as a drive calls
    it; an exception from it, as from a drive's, stops the drive.
    """
    if config.vla.provider != "fixture":
        raise ValueError("a timing drive answers from recordings: vla.provider must be 'fixture'")
    if not _non_negative_finite(answer_latency_s):
        raise ValueError("answer_latency_s must be finite and non-negative")
    if not _non_negative_finite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    faults = tuple(faults)
    if any(not isinstance(window, FaultWindow) for window in faults):
        raise TypeError("faults must be FaultWindow values")
    step_s = control_timestep_s(config.simulator)
    delayed: list[_DelayedProvider] = []

    def delay(provider: ModelProvider) -> ModelProvider:
        delayed.append(_DelayedProvider(provider, answer_latency_s))
        return delayed[0]

    scheduler = build_vla_scheduler(config.vla, inject_faults=True, around_provider=delay)
    capped = scheduler.pipeline.provider
    fault_provider = capped.provider if isinstance(capped, RequestCappedModelProvider) else capped
    assert isinstance(fault_provider, FaultInjectingModelProvider)
    model = delayed[0]
    runtime = VLACommandRuntime(
        scheduler,
        build_high_level_safety_policy(config.command_validation),
        VLACommandExecutor(config.controller.target_speed_mps),
        run_id="timing",
        camera=_GreyCamera(),  # type: ignore[arg-type]
        fault_provider=fault_provider,
        request_policy=policy,
    )
    cruise_speed_mps = config.controller.target_speed_mps
    fault_specs = tuple(
        FaultSpec(
            fault_id=f"{window.kind.value}-{number}",
            kind=window.kind,
            start_step=round(window.start_s / step_s),
            duration_steps=max(
                1,
                round(((duration_s if window.end_s is None else window.end_s) - window.start_s) / step_s),
            ),
        )
        for number, window in enumerate(faults, start=1)
    )
    ticks: list[TimedTick] = []
    sent: dict[str, tuple[float, LocalScene, tuple[FaultKind, ...]]] = {}
    collected: dict[str, tuple[float, ModelFailureCategory | None]] = {}
    authority: dict[str, list[float]] = {}
    holder: str | None = None
    try:
        for index in range(int(math.floor(duration_s / step_s + 1e-9))):
            now_s = round(index * step_s, 9)
            model.release_due(now_s, scheduler)
            scene = scene_at(now_s)
            if not isinstance(scene, LocalScene):
                raise TypeError(f"scene_at({now_s:g}) must return a LocalScene")
            active = tuple(
                spec for spec, window in zip(fault_specs, faults) if window.covers(now_s)
            )
            update = runtime.update(
                None,
                scene,
                now_s=now_s,
                ego_speed_mps=scene.ego_speed_mps,
                cruise_speed_mps=cruise_speed_mps,
                active_faults=active,
            )
            completion = update.completion
            if isinstance(completion, InferenceSuccess):
                collected[completion.request_id] = (now_s, None)
            elif isinstance(completion, (InferenceFailure, InferenceDiscarded)):
                error = completion.error
                collected[completion.request_id] = (
                    now_s,
                    None if error is None else model_failure_category(error),
                )
            current = (
                None
                if update.active_validation is None
                else update.active_validation.effective_command.command_id
            )
            if current != holder:
                if holder is not None:
                    authority[holder][1] = now_s
                if current is not None:
                    authority.setdefault(current, [now_s, duration_s])
                holder = current
            disposition = update.submission_disposition
            if disposition is InferenceSubmitDisposition.STARTED:
                assert update.submitted_request_id is not None
                sent[update.submitted_request_id] = (
                    now_s,
                    scene,
                    tuple(window.kind for window in faults if window.covers(now_s)),
                )
                model.wait_until_received(scheduler)
            request = update.request
            ticks.append(
                TimedTick(
                    now_s=now_s,
                    asked=bool(request and request.request),
                    problem=None if request is None else request.problem,
                    elapsed_s=0.0 if request is None else request.elapsed_s,
                    disposition=None if disposition is None else disposition.value,
                    requests_remaining=_remaining(runtime),
                    authority=current,
                )
            )
    finally:
        model.release_all()
        scheduler.close()
    requests = tuple(
        TimedRequest(
            request_id=request_id,
            sent_s=sent_s,
            scene=scene,
            faults=kinds,
            collected_s=collected.get(request_id, (None, None))[0],
            failure=collected.get(request_id, (None, None))[1],
            authority=(
                None
                if request_id not in authority
                else (authority[request_id][0], authority[request_id][1])
            ),
        )
        for request_id, (sent_s, scene, kinds) in sent.items()
    )
    return TimedDrive(
        control_step_s=step_s,
        duration_s=duration_s,
        action_horizon_s=config.vla.action_horizon_s,
        minimum_interval_s=config.vla.minimum_interval_s,
        request_budget=config.vla.maximum_requests_per_run,
        answer_latency_s=answer_latency_s,
        faults=faults,
        ticks=tuple(ticks),
        requests=requests,
    )


def _remaining(runtime: VLACommandRuntime) -> int | None:
    budget = runtime.scheduler.request_budget
    return None if budget is None else budget.remaining_requests


class _DelayedProvider:
    """Holds each outcome of the provider it wraps until the drive's clock
    reaches the time it is due, so that it reaches the car on that tick.

    It stands where the model would, inside the fault wrapper: a refused request
    never reaches it, and comes back on the next tick.
    """

    def __init__(self, provider: ModelProvider, answer_latency_s: float) -> None:
        self.provider = provider
        self.answer_latency_s = answer_latency_s
        self._lock = Lock()
        self._received = Event()
        self._pending: tuple[float, Event] | None = None

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        error: Exception | None = None
        response: ModelResponse | None = None
        try:
            response = self.provider.generate(request, timeout_s=timeout_s)
        except Exception as exc:
            error = exc
        latency_s = self.answer_latency_s
        if error is None and latency_s > timeout_s:
            latency_s = timeout_s
            error = ModelTimeoutError(f"no answer within the {timeout_s:g} s request timeout")
        release = Event()
        with self._lock:
            self._pending = (request.frame.timestamp_s + latency_s, release)
        self._received.set()
        release.wait(_WORKER_TIMEOUT_S)
        if error is not None:
            raise error
        assert response is not None
        return response

    def wait_until_received(self, scheduler: object) -> None:
        """Wait until the request just started has reached this provider, or
        finished without reaching it, refused for a fault."""
        for _ in range(int(_WORKER_TIMEOUT_S / 0.001)):
            if self._received.wait(0.001):
                self._received.clear()
                return
            if scheduler.wait_for_completion(0.0):  # type: ignore[attr-defined]
                return
        raise RuntimeError("DriveBench's inference worker did not take the request")

    def release_due(self, now_s: float, scheduler: object) -> None:
        """Let the outcome held back go, if it is due by ``now_s``, and wait
        until DriveBench's worker has finished with it."""
        with self._lock:
            pending = self._pending
            if pending is None or pending[0] > now_s + 1e-9:
                return
            self._pending = None
        pending[1].set()
        if not scheduler.wait_for_completion(_WORKER_TIMEOUT_S):  # type: ignore[attr-defined]
            raise RuntimeError("DriveBench's inference worker did not finish the request")

    def release_all(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            pending[1].set()


class _GreyCamera:
    """A one-pixel grey frame at the tick's time: the recorded answers never look."""

    def capture(self, env: object, *, timestamp_s: float) -> RGBFrame:
        return RGBFrame(timestamp_s, 1, 1, b"\x80\x80\x80")


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


__all__ = [
    "FAULT_KINDS",
    "FaultWindow",
    "TimedDrive",
    "TimedRequest",
    "TimedTick",
    "drive_request_policy",
]
