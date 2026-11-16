"""The request-policy seam: when to ask the model.

A ``RequestPolicy`` is asked on the control loop, every control tick, whether
DriveBench should ask the model now. It decides before the camera is captured,
so a tick it declines costs no frame. Its answer can only hold a request back:
when it answers True, the request still goes through the instructor-owned floor
below it, which keeps one request in flight at most, waits
``vla.minimum_interval_s`` between requests, and stops at the request budget. A
policy can therefore make DriveBench ask less often, never more often, than it
asked before the seam (ADR-0005).

The runtime checks every answer and holds the request back itself when the
policy raises or answers anything but True or False, so a broken policy can
starve the vehicle of model authority but never exceed the budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, TypeAlias, runtime_checkable

from metadrive_starter.perception.scene import LocalScene
from metadrive_starter.vla.assessment import VLAAssessment
from metadrive_starter.vla.provider import ModelFailureCategory

if TYPE_CHECKING:
    from metadrive_starter.config import VLASettings


class RequestPolicyError(RuntimeError):
    """A request policy could not answer; the message says why in full."""


@dataclass(frozen=True)
class RequestPolicySettings:
    """The instructor's request limits, which a policy works within."""

    # Model requests one episode may attempt; None when unbounded. Every
    # DriveBench run drives one episode, so this is vla.maximum_requests_per_run.
    request_budget: int | None
    # The least simulation time between two requests; DriveBench never asks
    # sooner, whatever the policy answers.
    minimum_interval_s: float
    # How long one request may take before it fails as a timeout.
    request_timeout_s: float
    # How long a command derived from one assessment may hold authority.
    action_horizon_s: float

    @classmethod
    def from_vla(cls, settings: VLASettings) -> RequestPolicySettings:
        return cls(
            request_budget=settings.maximum_requests_per_run,
            minimum_interval_s=settings.minimum_interval_s,
            request_timeout_s=settings.request_timeout_s,
            action_horizon_s=settings.action_horizon_s,
        )


@dataclass(frozen=True)
class RequestTick:
    """One control tick, and what a request policy can decide from."""

    # Simulation time now.
    now_s: float
    # The local scene measured now.
    scene: LocalScene
    # The measured ego speed.
    ego_speed_mps: float
    # The desired clear-road cruise speed.
    cruise_speed_mps: float
    # The assessment, as arbitrated, behind the model command that holds
    # authority after this tick's review; None while the local fallback drives.
    active: VLAAssessment | None
    # When the model was asked for that assessment; None with it.
    issued_at_s: float | None
    # Why a model request collected on this tick produced no assessment: it
    # timed out, met a quota or an outage, failed, or answered in a form the
    # decoder refused. None otherwise.
    failure: ModelFailureCategory | None
    # True while an earlier request is in flight: asking now would be refused.
    in_flight: bool
    # When this episode's latest request was sent; None before its first.
    last_requested_at_s: float | None
    # Requests left in the budget, counting one in flight as spent; None when
    # unbounded. Asking at 0 is refused.
    requests_remaining: int | None
    # The planner's speed limit for the road curvature within its preview, from
    # the latest plan, made on the previous tick; None on a straight road and
    # on an episode's first tick.
    route_speed_cap_mps: float | None


@runtime_checkable
class RequestPolicy(Protocol):
    """Replaceable request timing: one control tick in, ask now or not out."""

    def begin_episode(self) -> None:
        """Forget every earlier tick. Called before an episode's first one."""
        ...

    def should_request(self, tick: RequestTick) -> bool:
        """Return True to ask the model now, False to hold the request back.

        Called on the control loop every control tick, before the camera is
        captured. It must return within one control step.
        """
        ...


RequestPolicyFactory: TypeAlias = Callable[[RequestPolicySettings], RequestPolicy]


class DefaultRequestPolicy:
    """Asks whenever DriveBench may, so that ``vla.minimum_interval_s`` alone
    paces the requests: DriveBench as it asked before the request policy.
    Runtimes given no policy use it."""

    def begin_episode(self) -> None:
        pass

    def should_request(self, tick: RequestTick) -> bool:
        return True


def build_request_policy(
    settings: RequestPolicySettings,
    *,
    factory: RequestPolicyFactory,
) -> RequestPolicy:
    """Build a request policy from its factory, checking it implements the protocol."""
    policy = factory(settings)
    if not isinstance(policy, RequestPolicy):
        raise TypeError(
            "request policy must implement begin_episode() and should_request(tick)"
        )
    return policy


@dataclass(frozen=True)
class RequestDecision:
    """One control tick as the request policy decided it."""

    # True when the policy asked for a request and its answer stood.
    request: bool
    # Why the runtime held the request back on the policy's behalf; None when
    # its answer stood.
    problem: str | None
    # Wall-clock time the policy took to answer.
    elapsed_s: float


def decide_request(policy: RequestPolicy, tick: RequestTick) -> RequestDecision:
    """Ask ``policy`` whether to request the model on ``tick``, holding the
    request back unless the answer is True."""
    answer: object = None
    problem: str | None = None
    started_s = time.perf_counter()
    try:
        answer = policy.should_request(tick)
    except RequestPolicyError as exc:
        problem = str(exc)
    except Exception as exc:
        problem = f"should_request(tick) raised {type(exc).__name__}: {exc}"
    elapsed_s = time.perf_counter() - started_s
    if problem is None:
        problem = request_violation(answer)
    return RequestDecision(
        request=problem is None and answer is True,
        problem=problem,
        elapsed_s=elapsed_s,
    )


def request_violation(answer: object) -> str | None:
    """Why ``answer`` cannot stand as a request decision, or None when it is
    True or False."""
    if isinstance(answer, bool):
        return None
    return f"should_request(tick) must return True or False; got {answer!r}"
