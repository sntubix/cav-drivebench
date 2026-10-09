"""The arbitration seam: how far to trust one assessment.

An ``AssessmentArbiter`` receives each assessment the model returns, with the
local scene measured when the runtime collects it, and answers with the
assessment the safety floor validates. It runs on the control loop, upstream of
command derivation, so the command and everything that consumes it are unchanged.
It also reviews every control tick, whether or not an assessment arrived, and
may end the model command that holds authority: when the model fails, when that
command grows old, or when the scene no longer supports it.

An arbiter may only take authority away (the monotone restriction, ADR-0001): lower
the proposed target speed, lower the confidence, or decline by replacing the
proposed action with ``REQUEST_FALLBACK``. Only those three fields of its answer
reach the vehicle; everything else stays as the model reported it. A review may
keep the command that holds authority or end it, never extend it. The runtime
checks every answer, declines the assessment itself when the arbiter raises or
answers outside the proposal, and ends the command when a review raises or does
not answer True or False, so a broken arbiter can be over-conservative but never
unsafe.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING, Callable, Protocol, TypeAlias, runtime_checkable

from metadrive_starter.perception.scene import LocalScene
from metadrive_starter.vla.assessment import VLAAssessment
from metadrive_starter.vla.commands import HighLevelAction
from metadrive_starter.vla.provider import ModelFailureCategory

if TYPE_CHECKING:
    from metadrive_starter.safety import CommandValidationSettings


class ArbitrationError(RuntimeError):
    """An arbiter could not answer; the message says why in full."""


@dataclass(frozen=True)
class ArbitrationRequest:
    """One assessment to arbitrate, and what it can be judged against."""

    # The model's assessment, exactly as decoded.
    assessment: VLAAssessment
    # The local scene measured now: the one the safety floor validates against.
    scene: LocalScene
    # Simulation time now.
    now_s: float
    # When the model was asked, with the frame it assessed; now_s minus this is
    # the assessment's age.
    issued_at_s: float
    # The desired clear-road cruise speed.
    cruise_speed_mps: float


@dataclass(frozen=True)
class ReviewRequest:
    """One control tick, and the model command that holds authority, if any."""

    # The assessment, as arbitrated, behind the command that holds authority;
    # None while the local fallback drives.
    active: VLAAssessment | None
    # When the model was asked for that assessment; None with it.
    issued_at_s: float | None
    # The local scene measured now.
    scene: LocalScene
    # Simulation time now.
    now_s: float
    # The desired clear-road cruise speed.
    cruise_speed_mps: float
    # Why a model request collected on this tick produced no assessment: it timed
    # out, failed, or answered in a form the decoder refused. None otherwise.
    failure: ModelFailureCategory | None


@runtime_checkable
class AssessmentArbiter(Protocol):
    """Replaceable arbitration: one assessment in, the assessment the floor validates out."""

    def begin_episode(self) -> None:
        """Forget every earlier assessment. Called before an episode's first one."""
        ...

    def arbitrate(self, request: ArbitrationRequest) -> VLAAssessment:
        """Return ``request.assessment``, or a copy with less authority.

        Called on the control loop for every assessment the model returns. It
        must return within one control step.
        """
        ...

    def review(self, request: ReviewRequest) -> bool:
        """Return True to keep the model command that holds authority, False to end it.

        Called on the control loop every control tick, after any assessment that
        arrived on it has been arbitrated. It must return within one control step.
        """
        ...


AssessmentArbiterFactory: TypeAlias = Callable[
    ["CommandValidationSettings"], AssessmentArbiter
]


class DefaultArbiter:
    """Endorses every assessment: DriveBench as it drove before arbitration.
    Runtimes given no arbiter use it."""

    def begin_episode(self) -> None:
        pass

    def arbitrate(self, request: ArbitrationRequest) -> VLAAssessment:
        return request.assessment

    def review(self, request: ReviewRequest) -> bool:
        return True


def build_arbiter(
    settings: CommandValidationSettings,
    *,
    factory: AssessmentArbiterFactory,
) -> AssessmentArbiter:
    """Build an arbiter from its factory, checking it implements the protocol."""
    arbiter = factory(settings)
    if not isinstance(arbiter, AssessmentArbiter):
        raise TypeError(
            "arbiter must implement begin_episode(), arbitrate(request), and review(request)"
        )
    return arbiter


class ArbitrationOutcome(str, Enum):
    """What an arbitrated assessment kept of the model's authority."""

    ENDORSED = "endorsed"
    # A lower target speed or confidence, with the proposed action.
    RESTRICTED = "restricted"
    # The proposed action replaced by REQUEST_FALLBACK.
    DECLINED = "declined"


@dataclass(frozen=True)
class ArbitrationDecision:
    """One assessment as arbitrated: what the safety floor receives, and why."""

    outcome: ArbitrationOutcome
    # The model's assessment with the action, target speed, and confidence the
    # arbiter answered, or declined when its answer could not stand.
    assessment: VLAAssessment
    # Why the runtime declined on the arbiter's behalf; None when its answer stood.
    problem: str | None
    # Wall-clock time the arbiter took to answer.
    elapsed_s: float


def arbitrate_assessment(
    arbiter: AssessmentArbiter,
    request: ArbitrationRequest,
) -> ArbitrationDecision:
    """Ask ``arbiter`` about ``request``, and keep its answer only if it stays
    within the model's proposal; otherwise decline."""
    proposal = request.assessment
    answer: object = None
    problem: str | None = None
    started_s = time.perf_counter()
    try:
        answer = arbiter.arbitrate(request)
    except ArbitrationError as exc:
        problem = str(exc)
    except Exception as exc:
        problem = f"arbitrate(request) raised {type(exc).__name__}: {exc}"
    elapsed_s = time.perf_counter() - started_s
    if problem is None:
        violations = monotone_violations(proposal, answer)
        problem = "; ".join(violations) if violations else None
    if problem is None:
        assert isinstance(answer, VLAAssessment)
        assessment = replace(
            proposal,
            proposed_action=answer.proposed_action,
            proposed_target_speed_mps=float(answer.proposed_target_speed_mps),
            confidence=float(answer.confidence),
        )
    else:
        assessment = declined(proposal)
    return ArbitrationDecision(
        outcome=_outcome(proposal, assessment),
        assessment=assessment,
        problem=problem,
        elapsed_s=elapsed_s,
    )


@dataclass(frozen=True)
class ReviewDecision:
    """One control tick as reviewed."""

    # True when the review ended the model command that held authority.
    revoked: bool
    # Why the runtime ended that command on the arbiter's behalf, or why the
    # review could not stand; None when its answer stood.
    problem: str | None
    # Wall-clock time the arbiter took to answer.
    elapsed_s: float


def review_command(arbiter: AssessmentArbiter, request: ReviewRequest) -> ReviewDecision:
    """Ask ``arbiter`` whether the model command that holds authority stands,
    ending it unless the answer is True."""
    answer: object = None
    problem: str | None = None
    started_s = time.perf_counter()
    try:
        answer = arbiter.review(request)
    except ArbitrationError as exc:
        problem = str(exc)
    except Exception as exc:
        problem = f"review(request) raised {type(exc).__name__}: {exc}"
    elapsed_s = time.perf_counter() - started_s
    if problem is None:
        problem = review_violation(answer)
    return ReviewDecision(
        revoked=request.active is not None and (problem is not None or answer is False),
        problem=problem,
        elapsed_s=elapsed_s,
    )


def review_violation(answer: object) -> str | None:
    """Why ``answer`` cannot stand as a review, or None when it is True or False."""
    if isinstance(answer, bool):
        return None
    return f"review(request) must return True or False; got {answer!r}"


def declined(assessment: VLAAssessment) -> VLAAssessment:
    """``assessment`` with its action replaced by ``REQUEST_FALLBACK``."""
    return replace(assessment, proposed_action=HighLevelAction.REQUEST_FALLBACK)


def monotone_violations(proposal: VLAAssessment, answer: object) -> tuple[str, ...]:
    """Every way ``answer`` grants more authority than ``proposal``, or cannot be checked.

    Empty when ``answer`` keeps or lowers the target speed and the confidence, and
    keeps the proposed action or replaces it with ``REQUEST_FALLBACK``. Actions are
    not ordered by caution: stopping is safer than keeping lane on an empty road and
    less safe with traffic behind, so only surrendering model authority is always a
    restriction.
    """
    if not isinstance(answer, VLAAssessment):
        return (
            "arbitrate(request) must return a VLAAssessment; "
            f"got {_with_article(type(answer).__name__)}",
        )
    violations: list[str] = []
    action = answer.proposed_action
    if not isinstance(action, HighLevelAction):
        violations.append(f"proposed_action must be a HighLevelAction; got {action!r}")
    elif action not in {proposal.proposed_action, HighLevelAction.REQUEST_FALLBACK}:
        violations.append(
            f"proposed_action changed from {proposal.proposed_action.value} to "
            f"{action.value}; only {HighLevelAction.REQUEST_FALLBACK.value} may replace it"
        )
    for name, value, limit, unit in (
        (
            "proposed_target_speed_mps",
            answer.proposed_target_speed_mps,
            proposal.proposed_target_speed_mps,
            " m/s",
        ),
        ("confidence", answer.confidence, proposal.confidence, ""),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            violations.append(f"{name} must be a number; got {value!r}")
        elif not math.isfinite(value) or value < 0.0:
            violations.append(f"{name} must be finite and at least 0; got {value!r}")
        elif value > limit:
            violations.append(f"{name} raised from {limit:g} to {value:g}{unit}")
    return tuple(violations)


def _outcome(proposal: VLAAssessment, assessment: VLAAssessment) -> ArbitrationOutcome:
    fallback = HighLevelAction.REQUEST_FALLBACK
    if assessment.proposed_action is fallback and proposal.proposed_action is not fallback:
        return ArbitrationOutcome.DECLINED
    if (
        assessment.proposed_target_speed_mps < proposal.proposed_target_speed_mps
        or assessment.confidence < proposal.confidence
    ):
        return ArbitrationOutcome.RESTRICTED
    return ArbitrationOutcome.ENDORSED


def _with_article(noun: str) -> str:
    return f"{'an' if noun[:1] in 'AEIOU' else 'a'} {noun}"
