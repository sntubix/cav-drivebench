"""Structural gates: mechanical pass/fail checks on a team's submission.

Teams run these gates locally against the public scenarios, and instructors
grade with the same code against hidden ones, so a hidden failure means a team
skipped a check it had. Gates catch what breaks a run. How well a submission
drives is its score, which the same run reports: half from implementation checks
of its controller, half from how it drove the manifest's routes. The score never
decides a gate.

Where routes are scored, a route that crashes, leaves the road, or does not
arrive is part of how the submission drove: it earns no driving credit, and only
a run an error stopped fails a gate. A team cannot drive the hidden routes, and a
tuning choice that leaves one of them should cost that route, not every mark.
Where routes are not scored, completing them is the gate.

When the base configuration enables the VLA subsystem, the routes are driven
with the model asked through the team's observation.py and its assessments
arbitrated by the team's arbitration.py, and three more gates check them: every
observation built could be sent; neither held the control loop up, an
observation past the request timeout or an arbiter call past one control step;
and no arbiter answer granted more authority than the model proposed. An
observation without every output-contract line can be sent, so it fails
nothing; the first gate's detail counts them.

A manifest may also hold fault scenarios: routes driven under a schedule of
injected camera, LiDAR, or model-provider faults. One more gate judges them, that
no fault crashed the vehicle, sent it off the road, or stopped the run. They need
not arrive, and they never count toward the routes gate or the score.

A manifest that declares ``no_score`` reports no score, only its reason: the
score above is Assignment 1's, and another assignment's run would only mislead.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml

from metadrive_starter.implementation_checks import CheckResult, run_implementation_checks
from metadrive_starter.config import (
    AppConfig,
    ControllerSettings,
    EpisodeRecordingSettings,
    EventLogSettings,
    ObservationSettings,
)
from metadrive_starter.controllers import (
    ControllerOutputError,
    VehicleController,
    VehicleControllerFactory,
)
from metadrive_starter.env import control_timestep_s
from metadrive_starter.events import to_json_value
from metadrive_starter.faults import FaultSpec
from metadrive_starter.safety import CommandValidationSettings
from metadrive_starter.simulation import RunSummary, run_simulation
from metadrive_starter.submission import (
    ARBITRATION_SEAM,
    OBSERVATION_SEAM,
    PROJECT_ROOT,
    SeamSpec,
    SubmissionError,
    apply_submission_overlay,
    load_arbitration_for,
    load_controller,
    load_observation_for,
)
from metadrive_starter.types import Action, ControlTick
from metadrive_starter.vla import (
    ArbitrationError,
    ArbitrationRequest,
    AssessmentArbiter,
    AssessmentArbiterFactory,
    Observation,
    ObservationBuilder,
    ObservationBuilderFactory,
    ObservationError,
    ObservationRequest,
    ReviewRequest,
    VLAAssessment,
    checked_observation,
    describe_missing_contract,
    missing_contract_lines,
    monotone_violations,
    review_violation,
)

GATE_MANIFEST_VERSION = 1
SUBMISSION_LOADS = "submission loads"
BOUNDED_OUTPUTS = "bounded control outputs"
ROUTE_COMPLETES = "route completes"
RUNS_FINISH = "no run stops on an error"
VALID_OBSERVATIONS = "valid observations emitted"
CONTROL_LOOP_UNBLOCKED = "control loop never blocked"
MONOTONE_RESTRICTION = "monotone restriction preserved"
FAULTS_SURVIVED = "faults never break the run"
TREE_UNCHANGED = "nothing outside submission/ was modified"
RELEASE_MANIFEST = "release-manifest.json"
_GENERATED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}
# Jupyter rewrites a notebook every time it runs, and no run ever imports one,
# so notebooks are never part of what the tree gate checks.
_NOTEBOOK_SUFFIX = ".ipynb"
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class CreditBounds:
    """Full credit at or below ``full``, none at or above ``zero``, linear between."""

    full: float
    zero: float
    weight: float

    def credit(self, value: float) -> float:
        return min(1.0, max(0.0, (self.zero - value) / (self.zero - self.full)))

    def weighted_credit(self, value: float) -> float:
        return self.weight * self.credit(value)


# Assignment 1's score. A scenario's driving credit weighs its time against par,
# its speed error once launched, and its lateral error.
# Full credit sits below what the reference reaches on its tuned gains, so
# tuning earns driving credit; a sweep of gains on 6 October 2026 halved the
# speed error and cut the lateral error to a fifth, on the hidden routes as on
# the public ones. Time earns full credit at par and no more, so no setting buys
# credit by finishing faster than the reference, and lowering the target speed
# costs more time credit than it saves in speed error.
IMPLEMENTATION_SHARE = 0.5
LAUNCH_S = 5.0
TIME_CREDIT = CreditBounds(full=1.0, zero=1.3, weight=0.4)  # steps / par steps
SPEED_CREDIT = CreditBounds(full=0.1, zero=0.6, weight=0.4)  # m/s, after LAUNCH_S
LATERAL_CREDIT = CreditBounds(full=0.02, zero=0.15, weight=0.2)  # m, RMS


@dataclass(frozen=True)
class GateScenario:
    """One route a submission must drive, with the world held fixed.

    A scenario with faults is a fault scenario: driven under that schedule and
    judged only by whether a fault broke the run.
    """

    scenario_id: str
    map: str
    seed: int
    horizon: int
    # The reference controller's steps to arrival; the score's time credit
    # compares a submission against it. None for a fault scenario, or a route
    # in a plan that reports no score.
    par_steps: int | None = None
    traffic_density: float = 0.0
    obstacle_probability: float = 0.0
    stopped_vehicle_ahead_m: float | None = None
    # Overrides perception.safety_source; lidar_dropout acts only on "lidar".
    safety_source: str | None = None
    faults: tuple[FaultSpec, ...] = ()

    @property
    def is_fault_scenario(self) -> bool:
        return bool(self.faults)

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not _ID_PATTERN.fullmatch(
            self.scenario_id
        ):
            raise ValueError(f"gate scenario id is invalid: {self.scenario_id!r}")
        if not isinstance(self.map, str) or not self.map.strip():
            raise ValueError("gate scenario map must not be empty")
        if not _non_negative_integer(self.seed):
            raise ValueError("gate scenario seed must be a non-negative integer")
        if not _non_negative_integer(self.horizon) or self.horizon == 0:
            raise ValueError("gate scenario horizon must be a positive integer")
        if self.par_steps is not None and (
            not _non_negative_integer(self.par_steps) or self.par_steps == 0
        ):
            raise ValueError("gate scenario par_steps must be a positive integer")
        if self.par_steps is not None and self.par_steps > self.horizon:
            raise ValueError("gate scenario par_steps must not exceed the horizon")
        if self.stopped_vehicle_ahead_m is not None and (
            isinstance(self.stopped_vehicle_ahead_m, bool)
            or not isinstance(self.stopped_vehicle_ahead_m, (int, float))
            or not math.isfinite(self.stopped_vehicle_ahead_m)
            or self.stopped_vehicle_ahead_m <= 0.0
        ):
            raise ValueError(
                "gate scenario stopped_vehicle_ahead_m must be finite and positive"
            )
        if self.safety_source not in {None, "oracle", "lidar"}:
            raise ValueError("gate scenario safety_source must be 'oracle' or 'lidar'")
        if not isinstance(self.faults, tuple) or any(
            not isinstance(fault, FaultSpec) for fault in self.faults
        ):
            raise ValueError("gate scenario faults must be a tuple of FaultSpec values")
        fault_ids = [fault.fault_id for fault in self.faults]
        if len(set(fault_ids)) != len(fault_ids):
            raise ValueError("gate scenario fault ids must be unique")
        late = [fault.fault_id for fault in self.faults if fault.start_step >= self.horizon]
        if late:
            raise ValueError(
                f"gate scenario fault {late[0]!r} starts after the horizon, so it never acts"
            )
        for name in ("traffic_density", "obstacle_probability"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"gate scenario {name} must be between 0 and 1")


@dataclass(frozen=True)
class GatePlan:
    """The scenarios one assignment's gates run a submission through."""

    plan_id: str
    visibility: str
    scenarios: tuple[GateScenario, ...]
    # Why this plan reports no score, and where the assignment is marked instead;
    # None when its runs are scored.
    no_score: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not _ID_PATTERN.fullmatch(self.plan_id):
            raise ValueError(f"gate manifest id is invalid: {self.plan_id!r}")
        if self.visibility not in {"public", "hidden"}:
            raise ValueError("gate manifest visibility must be 'public' or 'hidden'")
        if not self.scenarios:
            raise ValueError("gate manifest scenarios must not be empty")
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("gate scenario ids must be unique")
        if self.no_score is not None and (
            not isinstance(self.no_score, str) or not self.no_score.strip()
        ):
            raise ValueError("gate manifest no_score must be text saying why")
        if self.no_score is None:
            unscored = [
                scenario.scenario_id
                for scenario in self.scenarios
                if scenario.par_steps is None and not scenario.is_fault_scenario
            ]
            if unscored:
                raise ValueError(
                    f"gate scenario {unscored[0]!r} needs par_steps, "
                    "unless the manifest declares no_score"
                )


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not run"


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    detail: str = ""


@dataclass(frozen=True)
class ScenarioScore:
    """How one route was driven. ``credit`` runs from 0 to 1."""

    scenario_id: str
    credit: float
    steps: int | None
    par_steps: int
    # Mean |target - speed| after the first LAUNCH_S seconds.
    speed_error_mps: float | None
    # Root mean square over the whole route.
    lateral_error_m: float | None
    # Why the route earned no credit: it crashed, left the road, did not arrive,
    # or its run stopped. None when it completed.
    problem: str | None = None


@dataclass(frozen=True)
class Score:
    """A quality measure out of 100, never a gate."""

    total: float
    implementation: float
    driving: float
    checks: tuple[CheckResult, ...]
    scenarios: tuple[ScenarioScore, ...]


@dataclass(frozen=True)
class GateReport:
    plan_id: str
    visibility: str
    submission: str
    submission_sha256: Mapping[str, str]
    results: tuple[GateResult, ...]
    # None when the submission did not load, or the plan reports no score.
    score: Score | None = None
    # The plan's reason for reporting no score.
    no_score: str | None = None
    # The scenarios driven, in manifest order; a partial run drove fewer than
    # the manifest holds, which only iteration ever asks for.
    scenarios_run: tuple[str, ...] = ()
    partial: bool = False

    @property
    def passed(self) -> bool:
        return all(result.status is not GateStatus.FAILED for result in self.results)

    def to_dict(self) -> dict[str, object]:
        value = to_json_value(self)
        assert isinstance(value, dict)
        value["passed"] = self.passed
        return value


@dataclass
class _ObservationLog:
    """What a submitted observation builder did during one drive."""

    emitted: int = 0
    # Emitted without every output-contract line, and the lines the first lacked.
    without_contract: int = 0
    first_missing_contract: tuple[str, ...] = ()
    slowest_s: float = 0.0
    # Why an observation could not be sent, per build that produced none.
    problems: list[str] = field(default_factory=list)
    # The first call that ran past the deadline; later calls are refused.
    overrun: str | None = None


@dataclass
class _ArbitrationLog:
    """What a submitted arbiter did during one drive."""

    # arbitrate calls that returned, whatever they returned.
    answered: int = 0
    # review calls that returned, whatever they returned.
    reviewed: int = 0
    slowest_s: float = 0.0
    # Why an answer could not stand, per call that raised, answered with more
    # authority than the model proposed, or reviewed with neither True nor False.
    problems: list[str] = field(default_factory=list)
    # The first call that ran past the control step; later calls are refused.
    overrun: str | None = None


@dataclass(frozen=True)
class _Drive:
    scenario: GateScenario
    summary: RunSummary | None
    error: Exception | None
    # Every control tick the submitted controller saw.
    ticks: tuple[ControlTick, ...] = ()
    # None when the drive never asks the model.
    observations: _ObservationLog | None = None
    arbitrations: _ArbitrationLog | None = None


class _Recorder:
    """Passes every call through to a submitted controller, keeping its ticks."""

    def __init__(self, controller: VehicleController) -> None:
        self._controller = controller
        self.ticks: list[ControlTick] = []

    def update(self, tick: ControlTick) -> Action:
        self.ticks.append(tick)
        return self._controller.update(tick)

    def reset_speed_control(self) -> None:
        self._controller.reset_speed_control()


class _Overrun(Exception):
    """A submitted call had not returned by its deadline; the message says which."""


class _ObservationRecorder:
    """Passes every build through to a submitted observation builder, within a
    deadline, and logs what it returned. The pipeline then checks and sends the
    observation exactly as it would without the recorder."""

    def __init__(
        self,
        builder: ObservationBuilder,
        log: _ObservationLog,
        *,
        deadline_s: float,
    ) -> None:
        self._builder = builder
        self._log = log
        self._deadline_s = deadline_s

    def build(self, request: ObservationRequest) -> Observation:
        log = self._log
        if log.overrun is not None:
            raise ObservationError(
                "the observation builder already ran past its deadline, "
                "so it is no longer called"
            )
        started_s = time.perf_counter()
        try:
            observation = _within(
                self._deadline_s, "build(request)", self._builder.build, request
            )
        except _Overrun as overrun:
            log.overrun = str(overrun)
            raise ObservationError(log.overrun) from None
        except Exception as exc:
            log.problems.append(f"build(request) raised {type(exc).__name__}: {exc}")
            raise
        finally:
            log.slowest_s = max(log.slowest_s, time.perf_counter() - started_s)
        try:
            checked_observation(observation, request)
        except ObservationError as exc:
            log.problems.append(str(exc))
            return observation
        log.emitted += 1
        missing = missing_contract_lines(observation.prompt, request.contract)
        if missing:
            log.without_contract += 1
            log.first_missing_contract = log.first_missing_contract or missing
        return observation


class ControlStepOverrun(ArbitrationError):
    """A submitted arbiter ran past the control step, so the gates stopped calling it."""


class _ArbitrationRecorder:
    """Passes every call through to a submitted arbiter, within one control step,
    and logs what it answered. The runtime then checks the answer and derives
    the command exactly as it would without the recorder."""

    def __init__(
        self,
        arbiter: AssessmentArbiter,
        log: _ArbitrationLog,
        *,
        budget_s: float,
    ) -> None:
        self._arbiter = arbiter
        self._log = log
        self._budget_s = budget_s

    def begin_episode(self) -> None:
        self._within("begin_episode()", self._arbiter.begin_episode)

    def arbitrate(self, request: ArbitrationRequest) -> VLAAssessment:
        log = self._log
        try:
            answer = self._within("arbitrate(request)", self._arbiter.arbitrate, request)
        except ControlStepOverrun:
            raise
        except Exception as exc:
            log.problems.append(f"arbitrate(request) raised {type(exc).__name__}: {exc}")
            raise
        log.answered += 1
        violations = monotone_violations(request.assessment, answer)
        if violations:
            log.problems.append("; ".join(violations))
        return answer

    def review(self, request: ReviewRequest) -> bool:
        log = self._log
        try:
            answer = self._within("review(request)", self._arbiter.review, request)
        except ControlStepOverrun:
            raise
        except Exception as exc:
            log.problems.append(f"review(request) raised {type(exc).__name__}: {exc}")
            raise
        log.reviewed += 1
        violation = review_violation(answer)
        if violation is not None:
            log.problems.append(violation)
        return answer

    def _within(self, call: str, function: Callable[..., Any], *arguments: Any) -> Any:
        log = self._log
        if log.overrun is not None:
            raise ControlStepOverrun(
                "the arbiter already ran past the control step, so it is no longer called"
            )
        started_s = time.perf_counter()
        try:
            return _within(
                self._budget_s, call, function, *arguments, limit="the control step"
            )
        except _Overrun as overrun:
            log.overrun = str(overrun)
            raise ControlStepOverrun(log.overrun) from None
        finally:
            log.slowest_s = max(log.slowest_s, time.perf_counter() - started_s)


def _within(
    deadline_s: float,
    call: str,
    function: Callable[..., Any],
    *arguments: Any,
    limit: str = "the request timeout",
) -> Any:
    """Return ``function(*arguments)``, or raise _Overrun once ``deadline_s`` passes.

    ``call`` names the submitted call and ``limit`` what the deadline is, as the
    overrun message quotes them. The call runs on a daemon thread that is
    abandoned, never awaited, if it overruns, so a submission that never returns
    cannot hang the gate run.
    """
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = function(*arguments)
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run, name="gate-submission-call", daemon=True)
    thread.start()
    thread.join(deadline_s)
    if thread.is_alive():
        raise _Overrun(f"{call} had not returned after {deadline_s:g} s, {limit}")
    if "error" in outcome:
        raise outcome["error"]
    if "value" not in outcome:
        raise ObservationError("the call ended without returning or raising an exception")
    return outcome["value"]


def run_gates(
    submission_dir: Path | str,
    plan: GatePlan,
    *,
    base_config: AppConfig,
    release_root: Path | str = PROJECT_ROOT,
    scenario_ids: Sequence[str] | None = None,
) -> GateReport:
    """Run every gate against one submission; a failed gate never hides another.

    ``scenario_ids`` drives only those scenarios, for a quicker check while
    iterating; the report marks the run partial. Grading drives them all.
    """
    submission = Path(submission_dir)
    check_gate_plan(plan, base_config)
    scenarios = select_gate_scenarios(plan, scenario_ids)
    results: list[GateResult] = []
    score: Score | None = None
    # A plan that scores its routes counts an incomplete route against its score.
    route_gate = ROUTE_COMPLETES if plan.no_score is not None else RUNS_FINISH
    # Only drives that ask the model build observations and arbitrate
    # assessments; the overlay cannot switch the VLA subsystem on or off.
    model_gates = (
        (VALID_OBSERVATIONS, CONTROL_LOOP_UNBLOCKED, MONOTONE_RESTRICTION)
        if base_config.vla.enabled
        else ()
    )
    fault_gates = (
        (FAULTS_SURVIVED,)
        if any(scenario.is_fault_scenario for scenario in plan.scenarios)
        else ()
    )
    try:
        config = apply_submission_overlay(base_config, submission)
        factory = load_controller(submission)
        # Build one controller now, so a failing constructor fails this gate
        # rather than every route.
        factory(config.controller)
        observation_factory = _load_observation(submission, config)
        arbitration_factory = _load_arbitration(submission, config)
    except SubmissionError as exc:
        results.append(GateResult(SUBMISSION_LOADS, GateStatus.FAILED, str(exc)))
        results.extend(
            GateResult(gate, GateStatus.NOT_RUN, "the submission did not load")
            for gate in (BOUNDED_OUTPUTS, route_gate, *model_gates, *fault_gates)
        )
    else:
        drives = [
            _drive(config, scenario, factory, observation_factory, arbitration_factory)
            for scenario in scenarios
        ]
        routes = [drive for drive in drives if not drive.scenario.is_fault_scenario]
        results.append(GateResult(SUBMISSION_LOADS, GateStatus.PASSED))
        results.append(_bounded_outputs(drives))
        results.append(
            _route_completes(routes) if plan.no_score is not None else _runs_finish(routes)
        )
        if model_gates:
            results.append(_valid_observations(drives))
            results.append(
                _control_loop_unblocked(
                    drives,
                    config.vla.request_timeout_s,
                    control_timestep_s(config.simulator),
                )
            )
            results.append(_monotone_restriction(drives))
        if fault_gates:
            results.append(
                _faults_survived(
                    [drive for drive in drives if drive.scenario.is_fault_scenario]
                )
            )
        if plan.no_score is None:
            score = _score(run_implementation_checks(factory), routes)
    results.append(_tree_unchanged(Path(release_root)))
    return GateReport(
        plan_id=plan.plan_id,
        visibility=plan.visibility,
        submission=str(submission),
        submission_sha256=_file_hashes(submission),
        results=tuple(results),
        score=score,
        no_score=plan.no_score,
        scenarios_run=tuple(scenario.scenario_id for scenario in scenarios),
        partial=len(scenarios) < len(plan.scenarios),
    )


def drive_route(
    factory: VehicleControllerFactory,
    scenario: GateScenario,
    *,
    config: AppConfig,
) -> ScenarioScore:
    """Drive one scored route as the gates do, with a controller ``factory``
    builds from ``config``, and score it as the gate report does.

    It measures a controller that is not a submission yet, such as the one a
    course notebook holds.
    """
    if scenario.par_steps is None or scenario.is_fault_scenario:
        raise ValueError(f"{scenario.scenario_id} is not a scored route")
    return _scenario_score(_drive(config, scenario, factory))


def check_gate_plan(plan: GatePlan, base_config: AppConfig) -> None:
    """Refuse a plan whose fault scenarios could not act under ``base_config``."""
    if not base_config.vla.enabled and any(
        scenario.is_fault_scenario for scenario in plan.scenarios
    ):
        raise ValueError(
            f"{plan.plan_id} has fault scenarios, which need a base configuration "
            "that enables vla, for example --config configs/demo-vla-fixture.yaml"
        )


def select_gate_scenarios(
    plan: GatePlan,
    scenario_ids: Sequence[str] | None,
) -> tuple[GateScenario, ...]:
    """The plan's scenarios named by ``scenario_ids``, in manifest order; all without any."""
    if scenario_ids is None:
        return plan.scenarios
    available = [scenario.scenario_id for scenario in plan.scenarios]
    unknown = [scenario_id for scenario_id in scenario_ids if scenario_id not in available]
    if unknown:
        raise ValueError(
            f"unknown gate scenario id: {', '.join(unknown)}; "
            f"{plan.plan_id} has {', '.join(available)}"
        )
    if not scenario_ids:
        raise ValueError("at least one gate scenario id is required")
    return tuple(scenario for scenario in plan.scenarios if scenario.scenario_id in scenario_ids)


def _load_observation(
    submission: Path,
    config: AppConfig,
) -> ObservationBuilderFactory | None:
    """Load the submitted observation builder, when the drives ask the model, and
    build one now, within the request timeout, so a failing or hanging
    constructor fails the load gate rather than every route."""
    factory = load_observation_for(submission, config)
    if factory is not None:
        _build_once(
            submission,
            OBSERVATION_SEAM,
            factory,
            config.observation,
            deadline_s=config.vla.request_timeout_s,
        )
    return factory


def _load_arbitration(
    submission: Path,
    config: AppConfig,
) -> AssessmentArbiterFactory | None:
    """Load the submitted arbiter, when the drives ask the model, and build one
    now, within the request timeout, so a failing or hanging constructor fails
    the load gate rather than every route."""
    factory = load_arbitration_for(submission, config)
    if factory is not None:
        _build_once(
            submission,
            ARBITRATION_SEAM,
            factory,
            config.command_validation,
            deadline_s=config.vla.request_timeout_s,
        )
    return factory


def _build_once(
    submission: Path,
    spec: SeamSpec,
    factory: Callable[[Any], Any],
    settings: Any,
    *,
    deadline_s: float,
) -> None:
    """Build one product of a submitted seam, failing the load if it hangs."""
    try:
        _within(deadline_s, spec.entry_call, factory, settings)
    except _Overrun as overrun:
        raise SubmissionError(f"{submission / spec.filename}: {overrun}") from None


def _drive(
    config: AppConfig,
    scenario: GateScenario,
    factory: VehicleControllerFactory,
    observation_factory: ObservationBuilderFactory | None = None,
    arbitration_factory: AssessmentArbiterFactory | None = None,
) -> _Drive:
    run_config = copy.deepcopy(config)
    simulator = run_config.simulator
    simulator.map = scenario.map
    simulator.start_seed = scenario.seed
    simulator.horizon = scenario.horizon
    simulator.traffic_density = scenario.traffic_density
    simulator.obstacle_probability = scenario.obstacle_probability
    simulator.headless = True
    simulator.manual_control = False
    simulator.realtime = False
    run_config.scenario.stopped_vehicle_ahead_m = scenario.stopped_vehicle_ahead_m
    if scenario.safety_source is not None:
        run_config.perception.safety_source = scenario.safety_source
    run_config.faults = scenario.faults
    run_config.event_log = EventLogSettings()
    run_config.episode_recording = EpisodeRecordingSettings()
    recorders: list[_Recorder] = []

    def recording_factory(settings: ControllerSettings) -> VehicleController:
        recorder = _Recorder(factory(settings))
        recorders.append(recorder)
        return recorder

    observations = None if observation_factory is None else _ObservationLog()
    deadline_s = run_config.vla.request_timeout_s

    def recording_observation_factory(settings: ObservationSettings) -> ObservationBuilder:
        assert observation_factory is not None and observations is not None
        try:
            builder = _within(
                deadline_s, OBSERVATION_SEAM.entry_call, observation_factory, settings
            )
        except _Overrun as overrun:
            observations.overrun = str(overrun)
            raise ObservationError(observations.overrun) from None
        return _ObservationRecorder(builder, observations, deadline_s=deadline_s)

    arbitrations = None if arbitration_factory is None else _ArbitrationLog()

    def recording_arbitration_factory(
        settings: CommandValidationSettings,
    ) -> AssessmentArbiter:
        assert arbitration_factory is not None and arbitrations is not None
        try:
            arbiter = _within(
                deadline_s, ARBITRATION_SEAM.entry_call, arbitration_factory, settings
            )
        except _Overrun as overrun:
            arbitrations.overrun = str(overrun)
            raise ArbitrationError(arbitrations.overrun) from None
        return _ArbitrationRecorder(
            arbiter,
            arbitrations,
            budget_s=control_timestep_s(run_config.simulator),
        )

    try:
        summary = run_simulation(
            run_config,
            controller_factory=recording_factory,
            observation_factory=(
                None if observation_factory is None else recording_observation_factory
            ),
            arbitration_factory=(
                None if arbitration_factory is None else recording_arbitration_factory
            ),
        )
    except Exception as exc:
        # Submitted code may raise anything; the route gate reports it.
        return _Drive(
            scenario, None, exc, _recorded_ticks(recorders), observations, arbitrations
        )
    return _Drive(
        scenario, summary, None, _recorded_ticks(recorders), observations, arbitrations
    )


def _recorded_ticks(recorders: list[_Recorder]) -> tuple[ControlTick, ...]:
    return tuple(tick for recorder in recorders for tick in recorder.ticks)


def _bounded_outputs(drives: list[_Drive]) -> GateResult:
    unusable = [
        f"{drive.scenario.scenario_id}: {drive.error}"
        for drive in drives
        if isinstance(drive.error, ControllerOutputError)
    ]
    if unusable:
        return GateResult(BOUNDED_OUTPUTS, GateStatus.FAILED, "; ".join(unusable))
    clamped = {
        drive.scenario.scenario_id: drive.summary.controller_outputs_clamped
        for drive in drives
        if drive.summary is not None and drive.summary.controller_outputs_clamped
    }
    if clamped:
        counts = ", ".join(f"{scenario}: {count}" for scenario, count in clamped.items())
        return GateResult(
            BOUNDED_OUTPUTS,
            GateStatus.FAILED,
            f"the actuator range clamped {_count(sum(clamped.values()), 'control tick')} "
            f"({counts})",
        )
    return GateResult(
        BOUNDED_OUTPUTS,
        GateStatus.PASSED,
        "every control output stayed within [-1, 1]",
    )


def _runs_finish(drives: list[_Drive]) -> GateResult:
    """The route gate of a plan that scores its routes: only an error fails it.

    A route that crashes, leaves the road, or does not arrive earns no driving
    credit instead, and its score says why.
    """
    if not drives:
        return GateResult(RUNS_FINISH, GateStatus.NOT_RUN, "no route scenario was driven")
    stopped = [
        problem
        for drive in drives
        if drive.error is not None and (problem := _route_problem(drive))
    ]
    if stopped:
        return GateResult(RUNS_FINISH, GateStatus.FAILED, "; ".join(stopped))
    detail = f"{_count(len(drives), 'route')} driven without an error"
    incomplete = [problem for drive in drives if (problem := _route_problem(drive))]
    if incomplete:
        detail += (
            f"; {_count(len(incomplete), 'route')} did not complete, so earned no "
            "driving credit: " + "; ".join(incomplete)
        )
    return GateResult(RUNS_FINISH, GateStatus.PASSED, detail)


def _route_completes(drives: list[_Drive]) -> GateResult:
    if not drives:
        return GateResult(ROUTE_COMPLETES, GateStatus.NOT_RUN, "no route scenario was driven")
    problems = [problem for drive in drives if (problem := _route_problem(drive))]
    if problems:
        return GateResult(ROUTE_COMPLETES, GateStatus.FAILED, "; ".join(problems))
    return GateResult(
        ROUTE_COMPLETES,
        GateStatus.PASSED,
        f"{_count(len(drives), 'scenario')} completed",
    )


def _valid_observations(drives: list[_Drive]) -> GateResult:
    problems: list[str] = []
    for drive in drives:
        log = drive.observations
        assert log is not None
        scenario = drive.scenario.scenario_id
        if log.problems:
            problems.append(
                f"{scenario}: {_count(len(log.problems), 'observation')} could not be "
                f"sent; the first: {log.problems[0]}"
            )
        elif log.emitted == 0 and drive.error is None and log.overrun is None:
            problems.append(f"{scenario}: no observation was built")
    if problems:
        return GateResult(VALID_OBSERVATIONS, GateStatus.FAILED, "; ".join(problems))
    logs = [drive.observations for drive in drives if drive.observations is not None]
    emitted = sum(log.emitted for log in logs)
    without_contract = sum(log.without_contract for log in logs)
    detail = f"{_count(emitted, 'observation')} built, each with a valid frame"
    if not without_contract:
        return GateResult(
            VALID_OBSERVATIONS,
            GateStatus.PASSED,
            f"{detail} and the output contract",
        )
    first_missing = next(log.first_missing_contract for log in logs if log.without_contract)
    # Reported, never failed: the output contract is a recommendation.
    return GateResult(
        VALID_OBSERVATIONS,
        GateStatus.PASSED,
        f"{detail}; in {without_contract} of them the prompt "
        f"{describe_missing_contract(first_missing)}",
    )


def _control_loop_unblocked(
    drives: list[_Drive],
    deadline_s: float,
    budget_s: float,
) -> GateResult:
    overruns = [
        f"{drive.scenario.scenario_id}: {log.overrun}"
        for drive in drives
        for log in (drive.observations, drive.arbitrations)
        if log is not None and log.overrun is not None
    ]
    if overruns:
        return GateResult(CONTROL_LOOP_UNBLOCKED, GateStatus.FAILED, "; ".join(overruns))
    slowest_build_s = max(
        (drive.observations.slowest_s for drive in drives if drive.observations),
        default=0.0,
    )
    slowest_answer_s = max(
        (drive.arbitrations.slowest_s for drive in drives if drive.arbitrations),
        default=0.0,
    )
    # Observations are built beside the control loop, which a slow one starves of
    # model authority; arbitration runs on the loop itself.
    return GateResult(
        CONTROL_LOOP_UNBLOCKED,
        GateStatus.PASSED,
        f"every observation was built within the {deadline_s:g} s request timeout, "
        f"the slowest in {1000.0 * slowest_build_s:.1f} ms; every arbiter call "
        f"returned on the control loop within the {budget_s:g} s control step, "
        f"the slowest in {1000.0 * slowest_answer_s:.1f} ms",
    )


def _monotone_restriction(drives: list[_Drive]) -> GateResult:
    problems = [
        f"{drive.scenario.scenario_id}: "
        f"{_count(len(log.problems), 'answer')} could not stand; the first: {log.problems[0]}"
        for drive in drives
        if (log := drive.arbitrations) is not None and log.problems
    ]
    if problems:
        return GateResult(MONOTONE_RESTRICTION, GateStatus.FAILED, "; ".join(problems))
    logs = [drive.arbitrations for drive in drives if drive.arbitrations is not None]
    answered = sum(log.answered for log in logs)
    reviewed = sum(log.reviewed for log in logs)
    return GateResult(
        MONOTONE_RESTRICTION,
        GateStatus.PASSED,
        f"{_count(answered, 'assessment')} arbitrated and {_count(reviewed, 'tick')} "
        "reviewed, none granting more authority than the model proposed",
    )


def _faults_survived(drives: list[_Drive]) -> GateResult:
    if not drives:
        return GateResult(FAULTS_SURVIVED, GateStatus.NOT_RUN, "no fault scenario was driven")
    problems = [problem for drive in drives if (problem := _fault_problem(drive))]
    if problems:
        return GateResult(FAULTS_SURVIVED, GateStatus.FAILED, "; ".join(problems))
    kinds = sorted({fault.kind.value for drive in drives for fault in drive.scenario.faults})
    return GateResult(
        FAULTS_SURVIVED,
        GateStatus.PASSED,
        f"{_count(len(drives), 'fault scenario')} driven without a crash, an off-road "
        f"exit, or a stopped run, under {', '.join(kinds)}",
    )


def _fault_problem(drive: _Drive) -> str | None:
    """Why a fault scenario broke the run; arriving is not required."""
    scenario = drive.scenario.scenario_id
    if drive.error is not None:
        return f"{scenario}: the run stopped: {type(drive.error).__name__}: {drive.error}"
    summary = drive.summary
    assert summary is not None
    if summary.crashed:
        return f"{scenario}: crashed"
    if summary.went_off_road:
        return f"{scenario}: left the road"
    return None


def _route_problem(drive: _Drive) -> str | None:
    failure = _route_failure(drive)
    return None if failure is None else f"{drive.scenario.scenario_id}: {failure}"


def _route_failure(drive: _Drive) -> str | None:
    if drive.error is not None:
        return f"the run stopped: {type(drive.error).__name__}: {drive.error}"
    summary = drive.summary
    assert summary is not None
    if summary.crashed:
        return "crashed"
    if summary.went_off_road:
        return "left the road"
    if not summary.arrived:
        return f"did not arrive within {drive.scenario.horizon} steps"
    return None


def _score(checks: tuple[CheckResult, ...], drives: list[_Drive]) -> Score:
    scenarios = tuple(_scenario_score(drive) for drive in drives)
    implementation = 100.0 * sum(check.passed for check in checks) / len(checks)
    # A partial run may drive fault scenarios only, which the score never judges.
    driving = (
        100.0 * sum(scenario.credit for scenario in scenarios) / len(scenarios)
        if scenarios
        else 0.0
    )
    return Score(
        total=IMPLEMENTATION_SHARE * implementation + (1.0 - IMPLEMENTATION_SHARE) * driving,
        implementation=implementation,
        driving=driving,
        checks=checks,
        scenarios=scenarios,
    )


def _scenario_score(drive: _Drive) -> ScenarioScore:
    steps = drive.summary.steps if drive.summary is not None else None
    speed_error = _speed_error(drive.ticks)
    lateral_error = _root_mean_square([tick.lateral_error_m for tick in drive.ticks])
    par_steps = drive.scenario.par_steps
    # Only routes are scored, and every route has a par.
    assert par_steps is not None
    credit = 0.0
    problem = _route_failure(drive)
    if problem is None:
        assert steps is not None and speed_error is not None and lateral_error is not None
        credit = (
            TIME_CREDIT.weighted_credit(steps / par_steps)
            + SPEED_CREDIT.weighted_credit(speed_error)
            + LATERAL_CREDIT.weighted_credit(lateral_error)
        )
    return ScenarioScore(
        scenario_id=drive.scenario.scenario_id,
        credit=credit,
        steps=steps,
        par_steps=par_steps,
        speed_error_mps=speed_error,
        lateral_error_m=lateral_error,
        problem=problem,
    )


def _speed_error(ticks: tuple[ControlTick, ...]) -> float | None:
    """Mean |target - speed| once the launch from standstill is over."""
    launched: list[ControlTick] = []
    elapsed_s = 0.0
    for tick in ticks:
        if elapsed_s >= LAUNCH_S:
            launched.append(tick)
        elapsed_s += tick.dt_s
    # A route shorter than the launch is judged over all of it.
    judged = launched or list(ticks)
    if not judged:
        return None
    return sum(abs(tick.target_speed_mps - tick.speed_mps) for tick in judged) / len(judged)


def _root_mean_square(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def write_release_manifest(root: Path | str) -> Path:
    """Record the SHA-256 of every shipped file outside submission/.

    Run it on a clean release tree; the tree gate then detects edits to those
    files, while new files such as run output are ignored. Notebooks are left
    out, because running one rewrites it.
    """
    release = Path(root)
    files = {
        path.relative_to(release).as_posix(): _sha256(path)
        for path in _shipped_files(release)
    }
    manifest = release / RELEASE_MANIFEST
    manifest.write_text(
        json.dumps({"version": 1, "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _shipped_files(release: Path) -> Iterator[Path]:
    for directory, subdirectories, files in os.walk(release):
        current = Path(directory)
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if name not in _GENERATED_DIRECTORIES
            and not (current == release and name == "submission")
        )
        for name in sorted(files):
            if name.endswith(_NOTEBOOK_SUFFIX):
                continue
            if not (current == release and name == RELEASE_MANIFEST):
                yield current / name


def _tree_unchanged(release: Path) -> GateResult:
    manifest = release / RELEASE_MANIFEST
    if not manifest.is_file():
        return GateResult(
            TREE_UNCHANGED,
            GateStatus.NOT_RUN,
            "no release manifest; this check runs in a student release",
        )
    try:
        files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        expected = {str(name): str(digest) for name, digest in files.items()}
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return GateResult(
            TREE_UNCHANGED,
            GateStatus.FAILED,
            f"{RELEASE_MANIFEST} is unreadable: {exc}",
        )
    changes: list[str] = []
    for name, digest in sorted(expected.items()):
        path = release / name
        if not path.is_file():
            changes.append(f"{name} deleted")
        elif _sha256(path) != digest:
            changes.append(f"{name} modified")
    if changes:
        return GateResult(TREE_UNCHANGED, GateStatus.FAILED, "; ".join(changes))
    return GateResult(
        TREE_UNCHANGED,
        GateStatus.PASSED,
        f"{_count(len(expected), 'shipped file')} unchanged",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _file_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        return {}
    return {
        path.name: _sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def load_gate_plan(path: Path | str) -> GatePlan:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("gate manifest must be a mapping")
    _reject_unknown(
        value, {"version", "id", "visibility", "no_score", "scenarios"}, "gate manifest"
    )
    if value.get("version") != GATE_MANIFEST_VERSION:
        raise ValueError(f"gate manifest version must be {GATE_MANIFEST_VERSION}")
    raw_scenarios = value.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ValueError("gate manifest scenarios must be a list")
    return GatePlan(
        plan_id=value.get("id"),
        visibility=value.get("visibility"),
        scenarios=tuple(_scenario(item) for item in raw_scenarios),
        no_score=value.get("no_score"),
    )


def _scenario(value: object) -> GateScenario:
    if not isinstance(value, dict):
        raise ValueError("gate scenarios must be mappings")
    _reject_unknown(
        value,
        {
            "id",
            "map",
            "seed",
            "horizon",
            "par_steps",
            "traffic_density",
            "obstacle_probability",
            "stopped_vehicle_ahead_m",
            "safety_source",
            "faults",
        },
        "gate scenario",
    )
    faults = value.get("faults", [])
    if not isinstance(faults, list):
        raise ValueError("gate scenario faults must be a list")
    return GateScenario(
        scenario_id=value.get("id"),
        map=value.get("map"),
        seed=value.get("seed"),
        horizon=value.get("horizon"),
        par_steps=value.get("par_steps"),
        traffic_density=value.get("traffic_density", 0.0),
        obstacle_probability=value.get("obstacle_probability", 0.0),
        stopped_vehicle_ahead_m=value.get("stopped_vehicle_ahead_m"),
        safety_source=value.get("safety_source"),
        faults=tuple(_fault(fault) for fault in faults),
    )


def _fault(value: object) -> FaultSpec:
    """One fault, written as in an evaluation manifest."""
    if not isinstance(value, dict):
        raise ValueError("gate scenario faults must be mappings")
    _reject_unknown(
        value,
        {"id", "kind", "start_step", "duration_steps", "latency_s"},
        "gate scenario fault",
    )
    try:
        return FaultSpec(
            fault_id=value["id"],
            kind=value["kind"],
            start_step=value["start_step"],
            duration_steps=value["duration_steps"],
            latency_s=value.get("latency_s", 0.0),
        )
    except KeyError as exc:
        raise ValueError(f"gate scenario fault is missing field {exc.args[0]!r}") from None


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _non_negative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
