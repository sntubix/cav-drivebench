import json
import math
import textwrap
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from metadrive_starter import submission
from metadrive_starter.config import AppConfig, EventLogSettings, config_from_dict, load_config
from metadrive_starter.events import read_event_log
from metadrive_starter.perception import LaneRelation, LocalScene, TrackedObject
from metadrive_starter.planning.command_executor import (
    CommandExecutionSource,
    VLACommandExecutor,
)
from metadrive_starter.replay import replay_event_log
from metadrive_starter.safety import (
    CommandDisposition,
    CommandValidationSettings,
    VLACommandValidator,
)
from metadrive_starter.simulation import run_simulation
from metadrive_starter.submission import (
    SubmissionError,
    apply_submission_overlay,
    load_arbitration,
    load_arbitration_for,
)
from metadrive_starter.types import ControlTick
from metadrive_starter.vla import (
    ArbitrationError,
    ArbitrationOutcome,
    ArbitrationRequest,
    DefaultArbiter,
    HazardType,
    HighLevelAction,
    ModelFailureCategory,
    ModelProviderError,
    ModelTimeoutError,
    RelativeLocation,
    ReviewRequest,
    RGBFrame,
    RiskLevel,
    ScriptedModelProvider,
    VLAAssessment,
    VLAHazard,
    VLAInferencePipeline,
    VLAInferenceScheduler,
    arbitrate_assessment,
    declined,
    monotone_violations,
    review_command,
)
from metadrive_starter.vla_control import VLACommandRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = PROJECT_ROOT / "configs" / "demo-vla-fixture.yaml"
FALLBACK = HighLevelAction.REQUEST_FALLBACK
HAZARD = VLAHazard(HazardType.VEHICLE, RelativeLocation.FRONT, RiskLevel.HIGH)
PROPOSAL = VLAAssessment(
    scene_summary="A stopped vehicle ahead.",
    relevant_hazards=(HAZARD,),
    proposed_action=HighLevelAction.KEEP_LANE,
    proposed_target_speed_mps=9.72,
    confidence=0.9,
    brief_justification="The lane looks clear.",
)


def _scene(timestamp_s: float, *objects: TrackedObject) -> LocalScene:
    return LocalScene(
        timestamp_s=timestamp_s,
        ego_speed_mps=5.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        objects=objects,
    )


STOPPED_LEAD = TrackedObject(
    object_id="lead",
    kind="vehicle",
    relative_position_m=(22.0, 0.0),
    relative_velocity_mps=(-5.0, 0.0),
    length_m=4.5,
    width_m=1.8,
    lane_relation=LaneRelation.SAME,
    in_path=True,
    path_distance_m=22.0,
    path_relative_velocity_mps=-5.0,
)


def _request(assessment: VLAAssessment = PROPOSAL) -> ArbitrationRequest:
    return ArbitrationRequest(
        assessment=assessment,
        scene=_scene(10.0),
        now_s=10.0,
        issued_at_s=9.4,
        cruise_speed_mps=9.72,
    )


class _Answers:
    """An arbiter that answers each assessment with ``answer(request)`` and each
    review with ``keep(request)``."""

    def __init__(self, answer, keep=lambda request: True) -> None:
        self.answer = answer
        self.keep = keep
        self.requests: list[ArbitrationRequest] = []
        self.reviews: list[ReviewRequest] = []
        self.episodes = 0

    def begin_episode(self) -> None:
        self.episodes += 1

    def arbitrate(self, request: ArbitrationRequest) -> VLAAssessment:
        self.requests.append(request)
        return self.answer(request)

    def review(self, request: ReviewRequest) -> bool:
        self.reviews.append(request)
        return self.keep(request)


# The monotone restriction, checked without any scene.


@pytest.mark.parametrize(
    "answer",
    [
        PROPOSAL,
        replace(PROPOSAL, proposed_target_speed_mps=4.0),
        replace(PROPOSAL, proposed_target_speed_mps=0),
        replace(PROPOSAL, confidence=0.0),
        replace(PROPOSAL, proposed_action=FALLBACK),
        replace(PROPOSAL, proposed_target_speed_mps=Fraction(1, 2)),
        # Descriptive fields never reach the vehicle, so rewriting them grants nothing.
        replace(PROPOSAL, relevant_hazards=(), brief_justification="Declined."),
    ],
    ids=["same", "slower", "stop", "no-confidence", "fallback", "any-real", "descriptive"],
)
def test_an_answer_within_the_proposal_has_no_violations(answer: VLAAssessment) -> None:
    assert monotone_violations(PROPOSAL, answer) == ()


@pytest.mark.parametrize(
    ("answer", "violation"),
    [
        (
            replace(PROPOSAL, proposed_target_speed_mps=13.9),
            "proposed_target_speed_mps raised from 9.72 to 13.9 m/s",
        ),
        (replace(PROPOSAL, confidence=0.95), "confidence raised from 0.9 to 0.95"),
        (
            replace(PROPOSAL, proposed_action=HighLevelAction.STOP),
            "proposed_action changed from KEEP_LANE to STOP; "
            "only REQUEST_FALLBACK may replace it",
        ),
        (
            replace(PROPOSAL, proposed_action="REQUEST_FALLBACK"),
            "proposed_action must be a HighLevelAction; got 'REQUEST_FALLBACK'",
        ),
        (
            replace(PROPOSAL, proposed_target_speed_mps=math.nan),
            "proposed_target_speed_mps must be finite and at least 0; got nan",
        ),
        (
            replace(PROPOSAL, proposed_target_speed_mps=-1.0),
            "proposed_target_speed_mps must be finite and at least 0; got -1.0",
        ),
        (replace(PROPOSAL, confidence=True), "confidence must be a number; got True"),
        (None, "arbitrate(request) must return a VLAAssessment; got a NoneType"),
        (
            PROPOSAL.to_command(command_id="c", issued_at_s=0.0, action_horizon_s=2.0),
            "arbitrate(request) must return a VLAAssessment; got a VLACommand",
        ),
    ],
    ids=[
        "faster",
        "more-confident",
        "another-action",
        "action-as-text",
        "nan-speed",
        "negative-speed",
        "boolean",
        "nothing",
        "a-command",
    ],
)
def test_an_answer_beyond_the_proposal_names_each_violation(
    answer: object,
    violation: str,
) -> None:
    assert monotone_violations(PROPOSAL, answer) == (violation,)


def test_every_violation_in_one_answer_is_named() -> None:
    answer = replace(
        PROPOSAL,
        proposed_action=HighLevelAction.OVERTAKE,
        proposed_target_speed_mps=12.0,
        confidence=1.0,
    )

    assert len(monotone_violations(PROPOSAL, answer)) == 3


def test_a_model_proposing_fallback_can_only_be_endorsed() -> None:
    proposal = replace(PROPOSAL, proposed_action=FALLBACK)

    assert monotone_violations(proposal, proposal) == ()
    assert monotone_violations(proposal, replace(proposal, proposed_action=HighLevelAction.STOP))


# What the runtime does with an answer.


def test_default_arbiter_endorses_every_assessment() -> None:
    decision = arbitrate_assessment(DefaultArbiter(), _request())

    assert decision.outcome is ArbitrationOutcome.ENDORSED
    assert decision.assessment == PROPOSAL
    assert decision.problem is None


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        (replace(PROPOSAL, proposed_target_speed_mps=4.0), ArbitrationOutcome.RESTRICTED),
        (replace(PROPOSAL, confidence=0.4), ArbitrationOutcome.RESTRICTED),
        (declined(PROPOSAL), ArbitrationOutcome.DECLINED),
    ],
)
def test_an_answer_within_the_proposal_is_what_the_floor_receives(
    answer: VLAAssessment,
    outcome: ArbitrationOutcome,
) -> None:
    decision = arbitrate_assessment(_Answers(lambda request: answer), _request())

    assert decision.outcome is outcome
    assert decision.assessment == answer
    assert decision.problem is None


def test_only_action_speed_and_confidence_of_an_answer_reach_the_floor() -> None:
    answer = replace(
        PROPOSAL,
        scene_summary="Rewritten.",
        relevant_hazards=(),
        proposed_target_speed_mps=Fraction(7, 2),
        brief_justification="Rewritten.",
    )

    decision = arbitrate_assessment(_Answers(lambda request: answer), _request())

    assert decision.assessment == replace(PROPOSAL, proposed_target_speed_mps=3.5)
    assert type(decision.assessment.proposed_target_speed_mps) is float


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        (
            lambda request: replace(request.assessment, confidence=1.0),
            "confidence raised from 0.9 to 1",
        ),
        (lambda request: 1 / 0, "arbitrate(request) raised ZeroDivisionError: division by zero"),
        (lambda request: {"meta_action": "STOP"}, "must return a VLAAssessment; got a dict"),
    ],
    ids=["more-authority", "raises", "not-an-assessment"],
)
def test_an_answer_that_cannot_stand_is_declined_for_the_arbiter(answer, problem: str) -> None:
    decision = arbitrate_assessment(_Answers(answer), _request())

    assert decision.outcome is ArbitrationOutcome.DECLINED
    assert decision.assessment == declined(PROPOSAL)
    assert decision.problem is not None and problem in decision.problem


def test_an_arbitration_error_is_reported_in_its_own_words() -> None:
    def refuse(request: ArbitrationRequest) -> VLAAssessment:
        raise ArbitrationError("the arbiter already ran past the control step")

    decision = arbitrate_assessment(_Answers(refuse), _request())

    assert decision.problem == "the arbiter already ran past the control step"


def test_arbitration_is_timed() -> None:
    def slow(request: ArbitrationRequest) -> VLAAssessment:
        time.sleep(0.02)
        return request.assessment

    decision = arbitrate_assessment(_Answers(slow), _request())

    assert decision.elapsed_s >= 0.02


# Reviews: keep or end the command that holds authority, every tick.


def _review(active: VLAAssessment | None = PROPOSAL) -> ReviewRequest:
    return ReviewRequest(
        active=active,
        issued_at_s=None if active is None else 9.4,
        scene=_scene(10.0),
        now_s=10.0,
        cruise_speed_mps=9.72,
        failure=None,
    )


@pytest.mark.parametrize(
    ("keep", "active", "revoked"),
    [
        (True, PROPOSAL, False),
        (False, PROPOSAL, True),
        # With no model command there is nothing to end.
        (False, None, False),
    ],
)
def test_a_review_keeps_or_ends_the_model_command(
    keep: bool,
    active: VLAAssessment | None,
    revoked: bool,
) -> None:
    decision = review_command(_Answers(None, lambda request: keep), _review(active))

    assert decision.revoked is revoked
    assert decision.problem is None
    assert DefaultArbiter().review(_review(active)) is True


@pytest.mark.parametrize(
    ("keep", "problem"),
    [
        (lambda request: None, "review(request) must return True or False; got None"),
        (lambda request: 1, "review(request) must return True or False; got 1"),
        (lambda request: 1 / 0, "review(request) raised ZeroDivisionError: division by zero"),
    ],
    ids=["nothing", "truthy", "raises"],
)
def test_a_review_that_cannot_stand_ends_the_model_command(keep, problem: str) -> None:
    decision = review_command(_Answers(None, keep), _review())
    idle = review_command(_Answers(None, keep), _review(active=None))

    assert (decision.revoked, decision.problem) == (True, problem)
    # Reported even when no command holds authority, so the gate still sees it.
    assert (idle.revoked, idle.problem) == (False, problem)


# The runtime: arbitration on the control loop, upstream of command derivation.


def _assessment_text(action: str, speed_mps: float, *, hazards: list[dict] = ()) -> str:
    return json.dumps(
        {
            "scene_summary": "Scripted scene.",
            "relevant_hazards": list(hazards),
            "meta_action": action,
            "target_speed_mps": speed_mps,
            "confidence": 0.9,
            "brief_justification": "Scripted decision.",
        }
    )


class _Camera:
    def capture(self, env: object, *, timestamp_s: float) -> RGBFrame:
        return RGBFrame(timestamp_s, 1, 1, b"\x01\x02\x03")


def _runtime(provider: ScriptedModelProvider, arbiter=None) -> VLACommandRuntime:
    return VLACommandRuntime(
        VLAInferenceScheduler(VLAInferencePipeline(provider), minimum_interval_s=0.0),
        VLACommandValidator(),
        VLACommandExecutor(9.72),
        run_id="run-1",
        camera=_Camera(),  # type: ignore[arg-type]
        arbiter=arbiter,
    )


def _collect(runtime: VLACommandRuntime, scene: LocalScene):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        update = runtime.update(
            object(),
            scene,
            now_s=scene.timestamp_s,
            ego_speed_mps=5.0,
            cruise_speed_mps=9.72,
        )
        if update.completion is not None:
            return update
        time.sleep(0.001)
    raise AssertionError("inference did not complete")


def test_the_arbiter_receives_the_reported_hazards_and_the_scene_measured_at_collection() -> None:
    arbiter = _Answers(lambda request: request.assessment)
    hazard = {"type": "vehicle", "relative_location": "front", "risk": "high"}
    runtime = _runtime(
        ScriptedModelProvider([_assessment_text("KEEP_LANE", 9.72, hazards=[hazard])]),
        arbiter,
    )
    try:
        runtime.update(object(), _scene(1.0), now_s=1.0, ego_speed_mps=5.0, cruise_speed_mps=9.72)
        collected_scene = _scene(1.1, STOPPED_LEAD)
        _collect(runtime, collected_scene)
    finally:
        runtime.scheduler.close()

    [request] = arbiter.requests
    assert request.assessment.relevant_hazards == (HAZARD,)
    assert request.scene is collected_scene
    assert (request.now_s, request.issued_at_s, request.cruise_speed_mps) == (1.1, 1.0, 9.72)


def test_without_an_arbiter_the_floor_validates_the_model_command_unchanged() -> None:
    runtime = _runtime(ScriptedModelProvider([_assessment_text("KEEP_LANE", 8.0)]))
    try:
        runtime.update(object(), _scene(1.0), now_s=1.0, ego_speed_mps=5.0, cruise_speed_mps=9.72)
        update = _collect(runtime, _scene(1.1))
    finally:
        runtime.scheduler.close()

    assert isinstance(runtime.arbiter, DefaultArbiter)
    assert update.arbitration is not None
    assert update.arbitration.outcome is ArbitrationOutcome.ENDORSED
    assert update.validation is not None
    assert update.validation.requested_command == update.completion.result.requested_command


def test_a_restricted_command_keeps_its_locally_owned_metadata() -> None:
    arbiter = _Answers(lambda request: replace(request.assessment, proposed_target_speed_mps=3.0))
    runtime = _runtime(ScriptedModelProvider([_assessment_text("KEEP_LANE", 8.0)]), arbiter)
    try:
        runtime.update(object(), _scene(1.0), now_s=1.0, ego_speed_mps=5.0, cruise_speed_mps=9.72)
        update = _collect(runtime, _scene(1.1))
    finally:
        runtime.scheduler.close()

    proposed = update.completion.result.requested_command
    assert update.validation is not None
    assert update.validation.requested_command == replace(proposed, target_speed_mps=3.0)
    assert update.execution.source is CommandExecutionSource.VLA
    assert update.execution.target_speed_mps == 3.0


def test_a_declined_assessment_revokes_model_authority() -> None:
    answers = iter([lambda assessment: assessment, declined])
    arbiter = _Answers(lambda request: next(answers)(request.assessment))
    runtime = _runtime(
        ScriptedModelProvider([_assessment_text("KEEP_LANE", 8.0)] * 2),
        arbiter,
    )
    try:
        runtime.update(object(), _scene(1.0), now_s=1.0, ego_speed_mps=5.0, cruise_speed_mps=9.72)
        endorsed = _collect(runtime, _scene(1.1))
        refused = _collect(runtime, _scene(1.2))
    finally:
        runtime.scheduler.close()

    assert endorsed.execution.source is CommandExecutionSource.VLA
    assert refused.arbitration is not None
    assert refused.arbitration.outcome is ArbitrationOutcome.DECLINED
    assert refused.validation is not None
    assert refused.validation.disposition is CommandDisposition.FALLBACK
    assert refused.validation.reasons == ("VLA requested local fallback",)
    # The endorsed command's horizon had not ended; declining still ends its authority.
    assert refused.execution.source is CommandExecutionSource.LOCAL_FALLBACK


def test_an_arbiter_granting_more_authority_is_declined_at_runtime() -> None:
    arbiter = _Answers(
        lambda request: replace(request.assessment, proposed_target_speed_mps=30.0)
    )
    runtime = _runtime(ScriptedModelProvider([_assessment_text("KEEP_LANE", 8.0)]), arbiter)
    try:
        runtime.update(object(), _scene(1.0), now_s=1.0, ego_speed_mps=5.0, cruise_speed_mps=9.72)
        update = _collect(runtime, _scene(1.1))
    finally:
        runtime.scheduler.close()

    assert update.arbitration is not None
    assert update.arbitration.problem == "proposed_target_speed_mps raised from 8 to 30 m/s"
    assert update.validation is not None
    assert update.validation.disposition is CommandDisposition.FALLBACK
    assert update.execution.source is CommandExecutionSource.LOCAL_FALLBACK


def test_every_episode_begins_with_a_reset_arbiter() -> None:
    arbiter = _Answers(lambda request: request.assessment)
    runtime = _runtime(ScriptedModelProvider([]), arbiter)
    try:
        assert arbiter.episodes == 1
        runtime.begin_episode()
        assert arbiter.episodes == 2
    finally:
        runtime.scheduler.close()


def test_every_tick_is_reviewed_with_the_command_that_holds_authority() -> None:
    arbiter = _Answers(lambda request: replace(request.assessment, confidence=0.8))
    runtime = _runtime(ScriptedModelProvider([_assessment_text("KEEP_LANE", 8.0)]), arbiter)
    try:
        waiting = runtime.update(
            object(), _scene(1.0), now_s=1.0, ego_speed_mps=5.0, cruise_speed_mps=9.72
        )
        collected = _collect(runtime, _scene(1.1))
    finally:
        runtime.scheduler.close()

    first, *_, last = arbiter.reviews
    assert waiting.review is not None and not waiting.review.revoked
    assert (first.active, first.issued_at_s, first.failure) == (None, None, None)
    # Reviewed after the arbitration on the same tick, so the arbitrated
    # assessment is already the one that holds authority.
    assert collected.arbitration is not None
    assert last.active == collected.arbitration.assessment
    assert last.active.confidence == 0.8
    assert (last.issued_at_s, last.now_s) == (1.0, 1.1)
    assert collected.review is not None and not collected.review.revoked
    assert len(arbiter.requests) == 1


@pytest.mark.parametrize(
    ("error", "failure"),
    [
        (ModelTimeoutError("no answer"), ModelFailureCategory.TIMEOUT),
        (ModelProviderError("offline"), ModelFailureCategory.TRANSPORT),
        ("Sure! I'd keep the lane.", ModelFailureCategory.INVALID_RESPONSE),
    ],
    ids=["timeout", "provider-error", "malformed-output"],
)
def test_a_review_can_end_a_command_when_the_model_fails(
    error: Exception | str,
    failure: ModelFailureCategory,
) -> None:
    arbiter = _Answers(
        lambda request: request.assessment,
        keep=lambda request: request.failure is None,
    )
    runtime = _runtime(
        ScriptedModelProvider([_assessment_text("KEEP_LANE", 8.0), error]),
        arbiter,
    )
    try:
        runtime.update(object(), _scene(1.0), now_s=1.0, ego_speed_mps=5.0, cruise_speed_mps=9.72)
        endorsed = _collect(runtime, _scene(1.1))
        failed = _collect(runtime, _scene(1.2))
    finally:
        runtime.scheduler.close()

    assert endorsed.execution.source is CommandExecutionSource.VLA
    assert arbiter.reviews[-1].failure is failure
    assert failed.review is not None and failed.review.revoked
    # Without the review the endorsed command would hold authority until 3.0 s.
    assert failed.active_validation is None
    assert failed.execution.source is CommandExecutionSource.LOCAL_FALLBACK


def test_a_review_that_raises_ends_the_command() -> None:
    def keep(request: ReviewRequest) -> bool:
        if request.active is not None:
            raise RuntimeError("history lost")
        return True

    runtime = _runtime(
        ScriptedModelProvider([_assessment_text("KEEP_LANE", 8.0)]),
        _Answers(lambda request: request.assessment, keep),
    )
    try:
        runtime.update(object(), _scene(1.0), now_s=1.0, ego_speed_mps=5.0, cruise_speed_mps=9.72)
        update = _collect(runtime, _scene(1.1))
    finally:
        runtime.scheduler.close()

    assert update.review is not None
    assert update.review.problem == "review(request) raised RuntimeError: history lost"
    assert update.execution.source is CommandExecutionSource.LOCAL_FALLBACK


def test_runtime_rejects_an_arbiter_without_the_protocol() -> None:
    with pytest.raises(TypeError, match="AssessmentArbiter"):
        _runtime(ScriptedModelProvider([]), object())


# Loading arbitration.py.

SUBMITTED_ARBITER = """
    class Arbiter:
        def __init__(self, settings):
            self.settings = settings

        def begin_episode(self):
            pass

        def arbitrate(self, request):
            return request.assessment

        def review(self, request):
            return True
"""


def _submission(directory: Path, **files: str) -> Path:
    for name, source in files.items():
        (directory / f"{name}.py").write_text(textwrap.dedent(source))
    return directory


def test_submitted_arbiter_is_built_from_the_floor_thresholds(tmp_path: Path) -> None:
    settings = CommandValidationSettings(minimum_confidence=0.7)

    arbiter = load_arbitration(_submission(tmp_path, arbitration=SUBMITTED_ARBITER))(settings)

    assert arbiter.settings is settings
    assert arbiter.arbitrate(_request()) is PROPOSAL


def test_missing_arbitration_file_names_the_expected_arbiter(tmp_path: Path) -> None:
    with pytest.raises(
        SubmissionError,
        match=(
            r"arbitration\.py not found; expected a file defining Arbiter\(settings\) "
            r"returning an AssessmentArbiter with begin_episode\(\), arbitrate\(request\) "
            r"and review\(request\)"
        ),
    ):
        load_arbitration(tmp_path)


def test_arbitration_file_loads_only_for_a_run_that_asks_the_model(tmp_path: Path) -> None:
    broken = _submission(tmp_path, arbitration="raise RuntimeError('never imported')")

    assert load_arbitration_for(broken, config_from_dict({})) is None
    with pytest.raises(SubmissionError, match="arbitration.py"):
        load_arbitration_for(broken, config_from_dict({"vla": {"enabled": True}}))


def test_arbiter_without_review_names_the_expected_method(tmp_path: Path) -> None:
    source = SUBMITTED_ARBITER.replace(
        "        def review(self, request):\n            return True\n", ""
    )
    factory = load_arbitration(_submission(tmp_path, arbitration=source))

    with pytest.raises(
        SubmissionError,
        match=r"returned an Arbiter without review\(request\); expected Arbiter\(settings\)",
    ):
        factory(CommandValidationSettings())


def test_arbiter_without_begin_episode_names_the_expected_method(tmp_path: Path) -> None:
    source = SUBMITTED_ARBITER.replace(
        "        def begin_episode(self):\n            pass\n", ""
    )
    factory = load_arbitration(_submission(tmp_path, arbitration=source))

    with pytest.raises(
        SubmissionError,
        match=r"returned an Arbiter without begin_episode\(\); expected Arbiter\(settings\)",
    ):
        factory(CommandValidationSettings())


def test_agent_yaml_tightens_the_thresholds_the_arbiter_receives(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text(
        "command_validation:\n  minimum_confidence: 0.8\n  lane_change_front_gap_m: 20\n"
    )
    built: list[CommandValidationSettings] = []

    def factory(settings: CommandValidationSettings) -> DefaultArbiter:
        built.append(settings)
        return DefaultArbiter()

    config = apply_submission_overlay(_fixture_config(), tmp_path)
    run_simulation(
        config,
        dry_run=True,
        controller_factory=lambda settings: _NeutralController(),
        arbitration_factory=factory,
    )

    assert built == [
        replace(
            _fixture_config().command_validation,
            minimum_confidence=0.8,
            lane_change_front_gap_m=20.0,
        )
    ]


@pytest.mark.parametrize(
    ("overlay", "message"),
    [
        ("minimum_confidence: 0.3", r"minimum_confidence may only be raised from 0\.5"),
        ("maximum_target_speed_mps: 20", r"maximum_target_speed_mps may only be lowered"),
        ("lane_change_minimum_ttc_s: 1", r"lane_change_minimum_ttc_s may only be raised"),
        ("minimum_confidence: 1.5", r"minimum_confidence must be at most 1"),
    ],
)
def test_agent_yaml_never_loosens_the_floor(tmp_path: Path, overlay: str, message: str) -> None:
    (tmp_path / "agent.yaml").write_text(f"command_validation:\n  {overlay}\n")

    with pytest.raises(SubmissionError, match=message):
        apply_submission_overlay(config_from_dict({}), tmp_path)


# Runs.


class _NeutralController:
    def update(self, tick: ControlTick) -> tuple[float, float]:
        return (0.0, 0.0)

    def reset_speed_control(self) -> None:
        pass


def _fixture_config() -> AppConfig:
    """The fixture-provider demo, logging nothing, runnable from any directory."""
    config = load_config(FIXTURE_CONFIG)
    config.event_log = EventLogSettings()
    config.vla = replace(
        config.vla,
        fixture=replace(
            config.vla.fixture,
            path=str(PROJECT_ROOT / config.vla.fixture.path),
        ),
    )
    return config


def test_run_without_the_vla_subsystem_builds_no_arbiter() -> None:
    built: list[CommandValidationSettings] = []

    summary = run_simulation(
        config_from_dict({}),
        dry_run=True,
        controller_factory=lambda settings: _NeutralController(),
        arbitration_factory=lambda settings: built.append(settings) or DefaultArbiter(),
    )

    assert built == []
    assert summary.arbiter is None


def test_vla_run_without_a_submitted_arbiter_endorses_everything(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A student release has no course/instructor/, and needs none for this.
    monkeypatch.setattr(submission, "INSTRUCTOR_DIR", tmp_path / "absent")

    summary = run_simulation(
        _fixture_config(),
        dry_run=True,
        controller_factory=lambda settings: _NeutralController(),
    )

    assert summary.arbiter == "metadrive_starter.vla.arbitration.DefaultArbiter"


DECLINING_ARBITER = """
    from metadrive_starter.vla import declined

    class Arbiter:
        def __init__(self, settings):
            pass

        def begin_episode(self):
            pass

        def arbitrate(self, request):
            return declined(request.assessment)

        def review(self, request):
            return True
"""


@pytest.mark.needs("fixtures")
def test_a_declining_arbiter_keeps_the_model_from_driving_and_the_log_replays(
    tmp_path: Path,
) -> None:
    config = _fixture_config()
    config.simulator.horizon = 12
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "events.jsonl"),
        scenario_id="arbitration-seam",
    )
    team = _submission(tmp_path, arbitration=DECLINING_ARBITER)

    summary = run_simulation(
        config,
        controller_factory=lambda settings: _NeutralController(),
        arbitration_factory=load_arbitration(team),
    )

    metrics = summary.vla_metrics
    assert summary.arbiter == "submission.arbitration.Arbiter"
    assert metrics.responses_succeeded > 0
    assert metrics.arbitrations_declined == metrics.responses_succeeded
    assert metrics.arbitration_problems == 0
    assert metrics.validations_fallback == metrics.responses_succeeded
    assert metrics.authority_steps == 0
    validations = [
        record.payload
        for record in read_event_log(config.event_log.path)
        if record.event_type == "command_validation"
    ]
    assert validations
    for payload in validations:
        # The model's assessment stays on record, beside what arbitration left.
        assert payload["assessment"]["proposed_action"] == "KEEP_LANE"
        assert payload["arbitration"]["outcome"] == "declined"
        assert payload["requested_command"]["action"] == "REQUEST_FALLBACK"
    replay = replay_event_log(config.event_log.path)
    assert replay.successful, replay.to_dict()
    assert replay.validation_events == len(validations)


ENDING_ARBITER = """
    class Arbiter:
        def __init__(self, settings):
            pass

        def begin_episode(self):
            pass

        def arbitrate(self, request):
            return request.assessment

        def review(self, request):
            return False
"""


@pytest.mark.needs("fixtures")
def test_a_review_that_ends_every_command_is_logged_and_the_log_replays(
    tmp_path: Path,
) -> None:
    config = _fixture_config()
    config.simulator.horizon = 12
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "events.jsonl"),
        scenario_id="arbitration-review",
    )
    team = _submission(tmp_path, arbitration=ENDING_ARBITER)

    summary = run_simulation(
        config,
        controller_factory=lambda settings: _NeutralController(),
        arbitration_factory=load_arbitration(team),
    )

    metrics = summary.vla_metrics
    assert metrics.arbitrations_endorsed == metrics.responses_succeeded > 0
    assert metrics.arbitration_revocations == metrics.responses_succeeded
    assert metrics.arbitration_problems == 0
    assert metrics.authority_steps == 0
    reviews = [
        record.payload["review"]
        for record in read_event_log(config.event_log.path)
        if record.event_type == "arbitration_review"
    ]
    assert len(reviews) == metrics.arbitration_revocations
    assert all(review["revoked"] for review in reviews)
    replay = replay_event_log(config.event_log.path)
    assert replay.successful, replay.to_dict()
