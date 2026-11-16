import json
import textwrap
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter import submission
from metadrive_starter.config import AppConfig, EventLogSettings, config_from_dict, load_config
from metadrive_starter.events import read_event_log
from metadrive_starter.perception import LocalScene
from metadrive_starter.planning.command_executor import (
    CommandExecutionSource,
    VLACommandExecutor,
)
from metadrive_starter.replay import replay_event_log
from metadrive_starter.safety import VLACommandValidator
from metadrive_starter.simulation import run_simulation
from metadrive_starter.submission import (
    SubmissionError,
    apply_submission_overlay,
    load_request_policy,
    load_request_policy_for,
)
from metadrive_starter.types import ControlTick
from metadrive_starter.vla import (
    DefaultRequestPolicy,
    InferenceSubmitDisposition,
    ModelFailureCategory,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelTimeoutError,
    RequestBudget,
    RequestCappedModelProvider,
    RequestPolicyError,
    RequestPolicySettings,
    RequestTick,
    RGBFrame,
    ScriptedModelProvider,
    VLAInferencePipeline,
    VLAInferenceScheduler,
    decide_request,
)
from metadrive_starter.vla_control import VLACommandRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = PROJECT_ROOT / "configs" / "demo-vla-fixture.yaml"


def _scene(timestamp_s: float) -> LocalScene:
    return LocalScene(
        timestamp_s=timestamp_s,
        ego_speed_mps=5.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
    )


def _tick(**values: object) -> RequestTick:
    fields: dict[str, object] = {
        "now_s": 1.0,
        "scene": _scene(1.0),
        "ego_speed_mps": 5.0,
        "cruise_speed_mps": 9.72,
        "active": None,
        "issued_at_s": None,
        "failure": None,
        "in_flight": False,
        "last_requested_at_s": None,
        "requests_remaining": None,
        "route_speed_cap_mps": None,
    }
    fields.update(values)
    return RequestTick(**fields)  # type: ignore[arg-type]


class _Policy:
    """A request policy that answers each tick with ``answer(tick)``."""

    def __init__(self, answer=lambda tick: True) -> None:
        self.answer = answer
        self.ticks: list[RequestTick] = []
        self.episodes = 0

    def begin_episode(self) -> None:
        self.episodes += 1

    def should_request(self, tick: RequestTick) -> bool:
        self.ticks.append(tick)
        return self.answer(tick)


# Decisions, checked without a runtime.


def test_default_policy_asks_on_every_tick() -> None:
    policy = DefaultRequestPolicy()
    policy.begin_episode()

    assert policy.should_request(_tick()) is True
    assert policy.should_request(_tick(in_flight=True, requests_remaining=0)) is True


@pytest.mark.parametrize("answer", [True, False])
def test_a_true_or_false_answer_stands(answer: bool) -> None:
    decision = decide_request(_Policy(lambda tick: answer), _tick())

    assert decision.request is answer
    assert decision.problem is None


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        (lambda tick: 1, "should_request(tick) must return True or False; got 1"),
        (lambda tick: None, "should_request(tick) must return True or False; got None"),
        (lambda tick: "yes", "should_request(tick) must return True or False; got 'yes'"),
        (lambda tick: 1 / 0, "should_request(tick) raised ZeroDivisionError: division by zero"),
    ],
    ids=["one", "none", "text", "raises"],
)
def test_an_answer_that_cannot_stand_holds_the_request_back(answer, problem: str) -> None:
    decision = decide_request(_Policy(answer), _tick())

    assert decision.request is False
    assert decision.problem == problem


def test_a_request_policy_error_is_reported_in_its_own_words() -> None:
    def refuse(tick: RequestTick) -> bool:
        raise RequestPolicyError("the gates stopped calling it")

    decision = decide_request(_Policy(refuse), _tick())

    assert decision.request is False
    assert decision.problem == "the gates stopped calling it"


def test_a_decision_is_timed() -> None:
    def slow(tick: RequestTick) -> bool:
        time.sleep(0.02)
        return True

    assert decide_request(_Policy(slow), _tick()).elapsed_s >= 0.02


def test_settings_carry_the_instructor_request_limits() -> None:
    vla = config_from_dict(
        {
            "vla": {
                "maximum_requests_per_run": 12,
                "minimum_interval_s": 0.75,
                "request_timeout_s": 3.0,
                "action_horizon_s": 1.5,
            }
        }
    ).vla

    assert RequestPolicySettings.from_vla(vla) == RequestPolicySettings(
        request_budget=12,
        minimum_interval_s=0.75,
        request_timeout_s=3.0,
        action_horizon_s=1.5,
    )


# The runtime: the policy decides on the control loop, before any frame is captured.


def _assessment_text(action: str = "KEEP_LANE", speed_mps: float = 8.0) -> str:
    return json.dumps(
        {
            "scene_summary": "Scripted scene.",
            "relevant_hazards": [],
            "meta_action": action,
            "target_speed_mps": speed_mps,
            "confidence": 0.9,
            "brief_justification": "Scripted decision.",
        }
    )


class _Camera:
    def __init__(self) -> None:
        self.captures = 0

    def capture(self, env: object, *, timestamp_s: float) -> RGBFrame:
        self.captures += 1
        return RGBFrame(timestamp_s, 1, 1, b"\x01\x02\x03")


def _runtime(
    provider,
    policy=None,
    *,
    minimum_interval_s: float = 0.0,
    budget: RequestBudget | None = None,
    arbiter=None,
) -> VLACommandRuntime:
    if budget is not None:
        provider = RequestCappedModelProvider(provider, budget)
    return VLACommandRuntime(
        VLAInferenceScheduler(
            VLAInferencePipeline(provider),
            minimum_interval_s=minimum_interval_s,
            request_budget=budget,
        ),
        VLACommandValidator(),
        VLACommandExecutor(9.72),
        run_id="run-1",
        camera=_Camera(),  # type: ignore[arg-type]
        arbiter=arbiter,
        request_policy=policy,
    )


def _update(runtime: VLACommandRuntime, now_s: float, **values: object):
    return runtime.update(
        object(),
        _scene(now_s),
        now_s=now_s,
        ego_speed_mps=5.0,
        cruise_speed_mps=9.72,
        **values,  # type: ignore[arg-type]
    )


def _collect(runtime: VLACommandRuntime, now_s: float):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        update = _update(runtime, now_s)
        if update.completion is not None:
            return update
        time.sleep(0.001)
    raise AssertionError("inference did not complete")


def test_a_held_tick_captures_no_frame_and_asks_nothing() -> None:
    provider = ScriptedModelProvider([_assessment_text()])
    runtime = _runtime(provider, _Policy(lambda tick: False))
    try:
        updates = [_update(runtime, 1.0 + 0.1 * step) for step in range(5)]
    finally:
        runtime.scheduler.close()

    assert {update.submission_disposition for update in updates} == {
        InferenceSubmitDisposition.HELD
    }
    assert runtime.camera.captures == 0  # type: ignore[attr-defined]
    assert provider.requests == ()


def test_asking_cannot_beat_the_minimum_interval() -> None:
    provider = ScriptedModelProvider([_assessment_text(), _assessment_text()])
    runtime = _runtime(provider, _Policy(), minimum_interval_s=0.5)
    try:
        first = _update(runtime, 1.0)
        _collect(runtime, 1.1)
        early = _update(runtime, 1.2)
        due = _update(runtime, 1.5)
    finally:
        runtime.scheduler.close()

    assert first.submission_disposition is InferenceSubmitDisposition.STARTED
    assert early.submission_disposition is InferenceSubmitDisposition.RATE_LIMITED
    assert due.submission_disposition is InferenceSubmitDisposition.STARTED


def test_asking_on_every_tick_never_exceeds_the_budget() -> None:
    provider = ScriptedModelProvider([_assessment_text()] * 5)
    runtime = _runtime(provider, _Policy(), budget=RequestBudget(2))
    dispositions = []
    try:
        for step in range(40):
            update = _update(runtime, 1.0 + 0.1 * step)
            dispositions.append(update.submission_disposition)
            time.sleep(0.002)
    finally:
        runtime.scheduler.close()

    assert len(provider.requests) == 2
    assert dispositions.count(InferenceSubmitDisposition.STARTED) == 2
    assert dispositions.count(InferenceSubmitDisposition.REQUEST_CAP_EXHAUSTED) == 1


class _HeldProvider:
    """Answers each request only once released, so a request stays in flight."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        self.release.wait(5.0)
        return ModelResponse(
            request_id=request.request_id,
            text=_assessment_text(),
            model_id="held-model",
            latency_s=0.0,
        )


def test_the_policy_sees_a_request_in_flight_as_spent() -> None:
    provider = _HeldProvider()
    policy = _Policy(lambda tick: tick.last_requested_at_s is None)
    runtime = _runtime(provider, policy, budget=RequestBudget(3))
    try:
        _update(runtime, 1.0)
        # However far the worker has got, the count is the same.
        time.sleep(0.05)
        _update(runtime, 1.1)
        provider.release.set()
        collected = _collect(runtime, 1.2)
    finally:
        runtime.scheduler.close()

    first, in_flight, *_, settled = policy.ticks
    assert (first.in_flight, first.requests_remaining, first.last_requested_at_s) == (
        False,
        3,
        None,
    )
    assert (in_flight.in_flight, in_flight.requests_remaining) == (True, 2)
    assert in_flight.last_requested_at_s == 1.0
    assert collected.completion is not None
    assert (settled.in_flight, settled.requests_remaining) == (False, 2)
    assert settled.active is not None and settled.issued_at_s == 1.0


@pytest.mark.parametrize(
    ("error", "failure"),
    [
        (ModelTimeoutError("slow"), ModelFailureCategory.TIMEOUT),
        (ModelProviderError("429", category=ModelFailureCategory.QUOTA), ModelFailureCategory.QUOTA),
        (
            ModelProviderError("503", category=ModelFailureCategory.MODEL_UNAVAILABLE),
            ModelFailureCategory.MODEL_UNAVAILABLE,
        ),
        ("Sure! I'd keep the lane.", ModelFailureCategory.INVALID_RESPONSE),
    ],
    ids=["timeout", "quota", "outage", "invalid-response"],
)
def test_the_policy_sees_why_a_request_failed_on_the_tick_it_is_collected(
    error: Exception | str,
    failure: ModelFailureCategory,
) -> None:
    policy = _Policy(lambda tick: tick.last_requested_at_s is None)
    runtime = _runtime(ScriptedModelProvider([error]), policy)
    try:
        _update(runtime, 1.0)
        _collect(runtime, 1.1)
        _update(runtime, 1.2)
    finally:
        runtime.scheduler.close()

    failures = [tick.failure for tick in policy.ticks]
    assert failures[-2] is failure
    assert failures[-1] is None
    assert failures.count(failure) == 1


class _EndingArbiter:
    def begin_episode(self) -> None:
        pass

    def arbitrate(self, request):
        return request.assessment

    def review(self, request) -> bool:
        return False


def test_the_policy_sees_the_command_that_holds_authority_after_review() -> None:
    kept = _Policy(lambda tick: tick.last_requested_at_s is None)
    ended = _Policy(lambda tick: tick.last_requested_at_s is None)
    keeping = _runtime(ScriptedModelProvider([_assessment_text()]), kept)
    ending = _runtime(ScriptedModelProvider([_assessment_text()]), ended, arbiter=_EndingArbiter())
    try:
        for runtime in (keeping, ending):
            _update(runtime, 1.0)
            _collect(runtime, 1.1)
    finally:
        keeping.scheduler.close()
        ending.scheduler.close()

    assert kept.ticks[-1].active is not None
    assert kept.ticks[-1].active.proposed_target_speed_mps == 8.0
    # The review ended the command on the tick it arrived, before the policy decided.
    assert ended.ticks[-1].active is None
    assert ended.ticks[-1].issued_at_s is None


def test_the_policy_sees_the_curve_speed_limit_it_is_given() -> None:
    policy = _Policy(lambda tick: False)
    runtime = _runtime(ScriptedModelProvider([]), policy)
    try:
        _update(runtime, 1.0, route_speed_cap_mps=6.5)
        _update(runtime, 1.1)
    finally:
        runtime.scheduler.close()

    assert [tick.route_speed_cap_mps for tick in policy.ticks] == [6.5, None]


def test_every_episode_begins_with_a_reset_policy() -> None:
    policy = _Policy(lambda tick: tick.last_requested_at_s is None)
    runtime = _runtime(ScriptedModelProvider([_assessment_text(), _assessment_text()]), policy)
    try:
        _update(runtime, 1.0)
        _collect(runtime, 1.1)
        runtime.begin_episode()
        _update(runtime, 0.0)
    finally:
        runtime.scheduler.close()

    assert policy.episodes == 2
    assert policy.ticks[-1].last_requested_at_s is None


def test_a_policy_that_raises_holds_the_request_and_the_vehicle_drives_on() -> None:
    def raises(tick: RequestTick) -> bool:
        raise RuntimeError("history lost")

    provider = ScriptedModelProvider([_assessment_text()])
    runtime = _runtime(provider, _Policy(raises))
    try:
        update = _update(runtime, 1.0)
    finally:
        runtime.scheduler.close()

    assert update.request is not None
    assert update.request.problem == "should_request(tick) raised RuntimeError: history lost"
    assert update.submission_disposition is InferenceSubmitDisposition.HELD
    assert update.execution.source is CommandExecutionSource.LOCAL_FALLBACK
    assert provider.requests == ()


def test_runtime_rejects_a_policy_without_the_protocol() -> None:
    with pytest.raises(TypeError, match="RequestPolicy"):
        _runtime(ScriptedModelProvider([]), object())


# Loading request_policy.py.

SUBMITTED_POLICY = """
    class RequestPolicy:
        def __init__(self, settings):
            self.settings = settings

        def begin_episode(self):
            pass

        def should_request(self, tick):
            return True
"""

SETTINGS = RequestPolicySettings(
    request_budget=20,
    minimum_interval_s=0.5,
    request_timeout_s=1.0,
    action_horizon_s=2.0,
)


def _submission(directory: Path, **files: str) -> Path:
    for name, source in files.items():
        (directory / f"{name}.py").write_text(textwrap.dedent(source))
    return directory


def test_submitted_policy_is_built_from_the_request_limits(tmp_path: Path) -> None:
    policy = load_request_policy(_submission(tmp_path, request_policy=SUBMITTED_POLICY))(SETTINGS)

    assert policy.settings is SETTINGS
    assert policy.should_request(_tick()) is True


def test_missing_request_policy_file_names_the_expected_policy(tmp_path: Path) -> None:
    with pytest.raises(
        SubmissionError,
        match=(
            r"request_policy\.py not found; expected a file defining "
            r"RequestPolicy\(settings\) returning a RequestPolicy with "
            r"begin_episode\(\) and should_request\(tick\)"
        ),
    ):
        load_request_policy(tmp_path)


def test_request_policy_file_loads_only_for_a_run_that_asks_the_model(tmp_path: Path) -> None:
    broken = _submission(tmp_path, request_policy="raise RuntimeError('never imported')")

    assert load_request_policy_for(broken, config_from_dict({})) is None
    with pytest.raises(SubmissionError, match="request_policy.py"):
        load_request_policy_for(broken, config_from_dict({"vla": {"enabled": True}}))


def test_policy_without_should_request_names_the_expected_method(tmp_path: Path) -> None:
    source = SUBMITTED_POLICY.replace(
        "        def should_request(self, tick):\n            return True\n", ""
    )
    factory = load_request_policy(_submission(tmp_path, request_policy=source))

    with pytest.raises(
        SubmissionError,
        match=(
            r"returned a RequestPolicy without should_request\(tick\); "
            r"expected RequestPolicy\(settings\)"
        ),
    ):
        factory(SETTINGS)


# agent.yaml may slow the requests, never speed them up.


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


def test_agent_yaml_slows_the_requests_the_policy_and_the_floor_see(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text("vla:\n  minimum_interval_s: 1.25\n")
    built: list[RequestPolicySettings] = []

    def factory(settings: RequestPolicySettings) -> DefaultRequestPolicy:
        built.append(settings)
        return DefaultRequestPolicy()

    config = apply_submission_overlay(_fixture_config(), tmp_path)
    run_simulation(
        config,
        dry_run=True,
        controller_factory=lambda settings: _NeutralController(),
        request_policy_factory=factory,
    )

    assert config.vla.minimum_interval_s == 1.25
    assert built == [RequestPolicySettings.from_vla(config.vla)]
    assert built[0].minimum_interval_s == 1.25


@pytest.mark.parametrize(
    ("overlay", "message"),
    [
        ("minimum_interval_s: 0.1", r"vla\.minimum_interval_s may only be raised from 0\.5"),
        ("maximum_requests_per_run: 100", r"vla\.maximum_requests_per_run is not permitted"),
    ],
    ids=["faster", "budget"],
)
def test_agent_yaml_never_asks_faster_or_for_more(
    tmp_path: Path,
    overlay: str,
    message: str,
) -> None:
    (tmp_path / "agent.yaml").write_text(f"vla:\n  {overlay}\n")

    with pytest.raises(SubmissionError, match=message):
        apply_submission_overlay(_fixture_config(), tmp_path)


# Runs.


def test_run_without_the_vla_subsystem_builds_no_policy() -> None:
    built: list[RequestPolicySettings] = []

    summary = run_simulation(
        config_from_dict({}),
        dry_run=True,
        controller_factory=lambda settings: _NeutralController(),
        request_policy_factory=lambda settings: built.append(settings) or DefaultRequestPolicy(),
    )

    assert built == []
    assert summary.request_policy is None


def test_vla_run_without_a_submitted_policy_asks_whenever_it_may(
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

    assert summary.request_policy == "metadrive_starter.vla.request_policy.DefaultRequestPolicy"


SILENT_POLICY = """
    class RequestPolicy:
        def __init__(self, settings):
            pass

        def begin_episode(self):
            pass

        def should_request(self, tick):
            return False
"""


def test_a_policy_that_never_asks_keeps_the_model_out_and_the_log_replays(
    tmp_path: Path,
) -> None:
    config = _fixture_config()
    config.simulator.horizon = 12
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "events.jsonl"),
        scenario_id="request-policy-seam",
    )
    team = _submission(tmp_path, request_policy=SILENT_POLICY)

    summary = run_simulation(
        config,
        controller_factory=lambda settings: _NeutralController(),
        request_policy_factory=load_request_policy(team),
    )

    metrics = summary.vla_metrics
    assert summary.request_policy == "submission.request_policy.RequestPolicy"
    assert metrics.requests_started == 0
    assert metrics.submissions_held == summary.steps == 12
    assert metrics.submissions_rate_limited == 0
    assert metrics.request_policy_problems == 0
    assert metrics.request_policy_slowest_s is not None
    assert metrics.authority_steps == 0
    events = [record.event_type for record in read_event_log(config.event_log.path)]
    assert "inference_submitted" not in events
    replay = replay_event_log(config.event_log.path)
    assert replay.successful, replay.to_dict()


def test_a_policy_that_raises_is_logged_and_the_run_completes(tmp_path: Path) -> None:
    config = _fixture_config()
    config.simulator.horizon = 6
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "events.jsonl"),
        scenario_id="request-policy-problem",
    )
    team = _submission(
        tmp_path,
        request_policy=SILENT_POLICY.replace("return False", "raise KeyError('lead')"),
    )

    summary = run_simulation(
        config,
        controller_factory=lambda settings: _NeutralController(),
        request_policy_factory=load_request_policy(team),
    )

    metrics = summary.vla_metrics
    assert summary.steps == 6
    assert metrics.requests_started == 0
    assert metrics.request_policy_problems == 6
    problems = [
        record.payload["decision"]
        for record in read_event_log(config.event_log.path)
        if record.event_type == "request_policy_problem"
    ]
    assert len(problems) == 6
    assert problems[0]["problem"] == "should_request(tick) raised KeyError: 'lead'"
    assert problems[0]["request"] is False
