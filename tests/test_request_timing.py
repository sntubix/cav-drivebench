"""A request policy driven on a scripted road, in simulated time."""

import threading
from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter.config import AppConfig, load_config
from metadrive_starter.faults import FaultKind
from metadrive_starter.perception import LocalScene
from metadrive_starter.request_timing import FaultWindow, drive_request_policy
from metadrive_starter.vla import (
    DefaultRequestPolicy,
    ModelFailureCategory,
    ModelRequest,
    ModelResponse,
    RGBFrame,
    VLAInferencePipeline,
    VLAInferenceScheduler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(**vla: object) -> AppConfig:
    config = load_config(PROJECT_ROOT / "configs" / "demo-vla-fixture.yaml")
    values = {
        "action_horizon_s": 4.0,
        "minimum_interval_s": 0.5,
        "request_timeout_s": 6.0,
        "maximum_requests_per_run": 60,
        **vla,
    }
    return replace(
        config,
        vla=replace(
            config.vla,
            fixture=replace(config.vla.fixture, path=str(PROJECT_ROOT / config.vla.fixture.path)),
            **values,
        ),
        command_validation=replace(config.command_validation, maximum_command_age_s=2.5),
    )


def _empty_road(now_s: float) -> LocalScene:
    return LocalScene(
        timestamp_s=now_s,
        ego_speed_mps=5.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
    )


def _drive(policy=None, *, config=None, latency_s=1.4, duration_s=10.0, faults=()):
    return drive_request_policy(
        DefaultRequestPolicy() if policy is None else policy,
        _config() if config is None else config,
        _empty_road,
        duration_s=duration_s,
        answer_latency_s=latency_s,
        faults=faults,
    )


class _Policy:
    def __init__(self, answer) -> None:
        self.answer = answer
        self.ticks = []
        self.episodes = 0

    def begin_episode(self) -> None:
        self.episodes += 1

    def should_request(self, tick):
        self.ticks.append(tick)
        return self.answer(tick)


def test_each_answer_arrives_its_latency_after_the_request_and_holds_to_the_next() -> None:
    drive = _drive()

    sent = [0.0, 1.4, 2.8, 4.2, 5.6, 7.0, 8.4, 9.8]
    assert [request.sent_s for request in drive.requests] == sent
    # The last answer would arrive after the drive ends.
    assert [request.collected_s for request in drive.requests] == [*sent[1:], None]
    assert {request.failure for request in drive.requests} == {None}
    # Each command holds from its answer until the next answer replaces it.
    assert [request.authority for request in drive.requests[:2]] == [(1.4, 2.8), (2.8, 4.2)]
    assert drive.requests[-2].authority == (9.8, 10.0)
    assert drive.requests[-1].authority is None
    assert len(drive.ticks) == 100
    assert drive.ticks[0].disposition == "started"
    assert drive.ticks[1].disposition == "busy"


def test_a_command_ends_at_its_horizon_and_the_policy_sees_it_end() -> None:
    policy = _Policy(lambda tick: tick.last_requested_at_s is None)
    drive = _drive(policy)

    (request,) = drive.requests
    # Asked at 0 s, so its 4 s horizon has passed on the first tick after 4 s.
    assert request.authority == (1.4, 4.1)
    seen = {round(tick.now_s, 1): tick for tick in policy.ticks}
    assert seen[4.0].active is not None and seen[4.0].issued_at_s == 0.0
    assert seen[4.1].active is None
    assert policy.episodes == 1


def test_the_minimum_interval_paces_answers_that_arrive_at_once() -> None:
    drive = _drive(latency_s=0.0, duration_s=2.0)

    assert [request.sent_s for request in drive.requests] == [0.0, 0.5, 1.0, 1.5]
    assert [request.collected_s for request in drive.requests] == [0.1, 0.6, 1.1, 1.6]


def test_a_request_an_outage_covers_is_refused_on_the_next_tick_and_costs_budget() -> None:
    outage = FaultWindow(FaultKind.PROVIDER_OUTAGE, 2.0, 4.0)
    policy = _Policy(lambda tick: True)
    drive = _drive(policy, faults=[outage], duration_s=6.0)

    refused = [request for request in drive.requests if request.failure is not None]
    assert [(request.sent_s, request.collected_s) for request in refused] == [
        (2.8, 2.9),
        (3.3, 3.4),
        (3.8, 3.9),
    ]
    assert {request.failure for request in refused} == {ModelFailureCategory.MODEL_UNAVAILABLE}
    assert {request.faults for request in refused} == {(FaultKind.PROVIDER_OUTAGE,)}
    assert all(request.authority is None for request in refused)
    # The policy sees each failure on the tick it is collected.
    failures = {round(tick.now_s, 1): tick.failure for tick in policy.ticks if tick.failure is not None}
    assert failures == {2.9: ModelFailureCategory.MODEL_UNAVAILABLE, 3.4: ModelFailureCategory.MODEL_UNAVAILABLE, 3.9: ModelFailureCategory.MODEL_UNAVAILABLE}
    assert drive.ticks[-1].requests_remaining == 60 - len(drive.requests)


def test_quota_refuses_every_request_to_the_end() -> None:
    quota = FaultWindow(FaultKind.PROVIDER_QUOTA, 3.0)
    drive = _drive(faults=[quota])

    late = [request for request in drive.requests if request.sent_s >= 3.0]
    assert late and {request.failure for request in late} == {ModelFailureCategory.QUOTA}


def test_a_drive_never_asks_beyond_its_budget() -> None:
    drive = _drive(config=_config(maximum_requests_per_run=3))

    assert len(drive.requests) == 3
    assert drive.request_budget == 3
    dispositions = [tick.disposition for tick in drive.ticks]
    assert dispositions.count("request_cap_exhausted") == 1
    assert drive.ticks[-1].requests_remaining == 0


def test_an_answer_later_than_the_request_timeout_fails_as_a_timeout() -> None:
    drive = _drive(config=_config(request_timeout_s=1.0), duration_s=3.0)

    first = drive.requests[0]
    assert (first.collected_s, first.failure) == (1.0, ModelFailureCategory.TIMEOUT)


def test_an_answer_that_cannot_stand_is_held_back_and_recorded() -> None:
    drive = _drive(_Policy(lambda tick: "yes"), duration_s=1.0)

    assert drive.requests == ()
    assert {tick.problem for tick in drive.ticks} == {
        "should_request(tick) must return True or False; got 'yes'"
    }
    assert {tick.asked for tick in drive.ticks} == {False}


def test_a_timing_drive_answers_from_recordings_only() -> None:
    config = _config()
    vertex = replace(config, vla=replace(config.vla, provider="vertex"))

    with pytest.raises(ValueError, match="vla.provider must be 'fixture'"):
        _drive(config=vertex)


@pytest.mark.parametrize(
    ("kind", "start_s", "end_s", "problem"),
    [
        (FaultKind.CAMERA_DROPOUT, 1.0, None, "must be one of: provider_outage, provider_quota"),
        (FaultKind.PROVIDER_QUOTA, 2.0, 2.0, "end_s must be None or later than start_s"),
        (FaultKind.PROVIDER_OUTAGE, -1.0, None, "start_s must be finite and non-negative"),
    ],
)
def test_a_fault_window_is_an_outage_or_quota_over_a_real_interval(kind, start_s, end_s, problem) -> None:
    with pytest.raises(ValueError, match=problem):
        FaultWindow(kind, start_s, end_s)


class _Gated:
    def __init__(self) -> None:
        self.release = threading.Event()

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        self.release.wait(5.0)
        return ModelResponse(
            request_id=request.request_id,
            text=(
                '{"scene_summary":"s","relevant_hazards":[],"meta_action":"KEEP_LANE",'
                '"target_speed_mps":5.0,"confidence":0.9,"brief_justification":"j"}'
            ),
            model_id="gated",
            latency_s=0.0,
        )


def test_the_scheduler_waits_for_its_outstanding_inference_only_when_asked() -> None:
    provider = _Gated()
    scheduler = VLAInferenceScheduler(VLAInferencePipeline(provider), minimum_interval_s=0.0)
    try:
        assert scheduler.wait_for_completion(0.0) is False
        scheduler.submit(RGBFrame(0.0, 1, 1, b"\x00\x00\x00"), now_s=0.0, request_id="r-1")
        assert scheduler.wait_for_completion(0.01) is False
        provider.release.set()
        assert scheduler.wait_for_completion(5.0) is True
        assert scheduler.poll() is not None
        assert scheduler.wait_for_completion(0.0) is False
    finally:
        scheduler.close()
