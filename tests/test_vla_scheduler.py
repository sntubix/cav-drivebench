from __future__ import annotations

import json
import time
from collections import deque
from threading import Event, Lock

import pytest

from metadrive_starter.vla.camera import RGBFrame
from metadrive_starter.vla.pipeline import VLAInferencePipeline
from metadrive_starter.vla.provider import (
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelTimeoutError,
    RequestBudget,
    RequestCappedModelProvider,
)
from metadrive_starter.vla.scheduler import (
    InferenceDiscarded,
    InferenceFailure,
    InferenceSubmitDisposition,
    InferenceSuccess,
    VLAInferenceScheduler,
)


def _frame(timestamp_s: float) -> RGBFrame:
    return RGBFrame(
        timestamp_s=timestamp_s,
        width=1,
        height=1,
        rgb_bytes=b"\x01\x02\x03",
    )


def _response(request: ModelRequest) -> ModelResponse:
    return ModelResponse(
        request_id=request.request_id,
        text=json.dumps(
            {
                "scene_summary": "Clear road.",
                "relevant_hazards": [],
                "meta_action": "KEEP_LANE",
                "target_speed_mps": 20.0,
                "confidence": 0.9,
                "brief_justification": "No hazards visible.",
            }
        ),
        model_id="scheduler-test",
        latency_s=0.0,
    )


class BlockingProvider:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.finished = Event()
        self._lock = Lock()
        self._active = 0
        self.maximum_active = 0

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        with self._lock:
            self._active += 1
            self.maximum_active = max(self.maximum_active, self._active)
        self.started.set()
        if not self.release.wait(timeout=1.0):
            raise ModelTimeoutError("test provider was not released")
        try:
            return _response(request)
        finally:
            with self._lock:
                self._active -= 1
            self.finished.set()


class SequenceProvider:
    def __init__(self, outcomes: list[Exception | None]) -> None:
        self._outcomes = deque(outcomes)

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        if not self._outcomes:
            raise ModelProviderError("test outcomes exhausted")
        outcome = self._outcomes.popleft()
        if outcome is not None:
            raise outcome
        return _response(request)


def _await_completion(
    scheduler: VLAInferenceScheduler,
) -> InferenceSuccess | InferenceFailure:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        completion = scheduler.poll()
        if completion is not None:
            return completion
        Event().wait(0.001)
    raise AssertionError("inference did not complete")


def test_scheduler_is_single_flight_and_does_not_queue_frames() -> None:
    provider = BlockingProvider()
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(provider),
        minimum_interval_s=1.0,
    )
    try:
        assert (
            scheduler.submit(_frame(10.0), now_s=10.0, request_id="first")
            is InferenceSubmitDisposition.STARTED
        )
        assert provider.started.wait(timeout=1.0)
        assert (
            scheduler.submit(_frame(11.0), now_s=11.0, request_id="not-queued")
            is InferenceSubmitDisposition.BUSY
        )

        provider.release.set()
        assert provider.finished.wait(timeout=1.0)
        completion = _await_completion(scheduler)

        assert isinstance(completion, InferenceSuccess)
        assert completion.request_id == "first"
        assert completion.episode_id == "default"
        assert completion.generation_id == 1
        assert completion.result.request.episode_id == "default"
        assert completion.result.request.generation_id == 1
        assert scheduler.poll() is None
        assert provider.maximum_active == 1
    finally:
        provider.release.set()
        scheduler.close()


def test_uncollected_completion_keeps_the_bounded_slot_busy() -> None:
    provider = BlockingProvider()
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(provider),
        minimum_interval_s=0.5,
    )
    try:
        scheduler.submit(_frame(10.0), now_s=10.0, request_id="first")
        provider.release.set()
        assert provider.finished.wait(timeout=1.0)

        assert (
            scheduler.submit(_frame(11.0), now_s=11.0, request_id="second")
            is InferenceSubmitDisposition.BUSY
        )
        assert isinstance(_await_completion(scheduler), InferenceSuccess)
    finally:
        provider.release.set()
        scheduler.close()


def test_scheduler_rate_limits_using_simulation_time() -> None:
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(SequenceProvider([None, None, None])),
        minimum_interval_s=1.0,
    )
    try:
        assert (
            scheduler.submit(_frame(10.0), now_s=10.0, request_id="first")
            is InferenceSubmitDisposition.STARTED
        )
        assert isinstance(_await_completion(scheduler), InferenceSuccess)
        assert (
            scheduler.submit(_frame(10.9), now_s=10.9, request_id="too-soon")
            is InferenceSubmitDisposition.RATE_LIMITED
        )
        assert (
            scheduler.submit(_frame(11.0), now_s=11.0, request_id="second")
            is InferenceSubmitDisposition.STARTED
        )
        assert isinstance(_await_completion(scheduler), InferenceSuccess)

        # Clock rollback resets rate limiting; episode identity changes only via begin_episode().
        assert (
            scheduler.submit(_frame(0.0), now_s=0.0, request_id="new-episode")
            is InferenceSubmitDisposition.STARTED
        )
        assert isinstance(_await_completion(scheduler), InferenceSuccess)
    finally:
        scheduler.close()


def test_episode_change_quarantines_old_completion_and_restarts_generation() -> None:
    provider = BlockingProvider()
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(provider),
        minimum_interval_s=10.0,
    )
    try:
        assert (
            scheduler.submit(_frame(5.0), now_s=5.0, request_id="old")
            is InferenceSubmitDisposition.STARTED
        )
        assert provider.started.wait(timeout=1.0)
        scheduler.begin_episode("episode-2")
        provider.release.set()

        discarded = _await_completion(scheduler)
        assert isinstance(discarded, InferenceDiscarded)
        assert discarded.episode_id == "default"
        assert discarded.generation_id == 1
        assert discarded.result is not None

        assert (
            scheduler.submit(_frame(0.0), now_s=0.0, request_id="new")
            is InferenceSubmitDisposition.STARTED
        )
        accepted = _await_completion(scheduler)
        assert isinstance(accepted, InferenceSuccess)
        assert accepted.episode_id == "episode-2"
        assert accepted.generation_id == 1
        assert accepted.result.request.episode_id == "episode-2"
        assert accepted.result.request.generation_id == 1
    finally:
        provider.release.set()
        scheduler.close()


@pytest.mark.parametrize("episode_id", ["", "bad\nepisode", 7])
def test_scheduler_rejects_invalid_episode_id(episode_id: object) -> None:
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(SequenceProvider([])),
        minimum_interval_s=0.0,
    )
    try:
        with pytest.raises(ValueError, match="episode_id"):
            scheduler.begin_episode(episode_id)  # type: ignore[arg-type]
    finally:
        scheduler.close()


def test_scheduler_rejects_episode_id_reuse() -> None:
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(SequenceProvider([])),
        minimum_interval_s=0.0,
    )
    try:
        scheduler.begin_episode("episode-2")
        scheduler.begin_episode("episode-3")
        with pytest.raises(ValueError, match="reused"):
            scheduler.begin_episode("episode-2")
    finally:
        scheduler.close()


def test_failure_is_returned_once_and_frees_the_slot() -> None:
    timeout = ModelTimeoutError("provider deadline exceeded")
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(SequenceProvider([timeout, None])),
        minimum_interval_s=0.5,
    )
    try:
        assert (
            scheduler.submit(_frame(10.0), now_s=10.0, request_id="failed")
            is InferenceSubmitDisposition.STARTED
        )
        completion = _await_completion(scheduler)

        assert isinstance(completion, InferenceFailure)
        assert completion.request_id == "failed"
        assert completion.error is timeout
        assert scheduler.poll() is None
        assert (
            scheduler.submit(_frame(10.5), now_s=10.5, request_id="recovered")
            is InferenceSubmitDisposition.STARTED
        )
        assert isinstance(_await_completion(scheduler), InferenceSuccess)
    finally:
        scheduler.close()


def test_close_is_idempotent_rejects_submissions_and_preserves_completion() -> None:
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(SequenceProvider([None])),
        minimum_interval_s=1.0,
    )
    assert (
        scheduler.submit(_frame(10.0), now_s=10.0, request_id="before-close")
        is InferenceSubmitDisposition.STARTED
    )

    scheduler.close()
    scheduler.close()

    assert (
        scheduler.submit(_frame(11.0), now_s=11.0, request_id="after-close")
        is InferenceSubmitDisposition.CLOSED
    )
    assert isinstance(scheduler.poll(), InferenceSuccess)
    assert scheduler.poll() is None


def test_context_manager_closes_scheduler() -> None:
    pipeline = VLAInferencePipeline(SequenceProvider([]))
    with VLAInferenceScheduler(pipeline, minimum_interval_s=1.0) as scheduler:
        pass

    assert (
        scheduler.submit(_frame(1.0), now_s=1.0, request_id="closed")
        is InferenceSubmitDisposition.CLOSED
    )


def test_scheduler_allows_zero_minimum_interval() -> None:
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(SequenceProvider([None, None])),
        minimum_interval_s=0.0,
    )
    try:
        assert (
            scheduler.submit(_frame(1.0), now_s=1.0, request_id="first")
            is InferenceSubmitDisposition.STARTED
        )
        assert isinstance(_await_completion(scheduler), InferenceSuccess)
        assert (
            scheduler.submit(_frame(1.0), now_s=1.0, request_id="second")
            is InferenceSubmitDisposition.STARTED
        )
        assert isinstance(_await_completion(scheduler), InferenceSuccess)
    finally:
        scheduler.close()


def test_request_cap_stops_new_work_and_survives_episode_changes() -> None:
    underlying = SequenceProvider([None, None])
    budget = RequestBudget(1)
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(RequestCappedModelProvider(underlying, budget)),
        minimum_interval_s=0.0,
        request_budget=budget,
    )
    try:
        assert (
            scheduler.submit(_frame(1.0), now_s=1.0, request_id="only-request")
            is InferenceSubmitDisposition.STARTED
        )
        assert isinstance(_await_completion(scheduler), InferenceSuccess)
        assert budget.used_requests == 1
        assert scheduler.request_cap_exhausted is True
        assert (
            scheduler.submit(_frame(2.0), now_s=2.0, request_id="blocked")
            is InferenceSubmitDisposition.REQUEST_CAP_EXHAUSTED
        )

        scheduler.begin_episode("episode-after-cap")
        assert (
            scheduler.submit(_frame(0.0), now_s=0.0, request_id="still-blocked")
            is InferenceSubmitDisposition.REQUEST_CAP_EXHAUSTED
        )
        assert budget.request_labels == ("only-request",)
    finally:
        scheduler.close()


@pytest.mark.parametrize("value", [-1.0, True, float("nan"), float("inf")])
def test_scheduler_rejects_invalid_minimum_interval(value: float) -> None:
    with pytest.raises(ValueError, match="minimum_interval_s"):
        VLAInferenceScheduler(
            VLAInferencePipeline(SequenceProvider([])),
            minimum_interval_s=value,
        )


@pytest.mark.parametrize("value", [-1.0, True, float("nan"), float("inf")])
def test_scheduler_rejects_invalid_simulation_time(value: float) -> None:
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(SequenceProvider([])),
        minimum_interval_s=1.0,
    )
    try:
        with pytest.raises(ValueError, match="now_s"):
            scheduler.submit(_frame(0.0), now_s=value, request_id="invalid")
    finally:
        scheduler.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame", object(), "frame"),
        ("request_id", "", "request_id"),
        ("request_id", 7, "request_id"),
        ("request_id", "bad\nheader", "control characters"),
        ("scene", object(), "scene"),
        ("ego_speed_mps", -1.0, "ego_speed_mps"),
        ("cruise_speed_mps", float("nan"), "cruise_speed_mps"),
    ],
)
def test_scheduler_rejects_invalid_submission_before_occupying_slot(
    field: str,
    value: object,
    message: str,
) -> None:
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(SequenceProvider([None])),
        minimum_interval_s=1.0,
    )
    kwargs: dict[str, object] = {
        "frame": _frame(1.0),
        "now_s": 1.0,
        "request_id": "valid-request",
        "scene": None,
        "ego_speed_mps": 0.0,
        "cruise_speed_mps": 35.0,
    }
    kwargs[field] = value
    try:
        with pytest.raises(ValueError, match=message):
            scheduler.submit(**kwargs)  # type: ignore[arg-type]
        assert (
            scheduler.submit(_frame(1.0), now_s=1.0, request_id="valid-request")
            is InferenceSubmitDisposition.STARTED
        )
        assert isinstance(_await_completion(scheduler), InferenceSuccess)
    finally:
        scheduler.close()
