from __future__ import annotations

import json
import time
from collections import deque

from metadrive_starter.faults import FaultSpec
from metadrive_starter.perception import LocalScene
from metadrive_starter.planning.command_executor import (
    CommandExecutionSource,
    VLACommandExecutor,
)
from metadrive_starter.safety import VLACommandValidator
from metadrive_starter.vla import (
    CameraCaptureError,
    InferenceSubmitDisposition,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    RequestBudget,
    RequestCappedModelProvider,
    RGBFrame,
    VLAInferencePipeline,
    VLAInferenceScheduler,
)
from metadrive_starter.vla_control import VLACommandRuntime


class FakeCamera:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.captures = 0

    def capture(self, env: object, *, timestamp_s: float) -> RGBFrame:
        del env
        self.captures += 1
        if self.error is not None:
            raise self.error
        return RGBFrame(timestamp_s, 1, 1, b"\x01\x02\x03")


class SequenceProvider:
    def __init__(self, actions: list[str | Exception]) -> None:
        self.actions = deque(actions)

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        del timeout_s
        outcome = self.actions.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return ModelResponse(
            request_id=request.request_id,
            text=json.dumps(
                {
                    "scene_summary": "Scripted test scene.",
                    "relevant_hazards": [],
                    "meta_action": outcome,
                    "target_speed_mps": 0.0 if outcome == "STOP" else 20.0,
                    "confidence": 0.9,
                    "brief_justification": "Scripted test decision.",
                }
            ),
            model_id="runtime-test",
            latency_s=0.0,
        )


def _scene(timestamp_s: float) -> LocalScene:
    return LocalScene(
        timestamp_s=timestamp_s,
        ego_speed_mps=5.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        valid=True,
    )


def _runtime(provider: SequenceProvider, camera: FakeCamera):
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(provider),
        minimum_interval_s=0.0,
    )
    runtime = VLACommandRuntime(
        scheduler,
        VLACommandValidator(),
        VLACommandExecutor(35.0),
        run_id="run-1",
        camera=camera,  # type: ignore[arg-type]
    )
    return runtime, scheduler


def _update_until_completion(runtime: VLACommandRuntime, *, now_s: float):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        update = runtime.update(
            object(),
            _scene(now_s),
            now_s=now_s,
            ego_speed_mps=18.0,
            cruise_speed_mps=35.0,
        )
        if update.completion is not None:
            return update
        time.sleep(0.001)
    raise AssertionError("inference did not complete")


def test_runtime_falls_back_while_inference_runs_then_applies_validated_result() -> None:
    runtime, scheduler = _runtime(SequenceProvider(["STOP", "STOP"]), FakeCamera())
    try:
        initial = runtime.update(
            object(),
            _scene(1.0),
            now_s=1.0,
            ego_speed_mps=18.0,
            cruise_speed_mps=35.0,
        )
        completed = _update_until_completion(runtime, now_s=1.1)

        assert initial.execution.source is CommandExecutionSource.LOCAL_FALLBACK
        assert initial.submitted_request_id == "run-1-vla-1"
        assert completed.validation is not None
        assert completed.execution.source is CommandExecutionSource.VLA
        assert completed.execution.target_speed_mps == 0.0
        assert completed.execution.command_id == "run-1-vla-1"
    finally:
        scheduler.close()


def test_runtime_reports_request_cap_once_then_expires_to_local_fallback() -> None:
    camera = FakeCamera()
    budget = RequestBudget(1)
    provider = RequestCappedModelProvider(SequenceProvider(["STOP"]), budget)
    scheduler = VLAInferenceScheduler(
        VLAInferencePipeline(provider),
        minimum_interval_s=0.0,
        request_budget=budget,
    )
    runtime = VLACommandRuntime(
        scheduler,
        VLACommandValidator(),
        VLACommandExecutor(35.0),
        run_id="capped-run",
        camera=camera,  # type: ignore[arg-type]
    )
    try:
        started = runtime.update(
            object(),
            _scene(1.0),
            now_s=1.0,
            ego_speed_mps=18.0,
            cruise_speed_mps=35.0,
        )
        completed = _update_until_completion(runtime, now_s=1.1)
        expired = runtime.update(
            object(),
            _scene(4.0),
            now_s=4.0,
            ego_speed_mps=0.0,
            cruise_speed_mps=35.0,
        )

        assert started.submission_disposition is InferenceSubmitDisposition.STARTED
        assert (
            completed.submission_disposition
            is InferenceSubmitDisposition.REQUEST_CAP_EXHAUSTED
        )
        assert completed.execution.source is CommandExecutionSource.VLA
        assert expired.submission_disposition is None
        assert expired.execution.source is CommandExecutionSource.LOCAL_FALLBACK
        assert camera.captures == 1
        assert budget.request_labels == ("capped-run-vla-1",)
    finally:
        scheduler.close()


def test_runtime_model_failure_retains_previous_authority_until_original_horizon() -> None:
    provider = SequenceProvider(["STOP", ModelProviderError("offline"), "STOP"])
    runtime, scheduler = _runtime(provider, FakeCamera())
    try:
        runtime.update(
            object(), _scene(1.0), now_s=1.0, ego_speed_mps=18.0, cruise_speed_mps=35.0
        )
        first = _update_until_completion(runtime, now_s=1.1)
        assert first.execution.source is CommandExecutionSource.VLA

        failed = _update_until_completion(runtime, now_s=1.2)

        assert failed.completion is not None
        assert failed.validation is None
        assert failed.execution.source is CommandExecutionSource.VLA
        assert failed.execution.command_id == first.execution.command_id
        assert failed.execution.valid_until_s == first.execution.valid_until_s
    finally:
        scheduler.close()


def test_runtime_explicit_fallback_revokes_previous_authority() -> None:
    runtime, scheduler = _runtime(
        SequenceProvider(["STOP", "REQUEST_FALLBACK", "STOP"]),
        FakeCamera(),
    )
    try:
        runtime.update(
            object(), _scene(1.0), now_s=1.0, ego_speed_mps=18.0, cruise_speed_mps=35.0
        )
        active = _update_until_completion(runtime, now_s=1.1)
        assert active.execution.source is CommandExecutionSource.VLA

        fallback = _update_until_completion(runtime, now_s=1.2)

        assert fallback.validation is not None
        assert fallback.validation.disposition.value == "fallback"
        assert fallback.active_validation is None
        assert fallback.execution.source is CommandExecutionSource.LOCAL_FALLBACK
    finally:
        scheduler.close()


def test_runtime_episode_reset_revokes_active_authority() -> None:
    runtime, scheduler = _runtime(SequenceProvider(["STOP", "STOP"]), FakeCamera())
    try:
        runtime.update(
            object(), _scene(1.0), now_s=1.0, ego_speed_mps=18.0, cruise_speed_mps=35.0
        )
        active = _update_until_completion(runtime, now_s=1.1)
        assert active.execution.source is CommandExecutionSource.VLA

        episode_id = runtime.begin_episode()
        reset = runtime.update(
            object(), _scene(0.0), now_s=0.0, ego_speed_mps=0.0, cruise_speed_mps=35.0
        )

        assert episode_id == "run-1-episode-2"
        assert reset.execution.source is CommandExecutionSource.LOCAL_FALLBACK
        assert reset.submission_disposition is InferenceSubmitDisposition.BUSY
        assert reset.submitted_episode_id is None
    finally:
        scheduler.close()


def test_runtime_clears_retained_validation_after_original_horizon() -> None:
    camera = FakeCamera()
    runtime, scheduler = _runtime(SequenceProvider(["STOP"]), camera)
    try:
        runtime.update(
            object(), _scene(1.0), now_s=1.0, ego_speed_mps=18.0, cruise_speed_mps=35.0
        )
        camera.error = CameraCaptureError("camera unavailable")
        active = _update_until_completion(runtime, now_s=1.1)
        assert active.execution.valid_until_s == 3.0

        expired = runtime.update(
            object(), _scene(3.000001), now_s=3.000001, ego_speed_mps=0.0, cruise_speed_mps=35.0
        )

        assert expired.active_validation is None
        assert expired.execution.source is CommandExecutionSource.LOCAL_FALLBACK
    finally:
        scheduler.close()


def test_runtime_camera_failure_falls_back_without_interrupting_control() -> None:
    runtime, scheduler = _runtime(
        SequenceProvider([]),
        FakeCamera(CameraCaptureError("camera unavailable")),
    )
    try:
        update = runtime.update(
            object(), _scene(1.0), now_s=1.0, ego_speed_mps=18.0, cruise_speed_mps=35.0
        )

        assert isinstance(update.runtime_error, CameraCaptureError)
        assert update.execution.source is CommandExecutionSource.LOCAL_FALLBACK
        assert update.submission_disposition is None
    finally:
        scheduler.close()


def test_runtime_injects_scheduled_camera_dropout_before_capture() -> None:
    runtime, scheduler = _runtime(SequenceProvider([]), FakeCamera())
    fault = FaultSpec("camera-loss-1", "camera_dropout", 0, 1)
    try:
        update = runtime.update(
            object(),
            _scene(1.0),
            now_s=1.0,
            ego_speed_mps=18.0,
            cruise_speed_mps=35.0,
            active_faults=(fault,),
        )

        assert isinstance(update.runtime_error, CameraCaptureError)
        assert "injected camera dropout" in str(update.runtime_error)
        assert update.active_fault_ids == ("camera-loss-1",)
        assert update.submission_disposition is None
        assert update.execution.source is CommandExecutionSource.LOCAL_FALLBACK
    finally:
        scheduler.close()
