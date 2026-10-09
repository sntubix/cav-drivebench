from __future__ import annotations

from dataclasses import dataclass

from metadrive_starter.faults import (
    FaultInjectingModelProvider,
    FaultKind,
    FaultSpec,
    has_fault,
)
from metadrive_starter.perception import LocalScene
from metadrive_starter.planning.command_executor import (
    CommandExecutionDecision,
    VLACommandExecutor,
)
from metadrive_starter.safety import (
    CommandDisposition,
    CommandValidationDecision,
    HighLevelSafetyPolicy,
)
from metadrive_starter.timing import exceeds_time_limit
from metadrive_starter.vla import (
    ArbitrationDecision,
    ArbitrationRequest,
    AssessmentArbiter,
    CameraCaptureError,
    DefaultArbiter,
    ReviewDecision,
    ReviewRequest,
    VLAAssessment,
    InferenceCompletion,
    InferenceFailure,
    InferenceDiscarded,
    InferenceSubmitDisposition,
    InferenceSuccess,
    MetaDriveCameraAdapter,
    VLAInferenceScheduler,
    arbitrate_assessment,
    model_failure_category,
    review_command,
)


@dataclass(frozen=True)
class VLAControlUpdate:
    execution: CommandExecutionDecision
    validation: CommandValidationDecision | None
    active_validation: CommandValidationDecision | None
    completion: InferenceCompletion | None
    submission_disposition: InferenceSubmitDisposition | None
    submitted_request_id: str | None
    submitted_episode_id: str | None
    submitted_generation_id: int | None
    active_fault_ids: tuple[str, ...] = ()
    runtime_error: Exception | None = None
    # The collected assessment as arbitrated; None when none was collected.
    arbitration: ArbitrationDecision | None = None
    # This tick's review of the model command that held authority.
    review: ReviewDecision | None = None


class VLACommandRuntime:
    """Non-blocking bridge from camera inference to validated speed objectives."""

    def __init__(
        self,
        scheduler: VLAInferenceScheduler,
        policy: HighLevelSafetyPolicy,
        executor: VLACommandExecutor,
        *,
        run_id: str,
        camera: MetaDriveCameraAdapter | None = None,
        fault_provider: FaultInjectingModelProvider | None = None,
        arbiter: AssessmentArbiter | None = None,
    ) -> None:
        if not isinstance(scheduler, VLAInferenceScheduler):
            raise TypeError("scheduler must be a VLAInferenceScheduler")
        if not isinstance(policy, HighLevelSafetyPolicy):
            raise TypeError("policy must implement HighLevelSafetyPolicy.validate()")
        if not isinstance(executor, VLACommandExecutor):
            raise TypeError("executor must be a VLACommandExecutor")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must not be empty")
        if arbiter is None:
            arbiter = DefaultArbiter()
        elif not isinstance(arbiter, AssessmentArbiter):
            raise TypeError(
                "arbiter must implement AssessmentArbiter.begin_episode() and arbitrate()"
            )
        self.scheduler = scheduler
        self.policy = policy
        self.executor = executor
        self.run_id = run_id
        self.camera = camera or MetaDriveCameraAdapter()
        if fault_provider is not None and not isinstance(
            fault_provider, FaultInjectingModelProvider
        ):
            raise TypeError("fault_provider must be a FaultInjectingModelProvider")
        self.fault_provider = fault_provider
        self.arbiter = arbiter
        self._active_validation: CommandValidationDecision | None = None
        # The arbitrated assessment behind the active validation.
        self._active_assessment: VLAAssessment | None = None
        self._next_request_number = 1
        self._next_episode_number = 2
        self._episode_id = f"{run_id}-episode-1"
        self._request_cap_exhaustion_reported = False
        self.scheduler.begin_episode(self._episode_id)
        self._used_episode_ids = {self._episode_id}
        self.arbiter.begin_episode()

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def begin_episode(self, episode_id: str | None = None) -> str:
        """Clear command authority and quarantine work from the previous episode."""

        if episode_id is None:
            while True:
                episode_id = f"{self.run_id}-episode-{self._next_episode_number}"
                self._next_episode_number += 1
                if episode_id not in self._used_episode_ids:
                    break
        elif episode_id in self._used_episode_ids:
            raise ValueError("episode_id must not be reused")
        self.scheduler.begin_episode(episode_id)
        self._episode_id = episode_id
        self._used_episode_ids.add(episode_id)
        self._end_authority()
        self.arbiter.begin_episode()
        return episode_id

    def update(
        self,
        env: object,
        scene: LocalScene,
        *,
        now_s: float,
        ego_speed_mps: float,
        cruise_speed_mps: float,
        active_faults: tuple[FaultSpec, ...] = (),
    ) -> VLAControlUpdate:
        if not isinstance(scene, LocalScene):
            raise TypeError("scene must be a LocalScene")

        self._expire_active_validation(now_s)
        completion = self.scheduler.poll()
        validation: CommandValidationDecision | None = None
        arbitration: ArbitrationDecision | None = None
        if isinstance(completion, InferenceSuccess):
            # Arbitration may only take authority away, and the command the floor
            # validates is derived from what it leaves.
            proposal = completion.result.requested_command
            arbitration = arbitrate_assessment(
                self.arbiter,
                ArbitrationRequest(
                    assessment=completion.result.assessment,
                    scene=scene,
                    now_s=now_s,
                    issued_at_s=proposal.issued_at_s,
                    cruise_speed_mps=cruise_speed_mps,
                ),
            )
            validation = self.policy.validate(
                arbitration.assessment.to_command(
                    command_id=proposal.command_id,
                    issued_at_s=proposal.issued_at_s,
                    action_horizon_s=proposal.action_horizon_s,
                ),
                scene,
                now_s=now_s,
            )
            if validation.disposition is CommandDisposition.FALLBACK:
                self._end_authority()
            else:
                self._active_validation = validation
                self._active_assessment = arbitration.assessment
        elif isinstance(completion, (InferenceFailure, InferenceDiscarded)):
            # Transient failures cannot extend or replace command authority. A
            # still-valid prior command remains usable through its original
            # horizon unless this tick's review ends it.
            pass

        runtime_error: Exception | None = None
        submission: InferenceSubmitDisposition | None = None
        submitted_request_id: str | None = None
        submitted_episode_id: str | None = None
        submitted_generation_id: int | None = None
        candidate_request_id = f"{self.run_id}-vla-{self._next_request_number}"
        if self.scheduler.request_cap_exhausted:
            if self.scheduler.has_outstanding_inference:
                submission = InferenceSubmitDisposition.BUSY
            elif not self._request_cap_exhaustion_reported:
                submission = InferenceSubmitDisposition.REQUEST_CAP_EXHAUSTED
                self._request_cap_exhaustion_reported = True
        else:
            try:
                if has_fault(active_faults, FaultKind.CAMERA_DROPOUT):
                    raise CameraCaptureError("injected camera dropout")
                frame = self.camera.capture(env, timestamp_s=now_s)
                if self.fault_provider is not None:
                    self.fault_provider.register(candidate_request_id, active_faults)
                submission = self.scheduler.submit(
                    frame,
                    now_s=now_s,
                    request_id=candidate_request_id,
                    scene=scene,
                    ego_speed_mps=ego_speed_mps,
                    cruise_speed_mps=cruise_speed_mps,
                )
                if (
                    self.fault_provider is not None
                    and submission is not InferenceSubmitDisposition.STARTED
                ):
                    self.fault_provider.unregister(candidate_request_id)
                if submission is InferenceSubmitDisposition.STARTED:
                    submitted_request_id = candidate_request_id
                    submitted_episode_id = self.scheduler.active_episode_id
                    submitted_generation_id = self.scheduler.latest_generation_id
                    self._next_request_number += 1
                elif submission is InferenceSubmitDisposition.REQUEST_CAP_EXHAUSTED:
                    self._request_cap_exhaustion_reported = True
            except Exception as exc:
                # Camera/scheduling faults cannot extend or replace authority, but a
                # prior valid command remains bounded by its original horizon.
                runtime_error = exc
                if self.fault_provider is not None:
                    self.fault_provider.unregister(candidate_request_id)

        self._expire_active_validation(now_s)
        active = self._active_validation
        review = review_command(
            self.arbiter,
            ReviewRequest(
                active=self._active_assessment,
                issued_at_s=(
                    None if active is None else active.effective_command.issued_at_s
                ),
                scene=scene,
                now_s=now_s,
                cruise_speed_mps=cruise_speed_mps,
                failure=(
                    model_failure_category(completion.error)
                    if isinstance(completion, InferenceFailure)
                    else None
                ),
            ),
        )
        if review.revoked:
            self._end_authority()
        execution = self.executor.execute(self._active_validation, now_s=now_s)
        return VLAControlUpdate(
            execution=execution,
            validation=validation,
            active_validation=self._active_validation,
            completion=completion,
            submission_disposition=submission,
            submitted_request_id=submitted_request_id,
            submitted_episode_id=submitted_episode_id,
            submitted_generation_id=submitted_generation_id,
            active_fault_ids=tuple(fault.fault_id for fault in active_faults),
            runtime_error=runtime_error,
            arbitration=arbitration,
            review=review,
        )

    def _expire_active_validation(self, now_s: float) -> None:
        if self._active_validation is None:
            return
        command = self._active_validation.effective_command
        if exceeds_time_limit(now_s - command.issued_at_s, command.action_horizon_s):
            self._end_authority()

    def _end_authority(self) -> None:
        self._active_validation = None
        self._active_assessment = None
