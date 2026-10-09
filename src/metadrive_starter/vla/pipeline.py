from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass

from metadrive_starter.perception import LocalScene
from metadrive_starter.timing import (
    exceeds_time_limit,
    precedes_time_limit,
)
from metadrive_starter.vla.assessment import VLAAssessment
from metadrive_starter.vla.camera import RGBFrame
from metadrive_starter.vla.commands import VLACommand
from metadrive_starter.vla.observation import (
    OUTPUT_CONTRACT,
    ObservationBuilder,
    ObservationRequest,
    OutputContractWarning,
    checked_observation,
    describe_missing_contract,
    missing_contract_lines,
)
from metadrive_starter.vla.prompting import DefaultObservationBuilder
from metadrive_starter.vla.provider import (
    ModelFailureCategory,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
)
from metadrive_starter.vla.response import VLAResponseError, decode_vla_assessment


class VLAInferenceError(RuntimeError):
    """Raised when one model inference cannot produce a bounded command."""


def model_failure_category(error: Exception) -> ModelFailureCategory:
    """The stable label for why one inference produced no assessment."""
    if isinstance(error, ModelProviderError):
        return error.category
    if isinstance(error, (VLAResponseError, VLAInferenceError)):
        return ModelFailureCategory.INVALID_RESPONSE
    return ModelFailureCategory.UNKNOWN


@dataclass(frozen=True)
class VLAInferenceResult:
    request: ModelRequest
    response: ModelResponse
    assessment: VLAAssessment
    requested_command: VLACommand


class VLAInferencePipeline:
    """Synchronous camera-to-command pipeline independent of model transport."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        timeout_s: float = 10.0,
        action_horizon_s: float = 2.0,
        max_scene_objects: int = 8,
        maximum_frame_age_s: float = 0.5,
        maximum_clock_skew_s: float = 0.05,
        prompt_policy: str = "",
        observation_builder: ObservationBuilder | None = None,
    ) -> None:
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must implement ModelProvider.generate()")
        if not _positive_finite(timeout_s):
            raise ValueError("timeout_s must be finite and positive")
        if not _positive_finite(action_horizon_s):
            raise ValueError("action_horizon_s must be finite and positive")
        if not _positive_finite(maximum_frame_age_s):
            raise ValueError("maximum_frame_age_s must be finite and positive")
        if not _non_negative_finite(maximum_clock_skew_s):
            raise ValueError("maximum_clock_skew_s must be finite and non-negative")
        if (
            isinstance(max_scene_objects, bool)
            or not isinstance(max_scene_objects, int)
            or max_scene_objects < 0
        ):
            raise ValueError("max_scene_objects must be a non-negative integer")
        if observation_builder is None:
            observation_builder = DefaultObservationBuilder()
        elif not isinstance(observation_builder, ObservationBuilder):
            raise TypeError("observation_builder must implement ObservationBuilder.build()")
        self.provider = provider
        self.timeout_s = float(timeout_s)
        self.action_horizon_s = float(action_horizon_s)
        self.max_scene_objects = max_scene_objects
        self.maximum_frame_age_s = float(maximum_frame_age_s)
        self.maximum_clock_skew_s = float(maximum_clock_skew_s)
        self.prompt_policy = prompt_policy
        self.observation_builder = observation_builder
        # Every observation built, and those sent without every output-contract line.
        self.observations_built = 0
        self.observations_without_contract = 0

    def infer(
        self,
        frame: RGBFrame,
        *,
        now_s: float,
        request_id: str,
        episode_id: str = "standalone",
        generation_id: int = 1,
        scene: LocalScene | None = None,
        ego_speed_mps: float | None = None,
        cruise_speed_mps: float | None = None,
    ) -> VLAInferenceResult:
        if not isinstance(frame, RGBFrame):
            raise ValueError("frame must be an RGBFrame")
        observation_request = ObservationRequest(
            frame=frame,
            now_s=now_s,
            action_horizon_s=self.action_horizon_s,
            contract=OUTPUT_CONTRACT,
            scene=scene,
            ego_speed_mps=ego_speed_mps,
            cruise_speed_mps=cruise_speed_mps,
            max_scene_objects=self.max_scene_objects,
            prompt_policy=self.prompt_policy,
        )
        # The model receives exactly the observation built here, or nothing.
        observation = checked_observation(
            self.observation_builder.build(observation_request),
            observation_request,
        )
        self._count_contract(observation.prompt, observation_request)
        now_value = float(now_s)
        # Judged on the frame as captured, whatever image the observation carries.
        frame_age_s = now_value - frame.timestamp_s
        if precedes_time_limit(frame_age_s, -self.maximum_clock_skew_s):
            raise VLAInferenceError("camera frame timestamp is in the future")
        if exceeds_time_limit(frame_age_s, self.maximum_frame_age_s):
            raise VLAInferenceError(f"camera frame is stale ({frame_age_s:.3f}s old)")

        request = ModelRequest(
            request_id=request_id,
            prompt=observation.prompt,
            frame=observation.frame,
            created_at_s=now_value,
            episode_id=episode_id,
            generation_id=generation_id,
        )
        response = self.provider.generate(request, timeout_s=self.timeout_s)
        if not isinstance(response, ModelResponse):
            raise ModelProviderError("model provider returned an invalid response type")
        if response.request_id != request.request_id:
            raise VLAInferenceError("model response request_id does not match the request")

        assessment = decode_vla_assessment(response.text)
        command = assessment.to_command(
            command_id=request.request_id,
            issued_at_s=now_value,
            action_horizon_s=self.action_horizon_s,
        )
        return VLAInferenceResult(
            request=request,
            response=response,
            assessment=assessment,
            requested_command=command,
        )


    def _count_contract(self, prompt: str, request: ObservationRequest) -> None:
        """Count an observation, warning the first time one lacks contract lines."""
        self.observations_built += 1
        missing = missing_contract_lines(prompt, request.contract)
        if not missing:
            return
        self.observations_without_contract += 1
        if self.observations_without_contract == 1:
            # No registry: warnings.warn shows one message once per source line, which
            # would silence every run after the first in evaluate, race, and the gates.
            caller = sys._getframe(1)
            warnings.warn_explicit(
                OutputContractWarning(
                    f"an observation {describe_missing_contract(missing)}. It is sent "
                    "anyway; the run counts every such observation in "
                    "vla_metrics.observations_without_output_contract."
                ),
                OutputContractWarning,
                caller.f_code.co_filename,
                caller.f_lineno,
                module=caller.f_globals["__name__"],
                registry=None,
                module_globals=caller.f_globals,
            )


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0.0
    )


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )
