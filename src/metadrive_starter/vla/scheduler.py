from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import Lock

from metadrive_starter.perception import LocalScene
from metadrive_starter.vla.camera import RGBFrame
from metadrive_starter.vla.pipeline import VLAInferencePipeline, VLAInferenceResult
from metadrive_starter.vla.provider import RequestBudget


class InferenceSubmitDisposition(str, Enum):
    """Immediate result of offering a frame to the inference scheduler."""

    STARTED = "started"
    BUSY = "busy"
    RATE_LIMITED = "rate_limited"
    REQUEST_CAP_EXHAUSTED = "request_cap_exhausted"
    CLOSED = "closed"


@dataclass(frozen=True)
class InferenceSuccess:
    """One completed inference that produced a decoded VLA command."""

    request_id: str
    episode_id: str
    generation_id: int
    submitted_at_s: float
    result: VLAInferenceResult


@dataclass(frozen=True)
class InferenceFailure:
    """One completed inference whose original exception is preserved."""

    request_id: str
    episode_id: str
    generation_id: int
    submitted_at_s: float
    error: Exception


@dataclass(frozen=True)
class InferenceDiscarded:
    """Completed inference quarantined because its episode/generation is obsolete."""

    request_id: str
    episode_id: str
    generation_id: int
    submitted_at_s: float
    reason: str
    result: VLAInferenceResult | None = None
    error: Exception | None = None


InferenceCompletion = InferenceSuccess | InferenceFailure | InferenceDiscarded


class VLAInferenceScheduler:
    """Run a synchronous inference pipeline without blocking the control loop.

    The scheduler owns one worker and permits exactly one outstanding inference.
    Frames submitted while that inference (or its uncollected completion) occupies
    the slot are rejected as ``BUSY`` rather than queued, so an old camera frame
    can never wait behind another request.

    Closing does not attempt to kill a running Python thread. Model providers must
    therefore honor the timeout passed to them by :class:`VLAInferencePipeline`.
    A completion remains available through :meth:`poll` after closing.
    """

    def __init__(
        self,
        pipeline: VLAInferencePipeline,
        *,
        minimum_interval_s: float,
        request_budget: RequestBudget | None = None,
    ) -> None:
        if not isinstance(pipeline, VLAInferencePipeline):
            raise TypeError("pipeline must be a VLAInferencePipeline")
        if not _non_negative_finite(minimum_interval_s):
            raise ValueError("minimum_interval_s must be finite and non-negative")
        if request_budget is not None and not isinstance(request_budget, RequestBudget):
            raise TypeError("request_budget must be a RequestBudget or None")

        self.pipeline = pipeline
        self.minimum_interval_s = float(minimum_interval_s)
        self.request_budget = request_budget
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vla-inference",
        )
        self._lock = Lock()
        self._future: Future[VLAInferenceResult] | None = None
        self._request_id: str | None = None
        self._request_episode_id: str | None = None
        self._request_generation_id: int | None = None
        self._submitted_at_s: float | None = None
        self._last_started_at_s: float | None = None
        self._active_episode_id = "default"
        self._used_episode_ids = {self._active_episode_id}
        self._next_generation_id = 1
        self._latest_generation_id = 0
        self._closed = False

    @property
    def active_episode_id(self) -> str:
        with self._lock:
            return self._active_episode_id

    @property
    def latest_generation_id(self) -> int:
        with self._lock:
            return self._latest_generation_id

    @property
    def request_cap_exhausted(self) -> bool:
        budget = self.request_budget
        return budget is not None and budget.remaining_requests == 0

    @property
    def has_outstanding_inference(self) -> bool:
        with self._lock:
            return self._future is not None

    def begin_episode(self, episode_id: str) -> None:
        """Activate a new episode and invalidate any older in-flight completion."""

        _validate_identifier(episode_id, "episode_id")
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot begin an episode on a closed scheduler")
            if episode_id in self._used_episode_ids:
                raise ValueError("episode_id must not be reused")
            self._active_episode_id = episode_id
            self._used_episode_ids.add(episode_id)
            self._next_generation_id = 1
            self._latest_generation_id = 0
            self._last_started_at_s = None

    def submit(
        self,
        frame: RGBFrame,
        *,
        now_s: float,
        request_id: str,
        scene: LocalScene | None = None,
        ego_speed_mps: float | None = None,
        cruise_speed_mps: float | None = None,
    ) -> InferenceSubmitDisposition:
        """Start immediately or explain why the frame was not accepted.

        ``now_s`` is simulation time. If it moves backwards after an episode
        reset, the new timestamp starts a fresh rate-limit epoch.
        """
        if not isinstance(frame, RGBFrame):
            raise ValueError("frame must be an RGBFrame")
        if not _non_negative_finite(now_s):
            raise ValueError("now_s must be finite and non-negative")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must not be empty")
        if _contains_control_characters(request_id):
            raise ValueError("request_id must not contain control characters")
        if scene is not None and not isinstance(scene, LocalScene):
            raise ValueError("scene must be a LocalScene or None")
        for name, value in {
            "ego_speed_mps": ego_speed_mps,
            "cruise_speed_mps": cruise_speed_mps,
        }.items():
            if value is not None and not _non_negative_finite(value):
                raise ValueError(f"{name} must be finite and non-negative or None")
        now_value = float(now_s)

        with self._lock:
            if self._closed:
                return InferenceSubmitDisposition.CLOSED
            if self._future is not None:
                return InferenceSubmitDisposition.BUSY
            if self.request_cap_exhausted:
                return InferenceSubmitDisposition.REQUEST_CAP_EXHAUSTED
            if (
                self._last_started_at_s is not None
                and now_value >= self._last_started_at_s
                and now_value - self._last_started_at_s < self.minimum_interval_s
            ):
                return InferenceSubmitDisposition.RATE_LIMITED

            self._future = self._executor.submit(
                self.pipeline.infer,
                frame,
                now_s=now_value,
                request_id=request_id,
                episode_id=self._active_episode_id,
                generation_id=self._next_generation_id,
                scene=scene,
                ego_speed_mps=ego_speed_mps,
                cruise_speed_mps=cruise_speed_mps,
            )
            self._request_id = request_id
            self._request_episode_id = self._active_episode_id
            self._request_generation_id = self._next_generation_id
            self._submitted_at_s = now_value
            self._last_started_at_s = now_value
            self._latest_generation_id = self._next_generation_id
            self._next_generation_id += 1
            return InferenceSubmitDisposition.STARTED

    def poll(self) -> InferenceCompletion | None:
        """Collect the current completion once, without raising inference errors."""
        with self._lock:
            future = self._future
            if future is None or not future.done():
                return None

            request_id = self._request_id
            episode_id = self._request_episode_id
            generation_id = self._request_generation_id
            submitted_at_s = self._submitted_at_s
            active_episode_id = self._active_episode_id
            latest_generation_id = self._latest_generation_id
            self._future = None
            self._request_id = None
            self._request_episode_id = None
            self._request_generation_id = None
            self._submitted_at_s = None

        # These values are set atomically with every accepted future.
        assert request_id is not None
        assert episode_id is not None
        assert generation_id is not None
        assert submitted_at_s is not None
        try:
            result = future.result()
        except Exception as exc:
            if episode_id != active_episode_id or generation_id != latest_generation_id:
                return InferenceDiscarded(
                    request_id=request_id,
                    episode_id=episode_id,
                    generation_id=generation_id,
                    submitted_at_s=submitted_at_s,
                    reason="completion belongs to an obsolete episode or generation",
                    error=exc,
                )
            return InferenceFailure(
                request_id=request_id,
                episode_id=episode_id,
                generation_id=generation_id,
                submitted_at_s=submitted_at_s,
                error=exc,
            )
        if episode_id != active_episode_id or generation_id != latest_generation_id:
            return InferenceDiscarded(
                request_id=request_id,
                episode_id=episode_id,
                generation_id=generation_id,
                submitted_at_s=submitted_at_s,
                reason="completion belongs to an obsolete episode or generation",
                result=result,
            )
        return InferenceSuccess(
            request_id=request_id,
            episode_id=episode_id,
            generation_id=generation_id,
            submitted_at_s=submitted_at_s,
            result=result,
        )

    def close(self, *, wait: bool = True) -> None:
        """Reject future submissions and shut down the owned worker once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def __enter__(self) -> VLAInferenceScheduler:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _validate_identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if _contains_control_characters(value):
        raise ValueError(f"{name} must not contain control characters")
