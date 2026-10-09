from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Iterable, Protocol, runtime_checkable

from metadrive_starter.vla.camera import RGBFrame
from metadrive_starter.vla.contracts import VLA_PROMPT_CONTRACT_VERSION


class ModelFailureCategory(str, Enum):
    """Stable provider failure labels suitable for metrics and reports."""

    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    QUOTA = "quota"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_RESPONSE = "invalid_response"
    TRANSPORT = "transport"
    UNKNOWN = "unknown"


class ModelProviderError(RuntimeError):
    """Base error raised when a model provider cannot produce a response."""

    def __init__(
        self,
        message: str,
        *,
        category: ModelFailureCategory = ModelFailureCategory.TRANSPORT,
    ) -> None:
        super().__init__(message)
        self.category = category


class ModelTimeoutError(ModelProviderError, TimeoutError):
    """Raised when model inference does not finish within its deadline."""

    def __init__(self, message: str) -> None:
        super().__init__(message, category=ModelFailureCategory.TIMEOUT)


class RequestBudgetError(RuntimeError):
    """Raised before a provider call would exceed an explicit request budget."""


class RequestBudget:
    """Thread-safe hard cap shared by all provider calls in one workflow."""

    def __init__(self, maximum_requests: int) -> None:
        if (
            isinstance(maximum_requests, bool)
            or not isinstance(maximum_requests, int)
            or maximum_requests < 1
        ):
            raise ValueError("maximum_requests must be a positive integer")
        self._maximum_requests = maximum_requests
        self._labels: list[str] = []
        self._lock = Lock()

    @property
    def maximum_requests(self) -> int:
        return self._maximum_requests

    @property
    def used_requests(self) -> int:
        with self._lock:
            return len(self._labels)

    @property
    def remaining_requests(self) -> int:
        return self.maximum_requests - self.used_requests

    @property
    def request_labels(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._labels)

    def require_capacity(self, planned_requests: int) -> None:
        """Fail before work starts when the complete plan cannot fit."""
        if (
            isinstance(planned_requests, bool)
            or not isinstance(planned_requests, int)
            or planned_requests < 0
        ):
            raise ValueError("planned_requests must be a non-negative integer")
        with self._lock:
            available = self._maximum_requests - len(self._labels)
            if planned_requests > available:
                raise RequestBudgetError(
                    f"request plan needs {planned_requests} call(s), but only "
                    f"{available} remain under the hard cap of "
                    f"{self._maximum_requests}"
                )

    def consume(self, label: str) -> None:
        """Count one attempted outbound request before invoking its provider."""
        if not isinstance(label, str) or not label.strip():
            raise ValueError("request budget label must not be empty")
        with self._lock:
            if len(self._labels) >= self._maximum_requests:
                raise RequestBudgetError(
                    f"hard request cap of {self._maximum_requests} exhausted"
                )
            self._labels.append(label.strip())


@dataclass(frozen=True)
class ModelResponseMetadata:
    """Optional transport metadata; never grants command authority."""

    provider: str | None = None
    response_id: str | None = None
    model_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    fixture_id: str | None = None
    fixture_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("provider", "response_id", "model_version", "fixture_id"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or _contains_control_characters(value)
            ):
                raise ValueError(
                    f"{name} must be non-empty text without control characters or None"
                )
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.fixture_sha256 is not None and (
            not isinstance(self.fixture_sha256, str)
            or len(self.fixture_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.fixture_sha256
            )
        ):
            raise ValueError("fixture_sha256 must be lowercase SHA-256 or None")


@dataclass(frozen=True)
class ModelRequest:
    """Transport-neutral input for one camera-conditioned model inference."""

    request_id: str
    prompt: str
    frame: RGBFrame
    created_at_s: float
    encoded_png_bytes: bytes | None = None
    episode_id: str = "standalone"
    generation_id: int = 1
    prompt_contract_version: str = VLA_PROMPT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if _contains_control_characters(self.request_id):
            raise ValueError("request_id must not contain control characters")
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if _contains_control_characters(self.episode_id):
            raise ValueError("episode_id must not contain control characters")
        if (
            not isinstance(self.prompt_contract_version, str)
            or not self.prompt_contract_version.strip()
            or _contains_control_characters(self.prompt_contract_version)
        ):
            raise ValueError(
                "prompt_contract_version must be non-empty text without control characters"
            )
        if (
            isinstance(self.generation_id, bool)
            or not isinstance(self.generation_id, int)
            or self.generation_id < 1
        ):
            raise ValueError("generation_id must be a positive integer")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not isinstance(self.frame, RGBFrame):
            raise ValueError("frame must be an RGBFrame")
        if self.encoded_png_bytes is not None and (
            not isinstance(self.encoded_png_bytes, bytes) or not self.encoded_png_bytes
        ):
            raise ValueError("encoded_png_bytes must be non-empty bytes or None")
        if self.encoded_png_bytes is not None and not self.encoded_png_bytes.startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise ValueError("encoded_png_bytes must contain a PNG image")
        if (
            isinstance(self.created_at_s, bool)
            or not isinstance(self.created_at_s, (int, float))
            or not math.isfinite(self.created_at_s)
            or self.created_at_s < 0.0
        ):
            raise ValueError("created_at_s must be finite and non-negative")


@dataclass(frozen=True)
class ModelResponse:
    """Raw text returned by a model before command decoding and validation."""

    request_id: str
    text: str
    model_id: str
    latency_s: float
    metadata: ModelResponseMetadata = field(default_factory=ModelResponseMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if (
            isinstance(self.latency_s, bool)
            or not isinstance(self.latency_s, (int, float))
            or not math.isfinite(self.latency_s)
            or self.latency_s < 0.0
        ):
            raise ValueError("latency_s must be finite and non-negative")
        if not isinstance(self.metadata, ModelResponseMetadata):
            raise ValueError("metadata must be ModelResponseMetadata")


@runtime_checkable
class ModelProvider(Protocol):
    """Synchronous model boundary; callers decide how inference is scheduled."""

    def generate(
        self,
        request: ModelRequest,
        *,
        timeout_s: float,
    ) -> ModelResponse:
        """Return raw model output or raise a ``ModelProviderError``."""
        ...


class RequestCappedModelProvider:
    """Model provider decorator that counts attempts and never retries."""

    def __init__(self, provider: ModelProvider, budget: RequestBudget) -> None:
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must implement ModelProvider.generate()")
        if not isinstance(budget, RequestBudget):
            raise TypeError("budget must be a RequestBudget")
        self._provider = provider
        self._budget = budget

    @property
    def provider(self) -> ModelProvider:
        return self._provider

    @property
    def budget(self) -> RequestBudget:
        return self._budget

    def generate(
        self,
        request: ModelRequest,
        *,
        timeout_s: float,
    ) -> ModelResponse:
        self._budget.consume(request.request_id)
        return self._provider.generate(request, timeout_s=timeout_s)


ScriptedResult = str | Exception


class ScriptedModelProvider:
    """Deterministic provider for tests and demonstrations without a model."""

    def __init__(
        self,
        results: Iterable[ScriptedResult],
        *,
        model_id: str = "scripted-model",
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must not be empty")
        self._results = deque(results)
        self._model_id = model_id
        self._requests: list[ModelRequest] = []

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        """Requests observed so far, in call order."""
        return tuple(self._requests)

    def generate(
        self,
        request: ModelRequest,
        *,
        timeout_s: float,
    ) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise ValueError("request must be a ModelRequest")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
        ):
            raise ValueError("timeout_s must be finite and positive")

        self._requests.append(request)
        if not self._results:
            raise ModelProviderError("scripted model responses exhausted")

        result = self._results.popleft()
        if isinstance(result, Exception):
            raise result
        if not isinstance(result, str):
            raise ModelProviderError(
                f"unsupported scripted result type: {type(result).__name__}"
            )
        return ModelResponse(
            request_id=request.request_id,
            text=result,
            model_id=self._model_id,
            latency_s=0.0,
            metadata=ModelResponseMetadata(provider="scripted"),
        )


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)
