from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Iterable

from metadrive_starter.vla.contracts import VLA_PROMPT_CONTRACT_VERSION
from metadrive_starter.vla.provider import (
    ModelFailureCategory,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelResponseMetadata,
    ModelTimeoutError,
)


PROVIDER_FIXTURE_SCHEMA_VERSION = 1
PROVIDER_FIXTURE_MAX_BYTES = 2 * 1024 * 1024
_FIXTURE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SENSITIVE_TEXT = re.compile(
    r"(?:"
    r"authorization\s*[:=]|"
    r"bearer\s+[A-Za-z0-9._~+/-]+|"
    r"api[_-]?key\s*[:=]|"
    r"x-goog-api-key|"
    r"access[_-]?token\s*[:=]|"
    r"refresh[_-]?token\s*[:=]|"
    r"client[_-]?secret\s*[:=]|"
    r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}|"
    r"\bAIza[A-Za-z0-9_-]{20,}|"
    r"\bya29\.[A-Za-z0-9_-]{20,}|"
    r"[?&](?:key|token|access_token)=[^\s&#]+|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r")",
    re.IGNORECASE,
)


class ProviderFixtureError(ValueError):
    """Raised when an offline provider fixture is invalid or unsafe."""


class ProviderFixtureReplayError(ModelProviderError):
    """Raised when a fixture cannot replay the request it received."""


@dataclass(frozen=True)
class ProviderFixtureResponse:
    text: str
    model_id: str
    latency_s: float
    metadata: ModelResponseMetadata = field(default_factory=ModelResponseMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("provider fixture response text must be a string")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("provider fixture model_id must not be empty")
        if (
            isinstance(self.latency_s, bool)
            or not isinstance(self.latency_s, (int, float))
            or not math.isfinite(self.latency_s)
            or self.latency_s < 0.0
        ):
            raise ValueError(
                "provider fixture latency_s must be finite and non-negative"
            )
        if not isinstance(self.metadata, ModelResponseMetadata):
            raise ValueError("provider fixture metadata must be ModelResponseMetadata")


@dataclass(frozen=True)
class ProviderFixtureFailure:
    category: ModelFailureCategory
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.category, ModelFailureCategory):
            raise ValueError("provider fixture failure category is invalid")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("provider fixture failure message must not be empty")


@dataclass(frozen=True)
class ProviderFixture:
    fixture_id: str
    source: str
    prompt_contract_version: str
    response: ProviderFixtureResponse | None = None
    failure: ProviderFixtureFailure | None = None
    prompt_sha256: str | None = None
    rgb_sha256: str | None = None
    artifact_sha256: str | None = None
    schema_version: int = PROVIDER_FIXTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != PROVIDER_FIXTURE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"provider fixture schema_version must be "
                f"{PROVIDER_FIXTURE_SCHEMA_VERSION}"
            )
        if not isinstance(self.fixture_id, str) or _FIXTURE_ID.fullmatch(
            self.fixture_id
        ) is None:
            raise ValueError(
                "provider fixture_id must contain only letters, numbers, '.', "
                "'_', or '-'"
            )
        if self.source not in {"synthetic", "sanitized_capture"}:
            raise ValueError(
                "provider fixture source must be 'synthetic' or 'sanitized_capture'"
            )
        if (
            not isinstance(self.prompt_contract_version, str)
            or not self.prompt_contract_version.strip()
            or _contains_control_characters(self.prompt_contract_version)
        ):
            raise ValueError("provider fixture prompt_contract_version is invalid")
        if (self.response is None) == (self.failure is None):
            raise ValueError(
                "provider fixture must contain exactly one response or failure"
            )
        if self.response is not None and not isinstance(
            self.response, ProviderFixtureResponse
        ):
            raise ValueError("provider fixture response is invalid")
        if self.response is not None and (
            self.response.metadata.fixture_id is not None
            or self.response.metadata.fixture_sha256 is not None
        ):
            raise ValueError(
                "fixture provenance is added during replay, not stored as provider data"
            )
        if self.failure is not None and not isinstance(
            self.failure, ProviderFixtureFailure
        ):
            raise ValueError("provider fixture failure is invalid")
        _validate_optional_sha256(self.prompt_sha256, "prompt_sha256")
        _validate_optional_sha256(self.rgb_sha256, "rgb_sha256")
        _validate_optional_sha256(self.artifact_sha256, "artifact_sha256")
        if self.source == "sanitized_capture" and (
            self.prompt_sha256 is None or self.rgb_sha256 is None
        ):
            raise ValueError(
                "sanitized captured fixtures require prompt_sha256 and rgb_sha256"
            )


class RecordedModelProvider:
    """Replay sanitized, integrity-checked outcomes without network I/O."""

    def __init__(
        self,
        fixtures: Iterable[ProviderFixture],
        *,
        repeat_last: bool = False,
    ) -> None:
        loaded = tuple(fixtures)
        if not loaded or any(not isinstance(item, ProviderFixture) for item in loaded):
            raise ValueError("fixtures must contain at least one ProviderFixture")
        if any(item.artifact_sha256 is None for item in loaded):
            raise ValueError("recorded provider requires integrity-checked fixtures")
        if not isinstance(repeat_last, bool):
            raise ValueError("repeat_last must be a boolean")
        for fixture in loaded:
            assert_provider_fixture_sanitized(fixture)
            if provider_fixture_sha256(fixture) != fixture.artifact_sha256:
                raise ValueError("recorded provider fixture integrity check failed")
        self._fixtures = loaded
        self._repeat_last = repeat_last
        self._requests: list[ModelRequest] = []
        self._next_index = 0
        self._lock = Lock()

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        repeat_last: bool = False,
    ) -> RecordedModelProvider:
        return cls(load_provider_fixtures(path), repeat_last=repeat_last)

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        with self._lock:
            return tuple(self._requests)

    @property
    def fixture_ids(self) -> tuple[str, ...]:
        return tuple(fixture.fixture_id for fixture in self._fixtures)

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

        with self._lock:
            self._requests.append(request)
            if self._next_index < len(self._fixtures):
                fixture = self._fixtures[self._next_index]
                self._next_index += 1
            elif self._repeat_last:
                fixture = self._fixtures[-1]
            else:
                raise ProviderFixtureReplayError(
                    "recorded provider fixtures exhausted",
                    category=ModelFailureCategory.MODEL_UNAVAILABLE,
                )

        _match_fixture_request(fixture, request)
        if fixture.failure is not None:
            if fixture.failure.category is ModelFailureCategory.TIMEOUT:
                raise ModelTimeoutError(fixture.failure.message)
            raise ModelProviderError(
                fixture.failure.message,
                category=fixture.failure.category,
            )
        assert fixture.response is not None
        return ModelResponse(
            request_id=request.request_id,
            text=fixture.response.text,
            model_id=fixture.response.model_id,
            latency_s=float(fixture.response.latency_s),
            metadata=replace(
                fixture.response.metadata,
                fixture_id=fixture.fixture_id,
                fixture_sha256=fixture.artifact_sha256,
            ),
        )


def load_provider_fixture(path: Path | str) -> ProviderFixture:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ProviderFixtureError(f"cannot read provider fixture: {source}") from exc
    if len(payload) > PROVIDER_FIXTURE_MAX_BYTES:
        raise ProviderFixtureError(
            f"provider fixture exceeds {PROVIDER_FIXTURE_MAX_BYTES} bytes"
        )
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ProviderFixtureError) as exc:
        raise ProviderFixtureError(f"invalid provider fixture JSON: {source}") from exc
    if not isinstance(document, dict):
        raise ProviderFixtureError("provider fixture document must be an object")
    _require_keys(
        document,
        {
            "schema_version",
            "fixture_id",
            "source",
            "prompt_contract_version",
            "request_match",
            "outcome",
            "artifact_sha256",
        },
        "provider fixture",
    )
    expected_sha256 = _required_string(document, "artifact_sha256")
    _validate_optional_sha256(expected_sha256, "artifact_sha256")
    core = dict(document)
    del core["artifact_sha256"]
    actual_sha256 = _document_sha256(core)
    if actual_sha256 != expected_sha256:
        raise ProviderFixtureError("provider fixture artifact_sha256 does not match")

    try:
        fixture = _parse_fixture(core, artifact_sha256=actual_sha256)
    except ProviderFixtureError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProviderFixtureError(f"invalid provider fixture: {exc}") from exc
    if provider_fixture_sha256(fixture) != actual_sha256:
        raise ProviderFixtureError("provider fixture is not in canonical form")
    assert_provider_fixture_sanitized(fixture)
    return fixture


def load_provider_fixtures(path: Path | str) -> tuple[ProviderFixture, ...]:
    source = Path(path)
    if source.is_file():
        return (load_provider_fixture(source),)
    if not source.is_dir():
        raise ProviderFixtureError(f"provider fixture path does not exist: {source}")
    fixture_paths = tuple(sorted(source.glob("*.json")))
    if not fixture_paths:
        raise ProviderFixtureError(f"provider fixture directory is empty: {source}")
    return tuple(load_provider_fixture(item) for item in fixture_paths)


def write_provider_fixture(
    path: Path | str,
    fixture: ProviderFixture,
) -> ProviderFixture:
    """Write one new sanitized fixture without replacing existing evidence."""

    if not isinstance(fixture, ProviderFixture):
        raise TypeError("fixture must be a ProviderFixture")
    assert_provider_fixture_sanitized(fixture)
    actual_sha256 = provider_fixture_sha256(fixture)
    if (
        fixture.artifact_sha256 is not None
        and fixture.artifact_sha256 != actual_sha256
    ):
        raise ProviderFixtureError("provider fixture artifact_sha256 is stale")
    checked = replace(fixture, artifact_sha256=actual_sha256)
    document = _fixture_core(checked)
    document["artifact_sha256"] = actual_sha256
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(
                document,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
    except FileExistsError as exc:
        raise ProviderFixtureError(
            f"provider fixture already exists: {destination}"
        ) from exc
    return checked


def provider_fixture_sha256(fixture: ProviderFixture) -> str:
    if not isinstance(fixture, ProviderFixture):
        raise TypeError("fixture must be a ProviderFixture")
    return _document_sha256(_fixture_core(fixture))


def assert_provider_fixture_sanitized(fixture: ProviderFixture) -> None:
    if not isinstance(fixture, ProviderFixture):
        raise TypeError("fixture must be a ProviderFixture")
    for path, value in _fixture_strings(_fixture_core(fixture)):
        if _SENSITIVE_TEXT.search(value):
            raise ProviderFixtureError(
                f"provider fixture contains credential-like text at {path}"
            )


def _parse_fixture(
    core: dict[str, object],
    *,
    artifact_sha256: str,
) -> ProviderFixture:
    request_match = core["request_match"]
    if not isinstance(request_match, dict):
        raise ProviderFixtureError("provider fixture request_match must be an object")
    _allow_keys(request_match, {"prompt_sha256", "rgb_sha256"}, "request_match")
    outcome = core["outcome"]
    if not isinstance(outcome, dict):
        raise ProviderFixtureError("provider fixture outcome must be an object")
    outcome_type = _required_string(outcome, "type")
    response: ProviderFixtureResponse | None = None
    failure: ProviderFixtureFailure | None = None
    if outcome_type == "response":
        _require_keys(
            outcome,
            {"type", "text", "model_id", "latency_s", "metadata"},
            "response outcome",
        )
        metadata = outcome["metadata"]
        if not isinstance(metadata, dict):
            raise ProviderFixtureError("provider fixture metadata must be an object")
        _require_keys(
            metadata,
            {
                "provider",
                "response_id",
                "model_version",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            },
            "provider metadata",
        )
        response = ProviderFixtureResponse(
            text=_required_string_allow_empty(outcome, "text"),
            model_id=_required_string(outcome, "model_id"),
            latency_s=outcome["latency_s"],  # type: ignore[arg-type]
            metadata=ModelResponseMetadata(**metadata),  # type: ignore[arg-type]
        )
    elif outcome_type == "failure":
        _require_keys(outcome, {"type", "category", "message"}, "failure outcome")
        try:
            category = ModelFailureCategory(_required_string(outcome, "category"))
        except ValueError as exc:
            raise ProviderFixtureError(
                "provider fixture failure category is invalid"
            ) from exc
        failure = ProviderFixtureFailure(
            category=category,
            message=_required_string(outcome, "message"),
        )
    else:
        raise ProviderFixtureError(
            "provider fixture outcome type must be 'response' or 'failure'"
        )
    try:
        return ProviderFixture(
            fixture_id=core["fixture_id"],  # type: ignore[arg-type]
            source=core["source"],  # type: ignore[arg-type]
            prompt_contract_version=core[
                "prompt_contract_version"
            ],  # type: ignore[arg-type]
            response=response,
            failure=failure,
            prompt_sha256=request_match.get("prompt_sha256"),  # type: ignore[arg-type]
            rgb_sha256=request_match.get("rgb_sha256"),  # type: ignore[arg-type]
            artifact_sha256=artifact_sha256,
            schema_version=core["schema_version"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ProviderFixtureError(f"invalid provider fixture: {exc}") from exc


def _fixture_core(fixture: ProviderFixture) -> dict[str, object]:
    if fixture.response is not None:
        outcome: dict[str, object] = {
            "type": "response",
            "text": fixture.response.text,
            "model_id": fixture.response.model_id,
            "latency_s": fixture.response.latency_s,
            "metadata": {
                "provider": fixture.response.metadata.provider,
                "response_id": fixture.response.metadata.response_id,
                "model_version": fixture.response.metadata.model_version,
                "input_tokens": fixture.response.metadata.input_tokens,
                "output_tokens": fixture.response.metadata.output_tokens,
                "total_tokens": fixture.response.metadata.total_tokens,
            },
        }
    else:
        assert fixture.failure is not None
        outcome = {
            "type": "failure",
            "category": fixture.failure.category.value,
            "message": fixture.failure.message,
        }
    return {
        "schema_version": fixture.schema_version,
        "fixture_id": fixture.fixture_id,
        "source": fixture.source,
        "prompt_contract_version": fixture.prompt_contract_version,
        "request_match": {
            "prompt_sha256": fixture.prompt_sha256,
            "rgb_sha256": fixture.rgb_sha256,
        },
        "outcome": outcome,
    }


def _match_fixture_request(fixture: ProviderFixture, request: ModelRequest) -> None:
    if request.prompt_contract_version != fixture.prompt_contract_version:
        raise ProviderFixtureReplayError(
            f"fixture {fixture.fixture_id!r} prompt contract does not match request",
            category=ModelFailureCategory.INVALID_RESPONSE,
        )
    if fixture.prompt_sha256 is not None:
        prompt_sha256 = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        if prompt_sha256 != fixture.prompt_sha256:
            raise ProviderFixtureReplayError(
                f"fixture {fixture.fixture_id!r} prompt hash does not match request",
                category=ModelFailureCategory.INVALID_RESPONSE,
            )
    if fixture.rgb_sha256 is not None:
        rgb_sha256 = hashlib.sha256(request.frame.rgb_bytes).hexdigest()
        if rgb_sha256 != fixture.rgb_sha256:
            raise ProviderFixtureReplayError(
                f"fixture {fixture.fixture_id!r} RGB hash does not match request",
                category=ModelFailureCategory.INVALID_RESPONSE,
            )


def _document_sha256(document: dict[str, object]) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ProviderFixtureError(f"duplicate provider fixture field: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ProviderFixtureError(f"provider fixture contains non-standard number {value}")


def _require_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ProviderFixtureError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ProviderFixtureError(f"{label} has unknown fields: {', '.join(unknown)}")


def _allow_keys(value: dict[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ProviderFixtureError(f"{label} has unknown fields: {', '.join(unknown)}")


def _required_string(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ProviderFixtureError(f"provider fixture {key} must be non-empty text")
    return item


def _required_string_allow_empty(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ProviderFixtureError(f"provider fixture {key} must be text")
    return item


def _validate_optional_sha256(value: object, label: str) -> None:
    if value is not None and (
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
    ):
        raise ProviderFixtureError(
            f"provider fixture {label} must be lowercase SHA-256 or null"
        )


def _fixture_strings(
    value: object,
    path: str = "$",
) -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _fixture_strings(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _fixture_strings(item, f"{path}[{index}]")


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def synthetic_fixture(
    fixture_id: str,
    *,
    response: ProviderFixtureResponse | None = None,
    failure: ProviderFixtureFailure | None = None,
    prompt_contract_version: str = VLA_PROMPT_CONTRACT_VERSION,
) -> ProviderFixture:
    """Construct an unhashed synthetic fixture for ``write_provider_fixture``."""

    return ProviderFixture(
        fixture_id=fixture_id,
        source="synthetic",
        prompt_contract_version=prompt_contract_version,
        response=response,
        failure=failure,
    )
