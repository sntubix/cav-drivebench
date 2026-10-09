from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from metadrive_starter.vla import (
    ModelResponseMetadata,
    ProviderFixture,
    ProviderFixtureError,
    ProviderFixtureResponse,
    assert_provider_fixture_sanitized,
    decode_vla_assessment,
    write_provider_fixture,
)
from metadrive_starter.vla_probe import (
    ProbeReplayInput,
    load_vla_probe_replay_inputs,
)


_FIXTURE_PREFIX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class ProviderFixtureConversion:
    scenario_id: str
    fixture_id: str
    path: str
    artifact_sha256: str
    semantic_passed: bool | None


@dataclass(frozen=True)
class ProviderFixtureConversionReport:
    source_dir: str
    output_dir: str
    response_ids_retained: bool
    fixtures: tuple[ProviderFixtureConversion, ...]

    @property
    def successful(self) -> bool:
        return bool(self.fixtures)

    def to_dict(self) -> dict[str, object]:
        return {
            "successful": self.successful,
            "source_dir": self.source_dir,
            "output_dir": self.output_dir,
            "response_ids_retained": self.response_ids_retained,
            "fixtures": [asdict(item) for item in self.fixtures],
        }


def convert_probe_capture_to_provider_fixtures(
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    fixture_prefix: str,
    scenario_ids: Sequence[str] | None = None,
    retain_response_ids: bool = False,
) -> ProviderFixtureConversionReport:
    """Convert reviewed successful probe outcomes into exact offline fixtures."""
    if (
        not isinstance(fixture_prefix, str)
        or _FIXTURE_PREFIX.fullmatch(fixture_prefix) is None
    ):
        raise ValueError(
            "fixture_prefix must contain only letters, numbers, '.', '_', or '-'"
        )
    if not isinstance(retain_response_ids, bool):
        raise ValueError("retain_response_ids must be a boolean")

    prepared = load_vla_probe_replay_inputs(
        source_dir,
        scenario_ids=scenario_ids,
        require_prompt_hash=True,
    )
    fixtures = tuple(
        _fixture_from_probe_input(
            item,
            fixture_id=f"{fixture_prefix}-{item.scenario.scenario_id}",
            retain_response_id=retain_response_ids,
        )
        for item in prepared
    )

    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(
            f"provider fixture output directory already exists: {root}"
        )
    root.mkdir(parents=True)

    conversions: list[ProviderFixtureConversion] = []
    for index, (replay_input, fixture) in enumerate(
        zip(prepared, fixtures),
        start=1,
    ):
        target = root / f"{index:02d}-{replay_input.scenario.scenario_id}.json"
        checked = write_provider_fixture(target, fixture)
        assert checked.artifact_sha256 is not None
        conversions.append(
            ProviderFixtureConversion(
                scenario_id=replay_input.scenario.scenario_id,
                fixture_id=checked.fixture_id,
                path=str(target),
                artifact_sha256=checked.artifact_sha256,
                semantic_passed=_semantic_passed(replay_input.source_metadata),
            )
        )

    return ProviderFixtureConversionReport(
        source_dir=str(Path(source_dir).resolve()),
        output_dir=str(root),
        response_ids_retained=retain_response_ids,
        fixtures=tuple(conversions),
    )


def _fixture_from_probe_input(
    replay_input: ProbeReplayInput,
    *,
    fixture_id: str,
    retain_response_id: bool,
) -> ProviderFixture:
    metadata = replay_input.source_metadata
    if metadata.get("mode") != "artifact_replay":
        raise ProviderFixtureError(
            "sanitized fixtures require exact artifact-replay evidence"
        )
    replay = _object(metadata.get("replay"), "probe replay")
    if replay.get("exact_prompt") is not True or replay.get("exact_png_bytes") is not True:
        raise ProviderFixtureError(
            "probe evidence must confirm exact prompt and PNG replay"
        )
    outcome = _object(metadata.get("outcome"), "probe outcome")
    if outcome.get("status") != "success":
        raise ProviderFixtureError(
            "only technically successful probe outcomes can become response fixtures"
        )
    if outcome.get("prompt_contract_version") != replay_input.request.prompt_contract_version:
        raise ProviderFixtureError("probe prompt contract versions do not match")

    raw_filename = _filename(outcome.get("raw_response_file"), "raw response file")
    raw_text = (replay_input.source_dir / raw_filename).read_text(encoding="utf-8")
    decode_vla_assessment(raw_text)

    provider_metadata = _object(
        outcome.get("provider_metadata"),
        "probe provider metadata",
    )
    unknown_provider_fields = sorted(
        set(provider_metadata)
        - {
            "provider",
            "response_id",
            "model_version",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "fixture_id",
            "fixture_sha256",
        }
    )
    if unknown_provider_fields:
        raise ProviderFixtureError(
            "probe provider metadata has unreviewed fields: "
            + ", ".join(unknown_provider_fields)
        )
    if provider_metadata.get("fixture_id") is not None or provider_metadata.get(
        "fixture_sha256"
    ) is not None:
        raise ProviderFixtureError(
            "captured provider metadata must not contain fixture provenance"
        )
    response_id = _optional_text(provider_metadata.get("response_id"), "response_id")
    typed_metadata = ModelResponseMetadata(
        provider=_optional_text(provider_metadata.get("provider"), "provider"),
        response_id=response_id if retain_response_id else None,
        model_version=_optional_text(
            provider_metadata.get("model_version"),
            "model_version",
        ),
        input_tokens=_optional_count(
            provider_metadata.get("input_tokens"),
            "input_tokens",
        ),
        output_tokens=_optional_count(
            provider_metadata.get("output_tokens"),
            "output_tokens",
        ),
        total_tokens=_optional_count(
            provider_metadata.get("total_tokens"),
            "total_tokens",
        ),
    )
    model_id = _text(outcome.get("model_id"), "model_id")
    latency_s = outcome.get("latency_s")
    fixture = ProviderFixture(
        fixture_id=fixture_id,
        source="sanitized_capture",
        prompt_contract_version=replay_input.request.prompt_contract_version,
        prompt_sha256=_text(
            _object(metadata.get("capture"), "probe capture").get(
                "prompt_sha256"
            ),
            "prompt_sha256",
        ),
        rgb_sha256=_text(
            _object(metadata.get("capture"), "probe capture").get("rgb_sha256"),
            "rgb_sha256",
        ),
        response=ProviderFixtureResponse(
            text=raw_text,
            model_id=model_id,
            latency_s=latency_s,  # type: ignore[arg-type]
            metadata=typed_metadata,
        ),
    )
    assert_provider_fixture_sanitized(fixture)
    return fixture


def _semantic_passed(metadata: dict[str, object]) -> bool | None:
    outcome = _object(metadata.get("outcome"), "probe outcome")
    evaluation = outcome.get("semantic_evaluation")
    if evaluation is None:
        return None
    value = _object(evaluation, "semantic evaluation").get("passed")
    if not isinstance(value, bool):
        raise ProviderFixtureError("semantic evaluation passed must be a boolean")
    return value


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProviderFixtureError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderFixtureError(f"{label} must be non-empty text")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _optional_count(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderFixtureError(f"{label} must be a non-negative integer or null")
    return value


def _filename(value: object, label: str) -> str:
    text = _text(value, label)
    if Path(text).name != text:
        raise ProviderFixtureError(f"{label} must be a filename")
    return text


__all__ = [
    "ProviderFixtureConversion",
    "ProviderFixtureConversionReport",
    "convert_probe_capture_to_provider_fixtures",
]
