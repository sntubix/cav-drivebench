from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from metadrive_starter.perception import LocalScene
from metadrive_starter.safety import CommandDisposition, VLACommandValidator
from metadrive_starter.vla import (
    HighLevelAction,
    ModelFailureCategory,
    ModelResponseMetadata,
    ModelTimeoutError,
    ProviderFixture,
    ProviderFixtureError,
    ProviderFixtureFailure,
    ProviderFixtureReplayError,
    ProviderFixtureResponse,
    RGBFrame,
    RecordedModelProvider,
    VLAInferencePipeline,
    VLAResponseFormatError,
    VLA_PROMPT_CONTRACT_VERSION,
    load_provider_fixture,
    load_provider_fixtures,
    synthetic_fixture,
    write_provider_fixture,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "fixtures" / "vla" / "providers"


def _frame() -> RGBFrame:
    return RGBFrame(
        timestamp_s=10.0,
        width=1,
        height=1,
        rgb_bytes=b"\x01\x02\x03",
    )


def _scene() -> LocalScene:
    return LocalScene(
        timestamp_s=10.0,
        ego_speed_mps=4.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        objects=(),
        left_lane_available=True,
        right_lane_available=True,
        valid=True,
    )


@pytest.mark.needs("fixtures")
def test_success_fixture_replays_through_decoder_and_validator() -> None:
    provider = RecordedModelProvider.from_path(
        FIXTURE_ROOT / "synthetic-keep-lane.json"
    )
    result = VLAInferencePipeline(provider).infer(
        _frame(),
        now_s=10.0,
        request_id="fixture-request-1",
    )
    decision = VLACommandValidator().validate(
        result.requested_command,
        _scene(),
        now_s=10.0,
    )

    assert result.assessment.proposed_action is HighLevelAction.KEEP_LANE
    assert result.response.request_id == "fixture-request-1"
    assert result.response.metadata.provider == "synthetic_fixture"
    assert result.response.metadata.total_tokens == 176
    assert result.response.metadata.fixture_id == "synthetic-keep-lane"
    assert (
        result.response.metadata.fixture_sha256
        == "f1bcab570fbd88a919366cd602898e567d39b58294cf3a17ed302579ed5084f5"
    )
    assert decision.disposition is CommandDisposition.ACCEPTED
    assert provider.fixture_ids == ("synthetic-keep-lane",)
    assert provider.requests == (result.request,)


@pytest.mark.needs("fixtures")
def test_fixture_directory_loads_in_stable_filename_order() -> None:
    fixtures = load_provider_fixtures(FIXTURE_ROOT)

    assert tuple(item.fixture_id for item in fixtures) == (
        "synthetic-keep-lane",
        "synthetic-malformed",
        "synthetic-timeout",
    )
    assert all(item.artifact_sha256 is not None for item in fixtures)


@pytest.mark.needs("fixtures")
def test_timeout_fixture_preserves_typed_failure_category() -> None:
    provider = RecordedModelProvider.from_path(
        FIXTURE_ROOT / "synthetic-timeout.json"
    )

    with pytest.raises(ModelTimeoutError, match="synthetic provider") as raised:
        VLAInferencePipeline(provider).infer(
            _frame(),
            now_s=10.0,
            request_id="fixture-timeout",
        )

    assert raised.value.category is ModelFailureCategory.TIMEOUT


@pytest.mark.needs("fixtures")
def test_malformed_fixture_reaches_strict_assessment_decoder() -> None:
    provider = RecordedModelProvider.from_path(
        FIXTURE_ROOT / "synthetic-malformed.json"
    )

    with pytest.raises(VLAResponseFormatError):
        VLAInferencePipeline(provider).infer(
            _frame(),
            now_s=10.0,
            request_id="fixture-malformed",
        )


@pytest.mark.needs("fixtures")
def test_fixture_hash_detects_modified_outcome(tmp_path: Path) -> None:
    source = FIXTURE_ROOT / "synthetic-keep-lane.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    document["outcome"]["model_id"] = "tampered-model"
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ProviderFixtureError, match="does not match"):
        load_provider_fixture(target)


def test_fixture_loader_rejects_duplicate_fields(tmp_path: Path) -> None:
    target = tmp_path / "duplicate.json"
    target.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )

    with pytest.raises(ProviderFixtureError, match="invalid provider fixture JSON"):
        load_provider_fixture(target)


def test_writer_rejects_credential_like_text_before_persisting(
    tmp_path: Path,
) -> None:
    unsafe = synthetic_fixture(
        "unsafe",
        failure=ProviderFixtureFailure(
            ModelFailureCategory.AUTHENTICATION,
            "Authorization: Bearer secret-value",
        ),
    )

    with pytest.raises(ProviderFixtureError, match="credential-like"):
        write_provider_fixture(tmp_path / "unsafe.json", unsafe)
    assert not (tmp_path / "unsafe.json").exists()


def test_writer_hashes_round_trips_and_never_overwrites(tmp_path: Path) -> None:
    fixture = synthetic_fixture(
        "round-trip",
        response=ProviderFixtureResponse(
            text="not JSON by design",
            model_id="synthetic-model",
            latency_s=0.0,
            metadata=ModelResponseMetadata(provider="synthetic_fixture"),
        ),
    )
    target = tmp_path / "round-trip.json"

    checked = write_provider_fixture(target, fixture)

    assert checked.artifact_sha256 is not None
    assert load_provider_fixture(target) == checked
    with pytest.raises(ProviderFixtureError, match="already exists"):
        write_provider_fixture(target, fixture)


def test_sanitized_capture_requires_exact_request_hashes() -> None:
    with pytest.raises(ValueError, match="require prompt_sha256"):
        ProviderFixture(
            fixture_id="captured",
            source="sanitized_capture",
            prompt_contract_version=VLA_PROMPT_CONTRACT_VERSION,
            failure=ProviderFixtureFailure(
                ModelFailureCategory.TRANSPORT,
                "sanitized transport failure",
            ),
        )


def test_bound_fixture_rejects_different_request(tmp_path: Path) -> None:
    fixture = ProviderFixture(
        fixture_id="bound",
        source="synthetic",
        prompt_contract_version=VLA_PROMPT_CONTRACT_VERSION,
        response=ProviderFixtureResponse(
            text="not JSON",
            model_id="synthetic-model",
            latency_s=0.0,
        ),
        prompt_sha256=hashlib.sha256(b"different prompt").hexdigest(),
        rgb_sha256=hashlib.sha256(_frame().rgb_bytes).hexdigest(),
    )
    target = tmp_path / "bound.json"
    write_provider_fixture(target, fixture)
    provider = RecordedModelProvider.from_path(target)

    with pytest.raises(ProviderFixtureReplayError, match="prompt hash") as raised:
        VLAInferencePipeline(provider).infer(
            _frame(),
            now_s=10.0,
            request_id="bound-request",
        )

    assert raised.value.category is ModelFailureCategory.INVALID_RESPONSE


@pytest.mark.needs("fixtures")
def test_recorded_provider_exhaustion_fails_closed() -> None:
    provider = RecordedModelProvider.from_path(
        FIXTURE_ROOT / "synthetic-keep-lane.json"
    )
    pipeline = VLAInferencePipeline(provider)
    pipeline.infer(_frame(), now_s=10.0, request_id="first")

    with pytest.raises(ProviderFixtureReplayError, match="exhausted") as raised:
        pipeline.infer(_frame(), now_s=10.0, request_id="second")

    assert raised.value.category is ModelFailureCategory.MODEL_UNAVAILABLE
