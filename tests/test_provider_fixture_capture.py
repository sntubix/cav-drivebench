from __future__ import annotations

import json
from pathlib import Path

import pytest

from metadrive_starter.config import config_from_dict, load_config
from metadrive_starter.provider_fixture_capture import (
    convert_probe_capture_to_provider_fixtures,
)
from metadrive_starter.vla import (
    HighLevelAction,
    ModelRequest,
    ModelResponse,
    ModelResponseMetadata,
    ProviderFixtureError,
    RGBFrame,
    RecordedModelProvider,
    VLAResponseFormatError,
    decode_vla_assessment,
    load_provider_fixture,
    load_provider_fixtures,
)
from metadrive_starter.vla_probe import (
    ProbeCapture,
    ProbeEvaluationSpec,
    ProbeScenario,
    load_vla_probe_replay_inputs,
    replay_vla_probe_artifacts,
    run_vla_probe_catalog,
)
from metadrive_starter.vla_runtime import build_vla_provider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_FIXTURE_ROOT = (
    PROJECT_ROOT
    / "fixtures"
    / "vla"
    / "providers"
    / "vertex-gemini-3.5-flash-lite-20260824"
)
LIVE_INPUT_ROOT = (
    PROJECT_ROOT
    / "fixtures"
    / "vla"
    / "probes"
    / "vertex-gemini-3.5-flash-lite-20260824"
)
LIVE_SCENARIO_IDS = (
    "clear-straight",
    "stopped-vehicle",
    "red-light-with-traffic",
)


class CapturedProvider:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        del timeout_s
        return ModelResponse(
            request_id=request.request_id,
            text=self.text,
            model_id="captured-model",
            latency_s=1.25,
            metadata=ModelResponseMetadata(
                provider="vertex",
                response_id="provider-response-123",
                model_version="captured-model-v1",
                input_tokens=100,
                output_tokens=25,
                total_tokens=125,
            ),
        )


def _capture(_config, _scenario) -> ProbeCapture:
    return ProbeCapture(
        frame=RGBFrame(1.0, 2, 1, b"\xff\x00\x00\x00\xff\x00"),
        timestamp_s=1.0,
    )


def _response() -> str:
    return json.dumps(
        {
            "scene_summary": "A lead vehicle is visible.",
            "relevant_hazards": [
                {"type": "vehicle", "relative_location": "front", "risk": "medium"}
            ],
            "meta_action": "FOLLOW",
            "target_speed_mps": 9.72,
            "confidence": 0.95,
            "brief_justification": "Follow the visible vehicle.",
        }
    )


def _exact_capture(tmp_path: Path) -> tuple[object, Path]:
    config = config_from_dict({})
    scenario = ProbeScenario(
        scenario_id="stopped-vehicle",
        description="A stopped lead vehicle.",
        expected_visual="A vehicle ahead.",
        map_name="S",
        evaluation=ProbeEvaluationSpec(
            acceptable_actions=(HighLevelAction.FOLLOW,),
            minimum_target_speed_mps=0.0,
            maximum_target_speed_mps=5.0,
        ),
    )
    inputs = tmp_path / "inputs"
    run_vla_probe_catalog(config, [scenario], inputs, capture=_capture)
    capture = tmp_path / "capture"
    replay_vla_probe_artifacts(
        config,
        inputs,
        capture,
        provider=CapturedProvider(_response()),
    )
    return config, capture


def test_converter_creates_exact_sanitized_fixture_and_keeps_semantic_miss(
    tmp_path: Path,
) -> None:
    _, capture = _exact_capture(tmp_path)
    output = tmp_path / "fixtures"

    report = convert_probe_capture_to_provider_fixtures(
        capture,
        output,
        fixture_prefix="vertex-course-20260824",
    )

    assert report.successful is True
    assert report.response_ids_retained is False
    assert report.fixtures[0].semantic_passed is False
    fixture = load_provider_fixture(output / "01-stopped-vehicle.json")
    assert fixture.source == "sanitized_capture"
    assert fixture.prompt_sha256 is not None
    assert fixture.rgb_sha256 is not None
    assert fixture.response is not None
    assert fixture.response.text == _response()
    assert fixture.response.metadata.provider == "vertex"
    assert fixture.response.metadata.response_id is None
    assert fixture.response.metadata.total_tokens == 125

    replay_input = load_vla_probe_replay_inputs(capture)[0]
    replayed = RecordedModelProvider((fixture,)).generate(
        replay_input.request,
        timeout_s=2.0,
    )
    assert replayed.text == _response()
    assert replayed.metadata.fixture_id == fixture.fixture_id


def test_converter_can_retain_reviewed_response_id(tmp_path: Path) -> None:
    _, capture = _exact_capture(tmp_path)
    output = tmp_path / "fixtures"

    convert_probe_capture_to_provider_fixtures(
        capture,
        output,
        fixture_prefix="reviewed",
        retain_response_ids=True,
    )

    fixture = load_provider_fixture(output / "01-stopped-vehicle.json")
    assert fixture.response is not None
    assert fixture.response.metadata.response_id == "provider-response-123"


def test_converter_rejects_changed_raw_response(tmp_path: Path) -> None:
    _, capture = _exact_capture(tmp_path)
    raw = capture / "stopped-vehicle" / "raw_response.txt"
    raw.write_text("not JSON", encoding="utf-8")

    with pytest.raises(VLAResponseFormatError):
        convert_probe_capture_to_provider_fixtures(
            capture,
            tmp_path / "fixtures",
            fixture_prefix="changed",
        )

    assert not (tmp_path / "fixtures").exists()


def test_converter_requires_exact_replay_evidence(tmp_path: Path) -> None:
    config = config_from_dict({})
    scenario = ProbeScenario(
        scenario_id="clear-road",
        description="Clear road.",
        expected_visual="Clear road.",
        map_name="S",
    )
    direct = tmp_path / "direct"
    run_vla_probe_catalog(
        config,
        [scenario],
        direct,
        infer=True,
        provider=CapturedProvider(_response()),
        capture=_capture,
    )

    with pytest.raises(ProviderFixtureError, match="exact artifact-replay"):
        convert_probe_capture_to_provider_fixtures(
            direct,
            tmp_path / "fixtures",
            fixture_prefix="direct",
        )


def test_converter_rejects_unreviewed_provider_metadata_fields(tmp_path: Path) -> None:
    _, capture = _exact_capture(tmp_path)
    metadata_path = capture / "stopped-vehicle" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["outcome"]["provider_metadata"]["project_id"] = "must-not-leak"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ProviderFixtureError, match="unreviewed fields"):
        convert_probe_capture_to_provider_fixtures(
            capture,
            tmp_path / "fixtures",
            fixture_prefix="unreviewed",
        )


@pytest.mark.needs("fixtures")
def test_checked_vertex_fixtures_are_sanitized_and_keep_reviewed_outcomes() -> None:
    fixtures = tuple(
        load_provider_fixture(path) for path in sorted(LIVE_FIXTURE_ROOT.glob("*.json"))
    )

    assert [fixture.fixture_id for fixture in fixtures] == [
        "vertex-gemini-3.5-flash-lite-20260824-clear-straight",
        "vertex-gemini-3.5-flash-lite-20260824-stopped-vehicle",
        "vertex-gemini-3.5-flash-lite-20260824-red-light-with-traffic",
    ]
    assert [fixture.source for fixture in fixtures] == ["sanitized_capture"] * 3
    assert [fixture.response.metadata.response_id for fixture in fixtures if fixture.response] == [
        None,
        None,
        None,
    ]
    assert [
        fixture.response.metadata.total_tokens for fixture in fixtures if fixture.response
    ] == [2101, 2105, 2136]
    assert [
        decode_vla_assessment(fixture.response.text).proposed_action
        for fixture in fixtures
        if fixture.response
    ] == [
        HighLevelAction.KEEP_LANE,
        HighLevelAction.FOLLOW,
        HighLevelAction.STOP,
    ]


@pytest.mark.needs("fixtures")
def test_checked_vertex_bundle_replays_offline_end_to_end(tmp_path: Path) -> None:
    config = load_config(
        PROJECT_ROOT / "configs" / "vla-probe-vertex-fixtures-20260824.yaml"
    )

    summary = replay_vla_probe_artifacts(
        config,
        LIVE_INPUT_ROOT,
        tmp_path / "replay",
        provider=build_vla_provider(config.vla),
        scenario_ids=LIVE_SCENARIO_IDS,
    )

    assert summary.successful is True
    assert summary.semantically_successful is False
    assert [result.semantic_passed for result in summary.scenarios] == [
        True,
        False,
        True,
    ]
    for scenario_id, fixture in zip(
        LIVE_SCENARIO_IDS,
        load_provider_fixtures(LIVE_FIXTURE_ROOT),
    ):
        metadata = json.loads(
            (tmp_path / "replay" / scenario_id / "metadata.json").read_text()
        )
        assert (
            metadata["outcome"]["provider_metadata"]["fixture_id"]
            == fixture.fixture_id
        )
        assert (
            metadata["outcome"]["provider_metadata"]["fixture_sha256"]
            == fixture.artifact_sha256
        )
