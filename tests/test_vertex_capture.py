from __future__ import annotations

import json
from pathlib import Path

import pytest

from metadrive_starter.cloud_preflight import PreflightCheck, VertexPreflightReport
from metadrive_starter.config import config_from_dict
from metadrive_starter.vla import (
    RGBFrame,
    RequestBudget,
    RequestBudgetError,
    ScriptedModelProvider,
)
from metadrive_starter.vla_probe import (
    ProbeCapture,
    ProbeScenario,
    run_vla_probe_catalog,
)
from metadrive_starter.vertex_capture import (
    VERTEX_CAPTURE_SCENARIO_IDS,
    run_vertex_capture,
)


def _capture(_config, _scenario) -> ProbeCapture:
    frame = RGBFrame(
        timestamp_s=1.0,
        width=2,
        height=1,
        rgb_bytes=b"\xff\x00\x00\x00\xff\x00",
    )
    return ProbeCapture(frame=frame, timestamp_s=1.0)


def _source_inputs(tmp_path: Path) -> tuple[object, Path]:
    config = config_from_dict(
        {"vla": {"prompt_policy": "Prefer conservative progress."}}
    )
    scenarios = tuple(
        ProbeScenario(
            scenario_id=scenario_id,
            description=f"Fixed {scenario_id} evidence input.",
            expected_visual="A deterministic test frame.",
            map_name="S",
        )
        for scenario_id in VERTEX_CAPTURE_SCENARIO_IDS
    )
    source = tmp_path / "source"
    run_vla_probe_catalog(config, scenarios, source, capture=_capture)
    return config, source


def _valid_response() -> str:
    return json.dumps(
        {
            "scene_summary": "Clear road.",
            "relevant_hazards": [],
            "meta_action": "KEEP_LANE",
            "target_speed_mps": 8.0,
            "confidence": 0.9,
            "brief_justification": "Continue safely.",
        }
    )


def test_vertex_capture_uses_exactly_five_attempts_under_one_budget(
    tmp_path: Path,
) -> None:
    config, source = _source_inputs(tmp_path)
    provider = ScriptedModelProvider([_valid_response()] * 3, model_id="vertex-test")

    def preflight(settings, **kwargs):
        assert kwargs["prompt_policy"] == "Prefer conservative progress."
        budget = kwargs["request_budget"]
        assert isinstance(budget, RequestBudget)
        budget.consume("vertex-preflight-text")
        budget.consume("vertex-preflight-image-1")
        return VertexPreflightReport(
            project_id="project-123",
            location=settings.location,
            model_id=settings.model_id,
            checks=(PreflightCheck("fake", True, "passed"),),
        )

    report = run_vertex_capture(
        config,
        source,
        tmp_path / "evidence",
        preflight_runner=preflight,
        provider_factory=lambda settings, project_id: provider,
    )

    assert report.successful is True
    assert report.planned_requests == 5
    assert report.used_requests == 5
    assert report.request_cap == 5
    assert report.request_labels[:2] == (
        "vertex-preflight-text",
        "vertex-preflight-image-1",
    )
    assert report.request_labels[2:] == tuple(
        f"probe-{scenario_id}-seed-0" for scenario_id in VERTEX_CAPTURE_SCENARIO_IDS
    )
    assert len(provider.requests) == 3
    saved = json.loads((tmp_path / "evidence" / "summary.json").read_text())
    assert saved["request_budget"] == {
        "hard_cap": 5,
        "planned": 5,
        "remaining": 0,
        "request_labels": list(report.request_labels),
        "retries": 0,
        "used": 5,
    }


def test_vertex_capture_rejects_insufficient_cap_before_any_provider_work(
    tmp_path: Path,
) -> None:
    called = False

    def preflight(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("preflight must not run")

    with pytest.raises(RequestBudgetError, match="needs 5"):
        run_vertex_capture(
            config_from_dict({}),
            tmp_path / "missing-source",
            tmp_path / "evidence",
            maximum_requests=4,
            preflight_runner=preflight,
        )

    assert called is False
    assert not (tmp_path / "evidence").exists()


def test_vertex_capture_validates_all_inputs_before_preflight(tmp_path: Path) -> None:
    config, source = _source_inputs(tmp_path)
    frame = source / VERTEX_CAPTURE_SCENARIO_IDS[-1] / "frame.png"
    frame.write_bytes(frame.read_bytes()[:-1] + b"x")
    called = False

    def preflight(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("preflight must not run")

    with pytest.raises(ValueError, match="PNG"):
        run_vertex_capture(
            config,
            source,
            tmp_path / "evidence",
            preflight_runner=preflight,
        )

    assert called is False
    assert not (tmp_path / "evidence").exists()


def test_vertex_capture_stops_after_failed_preflight(tmp_path: Path) -> None:
    config, source = _source_inputs(tmp_path)

    def preflight(settings, **kwargs):
        budget = kwargs["request_budget"]
        budget.consume("vertex-preflight-text")
        return VertexPreflightReport(
            project_id="project-123",
            location=settings.location,
            model_id=settings.model_id,
            checks=(PreflightCheck("text_round_trip", False, "failed"),),
        )

    report = run_vertex_capture(
        config,
        source,
        tmp_path / "evidence",
        preflight_runner=preflight,
        provider_factory=lambda *_: (_ for _ in ()).throw(
            AssertionError("probe provider must not be built")
        ),
    )

    assert report.successful is False
    assert report.used_requests == 1
    assert report.probes is None
    assert (tmp_path / "evidence" / "summary.json").is_file()
