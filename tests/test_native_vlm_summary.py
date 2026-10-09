from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.native_vlm_summary import AcceptanceError, build_acceptance_summary


def _write_evidence(tmp_path: Path, run: dict[str, object]) -> tuple[Path, Path, Path]:
    run_log = tmp_path / "run.log"
    time_file = tmp_path / "time.txt"
    resources = tmp_path / "server-resources.csv"
    run_log.write_text("engine preamble\n" + json.dumps(run) + "\n", encoding="utf-8")
    time_file.write_text("wall_seconds=8.0\nmax_rss_kib=4096\n", encoding="utf-8")
    resources.write_text(
        "timestamp_utc,pid,rss_kib,vsz_kib,cpu_percent,elapsed\n"
        "2026-09-09T00:00:00Z,10,1024,2048,50.0,00:01\n"
        "2026-09-09T00:00:01Z,10,3072,4096,80.0,00:02\n",
        encoding="utf-8",
    )
    return run_log, time_file, resources


def test_native_vlm_summary_accepts_real_model_authority(tmp_path: Path) -> None:
    paths = _write_evidence(
        tmp_path,
        {
            "simulation_time_s": 2.0,
            "crashed": False,
            "went_off_road": False,
            "vla_metrics": {
                "provider_requests_attempted": 3,
                "responses_succeeded": 2,
                "authority_steps": 7,
                "runtime_errors": 0,
            },
        },
    )

    summary = build_acceptance_summary(*paths)

    assert summary["successful"] is True
    assert summary["achieved_realtime_factor"] == 0.25
    assert summary["simulator_max_rss_kib"] == 4096
    assert summary["server_max_rss_kib"] == 3072


def test_native_vlm_summary_rejects_fallback_only_run(tmp_path: Path) -> None:
    paths = _write_evidence(
        tmp_path,
        {
            "simulation_time_s": 2.0,
            "crashed": False,
            "went_off_road": False,
            "vla_metrics": {
                "provider_requests_attempted": 3,
                "responses_succeeded": 0,
                "authority_steps": 0,
                "runtime_errors": 1,
            },
        },
    )

    summary = build_acceptance_summary(*paths)

    assert summary["successful"] is False
    assert "vla_metrics.responses_succeeded must be positive" in summary["failures"]
    assert "vla_metrics.authority_steps must be positive" in summary["failures"]
    assert "vla_metrics.runtime_errors must be zero" in summary["failures"]


def test_native_vlm_summary_can_record_authority_without_requiring_it(
    tmp_path: Path,
) -> None:
    paths = _write_evidence(
        tmp_path,
        {
            "simulation_time_s": 2.0,
            "crashed": False,
            "went_off_road": False,
            "vla_metrics": {
                "provider_requests_attempted": 3,
                "responses_succeeded": 3,
                "authority_steps": 0,
                "runtime_errors": 0,
            },
        },
    )

    summary = build_acceptance_summary(*paths, require_authority=False)

    assert summary["successful"] is True
    assert summary["authority_steps"] == 0
    assert summary["authority_required"] is False


def test_native_vlm_summary_records_the_schema_off_probe_as_evidence(
    tmp_path: Path,
) -> None:
    paths = _write_evidence(
        tmp_path,
        {
            "simulation_time_s": 2.0,
            "crashed": False,
            "went_off_road": False,
            "vla_metrics": {
                "provider_requests_attempted": 3,
                "responses_succeeded": 2,
                "authority_steps": 7,
                "runtime_errors": 0,
            },
        },
    )
    probe = tmp_path / "summary.json"
    probe.write_text(
        json.dumps(
            {
                "successful": False,
                "scenarios": [
                    {"scenario_id": "a", "status": "success"},
                    {"scenario_id": "b", "status": "failure", "error_category": "ModelTimeoutError"},
                    {
                        "scenario_id": "c",
                        "status": "failure",
                        "error_category": "VLAAssessmentPayloadError",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_acceptance_summary(*paths, schema_off_probe=probe)
    unavailable = build_acceptance_summary(*paths, schema_off_probe=tmp_path / "absent.json")

    assert summary["successful"] is True
    assert summary["schema_off_probe"] == {
        "scenes": 3,
        "decoded": 1,
        "failure_categories": {"ModelTimeoutError": 1, "VLAAssessmentPayloadError": 1},
    }
    assert unavailable["successful"] is True
    assert unavailable["schema_off_probe"]["scenes"] is None


def test_native_vlm_summary_requires_server_resource_samples(tmp_path: Path) -> None:
    paths = list(
        _write_evidence(
            tmp_path,
            {
                "simulation_time_s": 2.0,
                "crashed": False,
                "went_off_road": False,
                "vla_metrics": {
                    "provider_requests_attempted": 1,
                    "responses_succeeded": 1,
                    "authority_steps": 1,
                    "runtime_errors": 0,
                },
            },
        )
    )
    paths[2].write_text(
        "timestamp_utc,pid,rss_kib,vsz_kib,cpu_percent,elapsed\n",
        encoding="utf-8",
    )

    with pytest.raises(AcceptanceError, match="no samples"):
        build_acceptance_summary(*paths)
