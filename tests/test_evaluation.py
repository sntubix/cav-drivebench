from __future__ import annotations

import json
from pathlib import Path

import pytest

from metadrive_starter.config import config_from_dict
from metadrive_starter.evaluation import load_evaluation_plan, run_evaluation
from metadrive_starter.faults import FaultKind
from metadrive_starter.simulation import (
    RunSummary,
    VLAMetricCount,
    VLARunMetrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_loads_seed_scenario_matrix_and_faults(tmp_path) -> None:
    path = tmp_path / "evaluation.yaml"
    path.write_text(
        """
version: 1
seeds: [2, 5]
scenarios:
  - id: lidar-loss
    map: C
    horizon: 12
    safety_source: lidar
    faults:
      - id: loss-1
        kind: lidar_dropout
        start_step: 3
        duration_steps: 4
""",
        encoding="utf-8",
    )

    plan = load_evaluation_plan(path)

    assert plan.seeds == (2, 5)
    assert plan.scenarios[0].scenario_id == "lidar-loss"
    assert plan.scenarios[0].faults[0].kind is FaultKind.LIDAR_DROPOUT


@pytest.mark.needs("fixtures")
def test_public_lane_change_suite_manifest_has_deterministic_acceptance_cases() -> None:
    plan = load_evaluation_plan(PROJECT_ROOT / "configs" / "lane-change-scenarios.yaml")

    assert plan.seeds == (0,)
    assert [scenario.scenario_id for scenario in plan.scenarios] == [
        "clear-right-completion",
        "left-road-edge-rejection",
        "fast-rear-right-rejection",
        "target-lane-hazard-abort",
    ]
    for scenario in plan.scenarios:
        assert scenario.vla_fixture_path is not None
        assert (PROJECT_ROOT / scenario.vla_fixture_path).exists()
        assert scenario.expectations
    fast_rear = plan.scenarios[2].vehicles[0]
    assert fast_rear.vehicle_id == "fast-rear-right"
    assert fast_rear.longitudinal_offset_m == -18.0
    assert fast_rear.lane_offset == 1
    assert fast_rear.speed_mps == 12.0


def test_manifest_loads_lane_change_fixture_hazard_and_expectations(tmp_path) -> None:
    path = tmp_path / "lane-change-evaluation.yaml"
    path.write_text(
        """
version: 1
seeds: [0]
scenarios:
  - id: target-lane-hazard
    vla_fixture_path: fixtures/vla/providers/lane-change-left
    lane_change_hazard:
      enabled: true
      distance_ahead_m: 6.0
      vehicle_kind: truck
    expect:
      lane_changes_started: 1
      lane_changes_completed: 0
      lane_changes_aborted: 1
      crashed: false
""",
        encoding="utf-8",
    )

    scenario = load_evaluation_plan(path).scenarios[0]

    assert scenario.vla_fixture_path == "fixtures/vla/providers/lane-change-left"
    assert scenario.lane_change_hazard is not None
    assert scenario.lane_change_hazard.enabled is True
    assert scenario.lane_change_hazard.distance_ahead_m == 6.0
    assert scenario.lane_change_hazard.vehicle_kind == "truck"
    assert dict(scenario.expectations) == {
        "lane_changes_started": 1,
        "lane_changes_completed": 0,
        "lane_changes_aborted": 1,
        "crashed": False,
    }


def test_manifest_loads_relative_scenario_vehicles_and_evaluator_applies_them(
    tmp_path,
) -> None:
    manifest = tmp_path / "relative-vehicles.yaml"
    manifest.write_text(
        """
version: 1
seeds: [0]
scenarios:
  - id: fast-rear-right
    vehicles:
      - id: fast-rear-right
        longitudinal_offset_m: -18.0
        lane_offset: 1
        speed_mps: 12.0
        kind: car
""",
        encoding="utf-8",
    )
    received = []

    def runner(config, *, dry_run: bool):
        received.append((config, dry_run))
        return RunSummary(steps=1, total_reward=0.0, control_mode="vla")

    plan = load_evaluation_plan(manifest)
    report = run_evaluation(
        config_from_dict({}),
        plan,
        tmp_path / "results",
        runner=runner,
    )

    vehicle = plan.scenarios[0].vehicles[0]
    applied_vehicle = received[0][0].scenario.vehicles[0]
    assert vehicle.vehicle_id == "fast-rear-right"
    assert vehicle.longitudinal_offset_m == -18.0
    assert vehicle.lane_offset == 1
    assert vehicle.speed_mps == 12.0
    assert applied_vehicle == vehicle
    assert report.successful is True


def test_evaluator_applies_lane_change_scenario_and_accepts_matching_summary(
    tmp_path,
) -> None:
    manifest = tmp_path / "lane-change-evaluation.yaml"
    manifest.write_text(
        """
version: 1
seeds: [0]
scenarios:
  - id: target-lane-hazard
    vla_fixture_path: fixtures/vla/providers/lane-change-left
    lane_change_hazard:
      enabled: true
      distance_ahead_m: 6.0
    expect:
      lane_changes_started: 1
      lane_changes_completed: 0
      lane_changes_aborted: 1
      emergency_brakes: 1
      crashed: false
      went_off_road: false
""",
        encoding="utf-8",
    )
    received = []

    def runner(config, *, dry_run: bool):
        received.append((config, dry_run))
        return RunSummary(
            steps=20,
            total_reward=1.0,
            control_mode="vla",
            emergency_brakes=1,
            lane_changes_started=1,
            lane_changes_completed=0,
            lane_changes_aborted=1,
            crashed=False,
            went_off_road=False,
        )

    base_config = config_from_dict(
        {
            "vla": {
                "enabled": True,
                "provider": "fixture",
                "fixture": {"path": "fixtures/original.json"},
            },
            "planner": {"lane_change_enabled": True},
        }
    )
    report = run_evaluation(
        base_config,
        load_evaluation_plan(manifest),
        tmp_path / "results",
        runner=runner,
    )

    config, dry_run = received[0]
    assert dry_run is False
    assert config.vla.fixture.path == "fixtures/vla/providers/lane-change-left"
    assert config.scenario.lane_change_hazard.enabled is True
    assert config.scenario.lane_change_hazard.distance_ahead_m == 6.0
    assert report.runs[0].acceptance_passed is True
    assert report.runs[0].acceptance_failures == ()
    assert report.successful is True


def test_evaluator_reports_lane_change_acceptance_mismatch(tmp_path) -> None:
    manifest = tmp_path / "lane-change-evaluation.yaml"
    manifest.write_text(
        """
version: 1
seeds: [0]
scenarios:
  - id: clear-left
    expect:
      lane_changes_started: 1
      lane_changes_completed: 1
""",
        encoding="utf-8",
    )

    report = run_evaluation(
        config_from_dict({}),
        load_evaluation_plan(manifest),
        tmp_path / "results",
        runner=lambda config, dry_run: RunSummary(
            steps=10,
            total_reward=0.0,
            control_mode="vla",
        ),
    )

    assert report.runs[0].status == "completed"
    assert report.runs[0].acceptance_passed is False
    assert report.runs[0].acceptance_failures == (
        "lane_changes_started: expected 1, got 0",
        "lane_changes_completed: expected 1, got 0",
    )
    assert report.successful is False
    saved = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert saved["successful"] is False
    assert saved["runs"][0]["acceptance_passed"] is False


def test_evaluator_runs_cartesian_product_and_aggregates_without_stopping_on_failure(
    tmp_path,
) -> None:
    manifest = tmp_path / "evaluation.yaml"
    manifest.write_text(
        """
version: 1
seeds: [0, 1]
scenarios:
  - id: baseline
    map: S
    horizon: 5
  - id: stopped
    map: C
    horizon: 7
    stopped_vehicle_ahead_m: 20.0
""",
        encoding="utf-8",
    )
    plan = load_evaluation_plan(manifest)
    received = []

    def runner(config, *, dry_run: bool):
        received.append((config, dry_run))
        if config.simulator.start_seed == 1 and config.simulator.map == "C":
            raise RuntimeError("injected run failure")
        return RunSummary(
            steps=config.simulator.horizon,
            total_reward=10.0 + config.simulator.start_seed,
            control_mode="autopilot",
            route_completion=0.5,
            arrived=config.simulator.start_seed == 0,
            crashed=False,
            went_off_road=False,
            safety_interventions=2,
            lane_change_hazards_spawned=1,
            fault_activations=len(config.faults),
            vla_metrics=VLARunMetrics(
                enabled=True,
                request_cap=2,
                provider_requests_attempted=2,
                provider_requests_remaining=0,
                request_cap_exhausted=True,
                requests_started=2,
                responses_succeeded=1,
                responses_failed=1,
                validations_accepted=1,
                authority_steps=3,
                fallback_steps=1,
                observations_built=2,
                observations_without_output_contract=1,
                latency_p50_s=0.5,
                latency_p95_s=0.75,
                latency_max_s=0.75,
                responses_with_token_usage=1,
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
                providers=("vertex",),
                model_ids=("course-model",),
                model_versions=("course-model-v1",),
                fixture_ids=("recorded-course-response",),
                fixture_sha256s=("f" * 64,),
                failure_categories=(VLAMetricCount("quota", 1),),
                validation_reasons=(VLAMetricCount("speed capped", 1),),
            ),
        )

    output = tmp_path / "results"
    report = run_evaluation(
        config_from_dict({}),
        plan,
        output,
        dry_run=True,
        runner=runner,
    )

    assert len(received) == 4
    assert all(dry_run for _, dry_run in received)
    assert {(config.simulator.map, config.simulator.start_seed) for config, _ in received} == {
        ("S", 0),
        ("S", 1),
        ("C", 0),
        ("C", 1),
    }
    assert all(config.event_log.enabled for config, _ in received)
    assert report.successful is False
    assert report.aggregate["runs"] == 4
    assert report.aggregate["completed"] == 3
    assert report.aggregate["failed"] == 1
    assert report.aggregate["arrival_rate"] == pytest.approx(2 / 3)
    assert report.aggregate["safety_interventions"] == 6
    assert report.aggregate["lane_change_hazards_spawned"] == 3
    vla = report.aggregate["vla"]
    assert vla["enabled_runs"] == 3
    assert vla["request_caps"] == [2]
    assert vla["provider_requests_attempted"] == 6
    assert vla["request_cap_exhausted_runs"] == 3
    assert vla["requests_started"] == 6
    assert vla["responses_succeeded"] == 3
    assert vla["responses_failed"] == 3
    assert vla["authority_rate"] == 0.75
    assert vla["observations_built"] == 6
    assert vla["observations_without_output_contract"] == 3
    assert vla["maximum_latency_s"] == 0.75
    assert vla["total_tokens"] == 36
    assert vla["providers"] == ["vertex"]
    assert vla["fixture_ids"] == ["recorded-course-response"]
    assert vla["fixture_sha256s"] == ["f" * 64]
    assert vla["failure_categories"] == {"quota": 3}
    assert vla["validation_reasons"] == {"speed capped": 3}
    assert report.by_scenario["stopped"]["failed"] == 1
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["successful"] is False
    assert len(saved["runs"]) == 4


def test_evaluator_refuses_to_overwrite_existing_artifact_root(tmp_path) -> None:
    manifest = tmp_path / "evaluation.yaml"
    manifest.write_text(
        "version: 1\nseeds: [0]\nscenarios:\n  - id: baseline\n",
        encoding="utf-8",
    )
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        run_evaluation(
            config_from_dict({}),
            load_evaluation_plan(manifest),
            output,
            dry_run=True,
        )


@pytest.mark.parametrize(
    "content",
    [
        "version: 2\nseeds: [0]\nscenarios: [{id: baseline}]\n",
        "version: 1\nseeds: []\nscenarios: [{id: baseline}]\n",
        "version: 1\nseeds: [0, 0]\nscenarios: [{id: baseline}]\n",
        "version: 1\nseeds: [0]\nscenarios: [{id: baseline, extra: true}]\n",
    ],
)
def test_manifest_rejects_invalid_or_ambiguous_inputs(tmp_path, content: str) -> None:
    path = tmp_path / "evaluation.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_evaluation_plan(path)
