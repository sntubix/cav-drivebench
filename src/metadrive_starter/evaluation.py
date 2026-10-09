from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping

import yaml

from metadrive_starter.config import (
    AppConfig,
    EventLogSettings,
    LaneChangeHazardScenarioSettings,
    ScenarioVehicleSettings,
)
from metadrive_starter.events import to_json_value
from metadrive_starter.faults import FaultSpec
from metadrive_starter.simulation import RunSummary, run_simulation


EVALUATION_MANIFEST_VERSION = 1
_EXPECTATION_FIELDS = {
    "crashed",
    "emergency_brakes",
    "lane_changes_aborted",
    "lane_changes_completed",
    "lane_changes_started",
    "validations_rejected",
    "went_off_road",
}


@dataclass(frozen=True)
class EvaluationScenario:
    scenario_id: str
    map: str | None = None
    traffic_density: float | None = None
    obstacle_probability: float | None = None
    horizon: int | None = None
    stopped_vehicle_ahead_m: float | None = None
    safety_source: str | None = None
    faults: tuple[FaultSpec, ...] = ()
    vla_fixture_path: str | None = None
    lane_change_hazard: LaneChangeHazardScenarioSettings | None = None
    vehicles: tuple[ScenarioVehicleSettings, ...] = ()
    expectations: tuple[tuple[str, int | bool], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.scenario_id
        ) is None:
            raise ValueError(
                "evaluation scenario id must contain only letters, numbers, '.', '_', or '-'"
            )
        if self.map is not None and (
            not isinstance(self.map, str) or not self.map.strip()
        ):
            raise ValueError("evaluation scenario map must not be empty")
        for name in ("traffic_density", "obstacle_probability"):
            value = getattr(self, name)
            if value is not None and not _non_negative_finite(value):
                raise ValueError(f"evaluation scenario {name} must be finite and non-negative")
        if self.obstacle_probability is not None and self.obstacle_probability > 1.0:
            raise ValueError(
                "evaluation scenario obstacle_probability must be between 0 and 1"
            )
        if self.horizon is not None and not _positive_integer(self.horizon):
            raise ValueError("evaluation scenario horizon must be a positive integer")
        if self.stopped_vehicle_ahead_m is not None and not _positive_finite(
            self.stopped_vehicle_ahead_m
        ):
            raise ValueError(
                "evaluation scenario stopped_vehicle_ahead_m must be finite and positive"
            )
        if self.safety_source is not None and self.safety_source not in {
            "oracle",
            "lidar",
        }:
            raise ValueError("evaluation scenario safety_source must be 'oracle' or 'lidar'")
        if self.vla_fixture_path is not None and (
            not isinstance(self.vla_fixture_path, str)
            or not self.vla_fixture_path.strip()
        ):
            raise ValueError("evaluation scenario vla_fixture_path must not be empty")
        if self.lane_change_hazard is not None and not isinstance(
            self.lane_change_hazard,
            LaneChangeHazardScenarioSettings,
        ):
            raise ValueError(
                "evaluation scenario lane_change_hazard must be lane-change hazard settings"
            )
        if any(
            not isinstance(vehicle, ScenarioVehicleSettings)
            for vehicle in self.vehicles
        ):
            raise ValueError(
                "evaluation scenario vehicles must be scenario vehicle settings"
            )
        vehicle_ids = [vehicle.vehicle_id for vehicle in self.vehicles]
        if len(set(vehicle_ids)) != len(vehicle_ids):
            raise ValueError("evaluation scenario vehicle ids must be unique")
        expectation_names = [name for name, _ in self.expectations]
        if len(set(expectation_names)) != len(expectation_names):
            raise ValueError("evaluation scenario expectation fields must be unique")
        unknown_expectations = sorted(set(expectation_names) - _EXPECTATION_FIELDS)
        if unknown_expectations:
            raise ValueError(
                "evaluation scenario has unknown expectation fields: "
                + ", ".join(unknown_expectations)
            )
        for name, expected in self.expectations:
            if name in {"crashed", "went_off_road"}:
                if not isinstance(expected, bool):
                    raise ValueError(f"evaluation expectation {name} must be a boolean")
            elif not _non_negative_integer(expected):
                raise ValueError(
                    f"evaluation expectation {name} must be a non-negative integer"
                )
        fault_ids = [fault.fault_id for fault in self.faults]
        if len(set(fault_ids)) != len(fault_ids):
            raise ValueError("evaluation scenario fault ids must be unique")


@dataclass(frozen=True)
class EvaluationPlan:
    seeds: tuple[int, ...]
    scenarios: tuple[EvaluationScenario, ...]

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("evaluation seeds must not be empty")
        if any(not _non_negative_integer(seed) for seed in self.seeds):
            raise ValueError("evaluation seeds must be non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("evaluation seeds must be unique")
        if not self.scenarios:
            raise ValueError("evaluation scenarios must not be empty")
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(set(ids)) != len(ids):
            raise ValueError("evaluation scenario ids must be unique")


@dataclass(frozen=True)
class EvaluationRunResult:
    scenario_id: str
    seed: int
    status: str
    event_log_path: str
    summary: RunSummary | None = None
    error_category: str | None = None
    error_message: str | None = None
    acceptance_passed: bool | None = None
    acceptance_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationReport:
    output_dir: str
    runs: tuple[EvaluationRunResult, ...]
    aggregate: Mapping[str, object]
    by_scenario: Mapping[str, Mapping[str, object]]

    @property
    def successful(self) -> bool:
        return all(
            run.status == "completed" and run.acceptance_passed is not False
            for run in self.runs
        )

    def to_dict(self) -> dict[str, object]:
        value = to_json_value(self)
        assert isinstance(value, dict)
        value["successful"] = self.successful
        return value


def load_evaluation_plan(path: Path | str) -> EvaluationPlan:
    manifest_path = Path(path)
    value = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evaluation manifest must be a mapping")
    _reject_unknown(value, {"version", "seeds", "scenarios"}, "evaluation manifest")
    if value.get("version") != EVALUATION_MANIFEST_VERSION:
        raise ValueError(
            f"evaluation manifest version must be {EVALUATION_MANIFEST_VERSION}"
        )
    raw_seeds = value.get("seeds")
    raw_scenarios = value.get("scenarios")
    if not isinstance(raw_seeds, list):
        raise ValueError("evaluation seeds must be a list")
    if not isinstance(raw_scenarios, list):
        raise ValueError("evaluation scenarios must be a list")
    scenarios = tuple(_parse_scenario(item) for item in raw_scenarios)
    return EvaluationPlan(seeds=tuple(raw_seeds), scenarios=scenarios)


def run_evaluation(
    base_config: AppConfig,
    plan: EvaluationPlan,
    output_dir: Path | str,
    *,
    dry_run: bool = False,
    runner: Callable[..., RunSummary] = run_simulation,
) -> EvaluationReport:
    if not isinstance(base_config, AppConfig):
        raise TypeError("base_config must be an AppConfig")
    if not isinstance(plan, EvaluationPlan):
        raise TypeError("plan must be an EvaluationPlan")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    runs: list[EvaluationRunResult] = []

    for scenario in plan.scenarios:
        for seed in plan.seeds:
            config = _config_for_run(base_config, scenario, seed, root)
            event_log_path = config.event_log.path
            try:
                summary = runner(config, dry_run=dry_run)
            except Exception as exc:
                runs.append(
                    EvaluationRunResult(
                        scenario_id=scenario.scenario_id,
                        seed=seed,
                        status="failed",
                        event_log_path=event_log_path,
                        error_category=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
            else:
                acceptance_failures = _acceptance_failures(
                    summary,
                    scenario.expectations,
                )
                runs.append(
                    EvaluationRunResult(
                        scenario_id=scenario.scenario_id,
                        seed=seed,
                        status="completed",
                        event_log_path=event_log_path,
                        summary=summary,
                        acceptance_passed=(
                            not acceptance_failures
                            if scenario.expectations
                            else None
                        ),
                        acceptance_failures=acceptance_failures,
                    )
                )

    by_scenario = {
        scenario.scenario_id: _aggregate(
            tuple(run for run in runs if run.scenario_id == scenario.scenario_id)
        )
        for scenario in plan.scenarios
    }
    report = EvaluationReport(
        output_dir=str(root),
        runs=tuple(runs),
        aggregate=_aggregate(tuple(runs)),
        by_scenario=by_scenario,
    )
    (root / "summary.json").write_text(
        json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _config_for_run(
    base_config: AppConfig,
    scenario: EvaluationScenario,
    seed: int,
    root: Path,
) -> AppConfig:
    config = copy.deepcopy(base_config)
    config.simulator.start_seed = seed
    if scenario.map is not None:
        config.simulator.map = scenario.map
    if scenario.traffic_density is not None:
        config.simulator.traffic_density = scenario.traffic_density
    if scenario.obstacle_probability is not None:
        config.simulator.obstacle_probability = scenario.obstacle_probability
    if scenario.horizon is not None:
        config.simulator.horizon = scenario.horizon
    config.scenario.stopped_vehicle_ahead_m = scenario.stopped_vehicle_ahead_m
    config.scenario.vehicles = copy.deepcopy(scenario.vehicles)
    if scenario.safety_source is not None:
        config.perception.safety_source = scenario.safety_source
    if scenario.vla_fixture_path is not None:
        config.vla = replace(
            config.vla,
            fixture=replace(
                config.vla.fixture,
                path=scenario.vla_fixture_path,
            ),
        )
    if scenario.lane_change_hazard is not None:
        config.scenario.lane_change_hazard = copy.deepcopy(
            scenario.lane_change_hazard
        )
    config.faults = scenario.faults
    event_path = root / scenario.scenario_id / f"seed-{seed}" / "events.jsonl"
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(event_path),
        scenario_id=scenario.scenario_id,
    )
    config.__post_init__()
    return config


def _acceptance_failures(
    summary: RunSummary,
    expectations: tuple[tuple[str, int | bool], ...],
) -> tuple[str, ...]:
    actual: dict[str, int | bool] = {
        "crashed": summary.crashed,
        "emergency_brakes": summary.emergency_brakes,
        "lane_changes_aborted": summary.lane_changes_aborted,
        "lane_changes_completed": summary.lane_changes_completed,
        "lane_changes_started": summary.lane_changes_started,
        "validations_rejected": summary.vla_metrics.validations_rejected,
        "went_off_road": summary.went_off_road,
    }
    return tuple(
        f"{name}: expected {expected!r}, got {actual[name]!r}"
        for name, expected in expectations
        if actual[name] != expected
    )


def _aggregate(runs: tuple[EvaluationRunResult, ...]) -> dict[str, object]:
    completed = tuple(run for run in runs if run.summary is not None)
    summaries = tuple(run.summary for run in completed if run.summary is not None)
    return aggregate_run_summaries(summaries, total_runs=len(runs))


def aggregate_run_summaries(
    summaries: tuple[RunSummary, ...],
    *,
    total_runs: int,
) -> dict[str, object]:
    """Aggregate common episode outcomes for evaluators with different manifests."""
    if not _non_negative_integer(total_runs):
        raise ValueError("total_runs must be a non-negative integer")
    if len(summaries) > total_runs:
        raise ValueError("completed summary count must not exceed total_runs")
    gaps = tuple(
        summary.minimum_safety_gap_m
        for summary in summaries
        if summary.minimum_safety_gap_m is not None
    )
    count = len(summaries)
    return {
        "runs": total_runs,
        "completed": count,
        "failed": total_runs - count,
        "arrival_rate": _rate(sum(summary.arrived for summary in summaries), count),
        "crash_rate": _rate(sum(summary.crashed for summary in summaries), count),
        "off_road_rate": _rate(
            sum(summary.went_off_road for summary in summaries), count
        ),
        "mean_reward": _mean(tuple(summary.total_reward for summary in summaries)),
        "mean_steps": _mean(tuple(float(summary.steps) for summary in summaries)),
        "mean_simulation_time_s": _mean(
            tuple(summary.simulation_time_s for summary in summaries)
        ),
        "mean_route_completion": _mean(
            tuple(summary.route_completion for summary in summaries)
        ),
        "mean_minimum_safety_gap_m": _mean(gaps),
        "safety_interventions": sum(
            summary.safety_interventions for summary in summaries
        ),
        "emergency_brakes": sum(summary.emergency_brakes for summary in summaries),
        "headway_cap_would_intervene": sum(
            summary.headway_cap_would_intervene for summary in summaries
        ),
        "headway_cap_applied": sum(
            summary.headway_cap_applied for summary in summaries
        ),
        "lane_changes_started": sum(
            summary.lane_changes_started for summary in summaries
        ),
        "lane_changes_completed": sum(
            summary.lane_changes_completed for summary in summaries
        ),
        "lane_changes_aborted": sum(
            summary.lane_changes_aborted for summary in summaries
        ),
        "lane_change_hazards_spawned": sum(
            summary.lane_change_hazards_spawned for summary in summaries
        ),
        "fault_activations": sum(summary.fault_activations for summary in summaries),
        "vla": _aggregate_vla_run_metrics(summaries),
    }


def _aggregate_vla_run_metrics(
    summaries: tuple[RunSummary, ...],
) -> dict[str, object]:
    enabled = tuple(
        summary.vla_metrics
        for summary in summaries
        if summary.vla_metrics.enabled
    )
    failure_categories = Counter[str]()
    runtime_error_categories = Counter[str]()
    validation_reasons = Counter[str]()
    providers: set[str] = set()
    model_ids: set[str] = set()
    model_versions: set[str] = set()
    fixture_ids: set[str] = set()
    fixture_sha256s: set[str] = set()
    prompt_versions: set[str] = set()
    for metrics in enabled:
        failure_categories.update(
            {item.label: item.count for item in metrics.failure_categories}
        )
        runtime_error_categories.update(
            {item.label: item.count for item in metrics.runtime_error_categories}
        )
        validation_reasons.update(
            {item.label: item.count for item in metrics.validation_reasons}
        )
        providers.update(metrics.providers)
        model_ids.update(metrics.model_ids)
        model_versions.update(metrics.model_versions)
        fixture_ids.update(metrics.fixture_ids)
        fixture_sha256s.update(metrics.fixture_sha256s)
        prompt_versions.add(metrics.prompt_contract_version)

    authority_steps = sum(metrics.authority_steps for metrics in enabled)
    fallback_steps = sum(metrics.fallback_steps for metrics in enabled)
    controlled_steps = authority_steps + fallback_steps
    return {
        "enabled_runs": len(enabled),
        "prompt_contract_versions": sorted(prompt_versions),
        "request_caps": sorted(
            {
                metrics.request_cap
                for metrics in enabled
                if metrics.request_cap is not None
            }
        ),
        "provider_requests_attempted": sum(
            metrics.provider_requests_attempted or 0 for metrics in enabled
        ),
        "request_cap_exhausted_runs": sum(
            metrics.request_cap_exhausted for metrics in enabled
        ),
        "requests_started": sum(metrics.requests_started for metrics in enabled),
        "submissions_busy": sum(metrics.submissions_busy for metrics in enabled),
        "submissions_rate_limited": sum(
            metrics.submissions_rate_limited for metrics in enabled
        ),
        "submissions_closed": sum(metrics.submissions_closed for metrics in enabled),
        "responses_succeeded": sum(
            metrics.responses_succeeded for metrics in enabled
        ),
        "responses_failed": sum(metrics.responses_failed for metrics in enabled),
        "responses_discarded": sum(
            metrics.responses_discarded for metrics in enabled
        ),
        "validations_accepted": sum(
            metrics.validations_accepted for metrics in enabled
        ),
        "validations_modified": sum(
            metrics.validations_modified for metrics in enabled
        ),
        "validations_rejected": sum(
            metrics.validations_rejected for metrics in enabled
        ),
        "validations_fallback": sum(
            metrics.validations_fallback for metrics in enabled
        ),
        "authority_steps": authority_steps,
        "fallback_steps": fallback_steps,
        "authority_rate": _rate(authority_steps, controlled_steps),
        "runtime_errors": sum(metrics.runtime_errors for metrics in enabled),
        "observations_built": sum(metrics.observations_built for metrics in enabled),
        "observations_without_output_contract": sum(
            metrics.observations_without_output_contract for metrics in enabled
        ),
        "mean_run_latency_p50_s": _mean(
            tuple(
                metrics.latency_p50_s
                for metrics in enabled
                if metrics.latency_p50_s is not None
            )
        ),
        "mean_run_latency_p95_s": _mean(
            tuple(
                metrics.latency_p95_s
                for metrics in enabled
                if metrics.latency_p95_s is not None
            )
        ),
        "maximum_latency_s": max(
            (
                metrics.latency_max_s
                for metrics in enabled
                if metrics.latency_max_s is not None
            ),
            default=None,
        ),
        "responses_with_token_usage": sum(
            metrics.responses_with_token_usage for metrics in enabled
        ),
        "input_tokens": sum(metrics.input_tokens for metrics in enabled),
        "output_tokens": sum(metrics.output_tokens for metrics in enabled),
        "total_tokens": sum(metrics.total_tokens for metrics in enabled),
        "providers": sorted(providers),
        "model_ids": sorted(model_ids),
        "model_versions": sorted(model_versions),
        "fixture_ids": sorted(fixture_ids),
        "fixture_sha256s": sorted(fixture_sha256s),
        "failure_categories": dict(sorted(failure_categories.items())),
        "runtime_error_categories": dict(sorted(runtime_error_categories.items())),
        "validation_reasons": dict(sorted(validation_reasons.items())),
    }


def _parse_scenario(value: object) -> EvaluationScenario:
    if not isinstance(value, dict):
        raise ValueError("evaluation scenario must be a mapping")
    allowed = {
        "id",
        "map",
        "traffic_density",
        "obstacle_probability",
        "horizon",
        "stopped_vehicle_ahead_m",
        "safety_source",
        "faults",
        "vla_fixture_path",
        "lane_change_hazard",
        "vehicles",
        "expect",
    }
    _reject_unknown(value, allowed, "evaluation scenario")
    raw_faults = value.get("faults", [])
    if not isinstance(raw_faults, list):
        raise ValueError("evaluation scenario faults must be a list")
    try:
        scenario_id = value["id"]
    except KeyError as exc:
        raise ValueError("evaluation scenario is missing field 'id'") from exc
    return EvaluationScenario(
        scenario_id=scenario_id,
        map=value.get("map"),
        traffic_density=value.get("traffic_density"),
        obstacle_probability=value.get("obstacle_probability"),
        horizon=value.get("horizon"),
        stopped_vehicle_ahead_m=value.get("stopped_vehicle_ahead_m"),
        safety_source=value.get("safety_source"),
        faults=tuple(_parse_fault(fault) for fault in raw_faults),
        vla_fixture_path=value.get("vla_fixture_path"),
        lane_change_hazard=_parse_lane_change_hazard(
            value.get("lane_change_hazard")
        ),
        vehicles=_parse_scenario_vehicles(value.get("vehicles")),
        expectations=_parse_expectations(value.get("expect")),
    )


def _parse_scenario_vehicles(
    value: object,
) -> tuple[ScenarioVehicleSettings, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("evaluation scenario vehicles must be a list")
    vehicles: list[ScenarioVehicleSettings] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("evaluation scenario vehicle must be a mapping")
        _reject_unknown(
            item,
            {"id", "longitudinal_offset_m", "lane_offset", "speed_mps", "kind"},
            "evaluation scenario vehicle",
        )
        if "id" not in item:
            raise ValueError("evaluation scenario vehicle is missing field 'id'")
        if "longitudinal_offset_m" not in item:
            raise ValueError(
                "evaluation scenario vehicle is missing field 'longitudinal_offset_m'"
            )
        vehicles.append(
            ScenarioVehicleSettings(
                vehicle_id=item["id"],
                longitudinal_offset_m=item["longitudinal_offset_m"],
                lane_offset=item.get("lane_offset", 0),
                speed_mps=item.get("speed_mps", 0.0),
                kind=item.get("kind", "car"),
            )
        )
    return tuple(vehicles)


def _parse_lane_change_hazard(
    value: object,
) -> LaneChangeHazardScenarioSettings | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("evaluation scenario lane_change_hazard must be a mapping")
    _reject_unknown(
        value,
        {"enabled", "distance_ahead_m", "vehicle_kind"},
        "evaluation scenario lane_change_hazard",
    )
    return LaneChangeHazardScenarioSettings(**value)


def _parse_expectations(value: object) -> tuple[tuple[str, int | bool], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("evaluation scenario expect must be a mapping")
    return tuple((name, expected) for name, expected in value.items())


def _parse_fault(value: object) -> FaultSpec:
    if not isinstance(value, dict):
        raise ValueError("evaluation fault must be a mapping")
    _reject_unknown(
        value,
        {"id", "kind", "start_step", "duration_steps", "latency_s"},
        "evaluation fault",
    )
    try:
        return FaultSpec(
            fault_id=value["id"],
            kind=value["kind"],
            start_step=value["start_step"],
            duration_steps=value["duration_steps"],
            latency_s=value.get("latency_s", 0.0),
        )
    except KeyError as exc:
        raise ValueError(f"evaluation fault is missing field {exc.args[0]!r}") from exc


def _reject_unknown(
    value: Mapping[str, object], allowed: set[str], label: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _mean(values: tuple[float, ...]) -> float | None:
    return fmean(values) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _positive_finite(value: object) -> bool:
    return _non_negative_finite(value) and value > 0.0  # type: ignore[operator]


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
