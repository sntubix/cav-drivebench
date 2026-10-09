from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import yaml

from metadrive_starter.config import (
    AppConfig,
    EpisodeRecordingSettings,
    EventLogSettings,
    ScenarioSettings,
)
from metadrive_starter.evaluation import aggregate_run_summaries
from metadrive_starter.events import to_json_value
from metadrive_starter.simulation import RunSummary, run_simulation


RACE_MANIFEST_VERSION = 1
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class RaceRuntime:
    decision_repeat: int
    physics_world_step_size: float
    realtime_factor: float

    def __post_init__(self) -> None:
        if not _positive_integer(self.decision_repeat):
            raise ValueError("race runtime decision_repeat must be a positive integer")
        if not _positive_finite(self.physics_world_step_size):
            raise ValueError(
                "race runtime physics_world_step_size must be finite and positive"
            )
        if not _positive_finite(self.realtime_factor):
            raise ValueError("race runtime realtime_factor must be finite and positive")
        if self.realtime_factor != 1.0:
            raise ValueError("race runtime realtime_factor must be 1.0")


@dataclass(frozen=True)
class RaceScenario:
    scenario_id: str
    seed: int
    map: str
    traffic_density: float
    obstacle_probability: float
    horizon: int
    stopped_vehicle_ahead_m: float | None = None

    def __post_init__(self) -> None:
        _validate_id(self.scenario_id, "race scenario id")
        if not _non_negative_integer(self.seed):
            raise ValueError("race scenario seed must be a non-negative integer")
        if not isinstance(self.map, str) or not self.map.strip():
            raise ValueError("race scenario map must not be empty")
        if not _unit_interval(self.traffic_density):
            raise ValueError(
                "race scenario traffic_density must be finite and between 0 and 1"
            )
        if not _unit_interval(self.obstacle_probability):
            raise ValueError(
                "race scenario obstacle_probability must be finite and between 0 and 1"
            )
        if not _positive_integer(self.horizon):
            raise ValueError("race scenario horizon must be a positive integer")
        if self.stopped_vehicle_ahead_m is not None and not _positive_finite(
            self.stopped_vehicle_ahead_m
        ):
            raise ValueError(
                "race scenario stopped_vehicle_ahead_m must be finite and positive"
            )


@dataclass(frozen=True)
class RacePlan:
    suite_id: str
    visibility: str
    runtime: RaceRuntime
    scenarios: tuple[RaceScenario, ...]

    def __post_init__(self) -> None:
        _validate_id(self.suite_id, "race suite id")
        if self.visibility not in {"public", "hidden"}:
            raise ValueError("race visibility must be 'public' or 'hidden'")
        if not isinstance(self.runtime, RaceRuntime):
            raise ValueError("race runtime must be RaceRuntime")
        if not self.scenarios:
            raise ValueError("race scenarios must not be empty")
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("race scenario ids must be unique")


@dataclass(frozen=True)
class RaceRunResult:
    scenario_id: str
    status: str
    environment_sha256: str
    event_log_path: str
    replay_artifact_path: str | None = None
    replay_artifact_sha256: str | None = None
    summary: RunSummary | None = None
    error_category: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class RaceReport:
    suite_id: str
    visibility: str
    rendered: bool
    manifest_sha256: str
    base_config_sha256: str
    output_dir: str
    runs: tuple[RaceRunResult, ...]
    aggregate: Mapping[str, object]

    @property
    def successful(self) -> bool:
        return all(run.status == "completed" for run in self.runs)

    def to_dict(self) -> dict[str, object]:
        value = to_json_value(self)
        assert isinstance(value, dict)
        value["successful"] = self.successful
        return value


def load_race_plan(path: Path | str) -> RacePlan:
    manifest_path = Path(path)
    value = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("race manifest must be a mapping")
    _reject_unknown(
        value,
        {"version", "id", "visibility", "runtime", "scenarios"},
        "race manifest",
    )
    if value.get("version") != RACE_MANIFEST_VERSION:
        raise ValueError(f"race manifest version must be {RACE_MANIFEST_VERSION}")

    runtime_value = value.get("runtime")
    if not isinstance(runtime_value, dict):
        raise ValueError("race runtime must be a mapping")
    _reject_unknown(
        runtime_value,
        {"decision_repeat", "physics_world_step_size", "realtime_factor"},
        "race runtime",
    )
    runtime = RaceRuntime(
        decision_repeat=_required(runtime_value, "decision_repeat", "race runtime"),
        physics_world_step_size=_required(
            runtime_value,
            "physics_world_step_size",
            "race runtime",
        ),
        realtime_factor=_required(runtime_value, "realtime_factor", "race runtime"),
    )

    raw_scenarios = value.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ValueError("race scenarios must be a list")
    return RacePlan(
        suite_id=_required(value, "id", "race manifest"),
        visibility=_required(value, "visibility", "race manifest"),
        runtime=runtime,
        scenarios=tuple(_parse_scenario(item) for item in raw_scenarios),
    )


def run_race(
    base_config: AppConfig,
    plan: RacePlan,
    output_dir: Path | str,
    *,
    dry_run: bool = False,
    render: bool = False,
    runner: Callable[..., RunSummary] = run_simulation,
) -> RaceReport:
    if not isinstance(base_config, AppConfig):
        raise TypeError("base_config must be an AppConfig")
    if not isinstance(plan, RacePlan):
        raise TypeError("plan must be a RacePlan")
    if not isinstance(render, bool):
        raise TypeError("render must be a boolean")

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    runs: list[RaceRunResult] = []

    for scenario in plan.scenarios:
        event_log_path = root / scenario.scenario_id / "events.jsonl"
        config = build_race_config(
            base_config,
            plan,
            scenario,
            event_log_path=event_log_path,
            render=render,
        )
        environment_sha256 = race_environment_sha256(
            plan,
            scenario,
            render=render,
        )
        try:
            summary = runner(config, dry_run=dry_run)
            if not dry_run and (
                summary.replay_artifact_path is None
                or summary.replay_artifact_sha256 is None
            ):
                raise RuntimeError("race run did not produce a replay artifact")
        except Exception as exc:
            runs.append(
                RaceRunResult(
                    scenario_id=scenario.scenario_id,
                    status="failed",
                    environment_sha256=environment_sha256,
                    event_log_path=str(event_log_path.resolve()),
                    error_category=type(exc).__name__,
                    error_message=str(exc),
                )
            )
        else:
            runs.append(
                RaceRunResult(
                    scenario_id=scenario.scenario_id,
                    status="completed",
                    environment_sha256=environment_sha256,
                    event_log_path=str(event_log_path.resolve()),
                    replay_artifact_path=summary.replay_artifact_path,
                    replay_artifact_sha256=summary.replay_artifact_sha256,
                    summary=summary,
                )
            )

    summaries = tuple(run.summary for run in runs if run.summary is not None)
    report = RaceReport(
        suite_id=plan.suite_id,
        visibility=plan.visibility,
        rendered=render,
        manifest_sha256=race_manifest_sha256(plan),
        base_config_sha256=_sha256(base_config.to_dict()),
        output_dir=str(root),
        runs=tuple(runs),
        aggregate=aggregate_run_summaries(summaries, total_runs=len(runs)),
    )
    (root / "summary.json").write_text(
        json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_race_config(
    base_config: AppConfig,
    plan: RacePlan,
    scenario: RaceScenario,
    *,
    event_log_path: Path | str,
    render: bool = False,
) -> AppConfig:
    """Apply an immutable race world around a tunable controller/model config."""
    if scenario not in plan.scenarios:
        raise ValueError("race scenario does not belong to plan")
    if not isinstance(render, bool):
        raise TypeError("render must be a boolean")
    config = copy.deepcopy(base_config)
    simulator = config.simulator
    simulator.map = scenario.map
    simulator.traffic_density = scenario.traffic_density
    simulator.random_traffic = False
    simulator.traffic_mode = "trigger"
    simulator.obstacle_probability = scenario.obstacle_probability
    simulator.num_scenarios = 1
    simulator.start_seed = scenario.seed
    simulator.decision_repeat = plan.runtime.decision_repeat
    simulator.physics_world_step_size = plan.runtime.physics_world_step_size
    simulator.horizon = scenario.horizon
    simulator.headless = not render
    simulator.manual_control = False
    simulator.realtime = True
    simulator.realtime_factor = plan.runtime.realtime_factor
    simulator.out_of_road_done = True
    simulator.crash_vehicle_done = True
    simulator.crash_object_done = True

    config.scenario = ScenarioSettings(
        stopped_vehicle_ahead_m=scenario.stopped_vehicle_ahead_m,
    )
    config.faults = ()
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(Path(event_log_path).resolve()),
        scenario_id=scenario.scenario_id,
    )
    config.episode_recording = EpisodeRecordingSettings(
        enabled=True,
        path=str((Path(event_log_path).resolve().parent / "replay")),
    )
    config.__post_init__()
    return config


def race_manifest_sha256(plan: RacePlan) -> str:
    return _sha256({"version": RACE_MANIFEST_VERSION, "plan": plan})


def race_environment_sha256(
    plan: RacePlan,
    scenario: RaceScenario,
    *,
    render: bool = False,
) -> str:
    if scenario not in plan.scenarios:
        raise ValueError("race scenario does not belong to plan")
    if not isinstance(render, bool):
        raise TypeError("render must be a boolean")
    return _sha256(
        {
            "runtime": plan.runtime,
            "world": {
                "map": scenario.map,
                "traffic_density": scenario.traffic_density,
                "random_traffic": False,
                "traffic_mode": "trigger",
                "obstacle_probability": scenario.obstacle_probability,
                "num_scenarios": 1,
                "seed": scenario.seed,
                "horizon": scenario.horizon,
                "headless": not render,
                "manual_control": False,
                "realtime": True,
                "out_of_road_done": True,
                "crash_vehicle_done": True,
                "crash_object_done": True,
                "stopped_vehicle_ahead_m": scenario.stopped_vehicle_ahead_m,
            },
        }
    )


def _parse_scenario(value: object) -> RaceScenario:
    if not isinstance(value, dict):
        raise ValueError("race scenario must be a mapping")
    allowed = {
        "id",
        "seed",
        "map",
        "traffic_density",
        "obstacle_probability",
        "horizon",
        "stopped_vehicle_ahead_m",
    }
    _reject_unknown(value, allowed, "race scenario")
    return RaceScenario(
        scenario_id=_required(value, "id", "race scenario"),
        seed=_required(value, "seed", "race scenario"),
        map=_required(value, "map", "race scenario"),
        traffic_density=_required(value, "traffic_density", "race scenario"),
        obstacle_probability=_required(
            value,
            "obstacle_probability",
            "race scenario",
        ),
        horizon=_required(value, "horizon", "race scenario"),
        stopped_vehicle_ahead_m=value.get("stopped_vehicle_ahead_m"),
    )


def _sha256(value: object) -> str:
    primitive = to_json_value(value)
    encoded = json.dumps(
        primitive,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required(value: Mapping[str, object], field: str, label: str) -> object:
    try:
        return value[field]
    except KeyError as exc:
        raise ValueError(f"{label} is missing field {field!r}") from exc


def _reject_unknown(
    value: Mapping[str, object],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _validate_id(value: object, label: str) -> None:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{label} must contain only letters, numbers, '.', '_', or '-'"
        )


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0.0
    )


def _unit_interval(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )
