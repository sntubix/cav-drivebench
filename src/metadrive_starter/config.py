from __future__ import annotations

import copy
import functools
import math
import re
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from metadrive_starter.faults import FaultSpec
from metadrive_starter.prompt_policy import normalize_prompt_policy
from metadrive_starter.safety import CommandValidationSettings, SafetySettings


@dataclass
class PIDSettings:
    kp: float
    ki: float
    kd: float
    output_min: float = -1.0
    output_max: float = 1.0
    anti_windup: bool = True

    def __post_init__(self) -> None:
        for name in ("kp", "ki", "kd", "output_min", "output_max"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"controller PID {name} must be finite")
        if self.output_min >= self.output_max:
            raise ValueError("controller PID output_min must be below output_max")
        if not isinstance(self.anti_windup, bool):
            raise ValueError("controller PID anti_windup must be a boolean")


# The starting PID gains, and the speed loop's reset threshold, are written
# only in configs/pid.yaml. Every configuration takes them from there, and a
# configuration file that repeats one is refused (#28). The reference tunes its
# own in its agent.yaml, as a team does (ADR-0007).
PID_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "pid.yaml"
_PID_LOOPS = ("speed_pid", "steering_pid", "lateral_pid")
_PID_CONFIG_KEYS = ("speed_pid_reset_threshold_mps", *_PID_LOOPS)


def instructor_pid_values() -> dict[str, Any]:
    """Return the controller values configs/pid.yaml sets, checked."""
    try:
        stamp = PID_CONFIG.stat()
    except FileNotFoundError:
        raise ValueError(
            f"{PID_CONFIG} is missing; it holds the starting PID gains"
        ) from None
    return copy.deepcopy(_read_pid_config(PID_CONFIG, stamp.st_mtime_ns, stamp.st_size))


# Keyed on the file's modification time and size, so an edit takes effect in a
# running notebook without a restart.
@functools.lru_cache(maxsize=4)
def _read_pid_config(path: Path, mtime_ns: int, size: int) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    controller = data.get("controller") if isinstance(data, dict) else None
    if not isinstance(controller, dict) or set(data) != {"controller"}:
        raise ValueError(f"{path} must hold one controller section and nothing else")
    if set(controller) != set(_PID_CONFIG_KEYS):
        raise ValueError(
            f"{path} must set exactly these controller keys: "
            + ", ".join(_PID_CONFIG_KEYS)
        )
    loop_keys = [item.name for item in fields(PIDSettings)]
    for loop in _PID_LOOPS:
        if not isinstance(controller[loop], dict) or set(controller[loop]) != set(loop_keys):
            raise ValueError(
                f"{path}: controller.{loop} must set exactly " + ", ".join(loop_keys)
            )
    return controller


def _instructor_pid(loop: str) -> PIDSettings:
    return PIDSettings(**instructor_pid_values()[loop])


@dataclass
class SimulatorSettings:
    map: str = "XTOC"
    traffic_density: float = 0.1
    random_traffic: bool = False
    traffic_mode: str = "trigger"
    obstacle_probability: float = 0.0
    num_scenarios: int = 1
    start_seed: int = 0
    spawn_longitude_m: float | None = None
    decision_repeat: int = 5
    physics_world_step_size: float = 0.02
    horizon: int = 1000
    use_lidar: bool = True
    headless: bool = True
    manual_control: bool = False
    realtime: bool = False
    realtime_factor: float = 1.0
    out_of_road_done: bool = True
    crash_vehicle_done: bool = True
    crash_object_done: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.random_traffic, bool):
            raise ValueError("simulator.random_traffic must be a boolean")
        if self.spawn_longitude_m is not None and not _non_negative_finite(
            self.spawn_longitude_m
        ):
            raise ValueError(
                "simulator.spawn_longitude_m must be finite and non-negative or null"
            )
        if self.traffic_mode not in {"trigger", "hybrid", "respawn"}:
            raise ValueError(
                "simulator.traffic_mode must be 'trigger', 'hybrid', or 'respawn'"
            )
        if not 0.0 <= self.obstacle_probability <= 1.0:
            raise ValueError("obstacle_probability must be between 0 and 1")
        if not _positive_integer(self.decision_repeat):
            raise ValueError("simulator.decision_repeat must be a positive integer")
        if not _positive_finite(self.physics_world_step_size):
            raise ValueError(
                "simulator.physics_world_step_size must be finite and positive"
            )
        if not isinstance(self.realtime, bool):
            raise ValueError("simulator.realtime must be a boolean")
        if (
            not _positive_finite(self.realtime_factor)
            or self.realtime_factor > 1.0
        ):
            raise ValueError(
                "simulator.realtime_factor must be finite and between 0 and 1"
            )


@dataclass(frozen=True)
class CameraSettings:
    enabled: bool = False
    width: int = 512
    height: int = 288

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("camera.enabled must be a boolean")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("camera.width must be a positive integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("camera.height must be a positive integer")


@dataclass(frozen=True)
class DisplaySettings:
    speed_unit: str = "mps"

    def __post_init__(self) -> None:
        if self.speed_unit not in {"mps", "kph"}:
            raise ValueError("display.speed_unit must be 'mps' or 'kph'")


@dataclass(frozen=True)
class HTTPProviderSettings:
    endpoint_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    model_id: str = "local-vlm"
    api_key_env: str | None = None
    max_response_bytes: int = 1024 * 1024
    max_tokens: int = 256
    allow_insecure_http: bool = False
    structured_output: bool = False
    # Sampling temperature; 0 asks for the most likely answer every time.
    temperature: float = 0.0
    # Fixes the server's sampling, so a run at a temperature above 0 can be
    # reproduced; None leaves the server to choose.
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_url, str) or not self.endpoint_url.strip():
            raise ValueError("vla.http.endpoint_url must not be empty")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("vla.http.model_id must not be empty")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env) is None
        ):
            raise ValueError("vla.http.api_key_env must be a valid environment variable name")
        if not _positive_integer(self.max_response_bytes):
            raise ValueError("vla.http.max_response_bytes must be a positive integer")
        if not _positive_integer(self.max_tokens):
            raise ValueError("vla.http.max_tokens must be a positive integer")
        if not isinstance(self.allow_insecure_http, bool):
            raise ValueError("vla.http.allow_insecure_http must be a boolean")
        if not isinstance(self.structured_output, bool):
            raise ValueError("vla.http.structured_output must be a boolean")
        if not _non_negative_finite(self.temperature) or self.temperature > 2.0:
            raise ValueError("vla.http.temperature must be between 0 and 2")
        if self.seed is not None and not (
            isinstance(self.seed, int) and not isinstance(self.seed, bool) and self.seed >= 0
        ):
            raise ValueError("vla.http.seed must be a non-negative integer or null")


@dataclass(frozen=True)
class VertexProviderSettings:
    project_env: str = "GOOGLE_CLOUD_PROJECT"
    location: str = "global"
    model_id: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    max_output_tokens: int = 256
    thinking_budget: int | None = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.project_env, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.project_env) is None
        ):
            raise ValueError("vla.vertex.project_env must be a valid environment variable name")
        for name, value in {
            "location": self.location,
            "model_id": self.model_id,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"vla.vertex.{name} must not be empty")
        if not _non_negative_finite(self.temperature) or self.temperature > 2.0:
            raise ValueError("vla.vertex.temperature must be between 0 and 2")
        if not _positive_integer(self.max_output_tokens):
            raise ValueError("vla.vertex.max_output_tokens must be a positive integer")
        if self.thinking_budget is not None and not _non_negative_integer(
            self.thinking_budget
        ):
            raise ValueError("vla.vertex.thinking_budget must be a non-negative integer or null")


@dataclass(frozen=True)
class FixtureProviderSettings:
    path: str = "fixtures/vla/providers/synthetic-keep-lane.json"
    repeat_last: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("vla.fixture.path must not be empty")
        if not isinstance(self.repeat_last, bool):
            raise ValueError("vla.fixture.repeat_last must be a boolean")


@dataclass(frozen=True)
class VLASettings:
    enabled: bool = False
    provider: str = "http"
    prompt_policy: str = ""
    maximum_requests_per_run: int | None = None
    request_timeout_s: float = 10.0
    minimum_interval_s: float = 0.5
    action_horizon_s: float = 2.0
    max_scene_objects: int = 8
    maximum_frame_age_s: float = 0.5
    maximum_clock_skew_s: float = 0.05
    action_speed_policy_mode: str = "off"
    http: HTTPProviderSettings = field(default_factory=HTTPProviderSettings)
    vertex: VertexProviderSettings = field(default_factory=VertexProviderSettings)
    fixture: FixtureProviderSettings = field(default_factory=FixtureProviderSettings)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("vla.enabled must be a boolean")
        if self.provider not in {"http", "vertex", "fixture"}:
            raise ValueError("vla.provider must be 'http', 'vertex', or 'fixture'")
        object.__setattr__(
            self,
            "prompt_policy",
            normalize_prompt_policy(self.prompt_policy, label="vla.prompt_policy"),
        )
        if self.action_speed_policy_mode not in {"off", "shadow", "enforce"}:
            raise ValueError(
                "vla.action_speed_policy_mode must be 'off', 'shadow', or 'enforce'"
            )
        if self.maximum_requests_per_run is not None and not _positive_integer(
            self.maximum_requests_per_run
        ):
            raise ValueError(
                "vla.maximum_requests_per_run must be a positive integer or null"
            )
        for name in (
            "request_timeout_s",
            "action_horizon_s",
            "maximum_frame_age_s",
        ):
            if not _positive_finite(getattr(self, name)):
                raise ValueError(f"vla.{name} must be finite and positive")
        for name in ("minimum_interval_s", "maximum_clock_skew_s"):
            if not _non_negative_finite(getattr(self, name)):
                raise ValueError(f"vla.{name} must be finite and non-negative")
        if not _non_negative_integer(self.max_scene_objects):
            raise ValueError("vla.max_scene_objects must be a non-negative integer")
        if not isinstance(self.http, HTTPProviderSettings):
            raise ValueError("vla.http must be HTTP provider settings")
        if not isinstance(self.vertex, VertexProviderSettings):
            raise ValueError("vla.vertex must be Vertex provider settings")
        if not isinstance(self.fixture, FixtureProviderSettings):
            raise ValueError("vla.fixture must be Fixture provider settings")


@dataclass(frozen=True)
class SceneFieldSettings:
    """Which parts of the measured scene an observation's scene context includes."""

    lanes: bool = True
    traffic_controls: bool = True
    objects: bool = True

    def __post_init__(self) -> None:
        for name in ("lanes", "traffic_controls", "objects"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"observation.scene_fields.{name} must be a boolean")


@dataclass(frozen=True)
class ObservationSettings:
    """The values an observation builder assembles each observation from.

    The defaults reproduce DriveBench's original observation.
    """

    # Measured ego speed and the clear-road cruise speed.
    driving_context: bool = True
    # The measured local scene.
    scene_context: bool = True
    scene_fields: SceneFieldSettings = field(default_factory=SceneFieldSettings)

    def __post_init__(self) -> None:
        for name in ("driving_context", "scene_context"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"observation.{name} must be a boolean")
        if not isinstance(self.scene_fields, SceneFieldSettings):
            raise ValueError("observation.scene_fields must be scene field settings")


_TARGET_SPEED_MPS = 9.722222222222221


@dataclass
class ControllerSettings:
    target_speed_mps: float = _TARGET_SPEED_MPS
    # Every other default is read from configs/pid.yaml.
    speed_pid_reset_threshold_mps: float = field(
        default_factory=lambda: instructor_pid_values()["speed_pid_reset_threshold_mps"]
    )
    speed_pid: PIDSettings = field(default_factory=lambda: _instructor_pid("speed_pid"))
    steering_pid: PIDSettings = field(default_factory=lambda: _instructor_pid("steering_pid"))
    lateral_pid: PIDSettings = field(default_factory=lambda: _instructor_pid("lateral_pid"))

    def __post_init__(self) -> None:
        if not _non_negative_finite(self.target_speed_mps):
            raise ValueError("controller.target_speed_mps must be finite and non-negative")
        if not _positive_finite(self.speed_pid_reset_threshold_mps):
            raise ValueError(
                "controller.speed_pid_reset_threshold_mps must be finite and positive"
            )


@dataclass
class PlannerSettings:
    route_source: str = "map"
    lookahead_m: float = 15.0
    waypoint_spacing_m: float = 2.0
    curvature_preview_m: float = 40.0
    maximum_lateral_acceleration_mps2: float = 2.0
    minimum_curve_speed_mps: float = 4.0
    lane_change_enabled: bool = False
    lane_change_transition_m: float = 18.0
    lane_change_timeout_s: float = 8.0
    lane_change_completion_tolerance_m: float = 0.6
    default_route: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)]
    )

    def __post_init__(self) -> None:
        if self.route_source not in {"map", "waypoints"}:
            raise ValueError("planner.route_source must be 'map' or 'waypoints'")
        for name in (
            "lookahead_m",
            "waypoint_spacing_m",
            "curvature_preview_m",
            "maximum_lateral_acceleration_mps2",
        ):
            if not _positive_finite(getattr(self, name)):
                raise ValueError(f"planner.{name} must be finite and positive")
        if not _non_negative_finite(self.minimum_curve_speed_mps):
            raise ValueError(
                "planner.minimum_curve_speed_mps must be finite and non-negative"
            )
        if not isinstance(self.lane_change_enabled, bool):
            raise ValueError("planner.lane_change_enabled must be a boolean")
        for name in (
            "lane_change_transition_m",
            "lane_change_timeout_s",
            "lane_change_completion_tolerance_m",
        ):
            if not _positive_finite(getattr(self, name)):
                raise ValueError(f"planner.{name} must be finite and positive")
        if self.lane_change_enabled and self.route_source != "map":
            raise ValueError("planner lane changes require route_source 'map'")


@dataclass
class PerceptionSettings:
    prefer_info: bool = True
    safety_source: str = "oracle"
    detection_radius_m: float = 50.0
    corridor_margin_m: float = 0.5

    def __post_init__(self) -> None:
        if self.safety_source not in {"oracle", "lidar"}:
            raise ValueError("perception.safety_source must be 'oracle' or 'lidar'")
        if self.detection_radius_m <= 0.0:
            raise ValueError("perception.detection_radius_m must be positive")
        if self.corridor_margin_m < 0.0:
            raise ValueError("perception.corridor_margin_m must not be negative")


@dataclass
class TrafficLightScenarioSettings:
    enabled: bool = False
    distance_m: float = 38.0
    visual_scale: float = 1.75
    initial_state: str = "red"
    red_duration_s: float = 12.0
    green_duration_s: float = 10.0
    yellow_duration_s: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("scenario.traffic_light.enabled must be a boolean")
        for name, value in {
            "distance_m": self.distance_m,
            "visual_scale": self.visual_scale,
            "red_duration_s": self.red_duration_s,
            "green_duration_s": self.green_duration_s,
            "yellow_duration_s": self.yellow_duration_s,
        }.items():
            if not _positive_finite(value):
                raise ValueError(f"scenario.traffic_light.{name} must be finite and positive")
        if self.initial_state not in {"red", "green", "yellow"}:
            raise ValueError(
                "scenario.traffic_light.initial_state must be 'red', 'green', or 'yellow'"
            )


@dataclass
class LaneChangeHazardScenarioSettings:
    """Deterministic target-lane obstacle injected after a lane request latches."""

    enabled: bool = False
    distance_ahead_m: float = 6.0
    vehicle_kind: str = "car"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("scenario.lane_change_hazard.enabled must be a boolean")
        if not _positive_finite(self.distance_ahead_m):
            raise ValueError(
                "scenario.lane_change_hazard.distance_ahead_m must be finite and positive"
            )
        if self.vehicle_kind not in {"car", "truck"}:
            raise ValueError(
                "scenario.lane_change_hazard.vehicle_kind must be 'car' or 'truck'"
            )


@dataclass(frozen=True)
class ScenarioVehicleSettings:
    """One deterministic actor placed relative to the ego on its current road."""

    vehicle_id: str
    longitudinal_offset_m: float
    lane_offset: int = 0
    speed_mps: float = 0.0
    kind: str = "car"

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*",
            self.vehicle_id,
        ) is None:
            raise ValueError(
                "scenario vehicle id must contain only letters, numbers, '.', '_', or '-'"
            )
        if not _finite(self.longitudinal_offset_m):
            raise ValueError(
                "scenario vehicle longitudinal_offset_m must be finite"
            )
        if isinstance(self.lane_offset, bool) or not isinstance(self.lane_offset, int):
            raise ValueError("scenario vehicle lane_offset must be an integer")
        if not _non_negative_finite(self.speed_mps):
            raise ValueError("scenario vehicle speed_mps must be finite and non-negative")
        if self.kind not in {"car", "truck"}:
            raise ValueError("scenario vehicle kind must be 'car' or 'truck'")


@dataclass(frozen=True)
class StoppedVehicleSettings:
    """One static car or truck placed along the ego route, as a probe scene places it."""

    vehicle_id: str
    distance_m: float
    lane_offset: int = 0
    # Offset from the lane centre, positive to the left.
    lateral_m: float = 0.0
    kind: str = "car"

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*",
            self.vehicle_id,
        ) is None:
            raise ValueError(
                "scenario stopped vehicle_id must contain only letters, numbers, "
                "'.', '_', or '-'"
            )
        if not _finite(self.distance_m) or self.distance_m <= 0.0:
            raise ValueError("scenario stopped vehicle distance_m must be finite and positive")
        if isinstance(self.lane_offset, bool) or not isinstance(self.lane_offset, int):
            raise ValueError("scenario stopped vehicle lane_offset must be an integer")
        if not _finite(self.lateral_m):
            raise ValueError("scenario stopped vehicle lateral_m must be finite")
        if self.kind not in {"car", "truck"}:
            raise ValueError("scenario stopped vehicle kind must be 'car' or 'truck'")


@dataclass
class ScenarioSettings:
    stopped_vehicle_ahead_m: float | None = None
    traffic_light: TrafficLightScenarioSettings = field(
        default_factory=TrafficLightScenarioSettings
    )
    lane_change_hazard: LaneChangeHazardScenarioSettings = field(
        default_factory=LaneChangeHazardScenarioSettings
    )
    vehicles: tuple[ScenarioVehicleSettings, ...] = ()
    # Static vehicles along the route, beyond the short road the ego starts on.
    stopped_vehicles: tuple[StoppedVehicleSettings, ...] = ()

    def __post_init__(self) -> None:
        if self.stopped_vehicle_ahead_m is not None and self.stopped_vehicle_ahead_m <= 0.0:
            raise ValueError("scenario.stopped_vehicle_ahead_m must be positive")
        if not isinstance(self.traffic_light, TrafficLightScenarioSettings):
            raise ValueError("scenario.traffic_light must be traffic-light settings")
        if not isinstance(
            self.lane_change_hazard,
            LaneChangeHazardScenarioSettings,
        ):
            raise ValueError(
                "scenario.lane_change_hazard must be lane-change hazard settings"
            )
        if not isinstance(self.vehicles, tuple) or any(
            not isinstance(vehicle, ScenarioVehicleSettings)
            for vehicle in self.vehicles
        ):
            raise ValueError(
                "scenario.vehicles must be a tuple of scenario vehicle settings"
            )
        if not isinstance(self.stopped_vehicles, tuple) or any(
            not isinstance(vehicle, StoppedVehicleSettings)
            for vehicle in self.stopped_vehicles
        ):
            raise ValueError(
                "scenario.stopped_vehicles must be a tuple of stopped vehicle settings"
            )
        vehicle_ids = [
            vehicle.vehicle_id for vehicle in (*self.vehicles, *self.stopped_vehicles)
        ]
        if len(set(vehicle_ids)) != len(vehicle_ids):
            raise ValueError("scenario vehicle ids must be unique")


@dataclass(frozen=True)
class EventLogSettings:
    enabled: bool = False
    path: str = "tmp/events.jsonl"
    scenario_id: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("event_log.enabled must be a boolean")
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("event_log.path must not be empty")
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("event_log.scenario_id must not be empty")


@dataclass(frozen=True)
class EpisodeRecordingSettings:
    enabled: bool = False
    path: str = "tmp/episode-replay"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("episode_recording.enabled must be a boolean")
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("episode_recording.path must not be empty")


@dataclass
class AppConfig:
    simulator: SimulatorSettings = field(default_factory=SimulatorSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    vla: VLASettings = field(default_factory=VLASettings)
    observation: ObservationSettings = field(default_factory=ObservationSettings)
    controller: ControllerSettings = field(default_factory=ControllerSettings)
    planner: PlannerSettings = field(default_factory=PlannerSettings)
    perception: PerceptionSettings = field(default_factory=PerceptionSettings)
    safety: SafetySettings = field(default_factory=SafetySettings)
    command_validation: CommandValidationSettings = field(
        default_factory=CommandValidationSettings
    )
    scenario: ScenarioSettings = field(default_factory=ScenarioSettings)
    faults: tuple[FaultSpec, ...] = ()
    event_log: EventLogSettings = field(default_factory=EventLogSettings)
    episode_recording: EpisodeRecordingSettings = field(
        default_factory=EpisodeRecordingSettings
    )

    def __post_init__(self) -> None:
        if not isinstance(self.faults, tuple) or any(
            not isinstance(fault, FaultSpec) for fault in self.faults
        ):
            raise ValueError("faults must be a tuple of FaultSpec values")
        fault_ids = [fault.fault_id for fault in self.faults]
        if len(set(fault_ids)) != len(fault_ids):
            raise ValueError("fault_id values must be unique")
        if self.perception.safety_source == "lidar" and not self.simulator.use_lidar:
            raise ValueError("simulator.use_lidar must be true when safety_source is 'lidar'")
        if (
            self.scenario.lane_change_hazard.enabled
            and not self.planner.lane_change_enabled
        ):
            raise ValueError(
                "scenario.lane_change_hazard requires planner.lane_change_enabled"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path | str = "configs/default.yaml") -> AppConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    _refuse_pid_values(data or {}, config_path)
    return config_from_dict(data or {})


def _refuse_pid_values(data: object, path: Path) -> None:
    """Refuse a configuration file that repeats a value configs/pid.yaml owns."""
    controller = data.get("controller") if isinstance(data, dict) else None
    if not isinstance(controller, dict):
        return
    repeated = [f"controller.{key}" for key in _PID_CONFIG_KEYS if key in controller]
    if repeated:
        raise ValueError(
            f"{path} sets {', '.join(repeated)}; PID gains are written only in "
            f"{PID_CONFIG.name}, so remove them here and change {PID_CONFIG.name} "
            "or a submission's agent.yaml instead"
        )


def config_from_dict(data: dict[str, Any]) -> AppConfig:
    simulator = SimulatorSettings(**data.get("simulator", {}))
    camera = CameraSettings(**data.get("camera", {}))
    display = DisplaySettings(**data.get("display", {}))
    vla_data = data.get("vla", {})
    http = HTTPProviderSettings(**vla_data.get("http", {}))
    vertex = VertexProviderSettings(**vla_data.get("vertex", {}))
    fixture = FixtureProviderSettings(**vla_data.get("fixture", {}))
    vla = VLASettings(
        **{
            key: value
            for key, value in vla_data.items()
            if key not in {"http", "vertex", "fixture"}
        },
        http=http,
        vertex=vertex,
        fixture=fixture,
    )
    observation_data = dict(data.get("observation", {}))
    scene_fields = SceneFieldSettings(**observation_data.pop("scene_fields", {}))
    observation = ObservationSettings(**observation_data, scene_fields=scene_fields)

    controller_data = data.get("controller", {})
    instructor = instructor_pid_values()
    controller = ControllerSettings(
        target_speed_mps=_speed_mps(controller_data, "target_speed", default=_TARGET_SPEED_MPS),
        speed_pid_reset_threshold_mps=_speed_mps(
            controller_data,
            "speed_pid_reset_threshold",
            default=instructor["speed_pid_reset_threshold_mps"],
        ),
        **{
            loop: _pid_settings(controller_data.get(loop, {}), PIDSettings(**instructor[loop]))
            for loop in _PID_LOOPS
        },
    )

    planner = PlannerSettings(**_planner_values(data.get("planner", {})))

    perception = PerceptionSettings(**data.get("perception", {}))
    safety = SafetySettings(**data.get("safety", {}))
    command_validation_data = dict(data.get("command_validation", {}))
    command_validation_defaults = CommandValidationSettings()
    normalized_speed_settings = {
        field_name: _speed_mps(
            command_validation_data,
            field_name.removesuffix("_mps"),
            default=getattr(command_validation_defaults, field_name),
        )
        for field_name in (
            "maximum_target_speed_mps",
            "slow_down_speed_mps",
            "yield_speed_mps",
        )
    }
    for stem in ("maximum_target_speed", "slow_down_speed", "yield_speed"):
        command_validation_data.pop(f"{stem}_mps", None)
    command_validation = CommandValidationSettings(
        **command_validation_data,
        **normalized_speed_settings,
    )
    scenario_data = dict(data.get("scenario", {}))
    traffic_light = TrafficLightScenarioSettings(
        **scenario_data.pop("traffic_light", {})
    )
    lane_change_hazard = LaneChangeHazardScenarioSettings(
        **scenario_data.pop("lane_change_hazard", {})
    )
    raw_scenario_vehicles = scenario_data.pop("vehicles", [])
    if not isinstance(raw_scenario_vehicles, list):
        raise ValueError("scenario.vehicles must be a list")
    scenario_vehicles = tuple(
        _scenario_vehicle_settings(raw_vehicle)
        for raw_vehicle in raw_scenario_vehicles
    )
    raw_stopped_vehicles = scenario_data.pop("stopped_vehicles", [])
    if not isinstance(raw_stopped_vehicles, list) or any(
        not isinstance(raw_vehicle, dict) for raw_vehicle in raw_stopped_vehicles
    ):
        raise ValueError("scenario.stopped_vehicles must be a list of mappings")
    scenario = ScenarioSettings(
        **scenario_data,
        traffic_light=traffic_light,
        lane_change_hazard=lane_change_hazard,
        vehicles=scenario_vehicles,
        stopped_vehicles=tuple(
            StoppedVehicleSettings(**raw_vehicle) for raw_vehicle in raw_stopped_vehicles
        ),
    )
    raw_faults = data.get("faults", [])
    if not isinstance(raw_faults, list):
        raise ValueError("faults must be a list")
    if any(not isinstance(fault, dict) for fault in raw_faults):
        raise ValueError("faults entries must be mappings")
    faults = tuple(FaultSpec(**fault) for fault in raw_faults)
    event_log = EventLogSettings(**data.get("event_log", {}))
    episode_recording = EpisodeRecordingSettings(
        **data.get("episode_recording", {})
    )
    return AppConfig(
        simulator=simulator,
        camera=camera,
        display=display,
        vla=vla,
        observation=observation,
        controller=controller,
        planner=planner,
        perception=perception,
        safety=safety,
        command_validation=command_validation,
        scenario=scenario,
        faults=faults,
        event_log=event_log,
        episode_recording=episode_recording,
    )


def _pid_settings(data: dict[str, Any], defaults: PIDSettings) -> PIDSettings:
    merged = asdict(defaults)
    merged.update(data)
    return PIDSettings(**merged)


_PLANNER_FLOATS = (
    "lookahead_m",
    "waypoint_spacing_m",
    "curvature_preview_m",
    "maximum_lateral_acceleration_mps2",
    "minimum_curve_speed_mps",
    "lane_change_transition_m",
    "lane_change_timeout_s",
    "lane_change_completion_tolerance_m",
)


def _planner_values(data: dict[str, Any]) -> dict[str, Any]:
    """Return the planner values a configuration sets, typed as the parser always has.

    Only keys present are passed on, so PlannerSettings holds every default. A key
    PlannerSettings does not know is ignored, as it always was.
    """
    values: dict[str, Any] = {name: float(data[name]) for name in _PLANNER_FLOATS if name in data}
    if "route_source" in data:
        values["route_source"] = str(data["route_source"])
    if "lane_change_enabled" in data:
        values["lane_change_enabled"] = data["lane_change_enabled"]
    if "default_route" in data:
        values["default_route"] = [
            (float(point[0]), float(point[1])) for point in data["default_route"]
        ]
    return values


def _speed_mps(
    data: dict[str, Any],
    stem: str,
    *,
    default: float,
) -> float:
    mps_key = f"{stem}_mps"
    kmh_key = f"{stem}_kmh"
    if kmh_key in data:
        raise ValueError(f"{kmh_key} is not supported; use {mps_key}")
    if mps_key not in data:
        return default
    raw_value = data[mps_key]
    if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
        raise ValueError(f"{mps_key} must be numeric")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"{mps_key} must be finite")
    return value


def _positive_finite(value: object) -> bool:
    return _non_negative_finite(value) and value > 0.0  # type: ignore[operator]


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def _scenario_vehicle_settings(value: object) -> ScenarioVehicleSettings:
    if not isinstance(value, dict):
        raise ValueError("scenario.vehicles entries must be mappings")
    # YAML spells the id "id"; AppConfig.to_dict(), and so every run_started
    # event that replay parses, spells it "vehicle_id". Accept either.
    if "vehicle_id" in value:
        if "id" in value:
            raise ValueError(
                "scenario.vehicles entry must give only one of id and vehicle_id"
            )
        value = dict(value)
        value["id"] = value.pop("vehicle_id")
    allowed = {"id", "longitudinal_offset_m", "lane_offset", "speed_mps", "kind"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            "scenario.vehicles entry has unknown fields: " + ", ".join(unknown)
        )
    missing = [name for name in ("id", "longitudinal_offset_m") if name not in value]
    if missing:
        raise ValueError(
            "scenario.vehicles entry is missing fields: " + ", ".join(missing)
        )
    try:
        return ScenarioVehicleSettings(
            vehicle_id=value["id"],
            longitudinal_offset_m=value["longitudinal_offset_m"],
            lane_offset=value.get("lane_offset", 0),
            speed_mps=value.get("speed_mps", 0.0),
            kind=value.get("kind", "car"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scenario.vehicles entry is invalid: {exc}") from exc


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def apply_overrides(
    config: AppConfig,
    *,
    headless: bool | None = None,
    manual_control: bool | None = None,
    realtime: bool | None = None,
    realtime_factor: float | None = None,
    out_of_road_done: bool | None = None,
    crash_vehicle_done: bool | None = None,
    crash_object_done: bool | None = None,
    steps: int | None = None,
    traffic_density: float | None = None,
    obstacle_probability: float | None = None,
    map_name: str | None = None,
    event_log_path: Path | str | None = None,
    display_speed_unit: str | None = None,
    headway_speed_cap_mode: str | None = None,
    action_speed_policy_mode: str | None = None,
    http_seed: int | None = None,
) -> AppConfig:
    if headless is not None:
        config.simulator.headless = headless
    if manual_control is not None:
        config.simulator.manual_control = manual_control
    if realtime is not None:
        config.simulator.realtime = realtime
    if realtime_factor is not None:
        if not _positive_finite(realtime_factor) or realtime_factor > 1.0:
            raise ValueError("realtime_factor must be finite and between 0 and 1")
        config.simulator.realtime_factor = float(realtime_factor)
    if out_of_road_done is not None:
        config.simulator.out_of_road_done = out_of_road_done
    if crash_vehicle_done is not None:
        config.simulator.crash_vehicle_done = crash_vehicle_done
    if crash_object_done is not None:
        config.simulator.crash_object_done = crash_object_done
    if steps is not None:
        config.simulator.horizon = steps
    if traffic_density is not None:
        config.simulator.traffic_density = traffic_density
    if obstacle_probability is not None:
        if not 0.0 <= obstacle_probability <= 1.0:
            raise ValueError("obstacle_probability must be between 0 and 1")
        config.simulator.obstacle_probability = obstacle_probability
    if map_name is not None:
        config.simulator.map = map_name
    if event_log_path is not None:
        path = str(event_log_path)
        if not path.strip():
            raise ValueError("event log path must not be empty")
        config.event_log = EventLogSettings(
            enabled=True,
            path=path,
            scenario_id=config.event_log.scenario_id,
        )
    if display_speed_unit is not None:
        config.display = DisplaySettings(speed_unit=display_speed_unit)
    if headway_speed_cap_mode is not None:
        config.safety = replace(
            config.safety,
            headway_speed_cap_mode=headway_speed_cap_mode,
        )
    if action_speed_policy_mode is not None:
        config.vla = replace(
            config.vla,
            action_speed_policy_mode=action_speed_policy_mode,
        )
    if http_seed is not None:
        config.vla = replace(config.vla, http=replace(config.vla.http, seed=http_seed))
    return config
