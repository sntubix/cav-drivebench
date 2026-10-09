from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import yaml

from metadrive_starter.config import AppConfig, ObservationSettings
from metadrive_starter.env import (
    make_env,
    navigation_waypoints,
    simulation_time_s,
    spawn_static_vehicle_ahead,
    spawn_traffic_light_group_ahead,
)
from metadrive_starter.perception import LidarSceneAdapter, LocalScene, OracleSceneAdapter
from metadrive_starter.planning import PolylineFuturePath
from metadrive_starter.timing import times_equal
from metadrive_starter.traffic_lights import TrafficLightState, set_traffic_light_state
from metadrive_starter.vla import (
    OUTPUT_CONTRACT,
    DefaultObservationBuilder,
    HazardType,
    HighLevelAction,
    MetaDriveCameraAdapter,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    Observation,
    ObservationBuilder,
    ObservationRequest,
    RGBFrame,
    RelativeLocation,
    VLACommand,
    VLAAssessment,
    VLAInferencePipeline,
    VLA_PROMPT_CONTRACT_VERSION,
    build_vla_scene_context,
    checked_observation,
    decode_vla_assessment,
    encode_rgb_frame_png,
)


_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_MANIFEST_SCHEMA_VERSION = 2
_SUPPORTED_MANIFEST_SCHEMA_VERSIONS = {1, _MANIFEST_SCHEMA_VERSION}
# Version 4 saves whatever observation was sent, so its prompt may hold the scene
# context anywhere or not at all; versions 2 and 3 saved DriveBench's own layout.
_ARTIFACT_SCHEMA_VERSION = 4
_SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = {2, 3, _ARTIFACT_SCHEMA_VERSION}
_PROBE_DIMENSIONS = {
    "semantic": "semantic_passed",
    "perception": "perception_passed",
    "tactical_action": "tactical_action_passed",
    "raw_speed": "raw_speed_passed",
}


@dataclass(frozen=True)
class ProbeVehicleSpec:
    object_id: str
    distance_m: float
    lane_offset: int = 0
    lateral_m: float = 0.0
    kind: str = "car"

    def __post_init__(self) -> None:
        _validate_identifier(self.object_id, "probe vehicle id")
        if not _finite_number(self.distance_m) or self.distance_m <= 0.0:
            raise ValueError("probe vehicle distance_m must be finite and positive")
        if isinstance(self.lane_offset, bool) or not isinstance(self.lane_offset, int):
            raise ValueError("probe vehicle lane_offset must be an integer")
        if not _finite_number(self.lateral_m):
            raise ValueError("probe vehicle lateral_m must be finite")
        if self.kind not in {"car", "truck"}:
            raise ValueError("probe vehicle kind must be 'car' or 'truck'")


@dataclass(frozen=True)
class ProbeTrafficLightSpec:
    distance_m: float
    state: TrafficLightState
    visual_scale: float = 1.75

    def __post_init__(self) -> None:
        if not _finite_number(self.distance_m) or self.distance_m <= 0.0:
            raise ValueError("probe traffic light distance_m must be finite and positive")
        if (
            not isinstance(self.state, TrafficLightState)
            or self.state is TrafficLightState.UNKNOWN
        ):
            raise ValueError("probe traffic light state must be red, green, or yellow")
        if not _finite_number(self.visual_scale) or self.visual_scale <= 0.0:
            raise ValueError("probe traffic light visual_scale must be finite and positive")


@dataclass(frozen=True)
class ProbeHazardExpectation:
    hazard_type: HazardType
    relative_location: RelativeLocation

    def __post_init__(self) -> None:
        if not isinstance(self.hazard_type, HazardType):
            raise ValueError("probe hazard expectation type must be a VLA hazard type")
        if not isinstance(self.relative_location, RelativeLocation):
            raise ValueError(
                "probe hazard expectation relative_location must be a VLA relative location"
            )


@dataclass(frozen=True)
class ProbeEvaluationSpec:
    acceptable_actions: tuple[HighLevelAction, ...]
    minimum_target_speed_mps: float
    maximum_target_speed_mps: float
    required_hazards: tuple[ProbeHazardExpectation, ...] = ()
    raw_speed_required: bool = True

    def __post_init__(self) -> None:
        if not self.acceptable_actions or not all(
            isinstance(action, HighLevelAction) for action in self.acceptable_actions
        ):
            raise ValueError("probe evaluation acceptable_actions must contain VLA actions")
        if len(self.acceptable_actions) != len(set(self.acceptable_actions)):
            raise ValueError("probe evaluation acceptable_actions must be unique")
        for name, value in {
            "minimum_target_speed_mps": self.minimum_target_speed_mps,
            "maximum_target_speed_mps": self.maximum_target_speed_mps,
        }.items():
            if not _finite_number(value) or value < 0.0:
                raise ValueError(f"probe evaluation {name} must be finite and non-negative")
        if self.minimum_target_speed_mps > self.maximum_target_speed_mps:
            raise ValueError("probe evaluation target speed range must not be inverted")
        if not isinstance(self.required_hazards, tuple) or not all(
            isinstance(hazard, ProbeHazardExpectation) for hazard in self.required_hazards
        ):
            raise ValueError(
                "probe evaluation required_hazards must contain hazard expectations"
            )
        if len(self.required_hazards) != len(set(self.required_hazards)):
            raise ValueError("probe evaluation required_hazards must be unique")
        if not isinstance(self.raw_speed_required, bool):
            raise ValueError("probe evaluation raw_speed_required must be a boolean")


@dataclass(frozen=True)
class ProbeScenario:
    scenario_id: str
    description: str
    expected_visual: str
    map_name: str
    seed: int = 0
    settle_steps: int = 1
    traffic_density: float = 0.0
    vehicles: tuple[ProbeVehicleSpec, ...] = ()
    traffic_light: ProbeTrafficLightSpec | None = None
    evaluation: ProbeEvaluationSpec | None = None
    benchmark_version: int = _MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_identifier(self.scenario_id, "probe scenario id")
        for name, value in {
            "description": self.description,
            "expected_visual": self.expected_visual,
            "map": self.map_name,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"probe scenario {name} must not be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("probe scenario seed must be a non-negative integer")
        if (
            isinstance(self.settle_steps, bool)
            or not isinstance(self.settle_steps, int)
            or self.settle_steps < 0
        ):
            raise ValueError("probe scenario settle_steps must be a non-negative integer")
        if not _finite_number(self.traffic_density) or not 0.0 <= self.traffic_density <= 1.0:
            raise ValueError("probe scenario traffic_density must be between 0 and 1")
        if not isinstance(self.vehicles, tuple) or not all(
            isinstance(vehicle, ProbeVehicleSpec) for vehicle in self.vehicles
        ):
            raise ValueError("probe scenario vehicles must contain ProbeVehicleSpec values")
        if self.evaluation is not None and not isinstance(
            self.evaluation, ProbeEvaluationSpec
        ):
            raise ValueError("probe scenario evaluation must be ProbeEvaluationSpec or None")
        if self.traffic_light is not None and not isinstance(
            self.traffic_light, ProbeTrafficLightSpec
        ):
            raise ValueError("probe scenario traffic_light must be ProbeTrafficLightSpec or None")
        if self.benchmark_version not in _SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError("probe scenario benchmark_version must be 1 or 2")
        object_ids = [vehicle.object_id for vehicle in self.vehicles]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(f"probe scenario {self.scenario_id!r} has duplicate vehicle ids")


@dataclass(frozen=True)
class ProbeCapture:
    frame: RGBFrame
    timestamp_s: float
    ego_speed_mps: float = 0.0
    scene: LocalScene | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame, RGBFrame):
            raise ValueError("probe capture frame must be an RGBFrame")
        if not _finite_number(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("probe capture timestamp_s must be finite and non-negative")
        if self.frame.timestamp_s != self.timestamp_s:
            raise ValueError("probe capture timestamp must match the frame timestamp")
        if not _finite_number(self.ego_speed_mps) or self.ego_speed_mps < 0.0:
            raise ValueError("probe capture ego_speed_mps must be finite and non-negative")
        if self.scene is not None:
            if not isinstance(self.scene, LocalScene):
                raise ValueError("probe capture scene must be a LocalScene or None")
            if not times_equal(self.scene.timestamp_s, self.timestamp_s):
                raise ValueError("probe capture scene timestamp must match the frame timestamp")


@dataclass(frozen=True)
class ProbeReplayInput:
    """Integrity-checked prompt and image bytes ready for exact replay."""

    scenario: ProbeScenario
    source_dir: Path
    source_metadata: dict[str, object]
    request: ModelRequest
    png_bytes: bytes
    scene_bytes: bytes | None = None


@dataclass(frozen=True)
class ProbeScenarioResult:
    scenario_id: str
    status: str
    artifact_dir: str
    error_category: str | None = None
    semantic_passed: bool | None = None
    perception_passed: bool | None = None
    tactical_action_passed: bool | None = None
    raw_speed_passed: bool | None = None


@dataclass(frozen=True)
class ProbeRunSummary:
    output_dir: str
    inference_attempted: bool
    scenarios: tuple[ProbeScenarioResult, ...]

    @property
    def successful(self) -> bool:
        return all(result.status in {"captured", "success"} for result in self.scenarios)

    @property
    def semantically_successful(self) -> bool | None:
        if not self.inference_attempted:
            return None
        if all(result.semantic_passed is None for result in self.scenarios):
            return None
        return all(result.semantic_passed is True for result in self.scenarios)

    @property
    def perception_successful(self) -> bool | None:
        return _dimension_success(self.scenarios, "perception_passed")

    @property
    def tactically_successful(self) -> bool | None:
        return _dimension_success(self.scenarios, "tactical_action_passed")

    @property
    def raw_speed_successful(self) -> bool | None:
        return _dimension_success(self.scenarios, "raw_speed_passed")

    @property
    def scores(self) -> dict[str, dict[str, int]]:
        """Per dimension, the scenes that passed out of those it scored."""
        scores: dict[str, dict[str, int]] = {}
        for name, field_name in _PROBE_DIMENSIONS.items():
            scored = [
                value
                for result in self.scenarios
                if (value := getattr(result, field_name)) is not None
            ]
            scores[name] = {"passed": sum(scored), "scored": len(scored)}
        return scores

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "inference_attempted": self.inference_attempted,
            "successful": self.successful,
            "semantically_successful": self.semantically_successful,
            "perception_successful": self.perception_successful,
            "tactically_successful": self.tactically_successful,
            "raw_speed_successful": self.raw_speed_successful,
            "scores": self.scores,
            "scenarios": [asdict(result) for result in self.scenarios],
        }


CaptureFunction = Callable[[AppConfig, ProbeScenario], ProbeCapture]


def load_probe_scenarios(path: Path | str) -> tuple[ProbeScenario, ...]:
    manifest_path = Path(path)
    data = yaml.safe_load(manifest_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("probe manifest must be a YAML mapping")
    _reject_unknown_keys(data, {"version", "scenarios"}, "probe manifest")
    manifest_version = data.get("version")
    if (
        not isinstance(manifest_version, int)
        or isinstance(manifest_version, bool)
        or manifest_version not in _SUPPORTED_MANIFEST_SCHEMA_VERSIONS
    ):
        raise ValueError("probe manifest version must be 1 or 2")
    entries = data.get("scenarios")
    if not isinstance(entries, list) or not entries:
        raise ValueError("probe manifest scenarios must be a non-empty list")

    assert isinstance(manifest_version, int)
    scenarios = tuple(
        _parse_scenario(entry, benchmark_version=manifest_version) for entry in entries
    )
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("probe manifest scenario ids must be unique")
    return scenarios


def run_vla_probe_catalog(
    config: AppConfig,
    scenarios: Sequence[ProbeScenario],
    output_dir: Path | str,
    *,
    scenario_ids: Sequence[str] | None = None,
    infer: bool = False,
    provider: ModelProvider | None = None,
    capture: CaptureFunction | None = None,
    observation_builder: ObservationBuilder | None = None,
) -> ProbeRunSummary:
    """Capture deterministic probe inputs and optionally perform open-loop inference.

    Each scene's observation is built by ``observation_builder``, a team's
    observation.py, and saved exactly as sent. Without one, DriveBench builds its
    original observation, which reads no observation settings, so changed ones
    are refused rather than ignored.
    """
    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if observation_builder is None:
        if config.observation != ObservationSettings():
            raise ValueError(
                "observation settings are read by a submitted observation.py; "
                "probe with --submission, or leave the observation section at its defaults"
            )
        observation_builder = DefaultObservationBuilder()
    elif not isinstance(observation_builder, ObservationBuilder):
        raise TypeError("observation_builder must implement ObservationBuilder.build()")
    if not isinstance(infer, bool):
        raise ValueError("infer must be a boolean")
    if infer and provider is None:
        raise ValueError("provider is required when infer is true")
    if not infer and provider is not None:
        raise ValueError("provider must be omitted in capture-only mode")

    selected = _select_scenarios(scenarios, scenario_ids)
    root = Path(output_dir).resolve()
    targets = [root / scenario.scenario_id for scenario in selected]
    existing = [target for target in targets if target.exists()]
    if existing:
        raise FileExistsError(f"probe artifact directory already exists: {existing[0]}")
    root.mkdir(parents=True, exist_ok=True)

    capture_scenario = capture or capture_probe_scenario
    results: list[ProbeScenarioResult] = []
    for scenario, target in zip(selected, targets):
        target.mkdir()
        try:
            probe_capture = capture_scenario(config, scenario)
            result = _write_probe_artifacts(
                config,
                scenario,
                probe_capture,
                target,
                infer=infer,
                provider=provider,
                observation_builder=observation_builder,
            )
        except Exception as exc:
            metadata = _base_metadata(scenario, infer=infer)
            metadata["outcome"] = {
                "status": "failure",
                "error_category": type(exc).__name__,
                "error_message": str(exc),
            }
            _write_json(target / "metadata.json", metadata)
            result = ProbeScenarioResult(
                scenario_id=scenario.scenario_id,
                status="failure",
                artifact_dir=str(target),
                error_category=type(exc).__name__,
                semantic_passed=False if infer and scenario.evaluation is not None else None,
            )
        results.append(result)

    summary = ProbeRunSummary(
        output_dir=str(root),
        inference_attempted=infer,
        scenarios=tuple(results),
    )
    _write_json(root / "summary.json", summary.to_dict())
    return summary


def replay_vla_probe_artifacts(
    config: AppConfig,
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    provider: ModelProvider,
    scenario_ids: Sequence[str] | None = None,
    prepared_inputs: Sequence[ProbeReplayInput] | None = None,
) -> ProbeRunSummary:
    """Run inference against saved prompt/PNG bytes without rerendering MetaDrive."""
    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not isinstance(provider, ModelProvider):
        raise TypeError("provider must implement ModelProvider.generate()")
    source_root = Path(source_dir).resolve()
    available, selected_ids = _probe_artifact_selection(source_root, scenario_ids)
    prepared_by_id: dict[str, ProbeReplayInput] | None = None
    if prepared_inputs is not None:
        prepared_by_id = {
            item.scenario.scenario_id: item for item in prepared_inputs
        }
        if len(prepared_by_id) != len(prepared_inputs) or tuple(
            prepared_by_id
        ) != selected_ids:
            raise ValueError("prepared probe inputs do not match selected scenarios")
        if any(
            item.source_dir != available[scenario_id]
            for scenario_id, item in prepared_by_id.items()
        ):
            raise ValueError("prepared probe inputs do not match the source directory")
    root = Path(output_dir).resolve()
    targets = [root / scenario_id for scenario_id in selected_ids]
    existing = [target for target in targets if target.exists()]
    if existing:
        raise FileExistsError(f"probe artifact directory already exists: {existing[0]}")
    root.mkdir(parents=True, exist_ok=True)

    results: list[ProbeScenarioResult] = []
    for scenario_id, target in zip(selected_ids, targets):
        target.mkdir()
        source = available[scenario_id]
        recorder = _RecordingModelProvider(provider)
        semantic_passed: bool | None = None
        evaluation_defined = False
        metadata: dict[str, object] = {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "mode": "artifact_replay",
            "scenario": {"id": scenario_id},
            "replay": {"source_artifact_dir": str(source)},
        }
        try:
            replay_input = (
                prepared_by_id[scenario_id]
                if prepared_by_id is not None
                else _load_probe_replay_input(
                    source,
                    scenario_id,
                    require_prompt_hash=False,
                )
            )
            source_metadata = replay_input.source_metadata
            scenario = replay_input.scenario
            evaluation_defined = scenario.evaluation is not None
            request = replay_input.request
            png_bytes = replay_input.png_bytes
            scene_bytes = replay_input.scene_bytes
            prompt = request.prompt
            metadata = dict(source_metadata)
            metadata["mode"] = "artifact_replay"
            metadata["replay"] = {
                "source_artifact_dir": str(source),
                "exact_prompt": True,
                "exact_png_bytes": True,
                "exact_scene_context": scene_bytes is not None,
            }

            (target / "frame.png").write_bytes(png_bytes)
            (target / "prompt.txt").write_text(prompt, encoding="utf-8")
            if scene_bytes is not None:
                (target / "scene.json").write_bytes(scene_bytes)
            response = recorder.generate(request, timeout_s=config.vla.request_timeout_s)
            (target / "raw_response.txt").write_text(response.text, encoding="utf-8")
            assessment = decode_vla_assessment(response.text)
            command = assessment.to_command(
                command_id=request.request_id,
                issued_at_s=request.created_at_s,
                action_horizon_s=config.vla.action_horizon_s,
            )

            semantic_evaluation = _evaluate_assessment(scenario.evaluation, assessment)
            semantic_passed = (
                semantic_evaluation["passed"] if semantic_evaluation is not None else None
            )
            assert semantic_passed is None or isinstance(semantic_passed, bool)
            command_data = asdict(command)
            command_data["action"] = command.action.value
            metadata["outcome"] = {
                "status": "success",
                "model_id": response.model_id,
                "latency_s": response.latency_s,
                "provider_metadata": asdict(response.metadata),
                "prompt_contract_version": request.prompt_contract_version,
                "raw_response_file": "raw_response.txt",
                "parsed_assessment": _assessment_data(assessment),
                "parsed_command": command_data,
                "semantic_evaluation": semantic_evaluation,
            }
            _write_json(target / "metadata.json", metadata)
            result = ProbeScenarioResult(
                scenario_id,
                "success",
                str(target),
                semantic_passed=semantic_passed,
                **_result_dimensions(semantic_evaluation),
            )
        except Exception as exc:
            if recorder.response is not None:
                (target / "raw_response.txt").write_text(
                    recorder.response.text,
                    encoding="utf-8",
                )
            outcome: dict[str, object] = {
                "status": "failure",
                "error_category": type(exc).__name__,
                "error_message": str(exc),
                "raw_response_file": (
                    "raw_response.txt" if recorder.response is not None else None
                ),
            }
            if recorder.response is not None:
                outcome["model_id"] = recorder.response.model_id
                outcome["latency_s"] = recorder.response.latency_s
                outcome["provider_metadata"] = asdict(recorder.response.metadata)
            metadata["outcome"] = outcome
            _write_json(target / "metadata.json", metadata)
            result = ProbeScenarioResult(
                scenario_id,
                "failure",
                str(target),
                type(exc).__name__,
                semantic_passed=False if evaluation_defined else None,
            )
        results.append(result)

    summary = ProbeRunSummary(
        output_dir=str(root),
        inference_attempted=True,
        scenarios=tuple(results),
    )
    _write_json(root / "summary.json", summary.to_dict())
    return summary


def load_vla_probe_replay_inputs(
    source_dir: Path | str,
    *,
    scenario_ids: Sequence[str] | None = None,
    require_prompt_hash: bool = True,
) -> tuple[ProbeReplayInput, ...]:
    """Validate every selected artifact before any provider request is allowed."""
    if not isinstance(require_prompt_hash, bool):
        raise ValueError("require_prompt_hash must be a boolean")
    source_root = Path(source_dir).resolve()
    available, selected_ids = _probe_artifact_selection(source_root, scenario_ids)
    return tuple(
        _load_probe_replay_input(
            available[scenario_id],
            scenario_id,
            require_prompt_hash=require_prompt_hash,
        )
        for scenario_id in selected_ids
    )


def capture_probe_scenario(config: AppConfig, scenario: ProbeScenario) -> ProbeCapture:
    """Launch one isolated MetaDrive fixture and capture its settled front camera."""
    simulator = replace(
        config.simulator,
        map=scenario.map_name,
        traffic_density=scenario.traffic_density,
        obstacle_probability=0.0,
        num_scenarios=1,
        start_seed=scenario.seed,
        horizon=max(2, scenario.settle_steps + 1),
        use_lidar=config.simulator.use_lidar,
        headless=True,
        manual_control=False,
        out_of_road_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
    )
    camera = replace(config.camera, enabled=True)
    env = make_env(simulator, camera)
    try:
        env.reset()
        for vehicle in scenario.vehicles:
            spawn_static_vehicle_ahead(
                env,
                vehicle.distance_m,
                lane_offset=vehicle.lane_offset,
                lateral_m=vehicle.lateral_m,
                vehicle_kind=vehicle.kind,
                name=f"probe-{scenario.scenario_id}-{vehicle.object_id}",
            )
        if scenario.traffic_light is not None:
            lights = spawn_traffic_light_group_ahead(
                env,
                scenario.traffic_light.distance_m,
                name_prefix=f"probe-{scenario.scenario_id}-traffic-light",
                visual_scale=scenario.traffic_light.visual_scale,
            )
            for light in lights:
                set_traffic_light_state(light, scenario.traffic_light.state)
        for _ in range(scenario.settle_steps):
            env.step((0.0, 0.0))
        timestamp_s = simulation_time_s(env)
        frame = MetaDriveCameraAdapter().capture(env, timestamp_s=timestamp_s)
        route = (
            navigation_waypoints(env, config.planner.waypoint_spacing_m)
            if config.planner.route_source == "map"
            else config.planner.default_route
        )
        adapter_type = (
            OracleSceneAdapter
            if config.perception.safety_source == "oracle"
            else LidarSceneAdapter
        )
        scene = adapter_type(
            detection_radius_m=config.perception.detection_radius_m,
            corridor_margin_m=config.perception.corridor_margin_m,
            future_path=PolylineFuturePath(route),
        ).observe(env, timestamp_s=timestamp_s)
        return ProbeCapture(
            frame=frame,
            timestamp_s=timestamp_s,
            ego_speed_mps=float(env.agent.speed),
            scene=scene,
        )
    finally:
        env.close()


def _write_probe_artifacts(
    config: AppConfig,
    scenario: ProbeScenario,
    capture: ProbeCapture,
    target: Path,
    *,
    infer: bool,
    provider: ModelProvider | None,
    observation_builder: ObservationBuilder,
) -> ProbeScenarioResult:
    request_id = f"probe-{scenario.scenario_id}-seed-{scenario.seed}"
    observation_request = ObservationRequest(
        frame=capture.frame,
        now_s=capture.timestamp_s,
        action_horizon_s=config.vla.action_horizon_s,
        contract=OUTPUT_CONTRACT,
        scene=capture.scene,
        ego_speed_mps=capture.ego_speed_mps,
        cruise_speed_mps=config.controller.target_speed_mps,
        max_scene_objects=config.vla.max_scene_objects,
        prompt_policy=config.vla.prompt_policy,
    )
    # Built once and saved exactly as the model receives it.
    observation = checked_observation(
        observation_builder.build(observation_request),
        observation_request,
    )
    prompt = observation.prompt
    png_bytes = encode_rgb_frame_png(observation.frame)
    scene_context = build_vla_scene_context(
        capture.scene,
        max_scene_objects=config.vla.max_scene_objects,
    )
    scene_bytes = (
        None
        if scene_context is None
        else (
            json.dumps(
                scene_context,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )
    (target / "frame.png").write_bytes(png_bytes)
    (target / "prompt.txt").write_text(prompt, encoding="utf-8")
    if scene_bytes is not None:
        (target / "scene.json").write_bytes(scene_bytes)

    metadata = _base_metadata(scenario, infer=infer)
    metadata["capture"] = {
        "timestamp_s": capture.timestamp_s,
        # The image and prompt the observation carried, which replay sends again.
        "observation_builder": _qualified_name(observation_builder),
        "width": observation.frame.width,
        "height": observation.frame.height,
        "rgb_sha256": hashlib.sha256(observation.frame.rgb_bytes).hexdigest(),
        "png_sha256": hashlib.sha256(png_bytes).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_file": "frame.png",
        "prompt_file": "prompt.txt",
        "scene_context_included": scene_bytes is not None,
        "scene_file": "scene.json" if scene_bytes is not None else None,
        "scene_sha256": (
            hashlib.sha256(scene_bytes).hexdigest() if scene_bytes is not None else None
        ),
        "ego_speed_mps": capture.ego_speed_mps,
        "cruise_speed_mps": config.controller.target_speed_mps,
    }
    metadata["request"] = {
        "request_id": request_id,
        "created_at_s": capture.timestamp_s,
        "prompt_contract_version": VLA_PROMPT_CONTRACT_VERSION,
    }

    if not infer:
        metadata["outcome"] = {"status": "captured"}
        _write_json(target / "metadata.json", metadata)
        return ProbeScenarioResult(scenario.scenario_id, "captured", str(target))

    assert provider is not None
    recorder = _RecordingModelProvider(provider)
    pipeline = VLAInferencePipeline(
        recorder,
        timeout_s=config.vla.request_timeout_s,
        action_horizon_s=config.vla.action_horizon_s,
        max_scene_objects=config.vla.max_scene_objects,
        maximum_frame_age_s=config.vla.maximum_frame_age_s,
        maximum_clock_skew_s=config.vla.maximum_clock_skew_s,
        prompt_policy=config.vla.prompt_policy,
        observation_builder=_SavedObservation(observation),
    )
    try:
        inference = pipeline.infer(
            capture.frame,
            now_s=capture.timestamp_s,
            request_id=request_id,
            scene=capture.scene,
            ego_speed_mps=capture.ego_speed_mps,
            cruise_speed_mps=config.controller.target_speed_mps,
        )
        if (
            inference.request.prompt != prompt
            or inference.request.frame != observation.frame
        ):
            raise RuntimeError("probe artifacts do not match the model request")
        (target / "raw_response.txt").write_text(
            inference.response.text,
            encoding="utf-8",
        )
        command = asdict(inference.requested_command)
        command["action"] = inference.requested_command.action.value
        semantic_evaluation = _evaluate_assessment(
            scenario.evaluation,
            inference.assessment,
        )
        metadata["outcome"] = {
            "status": "success",
            "model_id": inference.response.model_id,
            "latency_s": inference.response.latency_s,
            "provider_metadata": asdict(inference.response.metadata),
            "prompt_contract_version": inference.request.prompt_contract_version,
            "raw_response_file": "raw_response.txt",
            "parsed_assessment": _assessment_data(inference.assessment),
            "parsed_command": command,
            "semantic_evaluation": semantic_evaluation,
        }
        _write_json(target / "metadata.json", metadata)
        semantic_passed = (
            semantic_evaluation["passed"] if semantic_evaluation is not None else None
        )
        assert semantic_passed is None or isinstance(semantic_passed, bool)
        return ProbeScenarioResult(
            scenario.scenario_id,
            "success",
            str(target),
            semantic_passed=semantic_passed,
            **_result_dimensions(semantic_evaluation),
        )
    except Exception as exc:
        if recorder.response is not None:
            (target / "raw_response.txt").write_text(
                recorder.response.text,
                encoding="utf-8",
            )
        outcome: dict[str, object] = {
            "status": "failure",
            "error_category": type(exc).__name__,
            "error_message": str(exc),
            "raw_response_file": "raw_response.txt" if recorder.response is not None else None,
        }
        if recorder.response is not None:
            outcome["model_id"] = recorder.response.model_id
            outcome["latency_s"] = recorder.response.latency_s
            outcome["provider_metadata"] = asdict(recorder.response.metadata)
        metadata["outcome"] = outcome
        _write_json(target / "metadata.json", metadata)
        return ProbeScenarioResult(
            scenario.scenario_id,
            "failure",
            str(target),
            type(exc).__name__,
            semantic_passed=False if scenario.evaluation is not None else None,
        )


class _SavedObservation:
    """Hands the pipeline the observation already built and saved."""

    def __init__(self, observation: Observation) -> None:
        self._observation = observation

    def build(self, request: ObservationRequest) -> Observation:
        return self._observation


def _qualified_name(component: object) -> str:
    component_type = type(component)
    return f"{component_type.__module__}.{component_type.__qualname__}"


class _RecordingModelProvider:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider
        self.request: ModelRequest | None = None
        self.response: ModelResponse | None = None

    def generate(self, request: ModelRequest, *, timeout_s: float) -> ModelResponse:
        self.request = request
        response = self.provider.generate(request, timeout_s=timeout_s)
        self.response = response
        return response


def _base_metadata(scenario: ProbeScenario, *, infer: bool) -> dict[str, object]:
    scenario_data = asdict(scenario)
    benchmark_version = scenario_data.pop("benchmark_version")
    scenario_data["id"] = scenario_data.pop("scenario_id")
    scenario_data["map"] = scenario_data.pop("map_name")
    for vehicle in scenario_data["vehicles"]:  # type: ignore[union-attr]
        vehicle["id"] = vehicle.pop("object_id")
    traffic_light = scenario_data["traffic_light"]
    if traffic_light is not None:
        traffic_light["state"] = traffic_light["state"].value
    evaluation = scenario_data["evaluation"]
    if evaluation is not None:
        for hazard in evaluation["required_hazards"]:
            hazard["type"] = hazard.pop("hazard_type")
    return {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "benchmark_version": benchmark_version,
        "scenario": scenario_data,
        "mode": "http_inference" if infer else "capture_only",
    }


def _select_scenarios(
    scenarios: Sequence[ProbeScenario],
    scenario_ids: Sequence[str] | None,
) -> tuple[ProbeScenario, ...]:
    if not scenarios or not all(isinstance(scenario, ProbeScenario) for scenario in scenarios):
        raise ValueError("scenarios must contain at least one ProbeScenario")
    available = {scenario.scenario_id: scenario for scenario in scenarios}
    if len(available) != len(scenarios):
        raise ValueError("scenario ids must be unique")
    if scenario_ids is None:
        return tuple(scenarios)
    selected: list[ProbeScenario] = []
    seen: set[str] = set()
    for scenario_id in scenario_ids:
        if scenario_id not in available:
            raise ValueError(f"unknown probe scenario id: {scenario_id}")
        if scenario_id not in seen:
            selected.append(available[scenario_id])
            seen.add(scenario_id)
    if not selected:
        raise ValueError("at least one probe scenario id is required")
    return tuple(selected)


def _select_artifact_ids(
    available: dict[str, Path],
    scenario_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if scenario_ids is None:
        return tuple(sorted(available))
    selected: list[str] = []
    seen: set[str] = set()
    for scenario_id in scenario_ids:
        _validate_identifier(scenario_id, "probe scenario id")
        if scenario_id not in available:
            raise ValueError(f"unknown probe scenario id: {scenario_id}")
        if scenario_id not in seen:
            selected.append(scenario_id)
            seen.add(scenario_id)
    if not selected:
        raise ValueError("at least one probe scenario id is required")
    return tuple(selected)


def _probe_artifact_selection(
    source_root: Path,
    scenario_ids: Sequence[str] | None,
) -> tuple[dict[str, Path], tuple[str, ...]]:
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"probe artifact directory does not exist: {source_root}"
        )
    available = {
        child.name: child
        for child in source_root.iterdir()
        if child.is_dir() and (child / "metadata.json").is_file()
    }
    if not available:
        raise ValueError("probe artifact directory contains no scenario artifacts")
    return available, _select_artifact_ids(available, scenario_ids)


def _load_probe_replay_input(
    source: Path,
    scenario_id: str,
    *,
    require_prompt_hash: bool,
) -> ProbeReplayInput:
    source_metadata = _read_json_object(source / "metadata.json")
    artifact_schema_version = source_metadata.get("schema_version")
    if artifact_schema_version not in _SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(
            "probe artifact schema is incompatible; recapture with the assessment contract"
        )
    benchmark_version = source_metadata.get("benchmark_version", 1)
    if (
        not isinstance(benchmark_version, int)
        or isinstance(benchmark_version, bool)
        or benchmark_version not in _SUPPORTED_MANIFEST_SCHEMA_VERSIONS
    ):
        raise ValueError("probe artifact benchmark_version must be 1 or 2")
    scenario = _parse_scenario(
        source_metadata.get("scenario"),
        benchmark_version=benchmark_version,
    )
    if scenario.scenario_id != scenario_id:
        raise ValueError("scenario directory and metadata ids do not match")
    capture = _json_object(source_metadata.get("capture"), "capture")
    request_data = _json_object(source_metadata.get("request"), "request")
    image_file = _artifact_filename(capture, "image_file")
    prompt_file = _artifact_filename(capture, "prompt_file")
    png_bytes = (source / image_file).read_bytes()
    prompt = (source / prompt_file).read_text(encoding="utf-8")
    expected_prompt_sha256 = capture.get("prompt_sha256")
    if expected_prompt_sha256 is None:
        if require_prompt_hash:
            raise ValueError(
                "probe artifact is missing prompt_sha256; recapture before live use"
            )
    elif (
        not isinstance(expected_prompt_sha256, str)
        or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        != expected_prompt_sha256
    ):
        raise ValueError("saved prompt hash does not match metadata")
    timestamp_s = _artifact_number(capture, "timestamp_s")
    frame = _decode_probe_png(png_bytes, timestamp_s=timestamp_s)
    _verify_probe_hashes(capture, frame, png_bytes)
    scene_bytes = _load_probe_scene_bytes(
        source,
        capture,
        prompt,
        required=artifact_schema_version >= 3,
        embedded_in_prompt=artifact_schema_version <= 3,
    )
    request = ModelRequest(
        request_id=_artifact_string(request_data, "request_id"),
        prompt=prompt,
        frame=frame,
        created_at_s=_artifact_number(request_data, "created_at_s"),
        encoded_png_bytes=png_bytes,
        prompt_contract_version=(
            request_data.get("prompt_contract_version")
            if isinstance(request_data.get("prompt_contract_version"), str)
            else "legacy-unversioned"
        ),
    )
    if not times_equal(request.created_at_s, timestamp_s):
        raise ValueError("saved request and capture timestamps do not match")
    return ProbeReplayInput(
        scenario=scenario,
        source_dir=source,
        source_metadata=source_metadata,
        request=request,
        png_bytes=png_bytes,
        scene_bytes=scene_bytes,
    )


def _decode_probe_png(png_bytes: bytes, *, timestamp_s: float) -> RGBFrame:
    signature = b"\x89PNG\r\n\x1a\n"
    if not isinstance(png_bytes, bytes) or not png_bytes.startswith(signature):
        raise ValueError("saved frame is not a PNG")
    position = len(signature)
    ihdr: bytes | None = None
    compressed = bytearray()
    saw_end = False
    while position < len(png_bytes):
        if position + 12 > len(png_bytes):
            raise ValueError("saved PNG has a truncated chunk")
        length = struct.unpack(">I", png_bytes[position : position + 4])[0]
        kind = png_bytes[position + 4 : position + 8]
        data_start = position + 8
        data_end = data_start + length
        checksum_end = data_end + 4
        if checksum_end > len(png_bytes):
            raise ValueError("saved PNG has a truncated chunk")
        data = png_bytes[data_start:data_end]
        expected_crc = struct.unpack(">I", png_bytes[data_end:checksum_end])[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != expected_crc:
            raise ValueError("saved PNG has an invalid chunk checksum")
        if kind == b"IHDR":
            if ihdr is not None:
                raise ValueError("saved PNG has multiple IHDR chunks")
            ihdr = data
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            saw_end = True
            position = checksum_end
            break
        position = checksum_end
    if ihdr is None or len(ihdr) != 13 or not compressed or not saw_end:
        raise ValueError("saved PNG is missing required chunks")
    if position != len(png_bytes):
        raise ValueError("saved PNG has trailing data")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise ValueError("saved PNG is not an 8-bit non-interlaced RGB image")
    expected_size = height * (1 + width * 3)
    if expected_size <= 0 or expected_size > 256 * 1024 * 1024:
        raise ValueError("saved PNG dimensions are outside replay bounds")
    try:
        decompressor = zlib.decompressobj()
        scanlines = decompressor.decompress(bytes(compressed), expected_size + 1)
    except zlib.error as exc:
        raise ValueError("saved PNG has invalid compressed pixels") from exc
    if (
        len(scanlines) != expected_size
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError("saved PNG pixel data has an unexpected size")
    row_size = width * 3
    rows: list[bytes] = []
    for offset in range(0, len(scanlines), row_size + 1):
        if scanlines[offset] != 0:
            raise ValueError("saved PNG uses an unsupported row filter")
        rows.append(scanlines[offset + 1 : offset + 1 + row_size])
    return RGBFrame(timestamp_s, width, height, b"".join(rows))


def _verify_probe_hashes(
    capture: dict[str, object],
    frame: RGBFrame,
    png_bytes: bytes,
) -> None:
    expected_rgb = _artifact_string(capture, "rgb_sha256")
    expected_png = _artifact_string(capture, "png_sha256")
    if hashlib.sha256(frame.rgb_bytes).hexdigest() != expected_rgb:
        raise ValueError("saved frame RGB hash does not match metadata")
    if hashlib.sha256(png_bytes).hexdigest() != expected_png:
        raise ValueError("saved frame PNG hash does not match metadata")
    if frame.width != int(_artifact_number(capture, "width")) or frame.height != int(
        _artifact_number(capture, "height")
    ):
        raise ValueError("saved frame dimensions do not match metadata")


def _load_probe_scene_bytes(
    source: Path,
    capture: dict[str, object],
    prompt: str,
    *,
    required: bool,
    embedded_in_prompt: bool,
) -> bytes | None:
    """The measured scene context saved beside the prompt, checked against its hash.

    Artifacts before version 4 saved DriveBench's own prompt, which ends with that
    context, so their prompt must still match it.
    """
    included = capture.get("scene_context_included")
    if not isinstance(included, bool):
        if required:
            raise ValueError("probe artifact scene_context_included must be a boolean")
        return None
    if not included:
        if required and (
            capture.get("scene_file") is not None
            or capture.get("scene_sha256") is not None
        ):
            raise ValueError("probe artifact without scene context must not name scene evidence")
        return None

    scene_file = _artifact_filename(capture, "scene_file")
    expected_sha256 = _artifact_string(capture, "scene_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("artifact scene_sha256 must be lowercase SHA-256")
    scene_bytes = (source / scene_file).read_bytes()
    if hashlib.sha256(scene_bytes).hexdigest() != expected_sha256:
        raise ValueError("saved scene hash does not match metadata")
    try:
        scene_context = json.loads(scene_bytes)
        compact = json.dumps(
            scene_context,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("saved scene context is not valid finite JSON") from exc
    if embedded_in_prompt and not prompt.endswith(f"LOCAL_SCENE_CONTEXT:\n{compact}"):
        raise ValueError("saved scene context does not match the exact prompt")
    return scene_bytes


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _json_object(value, path.name)


def _json_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _artifact_string(data: dict[str, object], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact {name} must be a non-empty string")
    return value


def _artifact_filename(data: dict[str, object], name: str) -> str:
    value = _artifact_string(data, name)
    if Path(value).name != value:
        raise ValueError(f"artifact {name} must be a filename")
    return value


def _artifact_number(data: dict[str, object], name: str) -> float:
    value = data.get(name)
    if not _finite_number(value):
        raise ValueError(f"artifact {name} must be finite and numeric")
    return float(value)


def _parse_scenario(value: object, *, benchmark_version: int = 1) -> ProbeScenario:
    if not isinstance(value, dict):
        raise ValueError("each probe scenario must be a mapping")
    allowed = {
        "id",
        "description",
        "expected_visual",
        "map",
        "seed",
        "settle_steps",
        "traffic_density",
        "vehicles",
        "traffic_light",
        "evaluation",
    }
    _reject_unknown_keys(value, allowed, "probe scenario")
    vehicles_data = value.get("vehicles", [])
    if not isinstance(vehicles_data, list):
        raise ValueError("probe scenario vehicles must be a list")
    vehicles = tuple(_parse_vehicle(vehicle) for vehicle in vehicles_data)
    try:
        return ProbeScenario(
            scenario_id=value["id"],
            description=value["description"],
            expected_visual=value["expected_visual"],
            map_name=value["map"],
            seed=value.get("seed", 0),
            settle_steps=value.get("settle_steps", 1),
            traffic_density=value.get("traffic_density", 0.0),
            vehicles=vehicles,
            traffic_light=_parse_traffic_light(value.get("traffic_light")),
            evaluation=_parse_evaluation(value.get("evaluation")),
            benchmark_version=benchmark_version,
        )
    except KeyError as exc:
        raise ValueError(f"probe scenario is missing field {exc.args[0]!r}") from exc


def _parse_vehicle(value: object) -> ProbeVehicleSpec:
    if not isinstance(value, dict):
        raise ValueError("each probe vehicle must be a mapping")
    _reject_unknown_keys(
        value,
        {"id", "distance_m", "lane_offset", "lateral_m", "kind"},
        "probe vehicle",
    )
    try:
        return ProbeVehicleSpec(
            object_id=value["id"],
            distance_m=value["distance_m"],
            lane_offset=value.get("lane_offset", 0),
            lateral_m=value.get("lateral_m", 0.0),
            kind=value.get("kind", "car"),
        )
    except KeyError as exc:
        raise ValueError(f"probe vehicle is missing field {exc.args[0]!r}") from exc


def _parse_traffic_light(value: object) -> ProbeTrafficLightSpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("probe traffic_light must be a mapping")
    _reject_unknown_keys(
        value,
        {"distance_m", "state", "visual_scale"},
        "probe traffic_light",
    )
    try:
        return ProbeTrafficLightSpec(
            distance_m=value["distance_m"],
            state=TrafficLightState(value["state"]),
            visual_scale=value.get("visual_scale", 1.75),
        )
    except KeyError as exc:
        raise ValueError(f"probe traffic_light is missing field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("probe traffic light state must be red, green, or yellow") from exc


def _parse_evaluation(value: object) -> ProbeEvaluationSpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("probe evaluation must be a mapping")
    _reject_unknown_keys(
        value,
        {
            "acceptable_actions",
            "minimum_target_speed_mps",
            "maximum_target_speed_mps",
            "required_hazards",
            "raw_speed_required",
        },
        "probe evaluation",
    )
    actions = value.get("acceptable_actions")
    if not isinstance(actions, list):
        raise ValueError("probe evaluation acceptable_actions must be a list")
    try:
        acceptable_actions = tuple(HighLevelAction(action) for action in actions)
    except (TypeError, ValueError) as exc:
        raise ValueError("probe evaluation contains an unknown VLA action") from exc
    hazards_data = value.get("required_hazards", [])
    if not isinstance(hazards_data, list):
        raise ValueError("probe evaluation required_hazards must be a list")
    required_hazards = tuple(
        _parse_hazard_expectation(item) for item in hazards_data
    )
    try:
        return ProbeEvaluationSpec(
            acceptable_actions=acceptable_actions,
            minimum_target_speed_mps=_probe_speed_mps(value, "minimum_target_speed"),
            maximum_target_speed_mps=_probe_speed_mps(value, "maximum_target_speed"),
            required_hazards=required_hazards,
            raw_speed_required=value.get("raw_speed_required", True),
        )
    except KeyError as exc:
        raise ValueError(f"probe evaluation is missing field {exc.args[0]!r}") from exc


def _parse_hazard_expectation(value: object) -> ProbeHazardExpectation:
    if not isinstance(value, dict):
        raise ValueError("each required hazard must be a mapping")
    _reject_unknown_keys(
        value,
        {"type", "hazard_type", "relative_location"},
        "probe required hazard",
    )
    if "type" in value and "hazard_type" in value:
        raise ValueError("probe required hazard must use only field 'type'")
    try:
        hazard_type = value.get("type", value.get("hazard_type"))
        if hazard_type is None:
            raise KeyError("type")
        return ProbeHazardExpectation(
            hazard_type=HazardType(hazard_type),
            relative_location=RelativeLocation(value["relative_location"]),
        )
    except KeyError as exc:
        raise ValueError(f"probe required hazard is missing field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("probe required hazard contains an unknown value") from exc


def _evaluate_assessment(
    evaluation: ProbeEvaluationSpec | None,
    assessment: VLAAssessment,
) -> dict[str, object] | None:
    if evaluation is None:
        return None
    tactical_action_passed = assessment.proposed_action in evaluation.acceptable_actions
    raw_speed_passed = (
        evaluation.minimum_target_speed_mps
        <= assessment.proposed_target_speed_mps
        <= evaluation.maximum_target_speed_mps
    )
    observed_hazards = {
        (hazard.hazard_type, hazard.relative_location)
        for hazard in assessment.relevant_hazards
    }
    perception_passed = (
        all(
            (hazard.hazard_type, hazard.relative_location) in observed_hazards
            for hazard in evaluation.required_hazards
        )
        if evaluation.required_hazards
        else None
    )
    passed = tactical_action_passed
    if perception_passed is not None:
        passed = passed and perception_passed
    if evaluation.raw_speed_required:
        passed = passed and raw_speed_passed
    return {
        "passed": passed,
        "perception_passed": perception_passed,
        "tactical_action_passed": tactical_action_passed,
        "raw_speed_passed": raw_speed_passed,
        "raw_speed_required": evaluation.raw_speed_required,
        "action_passed": tactical_action_passed,
        "target_speed_passed": raw_speed_passed,
        "required_hazards": [
            {
                "type": hazard.hazard_type.value,
                "relative_location": hazard.relative_location.value,
            }
            for hazard in evaluation.required_hazards
        ],
        "acceptable_actions": [action.value for action in evaluation.acceptable_actions],
        "minimum_target_speed_mps": evaluation.minimum_target_speed_mps,
        "maximum_target_speed_mps": evaluation.maximum_target_speed_mps,
    }


def _result_dimensions(
    semantic_evaluation: dict[str, object] | None,
) -> dict[str, bool | None]:
    if semantic_evaluation is None:
        return {
            "perception_passed": None,
            "tactical_action_passed": None,
            "raw_speed_passed": None,
        }
    return {
        "perception_passed": _optional_bool(semantic_evaluation["perception_passed"]),
        "tactical_action_passed": _optional_bool(
            semantic_evaluation["tactical_action_passed"]
        ),
        "raw_speed_passed": _optional_bool(semantic_evaluation["raw_speed_passed"]),
    }


def _optional_bool(value: object) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise TypeError("semantic evaluation dimension must be boolean or None")


def _dimension_success(
    scenarios: Sequence[ProbeScenarioResult],
    field: str,
) -> bool | None:
    values = [getattr(result, field) for result in scenarios]
    scored = [value for value in values if value is not None]
    return all(scored) if scored else None


def _assessment_data(assessment: VLAAssessment) -> dict[str, object]:
    data = asdict(assessment)
    data["proposed_action"] = assessment.proposed_action.value
    data["uncertainty"] = assessment.uncertainty
    return data


def _probe_speed_mps(value: dict[object, object], stem: str) -> float:
    mps_key = f"{stem}_mps"
    if mps_key not in value:
        raise ValueError(f"probe evaluation is missing field {mps_key!r}")
    raw_value = value[mps_key]
    if not _finite_number(raw_value):
        raise ValueError(f"probe evaluation {mps_key} must be finite")
    return float(raw_value)


def _reject_unknown_keys(value: dict[object, object], allowed: set[str], context: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _validate_identifier(value: object, context: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{context} must match {_IDENTIFIER.pattern!r}")


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
