from __future__ import annotations

import hashlib
import math
import uuid
from collections import Counter
from dataclasses import dataclass, field, replace
from statistics import median

from metadrive_starter.config import AppConfig, ObservationSettings
from metadrive_starter.controllers import (
    VehicleController,
    VehicleControllerFactory,
    build_vehicle_controller,
    checked_action,
    saturated_action,
)
from metadrive_starter.episode_replay import write_episode_artifact
from metadrive_starter.env import (
    control_timestep_s,
    ego_state_info,
    lane_change_waypoints,
    lane_topology,
    make_env,
    navigation_waypoints,
    simulation_time_s,
    spawn_scenario_vehicle,
    spawn_static_vehicle_ahead,
    spawn_stopped_vehicle_ahead,
    spawn_traffic_light_group_ahead,
)
from metadrive_starter.events import EventLogger
from metadrive_starter.faults import (
    FaultInjectingModelProvider,
    FaultKind,
    FaultSchedule,
    has_fault,
    has_provider_faults,
)
from metadrive_starter.perception import (
    BasicPerception,
    LaneRelation,
    LidarSceneAdapter,
    LocalScene,
    OracleSceneAdapter,
)
from metadrive_starter.planning import (
    ActionSpeedPolicy,
    PolylineFuturePath,
    WaypointPathPlanner,
)
from metadrive_starter.planning.command_executor import (
    CommandExecutionDecision,
    CommandExecutionSource,
    VLACommandExecutor,
)
from metadrive_starter.planning.lane_change import (
    LaneChangeCoordinator,
    LaneChangeDecision,
    LaneChangePhase,
)
from metadrive_starter.safety import (
    CommandDisposition,
    EmergencyBrakingSupervisor,
    HeadwaySpeedCapDecision,
    LaneChangeClearanceDecision,
    SafetyDecision,
    SafetyLevel,
    TimeHeadwaySpeedCap,
    VLACommandValidator,
    build_high_level_safety_policy,
)
from metadrive_starter.submission import apply_reference_overlay, load_reference_controller
from metadrive_starter.timing import RealTimePacer
from metadrive_starter.traffic_lights import (
    TrafficLightController,
    TrafficLightCycle,
    TrafficLightState,
)
from metadrive_starter.types import Action, ControlCommand, ControlTick, Plan
from metadrive_starter.units import mps_to_kmh
from metadrive_starter.vla import (
    ArbitrationOutcome,
    AssessmentArbiter,
    AssessmentArbiterFactory,
    DefaultArbiter,
    DefaultObservationBuilder,
    HighLevelAction,
    InferenceCompletion,
    InferenceDiscarded,
    InferenceFailure,
    InferenceSubmitDisposition,
    InferenceSuccess,
    ModelResponse,
    ObservationBuilder,
    ObservationBuilderFactory,
    RequestBudget,
    VLAInferencePipeline,
    VLA_PROMPT_CONTRACT_VERSION,
    VLAInferenceScheduler,
    build_arbiter,
    build_observation_builder,
    model_failure_category,
)
from metadrive_starter.vla_control import VLACommandRuntime, VLAControlUpdate
from metadrive_starter.vla_runtime import build_vla_scheduler


@dataclass(frozen=True)
class VLAMetricCount:
    label: str
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("VLA metric label must not be empty")
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 1
        ):
            raise ValueError("VLA metric count must be a positive integer")


@dataclass(frozen=True)
class VLARunMetrics:
    """Cloud/local-model participation evidence for one simulation run."""

    enabled: bool = False
    prompt_contract_version: str = VLA_PROMPT_CONTRACT_VERSION
    request_cap: int | None = None
    provider_requests_attempted: int | None = None
    provider_requests_remaining: int | None = None
    request_cap_exhausted: bool = False
    requests_started: int = 0
    submissions_busy: int = 0
    submissions_rate_limited: int = 0
    submissions_closed: int = 0
    responses_succeeded: int = 0
    responses_failed: int = 0
    responses_discarded: int = 0
    validations_accepted: int = 0
    validations_modified: int = 0
    validations_rejected: int = 0
    validations_fallback: int = 0
    authority_steps: int = 0
    fallback_steps: int = 0
    runtime_errors: int = 0
    observations_built: int = 0
    # Sent although they lack an output-contract line; each run warns on the first.
    observations_without_output_contract: int = 0
    # Collected assessments by what arbitration left of them.
    arbitrations_endorsed: int = 0
    arbitrations_restricted: int = 0
    arbitrations_declined: int = 0
    # Model commands a review ended before their horizon.
    arbitration_revocations: int = 0
    # Arbiter answers that could not stand: an arbitrate that raised or answered
    # outside the model's proposal, which the runtime declined, or a review that
    # raised or did not answer True or False, which ended any model command.
    arbitration_problems: int = 0
    # The slowest arbitrate or review call.
    arbitration_slowest_s: float | None = None
    latency_p50_s: float | None = None
    latency_p95_s: float | None = None
    latency_max_s: float | None = None
    responses_with_token_usage: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    providers: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    model_versions: tuple[str, ...] = ()
    fixture_ids: tuple[str, ...] = ()
    fixture_sha256s: tuple[str, ...] = ()
    failure_categories: tuple[VLAMetricCount, ...] = ()
    runtime_error_categories: tuple[VLAMetricCount, ...] = ()
    validation_reasons: tuple[VLAMetricCount, ...] = ()


@dataclass(frozen=True)
class RunSummary:
    steps: int
    total_reward: float
    control_mode: str
    # Qualified class name of the vehicle controller; None under manual control.
    controller: str | None = None
    # Qualified class name of the observation builder; None without the VLA subsystem.
    observation: str | None = None
    # Qualified class name of the arbiter; None without the VLA subsystem.
    arbiter: str | None = None
    # Control ticks whose action the actuator range had to clamp.
    controller_outputs_clamped: int = 0
    route_completion: float = 0.0
    arrived: bool = False
    crashed: bool = False
    went_off_road: bool = False
    safety_interventions: int = 0
    emergency_brakes: int = 0
    minimum_safety_gap_m: float | None = None
    headway_cap_evaluations: int = 0
    headway_cap_would_intervene: int = 0
    headway_cap_applied: int = 0
    lane_changes_started: int = 0
    lane_changes_completed: int = 0
    lane_changes_aborted: int = 0
    lane_change_hazards_spawned: int = 0
    fault_activations: int = 0
    final_speed_mps: float = 0.0
    simulation_time_s: float = 0.0
    run_id: str | None = None
    event_log_path: str | None = None
    replay_artifact_path: str | None = None
    replay_artifact_sha256: str | None = None
    vla_metrics: VLARunMetrics = field(default_factory=VLARunMetrics)
    dry_run: bool = False


class _VLAMetricsCollector:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.request_budget: RequestBudget | None = None
        self.pipeline: VLAInferencePipeline | None = None
        self.requests_started = 0
        self.submissions_busy = 0
        self.submissions_rate_limited = 0
        self.submissions_closed = 0
        self.responses_succeeded = 0
        self.responses_failed = 0
        self.responses_discarded = 0
        self.validations = Counter[str]()
        self.arbitrations = Counter[ArbitrationOutcome]()
        self.arbitration_revocations = 0
        self.arbitration_problems = 0
        self.arbitration_slowest_s: float | None = None
        self.authority_steps = 0
        self.fallback_steps = 0
        self.runtime_errors = 0
        self.latencies_s: list[float] = []
        self.responses_with_token_usage = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.providers: set[str] = set()
        self.model_ids: set[str] = set()
        self.model_versions: set[str] = set()
        self.fixture_ids: set[str] = set()
        self.fixture_sha256s: set[str] = set()
        self.failure_categories = Counter[str]()
        self.runtime_error_categories = Counter[str]()
        self.validation_reasons = Counter[str]()

    def bind_request_budget(self, request_budget: RequestBudget | None) -> None:
        if request_budget is not None and not isinstance(request_budget, RequestBudget):
            raise TypeError("request_budget must be a RequestBudget or None")
        if self.request_budget is not None:
            raise RuntimeError("request budget is already bound")
        self.request_budget = request_budget

    def bind_pipeline(self, pipeline: VLAInferencePipeline) -> None:
        """Report the observations ``pipeline`` builds."""
        self.pipeline = pipeline

    def observe_update(self, update: VLAControlUpdate) -> None:
        submission = update.submission_disposition
        if submission is InferenceSubmitDisposition.STARTED:
            self.requests_started += 1
        elif submission is InferenceSubmitDisposition.BUSY:
            self.submissions_busy += 1
        elif submission is InferenceSubmitDisposition.RATE_LIMITED:
            self.submissions_rate_limited += 1
        elif submission is InferenceSubmitDisposition.CLOSED:
            self.submissions_closed += 1

        if update.completion is not None:
            self.observe_completion(update.completion)
        if update.arbitration is not None:
            self.arbitrations[update.arbitration.outcome] += 1
        if update.review is not None:
            self.arbitration_revocations += int(update.review.revoked)
        for answer in (update.arbitration, update.review):
            if answer is not None:
                self.arbitration_problems += int(answer.problem is not None)
                self.arbitration_slowest_s = max(
                    answer.elapsed_s,
                    self.arbitration_slowest_s or 0.0,
                )
        if update.validation is not None:
            disposition = update.validation.disposition.value
            self.validations[disposition] += 1
            if update.validation.disposition is not CommandDisposition.ACCEPTED:
                self.validation_reasons.update(update.validation.reasons)
        if update.execution.source is CommandExecutionSource.VLA:
            self.authority_steps += 1
        else:
            self.fallback_steps += 1
        if update.runtime_error is not None:
            self.runtime_errors += 1
            self.runtime_error_categories[
                model_failure_category(update.runtime_error).value
            ] += 1

    def observe_completion(
        self,
        completion: InferenceCompletion,
        *,
        discarded: bool = False,
    ) -> None:
        if discarded:
            self.responses_discarded += 1
        elif isinstance(completion, InferenceSuccess):
            self.responses_succeeded += 1
        elif isinstance(completion, InferenceFailure):
            self.responses_failed += 1
        elif isinstance(completion, InferenceDiscarded):
            self.responses_discarded += 1
        else:
            raise TypeError("completion must be a success, failure, or discard")

        if isinstance(completion, InferenceSuccess):
            self._observe_response(completion.result.response)
        elif isinstance(completion, InferenceFailure):
            category = model_failure_category(completion.error).value
            self.failure_categories[category] += 1
        elif completion.result is not None:
            self._observe_response(completion.result.response)
        elif completion.error is not None:
            category = model_failure_category(completion.error).value
            self.failure_categories[category] += 1

    def _observe_response(self, response: object) -> None:
        if not isinstance(response, ModelResponse):
            raise TypeError("response must be ModelResponse")
        self.latencies_s.append(response.latency_s)
        self.model_ids.add(response.model_id)
        metadata = response.metadata
        if metadata.provider is not None:
            self.providers.add(metadata.provider)
        if metadata.model_version is not None:
            self.model_versions.add(metadata.model_version)
        if metadata.fixture_id is not None:
            self.fixture_ids.add(metadata.fixture_id)
        if metadata.fixture_sha256 is not None:
            self.fixture_sha256s.add(metadata.fixture_sha256)
        token_counts = (
            metadata.input_tokens,
            metadata.output_tokens,
            metadata.total_tokens,
        )
        if any(value is not None for value in token_counts):
            self.responses_with_token_usage += 1
            self.input_tokens += metadata.input_tokens or 0
            self.output_tokens += metadata.output_tokens or 0
            self.total_tokens += metadata.total_tokens or 0

    def finish(self) -> VLARunMetrics:
        ordered = sorted(self.latencies_s)
        budget = self.request_budget
        return VLARunMetrics(
            enabled=self.enabled,
            request_cap=budget.maximum_requests if budget is not None else None,
            provider_requests_attempted=(
                budget.used_requests if budget is not None else None
            ),
            provider_requests_remaining=(
                budget.remaining_requests if budget is not None else None
            ),
            request_cap_exhausted=(
                budget.remaining_requests == 0 if budget is not None else False
            ),
            requests_started=self.requests_started,
            submissions_busy=self.submissions_busy,
            submissions_rate_limited=self.submissions_rate_limited,
            submissions_closed=self.submissions_closed,
            responses_succeeded=self.responses_succeeded,
            responses_failed=self.responses_failed,
            responses_discarded=self.responses_discarded,
            validations_accepted=self.validations[CommandDisposition.ACCEPTED.value],
            validations_modified=self.validations[CommandDisposition.MODIFIED.value],
            validations_rejected=self.validations[CommandDisposition.REJECTED.value],
            validations_fallback=self.validations[CommandDisposition.FALLBACK.value],
            authority_steps=self.authority_steps,
            fallback_steps=self.fallback_steps,
            runtime_errors=self.runtime_errors,
            observations_built=(
                self.pipeline.observations_built if self.pipeline is not None else 0
            ),
            observations_without_output_contract=(
                self.pipeline.observations_without_contract
                if self.pipeline is not None
                else 0
            ),
            arbitrations_endorsed=self.arbitrations[ArbitrationOutcome.ENDORSED],
            arbitrations_restricted=self.arbitrations[ArbitrationOutcome.RESTRICTED],
            arbitrations_declined=self.arbitrations[ArbitrationOutcome.DECLINED],
            arbitration_revocations=self.arbitration_revocations,
            arbitration_problems=self.arbitration_problems,
            arbitration_slowest_s=self.arbitration_slowest_s,
            latency_p50_s=median(ordered) if ordered else None,
            latency_p95_s=_nearest_rank_percentile(ordered, 0.95),
            latency_max_s=max(ordered) if ordered else None,
            responses_with_token_usage=self.responses_with_token_usage,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            providers=tuple(sorted(self.providers)),
            model_ids=tuple(sorted(self.model_ids)),
            model_versions=tuple(sorted(self.model_versions)),
            fixture_ids=tuple(sorted(self.fixture_ids)),
            fixture_sha256s=tuple(sorted(self.fixture_sha256s)),
            failure_categories=_metric_counts(self.failure_categories),
            runtime_error_categories=_metric_counts(self.runtime_error_categories),
            validation_reasons=_metric_counts(self.validation_reasons),
        )


def _metric_counts(values: Counter[str]) -> tuple[VLAMetricCount, ...]:
    return tuple(
        VLAMetricCount(label, count)
        for label, count in sorted(values.items())
        if count > 0
    )


def _nearest_rank_percentile(
    ordered_values: list[float],
    proportion: float,
) -> float | None:
    if not ordered_values:
        return None
    index = max(0, math.ceil(proportion * len(ordered_values)) - 1)
    return ordered_values[index]


def run_simulation(
    config: AppConfig,
    *,
    dry_run: bool = False,
    controller_factory: VehicleControllerFactory | None = None,
    observation_factory: ObservationBuilderFactory | None = None,
    arbitration_factory: AssessmentArbiterFactory | None = None,
) -> RunSummary:
    if controller_factory is None and not config.simulator.manual_control:
        # The reference drives, tuned as it tunes the starting gains.
        config = apply_reference_overlay(config)
    _validate_runtime_configuration(config)
    control_mode = (
        "manual"
        if config.simulator.manual_control
        else "vla_autopilot" if config.vla.enabled else "autopilot"
    )
    run_id = uuid.uuid4().hex
    event_logger = (
        EventLogger(
            config.event_log.path,
            run_id=run_id,
            scenario_id=config.event_log.scenario_id,
        )
        if config.event_log.enabled
        else None
    )
    event_log_path = str(event_logger.path) if event_logger is not None else None
    _event(
        event_logger,
        "run_started",
        sim_time_s=0.0,
        payload={
            "config": config.to_dict(),
            "control_mode": control_mode,
            "map": config.simulator.map,
            "seed": config.simulator.start_seed,
            "prompt_contract_version": VLA_PROMPT_CONTRACT_VERSION,
        },
    )

    try:
        controller = _build_controller(config, controller_factory)
        observation_builder = _build_observation_builder(config, observation_factory)
        arbiter = _build_arbiter(config, arbitration_factory)
        if dry_run:
            clamped = _dry_run_wiring(config, controller)
            summary = RunSummary(
                steps=0,
                total_reward=0.0,
                control_mode=control_mode,
                controller=_qualified_name(controller),
                observation=_qualified_name(observation_builder),
                arbiter=_qualified_name(arbiter),
                controller_outputs_clamped=clamped,
                run_id=run_id,
                event_log_path=event_log_path,
                vla_metrics=VLARunMetrics(
                    enabled=config.vla.enabled,
                    request_cap=config.vla.maximum_requests_per_run,
                    provider_requests_remaining=config.vla.maximum_requests_per_run,
                ),
                dry_run=True,
            )
        else:
            summary = _run_live_simulation(
                config,
                control_mode=control_mode,
                run_id=run_id,
                event_logger=event_logger,
                event_log_path=event_log_path,
                controller=controller,
                observation_builder=observation_builder,
                arbiter=arbiter,
            )
        _event(
            event_logger,
            "run_ended",
            sim_time_s=summary.simulation_time_s,
            payload={"summary": summary},
        )
        return summary
    except Exception as exc:
        _event(
            event_logger,
            "run_failed",
            sim_time_s=None,
            payload={
                "error_category": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    finally:
        if event_logger is not None:
            event_logger.close()


def _run_live_simulation(
    config: AppConfig,
    *,
    control_mode: str,
    run_id: str,
    event_logger: EventLogger | None,
    event_log_path: str | None,
    controller: VehicleController | None,
    observation_builder: ObservationBuilder | None,
    arbiter: AssessmentArbiter | None,
) -> RunSummary:
    perception = BasicPerception(prefer_info=config.perception.prefer_info)
    planner = WaypointPathPlanner(
        config.planner.default_route,
        lookahead_m=config.planner.lookahead_m,
        curvature_preview_m=config.planner.curvature_preview_m,
        maximum_lateral_acceleration_mps2=(
            config.planner.maximum_lateral_acceleration_mps2
        ),
        minimum_curve_speed_mps=config.planner.minimum_curve_speed_mps,
    )
    safety_supervisor = (
        EmergencyBrakingSupervisor(config.safety) if config.safety.enabled else None
    )
    headway_speed_cap = TimeHeadwaySpeedCap(
        mode=config.safety.headway_speed_cap_mode,
        minimum_gap_m=config.safety.minimum_gap_m,
        time_headway_s=config.safety.minimum_headway_s,
        scene_stale_after_s=config.safety.stale_after_s,
    )
    follow_speed_cap = TimeHeadwaySpeedCap(
        mode="enforce",
        minimum_gap_m=config.safety.minimum_gap_m,
        time_headway_s=config.safety.minimum_headway_s,
        scene_stale_after_s=config.safety.stale_after_s,
    )
    safety_scene_adapter: OracleSceneAdapter | LidarSceneAdapter | None = None
    future_path: PolylineFuturePath | None = None
    scheduler: VLAInferenceScheduler | None = None
    vla_runtime: VLACommandRuntime | None = None
    lane_change_coordinator = (
        LaneChangeCoordinator(
            timeout_s=config.planner.lane_change_timeout_s,
            completion_tolerance_m=config.planner.lane_change_completion_tolerance_m,
        )
        if config.planner.lane_change_enabled
        else None
    )
    lane_change_validator = (
        VLACommandValidator(config.command_validation)
        if lane_change_coordinator is not None
        else None
    )
    env = make_env(
        config.simulator,
        config.camera,
        record_episode=config.episode_recording.enabled,
    )
    pacer = (
        RealTimePacer(realtime_factor=config.simulator.realtime_factor)
        if config.simulator.realtime
        else None
    )

    total_reward = 0.0
    steps = 0
    crashed = False
    went_off_road = False
    safety_interventions = 0
    emergency_brakes = 0
    minimum_safety_gap_m: float | None = None
    headway_cap_evaluations = 0
    headway_cap_would_intervene = 0
    headway_cap_applied = 0
    lane_changes_started = 0
    lane_changes_completed = 0
    lane_changes_aborted = 0
    lane_change_hazards_spawned = 0
    lane_change_hazard_spawned = False
    fault_activations = 0
    controller_outputs_clamped = 0
    final_speed_mps = 0.0
    final_simulation_time_s = 0.0
    replay_artifact_path: str | None = None
    replay_artifact_sha256: str | None = None
    info: dict[str, object] = {}
    previous_safety_level: SafetyLevel | None = None
    previous_execution: CommandExecutionDecision | None = None
    previous_target_speed_mps: float | None = None
    previous_control_authority: str | None = None
    traffic_light_controller: TrafficLightController | None = None
    fault_schedule = FaultSchedule(config.faults)
    vla_metrics = _VLAMetricsCollector(enabled=config.vla.enabled)
    control_dt_s = control_timestep_s(config.simulator)
    try:
        observation, info = env.reset()
        if config.scenario.stopped_vehicle_ahead_m is not None:
            spawn_stopped_vehicle_ahead(env, config.scenario.stopped_vehicle_ahead_m)
        for stopped in config.scenario.stopped_vehicles:
            actor = spawn_static_vehicle_ahead(
                env,
                stopped.distance_m,
                lane_offset=stopped.lane_offset,
                lateral_m=stopped.lateral_m,
                vehicle_kind=stopped.kind,
                name=f"scenario-{stopped.vehicle_id}",
            )
            _event(
                event_logger,
                "scenario_vehicle_spawned",
                sim_time_s=0.0,
                payload={
                    "object_id": str(actor.id),
                    "vehicle_id": stopped.vehicle_id,
                    "distance_m": stopped.distance_m,
                    "lane_offset": stopped.lane_offset,
                    "lateral_m": stopped.lateral_m,
                    "speed_mps": 0.0,
                    "kind": stopped.kind,
                },
            )
        for scenario_vehicle in config.scenario.vehicles:
            actor = spawn_scenario_vehicle(env, scenario_vehicle)
            _event(
                event_logger,
                "scenario_vehicle_spawned",
                sim_time_s=0.0,
                payload={
                    "object_id": str(actor.id),
                    "vehicle_id": scenario_vehicle.vehicle_id,
                    "longitudinal_offset_m": (
                        scenario_vehicle.longitudinal_offset_m
                    ),
                    "lane_offset": scenario_vehicle.lane_offset,
                    "speed_mps": scenario_vehicle.speed_mps,
                    "kind": scenario_vehicle.kind,
                },
            )
        traffic_light_settings = config.scenario.traffic_light
        if traffic_light_settings.enabled:
            lights = spawn_traffic_light_group_ahead(
                env,
                traffic_light_settings.distance_m,
                visual_scale=traffic_light_settings.visual_scale,
            )
            traffic_light_controller = TrafficLightController(
                lights,
                TrafficLightCycle(
                    initial_state=TrafficLightState(
                        traffic_light_settings.initial_state
                    ),
                    red_duration_s=traffic_light_settings.red_duration_s,
                    green_duration_s=traffic_light_settings.green_duration_s,
                    yellow_duration_s=traffic_light_settings.yellow_duration_s,
                ),
            )
            initial_state = traffic_light_controller.update(0.0)
            assert initial_state is not None
            _event(
                event_logger,
                "traffic_light_state",
                sim_time_s=0.0,
                payload={
                    "light_ids": [str(light.id) for light in lights],
                    "state": initial_state.value,
                    "distance_ahead_m": traffic_light_settings.distance_m,
                },
            )
        if config.planner.route_source == "map":
            route = navigation_waypoints(env, config.planner.waypoint_spacing_m)
            planner.reset(route)
        else:
            route = planner.route
        future_path = PolylineFuturePath(route)

        if safety_supervisor is not None or config.vla.enabled:
            adapter_type = (
                OracleSceneAdapter
                if config.perception.safety_source == "oracle"
                else LidarSceneAdapter
            )
            safety_scene_adapter = adapter_type(
                detection_radius_m=config.perception.detection_radius_m,
                corridor_margin_m=config.perception.corridor_margin_m,
                future_path=future_path,
            )
        if config.vla.enabled:
            scheduler = build_vla_scheduler(
                config.vla,
                inject_faults=has_provider_faults(config.faults),
                observation_builder=observation_builder,
            )
            vla_metrics.bind_request_budget(scheduler.request_budget)
            vla_metrics.bind_pipeline(scheduler.pipeline)
            fault_provider = (
                scheduler.pipeline.provider
                if isinstance(
                    scheduler.pipeline.provider,
                    FaultInjectingModelProvider,
                )
                else None
            )
            vla_runtime = VLACommandRuntime(
                scheduler,
                build_high_level_safety_policy(config.command_validation),
                _vla_command_executor(config),
                run_id=run_id,
                fault_provider=fault_provider,
                arbiter=arbiter,
            )
        if pacer is not None:
            pacer.reset(simulation_time_s(env))

        for step_index in range(config.simulator.horizon):
            frame = None
            scene: LocalScene | None = None
            safety_decision: SafetyDecision | None = None
            execution: CommandExecutionDecision | None = None
            headway_cap_decision: HeadwaySpeedCapDecision | None = None
            headway_speed_cap_trigger: str | None = None
            lane_change_decision: LaneChangeDecision | None = None
            lane_change_clearance: LaneChangeClearanceDecision | None = None
            lane_change_context: dict[str, object] | None = None
            target_speed_mps = config.controller.target_speed_mps
            raw_target_speed_mps = target_speed_mps
            now_s = simulation_time_s(env)
            lane_change_hazard = config.scenario.lane_change_hazard
            target_lane_index = (
                lane_change_coordinator.target_lane_index
                if lane_change_coordinator is not None
                else None
            )
            if (
                lane_change_hazard.enabled
                and not lane_change_hazard_spawned
                and target_lane_index is not None
            ):
                current_lane_index, _, _ = lane_topology(env)
                lane_offset = target_lane_index - current_lane_index
                if lane_offset not in {-1, 0, 1}:
                    raise RuntimeError(
                        "active lane-change target is not current or adjacent"
                    )
                hazard = spawn_static_vehicle_ahead(
                    env,
                    lane_change_hazard.distance_ahead_m,
                    lane_offset=lane_offset,
                    vehicle_kind=lane_change_hazard.vehicle_kind,
                    name="scenario-lane-change-hazard",
                )
                lane_change_hazard_spawned = True
                lane_change_hazards_spawned += 1
                _event(
                    event_logger,
                    "lane_change_hazard_spawned",
                    sim_time_s=now_s,
                    payload={
                        "object_id": str(hazard.id),
                        "distance_ahead_m": lane_change_hazard.distance_ahead_m,
                        "vehicle_kind": lane_change_hazard.vehicle_kind,
                        "current_lane_index": current_lane_index,
                        "target_lane_index": target_lane_index,
                        "lane_offset": lane_offset,
                    },
                )
            if traffic_light_controller is not None:
                changed_state = traffic_light_controller.update(now_s)
                if changed_state is not None:
                    _event(
                        event_logger,
                        "traffic_light_state",
                        sim_time_s=now_s,
                        payload={
                            "light_ids": [
                                str(light.id)
                                for light in traffic_light_controller.lights
                            ],
                            "state": changed_state.value,
                            "distance_ahead_m": traffic_light_settings.distance_m,
                        },
                    )
            active_faults, fault_transitions = fault_schedule.advance(step_index)
            for transition in fault_transitions:
                _event(
                    event_logger,
                    "fault_activated" if transition.active else "fault_cleared",
                    sim_time_s=now_s,
                    payload={"step": step_index, "fault": transition.fault},
                )
                fault_activations += int(transition.active)

            if config.simulator.manual_control:
                action: Action = (0.0, 0.0)
            else:
                assert controller is not None
                frame = perception.observe(observation, ego_state_info(env, info))
                if safety_scene_adapter is not None:
                    scene = safety_scene_adapter.observe(env, timestamp_s=now_s)
                    if (
                        config.perception.safety_source == "lidar"
                        and has_fault(active_faults, FaultKind.LIDAR_DROPOUT)
                    ):
                        scene = replace(scene, valid=False, objects=())
                if vla_runtime is not None:
                    assert scene is not None
                    update = vla_runtime.update(
                        env,
                        scene,
                        now_s=now_s,
                        ego_speed_mps=frame.ego.speed_mps,
                        cruise_speed_mps=config.controller.target_speed_mps,
                        active_faults=active_faults,
                    )
                    execution = update.execution
                    target_speed_mps = execution.target_speed_mps
                    raw_target_speed_mps = target_speed_mps
                    vla_metrics.observe_update(update)
                    _log_vla_update(event_logger, update, scene=scene, now_s=now_s)
                    if execution != previous_execution:
                        _event(
                            event_logger,
                            "command_execution",
                            sim_time_s=now_s,
                            payload={
                                "now_s": now_s,
                                "validation_decision": update.active_validation,
                                "execution": execution,
                            },
                        )
                        previous_execution = execution

                    if execution.source is CommandExecutionSource.VLA:
                        active_speed_cap = (
                            follow_speed_cap
                            if execution.action is HighLevelAction.FOLLOW
                            else headway_speed_cap
                        )
                        headway_speed_cap_trigger = (
                            "follow_action"
                            if execution.action is HighLevelAction.FOLLOW
                            else "configured_mode"
                        )
                        headway_cap_decision = active_speed_cap.evaluate(
                            scene,
                            raw_target_speed_mps,
                            now_s=now_s,
                        )
                        target_speed_mps = (
                            headway_cap_decision.effective_target_speed_mps
                        )
                        headway_cap_evaluations += 1
                        headway_cap_would_intervene += int(
                            headway_cap_decision.would_intervene
                        )
                        headway_cap_applied += int(headway_cap_decision.applied)
                        if headway_cap_decision.would_intervene:
                            _event(
                                event_logger,
                                "headway_speed_cap",
                                sim_time_s=now_s,
                                payload={
                                    "command_id": execution.command_id,
                                    "decision": headway_cap_decision,
                                },
                            )

                if lane_change_coordinator is not None:
                    assert lane_change_validator is not None
                    input_target_lane_index = lane_change_coordinator.target_lane_index
                    current_lane_index, lane_count, target_lane_offset_m = lane_topology(
                        env,
                        target_lane_index=input_target_lane_index,
                    )
                    target_lane_clear: bool | None = None
                    clearance_reason: str | None = None
                    if input_target_lane_index is not None:
                        target_relation = (
                            LaneRelation.SAME
                            if input_target_lane_index == current_lane_index
                            else LaneRelation.LEFT
                            if input_target_lane_index < current_lane_index
                            else LaneRelation.RIGHT
                        )
                        if scene is None:
                            target_lane_clear = False
                            clearance_reason = (
                                "local scene is unavailable during lane change"
                            )
                        else:
                            lane_change_clearance = (
                                lane_change_validator.lane_change_clearance(
                                    scene,
                                    target_relation,
                                    now_s=now_s,
                                    include_planned_path=True,
                                )
                            )
                            target_lane_clear = lane_change_clearance.clear
                            clearance_reason = lane_change_clearance.reason
                    lane_change_decision = lane_change_coordinator.update(
                        execution,
                        now_s=now_s,
                        current_lane_index=current_lane_index,
                        lane_count=lane_count,
                        target_lane_offset_m=target_lane_offset_m,
                        target_lane_clear=target_lane_clear,
                        clearance_reason=clearance_reason,
                    )
                    forced_abort_reason: str | None = None
                    if lane_change_decision.phase is LaneChangePhase.STARTED:
                        assert lane_change_decision.source_lane_index is not None
                        assert lane_change_decision.target_lane_index is not None
                        try:
                            route = lane_change_waypoints(
                                env,
                                config.planner.waypoint_spacing_m,
                                lane_offset=(
                                    lane_change_decision.target_lane_index
                                    - lane_change_decision.source_lane_index
                                ),
                                transition_distance_m=(
                                    config.planner.lane_change_transition_m
                                ),
                            )
                        except (RuntimeError, ValueError) as exc:
                            forced_abort_reason = (
                                f"lane-change route unavailable: {exc}"
                            )
                            lane_change_decision = lane_change_coordinator.abort(
                                now_s=now_s,
                                reason=forced_abort_reason,
                            )
                            route = navigation_waypoints(
                                env,
                                config.planner.waypoint_spacing_m,
                            )
                        planner.reset(route)
                        assert future_path is not None
                        future_path.reset(route)

                    elif lane_change_decision.phase in {
                        LaneChangePhase.COMPLETED,
                        LaneChangePhase.ABORTED,
                    }:
                        route = navigation_waypoints(
                            env,
                            config.planner.waypoint_spacing_m,
                        )
                        planner.reset(route)
                        assert future_path is not None
                        future_path.reset(route)

                    lane_change_context = {
                        "current_lane_index": current_lane_index,
                        "lane_count": lane_count,
                        "target_lane_offset_m": target_lane_offset_m,
                        "clearance": lane_change_clearance,
                        "forced_abort_reason": forced_abort_reason,
                    }

                    if lane_change_decision.phase is not LaneChangePhase.IDLE:
                        lane_changes_started += int(
                            lane_change_decision.phase is LaneChangePhase.STARTED
                        )
                        lane_changes_completed += int(
                            lane_change_decision.phase is LaneChangePhase.COMPLETED
                        )
                        lane_changes_aborted += int(
                            lane_change_decision.phase is LaneChangePhase.ABORTED
                        )
                    if lane_change_decision.phase in {
                        LaneChangePhase.STARTED,
                        LaneChangePhase.COMPLETED,
                        LaneChangePhase.ABORTED,
                        LaneChangePhase.REJECTED,
                    }:
                        _event(
                            event_logger,
                            "lane_change_transition",
                            sim_time_s=now_s,
                            payload={
                                "decision": lane_change_decision,
                                "context": lane_change_context,
                            },
                        )

                plan = planner.plan(frame)
                requested_target_speed_mps = target_speed_mps
                if plan.route_speed_cap_mps is not None:
                    target_speed_mps = min(
                        target_speed_mps,
                        plan.route_speed_cap_mps,
                    )
                control_authority = (
                    execution.source.value if execution is not None else "local_autopilot"
                )
                pid_reset_reasons: list[str] = []
                if previous_target_speed_mps is not None and abs(
                    target_speed_mps - previous_target_speed_mps
                ) >= config.controller.speed_pid_reset_threshold_mps:
                    pid_reset_reasons.append("target speed changed substantially")
                if (
                    previous_control_authority is not None
                    and control_authority != previous_control_authority
                ):
                    pid_reset_reasons.append("control authority changed")
                if pid_reset_reasons:
                    controller.reset_speed_control()
                    _event(
                        event_logger,
                        "speed_pid_reset",
                        sim_time_s=now_s,
                        payload={"reasons": pid_reset_reasons},
                    )
                previous_target_speed_mps = target_speed_mps
                previous_control_authority = control_authority
                requested_action, action = _controller_action(
                    controller,
                    plan,
                    target_speed_mps=target_speed_mps,
                    speed_mps=frame.ego.speed_mps,
                    dt_s=control_dt_s,
                )
                if action != requested_action:
                    controller_outputs_clamped += 1
                    _event(
                        event_logger,
                        "controller_output_clamped",
                        sim_time_s=now_s,
                        payload={"requested": requested_action, "applied": action},
                    )
                proposed_command = _action_to_control_command(action)
                applied_command = proposed_command
                if safety_supervisor is not None and scene is not None:
                    safety_decision = safety_supervisor.evaluate(
                        scene,
                        proposed_command,
                        now_s=now_s,
                    )
                    applied_command = safety_decision.command
                    action = _control_command_to_action(applied_command)
                    safety_interventions += int(safety_decision.intervened)
                    emergency_brakes += int(
                        safety_decision.level
                        in {SafetyLevel.EMERGENCY, SafetyLevel.DEGRADED}
                    )
                    if safety_decision.minimum_gap_m is not None:
                        minimum_safety_gap_m = (
                            safety_decision.minimum_gap_m
                            if minimum_safety_gap_m is None
                            else min(
                                minimum_safety_gap_m,
                                safety_decision.minimum_gap_m,
                            )
                        )
                    if safety_decision.level != previous_safety_level:
                        _event(
                            event_logger,
                            "safety_transition",
                            sim_time_s=now_s,
                            payload={
                                "from_level": previous_safety_level,
                                "to_level": safety_decision.level,
                                "decision": safety_decision,
                            },
                        )
                        previous_safety_level = safety_decision.level
                    if safety_decision.intervened:
                        # The final safety layer replaced the PID output. Clear
                        # stored error every overridden tick so it cannot wind up
                        # behind the override or leak stale throttle on release.
                        controller.reset_speed_control()
                        _event(
                            event_logger,
                            "speed_pid_reset",
                            sim_time_s=now_s,
                            payload={
                                "reasons": ["local safety overrode PID output"],
                                "safety_level": safety_decision.level,
                            },
                        )
                        _event(
                            event_logger,
                            "safety_intervention",
                            sim_time_s=now_s,
                            payload={
                                "proposed_command": proposed_command,
                                "applied_command": applied_command,
                                "decision": safety_decision,
                            },
                        )

                _event(
                    event_logger,
                    "control_applied",
                    sim_time_s=now_s,
                    payload={
                        "now_s": now_s,
                        "ego": frame.ego,
                        "plan": plan,
                        "target_speed_mps": target_speed_mps,
                        "requested_target_speed_mps": requested_target_speed_mps,
                        "raw_target_speed_mps": raw_target_speed_mps,
                        "model_target_speed_mps": (
                            execution.action_speed.requested_target_speed_mps
                            if execution is not None
                            and execution.action_speed is not None
                            else raw_target_speed_mps
                        ),
                        "action_speed_policy": (
                            execution.action_speed
                            if execution is not None
                            else None
                        ),
                        "control_dt_s": control_dt_s,
                        "headway_speed_cap": headway_cap_decision,
                        "headway_speed_cap_trigger": headway_speed_cap_trigger,
                        "lane_change": lane_change_decision,
                        "lane_change_context": lane_change_context,
                        "execution": execution,
                        "proposed_command": proposed_command,
                        "applied_command": applied_command,
                        "scene": scene,
                        "safety_decision": safety_decision,
                    },
                )

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            crashed = crashed or bool(info.get("crash", False))
            went_off_road = went_off_road or bool(info.get("out_of_road", False))
            final_speed_mps = float(env.agent.speed)
            final_simulation_time_s = simulation_time_s(env)
            if not config.simulator.headless and not config.simulator.manual_control:
                assert frame is not None
                env.render(
                    {
                        "mode": (
                            "VLA + local validation + PID"
                            if config.vla.enabled
                            else "waypoint + PID autopilot"
                        ),
                        "target speed": _format_display_speed(
                            target_speed_mps,
                            config.display.speed_unit,
                        ),
                        "speed": _format_display_speed(
                            frame.ego.speed_mps,
                            config.display.speed_unit,
                        ),
                        "traffic light": (
                            traffic_light_controller.current_state.value
                            if traffic_light_controller is not None
                            and traffic_light_controller.current_state is not None
                            else "none"
                        ),
                        "steering": f"{action[0]:+.3f}",
                        "throttle/brake": f"{action[1]:+.3f}",
                        "VLA action": execution.action.value if execution is not None else "disabled",
                        "lane change": (
                            lane_change_decision.phase.value
                            if lane_change_decision is not None
                            else "disabled"
                        ),
                        "perception": config.perception.safety_source,
                        "safety": (
                            safety_decision.level.value
                            if safety_decision is not None
                            else "disabled"
                        ),
                    }
                )
            if pacer is not None:
                pacer.wait(final_simulation_time_s)
            if terminated or truncated:
                break
        if config.episode_recording.enabled:
            artifact = write_episode_artifact(
                config.episode_recording.path,
                env.engine.dump_episode(),
                config=config.to_dict(),
                run_id=run_id,
                scenario_id=config.event_log.scenario_id,
                recorded_steps=steps,
                simulation_time_s=final_simulation_time_s,
            )
            replay_artifact_path = artifact.artifact_dir
            replay_artifact_sha256 = artifact.payload_sha256
            _event(
                event_logger,
                "episode_artifact_written",
                sim_time_s=final_simulation_time_s,
                payload={"artifact": artifact},
            )
    finally:
        try:
            if scheduler is not None:
                scheduler.close()
                final_completion = scheduler.poll()
                if final_completion is not None:
                    vla_metrics.observe_completion(
                        final_completion,
                        discarded=True,
                    )
                    _log_inference_completion(
                        event_logger,
                        final_completion,
                        now_s=final_simulation_time_s,
                        disposition="discarded_simulation_ended",
                    )
        finally:
            env.close()

    return RunSummary(
        steps=steps,
        total_reward=total_reward,
        control_mode=control_mode,
        controller=_qualified_name(controller),
        observation=_qualified_name(observation_builder),
        arbiter=_qualified_name(arbiter),
        controller_outputs_clamped=controller_outputs_clamped,
        route_completion=float(info.get("route_completion", 0.0)),
        arrived=bool(info.get("arrive_dest", False)),
        crashed=crashed,
        went_off_road=went_off_road,
        safety_interventions=safety_interventions,
        emergency_brakes=emergency_brakes,
        minimum_safety_gap_m=minimum_safety_gap_m,
        headway_cap_evaluations=headway_cap_evaluations,
        headway_cap_would_intervene=headway_cap_would_intervene,
        headway_cap_applied=headway_cap_applied,
        lane_changes_started=lane_changes_started,
        lane_changes_completed=lane_changes_completed,
        lane_changes_aborted=lane_changes_aborted,
        lane_change_hazards_spawned=lane_change_hazards_spawned,
        fault_activations=fault_activations,
        final_speed_mps=final_speed_mps,
        simulation_time_s=final_simulation_time_s,
        run_id=run_id,
        event_log_path=event_log_path,
        replay_artifact_path=replay_artifact_path,
        replay_artifact_sha256=replay_artifact_sha256,
        vla_metrics=vla_metrics.finish(),
    )


def _dry_run_wiring(
    config: AppConfig,
    controller: VehicleController | None,
) -> int:
    """Exercise one tick of wiring; return 1 if the action had to be clamped."""
    perception = BasicPerception(prefer_info=config.perception.prefer_info)
    planner = WaypointPathPlanner(
        config.planner.default_route,
        lookahead_m=config.planner.lookahead_m,
        curvature_preview_m=config.planner.curvature_preview_m,
        maximum_lateral_acceleration_mps2=(
            config.planner.maximum_lateral_acceleration_mps2
        ),
        minimum_curve_speed_mps=config.planner.minimum_curve_speed_mps,
    )
    TimeHeadwaySpeedCap(
        mode=config.safety.headway_speed_cap_mode,
        minimum_gap_m=config.safety.minimum_gap_m,
        time_headway_s=config.safety.minimum_headway_s,
        scene_stale_after_s=config.safety.stale_after_s,
    )
    frame = perception.observe(
        {"lidar": [1.0, 0.5, 0.25]},
        {"position": (0.0, 0.0), "heading": 0.0, "speed_mps": 12.0},
    )
    plan = planner.plan(frame)
    target_speed_mps = config.controller.target_speed_mps
    if config.vla.enabled:
        executor = _vla_command_executor(config)
        target_speed_mps = executor.execute(None, now_s=0.0).target_speed_mps
        build_high_level_safety_policy(config.command_validation)
    if controller is None:
        return 0
    requested, applied = _controller_action(
        controller,
        plan,
        target_speed_mps=target_speed_mps,
        speed_mps=frame.ego.speed_mps,
        dt_s=control_timestep_s(config.simulator),
    )
    return int(applied != requested)


def _build_controller(
    config: AppConfig,
    factory: VehicleControllerFactory | None,
) -> VehicleController | None:
    """Build the controller that drives, or None under manual control.

    Without an explicit factory the instructor reference drives.
    """
    if config.simulator.manual_control:
        return None
    return build_vehicle_controller(
        config.controller,
        factory=load_reference_controller() if factory is None else factory,
    )


def _build_observation_builder(
    config: AppConfig,
    factory: ObservationBuilderFactory | None,
) -> ObservationBuilder | None:
    """Build what assembles each model observation, or None without the VLA subsystem.

    Without an explicit factory DriveBench builds its original observation,
    which reads no observation settings, so it refuses any that were changed
    rather than ignore them.
    """
    if not config.vla.enabled:
        return None
    if factory is None:
        if config.observation != ObservationSettings():
            raise ValueError(
                "observation settings are read by a submitted observation.py; "
                "run with --submission, or leave the observation section at its defaults"
            )
        return DefaultObservationBuilder()
    return build_observation_builder(config.observation, factory=factory)


def _build_arbiter(
    config: AppConfig,
    factory: AssessmentArbiterFactory | None,
) -> AssessmentArbiter | None:
    """Build what arbitrates each assessment, or None without the VLA subsystem.

    Without an explicit factory every assessment is endorsed, as before arbitration.
    """
    if not config.vla.enabled:
        return None
    if factory is None:
        return DefaultArbiter()
    return build_arbiter(config.command_validation, factory=factory)


def _qualified_name(component: object | None) -> str | None:
    if component is None:
        return None
    component_type = type(component)
    return f"{component_type.__module__}.{component_type.__qualname__}"


def _vla_command_executor(config: AppConfig) -> VLACommandExecutor:
    return VLACommandExecutor(
        config.controller.target_speed_mps,
        action_speed_policy=ActionSpeedPolicy(
            config.vla.action_speed_policy_mode,
            cruise_speed_mps=config.controller.target_speed_mps,
            slow_down_speed_mps=config.command_validation.slow_down_speed_mps,
            yield_speed_mps=config.command_validation.yield_speed_mps,
        ),
        lane_change_enabled=config.planner.lane_change_enabled,
    )


def _validate_runtime_configuration(config: AppConfig) -> None:
    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not config.vla.enabled:
        return
    if config.simulator.manual_control:
        raise ValueError("vla.enabled cannot be combined with manual control")
    if not config.camera.enabled:
        raise ValueError("camera.enabled must be true when vla.enabled is used by run")
    if not config.safety.enabled:
        raise ValueError("safety.enabled must be true when vla.enabled is used by run")


def _log_vla_update(
    logger: EventLogger | None,
    update: VLAControlUpdate,
    *,
    scene: LocalScene,
    now_s: float,
) -> None:
    completion = update.completion
    if isinstance(completion, InferenceSuccess):
        result = completion.result
        _log_inference_completion(logger, completion, now_s=now_s)
        assert update.validation is not None
        _event(
            logger,
            "command_validation",
            sim_time_s=now_s,
            payload={
                "now_s": now_s,
                # The model's assessment, and what arbitration left of it.
                "assessment": result.assessment,
                "arbitration": update.arbitration,
                # The command the safety floor validated, derived from the latter.
                "requested_command": update.validation.requested_command,
                "scene": scene,
                "decision": update.validation,
            },
        )
    elif isinstance(completion, (InferenceFailure, InferenceDiscarded)):
        _log_inference_completion(logger, completion, now_s=now_s)

    review = update.review
    if review is not None and (review.revoked or review.problem is not None):
        _event(
            logger,
            "arbitration_review",
            sim_time_s=now_s,
            payload={"now_s": now_s, "review": review},
        )

    if update.submission_disposition is InferenceSubmitDisposition.STARTED:
        _event(
            logger,
            "inference_submitted",
            sim_time_s=now_s,
            payload={
                "request_id": update.submitted_request_id,
                "episode_id": update.submitted_episode_id,
                "generation_id": update.submitted_generation_id,
                "active_fault_ids": update.active_fault_ids,
                "prompt_contract_version": VLA_PROMPT_CONTRACT_VERSION,
            },
        )
    elif update.submission_disposition is InferenceSubmitDisposition.CLOSED:
        _event(logger, "inference_scheduler_closed", sim_time_s=now_s)
    elif (
        update.submission_disposition
        is InferenceSubmitDisposition.REQUEST_CAP_EXHAUSTED
    ):
        _event(logger, "inference_request_cap_exhausted", sim_time_s=now_s)
    if update.runtime_error is not None:
        _event(
            logger,
            "vla_runtime_error",
            sim_time_s=now_s,
            payload={
                "error_category": type(update.runtime_error).__name__,
                "error_message": str(update.runtime_error),
            },
        )


def _log_inference_completion(
    logger: EventLogger | None,
    completion: InferenceCompletion,
    *,
    now_s: float,
    disposition: str | None = None,
) -> None:
    if isinstance(completion, InferenceSuccess):
        result = completion.result
        payload: dict[str, object] = {
            "request_id": completion.request_id,
            "episode_id": completion.episode_id,
            "generation_id": completion.generation_id,
            "submitted_at_s": completion.submitted_at_s,
            "model_id": result.response.model_id,
            "latency_s": result.response.latency_s,
            "provider_metadata": result.response.metadata,
            "prompt_contract_version": result.request.prompt_contract_version,
            "prompt_sha256": hashlib.sha256(
                result.request.prompt.encode("utf-8")
            ).hexdigest(),
            "rgb_sha256": hashlib.sha256(result.request.frame.rgb_bytes).hexdigest(),
            "assessment": result.assessment,
            "requested_command": result.requested_command,
        }
        event_type = "inference_completed"
    elif isinstance(completion, InferenceFailure):
        payload = {
            "request_id": completion.request_id,
            "episode_id": completion.episode_id,
            "generation_id": completion.generation_id,
            "submitted_at_s": completion.submitted_at_s,
            "error_category": type(completion.error).__name__,
            "error_message": str(completion.error),
        }
        event_type = "inference_failed"
    elif isinstance(completion, InferenceDiscarded):
        payload = {
            "request_id": completion.request_id,
            "episode_id": completion.episode_id,
            "generation_id": completion.generation_id,
            "submitted_at_s": completion.submitted_at_s,
            "reason": completion.reason,
        }
        if completion.result is not None:
            result = completion.result
            payload.update(
                {
                    "model_id": result.response.model_id,
                    "latency_s": result.response.latency_s,
                    "provider_metadata": result.response.metadata,
                    "prompt_contract_version": result.request.prompt_contract_version,
                    "prompt_sha256": hashlib.sha256(
                        result.request.prompt.encode("utf-8")
                    ).hexdigest(),
                    "rgb_sha256": hashlib.sha256(
                        result.request.frame.rgb_bytes
                    ).hexdigest(),
                    "assessment": result.assessment,
                    "requested_command": result.requested_command,
                }
            )
        if completion.error is not None:
            payload.update(
                {
                    "error_category": type(completion.error).__name__,
                    "error_message": str(completion.error),
                }
            )
        event_type = "inference_discarded"
    else:
        raise TypeError("completion must be a success, failure, or discard")
    payload.setdefault("prompt_contract_version", VLA_PROMPT_CONTRACT_VERSION)
    if disposition is not None:
        payload["disposition"] = disposition
    _event(logger, event_type, sim_time_s=now_s, payload=payload)


def _controller_action(
    controller: VehicleController,
    plan: Plan,
    *,
    target_speed_mps: float,
    speed_mps: float,
    dt_s: float,
) -> tuple[Action, Action]:
    """Return the controller's requested action and the saturated one to apply."""
    requested = checked_action(
        controller.update(
            ControlTick(
                target_speed_mps=target_speed_mps,
                speed_mps=speed_mps,
                heading_error_rad=plan.heading_error_rad,
                lateral_error_m=plan.lateral_error_m,
                dt_s=dt_s,
            )
        )
    )
    return requested, saturated_action(requested)


def _format_display_speed(speed_mps: float, unit: str) -> str:
    if unit == "mps":
        return f"{speed_mps:.1f} m/s"
    if unit == "kph":
        return f"{mps_to_kmh(speed_mps):.1f} km/h"
    raise ValueError("display speed unit must be 'mps' or 'kph'")


def _action_to_control_command(action: Action) -> ControlCommand:
    longitudinal = float(action[1])
    return ControlCommand(
        steering=float(action[0]),
        throttle=max(0.0, longitudinal),
        brake=max(0.0, -longitudinal),
    )


def _control_command_to_action(command: ControlCommand) -> Action:
    longitudinal = command.throttle if command.throttle > 0.0 else -command.brake
    return command.steering, longitudinal


def _event(
    logger: EventLogger | None,
    event_type: str,
    *,
    sim_time_s: float | None,
    payload: dict[str, object] | None = None,
) -> None:
    if logger is not None:
        logger.write(event_type, sim_time_s=sim_time_s, payload=payload)
