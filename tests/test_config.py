import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

import metadrive_starter.config as config_module
from metadrive_starter.config import (
    ControllerSettings,
    PIDSettings,
    PlannerSettings,
    apply_overrides,
    config_from_dict,
    load_config,
)
from metadrive_starter.events import to_json_value


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_config_loads_nested_pid_settings() -> None:
    config = config_from_dict(
        {
            "controller": {
                "target_speed_mps": 20.0,
                "speed_pid_reset_threshold_mps": 1.5,
                "speed_pid": {"kp": 0.1, "ki": 0.2, "kd": 0.3, "anti_windup": False},
            },
            "planner": {"route_source": "waypoints", "lookahead_m": 8.0, "waypoint_spacing_m": 1.0},
        }
    )

    assert config.controller.target_speed_mps == 20.0
    assert config.controller.speed_pid.kp == 0.1
    assert config.controller.speed_pid.anti_windup is False
    assert config.controller.speed_pid_reset_threshold_mps == 1.5
    assert config.controller.steering_pid.kp == 1.0
    assert config.controller.lateral_pid.kp == 0.35
    assert config.planner.route_source == "waypoints"
    assert config.planner.lookahead_m == 8.0
    assert config.planner.waypoint_spacing_m == 1.0
    assert config.planner.curvature_preview_m == 40.0
    assert config.planner.maximum_lateral_acceleration_mps2 == 2.0
    assert config.planner.minimum_curve_speed_mps == 4.0


@pytest.mark.parametrize(
    "data",
    [
        {"controller": {"target_speed_kmh": 36.0}},
        {"controller": {"speed_pid_reset_threshold_kmh": 3.6}},
        {"command_validation": {"maximum_target_speed_kmh": 54.0}},
    ],
)
def test_config_rejects_non_display_kmh_speed_fields(data: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="not supported.*_mps"):
        config_from_dict(data)


def test_cli_style_overrides() -> None:
    config = config_from_dict({})

    apply_overrides(
        config,
        headless=False,
        manual_control=True,
        realtime=True,
        realtime_factor=0.5,
        out_of_road_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        steps=42,
        traffic_density=0.4,
        obstacle_probability=0.25,
        map_name="C",
        headway_speed_cap_mode="shadow",
        action_speed_policy_mode="enforce",
    )

    assert config.simulator.headless is False
    assert config.simulator.manual_control is True
    assert config.simulator.realtime is True
    assert config.simulator.realtime_factor == 0.5
    assert config.simulator.out_of_road_done is False
    assert config.simulator.crash_vehicle_done is False
    assert config.simulator.crash_object_done is False
    assert config.simulator.horizon == 42
    assert config.simulator.traffic_density == 0.4
    assert config.simulator.obstacle_probability == 0.25
    assert config.simulator.map == "C"
    assert config.safety.headway_speed_cap_mode == "shadow"
    assert config.vla.action_speed_policy_mode == "enforce"


def test_simulator_traffic_generation_mode_is_explicit() -> None:
    config = config_from_dict(
        {"simulator": {"random_traffic": True, "traffic_mode": "respawn"}}
    )

    assert config.simulator.random_traffic is True
    assert config.simulator.traffic_mode == "respawn"


@pytest.mark.parametrize(
    "value",
    ["random", 1, None],
)
def test_config_rejects_unknown_traffic_mode(value: object) -> None:
    with pytest.raises(ValueError, match="traffic_mode"):
        config_from_dict({"simulator": {"traffic_mode": value}})


def test_config_rejects_non_boolean_random_traffic() -> None:
    with pytest.raises(ValueError, match="random_traffic"):
        config_from_dict({"simulator": {"random_traffic": 1}})


def test_episode_recording_config_is_typed_and_disabled_by_default() -> None:
    default = config_from_dict({})
    enabled = config_from_dict(
        {"episode_recording": {"enabled": True, "path": "tmp/replay-1"}}
    )

    assert default.episode_recording.enabled is False
    assert enabled.episode_recording.enabled is True
    assert enabled.episode_recording.path == "tmp/replay-1"


@pytest.mark.parametrize("value", [1, "yes", None])
def test_episode_recording_rejects_non_boolean_enabled(value: object) -> None:
    with pytest.raises(ValueError, match="episode_recording.enabled"):
        config_from_dict({"episode_recording": {"enabled": value}})


def test_config_accepts_lidar_as_safety_source() -> None:
    config = config_from_dict({"perception": {"safety_source": "lidar"}})

    assert config.perception.safety_source == "lidar"


def test_lidar_safety_source_requires_enabled_lidar() -> None:
    with pytest.raises(ValueError, match="use_lidar"):
        config_from_dict(
            {
                "simulator": {"use_lidar": False},
                "perception": {"safety_source": "lidar"},
            }
        )


def test_config_rejects_unknown_safety_source() -> None:
    with pytest.raises(ValueError, match="oracle.*lidar"):
        config_from_dict({"perception": {"safety_source": "camera"}})


def test_config_loads_command_validation_settings() -> None:
    config = config_from_dict(
        {
            "command_validation": {
                "minimum_confidence": 0.8,
                "maximum_target_speed_mps": 30.0,
                "lane_change_minimum_ttc_s": 4.5,
            }
        }
    )

    assert config.command_validation.minimum_confidence == 0.8
    assert config.command_validation.maximum_target_speed_mps == 30.0
    assert config.command_validation.lane_change_minimum_ttc_s == 4.5
    assert config.command_validation.lane_change_front_gap_m == 12.0


def test_config_rejects_invalid_command_validation_settings() -> None:
    with pytest.raises(ValueError, match="minimum_confidence"):
        config_from_dict({"command_validation": {"minimum_confidence": 1.1}})


@pytest.mark.parametrize("mode", ["off", "shadow", "enforce"])
def test_config_accepts_headway_speed_cap_modes(mode: str) -> None:
    config = config_from_dict(
        {"safety": {"enabled": True, "headway_speed_cap_mode": mode}}
    )

    assert config.safety.headway_speed_cap_mode == mode


def test_config_rejects_unknown_headway_speed_cap_mode() -> None:
    with pytest.raises(ValueError, match="headway_speed_cap_mode"):
        config_from_dict({"safety": {"headway_speed_cap_mode": "sometimes"}})


def test_config_rejects_non_boolean_pid_anti_windup() -> None:
    with pytest.raises(ValueError, match="anti_windup"):
        config_from_dict(
            {"controller": {"speed_pid": {"anti_windup": "sometimes"}}}
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_min": 1.0, "output_max": 1.0},
        {"kp": float("nan")},
        {"anti_windup": "yes"},
    ],
)
def test_pid_settings_reject_invalid_gains(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {"kp": 1.0, "ki": 0.0, "kd": 0.0}
    values.update(kwargs)
    with pytest.raises(ValueError):
        PIDSettings(**values)  # type: ignore[arg-type]


def test_default_yaml_exposes_command_validation_settings() -> None:
    config = load_config(PROJECT_ROOT / "configs/default.yaml")

    assert config.command_validation.minimum_confidence == 0.5
    assert config.command_validation.maximum_command_age_s == 2.0
    assert config.command_validation.lane_change_front_gap_m == 12.0
    assert config.controller.target_speed_mps == pytest.approx(35.0 / 3.6)
    assert config.command_validation.maximum_target_speed_mps == pytest.approx(50.0 / 3.6)
    assert config.controller.speed_pid.kp == pytest.approx(0.12)
    assert config.controller.speed_pid.anti_windup is True
    assert config.controller.speed_pid_reset_threshold_mps == 1.0
    assert config.simulator.realtime is False
    assert config.simulator.realtime_factor == 1.0
    assert config.simulator.physics_world_step_size == 0.02
    assert config.safety.headway_speed_cap_mode == "off"
    assert config.vla.action_speed_policy_mode == "off"
    assert config.vla.maximum_requests_per_run is None
    assert config.planner.lane_change_enabled is False
    assert config.scenario.traffic_light.enabled is False
    assert config.scenario.traffic_light.distance_m == 38.0
    assert config.scenario.lane_change_hazard.enabled is False
    assert config.scenario.lane_change_hazard.distance_ahead_m == 6.0
    assert config.faults == ()


def test_qwen_lane_choice_profile_enables_local_action_and_lane_policies() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/demo-vla-qwen3-vl-2b-lane-choice.yaml"
    )

    assert config.vla.action_speed_policy_mode == "enforce"
    assert config.planner.lane_change_enabled is True
    assert config.planner.lane_change_transition_m == 18.0
    assert config.scenario.stopped_vehicle_ahead_m == 28.0


def test_qwen_lane_hazard_profile_enables_dynamic_safety_fixture() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/demo-vla-qwen3-vl-2b-lane-hazard.yaml"
    )

    assert config.planner.lane_change_enabled is True
    assert config.scenario.lane_change_hazard.enabled is True
    assert config.scenario.lane_change_hazard.distance_ahead_m == 6.0
    assert config.event_log.scenario_id == "qwen3-vl-lane-hazard"


def test_config_rejects_lane_changes_for_fixed_waypoint_route() -> None:
    with pytest.raises(ValueError, match="route_source 'map'"):
        config_from_dict(
            {
                "planner": {
                    "route_source": "waypoints",
                    "lane_change_enabled": True,
                }
            }
        )


def test_config_rejects_unknown_action_speed_policy_mode() -> None:
    with pytest.raises(ValueError, match="action_speed_policy_mode"):
        config_from_dict({"vla": {"action_speed_policy_mode": "sometimes"}})


def test_config_loads_traffic_light_scenario() -> None:
    config = config_from_dict(
        {
            "scenario": {
                "traffic_light": {
                    "enabled": True,
                    "distance_m": 30.0,
                    "visual_scale": 1.5,
                    "initial_state": "green",
                    "red_duration_s": 9.0,
                    "green_duration_s": 8.0,
                    "yellow_duration_s": 1.5,
                }
            }
        }
    )

    assert config.scenario.traffic_light.enabled is True
    assert config.scenario.traffic_light.distance_m == 30.0
    assert config.scenario.traffic_light.visual_scale == 1.5
    assert config.scenario.traffic_light.initial_state == "green"


def test_config_rejects_unknown_traffic_light_state() -> None:
    with pytest.raises(ValueError, match="initial_state"):
        config_from_dict({"scenario": {"traffic_light": {"initial_state": "blue"}}})


def test_config_loads_lane_change_hazard_scenario() -> None:
    config = config_from_dict(
        {
            "planner": {"lane_change_enabled": True},
            "scenario": {
                "lane_change_hazard": {
                    "enabled": True,
                    "distance_ahead_m": 7.5,
                    "vehicle_kind": "truck",
                }
            },
        }
    )

    assert config.scenario.lane_change_hazard.enabled is True
    assert config.scenario.lane_change_hazard.distance_ahead_m == 7.5
    assert config.scenario.lane_change_hazard.vehicle_kind == "truck"


def test_config_loads_deterministic_relative_scenario_vehicle() -> None:
    config = config_from_dict(
        {
            "simulator": {"spawn_longitude_m": 30.0},
            "scenario": {
                "vehicles": [
                    {
                        "id": "fast-rear-left",
                        "longitudinal_offset_m": -18.0,
                        "lane_offset": 1,
                        "speed_mps": 12.0,
                        "kind": "car",
                    }
                ]
            },
        }
    )

    assert config.simulator.spawn_longitude_m == 30.0
    assert len(config.scenario.vehicles) == 1
    vehicle = config.scenario.vehicles[0]
    assert vehicle.vehicle_id == "fast-rear-left"
    assert vehicle.longitudinal_offset_m == -18.0
    assert vehicle.lane_offset == 1
    assert vehicle.speed_mps == 12.0
    assert vehicle.kind == "car"


def test_config_with_scenario_vehicles_round_trips_through_logged_form() -> None:
    config = config_from_dict(
        {
            "scenario": {
                "vehicles": [
                    {"id": "fast-rear-left", "longitudinal_offset_m": -18.0},
                    {
                        "id": "slow-ahead",
                        "longitudinal_offset_m": 25.0,
                        "lane_offset": -1,
                        "speed_mps": 4.0,
                        "kind": "truck",
                    },
                ]
            }
        }
    )

    logged = json.loads(json.dumps(to_json_value(config.to_dict())))

    assert config_from_dict(logged) == config


@pytest.mark.parametrize(
    "vehicles",
    [
        [42],
        [{"longitudinal_offset_m": -10.0}],
        [{"id": "actor"}],
        [{"id": "actor", "longitudinal_offset_m": 10.0, "extra": True}],
        [{"id": "actor", "vehicle_id": "actor", "longitudinal_offset_m": 10.0}],
    ],
)
def test_config_rejects_malformed_relative_scenario_vehicles(vehicles) -> None:
    with pytest.raises(ValueError, match="scenario.vehicles"):
        config_from_dict({"scenario": {"vehicles": vehicles}})


def test_lane_change_hazard_requires_lane_change_planner() -> None:
    with pytest.raises(ValueError, match="lane_change_enabled"):
        config_from_dict(
            {"scenario": {"lane_change_hazard": {"enabled": True}}}
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": "yes"},
        {"distance_ahead_m": 0.0},
        {"distance_ahead_m": float("nan")},
        {"vehicle_kind": "bus"},
    ],
)
def test_config_rejects_invalid_lane_change_hazard_settings(
    settings: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="lane_change_hazard"):
        config_from_dict({"scenario": {"lane_change_hazard": settings}})


def test_config_loads_scheduled_faults() -> None:
    config = config_from_dict(
        {
            "faults": [
                {
                    "fault_id": "timeout-1",
                    "kind": "provider_timeout",
                    "start_step": 4,
                    "duration_steps": 2,
                }
            ]
        }
    )

    assert config.faults[0].fault_id == "timeout-1"
    assert config.faults[0].end_step == 6


def test_config_rejects_duplicate_fault_ids() -> None:
    fault = {
        "fault_id": "duplicate",
        "kind": "camera_dropout",
        "start_step": 0,
        "duration_steps": 1,
    }
    with pytest.raises(ValueError, match="unique"):
        config_from_dict({"faults": [fault, fault]})


@pytest.mark.parametrize("value", [0, -0.01, float("nan"), True])
def test_config_rejects_invalid_physics_world_step_size(value: object) -> None:
    with pytest.raises(ValueError, match="physics_world_step_size"):
        config_from_dict({"simulator": {"physics_world_step_size": value}})


def test_config_loads_camera_settings() -> None:
    config = config_from_dict(
        {"camera": {"enabled": True, "width": 640, "height": 360}}
    )

    assert config.camera.enabled is True
    assert config.camera.width == 640
    assert config.camera.height == 360


def test_display_speed_unit_defaults_to_mps_and_accepts_kph() -> None:
    assert config_from_dict({}).display.speed_unit == "mps"
    assert config_from_dict({"display": {"speed_unit": "kph"}}).display.speed_unit == "kph"


def test_config_rejects_unknown_display_speed_unit() -> None:
    with pytest.raises(ValueError, match="display.speed_unit"):
        config_from_dict({"display": {"speed_unit": "mph"}})


def test_config_loads_vla_http_and_scheduler_settings() -> None:
    config = config_from_dict(
        {
            "vla": {
                "enabled": True,
                "prompt_policy": "Prefer smooth, safe progress.",
                "request_timeout_s": 3.0,
                "minimum_interval_s": 0.25,
                "action_horizon_s": 1.5,
                "max_scene_objects": 4,
                "maximum_frame_age_s": 0.2,
                "maximum_clock_skew_s": 0.01,
                "http": {
                    "endpoint_url": "https://models.example/v1/chat/completions",
                    "model_id": "demo-vlm",
                    "api_key_env": "DEMO_VLM_API_KEY",
                    "max_response_bytes": 4096,
                    "max_tokens": 128,
                },
            }
        }
    )

    assert config.vla.enabled is True
    assert config.vla.prompt_policy == "Prefer smooth, safe progress."
    assert config.vla.request_timeout_s == 3.0
    assert config.vla.minimum_interval_s == 0.25
    assert config.vla.action_horizon_s == 1.5
    assert config.vla.max_scene_objects == 4
    assert config.vla.http.endpoint_url.startswith("https://models.example")
    assert config.vla.http.model_id == "demo-vlm"
    assert config.vla.http.api_key_env == "DEMO_VLM_API_KEY"
    assert config.vla.http.max_response_bytes == 4096
    assert config.vla.http.max_tokens == 128


def test_config_loads_vertex_provider_settings() -> None:
    config = config_from_dict(
        {
            "vla": {
                "enabled": True,
                "provider": "vertex",
                "vertex": {
                    "project_env": "COURSE_GCP_PROJECT",
                    "location": "europe-west1",
                    "model_id": "gemini-course-model",
                    "temperature": 0.2,
                    "max_output_tokens": 512,
                    "thinking_budget": None,
                },
            }
        }
    )

    assert config.vla.provider == "vertex"
    assert config.vla.vertex.project_env == "COURSE_GCP_PROJECT"
    assert config.vla.vertex.location == "europe-west1"
    assert config.vla.vertex.model_id == "gemini-course-model"
    assert config.vla.vertex.temperature == 0.2
    assert config.vla.vertex.max_output_tokens == 512
    assert config.vla.vertex.thinking_budget is None


def test_config_loads_fixture_provider_settings() -> None:
    config = config_from_dict(
        {
            "vla": {
                "enabled": True,
                "provider": "fixture",
                "fixture": {
                    "path": "fixtures/vla/providers/synthetic-keep-lane.json",
                    "repeat_last": True,
                },
            }
        }
    )

    assert config.vla.provider == "fixture"
    assert config.vla.fixture.path.endswith("synthetic-keep-lane.json")
    assert config.vla.fixture.repeat_last is True


def test_default_yaml_exposes_disabled_local_vla_settings() -> None:
    config = load_config(PROJECT_ROOT / "configs/default.yaml")

    assert config.vla.enabled is False
    assert config.vla.minimum_interval_s == 0.5
    assert config.vla.http.endpoint_url.startswith("http://127.0.0.1:")
    assert config.vla.http.api_key_env is None
    assert config.vla.http.structured_output is False
    assert "Maintain safe forward progress" in config.vla.prompt_policy


def test_probe_local_yaml_enables_camera_and_http_inference() -> None:
    config = load_config(PROJECT_ROOT / "configs/vla-probe-local.yaml")

    assert config.camera.enabled is True
    assert (config.camera.width, config.camera.height) == (512, 288)
    assert config.vla.enabled is True
    assert config.vla.http.endpoint_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert config.vla.http.model_id == "local-vlm"


def test_fixture_demo_yaml_is_offline_and_repeats_checked_fixture() -> None:
    config = load_config(PROJECT_ROOT / "configs/demo-vla-fixture.yaml")

    assert config.camera.enabled is True
    assert config.vla.enabled is True
    assert config.vla.provider == "fixture"
    assert config.vla.fixture.path.endswith("synthetic-keep-lane.json")
    assert config.vla.fixture.repeat_last is True
    assert config.safety.enabled is True


def test_probe_smolvlm_mlx_yaml_targets_verified_loopback_server() -> None:
    config = load_config(PROJECT_ROOT / "configs/vla-probe-smolvlm-mlx.yaml")

    assert config.camera.enabled is True
    assert (config.camera.width, config.camera.height) == (512, 288)
    assert config.vla.enabled is True
    assert config.vla.request_timeout_s == 60.0
    assert config.vla.http.endpoint_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert config.vla.http.model_id == "mlx-community/SmolVLM-256M-Instruct-4bit"


def test_probe_qwen3_vl_mlx_yaml_targets_verified_loopback_server() -> None:
    config = load_config(PROJECT_ROOT / "configs/vla-probe-qwen3-vl-2b-mlx.yaml")

    assert config.camera.enabled is True
    assert (config.camera.width, config.camera.height) == (512, 288)
    assert config.vla.enabled is True
    assert config.vla.request_timeout_s == 60.0
    assert config.vla.http.endpoint_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert config.vla.http.model_id == "mlx-community/Qwen3-VL-2B-Instruct-4bit"
    assert config.vla.http.structured_output is False


def test_probe_qwen3_vl_structured_yaml_enforces_json_schema() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/vla-probe-qwen3-vl-2b-mlx-structured.yaml"
    )

    assert config.vla.enabled is True
    assert config.vla.http.model_id == "mlx-community/Qwen3-VL-2B-Instruct-4bit"
    assert config.vla.http.structured_output is True


@pytest.mark.parametrize(
    ("mlx", "llama"),
    [
        ("vla-probe-smolvlm-mlx.yaml", "vla-probe-smolvlm-llama.yaml"),
        ("vla-probe-qwen3-vl-2b-mlx.yaml", "vla-probe-qwen3-vl-2b-llama.yaml"),
        (
            "vla-probe-qwen3-vl-2b-mlx-structured.yaml",
            "vla-probe-qwen3-vl-2b-llama-structured.yaml",
        ),
        ("demo-vla-qwen3-vl-2b-mlx.yaml", "demo-vla-qwen3-vl-2b-llama.yaml"),
    ],
)
def test_mlx_profiles_sample_and_ask_as_the_llama_server_ones(mlx: str, llama: str) -> None:
    mlx_config = load_config(PROJECT_ROOT / "configs" / mlx)
    llama_config = load_config(PROJECT_ROOT / "configs" / llama)

    assert mlx_config.vla.http.temperature == llama_config.vla.http.temperature == 0.7
    assert mlx_config.vla.action_horizon_s == llama_config.vla.action_horizon_s
    assert mlx_config.vla.http.max_tokens == llama_config.vla.http.max_tokens
    assert mlx_config.vla.http.structured_output == llama_config.vla.http.structured_output


@pytest.mark.parametrize(
    ("filename", "structured_output"),
    [
        ("vla-probe-qwen3-vl-2b-llama.yaml", False),
        ("vla-probe-qwen3-vl-2b-llama-structured.yaml", True),
    ],
)
def test_probe_qwen3_vl_llama_yaml_targets_pinned_acceptance_alias(
    filename: str,
    structured_output: bool,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / filename)

    assert config.camera.enabled is True
    assert (config.camera.width, config.camera.height) == (512, 288)
    assert config.vla.enabled is True
    assert config.vla.request_timeout_s == 60.0
    assert config.vla.http.endpoint_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert config.vla.http.model_id == "drivebench-qwen3-vl-2b-q4-k-m"
    assert config.vla.http.max_tokens == 512
    assert config.vla.http.structured_output is structured_output


@pytest.mark.parametrize(
    ("filename", "structured_output"),
    [
        ("vla-probe-smolvlm-llama.yaml", False),
        ("vla-probe-smolvlm-llama-structured.yaml", True),
        ("demo-vla-smolvlm-llama.yaml", True),
    ],
)
def test_smolvlm_llama_profiles_differ_from_the_qwen_ones_only_in_the_model(
    filename: str,
    structured_output: bool,
) -> None:
    smolvlm = load_config(PROJECT_ROOT / "configs" / filename).to_dict()
    qwen = load_config(
        PROJECT_ROOT / "configs" / filename.replace("smolvlm", "qwen3-vl-2b")
    ).to_dict()

    assert smolvlm["vla"]["http"].pop("model_id") == "drivebench-smolvlm-256m-q8-0"
    qwen["vla"]["http"].pop("model_id")
    smolvlm.pop("event_log")
    qwen.pop("event_log")
    assert smolvlm == qwen
    assert smolvlm["vla"]["http"]["structured_output"] is structured_output


def test_lane_change_probe_profiles_are_bounded_and_provider_specific() -> None:
    local = load_config(
        PROJECT_ROOT / "configs" / "vla-probe-lane-change-qwen3-vl-2b-mlx.yaml"
    )
    vertex = load_config(
        PROJECT_ROOT / "configs" / "vla-probe-lane-change-vertex.yaml"
    )

    assert local.vla.provider == "http"
    assert local.vla.maximum_requests_per_run == 2
    assert local.vla.http.structured_output is True
    assert vertex.vla.provider == "vertex"
    assert vertex.vla.maximum_requests_per_run == 2
    assert vertex.vla.vertex.location == "global"
    assert vertex.vla.vertex.model_id == "gemini-3.5-flash-lite"
    assert local.controller.target_speed_mps == vertex.controller.target_speed_mps


def test_runtime_qwen_demo_enables_all_local_safety_boundaries() -> None:
    config = load_config(PROJECT_ROOT / "configs/demo-vla-qwen3-vl-2b-mlx.yaml")

    assert config.simulator.manual_control is False
    assert config.simulator.crash_vehicle_done is False
    assert config.simulator.crash_object_done is False
    assert config.camera.enabled is True
    assert config.vla.enabled is True
    assert config.vla.action_horizon_s == 3.0
    assert config.simulator.realtime is True
    assert config.simulator.realtime_factor == 1.0
    assert config.safety.enabled is True
    assert config.vla.http.model_id == "mlx-community/Qwen3-VL-2B-Instruct-4bit"
    assert config.event_log.enabled is True
    assert config.event_log.scenario_id == "qwen3-vl-runtime-demo"


def test_native_llama_acceptance_profile_is_paced_capped_and_headless() -> None:
    config = load_config(PROJECT_ROOT / "configs/demo-vla-qwen3-vl-2b-llama.yaml")

    assert config.simulator.headless is True
    assert config.simulator.realtime is True
    assert config.simulator.realtime_factor == 0.25
    assert config.simulator.horizon == 100
    assert config.vla.enabled is True
    assert config.vla.maximum_requests_per_run == 3
    assert config.vla.http.structured_output is True
    assert config.vla.http.model_id == "drivebench-qwen3-vl-2b-q4-k-m"
    assert config.safety.enabled is True
    assert config.event_log.enabled is True


def test_vertex_runtime_profile_is_paced_capped_and_fail_closed() -> None:
    config = load_config(PROJECT_ROOT / "configs/demo-vla-vertex.yaml")

    assert config.simulator.headless is True
    assert config.simulator.realtime is True
    assert config.simulator.realtime_factor == 0.5
    assert config.camera.enabled is True
    assert config.vla.enabled is True
    assert config.vla.provider == "vertex"
    assert config.vla.maximum_requests_per_run == 10
    assert config.vla.minimum_interval_s == 2.5
    assert config.vla.vertex.project_env == "GOOGLE_CLOUD_PROJECT"
    assert config.vla.vertex.location == "global"
    assert config.vla.vertex.model_id == "gemini-3.5-flash-lite"
    assert config.safety.enabled is True
    assert config.safety.headway_speed_cap_mode == "enforce"
    assert config.scenario.stopped_vehicle_ahead_m == 28.0
    assert config.event_log.enabled is True


def test_config_loads_event_log_settings() -> None:
    config = config_from_dict(
        {
            "event_log": {
                "enabled": True,
                "path": "tmp/test-events.jsonl",
                "scenario_id": "curve-test",
            }
        }
    )

    assert config.event_log.enabled is True
    assert config.event_log.path == "tmp/test-events.jsonl"
    assert config.event_log.scenario_id == "curve-test"


@pytest.mark.parametrize(
    "vla",
    [
        {"enabled": "yes"},
        {"request_timeout_s": 0.0},
        {"minimum_interval_s": -0.1},
        {"maximum_requests_per_run": 0},
        {"maximum_requests_per_run": True},
        {"maximum_requests_per_run": 1.5},
        {"action_horizon_s": float("inf")},
        {"max_scene_objects": -1},
        {"maximum_frame_age_s": 0.0},
        {"maximum_clock_skew_s": -0.1},
        {"prompt_policy": 42},
        {"prompt_policy": "x" * 4097},
        {"prompt_policy": "unsafe\x00policy"},
        {"http": {"model_id": ""}},
        {"http": {"api_key_env": "BAD-NAME"}},
        {"http": {"max_response_bytes": 0}},
        {"http": {"max_tokens": True}},
        {"http": {"allow_insecure_http": "yes"}},
        {"http": {"structured_output": "yes"}},
        {"provider": "unknown"},
        {"vertex": {"project_env": "BAD-NAME"}},
        {"vertex": {"location": ""}},
        {"vertex": {"model_id": ""}},
        {"vertex": {"temperature": 3.0}},
        {"vertex": {"max_output_tokens": 0}},
        {"vertex": {"thinking_budget": -1}},
        {"fixture": {"path": ""}},
        {"fixture": {"repeat_last": "yes"}},
    ],
)
def test_config_rejects_invalid_vla_settings(vla: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="vla"):
        config_from_dict({"vla": vla})


@pytest.mark.parametrize(
    "camera",
    [
        {"width": 0},
        {"height": -1},
        {"width": 3.5},
        {"enabled": "yes"},
    ],
)
def test_config_rejects_invalid_camera_settings(camera: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="camera"):
        config_from_dict({"camera": camera})


def test_config_rejects_non_boolean_realtime_setting() -> None:
    with pytest.raises(ValueError, match="simulator.realtime"):
        config_from_dict({"simulator": {"realtime": "yes"}})


@pytest.mark.parametrize("value", [0.0, -0.1, 1.01, True, float("nan")])
def test_config_rejects_invalid_realtime_factor(value: object) -> None:
    with pytest.raises(ValueError, match="realtime_factor"):
        config_from_dict({"simulator": {"realtime_factor": value}})


PID_YAML = PROJECT_ROOT / "configs" / "pid.yaml"
PID_YAML_TEXT = PID_YAML.read_text()


def test_every_configuration_drives_with_the_gains_in_pid_yaml() -> None:
    instructor = yaml.safe_load(PID_YAML_TEXT)["controller"]

    for path in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        if path == PID_YAML:
            continue
        controller = load_config(path).controller
        assert controller.speed_pid_reset_threshold_mps == instructor[
            "speed_pid_reset_threshold_mps"
        ], path.name
        for loop in ("speed_pid", "steering_pid", "lateral_pid"):
            assert asdict(getattr(controller, loop)) == instructor[loop], (path.name, loop)


@pytest.mark.parametrize(
    "repeated",
    [
        "  speed_pid:\n    kp: 0.5\n",
        "  steering_pid:\n    kd: 0.1\n",
        "  lateral_pid:\n    ki: 0.0\n",
        "  speed_pid_reset_threshold_mps: 2.0\n",
    ],
)
def test_a_configuration_file_that_repeats_a_pid_value_is_refused(
    tmp_path: Path, repeated: str
) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text("controller:\n  target_speed_mps: 5.0\n" + repeated)

    with pytest.raises(ValueError, match="written only in pid.yaml"):
        load_config(path)


def test_each_default_is_written_once() -> None:
    assert config_from_dict({}).controller == ControllerSettings()
    assert config_from_dict({}).planner == PlannerSettings()
    # default.yaml repeats the instructor target speed and planner on purpose, so
    # that teams can read them; this keeps those copies equal to the code.
    default = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    assert default.controller.target_speed_mps == ControllerSettings().target_speed_mps
    assert default.planner == PlannerSettings()


def test_planner_values_keep_their_types_and_unknown_keys_stay_ignored() -> None:
    planner = config_from_dict(
        {"planner": {"lookahead_m": 6, "default_route": [[0, 0], [10, 0]], "unknown_key": 1}}
    ).planner

    assert planner.lookahead_m == 6.0
    assert isinstance(planner.lookahead_m, float)
    assert planner.default_route == [(0.0, 0.0), (10.0, 0.0)]
    assert planner.waypoint_spacing_m == PlannerSettings().waypoint_spacing_m


def _use_pid_yaml(monkeypatch: pytest.MonkeyPatch, path: Path, text: str) -> None:
    path.write_text(text)
    monkeypatch.setattr(config_module, "PID_CONFIG", path)


def test_an_edit_to_pid_yaml_takes_effect_without_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pid.yaml"
    _use_pid_yaml(monkeypatch, path, PID_YAML_TEXT)
    assert config_from_dict({}).controller.speed_pid.kp == 0.12

    path.write_text(PID_YAML_TEXT.replace("kp: 0.12", "kp: 0.31415"))

    assert config_from_dict({}).controller.speed_pid.kp == 0.31415
    assert load_config(tmp_path / "absent.yaml").controller.speed_pid.kp == 0.31415


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (None, "is missing"),
        ("speed_pid: {}\n", "one controller section"),
        (
            PID_YAML_TEXT + "planner:\n  lookahead_m: 6.0\n",
            "one controller section",
        ),
        (
            PID_YAML_TEXT.replace("controller:\n", "controller:\n  target_speed_mps: 5.0\n"),
            "exactly these controller keys",
        ),
        (
            PID_YAML_TEXT.replace(
                "  steering_pid:\n    kp: 1.0\n    ki: 0.0\n    kd: 0.0\n",
                "  steering_pid:\n    kp: 1.0\n    ki: 0.0\n",
            ),
            "controller.steering_pid must set exactly",
        ),
    ],
)
def test_a_malformed_pid_yaml_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str | None, message: str
) -> None:
    path = tmp_path / "pid.yaml"
    monkeypatch.setattr(config_module, "PID_CONFIG", path)
    if text is not None:
        path.write_text(text)

    with pytest.raises(ValueError, match=message):
        config_from_dict({})
