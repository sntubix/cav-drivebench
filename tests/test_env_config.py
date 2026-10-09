from metadrive_starter.config import CameraSettings, SimulatorSettings
from types import SimpleNamespace

import pytest

from metadrive_starter.env import (
    _configure_filter_manager_render_scene_into,
    _configure_rgb_camera_shader_loader,
    _configure_simplepbr_init,
    control_timestep_s,
    ego_state_info,
    to_metadrive_config,
)
from metadrive_starter.perception import BasicPerception


def test_metadrive_config_uses_supported_top_level_keys() -> None:
    config = to_metadrive_config(SimulatorSettings(use_lidar=True, headless=True))

    assert config["use_render"] is False
    assert config["vehicle_config"]["enable_reverse"] is False
    assert config["out_of_road_done"] is True
    assert config["crash_vehicle_done"] is True
    assert config["crash_object_done"] is True
    assert config["accident_prob"] == 0.0
    assert config["random_traffic"] is False
    assert config["traffic_mode"] == "trigger"
    assert config["record_episode"] is False
    assert config["replay_episode"] is None
    assert config["only_reset_when_replay"] is False
    assert config["physics_world_step_size"] == 0.02
    assert "use_lidar" not in config


def test_control_timestep_comes_from_explicit_metadrive_timing() -> None:
    settings = SimulatorSettings(decision_repeat=4, physics_world_step_size=0.025)

    assert control_timestep_s(settings) == pytest.approx(0.1)
    config = to_metadrive_config(settings)
    assert config["decision_repeat"] * config["physics_world_step_size"] == pytest.approx(
        0.1
    )


def test_manual_control_enables_reverse() -> None:
    config = to_metadrive_config(SimulatorSettings(manual_control=True))

    assert config["manual_control"] is True
    assert config["vehicle_config"]["enable_reverse"] is True


def test_disabling_lidar_updates_metadrive_vehicle_config() -> None:
    config = to_metadrive_config(SimulatorSettings(use_lidar=False))

    assert config["vehicle_config"]["lidar"] == {"num_lasers": 0, "distance": 0}


def test_optional_spawn_longitude_is_forwarded_to_ego_vehicle() -> None:
    config = to_metadrive_config(SimulatorSettings(spawn_longitude_m=30.0))

    assert config["vehicle_config"]["spawn_longitude"] == 30.0


def test_enabling_camera_configures_single_uint8_rgb_observation() -> None:
    config = to_metadrive_config(
        SimulatorSettings(),
        CameraSettings(enabled=True, width=640, height=360),
    )

    sensor_class, width, height = config["sensors"]["rgb_camera"]
    assert sensor_class.__name__ == "RGBCamera"
    assert (width, height) == (640, 360)
    assert config["image_observation"] is True
    assert config["norm_pixel"] is False
    assert config["stack_size"] == 1
    assert config["vehicle_config"]["image_source"] == "rgb_camera"


def test_ego_state_info_uses_live_vehicle_state() -> None:
    env = SimpleNamespace(
        agent=SimpleNamespace(position=(2.0, 3.0), heading_theta=0.5, speed=12.0)
    )

    info = ego_state_info(env, {"route_completion": 0.25})

    assert info["position"] == (2.0, 3.0)
    assert info["heading"] == 0.5
    assert info["speed_mps"] == 12.0
    assert info["route_completion"] == 0.25


def test_basic_perception_prefers_explicit_mps_even_when_zero() -> None:
    frame = BasicPerception().observe(
        {},
        {"position": (0.0, 0.0), "heading": 0.0, "speed_mps": 0.0, "speed": 99.0},
    )

    assert frame.ego.speed_mps == 0.0


def test_simplepbr_is_configured_for_macos_core_profile() -> None:
    received: dict[str, object] = {}

    def init(**kwargs: object) -> str:
        received.update(kwargs)
        return "pipeline"

    compatible_init = _configure_simplepbr_init(init, maximum=4)

    assert compatible_init(msaa_samples=16, use_hardware_skinning=True) == "pipeline"
    assert received == {"msaa_samples": 4, "use_hardware_skinning": True, "use_330": True}
    assert compatible_init._metadrive_starter_msaa_compat is True


def test_camera_post_processing_msaa_is_capped() -> None:
    received: dict[str, object] = {}

    class FrameBuffer:
        multisamples = 16

        def get_multisamples(self) -> int:
            return self.multisamples

        def set_multisamples(self, value: int) -> None:
            self.multisamples = value

    def render(manager: object, **kwargs: object) -> str:
        received.update(manager=manager, **kwargs)
        return "quad"

    frame_buffer = FrameBuffer()
    compatible_render = _configure_filter_manager_render_scene_into(render, maximum=4)

    assert compatible_render("manager", fbprops=frame_buffer) == "quad"
    assert frame_buffer.multisamples == 4
    assert received == {"manager": "manager", "fbprops": frame_buffer}
    assert compatible_render._metadrive_starter_msaa_compat is True


def test_rgb_camera_post_processing_uses_core_profile_without_mutating_defines() -> None:
    received: dict[str, object] = {}

    def load_shader(shader_path: str, defines: dict[str, object]) -> str:
        received.update(shader_path=shader_path, defines=defines)
        return "shader"

    original_loader = lambda *_args, **_kwargs: "legacy shader"
    rgb_camera = SimpleNamespace(_load_shader_str=original_loader)
    original_defines = {"EXPOSURE": "1.0"}

    _configure_rgb_camera_shader_loader(rgb_camera, load_shader)
    compatible_loader = rgb_camera._load_shader_str

    assert compatible_loader("tonemap.frag", original_defines) == "shader"
    assert received == {
        "shader_path": "tonemap.frag",
        "defines": {"EXPOSURE": "1.0", "USE_330": ""},
    }
    assert original_defines == {"EXPOSURE": "1.0"}
    assert compatible_loader._metadrive_starter_camera_core_compat is True

    _configure_rgb_camera_shader_loader(rgb_camera, load_shader)
    assert rgb_camera._load_shader_str is compatible_loader
