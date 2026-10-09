from __future__ import annotations

import platform
import math
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from metadrive_starter.config import (
    CameraSettings,
    ScenarioVehicleSettings,
    SimulatorSettings,
)
from metadrive_starter.types import Point2D


_MACOS_MAX_MSAA_SAMPLES = 4


class MetaDriveUnavailable(RuntimeError):
    """Raised when MetaDrive is not installed or cannot be imported."""


def make_env(
    settings: SimulatorSettings,
    camera: CameraSettings | None = None,
    *,
    record_episode: bool = False,
    replay_episode: object | None = None,
):
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
    except Exception as exc:  # pragma: no cover - depends on optional native stack
        raise MetaDriveUnavailable(
            "MetaDrive could not be imported. Run `uv sync` with Python 3.10 or 3.11, "
            "then retry. Rendering may require additional OS graphics libraries."
        ) from exc

    camera = camera or CameraSettings()
    if not settings.headless or camera.enabled:
        _configure_macos_rendering()

    env = MetaDriveEnv(
        config=to_metadrive_config(
            settings,
            camera,
            record_episode=record_episode,
        )
    )
    # MetaDrive's Config.update rejects nested dictionaries passed at
    # construction, while its supported replay workflow assigns episode data
    # before reset.
    if replay_episode is not None:
        env.config["replay_episode"] = replay_episode
    return env


def ego_state_info(env: Any, info: dict[str, object] | None = None) -> dict[str, object]:
    """Add stable, correctly-unit-labelled ego state to MetaDrive's step info."""
    vehicle = env.agent
    enriched = dict(info or {})
    enriched.update(
        position=(float(vehicle.position[0]), float(vehicle.position[1])),
        heading=float(vehicle.heading_theta),
        speed_mps=float(vehicle.speed),
    )
    return enriched


def navigation_waypoints(env: Any, spacing_m: float) -> list[Point2D]:
    """Sample the route selected by MetaDrive navigation into world-space waypoints."""
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")

    vehicle = env.agent
    navigation = vehicle.navigation
    network = env.current_map.road_network
    lane_number = int(vehicle.lane_index[-1])
    route: list[Point2D] = []
    route_pairs = list(zip(navigation.checkpoints[:-1], navigation.checkpoints[1:]))
    current_road = tuple(vehicle.lane_index[:2])
    start_index = next(
        (index for index, pair in enumerate(route_pairs) if tuple(pair) == current_road),
        None,
    )
    if start_index is None:
        raise RuntimeError("ego lane is not part of the selected navigation route")

    for route_index, (start, end) in enumerate(
        route_pairs[start_index:],
        start=start_index,
    ):
        lanes = network.graph[start][end]
        lane = lanes[min(lane_number, len(lanes) - 1)]
        initial_longitudinal = (
            max(0.0, float(lane.local_coordinates(vehicle.position)[0]))
            if route_index == start_index
            else 0.0
        )
        remaining_m = max(0.0, float(lane.length) - initial_longitudinal)
        sample_count = max(1, math.ceil(remaining_m / spacing_m))
        for index in range(sample_count + 1):
            longitudinal = min(
                initial_longitudinal + index * spacing_m,
                float(lane.length),
            )
            position = lane.position(longitudinal, 0.0)
            point = (float(position[0]), float(position[1]))
            if not route or math.dist(route[-1], point) > 1e-6:
                route.append(point)

    if not route:
        raise RuntimeError("MetaDrive navigation did not provide a route")
    return route


def lane_change_waypoints(
    env: Any,
    spacing_m: float,
    *,
    lane_offset: int,
    transition_distance_m: float,
) -> list[Point2D]:
    """Build smooth local route from current lane into one adjacent lane."""
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    if lane_offset not in {-1, 1}:
        raise ValueError("lane_offset must be -1 or 1")
    if not math.isfinite(transition_distance_m) or transition_distance_m <= 0.0:
        raise ValueError("transition_distance_m must be finite and positive")

    ego_position = (float(env.agent.position[0]), float(env.agent.position[1]))
    route: list[Point2D] = [ego_position]
    distance_m = min(spacing_m, transition_distance_m)
    while distance_m <= transition_distance_m + 1e-9:
        current_lane, current_longitudinal = _route_position_ahead(env, distance_m)
        target_lane, target_longitudinal = _route_position_ahead(
            env,
            distance_m,
            lane_offset=lane_offset,
        )
        current = current_lane.position(current_longitudinal, 0.0)
        target = target_lane.position(target_longitudinal, 0.0)
        progress = min(1.0, distance_m / transition_distance_m)
        blend = progress * progress * (3.0 - 2.0 * progress)
        route.append(
            (
                float(current[0] + blend * (target[0] - current[0])),
                float(current[1] + blend * (target[1] - current[1])),
            )
        )
        distance_m += spacing_m

    if distance_m - spacing_m < transition_distance_m - 1e-9:
        distance_m = transition_distance_m
        target_lane, target_longitudinal = _route_position_ahead(
            env,
            distance_m,
            lane_offset=lane_offset,
        )
        target = target_lane.position(target_longitudinal, 0.0)
        route.append((float(target[0]), float(target[1])))

    distance_m = max(distance_m, transition_distance_m + spacing_m)
    while True:
        try:
            lane, longitudinal = _route_position_ahead(
                env,
                distance_m,
                lane_offset=lane_offset,
            )
        except ValueError:
            break
        position = lane.position(longitudinal, 0.0)
        point = (float(position[0]), float(position[1]))
        if math.dist(route[-1], point) > 1e-6:
            route.append(point)
        distance_m += spacing_m

    if len(route) < 2:
        raise RuntimeError("lane-change route did not provide enough waypoints")
    return route


def lane_topology(
    env: Any,
    *,
    target_lane_index: int | None = None,
) -> tuple[int, int, float | None]:
    """Return current lane index/count and lateral offset from an optional target."""
    ego = env.agent
    start, end, current_lane_index = ego.lane_index
    lanes = env.current_map.road_network.graph[start][end]
    target_offset_m: float | None = None
    if target_lane_index is not None and 0 <= target_lane_index < len(lanes):
        target_offset_m = float(lanes[target_lane_index].local_coordinates(ego.position)[1])
    return int(current_lane_index), len(lanes), target_offset_m


def simulation_time_s(env: Any) -> float:
    config = env.engine.global_config
    return float(env.engine.episode_step * config["decision_repeat"] * config["physics_world_step_size"])


def control_timestep_s(settings: SimulatorSettings) -> float:
    """Return one environment/control step in seconds from explicit simulator timing."""
    return float(settings.decision_repeat * settings.physics_world_step_size)


def spawn_stopped_vehicle_ahead(env: Any, distance_m: float) -> Any:
    """Spawn one deterministic static vehicle on the ego navigation route."""
    return spawn_static_vehicle_ahead(env, distance_m)


def spawn_traffic_light_ahead(
    env: Any,
    distance_m: float,
    *,
    name: str = "scenario-traffic-light",
) -> Any:
    """Spawn one stock MetaDrive traffic light on the ego navigation route."""
    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")
    if not isinstance(name, str) or not name:
        raise ValueError("name must not be empty")

    from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight

    lane, longitudinal = _route_position_ahead(env, distance_m)
    spawn_kwargs = {
        "lane": lane,
        "position": lane.position(longitudinal, 0.0),
        "force_spawn": True,
        "name": name,
    }
    light = _spawn_scenario_object(
        env,
        BaseTrafficLight,
        spawn_kwargs,
    )
    light.set_heading_theta(lane.heading_theta_at(longitudinal))
    _record_post_reset_scenario_object(env, light, BaseTrafficLight, spawn_kwargs)
    return light


def spawn_traffic_light_group_ahead(
    env: Any,
    distance_m: float,
    *,
    name_prefix: str = "scenario-traffic-light",
    visual_scale: float = 1.75,
) -> tuple[Any, ...]:
    """Spawn one synchronized stock signal above every lane on the route road."""
    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")
    if not isinstance(name_prefix, str) or not name_prefix:
        raise ValueError("name_prefix must not be empty")
    if not math.isfinite(visual_scale) or visual_scale <= 0.0:
        raise ValueError("visual_scale must be finite and positive")

    from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight

    selected_lane, selected_longitudinal = _route_position_ahead(env, distance_m)
    start, end, _ = selected_lane.index
    lanes = env.current_map.road_network.graph[start][end]
    ratio = selected_longitudinal / float(selected_lane.length)
    lights = []
    for lane_number, lane in enumerate(lanes):
        longitudinal = min(max(ratio * float(lane.length), 0.0), float(lane.length))
        spawn_kwargs = {
            "lane": lane,
            "position": lane.position(longitudinal, 0.0),
            "force_spawn": True,
            "name": f"{name_prefix}-{lane_number}",
        }
        light = _spawn_scenario_object(
            env,
            BaseTrafficLight,
            spawn_kwargs,
        )
        light.set_heading_theta(lane.heading_theta_at(longitudinal))
        light.drivebench_visual_scale = float(visual_scale)
        _record_post_reset_scenario_object(
            env,
            light,
            BaseTrafficLight,
            spawn_kwargs,
        )
        lights.append(light)
    _attach_traffic_light_gantry(env, lights, lanes)
    return tuple(lights)


def _attach_traffic_light_gantry(env: Any, lights: list[Any], lanes: list[Any]) -> None:
    """Mount lane signals on a simple non-physical overhead support."""
    if not lights or not lights[0].render:
        return

    from panda3d.core import NodePath, Point3

    from metadrive.engine.asset_loader import AssetLoader

    middle = lights[len(lights) // 2]
    anchor = NodePath("drivebench-traffic-light-gantry")
    anchor.reparentTo(env.engine.render)
    anchor.setPos(float(middle.position[0]), float(middle.position[1]), 0.0)
    anchor.setH(middle.origin.getH())
    middle._node_path_list.append(anchor)

    lane_widths = [float(lane.width_at(0.0)) for lane in lanes]
    half_span_m = sum(lane_widths) / 2.0 + 0.5
    top_m = 4.8
    support_thickness_m = 0.18

    def support_box(
        name: str,
        position: tuple[float, float, float],
        scale: tuple[float, float, float],
    ) -> None:
        box = middle.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
        box.setName(name)
        box.setPos(*position)
        box.setScale(*scale)
        box.setColor(0.12, 0.14, 0.16, 1.0)
        box.setTextureOff(1)
        box.reparentTo(anchor)

    support_box(
        "gantry-roadside-pole",
        (0.0, -half_span_m, top_m / 2.0),
        (support_thickness_m, support_thickness_m, top_m),
    )
    support_box(
        "gantry-crossbar",
        (0.0, 0.0, top_m),
        (support_thickness_m, 2.0 * half_span_m, support_thickness_m),
    )

    for index, light in enumerate(lights):
        local_position = anchor.getRelativePoint(
            env.engine.render,
            Point3(float(light.position[0]), float(light.position[1]), top_m),
        )
        support_box(
            f"gantry-signal-hanger-{index}",
            (float(local_position.x), float(local_position.y), top_m - 0.2),
            (0.1, 0.1, 0.4),
        )


def spawn_static_vehicle_ahead(
    env: Any,
    distance_m: float,
    *,
    lane_offset: int = 0,
    lateral_m: float = 0.0,
    vehicle_kind: str = "car",
    name: str = "scenario-stopped-vehicle",
) -> Any:
    """Spawn a deterministic static car or truck relative to the ego route."""
    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")
    if isinstance(lane_offset, bool) or not isinstance(lane_offset, int):
        raise ValueError("lane_offset must be an integer")
    if not math.isfinite(lateral_m):
        raise ValueError("lateral_m must be finite")
    if vehicle_kind not in {"car", "truck"}:
        raise ValueError("vehicle_kind must be 'car' or 'truck'")
    if not isinstance(name, str) or not name:
        raise ValueError("name must not be empty")

    from metadrive.component.vehicle.vehicle_type import StaticDefaultVehicle, XLVehicle

    vehicle_type = StaticDefaultVehicle if vehicle_kind == "car" else XLVehicle
    lane, longitudinal = _route_position_ahead(env, distance_m, lane_offset=lane_offset)
    spawn_kwargs = {
        "force_spawn": True,
        "name": name,
        "vehicle_config": {
            "spawn_lane_index": lane.index,
            "spawn_longitude": float(longitudinal),
            "spawn_lateral": float(lateral_m),
            "enable_reverse": False,
        },
    }
    obstacle = _spawn_scenario_object(
        env,
        vehicle_type,
        spawn_kwargs,
    )
    obstacle.set_velocity([0.0, 0.0])
    obstacle.set_static(True)
    _record_post_reset_scenario_object(env, obstacle, vehicle_type, spawn_kwargs)
    return obstacle


def spawn_scenario_vehicle(env: Any, settings: ScenarioVehicleSettings) -> Any:
    """Spawn one deterministic moving actor relative to ego on the current road."""

    if not isinstance(settings, ScenarioVehicleSettings):
        raise TypeError("settings must be ScenarioVehicleSettings")
    ego = env.agent
    start, end, current_lane_index = ego.lane_index
    lanes = env.current_map.road_network.graph[start][end]
    target_lane_index = int(current_lane_index) + settings.lane_offset
    if not 0 <= target_lane_index < len(lanes):
        raise ValueError(
            f"scenario vehicle {settings.vehicle_id!r} targets unavailable "
            f"lane offset {settings.lane_offset}"
        )
    current_lane = lanes[int(current_lane_index)]
    ego_longitudinal_m = float(current_lane.local_coordinates(ego.position)[0])
    target_longitudinal_m = ego_longitudinal_m + settings.longitudinal_offset_m
    target_lane = lanes[target_lane_index]
    if not 0.0 <= target_longitudinal_m <= float(target_lane.length):
        raise ValueError(
            f"scenario vehicle {settings.vehicle_id!r} longitudinal offset "
            "falls outside the current road"
        )

    from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle, XLVehicle

    vehicle_type = TrafficDefaultVehicle if settings.kind == "car" else XLVehicle
    spawn_kwargs = {
        "force_spawn": True,
        "name": f"scenario-{settings.vehicle_id}",
        "vehicle_config": {
            "spawn_lane_index": target_lane.index,
            "spawn_longitude": target_longitudinal_m,
            "spawn_lateral": 0.0,
            "spawn_velocity": [settings.speed_mps, 0.0],
            "spawn_velocity_car_frame": True,
            "enable_reverse": False,
        },
    }
    actor = _spawn_scenario_object(env, vehicle_type, spawn_kwargs)
    _record_post_reset_scenario_object(env, actor, vehicle_type, spawn_kwargs)
    return actor


def _spawn_scenario_object(
    env: Any,
    object_class: type,
    spawn_kwargs: dict[str, Any],
) -> Any:
    """Spawn post-reset fixtures without confusing MetaDrive's reset recorder."""
    engine = env.engine
    post_reset_recording = (
        bool(engine.global_config.get("record_episode", False))
        and not engine.replay_episode
        and engine.record_manager.current_frames is None
    )
    if not post_reset_recording:
        return engine.spawn_object(object_class, **spawn_kwargs)
    kwargs = dict(spawn_kwargs)
    kwargs["random_seed"] = engine.generate_seed()
    spawn_kwargs["random_seed"] = kwargs["random_seed"]
    return engine.spawn_object(
        object_class,
        auto_fill_random_seed=False,
        record=False,
        **kwargs,
    )


def _record_post_reset_scenario_object(
    env: Any,
    obj: Any,
    object_class: type,
    spawn_kwargs: dict[str, Any],
) -> None:
    engine = env.engine
    if (
        not bool(engine.global_config.get("record_episode", False))
        or engine.replay_episode
        or engine.record_manager.current_frames is not None
    ):
        return
    from copy import deepcopy

    from metadrive.constants import ObjectState

    reset_frame = engine.record_manager.episode_info["frame"][0][0]
    reset_frame.spawn_info[obj.name] = {
        ObjectState.CLASS: object_class,
        ObjectState.INIT_KWARGS: deepcopy(spawn_kwargs),
        ObjectState.NAME: obj.name,
    }
    reset_frame.step_info[obj.name] = obj.get_state()


def _route_position_ahead(
    env: Any,
    distance_m: float,
    *,
    lane_offset: int = 0,
) -> tuple[Any, float]:
    ego = env.agent
    navigation = ego.navigation
    network = env.current_map.road_network
    ego_lane_number = int(ego.lane_index[-1])
    route_pairs = list(zip(navigation.checkpoints[:-1], navigation.checkpoints[1:]))
    current_road = tuple(ego.lane_index[:2])
    start_index = next(
        (index for index, pair in enumerate(route_pairs) if tuple(pair) == current_road),
        None,
    )
    if start_index is None:
        raise RuntimeError("ego lane is not part of the selected navigation route")

    remaining = distance_m
    for route_index, (start, end) in enumerate(route_pairs[start_index:], start=start_index):
        lanes = network.graph[start][end]
        lane_number = ego_lane_number + lane_offset
        if lane_offset == 0:
            lane_number = min(lane_number, len(lanes) - 1)
        elif not 0 <= lane_number < len(lanes):
            raise ValueError(
                f"lane_offset={lane_offset} is unavailable on route segment {start!r}->{end!r}"
            )
        lane = lanes[lane_number]
        initial_longitudinal = float(lane.local_coordinates(ego.position)[0]) if route_index == start_index else 0.0
        available = float(lane.length) - initial_longitudinal
        if remaining <= available:
            return lane, initial_longitudinal + remaining
        remaining -= available

    raise ValueError(f"distance_m={distance_m} extends beyond the selected navigation route")


def _configure_macos_rendering() -> None:
    if platform.system() != "Darwin":
        return

    from panda3d.core import loadPrcFileData

    import metadrive.engine.core.engine_core as engine_core
    import metadrive.engine.core.sky_box as sky_box
    import metadrive.component.sensors.rgb_camera as rgb_camera
    import metadrive.third_party.simplepbr as simplepbr
    from metadrive.engine.core.terrain import Terrain

    # MetaDrive 0.4.3 mixes GLSL 120 and 330 while Panda3D defaults to an
    # OpenGL 2.1 context on macOS. Request a core context, then update the few
    # legacy shader calls that are invalid there. Cocoa exposes only 4x MSAA
    # for MetaDrive's requested framebuffer formats.
    loadPrcFileData(
        "metadrive-starter-macos",
        f"gl-version 3 3\nmultisamples {_MACOS_MAX_MSAA_SAMPLES}",
    )

    _configure_simplepbr_shader_loader(simplepbr)
    _configure_rgb_camera_shader_loader(rgb_camera, simplepbr._load_shader_str)
    _configure_terrain_shader_loader(Terrain)
    _configure_core_skybox(sky_box)
    _configure_filter_manager_msaa()

    if getattr(engine_core.init, "_metadrive_starter_msaa_compat", False):
        return

    engine_core.init = _configure_simplepbr_init(engine_core.init, _MACOS_MAX_MSAA_SAMPLES)


def _configure_simplepbr_init(init: Callable[..., Any], maximum: int) -> Callable[..., Any]:
    @wraps(init)
    def compatible_init(**kwargs: Any) -> Any:
        requested = kwargs.get("msaa_samples", maximum)
        kwargs["msaa_samples"] = min(requested, maximum)
        kwargs["use_330"] = True
        return init(**kwargs)

    compatible_init._metadrive_starter_msaa_compat = True  # type: ignore[attr-defined]
    return compatible_init


def _configure_filter_manager_msaa() -> None:
    """Cap MetaDrive camera post-processing buffers to macOS-supported MSAA."""
    from direct.filter.FilterManager import FilterManager

    if getattr(FilterManager.render_scene_into, "_metadrive_starter_msaa_compat", False):
        return

    FilterManager.render_scene_into = _configure_filter_manager_render_scene_into(
        FilterManager.render_scene_into,
        _MACOS_MAX_MSAA_SAMPLES,
    )


def _configure_filter_manager_render_scene_into(
    render_scene_into: Callable[..., Any],
    maximum: int,
) -> Callable[..., Any]:
    @wraps(render_scene_into)
    def compatible_render_scene_into(manager: Any, *args: Any, **kwargs: Any) -> Any:
        frame_buffer = kwargs.get("fbprops")
        if frame_buffer is not None and frame_buffer.get_multisamples() > maximum:
            frame_buffer.set_multisamples(maximum)
        return render_scene_into(manager, *args, **kwargs)

    compatible_render_scene_into._metadrive_starter_msaa_compat = True  # type: ignore[attr-defined]
    return compatible_render_scene_into


def _configure_simplepbr_shader_loader(simplepbr: Any) -> None:
    if getattr(simplepbr._load_shader_str, "_metadrive_starter_core_compat", False):
        return

    original_loader = simplepbr._load_shader_str

    @wraps(original_loader)
    def compatible_loader(shader_path: str, defines: dict[str, Any] | None = None) -> str:
        use_330 = defines is not None and "USE_330" in defines
        source = original_loader(shader_path, defines)
        return source.replace("texture2D(", "texture(") if use_330 else source

    compatible_loader._metadrive_starter_core_compat = True  # type: ignore[attr-defined]
    simplepbr._load_shader_str = compatible_loader


def _configure_rgb_camera_shader_loader(rgb_camera: Any, shader_loader: Callable[..., str]) -> None:
    """Load the RGB camera's post-processing shaders for the macOS core profile."""
    if getattr(rgb_camera._load_shader_str, "_metadrive_starter_camera_core_compat", False):
        return

    @wraps(shader_loader)
    def compatible_camera_loader(
        shader_path: str,
        defines: dict[str, Any] | None = None,
    ) -> str:
        core_defines = dict(defines or {})
        core_defines["USE_330"] = ""
        return shader_loader(shader_path, core_defines)

    compatible_camera_loader._metadrive_starter_camera_core_compat = True  # type: ignore[attr-defined]
    rgb_camera._load_shader_str = compatible_camera_loader


def _configure_terrain_shader_loader(terrain: Any) -> None:
    if getattr(terrain.make_render_state, "_metadrive_starter_core_compat", False):
        return

    from panda3d.core import NodePath, Shader

    from metadrive.engine.asset_loader import AssetLoader

    def compatible_render_state(engine: Any, vertex_name: str, fragment_name: str) -> Any:
        del engine
        vertex_path = Path(AssetLoader.file_path("../shaders", vertex_name))
        fragment_path = Path(AssetLoader.file_path("../shaders", fragment_name))
        vertex_source = vertex_path.read_text().replace("texture2D(", "texture(")
        fragment_source = fragment_path.read_text().replace("texture2D(", "texture(")
        shader = Shader.make(Shader.SL_GLSL, vertex=vertex_source, fragment=fragment_source)
        dummy = NodePath("metadrive-starter-terrain-shader")
        dummy.set_shader(shader)
        return dummy.get_state()

    compatible_render_state._metadrive_starter_core_compat = True  # type: ignore[attr-defined]
    terrain.make_render_state = staticmethod(compatible_render_state)


def _configure_core_skybox(sky_box: Any) -> None:
    if getattr(sky_box.is_mac, "_metadrive_starter_core_compat", False):
        return

    def use_generic_core_shader() -> bool:
        return False

    use_generic_core_shader._metadrive_starter_core_compat = True  # type: ignore[attr-defined]
    sky_box.is_mac = use_generic_core_shader


def to_metadrive_config(
    settings: SimulatorSettings,
    camera: CameraSettings | None = None,
    *,
    record_episode: bool = False,
    replay_episode: object | None = None,
) -> dict[str, Any]:
    if not isinstance(record_episode, bool):
        raise TypeError("record_episode must be a boolean")
    if record_episode and replay_episode is not None:
        raise ValueError("record_episode and replay_episode cannot both be enabled")
    camera = camera or CameraSettings()
    vehicle_config: dict[str, Any] = {"enable_reverse": settings.manual_control}
    if settings.spawn_longitude_m is not None:
        vehicle_config["spawn_longitude"] = settings.spawn_longitude_m
    if not settings.use_lidar:
        vehicle_config["lidar"] = {"num_lasers": 0, "distance": 0}

    config: dict[str, Any] = {
        "map": settings.map,
        "traffic_density": settings.traffic_density,
        "random_traffic": settings.random_traffic,
        "traffic_mode": settings.traffic_mode,
        "accident_prob": settings.obstacle_probability,
        "num_scenarios": settings.num_scenarios,
        "start_seed": settings.start_seed,
        "decision_repeat": settings.decision_repeat,
        "physics_world_step_size": settings.physics_world_step_size,
        "horizon": settings.horizon,
        "use_render": not settings.headless,
        "manual_control": settings.manual_control,
        # MetaDrive disables reverse by default, making S brake-only. Enable it
        # for keyboard play without changing autonomous action semantics.
        "vehicle_config": vehicle_config,
        "out_of_road_done": settings.out_of_road_done,
        "crash_vehicle_done": settings.crash_vehicle_done,
        "crash_object_done": settings.crash_object_done,
        "record_episode": record_episode,
        "replay_episode": None,
        "only_reset_when_replay": False,
    }
    if camera.enabled:
        from metadrive.component.sensors.rgb_camera import RGBCamera

        config.update(
            image_observation=True,
            norm_pixel=False,
            stack_size=1,
            sensors={"rgb_camera": (RGBCamera, camera.width, camera.height)},
        )
        vehicle_config["image_source"] = "rgb_camera"
    return config
