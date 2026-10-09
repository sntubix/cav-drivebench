from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from metadrive_starter.perception.object_scene import ObjectSceneBuilder, object_identifier
from metadrive_starter.perception.scene import LocalScene
from metadrive_starter.planning import FuturePathProjector


@dataclass(frozen=True)
class LidarSceneAdapter:
    """Build a local scene from MetaDrive's LiDAR-detected object set."""

    detection_radius_m: float = 50.0
    corridor_margin_m: float = 0.5
    future_path: FuturePathProjector | None = None

    def __post_init__(self) -> None:
        self._builder()

    def observe(self, env: Any, *, timestamp_s: float) -> LocalScene:
        ego = env.agent
        lidar_config = ego.config.get("lidar", {})
        num_lasers = int(lidar_config.get("num_lasers", 0))
        configured_distance = float(lidar_config.get("distance", 0.0))
        if num_lasers <= 0 or configured_distance <= 0.0:
            return self._builder().observe(env, (), timestamp_s=timestamp_s, valid=False)

        sensor = env.engine.get_sensor("lidar")
        _, detected_objects = sensor.perceive(
            ego,
            physics_world=env.engine.physics_world.dynamic_world,
            num_lasers=num_lasers,
            distance=min(self.detection_radius_m, configured_distance),
            show=bool(ego.config.get("show_lidar", False)),
        )
        ordered_objects = sorted(
            detected_objects,
            key=lambda candidate: str(getattr(candidate, "id", "")),
        )
        candidates = (
            (object_identifier(candidate, f"lidar-{index}"), candidate)
            for index, candidate in enumerate(ordered_objects)
        )
        return self._builder().observe(env, candidates, timestamp_s=timestamp_s)

    def _builder(self) -> ObjectSceneBuilder:
        return ObjectSceneBuilder(
            detection_radius_m=self.detection_radius_m,
            corridor_margin_m=self.corridor_margin_m,
            future_path=self.future_path,
        )
