from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from metadrive_starter.perception.object_scene import ObjectSceneBuilder, object_identifier
from metadrive_starter.perception.scene import LocalScene
from metadrive_starter.planning import FuturePathProjector


@dataclass(frozen=True)
class OracleSceneAdapter:
    """Convert all perfect simulator objects into the shared scene contract."""

    detection_radius_m: float = 50.0
    corridor_margin_m: float = 0.5
    future_path: FuturePathProjector | None = None

    def __post_init__(self) -> None:
        self._builder()

    def observe(self, env: Any, *, timestamp_s: float) -> LocalScene:
        candidates = (
            (object_identifier(candidate, object_id), candidate)
            for object_id, candidate in env.engine.get_objects().items()
        )
        return self._builder().observe(
            env,
            candidates,
            timestamp_s=timestamp_s,
        )

    def _builder(self) -> ObjectSceneBuilder:
        return ObjectSceneBuilder(
            detection_radius_m=self.detection_radius_m,
            corridor_margin_m=self.corridor_margin_m,
            future_path=self.future_path,
        )
