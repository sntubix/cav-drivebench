from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from metadrive_starter.types import EgoState, PerceptionFrame


@dataclass
class BasicPerception:
    prefer_info: bool = True

    def observe(self, observation: object, info: dict[str, object] | None = None) -> PerceptionFrame:
        info = info or {}
        source = info if self.prefer_info else {}

        position = _point2(source.get("position") or source.get("ego_position"), default=(0.0, 0.0))
        heading = _float(source.get("heading") or source.get("heading_theta"), default=0.0)
        speed = _float(
            _first_present(source, ("speed_mps", "speed", "velocity", "ego_speed")),
            default=0.0,
        )
        lidar = _extract_lidar(observation)

        return PerceptionFrame(
            ego=EgoState(position=position, heading_rad=heading, speed_mps=speed),
            lidar=lidar,
            raw_observation_type=type(observation).__name__,
        )


def _point2(value: object, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return default


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_present(source: dict[str, object], keys: tuple[str, ...]) -> object:
    return next((source[key] for key in keys if key in source and source[key] is not None), None)


def _extract_lidar(observation: object) -> Sequence[float] | None:
    if isinstance(observation, dict):
        lidar = observation.get("lidar") or observation.get("lidar_state")
        if isinstance(lidar, Sequence) and not isinstance(lidar, (str, bytes)):
            return lidar

    shape = getattr(observation, "shape", None)
    if shape is not None and len(shape) == 1:
        return observation  # type: ignore[return-value]

    return None
