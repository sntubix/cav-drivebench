from __future__ import annotations

from typing import TYPE_CHECKING, Any

from metadrive_starter.planning.future_path import (
    FuturePathProjector,
    PathProjection,
    PolylineFuturePath,
)
from metadrive_starter.planning.path_planner import WaypointPathPlanner

if TYPE_CHECKING:
    from metadrive_starter.planning.action_speed import (
        ActionSpeedDecision,
        ActionSpeedPolicy,
    )


def __getattr__(name: str) -> Any:
    if name in {
        "ACTION_SPEED_POLICY_MODES",
        "ActionSpeedDecision",
        "ActionSpeedPolicy",
    }:
        from metadrive_starter.planning import action_speed

        return getattr(action_speed, name)
    raise AttributeError(name)

__all__ = [
    "ACTION_SPEED_POLICY_MODES",
    "ActionSpeedDecision",
    "ActionSpeedPolicy",
    "FuturePathProjector",
    "PathProjection",
    "PolylineFuturePath",
    "WaypointPathPlanner",
]
