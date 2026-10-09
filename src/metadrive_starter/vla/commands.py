from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

class HighLevelAction(str, Enum):
    KEEP_LANE = "KEEP_LANE"
    FOLLOW = "FOLLOW"
    SLOW_DOWN = "SLOW_DOWN"
    STOP = "STOP"
    YIELD = "YIELD"
    CHANGE_LANE_LEFT = "CHANGE_LANE_LEFT"
    CHANGE_LANE_RIGHT = "CHANGE_LANE_RIGHT"
    OVERTAKE = "OVERTAKE"
    PULL_OVER = "PULL_OVER"
    REQUEST_FALLBACK = "REQUEST_FALLBACK"


# The actions the model is offered. No planner executes OVERTAKE or PULL_OVER, so
# the output contract leaves them out; the decoder still accepts them, so
# artifacts that hold them replay, and a command carrying one falls back.
MODEL_ACTIONS: tuple[HighLevelAction, ...] = tuple(
    action
    for action in HighLevelAction
    if action not in {HighLevelAction.OVERTAKE, HighLevelAction.PULL_OVER}
)


@dataclass(frozen=True)
class VLACommand:
    """Validated high-level command received from a local or cloud VLA."""

    action: HighLevelAction
    target_speed_mps: float
    issued_at_s: float
    action_horizon_s: float
    confidence: float
    command_id: str = ""
    justification: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action, HighLevelAction):
            raise ValueError("action must be a HighLevelAction")
        numeric = {
            "target_speed_mps": self.target_speed_mps,
            "issued_at_s": self.issued_at_s,
            "action_horizon_s": self.action_horizon_s,
            "confidence": self.confidence,
        }
        for name, value in numeric.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.target_speed_mps < 0.0:
            raise ValueError("target_speed_mps must not be negative")
        if self.action_horizon_s <= 0.0:
            raise ValueError("action_horizon_s must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> VLACommand:
        required = {
            "action",
            "target_speed_mps",
            "issued_at_s",
            "action_horizon_s",
            "confidence",
        }
        allowed = required | {"command_id", "justification"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"missing VLA command fields: {', '.join(missing)}")
        unknown = sorted(payload.keys() - allowed)
        if unknown:
            raise ValueError(f"unknown VLA command fields: {', '.join(unknown)}")
        try:
            action = HighLevelAction(str(payload["action"]).upper())
        except ValueError as exc:
            raise ValueError(f"unknown VLA action: {payload['action']!r}") from exc

        command_id = payload.get("command_id", "")
        justification = payload.get("justification", "")
        if not isinstance(command_id, str):
            raise ValueError("command_id must be a string")
        if not isinstance(justification, str):
            raise ValueError("justification must be a string")

        return cls(
            action=action,
            target_speed_mps=_number(payload, "target_speed_mps"),
            issued_at_s=_number(payload, "issued_at_s"),
            action_horizon_s=_number(payload, "action_horizon_s"),
            confidence=_number(payload, "confidence"),
            command_id=command_id,
            justification=justification,
        )


def _number(payload: Mapping[str, object], name: str) -> float:
    value = payload[name]
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
