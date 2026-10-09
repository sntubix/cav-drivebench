from __future__ import annotations

import math
from dataclasses import dataclass

from metadrive_starter.vla.commands import HighLevelAction


ACTION_SPEED_POLICY_MODES = frozenset({"off", "shadow", "enforce"})


@dataclass(frozen=True)
class ActionSpeedDecision:
    """Deterministic speed interpretation of one high-level VLA action."""

    mode: str
    action: HighLevelAction
    requested_target_speed_mps: float
    mapped_target_speed_mps: float | None
    effective_target_speed_mps: float
    would_intervene: bool
    applied: bool
    reason: str


@dataclass(frozen=True)
class ActionSpeedPolicy:
    """Map semantic actions to local speed objectives without model actuator authority."""

    mode: str
    cruise_speed_mps: float
    slow_down_speed_mps: float
    yield_speed_mps: float

    def __post_init__(self) -> None:
        if self.mode not in ACTION_SPEED_POLICY_MODES:
            raise ValueError("action speed policy mode must be off, shadow, or enforce")
        for name in (
            "cruise_speed_mps",
            "slow_down_speed_mps",
            "yield_speed_mps",
        ):
            if not _non_negative_finite(getattr(self, name)):
                raise ValueError(f"{name} must be finite and non-negative")

    def evaluate(
        self,
        action: HighLevelAction,
        requested_target_speed_mps: float,
    ) -> ActionSpeedDecision:
        if not isinstance(action, HighLevelAction):
            raise TypeError("action must be a HighLevelAction")
        if not _non_negative_finite(requested_target_speed_mps):
            raise ValueError("requested_target_speed_mps must be finite and non-negative")
        requested = float(requested_target_speed_mps)
        if self.mode == "off":
            return ActionSpeedDecision(
                mode=self.mode,
                action=action,
                requested_target_speed_mps=requested,
                mapped_target_speed_mps=None,
                effective_target_speed_mps=requested,
                would_intervene=False,
                applied=False,
                reason="action speed policy is disabled",
            )

        mapped = self._mapped_speed(action)
        would_intervene = not math.isclose(
            requested,
            mapped,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        applied = self.mode == "enforce" and would_intervene
        return ActionSpeedDecision(
            mode=self.mode,
            action=action,
            requested_target_speed_mps=requested,
            mapped_target_speed_mps=mapped,
            effective_target_speed_mps=mapped if self.mode == "enforce" else requested,
            would_intervene=would_intervene,
            applied=applied,
            reason=(
                "locally mapped action speed applied"
                if applied
                else "locally mapped action speed differs; shadow only"
                if would_intervene
                else "model target matches locally mapped action speed"
            ),
        )

    def _mapped_speed(self, action: HighLevelAction) -> float:
        if action in {
            HighLevelAction.KEEP_LANE,
            HighLevelAction.FOLLOW,
            HighLevelAction.CHANGE_LANE_LEFT,
            HighLevelAction.CHANGE_LANE_RIGHT,
            HighLevelAction.OVERTAKE,
        }:
            return float(self.cruise_speed_mps)
        if action is HighLevelAction.SLOW_DOWN:
            return float(self.slow_down_speed_mps)
        if action is HighLevelAction.YIELD:
            return float(self.yield_speed_mps)
        if action in {HighLevelAction.STOP, HighLevelAction.PULL_OVER}:
            return 0.0
        return float(self.cruise_speed_mps)


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )
