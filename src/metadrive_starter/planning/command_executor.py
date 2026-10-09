from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from metadrive_starter.safety import CommandValidationDecision
from metadrive_starter.timing import exceeds_time_limit, precedes_time_limit
from metadrive_starter.vla.commands import HighLevelAction

from metadrive_starter.planning.action_speed import (
    ActionSpeedDecision,
    ActionSpeedPolicy,
)


class CommandExecutionSource(str, Enum):
    VLA = "vla"
    LOCAL_FALLBACK = "local_fallback"
    UNSUPPORTED_ACTION = "unsupported_action"


@dataclass(frozen=True)
class CommandExecutionDecision:
    """Planner/controller objective derived only from a validated command."""

    source: CommandExecutionSource
    action: HighLevelAction
    target_speed_mps: float
    command_id: str
    reason: str
    valid_until_s: float | None = None
    action_speed: ActionSpeedDecision | None = None


class VLACommandExecutor:
    """Map validated high-level commands onto the current fixed-route controller.

    The baseline planner has no lateral manoeuvre API. Accepted lane-change,
    overtake, and pull-over commands therefore fail closed to the conventional
    local speed objective instead of being approximated with steering offsets.
    """

    _LONGITUDINAL_ACTIONS = {
        HighLevelAction.KEEP_LANE,
        HighLevelAction.FOLLOW,
        HighLevelAction.SLOW_DOWN,
        HighLevelAction.STOP,
        HighLevelAction.YIELD,
    }
    _UNSUPPORTED_ACTIONS = {
        HighLevelAction.CHANGE_LANE_LEFT,
        HighLevelAction.CHANGE_LANE_RIGHT,
        HighLevelAction.OVERTAKE,
        HighLevelAction.PULL_OVER,
    }

    def __init__(
        self,
        fallback_target_speed_mps: float,
        *,
        action_speed_policy: ActionSpeedPolicy | None = None,
        lane_change_enabled: bool = False,
    ) -> None:
        if not _non_negative_finite(fallback_target_speed_mps):
            raise ValueError("fallback_target_speed_mps must be finite and non-negative")
        self.fallback_target_speed_mps = float(fallback_target_speed_mps)
        self.action_speed_policy = action_speed_policy or ActionSpeedPolicy(
            "off",
            cruise_speed_mps=self.fallback_target_speed_mps,
            slow_down_speed_mps=self.fallback_target_speed_mps,
            yield_speed_mps=self.fallback_target_speed_mps,
        )
        if not isinstance(self.action_speed_policy, ActionSpeedPolicy):
            raise TypeError("action_speed_policy must be an ActionSpeedPolicy")
        if not isinstance(lane_change_enabled, bool):
            raise ValueError("lane_change_enabled must be a boolean")
        self.lane_change_enabled = lane_change_enabled

    def execute(
        self,
        decision: CommandValidationDecision | None,
        *,
        now_s: float,
    ) -> CommandExecutionDecision:
        if not _non_negative_finite(now_s):
            raise ValueError("now_s must be finite and non-negative")
        if decision is None:
            return self._fallback("no validated VLA command is available")
        if not isinstance(decision, CommandValidationDecision):
            raise TypeError("decision must be a CommandValidationDecision or None")

        command = decision.effective_command
        age_s = float(now_s) - command.issued_at_s
        if precedes_time_limit(age_s, 0.0):
            return self._fallback(
                "effective command timestamp is in the future",
                command_id=command.command_id,
            )
        if exceeds_time_limit(age_s, command.action_horizon_s):
            return self._fallback(
                "effective command action horizon has expired",
                command_id=command.command_id,
            )
        if command.action is HighLevelAction.REQUEST_FALLBACK:
            return self._fallback(
                "validated command requests conventional local fallback",
                command_id=command.command_id,
            )
        if command.action in self._UNSUPPORTED_ACTIONS and not (
            self.lane_change_enabled
            and command.action
            in {HighLevelAction.CHANGE_LANE_LEFT, HighLevelAction.CHANGE_LANE_RIGHT}
        ):
            return CommandExecutionDecision(
                source=CommandExecutionSource.UNSUPPORTED_ACTION,
                action=HighLevelAction.REQUEST_FALLBACK,
                target_speed_mps=self.fallback_target_speed_mps,
                command_id=command.command_id,
                reason=f"{command.action.value} is unsupported by the fixed-route planner",
            )
        if command.action not in self._LONGITUDINAL_ACTIONS and command.action not in {
            HighLevelAction.CHANGE_LANE_LEFT,
            HighLevelAction.CHANGE_LANE_RIGHT,
        }:
            return self._fallback(
                "effective command action is unknown to the executor",
                command_id=command.command_id,
            )
        action_speed = self.action_speed_policy.evaluate(
            command.action,
            command.target_speed_mps,
        )
        return CommandExecutionDecision(
            source=CommandExecutionSource.VLA,
            action=command.action,
            target_speed_mps=action_speed.effective_target_speed_mps,
            command_id=command.command_id,
            reason="validated effective command applied",
            valid_until_s=command.issued_at_s + command.action_horizon_s,
            action_speed=action_speed,
        )

    def _fallback(
        self,
        reason: str,
        *,
        command_id: str = "",
    ) -> CommandExecutionDecision:
        return CommandExecutionDecision(
            source=CommandExecutionSource.LOCAL_FALLBACK,
            action=HighLevelAction.REQUEST_FALLBACK,
            target_speed_mps=self.fallback_target_speed_mps,
            command_id=command_id,
            reason=reason,
        )


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )
