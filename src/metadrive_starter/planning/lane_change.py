from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from metadrive_starter.planning.command_executor import (
    CommandExecutionDecision,
    CommandExecutionSource,
)
from metadrive_starter.vla.commands import HighLevelAction


class LaneChangePhase(str, Enum):
    IDLE = "idle"
    STARTED = "started"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class LaneChangeDecision:
    """One deterministic update of the locally executed lane-change manoeuvre."""

    phase: LaneChangePhase
    action: HighLevelAction | None
    command_id: str
    source_lane_index: int | None
    target_lane_index: int | None
    started_at_s: float | None
    elapsed_s: float | None
    route_update_required: bool
    reason: str

    @property
    def active(self) -> bool:
        return self.phase in {LaneChangePhase.STARTED, LaneChangePhase.ACTIVE}


@dataclass
class _ActiveLaneChange:
    action: HighLevelAction
    command_id: str
    source_lane_index: int
    target_lane_index: int
    started_at_s: float


class LaneChangeCoordinator:
    """Latch one validated lane request until completion or timeout.

    VLA selects direction. Local route planner owns the continuous steering path.
    Repeated ticks carrying the same command cannot restart a finished manoeuvre.
    """

    def __init__(self, *, timeout_s: float, completion_tolerance_m: float) -> None:
        if not _positive_finite(timeout_s):
            raise ValueError("timeout_s must be finite and positive")
        if not _positive_finite(completion_tolerance_m):
            raise ValueError("completion_tolerance_m must be finite and positive")
        self.timeout_s = float(timeout_s)
        self.completion_tolerance_m = float(completion_tolerance_m)
        self._active: _ActiveLaneChange | None = None
        self._handled_command_id: str | None = None

    @property
    def target_lane_index(self) -> int | None:
        return self._active.target_lane_index if self._active is not None else None

    def abort(self, *, now_s: float, reason: str) -> LaneChangeDecision:
        if not _non_negative_finite(now_s):
            raise ValueError("now_s must be finite and non-negative")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must not be empty")
        if self._active is None:
            return self._idle(reason)
        active = self._active
        self._active = None
        return self._decision(
            LaneChangePhase.ABORTED,
            active,
            float(now_s) - active.started_at_s,
            route_update_required=True,
            reason=reason,
        )

    def update(
        self,
        execution: CommandExecutionDecision | None,
        *,
        now_s: float,
        current_lane_index: int,
        lane_count: int,
        target_lane_offset_m: float | None = None,
        target_lane_clear: bool | None = None,
        clearance_reason: str | None = None,
    ) -> LaneChangeDecision:
        if not _non_negative_finite(now_s):
            raise ValueError("now_s must be finite and non-negative")
        if isinstance(current_lane_index, bool) or not isinstance(current_lane_index, int):
            raise ValueError("current_lane_index must be an integer")
        if isinstance(lane_count, bool) or not isinstance(lane_count, int) or lane_count <= 0:
            raise ValueError("lane_count must be a positive integer")
        if target_lane_offset_m is not None and not math.isfinite(target_lane_offset_m):
            raise ValueError("target_lane_offset_m must be finite when provided")
        if target_lane_clear is not None and not isinstance(target_lane_clear, bool):
            raise ValueError("target_lane_clear must be a boolean or None")
        if clearance_reason is not None and (
            not isinstance(clearance_reason, str) or not clearance_reason
        ):
            raise ValueError("clearance_reason must be a non-empty string or None")

        if self._active is not None:
            active = self._active
            elapsed_s = float(now_s) - active.started_at_s
            if not 0 <= active.target_lane_index < lane_count:
                self._active = None
                return self._decision(
                    LaneChangePhase.ABORTED,
                    active,
                    elapsed_s,
                    route_update_required=True,
                    reason="target lane became unavailable during lane change",
                )
            centered = (
                current_lane_index == active.target_lane_index
                and target_lane_offset_m is not None
                and abs(target_lane_offset_m) <= self.completion_tolerance_m
            )
            if centered:
                self._active = None
                return self._decision(
                    LaneChangePhase.COMPLETED,
                    active,
                    elapsed_s,
                    route_update_required=True,
                    reason="ego centered in target lane",
                )
            if target_lane_clear is False:
                self._active = None
                return self._decision(
                    LaneChangePhase.ABORTED,
                    active,
                    elapsed_s,
                    route_update_required=True,
                    reason=clearance_reason or "target lane became unsafe",
                )
            if elapsed_s >= self.timeout_s:
                self._active = None
                return self._decision(
                    LaneChangePhase.ABORTED,
                    active,
                    elapsed_s,
                    route_update_required=True,
                    reason="lane-change timeout reached",
                )
            return self._decision(
                LaneChangePhase.ACTIVE,
                active,
                elapsed_s,
                route_update_required=False,
                reason="local lane-change route remains latched",
            )

        requested_action = (
            execution.action
            if execution is not None
            and execution.source is CommandExecutionSource.VLA
            and execution.action
            in {HighLevelAction.CHANGE_LANE_LEFT, HighLevelAction.CHANGE_LANE_RIGHT}
            else None
        )
        if requested_action is None:
            return self._idle("no new validated lane-change request")
        if execution is None:  # pragma: no cover - narrowed above
            return self._idle("no new validated lane-change request")
        if execution.command_id == self._handled_command_id:
            return self._idle("lane-change command was already handled")

        self._handled_command_id = execution.command_id
        # MetaDrive lane indices increase from physical left to physical right.
        lane_delta = -1 if requested_action is HighLevelAction.CHANGE_LANE_LEFT else 1
        target_lane_index = current_lane_index + lane_delta
        candidate = _ActiveLaneChange(
            action=requested_action,
            command_id=execution.command_id,
            source_lane_index=current_lane_index,
            target_lane_index=target_lane_index,
            started_at_s=float(now_s),
        )
        if not 0 <= target_lane_index < lane_count:
            return self._decision(
                LaneChangePhase.REJECTED,
                candidate,
                0.0,
                route_update_required=False,
                reason="requested adjacent lane is unavailable",
            )

        self._active = candidate
        return self._decision(
            LaneChangePhase.STARTED,
            candidate,
            0.0,
            route_update_required=True,
            reason="validated lane-change request latched by local planner",
        )

    @staticmethod
    def _decision(
        phase: LaneChangePhase,
        active: _ActiveLaneChange,
        elapsed_s: float,
        *,
        route_update_required: bool,
        reason: str,
    ) -> LaneChangeDecision:
        return LaneChangeDecision(
            phase=phase,
            action=active.action,
            command_id=active.command_id,
            source_lane_index=active.source_lane_index,
            target_lane_index=active.target_lane_index,
            started_at_s=active.started_at_s,
            elapsed_s=max(0.0, float(elapsed_s)),
            route_update_required=route_update_required,
            reason=reason,
        )

    @staticmethod
    def _idle(reason: str) -> LaneChangeDecision:
        return LaneChangeDecision(
            phase=LaneChangePhase.IDLE,
            action=None,
            command_id="",
            source_lane_index=None,
            target_lane_index=None,
            started_at_s=None,
            elapsed_s=None,
            route_update_required=False,
            reason=reason,
        )


def _positive_finite(value: object) -> bool:
    return _non_negative_finite(value) and value > 0.0


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )
