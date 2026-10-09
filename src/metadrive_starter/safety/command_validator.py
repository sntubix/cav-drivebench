from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from metadrive_starter.perception import LaneRelation, LocalScene, TrackedObject
from metadrive_starter.timing import exceeds_time_limit, precedes_time_limit
from metadrive_starter.vla import HighLevelAction, VLACommand


class CommandDisposition(str, Enum):
    ACCEPTED = "accepted"
    MODIFIED = "modified"
    REJECTED = "rejected"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class CommandValidationSettings:
    minimum_confidence: float = 0.5
    maximum_command_age_s: float = 2.0
    maximum_clock_skew_s: float = 0.25
    scene_stale_after_s: float = 0.2
    maximum_target_speed_mps: float = 13.88888888888889
    slow_down_speed_mps: float = 5.555555555555555
    yield_speed_mps: float = 2.7777777777777777
    lane_change_front_gap_m: float = 12.0
    lane_change_rear_gap_m: float = 10.0
    lane_change_minimum_ttc_s: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        positive = {
            "maximum_command_age_s": self.maximum_command_age_s,
            "scene_stale_after_s": self.scene_stale_after_s,
            "maximum_target_speed_mps": self.maximum_target_speed_mps,
            "slow_down_speed_mps": self.slow_down_speed_mps,
            "yield_speed_mps": self.yield_speed_mps,
            "lane_change_front_gap_m": self.lane_change_front_gap_m,
            "lane_change_rear_gap_m": self.lane_change_rear_gap_m,
            "lane_change_minimum_ttc_s": self.lane_change_minimum_ttc_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_clock_skew_s < 0.0:
            raise ValueError("maximum_clock_skew_s must not be negative")


@dataclass(frozen=True)
class CommandValidationDecision:
    disposition: CommandDisposition
    requested_command: VLACommand | None
    effective_command: VLACommand
    reasons: tuple[str, ...]

    @property
    def intervened(self) -> bool:
        return self.disposition is not CommandDisposition.ACCEPTED


@dataclass(frozen=True)
class LaneChangeClearanceDecision:
    """Deterministic target-lane clearance result used at admission and in flight."""

    clear: bool
    lane_relation: LaneRelation
    reason: str
    object_id: str | None = None
    gap_m: float | None = None
    minimum_gap_m: float | None = None
    ttc_s: float | None = None
    minimum_ttc_s: float | None = None


class VLACommandValidator:
    """Deterministic gate between high-level VLA output and local planning."""

    _LEFT_ACTIONS = {HighLevelAction.CHANGE_LANE_LEFT, HighLevelAction.OVERTAKE}
    _RIGHT_ACTIONS = {HighLevelAction.CHANGE_LANE_RIGHT, HighLevelAction.PULL_OVER}

    def __init__(self, settings: CommandValidationSettings | None = None):
        self.settings = settings or CommandValidationSettings()

    def validate_payload(
        self,
        payload: Mapping[str, object],
        scene: LocalScene,
        *,
        now_s: float,
    ) -> CommandValidationDecision:
        try:
            command = VLACommand.from_payload(payload)
        except (KeyError, TypeError, ValueError) as exc:
            return self._fallback(None, scene, now_s, f"invalid command payload: {exc}")
        return self.validate(command, scene, now_s=now_s)

    def validate(
        self,
        command: VLACommand,
        scene: LocalScene,
        *,
        now_s: float,
    ) -> CommandValidationDecision:
        if not scene.valid:
            return self._fallback(command, scene, now_s, "invalid local scene")
        scene_age_s = now_s - scene.timestamp_s
        if exceeds_time_limit(scene_age_s, self.settings.scene_stale_after_s):
            return self._fallback(command, scene, now_s, f"stale local scene ({scene_age_s:.3f}s old)")

        command_age_s = now_s - command.issued_at_s
        if precedes_time_limit(command_age_s, -self.settings.maximum_clock_skew_s):
            return self._fallback(command, scene, now_s, "command timestamp is in the future")
        if exceeds_time_limit(command_age_s, self.settings.maximum_command_age_s):
            return self._fallback(command, scene, now_s, f"stale command ({command_age_s:.3f}s old)")
        if exceeds_time_limit(command_age_s, command.action_horizon_s):
            return self._fallback(command, scene, now_s, "command action horizon has expired")
        if command.confidence < self.settings.minimum_confidence:
            return self._fallback(command, scene, now_s, "command confidence is below threshold")
        if command.action is HighLevelAction.REQUEST_FALLBACK:
            return self._fallback(command, scene, now_s, "VLA requested local fallback")

        if command.action in self._LEFT_ACTIONS:
            clearance = self.lane_change_clearance(
                scene,
                LaneRelation.LEFT,
                now_s=now_s,
            )
            if not clearance.clear:
                return self._reject_lane_change(command, scene, clearance.reason)
        elif command.action in self._RIGHT_ACTIONS:
            clearance = self.lane_change_clearance(
                scene,
                LaneRelation.RIGHT,
                now_s=now_s,
            )
            if not clearance.clear:
                return self._reject_lane_change(command, scene, clearance.reason)

        effective = command
        reasons: list[str] = []
        target_speed = min(command.target_speed_mps, self.settings.maximum_target_speed_mps)
        if target_speed != command.target_speed_mps:
            reasons.append("target speed limited by local maximum")
        if command.action is HighLevelAction.STOP:
            target_speed = 0.0
            if command.target_speed_mps != 0.0:
                reasons.append("STOP target speed forced to zero")
        elif command.action is HighLevelAction.SLOW_DOWN:
            limited = min(target_speed, self.settings.slow_down_speed_mps)
            if limited != target_speed:
                reasons.append("SLOW_DOWN target speed limited")
            target_speed = limited
        elif command.action is HighLevelAction.YIELD:
            limited = min(target_speed, self.settings.yield_speed_mps)
            if limited != target_speed:
                reasons.append("YIELD target speed limited")
            target_speed = limited

        if target_speed != command.target_speed_mps:
            effective = replace(command, target_speed_mps=target_speed)
        disposition = CommandDisposition.MODIFIED if reasons else CommandDisposition.ACCEPTED
        return CommandValidationDecision(disposition, command, effective, tuple(reasons or ["command accepted"]))

    def lane_change_clearance(
        self,
        scene: LocalScene,
        relation: LaneRelation,
        *,
        now_s: float,
        include_planned_path: bool = False,
    ) -> LaneChangeClearanceDecision:
        if not isinstance(scene, LocalScene):
            raise TypeError("scene must be a LocalScene")
        if relation not in {LaneRelation.LEFT, LaneRelation.RIGHT, LaneRelation.SAME}:
            raise ValueError("lane-change clearance relation must be left, right, or same")
        if not isinstance(include_planned_path, bool):
            raise ValueError("include_planned_path must be a boolean")
        if (
            isinstance(now_s, bool)
            or not isinstance(now_s, (int, float))
            or not math.isfinite(now_s)
            or now_s < 0.0
        ):
            raise ValueError("now_s must be finite and non-negative")
        if not scene.valid:
            return LaneChangeClearanceDecision(
                False,
                relation,
                "local scene is invalid during lane change",
            )
        scene_age_s = now_s - scene.timestamp_s
        if precedes_time_limit(scene_age_s, 0.0):
            return LaneChangeClearanceDecision(
                False,
                relation,
                "local scene timestamp is in the future during lane change",
            )
        if exceeds_time_limit(scene_age_s, self.settings.scene_stale_after_s):
            return LaneChangeClearanceDecision(
                False,
                relation,
                "local scene is stale during lane change",
            )

        if relation in {LaneRelation.LEFT, LaneRelation.RIGHT}:
            lane_available = (
                scene.left_lane_available
                if relation is LaneRelation.LEFT
                else scene.right_lane_available
            )
            if lane_available is not True:
                side = "left" if relation is LaneRelation.LEFT else "right"
                return LaneChangeClearanceDecision(
                    False,
                    relation,
                    f"{side} lane is unavailable or unknown",
                )

        for tracked in sorted(scene.objects, key=lambda item: item.object_id):
            if tracked.lane_relation is not relation and not (
                include_planned_path and tracked.in_path
            ):
                continue
            clearance = self._adjacent_object_clearance(scene, tracked, relation)
            if not clearance.clear:
                return clearance
        return LaneChangeClearanceDecision(
            True,
            relation,
            "target lane clearance requirements satisfied",
        )

    def _adjacent_object_clearance(
        self,
        scene: LocalScene,
        tracked: TrackedObject,
        relation: LaneRelation,
    ) -> LaneChangeClearanceDecision:
        center_distance = tracked.relative_position_m[0]
        footprint_m = (scene.ego_length_m + tracked.length_m) / 2.0
        relative_speed_mps = tracked.relative_velocity_mps[0]
        if center_distance >= 0.0:
            gap_m = max(0.0, center_distance - footprint_m)
            closing_speed_mps = max(0.0, -relative_speed_mps)
            minimum_gap_m = self.settings.lane_change_front_gap_m
        else:
            gap_m = max(0.0, -center_distance - footprint_m)
            closing_speed_mps = max(0.0, relative_speed_mps)
            minimum_gap_m = self.settings.lane_change_rear_gap_m

        ttc_s = gap_m / closing_speed_mps if closing_speed_mps > 1e-6 else None
        if gap_m < minimum_gap_m:
            return LaneChangeClearanceDecision(
                False,
                relation,
                f"object {tracked.object_id} violates lane-change gap",
                object_id=tracked.object_id,
                gap_m=gap_m,
                minimum_gap_m=minimum_gap_m,
                ttc_s=ttc_s,
                minimum_ttc_s=self.settings.lane_change_minimum_ttc_s,
            )
        if ttc_s is not None and ttc_s < self.settings.lane_change_minimum_ttc_s:
            return LaneChangeClearanceDecision(
                False,
                relation,
                f"object {tracked.object_id} violates lane-change TTC",
                object_id=tracked.object_id,
                gap_m=gap_m,
                minimum_gap_m=minimum_gap_m,
                ttc_s=ttc_s,
                minimum_ttc_s=self.settings.lane_change_minimum_ttc_s,
            )
        return LaneChangeClearanceDecision(
            True,
            relation,
            f"object {tracked.object_id} satisfies lane-change clearance",
            object_id=tracked.object_id,
            gap_m=gap_m,
            minimum_gap_m=minimum_gap_m,
            ttc_s=ttc_s,
            minimum_ttc_s=self.settings.lane_change_minimum_ttc_s,
        )

    def _reject_lane_change(
        self,
        command: VLACommand,
        scene: LocalScene,
        reason: str,
    ) -> CommandValidationDecision:
        effective = replace(
            command,
            action=HighLevelAction.KEEP_LANE,
            target_speed_mps=min(
                command.target_speed_mps,
                scene.ego_speed_mps,
                self.settings.maximum_target_speed_mps,
            ),
        )
        return CommandValidationDecision(
            disposition=CommandDisposition.REJECTED,
            requested_command=command,
            effective_command=effective,
            reasons=(reason, "unsafe lateral action replaced with KEEP_LANE"),
        )

    def _fallback(
        self,
        command: VLACommand | None,
        scene: LocalScene,
        now_s: float,
        reason: str,
    ) -> CommandValidationDecision:
        effective = VLACommand(
            action=HighLevelAction.REQUEST_FALLBACK,
            target_speed_mps=min(
                scene.ego_speed_mps,
                self.settings.maximum_target_speed_mps,
            ),
            issued_at_s=now_s,
            action_horizon_s=0.1,
            confidence=1.0,
            command_id=command.command_id if command is not None else "",
            justification=reason,
        )
        return CommandValidationDecision(
            disposition=CommandDisposition.FALLBACK,
            requested_command=command,
            effective_command=effective,
            reasons=(reason,),
        )
