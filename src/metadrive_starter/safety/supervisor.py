from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from metadrive_starter.perception import LocalScene, TrackedObject
from metadrive_starter.types import ControlCommand


class SafetyLevel(str, Enum):
    CLEAR = "clear"
    CAUTION = "caution"
    BRAKE = "brake"
    EMERGENCY = "emergency"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class SafetySettings:
    enabled: bool = False
    caution_ttc_s: float = 3.0
    emergency_ttc_s: float = 1.5
    minimum_gap_m: float = 3.0
    minimum_headway_s: float = 1.5
    reaction_time_s: float = 0.5
    maximum_deceleration_mps2: float = 6.0
    stale_after_s: float = 0.2
    release_hold_s: float = 0.3
    service_brake: float = 0.5
    headway_speed_cap_mode: str = "off"

    def __post_init__(self) -> None:
        positive = {
            "caution_ttc_s": self.caution_ttc_s,
            "emergency_ttc_s": self.emergency_ttc_s,
            "minimum_gap_m": self.minimum_gap_m,
            "minimum_headway_s": self.minimum_headway_s,
            "reaction_time_s": self.reaction_time_s,
            "maximum_deceleration_mps2": self.maximum_deceleration_mps2,
            "stale_after_s": self.stale_after_s,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.emergency_ttc_s >= self.caution_ttc_s:
            raise ValueError("emergency_ttc_s must be below caution_ttc_s")
        if self.release_hold_s < 0.0:
            raise ValueError("release_hold_s must not be negative")
        if not 0.0 < self.service_brake < 1.0:
            raise ValueError("service_brake must be between 0 and 1")
        if self.headway_speed_cap_mode not in {"off", "shadow", "enforce"}:
            raise ValueError(
                "headway_speed_cap_mode must be 'off', 'shadow', or 'enforce'"
            )


@dataclass(frozen=True)
class SafetyDecision:
    level: SafetyLevel
    command: ControlCommand
    reason: str
    intervened: bool
    minimum_ttc_s: float | None = None
    minimum_gap_m: float | None = None
    threat_object_id: str | None = None


@dataclass(frozen=True)
class _Threat:
    object_id: str
    level: SafetyLevel
    gap_m: float
    ttc_s: float | None


class EmergencyBrakingSupervisor:
    """Deterministic longitudinal safety filter over a local scene snapshot."""

    _SEVERITY = {
        SafetyLevel.CLEAR: 0,
        SafetyLevel.CAUTION: 1,
        SafetyLevel.BRAKE: 2,
        SafetyLevel.EMERGENCY: 3,
    }

    def __init__(self, settings: SafetySettings | None = None):
        self.settings = settings or SafetySettings()
        self._hold_brake_until_s = -math.inf

    def evaluate(
        self,
        scene: LocalScene,
        proposed: ControlCommand,
        *,
        now_s: float,
    ) -> SafetyDecision:
        age_s = now_s - scene.timestamp_s
        if not scene.valid:
            return self._degraded(proposed, "invalid perception scene")
        if age_s > self.settings.stale_after_s:
            return self._degraded(proposed, f"stale perception scene ({age_s:.3f}s old)")

        threats = [
            self._assess_object(scene, tracked_object)
            for tracked_object in scene.objects
            if tracked_object.in_path and _path_distance(tracked_object) > 0.0
        ]
        threats = [threat for threat in threats if threat is not None]

        if threats:
            threat = max(threats, key=lambda item: self._SEVERITY[item.level])
            minimum_ttc = min(
                (item.ttc_s for item in threats if item.ttc_s is not None),
                default=None,
            )
            minimum_gap = min(item.gap_m for item in threats)
            if threat.level in {SafetyLevel.BRAKE, SafetyLevel.EMERGENCY}:
                self._hold_brake_until_s = max(
                    self._hold_brake_until_s,
                    now_s + self.settings.release_hold_s,
                )
            return self._decision(
                threat.level,
                proposed,
                f"object {threat.object_id} requires {threat.level.value}",
                minimum_ttc_s=minimum_ttc,
                minimum_gap_m=minimum_gap,
                threat_object_id=threat.object_id,
            )

        if now_s < self._hold_brake_until_s:
            return self._decision(SafetyLevel.BRAKE, proposed, "brake release hold")

        return self._decision(SafetyLevel.CLEAR, proposed, "no hazard in predicted path")

    def _assess_object(self, scene: LocalScene, tracked_object: TrackedObject) -> _Threat | None:
        center_distance = _path_distance(tracked_object)
        gap_m = max(0.0, center_distance - (scene.ego_length_m + tracked_object.length_m) / 2.0)
        relative_velocity_mps = (
            tracked_object.path_relative_velocity_mps
            if tracked_object.path_relative_velocity_mps is not None
            else tracked_object.relative_velocity_mps[0]
        )
        closing_speed_mps = max(0.0, -relative_velocity_mps)
        ttc_s = gap_m / closing_speed_mps if closing_speed_mps > 1e-6 else None

        if gap_m <= self.settings.minimum_gap_m or (
            ttc_s is not None and ttc_s <= self.settings.emergency_ttc_s
        ):
            level = SafetyLevel.EMERGENCY
        elif closing_speed_mps > 0.0 and (
            gap_m <= self._stopping_distance(scene.ego_speed_mps)
            or (ttc_s is not None and ttc_s <= self.settings.caution_ttc_s)
        ):
            level = SafetyLevel.BRAKE
        elif scene.ego_speed_mps > 1e-6 and gap_m / scene.ego_speed_mps <= self.settings.minimum_headway_s:
            level = SafetyLevel.CAUTION
        else:
            level = SafetyLevel.CLEAR

        return _Threat(tracked_object.object_id, level, gap_m, ttc_s)

    def _stopping_distance(self, speed_mps: float) -> float:
        reaction_distance = speed_mps * self.settings.reaction_time_s
        braking_distance = speed_mps**2 / (2.0 * self.settings.maximum_deceleration_mps2)
        return reaction_distance + braking_distance + self.settings.minimum_gap_m

    def _degraded(self, proposed: ControlCommand, reason: str) -> SafetyDecision:
        command = ControlCommand(steering=proposed.steering, throttle=0.0, brake=1.0)
        return SafetyDecision(
            level=SafetyLevel.DEGRADED,
            command=command,
            reason=reason,
            intervened=command != proposed,
        )

    def _decision(
        self,
        level: SafetyLevel,
        proposed: ControlCommand,
        reason: str,
        *,
        minimum_ttc_s: float | None = None,
        minimum_gap_m: float | None = None,
        threat_object_id: str | None = None,
    ) -> SafetyDecision:
        if level is SafetyLevel.EMERGENCY:
            command = ControlCommand(steering=proposed.steering, throttle=0.0, brake=1.0)
        elif level is SafetyLevel.BRAKE:
            command = ControlCommand(
                steering=proposed.steering,
                throttle=0.0,
                brake=max(proposed.brake, self.settings.service_brake),
            )
        elif level is SafetyLevel.CAUTION:
            command = ControlCommand(steering=proposed.steering, throttle=0.0, brake=proposed.brake)
        else:
            command = proposed
        return SafetyDecision(
            level=level,
            command=command,
            reason=reason,
            intervened=command != proposed,
            minimum_ttc_s=minimum_ttc_s,
            minimum_gap_m=minimum_gap_m,
            threat_object_id=threat_object_id,
        )


def _path_distance(tracked_object: TrackedObject) -> float:
    if tracked_object.path_distance_m is not None:
        return tracked_object.path_distance_m
    return tracked_object.relative_position_m[0]
