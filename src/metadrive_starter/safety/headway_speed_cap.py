from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from metadrive_starter.perception import LocalScene, TrackedObject
from metadrive_starter.timing import exceeds_time_limit, precedes_time_limit


class HeadwaySpeedCapMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


@dataclass(frozen=True)
class HeadwaySpeedCapDecision:
    mode: HeadwaySpeedCapMode
    requested_target_speed_mps: float
    calculated_cap_mps: float | None
    effective_target_speed_mps: float
    available: bool
    would_intervene: bool
    applied: bool
    reason: str
    lead_object_id: str | None = None
    lead_gap_m: float | None = None
    lead_speed_mps: float | None = None
    desired_gap_m: float | None = None


class TimeHeadwaySpeedCap:
    """Optional proactive VLA speed filter; emergency braking remains final."""

    def __init__(
        self,
        *,
        mode: str | HeadwaySpeedCapMode,
        minimum_gap_m: float,
        time_headway_s: float,
        scene_stale_after_s: float,
    ) -> None:
        try:
            self.mode = HeadwaySpeedCapMode(mode)
        except ValueError as exc:
            raise ValueError("mode must be 'off', 'shadow', or 'enforce'") from exc
        for name, value in {
            "minimum_gap_m": minimum_gap_m,
            "time_headway_s": time_headway_s,
            "scene_stale_after_s": scene_stale_after_s,
        }.items():
            if not _positive_finite(value):
                raise ValueError(f"{name} must be finite and positive")
        self.minimum_gap_m = float(minimum_gap_m)
        self.time_headway_s = float(time_headway_s)
        self.scene_stale_after_s = float(scene_stale_after_s)

    def evaluate(
        self,
        scene: LocalScene,
        requested_target_speed_mps: float,
        *,
        now_s: float,
    ) -> HeadwaySpeedCapDecision:
        if not isinstance(scene, LocalScene):
            raise TypeError("scene must be a LocalScene")
        if not _non_negative_finite(requested_target_speed_mps):
            raise ValueError("requested_target_speed_mps must be finite and non-negative")
        if not _non_negative_finite(now_s):
            raise ValueError("now_s must be finite and non-negative")
        requested = float(requested_target_speed_mps)
        if self.mode is HeadwaySpeedCapMode.OFF:
            return self._unavailable(requested, "headway speed cap is disabled")
        if not scene.valid:
            return self._unavailable(requested, "local scene is invalid")
        scene_age_s = float(now_s) - scene.timestamp_s
        if precedes_time_limit(scene_age_s, 0.0):
            return self._unavailable(requested, "local scene timestamp is in the future")
        if exceeds_time_limit(scene_age_s, self.scene_stale_after_s):
            return self._unavailable(requested, "local scene is stale")

        leads = [
            lead
            for tracked in scene.objects
            if (lead := _lead_state(scene, tracked)) is not None
        ]
        if not leads:
            return self._unavailable(requested, "no lead object in path")
        lead = min(leads, key=lambda item: item.gap_m)
        desired_gap_m = self.minimum_gap_m + self.time_headway_s * scene.ego_speed_mps
        calculated_cap_mps = max(
            0.0,
            lead.speed_mps + (lead.gap_m - desired_gap_m) / self.time_headway_s,
        )
        would_intervene = calculated_cap_mps < requested
        applied = self.mode is HeadwaySpeedCapMode.ENFORCE and would_intervene
        effective = calculated_cap_mps if applied else requested
        return HeadwaySpeedCapDecision(
            mode=self.mode,
            requested_target_speed_mps=requested,
            calculated_cap_mps=calculated_cap_mps,
            effective_target_speed_mps=effective,
            available=True,
            would_intervene=would_intervene,
            applied=applied,
            reason=(
                "headway cap applied"
                if applied
                else "headway cap would intervene in enforce mode"
                if would_intervene
                else "requested target is within the headway cap"
            ),
            lead_object_id=lead.object_id,
            lead_gap_m=lead.gap_m,
            lead_speed_mps=lead.speed_mps,
            desired_gap_m=desired_gap_m,
        )

    def _unavailable(self, requested: float, reason: str) -> HeadwaySpeedCapDecision:
        return HeadwaySpeedCapDecision(
            mode=self.mode,
            requested_target_speed_mps=requested,
            calculated_cap_mps=None,
            effective_target_speed_mps=requested,
            available=False,
            would_intervene=False,
            applied=False,
            reason=reason,
        )


@dataclass(frozen=True)
class _LeadState:
    object_id: str
    gap_m: float
    speed_mps: float


def _lead_state(scene: LocalScene, tracked: TrackedObject) -> _LeadState | None:
    center_distance_m = (
        tracked.path_distance_m
        if tracked.path_distance_m is not None
        else tracked.relative_position_m[0]
    )
    if not tracked.in_path or center_distance_m <= 0.0:
        return None
    relative_speed_mps = (
        tracked.path_relative_velocity_mps
        if tracked.path_relative_velocity_mps is not None
        else tracked.relative_velocity_mps[0]
    )
    gap_m = max(
        0.0,
        center_distance_m - (scene.ego_length_m + tracked.length_m) / 2.0,
    )
    return _LeadState(
        object_id=tracked.object_id,
        gap_m=gap_m,
        speed_mps=max(0.0, scene.ego_speed_mps + relative_speed_mps),
    )


def _positive_finite(value: object) -> bool:
    return _non_negative_finite(value) and value > 0.0  # type: ignore[operator]


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )
