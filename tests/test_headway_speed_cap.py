from __future__ import annotations

import pytest

from metadrive_starter.perception import LaneRelation, LocalScene, TrackedObject
from metadrive_starter.safety import HeadwaySpeedCapMode, TimeHeadwaySpeedCap


def _tracked(
    object_id: str,
    *,
    center_distance_m: float,
    relative_speed_mps: float,
    in_path: bool = True,
) -> TrackedObject:
    return TrackedObject(
        object_id=object_id,
        kind="vehicle",
        relative_position_m=(center_distance_m + 50.0, 0.0),
        relative_velocity_mps=(relative_speed_mps + 10.0, 0.0),
        length_m=4.0,
        width_m=1.8,
        lane_relation=LaneRelation.SAME,
        in_path=in_path,
        path_distance_m=center_distance_m,
        path_relative_velocity_mps=relative_speed_mps,
    )


def _scene(*objects: TrackedObject, timestamp_s: float = 1.0, valid: bool = True) -> LocalScene:
    return LocalScene(
        timestamp_s=timestamp_s,
        ego_speed_mps=10.0,
        ego_length_m=4.0,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        objects=objects,
        valid=valid,
    )


def _cap(mode: str) -> TimeHeadwaySpeedCap:
    return TimeHeadwaySpeedCap(
        mode=mode,
        minimum_gap_m=3.0,
        time_headway_s=1.5,
        scene_stale_after_s=0.2,
    )


@pytest.mark.parametrize(
    ("mode", "effective_mps", "applied"),
    [("shadow", 15.0, False), ("enforce", 10.0, True)],
)
def test_shadow_reports_and_enforce_applies_same_headway_cap(
    mode: str,
    effective_mps: float,
    applied: bool,
) -> None:
    # Footprint gap 18 m = 3 m standstill gap + 1.5 s * 10 m/s.
    decision = _cap(mode).evaluate(
        _scene(_tracked("lead", center_distance_m=22.0, relative_speed_mps=0.0)),
        15.0,
        now_s=1.0,
    )

    assert decision.mode is HeadwaySpeedCapMode(mode)
    assert decision.calculated_cap_mps == pytest.approx(10.0)
    assert decision.effective_target_speed_mps == pytest.approx(effective_mps)
    assert decision.would_intervene is True
    assert decision.applied is applied
    assert decision.lead_gap_m == pytest.approx(18.0)
    assert decision.desired_gap_m == pytest.approx(18.0)


def test_off_mode_has_no_calculation_or_behavior_change() -> None:
    decision = _cap("off").evaluate(
        _scene(_tracked("lead", center_distance_m=8.0, relative_speed_mps=-10.0)),
        15.0,
        now_s=1.0,
    )

    assert decision.available is False
    assert decision.calculated_cap_mps is None
    assert decision.effective_target_speed_mps == 15.0


def test_cap_uses_path_fields_nearest_lead_and_clamps_at_zero() -> None:
    decision = _cap("enforce").evaluate(
        _scene(
            _tracked("far", center_distance_m=30.0, relative_speed_mps=-1.0),
            _tracked("receding", center_distance_m=35.0, relative_speed_mps=1.0),
            _tracked("adjacent", center_distance_m=5.0, relative_speed_mps=-1.0, in_path=False),
            _tracked("near", center_distance_m=8.0, relative_speed_mps=-4.0),
        ),
        12.0,
        now_s=1.0,
    )

    assert decision.lead_object_id == "near"
    assert decision.lead_gap_m == pytest.approx(4.0)
    assert decision.lead_speed_mps == pytest.approx(6.0)
    assert decision.calculated_cap_mps == 0.0
    assert decision.effective_target_speed_mps == 0.0


def test_cap_considers_receding_lead_when_proposed_target_would_close_again() -> None:
    decision = _cap("enforce").evaluate(
        _scene(_tracked("lead", center_distance_m=14.0, relative_speed_mps=1.0)),
        15.0,
        now_s=1.0,
    )

    assert decision.lead_speed_mps == 11.0
    assert decision.calculated_cap_mps == pytest.approx(11.0 + (10.0 - 18.0) / 1.5)
    assert decision.would_intervene is True
    assert decision.applied is True


@pytest.mark.parametrize(
    ("scene", "reason"),
    [
        (_scene(), "no lead"),
        (_scene(valid=False), "invalid"),
        (_scene(timestamp_s=0.0), "stale"),
    ],
)
def test_cap_fails_open_to_emergency_layer_when_scene_is_unavailable(
    scene: LocalScene,
    reason: str,
) -> None:
    decision = _cap("enforce").evaluate(scene, 12.0, now_s=1.0)

    assert decision.available is False
    assert decision.effective_target_speed_mps == 12.0
    assert reason in decision.reason


@pytest.mark.parametrize("mode", ["invalid", "", 7])
def test_cap_rejects_unknown_mode(mode: object) -> None:
    with pytest.raises(ValueError, match="mode"):
        TimeHeadwaySpeedCap(
            mode=mode,  # type: ignore[arg-type]
            minimum_gap_m=3.0,
            time_headway_s=1.5,
            scene_stale_after_s=0.2,
        )
