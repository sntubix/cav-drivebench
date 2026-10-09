from __future__ import annotations

from metadrive_starter.planning.command_executor import (
    CommandExecutionDecision,
    CommandExecutionSource,
)
from metadrive_starter.planning.lane_change import (
    LaneChangeCoordinator,
    LaneChangePhase,
)
from metadrive_starter.vla import HighLevelAction


def _execution(
    action: HighLevelAction,
    *,
    command_id: str = "lane-1",
) -> CommandExecutionDecision:
    return CommandExecutionDecision(
        source=CommandExecutionSource.VLA,
        action=action,
        target_speed_mps=5.0,
        command_id=command_id,
        reason="test",
    )


def test_lane_change_latches_then_completes_when_centered() -> None:
    coordinator = LaneChangeCoordinator(timeout_s=8.0, completion_tolerance_m=0.6)

    started = coordinator.update(
        _execution(HighLevelAction.CHANGE_LANE_RIGHT),
        now_s=1.0,
        current_lane_index=0,
        lane_count=2,
    )
    active = coordinator.update(
        _execution(HighLevelAction.KEEP_LANE, command_id="keep-2"),
        now_s=2.0,
        current_lane_index=0,
        lane_count=2,
        target_lane_offset_m=-1.2,
    )
    completed = coordinator.update(
        None,
        now_s=3.0,
        current_lane_index=1,
        lane_count=2,
        target_lane_offset_m=0.2,
    )

    assert started.phase is LaneChangePhase.STARTED
    assert started.target_lane_index == 1
    assert started.route_update_required is True
    assert active.phase is LaneChangePhase.ACTIVE
    assert active.command_id == "lane-1"
    assert completed.phase is LaneChangePhase.COMPLETED
    assert completed.route_update_required is True


def test_lane_change_times_out_and_same_command_does_not_restart() -> None:
    coordinator = LaneChangeCoordinator(timeout_s=2.0, completion_tolerance_m=0.5)
    command = _execution(HighLevelAction.CHANGE_LANE_RIGHT)
    coordinator.update(command, now_s=1.0, current_lane_index=0, lane_count=2)

    aborted = coordinator.update(
        command,
        now_s=3.0,
        current_lane_index=0,
        lane_count=2,
        target_lane_offset_m=-2.0,
    )
    repeated = coordinator.update(
        command,
        now_s=3.1,
        current_lane_index=0,
        lane_count=2,
    )

    assert aborted.phase is LaneChangePhase.ABORTED
    assert repeated.phase is LaneChangePhase.IDLE


def test_lane_change_rejects_road_edge_and_accepts_new_command_id() -> None:
    coordinator = LaneChangeCoordinator(timeout_s=8.0, completion_tolerance_m=0.6)

    rejected = coordinator.update(
        _execution(HighLevelAction.CHANGE_LANE_LEFT),
        now_s=0.0,
        current_lane_index=0,
        lane_count=2,
    )
    started = coordinator.update(
        _execution(HighLevelAction.CHANGE_LANE_RIGHT, command_id="lane-2"),
        now_s=0.1,
        current_lane_index=0,
        lane_count=2,
    )

    assert rejected.phase is LaneChangePhase.REJECTED
    assert started.phase is LaneChangePhase.STARTED


def test_non_vla_or_non_lane_execution_stays_idle() -> None:
    coordinator = LaneChangeCoordinator(timeout_s=8.0, completion_tolerance_m=0.6)
    fallback = CommandExecutionDecision(
        source=CommandExecutionSource.LOCAL_FALLBACK,
        action=HighLevelAction.CHANGE_LANE_LEFT,
        target_speed_mps=5.0,
        command_id="fallback",
        reason="test",
    )

    assert coordinator.update(
        fallback,
        now_s=0.0,
        current_lane_index=0,
        lane_count=2,
    ).phase is LaneChangePhase.IDLE
    assert coordinator.update(
        _execution(HighLevelAction.KEEP_LANE),
        now_s=0.1,
        current_lane_index=0,
        lane_count=2,
    ).phase is LaneChangePhase.IDLE


def test_active_lane_change_aborts_when_clearance_becomes_unsafe() -> None:
    coordinator = LaneChangeCoordinator(timeout_s=8.0, completion_tolerance_m=0.6)
    coordinator.update(
        _execution(HighLevelAction.CHANGE_LANE_RIGHT),
        now_s=1.0,
        current_lane_index=0,
        lane_count=2,
    )

    aborted = coordinator.update(
        None,
        now_s=1.1,
        current_lane_index=0,
        lane_count=2,
        target_lane_offset_m=-2.0,
        target_lane_clear=False,
        clearance_reason="object fast-rear violates lane-change TTC",
    )

    assert aborted.phase is LaneChangePhase.ABORTED
    assert aborted.route_update_required is True
    assert aborted.reason == "object fast-rear violates lane-change TTC"
    assert coordinator.target_lane_index is None


def test_active_lane_change_aborts_when_target_lane_disappears() -> None:
    coordinator = LaneChangeCoordinator(timeout_s=8.0, completion_tolerance_m=0.6)
    coordinator.update(
        _execution(HighLevelAction.CHANGE_LANE_RIGHT),
        now_s=1.0,
        current_lane_index=0,
        lane_count=2,
    )

    aborted = coordinator.update(
        None,
        now_s=1.1,
        current_lane_index=0,
        lane_count=1,
    )

    assert aborted.phase is LaneChangePhase.ABORTED
    assert aborted.reason == "target lane became unavailable during lane change"
