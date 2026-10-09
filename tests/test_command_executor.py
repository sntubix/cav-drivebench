from __future__ import annotations

import pytest

from metadrive_starter.perception import LocalScene
from metadrive_starter.planning.command_executor import (
    CommandExecutionSource,
    VLACommandExecutor,
)
from metadrive_starter.planning import ActionSpeedPolicy
from metadrive_starter.safety import VLACommandValidator
from metadrive_starter.vla import HighLevelAction, VLACommand


def _scene(timestamp_s: float = 1.0) -> LocalScene:
    return LocalScene(
        timestamp_s=timestamp_s,
        ego_speed_mps=5.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
        left_lane_available=True,
        right_lane_available=True,
    )


def _decision(
    action: HighLevelAction,
    *,
    target_speed_mps: float = 15.0,
    issued_at_s: float = 1.0,
    action_horizon_s: float = 2.0,
):
    scene = _scene(issued_at_s)
    command = VLACommand(
        action=action,
        target_speed_mps=target_speed_mps,
        issued_at_s=issued_at_s,
        action_horizon_s=action_horizon_s,
        confidence=0.9,
        command_id="model-7",
    )
    return VLACommandValidator().validate(command, scene, now_s=issued_at_s)


@pytest.mark.parametrize(
    "action",
    [
        HighLevelAction.KEEP_LANE,
        HighLevelAction.FOLLOW,
        HighLevelAction.SLOW_DOWN,
        HighLevelAction.STOP,
        HighLevelAction.YIELD,
    ],
)
def test_executor_applies_only_supported_effective_commands(action: HighLevelAction) -> None:
    decision = _decision(action, target_speed_mps=15.0)

    execution = VLACommandExecutor(35.0).execute(decision, now_s=1.5)

    assert execution.source is CommandExecutionSource.VLA
    assert execution.action is decision.effective_command.action
    assert execution.target_speed_mps == decision.effective_command.target_speed_mps
    assert execution.command_id == "model-7"
    assert execution.valid_until_s == 3.0


def test_executor_uses_effective_not_requested_speed() -> None:
    decision = _decision(HighLevelAction.SLOW_DOWN, target_speed_mps=45.0 / 3.6)

    execution = VLACommandExecutor(35.0).execute(decision, now_s=1.0)

    assert decision.requested_command is not None
    assert decision.requested_command.target_speed_mps == pytest.approx(45.0 / 3.6)
    assert decision.effective_command.target_speed_mps == pytest.approx(20.0 / 3.6)
    assert execution.target_speed_mps == pytest.approx(20.0 / 3.6)


def test_executor_can_enforce_locally_mapped_keep_lane_speed() -> None:
    decision = _decision(HighLevelAction.KEEP_LANE, target_speed_mps=0.0)
    policy = ActionSpeedPolicy(
        "enforce",
        cruise_speed_mps=10.0,
        slow_down_speed_mps=5.0,
        yield_speed_mps=2.0,
    )

    execution = VLACommandExecutor(
        10.0,
        action_speed_policy=policy,
    ).execute(decision, now_s=1.0)

    assert execution.target_speed_mps == 10.0
    assert execution.action_speed is not None
    assert execution.action_speed.requested_target_speed_mps == 0.0
    assert execution.action_speed.applied is True


@pytest.mark.parametrize(
    "action",
    [
        HighLevelAction.CHANGE_LANE_LEFT,
        HighLevelAction.CHANGE_LANE_RIGHT,
        HighLevelAction.OVERTAKE,
        HighLevelAction.PULL_OVER,
    ],
)
def test_executor_fails_closed_for_unsupported_lateral_actions(
    action: HighLevelAction,
) -> None:
    decision = _decision(action)

    execution = VLACommandExecutor(35.0).execute(decision, now_s=1.0)

    assert decision.effective_command.action is action
    assert execution.source is CommandExecutionSource.UNSUPPORTED_ACTION
    assert execution.action is HighLevelAction.REQUEST_FALLBACK
    assert execution.target_speed_mps == 35.0
    assert action.value in execution.reason


def test_executor_falls_back_without_a_decision_or_after_expiry() -> None:
    executor = VLACommandExecutor(35.0)

    missing = executor.execute(None, now_s=1.0)
    expired = executor.execute(_decision(HighLevelAction.STOP), now_s=3.1)

    assert missing.source is CommandExecutionSource.LOCAL_FALLBACK
    assert expired.source is CommandExecutionSource.LOCAL_FALLBACK
    assert missing.target_speed_mps == expired.target_speed_mps == 35.0
    assert "expired" in expired.reason


def test_executor_horizon_boundary_tolerates_only_float_noise() -> None:
    executor = VLACommandExecutor(35.0)
    decision = _decision(
        HighLevelAction.KEEP_LANE,
        issued_at_s=0.0,
        action_horizon_s=1.5,
    )

    boundary = executor.execute(decision, now_s=1.5000000000000004)
    late = executor.execute(decision, now_s=1.6)

    assert boundary.source is CommandExecutionSource.VLA
    assert late.source is CommandExecutionSource.LOCAL_FALLBACK


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_executor_rejects_invalid_fallback_speed(value: float) -> None:
    with pytest.raises(ValueError, match="fallback_target_speed_mps"):
        VLACommandExecutor(value)
