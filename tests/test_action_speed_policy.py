from __future__ import annotations

import pytest

from metadrive_starter.planning import ActionSpeedPolicy
from metadrive_starter.vla import HighLevelAction


def _policy(mode: str) -> ActionSpeedPolicy:
    return ActionSpeedPolicy(
        mode,
        cruise_speed_mps=10.0,
        slow_down_speed_mps=5.0,
        yield_speed_mps=2.0,
    )


def test_action_speed_policy_off_preserves_raw_target_without_mapping() -> None:
    decision = _policy("off").evaluate(HighLevelAction.KEEP_LANE, 0.0)

    assert decision.mapped_target_speed_mps is None
    assert decision.effective_target_speed_mps == 0.0
    assert decision.would_intervene is False
    assert decision.applied is False


def test_action_speed_policy_shadow_records_but_does_not_apply_mapping() -> None:
    decision = _policy("shadow").evaluate(HighLevelAction.KEEP_LANE, 0.0)

    assert decision.mapped_target_speed_mps == 10.0
    assert decision.effective_target_speed_mps == 0.0
    assert decision.would_intervene is True
    assert decision.applied is False


@pytest.mark.parametrize(
    ("action", "expected_speed_mps"),
    [
        (HighLevelAction.KEEP_LANE, 10.0),
        (HighLevelAction.FOLLOW, 10.0),
        (HighLevelAction.CHANGE_LANE_LEFT, 10.0),
        (HighLevelAction.CHANGE_LANE_RIGHT, 10.0),
        (HighLevelAction.SLOW_DOWN, 5.0),
        (HighLevelAction.YIELD, 2.0),
        (HighLevelAction.STOP, 0.0),
    ],
)
def test_action_speed_policy_enforces_local_action_mapping(
    action: HighLevelAction,
    expected_speed_mps: float,
) -> None:
    decision = _policy("enforce").evaluate(action, 7.0)

    assert decision.mapped_target_speed_mps == expected_speed_mps
    assert decision.effective_target_speed_mps == expected_speed_mps
    assert decision.would_intervene is (expected_speed_mps != 7.0)
    assert decision.applied is (expected_speed_mps != 7.0)


@pytest.mark.parametrize("mode", ["", "observe", "ENFORCE"])
def test_action_speed_policy_rejects_unknown_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        _policy(mode)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_action_speed_policy_rejects_invalid_requested_speed(value: float) -> None:
    with pytest.raises(ValueError, match="requested_target_speed_mps"):
        _policy("enforce").evaluate(HighLevelAction.KEEP_LANE, value)
