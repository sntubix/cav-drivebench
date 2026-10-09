from __future__ import annotations

from dataclasses import replace

import pytest

from metadrive_starter.perception import LocalScene
from metadrive_starter.safety import (
    CommandDisposition,
    CommandValidationDecision,
    CommandValidationSettings,
    HighLevelSafetyPolicy,
    VLACommandValidator,
    build_high_level_safety_policy,
)
from metadrive_starter.vla import HighLevelAction, VLACommand


class StopOnlyPolicy:
    """Small example of a policy injected without subclassing the baseline."""

    def __init__(self, settings: CommandValidationSettings):
        self.settings = settings

    def validate(
        self,
        command: VLACommand,
        scene: LocalScene,
        *,
        now_s: float,
    ) -> CommandValidationDecision:
        del scene, now_s
        effective = replace(command, action=HighLevelAction.STOP, target_speed_mps=0.0)
        return CommandValidationDecision(
            disposition=CommandDisposition.MODIFIED,
            requested_command=command,
            effective_command=effective,
            reasons=("example policy permits STOP only",),
        )


def _command() -> VLACommand:
    return VLACommand(
        action=HighLevelAction.KEEP_LANE,
        target_speed_mps=25.0,
        issued_at_s=1.0,
        action_horizon_s=2.0,
        confidence=0.9,
    )


def _scene() -> LocalScene:
    return LocalScene(
        timestamp_s=1.0,
        ego_speed_mps=5.0,
        ego_length_m=4.5,
        ego_width_m=1.8,
        lane_offset_m=0.0,
        heading_error_rad=0.0,
    )


def test_default_policy_is_baseline_validator() -> None:
    policy = build_high_level_safety_policy(CommandValidationSettings())

    assert isinstance(policy, VLACommandValidator)
    assert isinstance(policy, HighLevelSafetyPolicy)


def test_baseline_policy_uses_injected_settings() -> None:
    settings = CommandValidationSettings(maximum_target_speed_mps=20.0)
    policy = build_high_level_safety_policy(settings)

    decision = policy.validate(_command(), _scene(), now_s=1.0)

    assert decision.disposition is CommandDisposition.MODIFIED
    assert decision.effective_command.target_speed_mps == 20.0


def test_alternative_policy_can_be_injected_structurally() -> None:
    settings = CommandValidationSettings(maximum_target_speed_mps=32.0)
    policy = build_high_level_safety_policy(settings, factory=StopOnlyPolicy)

    decision = policy.validate(_command(), _scene(), now_s=1.0)

    assert isinstance(policy, StopOnlyPolicy)
    assert policy.settings is settings
    assert decision.effective_command.action is HighLevelAction.STOP
    assert decision.effective_command.target_speed_mps == 0.0


def test_factory_rejects_object_without_policy_contract() -> None:
    with pytest.raises(TypeError, match="implement validate"):
        build_high_level_safety_policy(factory=lambda settings: object())
