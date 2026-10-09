from __future__ import annotations

from typing import Callable, Protocol, TypeAlias, runtime_checkable

from metadrive_starter.perception import LocalScene
from metadrive_starter.safety.command_validator import (
    CommandValidationDecision,
    CommandValidationSettings,
    VLACommandValidator,
)
from metadrive_starter.vla import VLACommand


@runtime_checkable
class HighLevelSafetyPolicy(Protocol):
    """The safety floor's command validation, as the runtime composes it.

    Instructor-owned: no submission replaces it (ADR-0001). Teams restrict model
    authority above it, in arbitration.
    """

    def validate(
        self,
        command: VLACommand,
        scene: LocalScene,
        *,
        now_s: float,
    ) -> CommandValidationDecision:
        """Return the only command that downstream planning may execute."""
        ...


HighLevelSafetyPolicyFactory: TypeAlias = Callable[
    [CommandValidationSettings], HighLevelSafetyPolicy
]


def build_high_level_safety_policy(
    settings: CommandValidationSettings | None = None,
    *,
    factory: HighLevelSafetyPolicyFactory = VLACommandValidator,
) -> HighLevelSafetyPolicy:
    """Build the safety floor's validator; ``factory`` lets tests inject a stand-in."""
    policy = factory(settings or CommandValidationSettings())
    if not isinstance(policy, HighLevelSafetyPolicy):
        raise TypeError("high-level safety policy must implement validate()")
    return policy
