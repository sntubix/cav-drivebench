from __future__ import annotations

import math
from numbers import Real
from typing import Callable, Protocol, TypeAlias, runtime_checkable

from metadrive_starter.config import ControllerSettings
from metadrive_starter.types import Action, ControlTick


class ControllerOutputError(ValueError):
    """A controller returned an action the actuator cannot apply, even clamped."""


@runtime_checkable
class VehicleController(Protocol):
    """Replaceable low-level control: tracking errors in, one action out."""

    def update(self, tick: ControlTick) -> Action:
        """Return (steering, throttle_brake), each within [-1, 1]."""
        ...

    def reset_speed_control(self) -> None:
        """Forget speed-control history before it can wind up.

        Called when the target speed jumps, when its source changes (for
        example from a validated command to the local fallback), and on every
        tick the safety floor overrides throttle or brake.
        """
        ...


VehicleControllerFactory: TypeAlias = Callable[[ControllerSettings], VehicleController]


def build_vehicle_controller(
    settings: ControllerSettings,
    *,
    factory: VehicleControllerFactory,
) -> VehicleController:
    """Build a controller from its factory, checking it implements the protocol."""
    controller = factory(settings)
    if not isinstance(controller, VehicleController):
        raise TypeError(
            "vehicle controller must implement update(tick) and reset_speed_control()"
        )
    return controller


def checked_action(action: object) -> Action:
    """Return a controller's action as two finite floats, or explain what was expected."""
    if not isinstance(action, (tuple, list)) or len(action) != 2:
        raise ControllerOutputError(
            "vehicle controller update() must return (steering, throttle_brake); "
            f"got {action!r}"
        )
    for name, value in zip(("steering", "throttle_brake"), action):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ControllerOutputError(f"vehicle controller {name} must be a number; got {value!r}")
        if not math.isfinite(value):
            raise ControllerOutputError(
                f"vehicle controller {name} must be a finite number; got {value!r}"
            )
    return (float(action[0]), float(action[1]))


def saturated_action(action: Action) -> Action:
    """Clamp an action into the actuator range [-1, 1], as a real actuator saturates."""
    steering, throttle_brake = action
    return (max(-1.0, min(1.0, steering)), max(-1.0, min(1.0, throttle_brake)))
