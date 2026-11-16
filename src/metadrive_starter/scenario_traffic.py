"""Scenario vehicles that keep a set speed in their lane, driven every control step.

MetaDrive's IDM drives each one, with lane changes off and its desired speed set to
the configured one, so it follows its lane along the route through straights and
curves and still brakes for anything that stops in front of it. A manager registered
on the engine acts for it before each control step and updates its localisation
after, as MetaDrive's traffic manager does for its own vehicles: without that, a
spawned vehicle coasts to a stop and its lane index never moves.

This module imports MetaDrive at load, so import it only where MetaDrive runs.
"""

from __future__ import annotations

from typing import Any

from metadrive.manager.base_manager import BaseManager
from metadrive.policy.idm_policy import IDMPolicy

from metadrive_starter.units import mps_to_kmh

MANAGER_NAME = "drivebench_scenario_traffic"


class SetSpeedIDMPolicy(IDMPolicy):
    """MetaDrive's IDM with lane changes off and its desired speed set per vehicle."""

    def __init__(self, control_object: Any, speed_mps: float) -> None:
        # The seed only times IDM's lane changes, which are off.
        super().__init__(control_object, random_seed=0)
        # IDM works in km/h, and reset() restores target_speed from NORMAL_SPEED.
        self.NORMAL_SPEED = mps_to_kmh(speed_mps)
        self.target_speed = self.NORMAL_SPEED
        self.enable_lane_change = False


class ScenarioTrafficManager(BaseManager):
    """Steps the set-speed policy of every scenario vehicle that drives."""

    def __init__(self) -> None:
        super().__init__()
        self._driven: list[tuple[Any, SetSpeedIDMPolicy]] = []

    def add(self, vehicle: Any, speed_mps: float) -> None:
        self._driven.append((vehicle, SetSpeedIDMPolicy(vehicle, speed_mps)))

    def before_step(self, *args: Any, **kwargs: Any) -> dict:
        for vehicle, policy in self._driven:
            vehicle.before_step(policy.act())
        return {}

    def after_step(self, *args: Any, **kwargs: Any) -> dict:
        for vehicle, _ in self._driven:
            vehicle.after_step()
        return {}

    def before_reset(self) -> None:
        # The vehicles belong to the engine, as every scenario object does; a reset
        # only stops driving them.
        self._release()
        super().before_reset()

    def destroy(self) -> None:
        self._release()
        super().destroy()

    def _release(self) -> None:
        for _, policy in self._driven:
            policy.destroy()
        self._driven = []


def drive_at_set_speed(engine: Any, vehicle: Any, speed_mps: float) -> None:
    """Have the engine drive ``vehicle`` at ``speed_mps`` in its lane from the next step."""
    manager = getattr(engine, MANAGER_NAME, None)
    if manager is None:
        manager = ScenarioTrafficManager()
        engine.register_manager(MANAGER_NAME, manager)
    manager.add(vehicle, speed_mps)
