"""Assignment 1: your vehicle controller.

DriveBench loads this file when you pass ``--submission``:

    uv run metadrive-starter run --submission submission --headless

Each run builds one ``Controller(settings)``, where ``settings`` carries the
controller values from configuration. The simulator then calls ``update(tick)``
on every control tick, and ``reset_speed_control()`` whenever speed-control
history must be forgotten.

As shipped, every loop is proportional-only and the speed gain is deliberately
weak. The vehicle completes its route, badly: it settles well below its target
speed and takes far longer than it should.

Every place that needs your work is marked TODO. Tune the gains in agent.yaml.
"""

from __future__ import annotations

from dataclasses import replace

from metadrive_starter.config import ControllerSettings, PIDSettings
from metadrive_starter.types import Action, ControlTick


class PID:
    """One control loop. Only the proportional term is implemented."""

    def __init__(self, gains: PIDSettings) -> None:
        self.gains = gains
        # TODO: keep the history the integral and derivative terms need.

    def reset(self) -> None:
        """Forget loop history. A proportional-only loop keeps none."""
        # TODO: clear the history that __init__ keeps.

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement
        # TODO: add the integral term (ki) and the derivative term (kd). Then add
        # anti-windup: while the output is saturated in the direction the error
        # pushes, stop the integral growing, unless self.gains.anti_windup is False.
        output = self.gains.kp * error
        return max(self.gains.output_min, min(self.gains.output_max, output))


class Controller:
    """Throttle and brake from a speed loop; steering from heading and lateral loops."""

    def __init__(self, settings: ControllerSettings) -> None:
        # Deliberately weak, and it overrides the configured speed_pid.kp: a
        # proportional-only speed loop with this gain settles well below target.
        # TODO: remove this override, so the speed loop uses settings.speed_pid.
        self.speed_loop = PID(replace(settings.speed_pid, kp=0.03))
        self.heading_loop = PID(settings.steering_pid)
        self.lateral_loop = PID(settings.lateral_pid)

    def update(self, tick: ControlTick) -> Action:
        throttle_brake = self.speed_loop.update(
            tick.target_speed_mps,
            tick.speed_mps,
            tick.dt_s,
        )
        # Each steering loop drives its error to zero: the setpoint is 0 and the
        # measured deviation is -error, so a positive error steers left.
        steering = self.heading_loop.update(0.0, -tick.heading_error_rad, tick.dt_s)
        steering += self.lateral_loop.update(0.0, -tick.lateral_error_m, tick.dt_s)
        return (max(-1.0, min(1.0, steering)), throttle_brake)

    def reset_speed_control(self) -> None:
        self.speed_loop.reset()
