"""Implementation checks: a submitted controller on its own, on a toy vehicle.

MetaDrive's vehicle holds its speed with almost no throttle, so on the gate
routes a proportional-only controller drives as well as a complete PID: nothing
there needs an integral or a derivative term. The toy vehicle supplies what the
routes lack, such as a constant load and a long saturated start, so that each
term of a controller can be checked on its own.

Every check builds its controllers with gains of its own. It tests the
implementation, never a team's tuning: tuning belongs in agent.yaml and shows
on the routes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable

from metadrive_starter.config import ControllerSettings, PIDSettings
from metadrive_starter.controllers import (
    VehicleController,
    VehicleControllerFactory,
    checked_action,
)
from metadrive_starter.types import ControlTick

# Measured in MetaDrive 0.4.3 on a straight road: full throttle accelerates at a
# constant 2.76 m/s^2 up to at least 17 m/s, with no drag while any throttle is
# applied, and full brake decelerates at about 11.2 m/s^2.
THROTTLE_ACCELERATION_MPS2 = 2.76
BRAKE_DECELERATION_MPS2 = 11.2

SPEED_INTEGRAL = "speed loop integral term"
SPEED_DERIVATIVE = "speed loop derivative term"
STEERING_TERMS = "steering loops' integral and derivative terms"
LOAD_REJECTION = "speed held against a constant load"
ANTI_WINDUP = "no windup after a saturated start"
RESET = "reset_speed_control() clears speed history"

# Speed terms are checked at the simulator's 0.1 s control tick and at half of
# it, so a term that ignores dt, or assumes 0.1 s, fails at one of them.
_TICK_LENGTHS_S = (0.1, 0.05)
# How long a term check feeds its steady or steadily changing error.
_TERM_S = 2.0
# A term passes when it contributes within 10% of what it should.
_TOLERANCE = 0.1
_OFF = PIDSettings(0.0, 0.0, 0.0)
_CHECK_SETTINGS = ControllerSettings(speed_pid=_OFF, steering_pid=_OFF, lateral_pid=_OFF)


@dataclass(frozen=True)
class LongitudinalModel:
    """A toy vehicle whose acceleration is proportional to throttle or brake.

    ``load_mps2`` is a constant deceleration, such as a grade or a headwind,
    which MetaDrive's flat maps never have: a 5% grade is about 0.49 m/s^2.
    """

    throttle_acceleration_mps2: float = THROTTLE_ACCELERATION_MPS2
    brake_deceleration_mps2: float = BRAKE_DECELERATION_MPS2
    load_mps2: float = 0.0

    def step(self, speed_mps: float, throttle_brake: float, dt_s: float) -> float:
        """Return the speed ``dt_s`` later; the vehicle never rolls backwards."""
        command = max(-1.0, min(1.0, throttle_brake))
        gain = (
            self.throttle_acceleration_mps2
            if command >= 0.0
            else self.brake_deceleration_mps2
        )
        return max(0.0, speed_mps + (gain * command - self.load_mps2) * dt_s)


@dataclass(frozen=True)
class SpeedTrace:
    """One closed speed loop, sampled on every control tick.

    ``speed_mps`` is what the controller measured on the tick and
    ``throttle_brake`` what it returned.
    """

    time_s: tuple[float, ...]
    target_mps: tuple[float, ...]
    speed_mps: tuple[float, ...]
    throttle_brake: tuple[float, ...]


def drive_speed(
    controller: VehicleController,
    target_mps: float | Callable[[float], float],
    *,
    duration_s: float,
    dt_s: float = 0.1,
    initial_speed_mps: float = 0.0,
    model: LongitudinalModel = LongitudinalModel(),
) -> SpeedTrace:
    """Close ``controller``'s speed loop around ``model`` on a straight road.

    ``target_mps`` is a constant or a function of time in seconds. The tracking
    errors stay zero, so only the speed loop acts.
    """
    target = target_mps if callable(target_mps) else lambda _: float(target_mps)
    times: list[float] = []
    targets: list[float] = []
    speeds: list[float] = []
    outputs: list[float] = []
    speed = initial_speed_mps
    for step in range(round(duration_s / dt_s)):
        time_s = step * dt_s
        tick = ControlTick(target(time_s), speed, 0.0, 0.0, dt_s)
        _, throttle_brake = checked_action(controller.update(tick))
        times.append(time_s)
        targets.append(tick.target_speed_mps)
        speeds.append(speed)
        outputs.append(throttle_brake)
        speed = model.step(speed, throttle_brake, dt_s)
    return SpeedTrace(tuple(times), tuple(targets), tuple(speeds), tuple(outputs))


@dataclass(frozen=True)
class CheckResult:
    check: str
    passed: bool
    detail: str


# A check returns whether it passed and what it measured.
_Check = Callable[[VehicleControllerFactory], tuple[bool, str]]


def run_implementation_checks(factory: VehicleControllerFactory) -> tuple[CheckResult, ...]:
    """Run every implementation check against controllers built by ``factory``."""
    return tuple(_run(name, check, factory) for name, check in _CHECKS)


def _run(name: str, check: _Check, factory: VehicleControllerFactory) -> CheckResult:
    try:
        passed, detail = check(factory)
    except Exception as exc:
        # Submitted code may raise anything; the check reports it.
        return CheckResult(name, False, f"the controller raised {type(exc).__name__}: {exc}")
    return CheckResult(name, passed, detail)


def _speed_integral(factory: VehicleControllerFactory) -> tuple[bool, str]:
    for dt_s in _TICK_LENGTHS_S:
        problem = _integral_problem(factory, _SPEED, ki=0.2, error=0.5, dt_s=dt_s)
        if problem is not None:
            return False, problem
    return True, (
        "speed_pid.ki adds ki times the error integrated over seconds, "
        "with 0.1 s and 0.05 s ticks"
    )


def _speed_derivative(factory: VehicleControllerFactory) -> tuple[bool, str]:
    for dt_s in _TICK_LENGTHS_S:
        # The speed rises toward a fixed target, so the error falls.
        problem = _derivative_problem(
            factory, _SPEED, kd=0.1, start=2.0, rate=-1.0, dt_s=dt_s
        )
        if problem is not None:
            return False, problem
    return True, (
        "speed_pid.kd adds kd times the error's rate of change per second, "
        "with 0.1 s and 0.05 s ticks"
    )


def _steering_terms(factory: VehicleControllerFactory) -> tuple[bool, str]:
    for loop, error in ((_HEADING, 0.1), (_LATERAL, 0.2)):
        problem = _integral_problem(
            factory, loop, ki=0.2, error=error, dt_s=0.1
        ) or _derivative_problem(factory, loop, kd=0.1, start=0.0, rate=error, dt_s=0.1)
        if problem is not None:
            return False, problem
    return True, "steering_pid and lateral_pid each add their ki and kd terms to steering"


# The closed-loop checks drive the toy vehicle up a grade of about 5%.
_HILL = LongitudinalModel(load_mps2=0.5)
_CLOSED_LOOP_GAINS = PIDSettings(0.3, 0.1, 0.0)
_TARGET_MPS = 10.0
_CLOSED_LOOP_S = 40.0
_SETTLED_WITHIN_MPS = 0.05
# Conditional integration or back-calculation overshoots about 0.3 m/s from a
# standing start; an integral that winds up behind the saturated output
# overshoots 4.5 m/s, or 1.9 m/s when merely clamped to the output range.
_OVERSHOOT_WITHIN_MPS = 1.0


def _load_rejection(factory: VehicleControllerFactory) -> tuple[bool, str]:
    trace = _climb(factory, initial_speed_mps=_TARGET_MPS)
    error = _TARGET_MPS - trace.speed_mps[-1]
    load = f"against a steady {_HILL.load_mps2:g} m/s^2 load"
    if abs(error) > _SETTLED_WITHIN_MPS:
        return False, (
            f"{load}, with {_gains(_CLOSED_LOOP_GAINS)}, the speed ended "
            f"{_off_target(error)} after {_CLOSED_LOOP_S:g} s; "
            f"expected within {_SETTLED_WITHIN_MPS:g} m/s"
        )
    return True, (
        f"{load}, the speed settled within {_SETTLED_WITHIN_MPS:g} m/s of its "
        f"{_TARGET_MPS:g} m/s target"
    )


def _anti_windup(factory: VehicleControllerFactory) -> tuple[bool, str]:
    trace = _climb(factory, initial_speed_mps=0.0)
    overshoot = max(trace.speed_mps) - _TARGET_MPS
    error = _TARGET_MPS - trace.speed_mps[-1]
    start = (
        f"from a standstill against a steady {_HILL.load_mps2:g} m/s^2 load, with "
        f"{_gains(_CLOSED_LOOP_GAINS)}, the speed"
    )
    if overshoot > _OVERSHOOT_WITHIN_MPS:
        return False, (
            f"{start} overshot its {_TARGET_MPS:g} m/s target by {overshoot:.2f} m/s; "
            f"expected at most {_OVERSHOOT_WITHIN_MPS:g} m/s"
        )
    if abs(error) > _SETTLED_WITHIN_MPS:
        return False, (
            f"{start} ended {_off_target(error)} after {_CLOSED_LOOP_S:g} s; "
            f"expected within {_SETTLED_WITHIN_MPS:g} m/s"
        )
    return True, (
        f"{start} overshot its {_TARGET_MPS:g} m/s target by "
        f"{max(overshoot, 0.0):.2f} m/s and settled within {_SETTLED_WITHIN_MPS:g} m/s"
    )


def _climb(factory: VehicleControllerFactory, *, initial_speed_mps: float) -> SpeedTrace:
    """Drive the closed-loop gains up the hill toward the target."""
    return drive_speed(
        factory(replace(_CHECK_SETTINGS, speed_pid=_CLOSED_LOOP_GAINS)),
        _TARGET_MPS,
        duration_s=_CLOSED_LOOP_S,
        initial_speed_mps=initial_speed_mps,
        model=_HILL,
    )


def _reset(factory: VehicleControllerFactory) -> tuple[bool, str]:
    # Every term enabled, so a stale derivative counts as history too.
    settings = replace(_CHECK_SETTINGS, speed_pid=PIDSettings(0.3, 0.1, 0.1))
    reset, kept, fresh = factory(settings), factory(settings), factory(settings)
    error_mps, history_s, dt_s = 0.5, 5.0, 0.1
    behind = ControlTick(10.0, 10.0 - error_mps, 0.0, 0.0, dt_s)
    for _ in range(round(history_s / dt_s)):
        reset.update(behind)
        kept.update(behind)
    reset.reset_speed_control()
    after = [ControlTick(10.0, 9.8, 0.0, 0.0, dt_s)] * 10
    history = f"{history_s:g} s at a steady {error_mps:g} m/s speed error"
    reset_outputs, kept_outputs, fresh_outputs = (
        [checked_action(controller.update(tick))[1] for tick in after]
        for controller in (reset, kept, fresh)
    )
    if max(_differences(kept_outputs, fresh_outputs)) < 0.01:
        return False, (
            f"{history} left no trace in throttle_brake, so reset_speed_control() "
            "had no history to clear; expected the integral term to keep some"
        )
    difference = max(_differences(reset_outputs, fresh_outputs))
    if difference > 1e-6:
        return False, (
            f"after {history} and reset_speed_control(), throttle_brake differed "
            f"from a fresh controller's by up to {difference:.3f}; expected the same output"
        )
    return True, (
        "after reset_speed_control(), throttle_brake matched a fresh controller's "
        "tick for tick"
    )


_CHECKS: tuple[tuple[str, _Check], ...] = (
    (SPEED_INTEGRAL, _speed_integral),
    (SPEED_DERIVATIVE, _speed_derivative),
    (STEERING_TERMS, _steering_terms),
    (LOAD_REJECTION, _load_rejection),
    (ANTI_WINDUP, _anti_windup),
    (RESET, _reset),
)


_ACTION = ("steering", "throttle_brake")


@dataclass(frozen=True)
class _Loop:
    """One control loop as an implementation check drives it.

    ``settings_field`` names its gains in ControllerSettings, ``action_index``
    the part of the action it drives, and ``tick`` builds a tick carrying a
    given error for it.
    """

    settings_field: str
    kp: float
    action_index: int
    error: str
    unit: str
    tick: Callable[[float, float], ControlTick]

    @property
    def output(self) -> str:
        return _ACTION[self.action_index]


_SPEED = _Loop(
    "speed_pid",
    0.2,
    1,
    "speed",
    "m/s",
    lambda error, dt_s: ControlTick(10.0, 10.0 - error, 0.0, 0.0, dt_s),
)
_HEADING = _Loop(
    "steering_pid",
    0.5,
    0,
    "heading",
    "rad",
    lambda error, dt_s: ControlTick(10.0, 10.0, error, 0.0, dt_s),
)
_LATERAL = _Loop(
    "lateral_pid",
    0.3,
    0,
    "lateral",
    "m",
    lambda error, dt_s: ControlTick(10.0, 10.0, 0.0, error, dt_s),
)


def _integral_problem(
    factory: VehicleControllerFactory,
    loop: _Loop,
    *,
    ki: float,
    error: float,
    dt_s: float,
) -> str | None:
    """Explain how ``loop``'s ki misbehaves against a steady error, if it does."""
    expected = ki * error * _TERM_S
    ticks = [loop.tick(error, dt_s)] * round(_TERM_S / dt_s)
    added = _term_contribution(factory, loop, ticks, ki=ki)[-1]
    if _close(added, expected):
        return None
    return (
        f"with {dt_s:g} s ticks, {loop.settings_field}.ki = {ki:g} added {added:+.3f} "
        f"to {loop.output} after {_TERM_S:g} s at a steady {error:g} {loop.unit} "
        f"{loop.error} error; expected {expected:+.3f}"
    )


def _derivative_problem(
    factory: VehicleControllerFactory,
    loop: _Loop,
    *,
    kd: float,
    start: float,
    rate: float,
    dt_s: float,
) -> str | None:
    """Explain how ``loop``'s kd misbehaves against a steadily changing error."""
    expected = kd * rate
    steps = round(_TERM_S / dt_s)
    ticks = [loop.tick(start + rate * step * dt_s, dt_s) for step in range(steps)]
    # Judged over the second half, once any derivative filter has settled.
    added = _mean(_term_contribution(factory, loop, ticks, kd=kd)[steps // 2 :])
    if _close(added, expected):
        return None
    return (
        f"with {dt_s:g} s ticks, {loop.settings_field}.kd = {kd:g} added {added:+.3f} "
        f"to {loop.output} while the {loop.error} error changed by {rate:+g} "
        f"{loop.unit} every second; expected {expected:+.3f}"
    )


def _term_contribution(
    factory: VehicleControllerFactory,
    loop: _Loop,
    ticks: Iterable[ControlTick],
    *,
    ki: float = 0.0,
    kd: float = 0.0,
) -> list[float]:
    """Per tick, what ki or kd adds to ``loop``'s output.

    Two controllers that differ only in that gain see the same ticks, so
    anything else a controller adds, such as feedforward, cancels out.
    """
    with_term = factory(
        replace(_CHECK_SETTINGS, **{loop.settings_field: PIDSettings(loop.kp, ki, kd)})
    )
    without_term = factory(
        replace(_CHECK_SETTINGS, **{loop.settings_field: PIDSettings(loop.kp, 0.0, 0.0)})
    )
    return [
        checked_action(with_term.update(tick))[loop.action_index]
        - checked_action(without_term.update(tick))[loop.action_index]
        for tick in ticks
    ]


def _gains(gains: PIDSettings) -> str:
    return f"speed_pid kp = {gains.kp:g}, ki = {gains.ki:g}"


def _off_target(error: float) -> str:
    return f"{abs(error):.2f} m/s {'below' if error > 0 else 'above'} its {_TARGET_MPS:g} m/s target"


def _close(value: float, expected: float) -> bool:
    return abs(value - expected) <= _TOLERANCE * abs(expected)


def _differences(first: list[float], second: list[float]) -> list[float]:
    return [abs(a - b) for a, b in zip(first, second)]


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)
