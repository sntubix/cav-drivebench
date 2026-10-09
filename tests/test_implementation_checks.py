import pytest

from metadrive_starter.implementation_checks import (
    ANTI_WINDUP,
    LOAD_REJECTION,
    RESET,
    SPEED_DERIVATIVE,
    SPEED_INTEGRAL,
    STEERING_TERMS,
    CheckResult,
    LongitudinalModel,
    drive_speed,
    run_implementation_checks,
)


class ConstantThrottle:
    def __init__(self, throttle_brake: float) -> None:
        self.throttle_brake = throttle_brake
        self.ticks = []

    def update(self, tick):
        self.ticks.append(tick)
        return (0.0, self.throttle_brake)

    def reset_speed_control(self) -> None:
        pass


def test_toy_vehicle_accelerates_and_brakes_as_measured_in_metadrive() -> None:
    model = LongitudinalModel()

    assert model.step(5.0, 1.0, 0.1) == pytest.approx(5.276)
    assert model.step(5.0, 0.5, 0.1) == pytest.approx(5.138)
    assert model.step(5.0, -1.0, 0.1) == pytest.approx(3.88)


def test_toy_vehicle_load_slows_it_and_it_never_rolls_backwards() -> None:
    hill = LongitudinalModel(load_mps2=0.5)

    assert hill.step(5.0, 0.0, 0.1) == pytest.approx(4.95)
    assert hill.step(0.0, 0.0, 0.1) == 0.0
    assert LongitudinalModel().step(0.5, -1.0, 0.1) == 0.0


def test_drive_speed_feeds_the_controller_one_straight_road_tick_per_step() -> None:
    controller = ConstantThrottle(0.5)

    trace = drive_speed(
        controller,
        lambda time_s: 5.0 if time_s < 0.5 else 10.0,
        duration_s=1.0,
        dt_s=0.1,
    )

    assert trace.time_s == pytest.approx([0.1 * k for k in range(10)])
    assert trace.target_mps == (5.0,) * 5 + (10.0,) * 5
    # Half throttle accelerates at 1.38 m/s^2.
    assert trace.speed_mps == pytest.approx([0.138 * k for k in range(10)])
    assert trace.throttle_brake == (0.5,) * 10
    assert [tick.speed_mps for tick in controller.ticks] == list(trace.speed_mps)
    assert {
        (tick.heading_error_rad, tick.lateral_error_m, tick.dt_s)
        for tick in controller.ticks
    } == {(0.0, 0.0, 0.1)}


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class Proportional:
    """The shipped stub's structure: every loop proportional-only."""

    def __init__(self, settings) -> None:
        self.settings = settings

    def update(self, tick):
        speed = self.settings.speed_pid
        throttle = _clamp(
            speed.kp * (tick.target_speed_mps - tick.speed_mps),
            speed.output_min,
            speed.output_max,
        )
        steering = (
            self.settings.steering_pid.kp * tick.heading_error_rad
            + self.settings.lateral_pid.kp * tick.lateral_error_m
        )
        return (_clamp(steering), throttle)

    def reset_speed_control(self) -> None:
        pass


def _results(factory) -> dict[str, CheckResult]:
    return {result.check: result for result in run_implementation_checks(factory)}


def test_proportional_only_controller_has_no_speed_integral_term() -> None:
    result = _results(Proportional)[SPEED_INTEGRAL]

    assert not result.passed
    assert result.detail == (
        "with 0.1 s ticks, speed_pid.ki = 0.2 added +0.000 to throttle_brake after "
        "2 s at a steady 0.5 m/s speed error; expected +0.200"
    )


def test_proportional_only_controller_has_no_speed_derivative_term() -> None:
    result = _results(Proportional)[SPEED_DERIVATIVE]

    assert not result.passed
    assert result.detail == (
        "with 0.1 s ticks, speed_pid.kd = 0.1 added +0.000 to throttle_brake while "
        "the speed error changed by -1 m/s every second; expected -0.100"
    )


def test_proportional_only_controller_has_no_steering_integral_or_derivative_term() -> None:
    result = _results(Proportional)[STEERING_TERMS]

    assert not result.passed
    assert result.detail == (
        "with 0.1 s ticks, steering_pid.ki = 0.2 added +0.000 to steering after 2 s "
        "at a steady 0.1 rad heading error; expected +0.040"
    )


def test_proportional_only_controller_settles_short_of_its_target_under_load() -> None:
    result = _results(Proportional)[LOAD_REJECTION]

    assert not result.passed
    # kp = 0.3 settles where 0.3 * 2.76 * error balances the 0.5 m/s^2 load.
    assert result.detail == (
        "against a steady 0.5 m/s^2 load, with speed_pid kp = 0.3, ki = 0.1, the "
        "speed ended 0.60 m/s below its 10 m/s target after 40 s; expected within "
        "0.05 m/s"
    )


def test_proportional_only_controller_never_reaches_its_target_after_a_saturated_start() -> None:
    result = _results(Proportional)[ANTI_WINDUP]

    assert not result.passed
    assert result.detail == (
        "from a standstill against a steady 0.5 m/s^2 load, with speed_pid kp = 0.3, "
        "ki = 0.1, the speed ended 0.60 m/s below its 10 m/s target after 40 s; "
        "expected within 0.05 m/s"
    )


def test_proportional_only_controller_keeps_no_history_for_a_reset_to_clear() -> None:
    result = _results(Proportional)[RESET]

    assert not result.passed
    assert result.detail == (
        "5 s at a steady 0.5 m/s speed error left no trace in throttle_brake, so "
        "reset_speed_control() had no history to clear; expected the integral "
        "term to keep some"
    )


class Raising:
    def __init__(self, settings) -> None:
        pass

    def update(self, tick):
        raise RuntimeError("boom")

    def reset_speed_control(self) -> None:
        pass


def test_a_controller_that_raises_fails_every_check_with_its_error() -> None:
    results = run_implementation_checks(Raising)

    assert [result.check for result in results] == [
        SPEED_INTEGRAL,
        SPEED_DERIVATIVE,
        STEERING_TERMS,
        LOAD_REJECTION,
        ANTI_WINDUP,
        RESET,
    ]
    assert {(result.passed, result.detail) for result in results} == {
        (False, "the controller raised RuntimeError: boom")
    }
