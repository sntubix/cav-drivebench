from __future__ import annotations

import pytest

from metadrive_starter.timing import RealTimePacer


class FakeClock:
    def __init__(self, now_s: float = 100.0) -> None:
        self.now_s = now_s
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.sleeps.append(duration_s)
        self.now_s += duration_s


def test_pacer_sleeps_only_for_remaining_simulation_step_time() -> None:
    clock = FakeClock()
    pacer = RealTimePacer(clock=clock, sleeper=clock.sleep)
    pacer.reset(0.0)

    clock.now_s += 0.02
    first_sleep = pacer.wait(0.1)
    clock.now_s += 0.04
    second_sleep = pacer.wait(0.2)

    assert first_sleep == pytest.approx(0.08)
    assert second_sleep == pytest.approx(0.06)
    assert clock.sleeps == pytest.approx([0.08, 0.06])


def test_pacer_can_advance_simulation_slower_than_wall_time() -> None:
    clock = FakeClock()
    pacer = RealTimePacer(
        realtime_factor=0.5,
        clock=clock,
        sleeper=clock.sleep,
    )
    pacer.reset(0.0)

    clock.now_s += 0.02
    first_sleep = pacer.wait(0.1)
    clock.now_s += 0.04
    second_sleep = pacer.wait(0.2)

    assert first_sleep == pytest.approx(0.18)
    assert second_sleep == pytest.approx(0.16)
    assert clock.sleeps == pytest.approx([0.18, 0.16])


def test_pacer_reanchors_when_behind_instead_of_catching_up() -> None:
    clock = FakeClock()
    pacer = RealTimePacer(clock=clock, sleeper=clock.sleep)
    pacer.reset(0.0)

    clock.now_s += 0.25
    assert pacer.wait(0.1) == 0.0
    clock.now_s += 0.02
    assert pacer.wait(0.2) == pytest.approx(0.08)

    assert clock.sleeps == pytest.approx([0.08])


def test_pacer_resets_automatically_when_simulation_time_moves_backwards() -> None:
    clock = FakeClock()
    pacer = RealTimePacer(clock=clock, sleeper=clock.sleep)
    pacer.reset(1.0)

    clock.now_s += 0.1
    assert pacer.wait(0.0) == 0.0
    clock.now_s += 0.02
    assert pacer.wait(0.1) == pytest.approx(0.08)


def test_first_wait_establishes_anchor_without_sleeping() -> None:
    clock = FakeClock()
    pacer = RealTimePacer(clock=clock, sleeper=clock.sleep)

    assert pacer.wait(4.0) == 0.0
    assert clock.sleeps == []


@pytest.mark.parametrize("value", [-0.1, True, float("nan"), float("inf")])
def test_pacer_rejects_invalid_simulation_time(value: float) -> None:
    pacer = RealTimePacer()

    with pytest.raises(ValueError, match="simulation_time_s"):
        pacer.wait(value)


def test_pacer_rejects_non_finite_clock_value() -> None:
    pacer = RealTimePacer(clock=lambda: float("nan"))

    with pytest.raises(ValueError, match="clock"):
        pacer.reset()


@pytest.mark.parametrize("value", [0.0, -0.1, 1.01, True, float("nan")])
def test_pacer_rejects_invalid_realtime_factor(value: float) -> None:
    with pytest.raises(ValueError, match="realtime_factor"):
        RealTimePacer(realtime_factor=value)
