from __future__ import annotations

import math
import time
from collections.abc import Callable


# Simulation timestamps are repeatedly produced through floating-point
# multiplication and JSON round trips. This tolerance only absorbs that
# representation noise; policy-level clock skew remains separately configured.
FLOAT_TIME_TOLERANCE_S = 1e-9


class RealTimePacer:
    """Pace simulation time against monotonic wall time at a fixed rate.

    Falling behind re-anchors the schedule instead of running later steps faster
    to catch up. Simulation timestamps remain the source of truth for physics and
    command validity; the wall clock is used only to wait. A factor of 0.5 makes
    one simulation second consume two wall-clock seconds.
    """

    def __init__(
        self,
        *,
        realtime_factor: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            isinstance(realtime_factor, bool)
            or not isinstance(realtime_factor, (int, float))
            or not math.isfinite(realtime_factor)
            or not 0.0 < realtime_factor <= 1.0
        ):
            raise ValueError("realtime_factor must be finite and between 0 and 1")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(sleeper):
            raise TypeError("sleeper must be callable")
        self._clock = clock
        self._sleeper = sleeper
        self.realtime_factor = float(realtime_factor)
        self._simulation_anchor_s: float | None = None
        self._wall_anchor_s: float | None = None
        self._last_simulation_time_s: float | None = None

    def reset(self, simulation_time_s: float = 0.0) -> None:
        simulation_time = _non_negative_finite_time(
            simulation_time_s,
            "simulation_time_s",
        )
        wall_time = float(self._clock())
        if not math.isfinite(wall_time):
            raise ValueError("clock must return a finite value")
        self._simulation_anchor_s = simulation_time
        self._wall_anchor_s = wall_time
        self._last_simulation_time_s = simulation_time

    def wait(self, simulation_time_s: float) -> float:
        """Wait until wall time reaches ``simulation_time_s``; return requested sleep."""
        simulation_time = _non_negative_finite_time(
            simulation_time_s,
            "simulation_time_s",
        )
        if self._simulation_anchor_s is None or self._wall_anchor_s is None:
            self.reset(simulation_time)
            return 0.0
        assert self._last_simulation_time_s is not None
        if simulation_time < self._last_simulation_time_s:
            self.reset(simulation_time)
            return 0.0

        wall_time = float(self._clock())
        if not math.isfinite(wall_time):
            raise ValueError("clock must return a finite value")
        deadline = self._wall_anchor_s + (
            simulation_time - self._simulation_anchor_s
        ) / self.realtime_factor
        remaining_s = deadline - wall_time
        self._last_simulation_time_s = simulation_time
        if remaining_s > 0.0:
            self._sleeper(remaining_s)
            return remaining_s

        # Do not accelerate future steps to recover lost wall time.
        self._simulation_anchor_s = simulation_time
        self._wall_anchor_s = wall_time
        return 0.0


def times_equal(first_s: float, second_s: float) -> bool:
    """Return whether two finite timestamps differ only by representation noise."""

    return math.isclose(
        float(first_s),
        float(second_s),
        rel_tol=0.0,
        abs_tol=FLOAT_TIME_TOLERANCE_S,
    )


def exceeds_time_limit(value_s: float, limit_s: float) -> bool:
    """Return whether a value is materially beyond an inclusive time boundary."""

    return value_s > limit_s and not times_equal(value_s, limit_s)


def precedes_time_limit(value_s: float, limit_s: float) -> bool:
    """Return whether a value is materially before an inclusive time boundary."""

    return value_s < limit_s and not times_equal(value_s, limit_s)


def _non_negative_finite_time(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)
