"""Apply a team's agent.yaml overlay onto the instructor base configuration.

An allowlist enumerates the keys an overlay may set, each with optional bounds;
everything else is rejected. Enumerating what is permitted, rather than
stripping what is not, keeps a key nobody thought about locked.
"""

from __future__ import annotations

import copy
import difflib
import math
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Real
from typing import Any, Iterable, Iterator, Mapping

from metadrive_starter.config import AppConfig

# Endpoint and model pins: the instructor sets these at grading time, so no
# allowlist may ever permit them.
INSTRUCTOR_OWNED = (
    "vla.http",
    "vla.vertex.project_env",
    "vla.vertex.location",
    "vla.vertex.model_id",
)


class OverlayError(ValueError):
    """An overlay sets a key it may not, or a value outside its bounds."""


class Direction(str, Enum):
    """The only way a key may move away from its instructor base value."""

    LOWER = "lower"
    RAISE = "raise"


@dataclass(frozen=True)
class PermittedKey:
    """One overlay key, by dotted path through the configuration."""

    path: str
    minimum: float | None = None
    maximum: float | None = None
    only: Direction | None = None


@dataclass(frozen=True)
class Allowlist:
    name: str
    keys: tuple[PermittedKey, ...]

    def __post_init__(self) -> None:
        for key in self.keys:
            if any(
                key.path == owned or key.path.startswith(f"{owned}.")
                for owned in INSTRUCTOR_OWNED
            ):
                raise ValueError(f"{key.path} is instructor-owned and cannot be permitted")


def _pid_loop_keys(loop: str) -> tuple[PermittedKey, ...]:
    return (
        PermittedKey(f"controller.{loop}.kp", minimum=0.0),
        PermittedKey(f"controller.{loop}.ki", minimum=0.0),
        PermittedKey(f"controller.{loop}.kd", minimum=0.0),
        PermittedKey(f"controller.{loop}.anti_windup"),
        PermittedKey(f"controller.{loop}.output_min", minimum=-1.0, maximum=1.0),
        PermittedKey(f"controller.{loop}.output_max", minimum=-1.0, maximum=1.0),
    )


# Speed-raising values may only be lowered, so no team can buy time by
# driving faster than the instructor configuration allows.
ASSIGNMENT_1 = Allowlist(
    "assignment 1",
    (
        *_pid_loop_keys("speed_pid"),
        *_pid_loop_keys("steering_pid"),
        *_pid_loop_keys("lateral_pid"),
        PermittedKey("controller.speed_pid_reset_threshold_mps"),
        PermittedKey("controller.target_speed_mps", only=Direction.LOWER),
        # Not planner.lookahead_m: it moves only Plan.target, which no controller
        # receives, and the heading preview never looks further than 1 m.
        PermittedKey("planner.curvature_preview_m"),
        PermittedKey(
            "planner.maximum_lateral_acceleration_mps2",
            only=Direction.LOWER,
        ),
        PermittedKey("planner.minimum_curve_speed_mps", only=Direction.LOWER),
    ),
)

# Assignment 2 adds what the model sees, the configured prompt text and the
# observation's channels, and the safety floor's command thresholds, which only
# tighten: the monotone restriction (ADR-0001) applies to values as to code.
# None of them reaches a run without the VLA subsystem, so Assignment 1 runs
# stay as they were.
ASSIGNMENT_2 = Allowlist(
    "assignment 2",
    (
        *ASSIGNMENT_1.keys,
        PermittedKey("vla.prompt_policy"),
        PermittedKey("observation.driving_context"),
        PermittedKey("observation.scene_context"),
        PermittedKey("observation.scene_fields.lanes"),
        PermittedKey("observation.scene_fields.traffic_controls"),
        PermittedKey("observation.scene_fields.objects"),
        PermittedKey(
            "command_validation.minimum_confidence", maximum=1.0, only=Direction.RAISE
        ),
        PermittedKey("command_validation.maximum_command_age_s", only=Direction.LOWER),
        PermittedKey("command_validation.maximum_clock_skew_s", only=Direction.LOWER),
        PermittedKey("command_validation.scene_stale_after_s", only=Direction.LOWER),
        PermittedKey("command_validation.maximum_target_speed_mps", only=Direction.LOWER),
        PermittedKey("command_validation.slow_down_speed_mps", only=Direction.LOWER),
        PermittedKey("command_validation.yield_speed_mps", only=Direction.LOWER),
        PermittedKey("command_validation.lane_change_front_gap_m", only=Direction.RAISE),
        PermittedKey("command_validation.lane_change_rear_gap_m", only=Direction.RAISE),
        PermittedKey("command_validation.lane_change_minimum_ttc_s", only=Direction.RAISE),
    ),
)


def apply_overlay(
    config: AppConfig,
    overlay: Mapping[str, Any],
    allowlist: Allowlist,
) -> AppConfig:
    """Return a copy of ``config`` with the overlay's values applied."""
    if not isinstance(overlay, Mapping):
        raise OverlayError(
            f"the overlay must be a mapping of settings; got a {type(overlay).__name__}"
        )
    permitted = {key.path: key for key in allowlist.keys}
    problems: list[str] = []
    changes: dict[str, Any] = {}
    for path, value in _leaves(overlay):
        key = permitted.get(path)
        if key is None:
            problems.append(_not_permitted(path, permitted, allowlist.name))
            continue
        base = _lookup(config, path)
        problem = _problem(key, value, base=base)
        if problem is not None:
            problems.append(problem)
            continue
        _nest(changes, path.split("."), float(value) if isinstance(base, float) else value)
    if problems:
        raise OverlayError(
            problems[0]
            if len(problems) == 1
            else "\n".join(
                ["the overlay has several problems:", *(f"- {p}" for p in problems)]
            )
        )
    try:
        return _apply(copy.deepcopy(config), changes)
    except (TypeError, ValueError) as exc:
        raise OverlayError(f"the overlay makes an invalid configuration: {exc}") from exc


def _not_permitted(path: str, permitted: Iterable[str], allowlist_name: str) -> str:
    problem = f"{path} is not permitted in {allowlist_name}"
    close = difflib.get_close_matches(path, permitted, n=1, cutoff=0.8)
    return f"{problem}; did you mean {close[0]}?" if close else problem


def _problem(key: PermittedKey, value: Any, *, base: Any) -> str | None:
    if isinstance(base, bool):
        if not isinstance(value, bool):
            return f"{key.path} must be true or false; got {value!r}"
        return None
    if isinstance(base, str):
        if not isinstance(value, str):
            return f"{key.path} must be text; got {value!r}"
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return f"{key.path} must be a finite number; got {value!r}"
    if key.minimum is not None and value < key.minimum:
        return f"{key.path} must be at least {key.minimum:g}; got {value:g}"
    if key.maximum is not None and value > key.maximum:
        return f"{key.path} must be at most {key.maximum:g}; got {value:g}"
    if key.only is Direction.LOWER and value > base:
        return f"{key.path} may only be lowered from {base:g}; got {value:g}"
    if key.only is Direction.RAISE and value < base:
        return f"{key.path} may only be raised from {base:g}; got {value:g}"
    return None


def _lookup(config: AppConfig, path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _leaves(mapping: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for name, value in mapping.items():
        path = f"{prefix}{name}"
        if isinstance(value, Mapping):
            yield from _leaves(value, f"{path}.")
        else:
            yield path, value


def _nest(changes: dict[str, Any], parts: list[str], value: Any) -> None:
    for part in parts[:-1]:
        changes = changes.setdefault(part, {})
    changes[parts[-1]] = value


def _apply(settings: Any, changes: Mapping[str, Any]) -> Any:
    return replace(
        settings,
        **{
            name: _apply(getattr(settings, name), change)
            if isinstance(change, dict)
            else change
            for name, change in changes.items()
        },
    )
