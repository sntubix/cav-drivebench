from __future__ import annotations

import json
import math

from metadrive_starter.perception.scene import LaneRelation, LocalScene, TrackedObject
from metadrive_starter.prompt_policy import normalize_prompt_policy
from metadrive_starter.vla.contracts import (
    VLA_ASSESSMENT_MAX_HAZARDS,
    VLA_ASSESSMENT_MAX_TEXT_CHARACTERS,
)
from metadrive_starter.vla.observation import (
    OUTPUT_CONTRACT,
    Observation,
    ObservationRequest,
)


class PromptBuildError(ValueError):
    """Raised when a safe, bounded VLA prompt cannot be constructed."""


def build_vla_prompt(
    *,
    now_s: float,
    action_horizon_s: float,
    scene: LocalScene | None = None,
    max_scene_objects: int = 8,
    ego_speed_mps: float | None = None,
    cruise_speed_mps: float | None = None,
    prompt_policy: str = "",
) -> str:
    """Build a deterministic prompt for one rich, non-authoritative assessment."""

    if (
        not isinstance(now_s, (int, float))
        or isinstance(now_s, bool)
        or not math.isfinite(now_s)
        or now_s < 0.0
    ):
        raise PromptBuildError("now_s must be finite and non-negative")
    if (
        not isinstance(action_horizon_s, (int, float))
        or isinstance(action_horizon_s, bool)
        or not math.isfinite(action_horizon_s)
        or action_horizon_s <= 0.0
    ):
        raise PromptBuildError("action_horizon_s must be finite and positive")
    if isinstance(max_scene_objects, bool) or not isinstance(max_scene_objects, int):
        raise PromptBuildError("max_scene_objects must be an integer")
    if max_scene_objects < 0:
        raise PromptBuildError("max_scene_objects must not be negative")
    if scene is not None and not isinstance(scene, LocalScene):
        raise PromptBuildError("scene must be a LocalScene or None")
    for name, value in {
        "ego_speed_mps": ego_speed_mps,
        "cruise_speed_mps": cruise_speed_mps,
    }.items():
        if value is not None and not _non_negative_finite(value):
            raise PromptBuildError(f"{name} must be finite and non-negative or None")
    try:
        normalized_policy = normalize_prompt_policy(prompt_policy)
    except ValueError as exc:
        raise PromptBuildError(str(exc)) from exc

    effective_ego_speed_mps = ego_speed_mps
    if effective_ego_speed_mps is None and scene is not None:
        effective_ego_speed_mps = scene.ego_speed_mps

    context = build_vla_scene_context(scene, max_scene_objects=max_scene_objects)
    driving_context = None
    if effective_ego_speed_mps is not None or cruise_speed_mps is not None:
        driving_context = {
            "ego_speed_mps": effective_ego_speed_mps,
            "cruise_speed_mps": cruise_speed_mps,
        }
    try:
        context_json = json.dumps(
            context,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        driving_context_json = json.dumps(
            driving_context,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PromptBuildError("prompt context must contain only finite JSON values") from exc

    return "\n".join(
        (
            OUTPUT_CONTRACT.version,
            "You are assessing a driving scene and proposing one high-level manoeuvre.",
            "The attached front-camera image is the PRIMARY evidence for the driving decision.",
            (
                "LOCAL_SCENE_CONTEXT contains measured tactical facts. For lane availability, observed "
                "occupancy, distance, and relative speed, follow it when visual interpretation is ambiguous."
            ),
            OUTPUT_CONTRACT.actions,
            OUTPUT_CONTRACT.fields,
            (
                "Each relevant_hazards item must contain type, relative_location, and risk. "
                "Allowed types: vehicle, pedestrian, obstacle, road_feature, traffic_control, other."
            ),
            (
                f"Return at most {VLA_ASSESSMENT_MAX_HAZARDS} relevant hazards. "
                "Mention each observed object at most once."
            ),
            (
                "Allowed relative locations: front, front_left, front_right, left, right, rear, "
                "rear_left, rear_right. Allowed risks: low, medium, high."
            ),
            (
                "DRIVING_CONTEXT reports the measured ego speed and the desired clear-road cruise "
                "speed in m/s. target_speed_mps is the commanded speed for the next action horizon, not a "
                "copy of the current ego speed."
            ),
            *(("CONFIGURED_POLICY:", normalized_policy) if normalized_policy else ()),
            (
                "Propose a manoeuvre for the next "
                f"{json.dumps(float(action_horizon_s))} seconds. The runtime owns request IDs, "
                "timestamps, and command validity; do not return those fields."
            ),
            "target_speed_mps must be non-negative; confidence must be between 0 and 1.",
            (
                "scene_summary and brief_justification must each contain at most "
                f"{VLA_ASSESSMENT_MAX_TEXT_CHARACTERS} characters. Be concise and do not repeat."
            ),
            OUTPUT_CONTRACT.format,
            "DRIVING_CONTEXT:",
            driving_context_json,
            "LOCAL_SCENE_CONTEXT:",
            context_json,
        )
    )


class DefaultObservationBuilder:
    """DriveBench's original observation: the camera frame as captured, with the
    prompt ``build_vla_prompt`` writes. Pipelines given no builder use it."""

    def build(self, request: ObservationRequest) -> Observation:
        return Observation(
            frame=request.frame,
            prompt=build_vla_prompt(
                now_s=request.now_s,
                action_horizon_s=request.action_horizon_s,
                scene=request.scene,
                max_scene_objects=request.max_scene_objects,
                ego_speed_mps=request.ego_speed_mps,
                cruise_speed_mps=request.cruise_speed_mps,
                prompt_policy=request.prompt_policy,
            ),
        )


def build_vla_scene_context(
    scene: LocalScene | None,
    *,
    max_scene_objects: int,
) -> object:
    """Return the bounded provider-neutral scene payload embedded in VLA prompts.

    Instructor-owned: ``hazard_agreement`` uses this as the ground-truth description of
    the scene, so ``observation.py`` may call it but must never replace it.
    """
    if scene is not None and not isinstance(scene, LocalScene):
        raise PromptBuildError("scene must be a LocalScene or None")
    if isinstance(max_scene_objects, bool) or not isinstance(max_scene_objects, int):
        raise PromptBuildError("max_scene_objects must be an integer")
    if max_scene_objects < 0:
        raise PromptBuildError("max_scene_objects must not be negative")
    if scene is None:
        return None

    objects = sorted(scene.objects, key=_object_sort_key)[:max_scene_objects]
    return {
        "valid": scene.valid,
        "lanes": {
            "left": _lane_context(scene, side=LaneRelation.LEFT),
            "right": _lane_context(scene, side=LaneRelation.RIGHT),
        },
        "traffic_controls": [
            {
                "id": light.light_id,
                "state": light.state.value,
                "relative_position_m": list(light.relative_position_m),
                "in_path": light.in_path,
                "path_distance_m": light.path_distance_m,
                "confidence": light.confidence,
            }
            for light in sorted(scene.traffic_lights, key=lambda light: light.light_id)
        ],
        "objects": [_compact_object(tracked, scene.ego_speed_mps) for tracked in objects],
    }


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def _object_sort_key(tracked: TrackedObject) -> tuple[float, str]:
    if tracked.path_distance_m is not None:
        distance = tracked.path_distance_m
    else:
        x, y = tracked.relative_position_m
        distance = math.hypot(x, y)
    return (distance, tracked.object_id)


def _compact_object(tracked: TrackedObject, ego_speed_mps: float) -> dict[str, object]:
    relative_speed_mps = (
        tracked.path_relative_velocity_mps
        if tracked.path_relative_velocity_mps is not None
        else tracked.relative_velocity_mps[0]
    )
    distance_m = (
        tracked.path_distance_m
        if tracked.path_distance_m is not None
        else math.hypot(*tracked.relative_position_m)
    )
    return {
        "id": tracked.object_id,
        "kind": tracked.kind,
        "relative_location": _tactical_relative_location(tracked),
        "distance_m": _rounded(distance_m),
        "relative_speed_mps": _rounded(relative_speed_mps),
        "estimated_speed_mps": _rounded(max(0.0, ego_speed_mps + relative_speed_mps)),
        "in_path": tracked.in_path,
        "confidence": tracked.confidence,
    }


def _lane_context(scene: LocalScene, *, side: LaneRelation) -> dict[str, object]:
    available = (
        scene.left_lane_available
        if side is LaneRelation.LEFT
        else scene.right_lane_available
    )
    if not scene.valid or available is None:
        observed_clear: bool | None = None
    elif not available:
        observed_clear = False
    else:
        side_name = side.value
        observed_clear = not any(
            _tactical_relative_location(tracked) in {
                side_name,
                f"front_{side_name}",
                f"rear_{side_name}",
            }
            for tracked in scene.objects
        )
    return {"available": available, "observed_clear": observed_clear}


def _tactical_relative_location(tracked: TrackedObject) -> str:
    if tracked.lane_relation is LaneRelation.LEFT:
        side = "left"
    elif tracked.lane_relation is LaneRelation.RIGHT:
        side = "right"
    elif tracked.in_path or tracked.lane_relation in {
        LaneRelation.SAME,
        LaneRelation.CROSSING,
    }:
        side = ""
    elif tracked.relative_position_m[1] > 0.0:
        side = "left"
    elif tracked.relative_position_m[1] < 0.0:
        side = "right"
    else:
        side = ""

    longitudinal_m = tracked.relative_position_m[0]
    if tracked.in_path and tracked.path_distance_m is not None:
        longitudinal = "front"
    elif longitudinal_m > 0.0:
        longitudinal = "front"
    elif longitudinal_m < 0.0:
        longitudinal = "rear"
    else:
        longitudinal = ""
    return "_".join(part for part in (longitudinal, side) if part) or "front"


def _rounded(value: float) -> float:
    return round(float(value), 3)
