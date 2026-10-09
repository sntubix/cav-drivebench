from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from metadrive_starter.vla.commands import MODEL_ACTIONS, HighLevelAction, VLACommand
from metadrive_starter.vla.contracts import (
    VLA_ASSESSMENT_MAX_HAZARDS,
    VLA_ASSESSMENT_MAX_TEXT_CHARACTERS,
)


class HazardType(str, Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    OBSTACLE = "obstacle"
    ROAD_FEATURE = "road_feature"
    TRAFFIC_CONTROL = "traffic_control"
    OTHER = "other"


class RelativeLocation(str, Enum):
    FRONT = "front"
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    LEFT = "left"
    RIGHT = "right"
    REAR = "rear"
    REAR_LEFT = "rear_left"
    REAR_RIGHT = "rear_right"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class VLAHazard:
    """One model-reported hazard; descriptive, never actuator authority."""

    hazard_type: HazardType
    relative_location: RelativeLocation
    risk: RiskLevel

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> VLAHazard:
        required = {"type", "relative_location", "risk"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"missing VLA hazard fields: {', '.join(missing)}")
        unknown = sorted(payload.keys() - required)
        if unknown:
            raise ValueError(f"unknown VLA hazard fields: {', '.join(unknown)}")
        return cls(
            hazard_type=_enum_value(HazardType, payload["type"], "hazard type"),
            relative_location=_enum_value(
                RelativeLocation,
                payload["relative_location"],
                "relative location",
            ),
            risk=_enum_value(RiskLevel, payload["risk"], "risk level"),
        )


@dataclass(frozen=True)
class VLAAssessment:
    """Validated model assessment before local execution metadata is attached."""

    scene_summary: str
    relevant_hazards: tuple[VLAHazard, ...]
    proposed_action: HighLevelAction
    proposed_target_speed_mps: float
    confidence: float
    brief_justification: str

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.confidence

    def to_command(
        self,
        *,
        command_id: str,
        issued_at_s: float,
        action_horizon_s: float,
    ) -> VLACommand:
        """Copy proposal fields and stamp locally owned execution metadata."""

        return VLACommand(
            action=self.proposed_action,
            target_speed_mps=self.proposed_target_speed_mps,
            issued_at_s=float(issued_at_s),
            action_horizon_s=float(action_horizon_s),
            confidence=self.confidence,
            command_id=command_id,
            justification=self.brief_justification,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> VLAAssessment:
        required = {
            "scene_summary",
            "relevant_hazards",
            "meta_action",
            "target_speed_mps",
            "confidence",
            "brief_justification",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"missing VLA assessment fields: {', '.join(missing)}")
        unknown = sorted(payload.keys() - required)
        if unknown:
            raise ValueError(f"unknown VLA assessment fields: {', '.join(unknown)}")

        hazards = payload["relevant_hazards"]
        if not isinstance(hazards, list):
            raise ValueError("relevant_hazards must be an array")
        if len(hazards) > VLA_ASSESSMENT_MAX_HAZARDS:
            raise ValueError(
                "relevant_hazards must contain at most "
                f"{VLA_ASSESSMENT_MAX_HAZARDS} items"
            )
        parsed_hazards: list[VLAHazard] = []
        for hazard in hazards:
            if not isinstance(hazard, dict):
                raise ValueError("each relevant hazard must be an object")
            parsed_hazards.append(VLAHazard.from_payload(hazard))

        return cls(
            scene_summary=_bounded_text(payload, "scene_summary"),
            relevant_hazards=tuple(parsed_hazards),
            proposed_action=_enum_value(
                HighLevelAction,
                payload["meta_action"],
                "VLA action",
            ),
            proposed_target_speed_mps=_number(payload, "target_speed_mps", minimum=0.0),
            confidence=_number(payload, "confidence", minimum=0.0, maximum=1.0),
            brief_justification=_bounded_text(payload, "brief_justification"),
        )


def vla_assessment_json_schema() -> dict[str, object]:
    """Strict JSON Schema for OpenAI-compatible structured output."""

    return {
        "type": "object",
        "properties": {
            "scene_summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": VLA_ASSESSMENT_MAX_TEXT_CHARACTERS,
            },
            "relevant_hazards": {
                "type": "array",
                "maxItems": VLA_ASSESSMENT_MAX_HAZARDS,
                "items": _hazard_schema(vertex=False),
            },
            "meta_action": {
                "type": "string",
                "enum": [action.value for action in MODEL_ACTIONS],
            },
            "target_speed_mps": {"type": "number", "minimum": 0},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "brief_justification": {
                "type": "string",
                "minLength": 1,
                "maxLength": VLA_ASSESSMENT_MAX_TEXT_CHARACTERS,
            },
        },
        "required": [
            "scene_summary",
            "relevant_hazards",
            "meta_action",
            "target_speed_mps",
            "confidence",
            "brief_justification",
        ],
        "additionalProperties": False,
    }


def vertex_vla_assessment_schema() -> dict[str, object]:
    """Vertex response schema for the same model-facing assessment contract."""

    return {
        "type": "OBJECT",
        "properties": {
            "scene_summary": {"type": "STRING"},
            "relevant_hazards": {
                "type": "ARRAY",
                "items": _hazard_schema(vertex=True),
            },
            "meta_action": {
                "type": "STRING",
                "enum": [action.value for action in MODEL_ACTIONS],
            },
            "target_speed_mps": {"type": "NUMBER"},
            "confidence": {"type": "NUMBER"},
            "brief_justification": {"type": "STRING"},
        },
        "required": [
            "scene_summary",
            "relevant_hazards",
            "meta_action",
            "target_speed_mps",
            "confidence",
            "brief_justification",
        ],
    }


def _hazard_schema(*, vertex: bool) -> dict[str, object]:
    object_type = "OBJECT" if vertex else "object"
    string_type = "STRING" if vertex else "string"
    schema: dict[str, object] = {
        "type": object_type,
        "properties": {
            "type": {"type": string_type, "enum": [item.value for item in HazardType]},
            "relative_location": {
                "type": string_type,
                "enum": [item.value for item in RelativeLocation],
            },
            "risk": {"type": string_type, "enum": [item.value for item in RiskLevel]},
        },
        "required": ["type", "relative_location", "risk"],
    }
    if not vertex:
        schema["additionalProperties"] = False
    return schema


def _enum_value(enum_type: type[Enum], value: object, name: str):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unknown {name}: {value!r}") from exc


def _number(
    payload: Mapping[str, object],
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < minimum or maximum is not None and result > maximum:
        bound = (
            f"between {minimum:g} and {maximum:g}"
            if maximum is not None
            else f"at least {minimum:g}"
        )
        raise ValueError(f"{name} must be {bound}")
    return result


def _bounded_text(payload: Mapping[str, object], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    if len(value) > VLA_ASSESSMENT_MAX_TEXT_CHARACTERS:
        raise ValueError(
            f"{name} must contain at most {VLA_ASSESSMENT_MAX_TEXT_CHARACTERS} characters"
        )
    return value
