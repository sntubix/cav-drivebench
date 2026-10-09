from __future__ import annotations

import json
import re
from collections.abc import Mapping

from metadrive_starter.vla.assessment import VLAAssessment
from metadrive_starter.vla.commands import VLACommand


class VLAResponseError(ValueError):
    """Base error for a model response that cannot produce a VLA command."""


class VLAResponseFormatError(VLAResponseError):
    """Raised when a response is not one unambiguous JSON object."""


class VLACommandPayloadError(VLAResponseError):
    """Raised when the decoded JSON does not satisfy the VLA command schema."""


class VLAAssessmentPayloadError(VLAResponseError):
    """Raised when decoded JSON does not satisfy the VLA assessment schema."""


_JSON_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def decode_vla_response(text: str) -> VLACommand:
    """Decode one direct or singly wrapped JSON command from a model response."""

    decoded = _decode_object(text)

    payload: Mapping[str, object]
    if "command" in decoded:
        if set(decoded) != {"command"}:
            raise VLAResponseFormatError("wrapped response must contain only the command field")
        wrapped = decoded["command"]
        if not isinstance(wrapped, dict):
            raise VLAResponseFormatError("wrapped command must be a JSON object")
        payload = wrapped
    else:
        payload = decoded

    try:
        return VLACommand.from_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise VLACommandPayloadError(f"invalid VLA command payload: {exc}") from exc


def decode_vla_assessment(text: str) -> VLAAssessment:
    """Decode one rich assessment; executable metadata is not accepted."""

    decoded = _decode_object(text)
    payload: Mapping[str, object]
    if "assessment" in decoded:
        if set(decoded) != {"assessment"}:
            raise VLAResponseFormatError(
                "wrapped response must contain only the assessment field"
            )
        wrapped = decoded["assessment"]
        if not isinstance(wrapped, dict):
            raise VLAResponseFormatError("wrapped assessment must be a JSON object")
        payload = wrapped
    else:
        payload = decoded

    try:
        return VLAAssessment.from_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise VLAAssessmentPayloadError(f"invalid VLA assessment payload: {exc}") from exc


def _decode_object(text: str) -> dict[str, object]:

    if not isinstance(text, str):
        raise VLAResponseFormatError("VLA response must be text")
    stripped = text.strip()
    if not stripped:
        raise VLAResponseFormatError("VLA response is empty")

    fence = _JSON_FENCE.fullmatch(stripped)
    if stripped.startswith("```"):
        if fence is None:
            raise VLAResponseFormatError("response must contain exactly one JSON fence and no prose")
        candidate = fence.group("body").strip()
    else:
        candidate = stripped

    try:
        decoded = json.loads(
            candidate,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise VLAResponseFormatError(f"response is not valid JSON: {detail}") from exc

    if not isinstance(decoded, dict):
        raise VLAResponseFormatError("VLA response JSON must be an object")

    return decoded


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON number: {value}")
