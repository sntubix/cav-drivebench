from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


EVENT_SCHEMA_VERSION = 1


class EventLogError(ValueError):
    """Raised when an event log cannot be decoded safely."""


@dataclass(frozen=True)
class EventRecord:
    schema_version: int
    run_id: str
    scenario_id: str
    sequence: int
    event_type: str
    sim_time_s: float | None
    wall_time_utc: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise EventLogError(
                f"unsupported event schema version: {self.schema_version}"
            )
        for name, value in {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "event_type": self.event_type,
            "wall_time_utc": self.wall_time_utc,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise EventLogError(f"{name} must not be empty")
            if _contains_control_characters(value):
                raise EventLogError(f"{name} must not contain control characters")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise EventLogError("sequence must be a positive integer")
        if self.sim_time_s is not None and not _non_negative_finite(self.sim_time_s):
            raise EventLogError("sim_time_s must be finite and non-negative or null")
        if not isinstance(self.payload, Mapping):
            raise EventLogError("payload must be a JSON object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "sim_time_s": self.sim_time_s,
            "wall_time_utc": self.wall_time_utc,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EventRecord:
        expected = {
            "schema_version",
            "run_id",
            "scenario_id",
            "sequence",
            "event_type",
            "sim_time_s",
            "wall_time_utc",
            "payload",
        }
        unknown = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if missing:
            raise EventLogError(f"event is missing fields: {', '.join(missing)}")
        if unknown:
            raise EventLogError(f"event has unknown fields: {', '.join(unknown)}")
        return cls(**{name: value[name] for name in expected})


class EventLogger:
    """Thread-safe append-only JSON Lines recorder for one simulation run."""

    def __init__(
        self,
        path: Path | str,
        *,
        run_id: str | None = None,
        scenario_id: str = "default",
    ) -> None:
        self.path = Path(path).resolve()
        self.run_id = run_id or uuid.uuid4().hex
        self.scenario_id = scenario_id
        if not self.run_id.strip() or _contains_control_characters(self.run_id):
            raise ValueError("run_id must not be empty or contain control characters")
        if not scenario_id.strip() or _contains_control_characters(scenario_id):
            raise ValueError("scenario_id must not be empty or contain control characters")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._sequence = 0
        self._closed = False
        self._lock = threading.Lock()

    def write(
        self,
        event_type: str,
        *,
        sim_time_s: float | None,
        payload: Mapping[str, Any] | None = None,
    ) -> EventRecord:
        primitive = to_json_value(dict(payload or {}))
        assert isinstance(primitive, dict)
        with self._lock:
            if self._closed:
                raise RuntimeError("event logger is closed")
            record = EventRecord(
                schema_version=EVENT_SCHEMA_VERSION,
                run_id=self.run_id,
                scenario_id=self.scenario_id,
                sequence=self._sequence + 1,
                event_type=event_type,
                sim_time_s=sim_time_s,
                wall_time_utc=datetime.now(timezone.utc).isoformat(),
                payload=primitive,
            )
            encoded = json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            self._stream.write(encoded + "\n")
            self._stream.flush()
            self._sequence = record.sequence
            return record

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stream.close()

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def read_event_log(path: Path | str) -> tuple[EventRecord, ...]:
    log_path = Path(path)
    records: list[EventRecord] = []
    previous_sequence: dict[str, int] = {}
    try:
        lines: Iterable[str] = log_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise EventLogError(f"cannot open event log {log_path}: {exc}") from exc

    with lines as stream:  # type: ignore[attr-defined]
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise EventLogError(f"blank event-log line at {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EventLogError(
                    f"invalid JSON on event-log line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise EventLogError(f"event-log line {line_number} must be a JSON object")
            try:
                record = EventRecord.from_dict(value)
            except (EventLogError, TypeError) as exc:
                raise EventLogError(f"invalid event-log line {line_number}: {exc}") from exc
            expected_sequence = previous_sequence.get(record.run_id, 0) + 1
            if record.sequence != expected_sequence:
                raise EventLogError(
                    f"run {record.run_id!r} has sequence {record.sequence} on line "
                    f"{line_number}; expected {expected_sequence}"
                )
            previous_sequence[record.run_id] = record.sequence
            records.append(record)
    return tuple(records)


def to_json_value(value: Any) -> Any:
    """Convert supported typed project values into strict JSON primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("event values must not contain non-finite numbers")
        return value
    if isinstance(value, Enum):
        return to_json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("event object keys must be strings")
            result[key] = to_json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    raise TypeError(f"unsupported event value type: {type(value).__name__}")


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)
