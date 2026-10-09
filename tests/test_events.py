from __future__ import annotations

import json
from pathlib import Path

import pytest

from metadrive_starter.events import EventLogError, EventLogger, read_event_log
from metadrive_starter.vla import HighLevelAction


def test_event_logger_appends_strict_versioned_records(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "events.jsonl"
    with EventLogger(path, run_id="run-1", scenario_id="curve") as logger:
        first = logger.write(
            "run_started",
            sim_time_s=0.0,
            payload={"action": HighLevelAction.KEEP_LANE},
        )
        second = logger.write("run_ended", sim_time_s=1.5, payload={"steps": 3})

    records = read_event_log(path)

    assert [record.sequence for record in records] == [1, 2]
    assert first.sequence == 1
    assert second.sequence == 2
    assert records[0].payload == {"action": "KEEP_LANE"}
    assert records[0].scenario_id == "curve"
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_two_runs_can_be_appended_with_independent_sequences(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with EventLogger(path, run_id="run-1") as logger:
        logger.write("run_started", sim_time_s=0.0)
    with EventLogger(path, run_id="run-2") as logger:
        logger.write("run_started", sim_time_s=0.0)

    assert [(record.run_id, record.sequence) for record in read_event_log(path)] == [
        ("run-1", 1),
        ("run-2", 1),
    ]


def test_event_logger_rejects_non_finite_payloads(tmp_path: Path) -> None:
    with EventLogger(tmp_path / "events.jsonl") as logger:
        with pytest.raises(ValueError, match="non-finite"):
            logger.write("invalid", sim_time_s=0.0, payload={"value": float("nan")})


def test_event_reader_rejects_sequence_gaps(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    event = {
        "schema_version": 1,
        "run_id": "run-1",
        "scenario_id": "default",
        "sequence": 2,
        "event_type": "run_started",
        "sim_time_s": 0.0,
        "wall_time_utc": "2026-01-01T00:00:00+00:00",
        "payload": {},
    }
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(EventLogError, match="expected 1"):
        read_event_log(path)
