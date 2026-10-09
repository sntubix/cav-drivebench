#!/usr/bin/env python3
"""Validate and summarize one native local-VLM closed-loop acceptance run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


class AcceptanceError(ValueError):
    """Raised when closed-loop evidence does not meet the native VLM contract."""


def _final_run_summary(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and "simulation_time_s" in value
            and "vla_metrics" in value
        ):
            candidates.append(value)
    if not candidates:
        raise AcceptanceError("closed-loop log contains no DriveBench run summary")
    return candidates[-1]


def _time_metrics(path: Path) -> tuple[float, int]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw_line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    try:
        wall_seconds = float(values["wall_seconds"])
        max_rss_kib = int(values["max_rss_kib"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AcceptanceError("GNU time evidence is missing or invalid") from exc
    if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise AcceptanceError("wall_seconds must be finite and positive")
    if max_rss_kib < 0:
        raise AcceptanceError("max_rss_kib must be non-negative")
    return wall_seconds, max_rss_kib


def _server_max_rss_kib(path: Path) -> int:
    maximum = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "timestamp_utc,pid,rss_kib,vsz_kib,cpu_percent,elapsed":
        raise AcceptanceError("server resource evidence has an invalid header")
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) != 6:
            raise AcceptanceError("server resource evidence has an invalid row")
        try:
            maximum = max(maximum, int(fields[2]))
        except ValueError as exc:
            raise AcceptanceError("server resource RSS is invalid") from exc
    if len(lines) < 2:
        raise AcceptanceError("server resource evidence contains no samples")
    return maximum


def _probe_decoding(path: Path) -> dict[str, Any]:
    """How many scenes of a probe run decoded: evidence, never an acceptance condition."""
    try:
        scenarios = json.loads(path.read_text(encoding="utf-8"))["scenarios"]
        statuses = [(item["status"], item.get("error_category")) for item in scenarios]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"scenes": None, "decoded": None, "unavailable": f"{type(exc).__name__}: {exc}"}
    failures = Counter(category for status, category in statuses if status != "success")
    return {
        "scenes": len(statuses),
        "decoded": sum(status == "success" for status, _ in statuses),
        "failure_categories": dict(sorted(failures.items(), key=lambda item: str(item[0]))),
    }


def build_acceptance_summary(
    run_log: Path,
    time_file: Path,
    server_resources: Path,
    *,
    require_authority: bool = True,
    schema_off_probe: Path | None = None,
) -> dict[str, Any]:
    """Summarize one closed loop; ``require_authority=False`` still records the
    authority steps, for a model whose answers the safety floor is expected to reject.
    A ``schema_off_probe`` summary is recorded as evidence and never decides the result."""
    run = _final_run_summary(run_log.read_text(encoding="utf-8"))
    wall_seconds, simulator_max_rss_kib = _time_metrics(time_file)
    server_max_rss_kib = _server_max_rss_kib(server_resources)

    metrics = run.get("vla_metrics")
    if not isinstance(metrics, dict):
        raise AcceptanceError("run summary vla_metrics must be an object")

    failures: list[str] = []
    simulation_time_s = run.get("simulation_time_s")
    if (
        isinstance(simulation_time_s, bool)
        or not isinstance(simulation_time_s, (int, float))
        or not math.isfinite(simulation_time_s)
        or simulation_time_s <= 0.0
    ):
        failures.append("simulation_time_s must be finite and positive")
        simulation_time_s = 0.0

    positive = ["provider_requests_attempted", "responses_succeeded"]
    if require_authority:
        positive.append("authority_steps")
    for key in positive:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            failures.append(f"vla_metrics.{key} must be positive")
    runtime_errors = metrics.get("runtime_errors")
    if isinstance(runtime_errors, bool) or not isinstance(runtime_errors, int):
        failures.append("vla_metrics.runtime_errors must be an integer")
    elif runtime_errors != 0:
        failures.append("vla_metrics.runtime_errors must be zero")
    if run.get("crashed") is not False:
        failures.append("crashed must be false")
    if run.get("went_off_road") is not False:
        failures.append("went_off_road must be false")

    summary = {
        "schema_version": 1,
        "successful": not failures,
        "failures": failures,
        "simulation_time_s": simulation_time_s,
        "wall_seconds": wall_seconds,
        "achieved_realtime_factor": simulation_time_s / wall_seconds,
        "simulator_max_rss_kib": simulator_max_rss_kib,
        "server_max_rss_kib": server_max_rss_kib,
        "provider_requests_attempted": metrics.get("provider_requests_attempted"),
        "responses_succeeded": metrics.get("responses_succeeded"),
        "authority_steps": metrics.get("authority_steps"),
        "authority_required": require_authority,
        "runtime_errors": metrics.get("runtime_errors"),
        "crashed": run.get("crashed"),
        "went_off_road": run.get("went_off_road"),
    }
    if schema_off_probe is not None:
        summary["schema_off_probe"] = _probe_decoding(schema_off_probe)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--time-file", type=Path, required=True)
    parser.add_argument("--server-resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-zero-authority",
        action="store_true",
        help="record, but do not require, model authority",
    )
    parser.add_argument(
        "--schema-off-probe",
        type=Path,
        help="probe summary.json whose decode count is recorded as evidence only",
    )
    args = parser.parse_args()

    try:
        summary = build_acceptance_summary(
            args.run_log,
            args.time_file,
            args.server_resources,
            require_authority=not args.allow_zero_authority,
            schema_off_probe=args.schema_off_probe,
        )
    except (AcceptanceError, OSError) as exc:
        parser.exit(1, f"native-vlm-summary: {exc}\n")
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
