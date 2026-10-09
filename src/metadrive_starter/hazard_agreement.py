"""Offline hazard agreement and confidence calibration over event logs.

Every ``command_validation`` event holds the assessment a VLM returned beside the
``LocalScene`` measured when that assessment was validated. Scoring one against the
other asks whether the model's hazard reports, and the confidence it attaches to
them, carry signal that arbitration could use. Nothing here runs the simulator or
a model: it reads event logs that already exist.

The scene is the one logged at validation, which is what an arbiter sees. The model
observed its frame earlier, when the request was issued; each scored assessment
records that gap as its observation age.

Live-model evidence is never pooled with synthetic evidence. An assessment is live
only when its run was configured for a model provider and its response carries no
fixture identity. Everything else, a replayed capture of a real model included, is
synthetic: it measures whoever authored or first recorded the answer, not a model
call, and counting a replay again would inflate the live corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from metadrive_starter.events import (
    EventLogError,
    EventRecord,
    read_event_log,
    to_json_value,
)
from metadrive_starter.perception.scene import LocalScene
from metadrive_starter.replay import scene_from_payload
from metadrive_starter.traffic_lights import TrafficLightState
from metadrive_starter.vla.assessment import (
    HazardType,
    RelativeLocation,
    RiskLevel,
    VLAHazard,
)
from metadrive_starter.vla.prompting import build_vla_scene_context


class EvidenceClass(str, Enum):
    LIVE = "live"
    SYNTHETIC = "synthetic"


# Hazard types a LocalScene can confirm or refute. It has no counterpart for
# road_feature or other, so those claims are counted but never scored.
CHECKABLE_HAZARD_TYPES = frozenset(
    {
        HazardType.VEHICLE,
        HazardType.PEDESTRIAN,
        HazardType.OBSTACLE,
        HazardType.TRAFFIC_CONTROL,
    }
)

CALIBRATION_BIN_COUNT = 10

_LOCATION_RING = (
    RelativeLocation.FRONT,
    RelativeLocation.FRONT_LEFT,
    RelativeLocation.LEFT,
    RelativeLocation.REAR_LEFT,
    RelativeLocation.REAR,
    RelativeLocation.REAR_RIGHT,
    RelativeLocation.RIGHT,
    RelativeLocation.FRONT_RIGHT,
)
_PEDESTRIAN_KINDS = frozenset({"pedestrian", "cyclist"})
_MODEL_PROVIDERS = frozenset({"http", "vertex"})
# Response metadata written by a live model call; logs older than the field have none.
_LIVE_RESPONSE_PROVIDERS = frozenset({None, "openai_compatible_http", "vertex"})


@dataclass(frozen=True)
class SceneHazard:
    """One entry of the local scene, described in the vocabulary of a model hazard.

    ``required`` is false only for a green light: it supports a traffic-control
    claim, but a model that leaves it out has missed nothing.
    """

    source_id: str
    hazard_type: HazardType
    relative_location: RelativeLocation
    in_path: bool
    distance_m: float
    required: bool = True


@dataclass(frozen=True)
class HazardAgreement:
    """One assessment's hazards scored against one local scene."""

    claims: tuple[VLAHazard, ...]
    unverifiable_claims: int
    scene_hazards: tuple[SceneHazard, ...]
    matched: tuple[SceneHazard, ...]
    false_claims: tuple[VLAHazard, ...]
    missed: tuple[SceneHazard, ...]

    @property
    def agrees(self) -> bool:
        """No hazard invented, and none in the ego path left out."""
        return not self.false_claims and not any(hazard.in_path for hazard in self.missed)


@dataclass(frozen=True)
class ScoredAssessment:
    source: str
    run_id: str
    sim_time_s: float
    evidence: EvidenceClass
    profile: str
    confidence: float
    observation_age_s: float
    agreement: HazardAgreement


@dataclass(frozen=True)
class SkippedEvent:
    source: str
    run_id: str
    sequence: int
    evidence: EvidenceClass
    profile: str
    reason: str


@dataclass(frozen=True)
class SkippedFile:
    source: str
    reason: str


@dataclass(frozen=True)
class CorpusScan:
    logs_read: int
    assessments: tuple[ScoredAssessment, ...]
    skipped_events: tuple[SkippedEvent, ...]
    skipped_files: tuple[SkippedFile, ...]


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    assessments: int
    mean_confidence: float
    agreement_rate: float


@dataclass(frozen=True)
class Calibration:
    """Confidence against hazard agreement.

    ``auroc`` is the chance that an agreeing assessment is more confident than a
    disagreeing one, ties counting half: 0.5 means confidence cannot tell them
    apart. It is None unless both kinds occur.
    """

    bins: tuple[CalibrationBin, ...]
    distinct_confidences: int
    minimum_confidence: float
    maximum_confidence: float
    expected_calibration_error: float
    auroc: float | None


@dataclass(frozen=True)
class ProfileSummary:
    evidence: EvidenceClass
    profile: str
    runs: int
    assessments: int
    scenes_with_hazards: int
    scenes_with_in_path_hazards: int
    claims: int
    matched_claims: int
    unverifiable_claims: int
    scene_hazards: int
    reported_scene_hazards: int
    in_path_hazards: int
    reported_in_path_hazards: int
    agreeing: int
    precision: float | None
    recall: float | None
    in_path_recall: float | None
    agreement_rate: float
    median_observation_age_s: float
    maximum_observation_age_s: float
    calibration: Calibration
    disagreements: tuple[ScoredAssessment, ...]


@dataclass(frozen=True)
class AgreementReport:
    logs_read: int
    profiles: tuple[ProfileSummary, ...]
    skipped_events: tuple[SkippedEvent, ...]
    skipped_files: tuple[SkippedFile, ...]


def scene_hazards(scene: LocalScene) -> tuple[SceneHazard, ...]:
    """List the scene's objects and signals as hazards, nearest in-path first.

    Objects are located exactly as the prompt describes them to the model, so a
    claim is judged in the vocabulary the model was given. A signal is front or
    rear only; the one-sector matching tolerance covers its side of the road.
    """
    context = build_vla_scene_context(scene, max_scene_objects=len(scene.objects))
    assert isinstance(context, dict)
    locations = {item["id"]: item for item in context["objects"]}
    hazards = [
        SceneHazard(
            source_id=tracked.object_id,
            hazard_type=_object_hazard_type(tracked.kind),
            relative_location=RelativeLocation(
                locations[tracked.object_id]["relative_location"]
            ),
            in_path=tracked.in_path,
            distance_m=float(locations[tracked.object_id]["distance_m"]),
        )
        for tracked in scene.objects
    ]
    for light in scene.traffic_lights:
        longitudinal_m, lateral_m = light.relative_position_m
        hazards.append(
            SceneHazard(
                source_id=light.light_id,
                hazard_type=HazardType.TRAFFIC_CONTROL,
                relative_location=(
                    RelativeLocation.FRONT
                    if longitudinal_m >= 0.0
                    else RelativeLocation.REAR
                ),
                in_path=light.in_path,
                distance_m=(
                    light.path_distance_m
                    if light.path_distance_m is not None
                    else math.hypot(longitudinal_m, lateral_m)
                ),
                required=light.state is not TrafficLightState.GREEN,
            )
        )
    return tuple(
        sorted(
            hazards,
            key=lambda hazard: (not hazard.required, not hazard.in_path, hazard.distance_m),
        )
    )


def score_hazards(hazards: Sequence[VLAHazard], scene: LocalScene) -> HazardAgreement:
    """Pair each checkable claim with one scene hazard of its type.

    A claim matches when its location is at most one 45-degree sector from the
    scene hazard's. Exact locations pair first, so a near miss never takes a
    hazard another claim names exactly. A claim left over is false; a required
    scene hazard left over is missed. Risk levels are not compared.
    """
    truth = scene_hazards(scene)
    claims = tuple(
        hazard for hazard in hazards if hazard.hazard_type in CHECKABLE_HAZARD_TYPES
    )
    unmatched = list(range(len(truth)))
    pending = list(claims)
    for tolerance in (0, 1):
        remaining: list[VLAHazard] = []
        for claim in pending:
            hit = next(
                (
                    index
                    for index in unmatched
                    if truth[index].hazard_type is claim.hazard_type
                    and _sector_distance(
                        truth[index].relative_location, claim.relative_location
                    )
                    <= tolerance
                ),
                None,
            )
            if hit is None:
                remaining.append(claim)
            else:
                unmatched.remove(hit)
        pending = remaining
    return HazardAgreement(
        claims=claims,
        unverifiable_claims=len(hazards) - len(claims),
        scene_hazards=truth,
        matched=tuple(
            hazard for index, hazard in enumerate(truth) if index not in unmatched
        ),
        false_claims=tuple(pending),
        missed=tuple(truth[index] for index in unmatched if truth[index].required),
    )


def scan_event_logs(paths: Iterable[Path | str]) -> CorpusScan:
    """Score every logged assessment in the given logs or directories of ``*.jsonl``.

    A JSONL file that is not an event log is skipped and listed, not fatal, so a
    whole evidence directory can be scanned at once.
    """
    assessments: list[ScoredAssessment] = []
    skipped_events: list[SkippedEvent] = []
    skipped_files: list[SkippedFile] = []
    logs_read = 0
    for path, label in _event_log_paths(paths):
        try:
            records = read_event_log(path)
        except EventLogError as exc:
            skipped_files.append(SkippedFile(label, str(exc)))
            continue
        logs_read += 1
        for outcome in _score_records(records, source=label):
            if isinstance(outcome, ScoredAssessment):
                assessments.append(outcome)
            else:
                skipped_events.append(outcome)
    return CorpusScan(
        logs_read=logs_read,
        assessments=tuple(assessments),
        skipped_events=tuple(skipped_events),
        skipped_files=tuple(skipped_files),
    )


def calibrate(assessments: Sequence[ScoredAssessment]) -> Calibration:
    """Bin confidence in tenths and compare each bin with its agreement rate."""
    if not assessments:
        raise ValueError("calibration needs at least one assessment")
    grouped: dict[int, list[ScoredAssessment]] = {}
    for assessment in assessments:
        index = min(
            int(assessment.confidence * CALIBRATION_BIN_COUNT),
            CALIBRATION_BIN_COUNT - 1,
        )
        grouped.setdefault(index, []).append(assessment)
    bins = tuple(
        CalibrationBin(
            lower=index / CALIBRATION_BIN_COUNT,
            upper=(index + 1) / CALIBRATION_BIN_COUNT,
            assessments=len(members),
            mean_confidence=statistics.fmean(item.confidence for item in members),
            agreement_rate=_rate(members),
        )
        for index, members in sorted(grouped.items())
    )
    confidences = [assessment.confidence for assessment in assessments]
    return Calibration(
        bins=bins,
        distinct_confidences=len(set(confidences)),
        minimum_confidence=min(confidences),
        maximum_confidence=max(confidences),
        expected_calibration_error=sum(
            item.assessments / len(assessments)
            * abs(item.agreement_rate - item.mean_confidence)
            for item in bins
        ),
        auroc=_auroc(assessments),
    )


def summarize(scan: CorpusScan) -> AgreementReport:
    """Summarize per evidence class and model profile, live profiles first."""
    grouped: dict[tuple[EvidenceClass, str], list[ScoredAssessment]] = {}
    for assessment in scan.assessments:
        grouped.setdefault((assessment.evidence, assessment.profile), []).append(assessment)
    order = {EvidenceClass.LIVE: 0, EvidenceClass.SYNTHETIC: 1}
    profiles = tuple(
        _profile_summary(evidence, profile, members)
        for (evidence, profile), members in sorted(
            grouped.items(), key=lambda item: (order[item[0][0]], item[0][1])
        )
    )
    return AgreementReport(
        logs_read=scan.logs_read,
        profiles=profiles,
        skipped_events=scan.skipped_events,
        skipped_files=scan.skipped_files,
    )


def format_report(report: AgreementReport, *, disagreement_limit: int = 5) -> str:
    lines = [
        "Hazard agreement: logged assessments against the LocalScene at validation",
        f"Read {_count(report.logs_read, 'event log')}; "
        f"{_count(len(report.skipped_files), 'other JSONL file')} skipped.",
    ]
    for evidence, heading in (
        (EvidenceClass.LIVE, "Live model evidence"),
        (
            EvidenceClass.SYNTHETIC,
            "Synthetic evidence (fixtures and replays, never pooled with live)",
        ),
    ):
        profiles = [item for item in report.profiles if item.evidence is evidence]
        skipped = [item for item in report.skipped_events if item.evidence is evidence]
        lines += [
            "",
            f"{heading}: "
            f"{_count(sum(item.assessments for item in profiles), 'assessment')}, "
            f"{_count(len(profiles), 'profile')}, "
            f"{_count(sum(item.runs for item in profiles), 'run')}",
        ]
        reasons: dict[str, int] = {}
        for item in skipped:
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
        for reason, count in sorted(reasons.items()):
            lines.append(
                f"  {_count(count, 'command_validation event')} skipped: {reason}"
            )
        for summary in profiles:
            lines += _format_profile(summary, disagreement_limit)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m metadrive_starter.hazard_agreement",
        description=(
            "Score the hazards in every logged assessment against the local scene "
            "it was validated in, and report precision, recall, and confidence "
            "calibration per model profile, keeping live-model evidence apart "
            "from fixtures and replays."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="event logs, or directories searched recursively for *.jsonl",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    try:
        report = summarize(scan_event_logs(args.paths))
    except FileNotFoundError as exc:
        parser.exit(1, f"hazard_agreement: {exc}\n")
    if args.json:
        print(json.dumps(to_json_value(report), indent=2))
    else:
        print(format_report(report))
    return 0


def _event_log_paths(paths: Iterable[Path | str]) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            found.extend(
                (candidate, str(candidate.relative_to(path)))
                for candidate in sorted(path.rglob("*.jsonl"))
                if candidate.is_file()
            )
        elif path.is_file():
            found.append((path, str(path)))
        else:
            raise FileNotFoundError(f"no event log or directory at {path}")
    return found


@dataclass(frozen=True)
class _RunContext:
    provider: str
    model_id: str | None


def _score_records(
    records: Sequence[EventRecord], *, source: str
) -> list[ScoredAssessment | SkippedEvent]:
    runs: dict[str, _RunContext] = {}
    responses: dict[tuple[str, str], Mapping[str, Any]] = {}
    outcomes: list[ScoredAssessment | SkippedEvent] = []
    for record in records:
        payload = record.payload
        if record.event_type == "run_started":
            runs[record.run_id] = _run_context(payload)
        elif record.event_type == "inference_completed":
            request_id = payload.get("request_id")
            if isinstance(request_id, str):
                responses[(record.run_id, request_id)] = payload
        elif record.event_type == "command_validation":
            outcomes.append(
                _score_event(record, runs.get(record.run_id), responses, source=source)
            )
    return outcomes


def _run_context(payload: Mapping[str, Any]) -> _RunContext:
    config = payload.get("config")
    vla = config.get("vla") if isinstance(config, Mapping) else None
    if not isinstance(vla, Mapping):
        vla = {}
    # Logs older than vla.provider were HTTP-only, which is still its default.
    provider = str(vla.get("provider", "http"))
    settings = vla.get(provider)
    model_id = settings.get("model_id") if isinstance(settings, Mapping) else None
    return _RunContext(provider, model_id if isinstance(model_id, str) else None)


def _score_event(
    record: EventRecord,
    run: _RunContext | None,
    responses: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    source: str,
) -> ScoredAssessment | SkippedEvent:
    payload = record.payload
    command = payload.get("requested_command")
    if not isinstance(command, Mapping):
        command = {}
    response = responses.get((record.run_id, str(command.get("command_id"))), {})
    evidence, profile = _identify(run, response)

    def skip(reason: str) -> SkippedEvent:
        return SkippedEvent(source, record.run_id, record.sequence, evidence, profile, reason)

    assessment = payload.get("assessment")
    if not isinstance(assessment, Mapping):
        return skip("no assessment recorded")
    try:
        scene = scene_from_payload(payload.get("scene"))
        hazards = tuple(_hazard(item) for item in assessment["relevant_hazards"])
        confidence = _finite(assessment["confidence"])
        now_s = _finite(payload["now_s"])
        issued_at_s = _finite(command["issued_at_s"])
    except (KeyError, TypeError, ValueError) as exc:
        return skip(f"undecodable: {exc}")
    if not scene.valid:
        return skip("scene invalid")
    return ScoredAssessment(
        source=source,
        run_id=record.run_id,
        sim_time_s=now_s,
        evidence=evidence,
        profile=profile,
        confidence=confidence,
        observation_age_s=now_s - issued_at_s,
        agreement=score_hazards(hazards, scene),
    )


def _identify(
    run: _RunContext | None, response: Mapping[str, Any]
) -> tuple[EvidenceClass, str]:
    metadata = response.get("provider_metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    model_id = response.get("model_id") or (run.model_id if run else None) or "unknown"
    provider = run.provider if run else "unknown"
    live = (
        provider in _MODEL_PROVIDERS
        and metadata.get("provider") in _LIVE_RESPONSE_PROVIDERS
        and not metadata.get("fixture_id")
        and not metadata.get("fixture_sha256")
    )
    evidence = EvidenceClass.LIVE if live else EvidenceClass.SYNTHETIC
    return evidence, f"{provider}:{model_id}"


def _hazard(value: object) -> VLAHazard:
    if not isinstance(value, Mapping):
        raise ValueError("each hazard must be an object")
    return VLAHazard(
        hazard_type=HazardType(value["hazard_type"]),
        relative_location=RelativeLocation(value["relative_location"]),
        risk=RiskLevel(value["risk"]),
    )


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected a finite number, got {value!r}")
    return result


def _object_hazard_type(kind: str) -> HazardType:
    if kind == "vehicle":
        return HazardType.VEHICLE
    if kind in _PEDESTRIAN_KINDS:
        return HazardType.PEDESTRIAN
    return HazardType.OBSTACLE


def _sector_distance(first: RelativeLocation, second: RelativeLocation) -> int:
    steps = abs(_LOCATION_RING.index(first) - _LOCATION_RING.index(second))
    return min(steps, len(_LOCATION_RING) - steps)


def _rate(assessments: Sequence[ScoredAssessment]) -> float:
    return sum(item.agreement.agrees for item in assessments) / len(assessments)


def _auroc(assessments: Sequence[ScoredAssessment]) -> float | None:
    agreeing = [item.confidence for item in assessments if item.agreement.agrees]
    disagreeing = [item.confidence for item in assessments if not item.agreement.agrees]
    if not agreeing or not disagreeing:
        return None
    wins = sum(
        1.0 if high > low else 0.5 if high == low else 0.0
        for high in agreeing
        for low in disagreeing
    )
    return wins / (len(agreeing) * len(disagreeing))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _profile_summary(
    evidence: EvidenceClass, profile: str, members: Sequence[ScoredAssessment]
) -> ProfileSummary:
    agreements = [item.agreement for item in members]
    required = [
        hazard for agreement in agreements for hazard in agreement.scene_hazards if hazard.required
    ]
    reported = [
        hazard for agreement in agreements for hazard in agreement.matched if hazard.required
    ]
    claims = sum(len(agreement.claims) for agreement in agreements)
    matched_claims = sum(len(agreement.matched) for agreement in agreements)
    in_path = sum(hazard.in_path for hazard in required)
    reported_in_path = sum(hazard.in_path for hazard in reported)
    ages = [item.observation_age_s for item in members]
    return ProfileSummary(
        evidence=evidence,
        profile=profile,
        runs=len({(item.source, item.run_id) for item in members}),
        assessments=len(members),
        scenes_with_hazards=sum(
            any(hazard.required for hazard in agreement.scene_hazards)
            for agreement in agreements
        ),
        scenes_with_in_path_hazards=sum(
            any(hazard.required and hazard.in_path for hazard in agreement.scene_hazards)
            for agreement in agreements
        ),
        claims=claims,
        matched_claims=matched_claims,
        unverifiable_claims=sum(agreement.unverifiable_claims for agreement in agreements),
        scene_hazards=len(required),
        reported_scene_hazards=len(reported),
        in_path_hazards=in_path,
        reported_in_path_hazards=reported_in_path,
        agreeing=sum(agreement.agrees for agreement in agreements),
        precision=_ratio(matched_claims, claims),
        recall=_ratio(len(reported), len(required)),
        in_path_recall=_ratio(reported_in_path, in_path),
        agreement_rate=_rate(members),
        median_observation_age_s=statistics.median(ages),
        maximum_observation_age_s=max(ages),
        calibration=calibrate(members),
        disagreements=tuple(item for item in members if not item.agreement.agrees),
    )


def _format_profile(summary: ProfileSummary, disagreement_limit: int) -> list[str]:
    calibration = summary.calibration
    auroc = "n/a" if calibration.auroc is None else f"{calibration.auroc:.2f}"
    lines = [
        "",
        f"  {summary.profile}: {_count(summary.assessments, 'assessment')} "
        f"in {_count(summary.runs, 'run')}",
        f"    scenes          {summary.scenes_with_hazards} with a hazard, "
        f"{summary.scenes_with_in_path_hazards} with one in path",
        f"    precision       {_share(summary.precision)}  "
        f"({summary.matched_claims} of {summary.claims} checkable claims; "
        f"{summary.unverifiable_claims} road_feature/other not checkable)",
        f"    recall          {_share(summary.recall)}  "
        f"({summary.reported_scene_hazards} of {summary.scene_hazards} scene hazards)",
        f"    in-path recall  {_share(summary.in_path_recall)}  "
        f"({summary.reported_in_path_hazards} of {summary.in_path_hazards})",
        f"    agreement       {_share(summary.agreement_rate)}  "
        f"({summary.agreeing} of {summary.assessments} assessments)",
        f"    observation age median {summary.median_observation_age_s:.2f} s, "
        f"max {summary.maximum_observation_age_s:.2f} s",
        f"    confidence      {_count(calibration.distinct_confidences, 'distinct value')}, "
        f"{calibration.minimum_confidence:.2f} to {calibration.maximum_confidence:.2f}; "
        f"ECE {calibration.expected_calibration_error:.2f}; AUROC {auroc}",
        "      bin        n  mean conf  agreement",
    ]
    lines += [
        f"      {item.lower:.1f}-{item.upper:.1f}  {item.assessments:>3}  "
        f"{item.mean_confidence:>9.2f}  {item.agreement_rate:>9.2f}"
        for item in calibration.bins
    ]
    if summary.disagreements:
        lines.append("    disagreements")
        for item in summary.disagreements[:disagreement_limit]:
            lines.append(f"      {_describe_disagreement(item)}")
        hidden = len(summary.disagreements) - disagreement_limit
        if hidden > 0:
            lines.append(f"      ... and {hidden} more")
    return lines


def _describe_disagreement(item: ScoredAssessment) -> str:
    agreement = item.agreement
    parts = []
    missed = [hazard for hazard in agreement.missed if hazard.in_path]
    if missed:
        parts.append(
            "missed "
            + ", ".join(
                f"{hazard.hazard_type.value} {hazard.relative_location.value} "
                f"in path {hazard.distance_m:.1f} m"
                for hazard in missed
            )
        )
    if agreement.false_claims:
        parts.append(
            "no scene support for "
            + ", ".join(
                f"{hazard.hazard_type.value} {hazard.relative_location.value} "
                f"{hazard.risk.value}"
                for hazard in agreement.false_claims
            )
        )
    return f"{item.source} @ {item.sim_time_s:.1f} s: " + "; ".join(parts)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" + ("" if number == 1 else "s")


def _share(value: float | None) -> str:
    return "n/a " if value is None else f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
