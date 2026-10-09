from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Callable

from metadrive_starter.cloud_preflight import run_vertex_preflight
from metadrive_starter.config import AppConfig, apply_overrides, load_config
from metadrive_starter.evaluation import load_evaluation_plan, run_evaluation
from metadrive_starter.gates import (
    GateReport,
    GateStatus,
    ScenarioScore,
    Score,
    check_gate_plan,
    load_gate_plan,
    run_gates,
    select_gate_scenarios,
)
from metadrive_starter.episode_replay import (
    replay_episode_artifact,
    replay_race_output,
)
from metadrive_starter.provider_fixture_capture import (
    convert_probe_capture_to_provider_fixtures,
)
from metadrive_starter.race import load_race_plan, run_race
from metadrive_starter.replay import replay_event_log
from metadrive_starter.simulation import RunSummary, run_simulation
from metadrive_starter.submission import (
    SubmissionError,
    apply_reference_overlay,
    apply_submission_overlay,
    load_arbitration_for,
    load_controller,
    load_observation,
    load_observation_for,
    load_reference_controller,
)
from metadrive_starter.vla import ObservationBuilder, build_observation_builder
from metadrive_starter.vla_probe import (
    load_probe_scenarios,
    replay_vla_probe_artifacts,
    run_vla_probe_catalog,
)
from metadrive_starter.vla_runtime import apply_vla_request_cap, build_vla_provider
from metadrive_starter.vertex_capture import (
    RequestBudgetError,
    run_vertex_capture,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MetaDrive starter simulator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a MetaDrive simulation")
    _add_common_options(run_parser)
    _add_submission_option(run_parser)
    run_parser.add_argument("--dry-run", action="store_true", help="validate wiring without launching MetaDrive")

    smoke_parser = subparsers.add_parser("smoke", help="run a quick starter smoke check")
    _add_common_options(smoke_parser)
    _add_submission_option(smoke_parser)
    smoke_parser.add_argument("--dry-run", action="store_true", default=True, help="skip launching MetaDrive")

    config_parser = subparsers.add_parser("config", help="print the effective configuration")
    _add_common_options(config_parser)
    _add_submission_option(config_parser)

    probe_parser = subparsers.add_parser(
        "probe",
        help="capture deterministic VLA inputs and optionally run open-loop HTTP inference",
    )
    probe_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="path to YAML config containing camera and VLA settings",
    )
    probe_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/vla-probe-scenarios.yaml"),
        help="path to the deterministic probe scenario manifest",
    )
    probe_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/vla-probe"),
        help="new artifact root; existing scenario directories are not overwritten",
    )
    probe_parser.add_argument(
        "--scenario",
        dest="scenario_ids",
        action="append",
        help="scenario id to run; repeat to select multiple (default: all)",
    )
    probe_parser.add_argument(
        "--infer",
        action="store_true",
        help="send each captured input to the configured HTTP provider",
    )
    probe_parser.add_argument(
        "--replay-from",
        type=Path,
        help="reuse exact saved prompt and PNG artifacts instead of launching MetaDrive",
    )
    probe_parser.add_argument(
        "--submission",
        type=Path,
        help=(
            "build each scene's observation with this directory's observation.py, "
            "after its agent.yaml; without it DriveBench builds its original one"
        ),
    )
    _add_seed_option(probe_parser)

    replay_parser = subparsers.add_parser(
        "replay",
        help="deterministically replay validation and control decisions from an event log",
    )
    replay_parser.add_argument(
        "--event-log",
        type=Path,
        required=True,
        help="append-only JSONL event log to replay",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="run a scenario-by-seed evaluation manifest and aggregate outcomes",
    )
    evaluate_parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml")
    )
    evaluate_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/evaluation-scenarios.yaml"),
    )
    evaluate_parser.add_argument(
        "--output-dir", type=Path, default=Path("tmp/evaluation")
    )
    _add_submission_option(evaluate_parser)
    evaluate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every run configuration without launching MetaDrive",
    )

    gates_parser = subparsers.add_parser(
        "gates",
        help="run the structural gates against a submission",
    )
    gates_parser.add_argument(
        "--submission",
        type=Path,
        default=Path("submission"),
        help="submission directory to check",
    )
    gates_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/gates-assignment-1.yaml"),
        help="gate scenarios; instructors grade with a hidden manifest",
    )
    gates_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="instructor base configuration the submission's agent.yaml applies to",
    )
    gates_parser.add_argument(
        "--scenario",
        dest="scenario_ids",
        action="append",
        help=(
            "scenario id to drive; repeat to select several (default: all). A partial "
            "run is for iterating: grading drives every scenario"
        ),
    )
    gates_parser.add_argument(
        "--json",
        action="store_true",
        help="print the full report as JSON",
    )

    race_parser = subparsers.add_parser(
        "race",
        help="run one locked public or instructor-only race suite",
    )
    race_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="tunable agent/controller config; race world settings are overridden",
    )
    race_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/race-public-v1.yaml"),
        help="strict race manifest",
    )
    race_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/race"),
        help="new artifact root",
    )
    race_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate locked configs and artifact layout without launching MetaDrive",
    )
    race_parser.add_argument(
        "--render",
        action="store_true",
        help="show simulator windows sequentially; demonstration mode, not scoring mode",
    )
    _add_submission_option(race_parser)

    race_replay_parser = subparsers.add_parser(
        "race-replay",
        help="verify and replay one recorded race episode without rerunning its agent",
    )
    replay_source = race_replay_parser.add_mutually_exclusive_group(required=True)
    replay_source.add_argument(
        "--artifact",
        type=Path,
        help="trusted replay artifact directory produced by a race run",
    )
    replay_source.add_argument(
        "--race-output",
        type=Path,
        help="race output root; replay every scenario artifact in manifest order",
    )
    race_replay_parser.add_argument(
        "--render",
        action="store_true",
        help="show recorded world states in a simulator window",
    )
    race_replay_parser.add_argument(
        "--unpaced",
        action="store_true",
        help="replay as fast as possible instead of recorded simulation speed",
    )
    race_replay_parser.add_argument(
        "--trust-artifact",
        action="store_true",
        help="allow loading local pickle payload; never use with untrusted artifacts",
    )

    cloud_parser = subparsers.add_parser(
        "cloud-check",
        help="check Vertex credentials, schema round trips, and latency",
    )
    cloud_parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    cloud_parser.add_argument("--project", help="GCP project ID; defaults to configured environment variable")
    cloud_parser.add_argument("--location", help="Vertex location override for this check")
    cloud_parser.add_argument("--model", help="Vertex model override for this check")
    cloud_parser.add_argument("--samples", type=_preflight_samples, default=3)

    capture_parser = subparsers.add_parser(
        "vertex-capture",
        help="run the fixed five-request Vertex evidence capture",
    )
    capture_parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml")
    )
    capture_parser.add_argument(
        "--replay-from",
        type=Path,
        required=True,
        help="capture-only probe root containing the exact saved inputs",
    )
    capture_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/vertex-capture-01"),
        help="new evidence root; never overwritten",
    )
    capture_parser.add_argument(
        "--project",
        help="GCP project ID; defaults to configured environment variable or ADC",
    )
    capture_parser.add_argument("--location", help="Vertex location override")
    capture_parser.add_argument("--model", help="Vertex model override")
    capture_parser.add_argument(
        "--max-requests",
        type=_vertex_capture_cap,
        default=5,
        help="hard provider-attempt cap; the fixed plan requires exactly 5",
    )
    capture_parser.add_argument(
        "--execute",
        action="store_true",
        help="confirm execution of the five live, potentially billable requests",
    )

    fixture_parser = subparsers.add_parser(
        "provider-fixtures",
        help="sanitize reviewed probe responses into exact offline fixtures",
    )
    fixture_parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="successful exact-replay probe artifact root",
    )
    fixture_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new fixture directory; never overwritten",
    )
    fixture_parser.add_argument(
        "--fixture-prefix",
        required=True,
        help="safe fixture ID prefix, normally provider/model/date",
    )
    fixture_parser.add_argument(
        "--scenario",
        dest="scenario_ids",
        action="append",
        help="scenario id to convert; repeat to preserve an explicit order",
    )
    fixture_parser.add_argument(
        "--retain-response-ids",
        action="store_true",
        help="retain provider response IDs after policy review; omitted by default",
    )
    fixture_parser.add_argument(
        "--reviewed",
        action="store_true",
        help="confirm raw responses and metadata were reviewed for release",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "replay":
        summary = replay_event_log(args.event_log)
        print(json.dumps(summary.to_dict(), indent=2))
        return 0 if summary.successful else 1

    if args.command == "race-replay":
        if not args.trust_artifact:
            parser.error(
                "race-replay requires --trust-artifact because MetaDrive episode "
                "payloads use pickle"
            )
        if args.race_output is not None:
            summary = replay_race_output(
                args.race_output,
                trusted=True,
                render=args.render,
                paced=not args.unpaced,
            )
        else:
            summary = replay_episode_artifact(
                args.artifact,
                trusted=True,
                render=args.render,
                paced=not args.unpaced,
            )
        print(json.dumps(summary.to_dict(), indent=2))
        return 0 if summary.successful else 1

    if args.command == "provider-fixtures":
        if not args.reviewed:
            parser.error(
                "provider-fixtures requires --reviewed before captured provider "
                "data can be released"
            )
        report = convert_probe_capture_to_provider_fixtures(
            args.source_dir,
            args.output_dir,
            fixture_prefix=args.fixture_prefix,
            scenario_ids=args.scenario_ids,
            retain_response_ids=args.retain_response_ids,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.successful else 1

    if args.command == "cloud-check":
        config = load_config(args.config)
        vertex = replace(
            config.vla.vertex,
            location=args.location or config.vla.vertex.location,
            model_id=args.model or config.vla.vertex.model_id,
        )
        report = run_vertex_preflight(
            vertex,
            project_id=args.project,
            samples=args.samples,
            request_timeout_s=config.vla.request_timeout_s,
            prompt_policy=config.vla.prompt_policy,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.successful else 1

    if args.command == "vertex-capture":
        if not args.execute:
            parser.error(
                "vertex-capture requires --execute because it makes five live, "
                "potentially billable requests"
            )
        config = load_config(args.config)
        vertex = replace(
            config.vla.vertex,
            location=args.location or config.vla.vertex.location,
            model_id=args.model or config.vla.vertex.model_id,
        )
        try:
            report = run_vertex_capture(
                config,
                args.replay_from,
                args.output_dir,
                project_id=args.project,
                vertex_settings=vertex,
                maximum_requests=args.max_requests,
            )
        except RequestBudgetError as exc:
            parser.error(str(exc))
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.successful else 1

    if args.command in {"evaluate", "race"}:
        try:
            config, runner = _prepare_runs(load_config(args.config), args.submission)
        except SubmissionError as exc:
            _report_submission_error(exc)
            return 1
        if args.command == "evaluate":
            report = run_evaluation(
                config,
                load_evaluation_plan(args.manifest),
                args.output_dir,
                dry_run=args.dry_run,
                runner=runner,
            )
        else:
            report = run_race(
                config,
                load_race_plan(args.manifest),
                args.output_dir,
                dry_run=args.dry_run,
                render=args.render,
                runner=runner,
            )
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.successful else 1

    if args.command == "gates":
        plan = load_gate_plan(args.manifest)
        base_config = load_config(args.config)
        try:
            check_gate_plan(plan, base_config)
            select_gate_scenarios(plan, args.scenario_ids)
        except ValueError as exc:
            parser.error(str(exc))
        report = run_gates(
            args.submission,
            plan,
            base_config=base_config,
            scenario_ids=args.scenario_ids,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            _print_gate_report(report)
        return 0 if report.passed else 1

    config = _apply_seed(parser, load_config(args.config), args.seed)
    if args.command == "probe":
        observation_builder: ObservationBuilder | None = None
        if args.submission is not None:
            if args.replay_from is not None:
                parser.error(
                    "probe --replay-from sends the saved observations again; "
                    "--submission applies only when capturing"
                )
            try:
                config = apply_submission_overlay(config, args.submission)
                observation_builder = build_observation_builder(
                    config.observation,
                    factory=load_observation(args.submission),
                )
            except SubmissionError as exc:
                _report_submission_error(exc)
                return 1
        provider = build_vla_provider(config.vla) if args.infer else None
        if provider is not None:
            provider, _ = apply_vla_request_cap(config.vla, provider)
        if args.replay_from is not None:
            if provider is None:
                parser.error("probe --replay-from requires --infer")
            summary = replay_vla_probe_artifacts(
                config,
                args.replay_from,
                args.output_dir,
                scenario_ids=args.scenario_ids,
                provider=provider,
            )
        else:
            scenarios = load_probe_scenarios(args.manifest)
            summary = run_vla_probe_catalog(
                config,
                scenarios,
                args.output_dir,
                scenario_ids=args.scenario_ids,
                infer=args.infer,
                provider=provider,
                observation_builder=observation_builder,
            )
        print(json.dumps(summary.to_dict(), indent=2))
        return 0 if summary.successful else 1

    try:
        if args.submission is not None:
            config = apply_submission_overlay(config, args.submission)
        else:
            config = apply_reference_overlay(config)
        config = apply_overrides(
            config,
            headless=_headless_override(args),
            manual_control=args.manual_control,
            realtime=True if args.realtime_factor is not None else args.realtime,
            realtime_factor=args.realtime_factor,
            out_of_road_done=False if args.continue_off_road else None,
            crash_vehicle_done=False if args.continue_after_collision else None,
            crash_object_done=False if args.continue_after_collision else None,
            steps=args.steps,
            traffic_density=args.traffic_density,
            obstacle_probability=args.obstacle_probability,
            map_name=args.map,
            event_log_path=args.event_log,
            display_speed_unit=args.speed_unit,
            headway_speed_cap_mode=args.headway_speed_cap,
            action_speed_policy_mode=args.action_speed_policy,
        )
        if args.command == "config":
            print(json.dumps(config.to_dict(), indent=2))
            return 0
        controller_factory = (
            None if args.submission is None else load_controller(args.submission)
        )
        observation_factory = (
            None
            if args.submission is None
            else load_observation_for(args.submission, config)
        )
        arbitration_factory = (
            None
            if args.submission is None
            else load_arbitration_for(args.submission, config)
        )
        summary = run_simulation(
            config,
            dry_run=args.dry_run,
            controller_factory=controller_factory,
            observation_factory=observation_factory,
            arbitration_factory=arbitration_factory,
        )
    except SubmissionError as exc:
        _report_submission_error(exc)
        return 1
    print(json.dumps(asdict(summary), indent=2))
    return 0


def _print_gate_report(report: GateReport) -> None:
    print(f"Structural gates {report.plan_id} for {report.submission}")
    if report.partial:
        print(
            f"  PARTIAL RUN: {', '.join(report.scenarios_run)} only; "
            "a full run drives every scenario, as grading does"
        )
    for result in report.results:
        line = f"  {result.status.value:<8} {result.gate}"
        print(f"{line}: {result.detail}" if result.detail else line)
    failed = sum(result.status is GateStatus.FAILED for result in report.results)
    print("PASSED" if report.passed else f"FAILED: {failed} of {len(report.results)} gates")
    if report.score is not None:
        _print_score(report.score)
    elif report.no_score is not None:
        print()
        print(f"No score: {report.no_score}")


def _print_score(score: Score) -> None:
    print()
    print(f"Score {score.total:.1f} of 100, a quality measure that never decides a gate")
    passed = sum(check.passed for check in score.checks)
    print(
        f"  implementation checks {score.implementation:.1f}: "
        f"{passed} of {len(score.checks)} passed"
    )
    for check in score.checks:
        print(f"    {'passed' if check.passed else 'failed':<8} {check.check}: {check.detail}")
    print(f"  driving {score.driving:.1f}")
    for scenario in score.scenarios:
        print(
            f"    {100.0 * scenario.credit:5.1f}  {scenario.scenario_id}: "
            f"{_describe_drive(scenario)}"
        )


def _describe_drive(scenario: ScenarioScore) -> str:
    if scenario.problem is not None:
        if scenario.steps is None:
            return scenario.problem
        return f"{scenario.problem} after {scenario.steps} steps"
    if scenario.steps is None:
        return "the run stopped"
    parts = [f"{scenario.steps} steps against par {scenario.par_steps}"]
    if scenario.speed_error_mps is not None:
        parts.append(f"speed error {scenario.speed_error_mps:.2f} m/s")
    if scenario.lateral_error_m is not None:
        parts.append(f"lateral error {scenario.lateral_error_m:.2f} m")
    return ", ".join(parts)


def _add_submission_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--submission",
        type=Path,
        help=(
            "directory holding a team's submission files; without it the instructor "
            "reference drives, and a student release, which has none, stops"
        ),
    )


def _prepare_runs(
    config: AppConfig,
    submission: Path | None,
) -> tuple[AppConfig, Callable[..., RunSummary]]:
    """Apply a submission's agent.yaml, and load the controller every run drives
    and, when the model is asked, the observation builder and the arbiter.

    Without a submission the instructor reference drives, as in ``run``. Either
    fails here, before any run starts or writes output.
    """
    observation_factory = None
    arbitration_factory = None
    if submission is None:
        factory = load_reference_controller()
        config = apply_reference_overlay(config)
    else:
        config = apply_submission_overlay(config, submission)
        factory = load_controller(submission)
        observation_factory = load_observation_for(submission, config)
        arbitration_factory = load_arbitration_for(submission, config)
    return config, partial(
        run_simulation,
        controller_factory=factory,
        observation_factory=observation_factory,
        arbitration_factory=arbitration_factory,
    )


def _add_seed_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed",
        type=_seed,
        help=(
            "sampling seed sent with every HTTP model request; rerun under a new one "
            "each time, as grading does"
        ),
    )


def _apply_seed(
    parser: argparse.ArgumentParser,
    config: AppConfig,
    seed: int | None,
) -> AppConfig:
    """Seed every HTTP model request, as grading seeds each rerun.

    The seed stays instructor-owned in agent.yaml; this sets it for one run only.
    """
    if seed is None:
        return config
    if config.vla.provider != "http":
        parser.error(
            "--seed seeds the HTTP model provider; this configuration uses the "
            f"{config.vla.provider} provider"
        )
    return apply_overrides(config, http_seed=seed)


def _report_submission_error(error: SubmissionError) -> None:
    if error.__cause__ is not None:
        traceback.print_exception(error.__cause__, file=sys.stderr)
    print(f"submission error: {error}", file=sys.stderr)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="path to YAML config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="disable simulator rendering")
    mode.add_argument("--render", action="store_true", help="enable simulator rendering")
    control = parser.add_mutually_exclusive_group()
    control.add_argument(
        "--manual",
        dest="manual_control",
        action="store_true",
        default=None,
        help="use keyboard/manual control",
    )
    control.add_argument(
        "--autopilot",
        dest="manual_control",
        action="store_false",
        help="use waypoint planning and PID control",
    )
    pacing = parser.add_mutually_exclusive_group()
    pacing.add_argument(
        "--realtime",
        dest="realtime",
        action="store_true",
        default=None,
        help="pace simulation time to monotonic wall time",
    )
    pacing.add_argument(
        "--unpaced",
        dest="realtime",
        action="store_false",
        help="run as fast as the simulator permits",
    )
    pacing.add_argument(
        "--realtime-factor",
        type=_realtime_factor,
        help="pace at this fraction of real time; 0.5 gives model twice the wall time",
    )
    parser.add_argument(
        "--continue-off-road",
        action="store_true",
        help="keep the episode running when the ego vehicle leaves the road",
    )
    parser.add_argument(
        "--continue-after-collision",
        "--continue-after-crash",
        dest="continue_after_collision",
        action="store_true",
        help="keep the episode running after vehicle or obstacle collisions",
    )
    parser.add_argument("--steps", type=int, help="maximum simulation steps")
    parser.add_argument("--traffic-density", type=float, help="traffic density override")
    parser.add_argument(
        "--obstacle-probability",
        "--accident-probability",
        dest="obstacle_probability",
        type=_probability,
        help="probability per eligible road block of generating a construction or accident scene",
    )
    parser.add_argument("--map", help="MetaDrive map override, such as S, C, X, O, or R")
    parser.add_argument(
        "--speed-unit",
        choices=("mps", "kph"),
        help="rendered speed display unit; internal speed remains m/s",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        help="append structured runtime events to this JSONL file",
    )
    parser.add_argument(
        "--headway-speed-cap",
        dest="headway_speed_cap",
        choices=("off", "shadow", "enforce"),
        help="override proactive VLA headway speed filtering",
    )
    parser.add_argument(
        "--action-speed-policy",
        choices=("off", "shadow", "enforce"),
        help="override deterministic high-level-action speed mapping",
    )
    _add_seed_option(parser)


def _headless_override(args: argparse.Namespace) -> bool | None:
    if args.headless:
        return True
    if args.render:
        return False
    return None


def _probability(value: str) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise argparse.ArgumentTypeError("probability must be between 0 and 1")
    return probability


def _preflight_samples(value: str) -> int:
    samples = int(value)
    if not 1 <= samples <= 20:
        raise argparse.ArgumentTypeError("samples must be between 1 and 20")
    return samples


def _vertex_capture_cap(value: str) -> int:
    cap = int(value)
    if not 1 <= cap <= 5:
        raise argparse.ArgumentTypeError("Vertex capture cap must be between 1 and 5")
    return cap


def _seed(value: str) -> int:
    seed = int(value)
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer")
    return seed


def _realtime_factor(value: str) -> float:
    factor = float(value)
    if not 0.0 < factor <= 1.0:
        raise argparse.ArgumentTypeError("realtime factor must be between 0 and 1")
    return factor


if __name__ == "__main__":
    raise SystemExit(main())
