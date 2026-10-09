from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter.cli import _print_gate_report
from metadrive_starter.config import AppConfig, EventLogSettings, config_from_dict, load_config
from metadrive_starter.faults import FaultKind, FaultSpec
from metadrive_starter.gates import (
    FAULTS_SURVIVED,
    GatePlan,
    GateReport,
    GateResult,
    GateScenario,
    GateStatus,
    check_gate_plan,
    drive_route,
    load_gate_plan,
    run_gates,
    select_gate_scenarios,
    write_release_manifest,
)
from metadrive_starter.submission import load_controller
from metadrive_starter.vla import OUTPUT_CONTRACT, OutputContractWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Proportional control on the configured gains, saturated by the controller.
WELL_BEHAVED = """
class Controller:
    def __init__(self, settings):
        self.speed_kp = settings.speed_pid.kp
        self.heading_kp = settings.steering_pid.kp
        self.lateral_kp = settings.lateral_pid.kp

    def update(self, tick):
        throttle = self.speed_kp * (tick.target_speed_mps - tick.speed_mps)
        steering = (
            self.heading_kp * tick.heading_error_rad
            + self.lateral_kp * tick.lateral_error_m
        )
        return (max(-1.0, min(1.0, steering)), max(-1.0, min(1.0, throttle)))

    def reset_speed_control(self):
        pass
"""


def _submission(directory: Path, controller_source: str) -> Path:
    directory.mkdir(exist_ok=True)
    (directory / "controller.py").write_text(controller_source)
    return directory


def _straight_plan() -> GatePlan:
    return GatePlan(
        "test-gates",
        "public",
        (
            GateScenario(
                scenario_id="straight", map="S", seed=0, horizon=600, par_steps=134
            ),
        ),
    )


def _statuses(report: GateReport) -> dict[str, GateStatus]:
    return {result.gate: result.status for result in report.results}


def test_public_assignment_1_gates_drive_the_canonical_maps_without_traffic() -> None:
    plan = load_gate_plan(PROJECT_ROOT / "configs" / "gates-assignment-1.yaml")

    assert plan.visibility == "public"
    assert [scenario.map for scenario in plan.scenarios] == ["S", "C", "X", "O", "XTOC"]
    assert all(scenario.traffic_density == 0.0 for scenario in plan.scenarios)
    # Par is the reference controller's time on each route.
    assert [scenario.par_steps for scenario in plan.scenarios] == [134, 257, 172, 228, 503]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("version: 2\nid: g\nvisibility: public\nscenarios: []\n", "version must be 1"),
        (
            "version: 1\nid: g\nvisibility: public\nscenarios: []\nfaults: []\n",
            "unknown fields: faults",
        ),
        ("version: 1\nid: g\nvisibility: secret\nscenarios: []\n", "'public' or 'hidden'"),
        ("version: 1\nid: g\nvisibility: public\nscenarios: []\n", "must not be empty"),
        (
            "version: 1\nid: g\nvisibility: public\nscenarios:\n"
            "  - {id: a, map: S, seed: 0, horizon: 0, par_steps: 1}\n",
            "horizon must be a positive integer",
        ),
        (
            "version: 1\nid: g\nvisibility: public\nscenarios:\n"
            "  - {id: a, map: S, seed: 0, horizon: 600}\n",
            "gate scenario 'a' needs par_steps, unless the manifest declares no_score",
        ),
        (
            "version: 1\nid: g\nvisibility: public\nscenarios:\n"
            "  - {id: a, map: S, seed: 0, horizon: 600, par_steps: 601}\n",
            "par_steps must not exceed the horizon",
        ),
    ],
    ids=[
        "version",
        "unknown-field",
        "visibility",
        "no-scenarios",
        "horizon",
        "no-par",
        "par-beyond-horizon",
    ],
)
def test_malformed_gate_manifest_is_rejected(
    tmp_path: Path,
    manifest: str,
    message: str,
) -> None:
    path = tmp_path / "gates.yaml"
    path.write_text(manifest)

    with pytest.raises(ValueError, match=message):
        load_gate_plan(path)


def test_well_behaved_submission_passes_every_gate(tmp_path: Path) -> None:
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)

    report = run_gates(
        submission,
        _straight_plan(),
        base_config=config_from_dict({}),
        release_root=tmp_path,
    )

    assert _statuses(report) == {
        "submission loads": GateStatus.PASSED,
        "bounded control outputs": GateStatus.PASSED,
        "no run stops on an error": GateStatus.PASSED,
        "nothing outside submission/ was modified": GateStatus.NOT_RUN,
    }
    assert report.passed
    assert report.scenarios_run == ("straight",)
    assert report.partial is False


def test_a_team_file_no_seam_names_is_recorded_and_never_gated(
    tmp_path: Path,
) -> None:
    # Teams may keep notes beside their code; no gate may need or refuse them.
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)
    (submission / "notes.md").write_text("# Our controller\n")

    report = run_gates(
        submission,
        _straight_plan(),
        base_config=config_from_dict({}),
        release_root=tmp_path,
    )

    assert report.passed
    assert set(report.submission_sha256) == {"notes.md", "controller.py"}


def _two_route_plan() -> GatePlan:
    return GatePlan(
        "test-gates",
        "public",
        (
            GateScenario(scenario_id="straight", map="S", seed=0, horizon=600, par_steps=134),
            GateScenario(scenario_id="curve", map="C", seed=0, horizon=900, par_steps=257),
        ),
    )


def test_selected_scenarios_drive_alone_and_mark_the_run_partial(tmp_path: Path) -> None:
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)

    report = run_gates(
        submission,
        _two_route_plan(),
        base_config=config_from_dict({}),
        release_root=tmp_path,
        scenario_ids=["straight"],
    )

    assert report.passed
    assert report.partial is True
    assert report.scenarios_run == ("straight",)
    assert report.to_dict()["partial"] is True
    assert report.score is not None
    assert [scenario.scenario_id for scenario in report.score.scenarios] == ["straight"]


def test_every_named_scenario_is_a_full_run_in_manifest_order() -> None:
    plan = _two_route_plan()

    assert select_gate_scenarios(plan, ["curve", "straight"]) == plan.scenarios
    assert select_gate_scenarios(plan, None) == plan.scenarios


@pytest.mark.parametrize(
    ("scenario_ids", "message"),
    [
        (["roundabout"], "unknown gate scenario id: roundabout; test-gates has straight, curve"),
        ([], "at least one gate scenario id is required"),
    ],
)
def test_gate_scenario_selection_names_what_is_available(
    scenario_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_gate_scenarios(_two_route_plan(), scenario_ids)


UNSATURATED = WELL_BEHAVED.replace(
    "return (max(-1.0, min(1.0, steering)), max(-1.0, min(1.0, throttle)))",
    "return (max(-1.0, min(1.0, steering)), throttle)",
)
STATIONARY = WELL_BEHAVED.replace(
    "throttle = self.speed_kp * (tick.target_speed_mps - tick.speed_mps)",
    "throttle = 0.0",
)
LOADS = "submission loads"
BOUNDED = "bounded control outputs"
ROUTE = "route completes"
RUNS = "no run stops on an error"


@pytest.mark.parametrize(
    ("controller_source", "expected"),
    [
        (
            "class Controller(:\n",
            {LOADS: GateStatus.FAILED, BOUNDED: GateStatus.NOT_RUN, RUNS: GateStatus.NOT_RUN},
        ),
        (
            UNSATURATED,
            {LOADS: GateStatus.PASSED, BOUNDED: GateStatus.FAILED, RUNS: GateStatus.PASSED},
        ),
    ],
    ids=["does-not-load", "unsaturated-throttle"],
)
def test_broken_submission_fails_only_the_relevant_gate(
    tmp_path: Path,
    controller_source: str,
    expected: dict[str, GateStatus],
) -> None:
    submission = _submission(tmp_path / "submission", controller_source)

    report = run_gates(
        submission,
        _straight_plan(),
        base_config=config_from_dict({}),
        release_root=tmp_path,
    )

    statuses = _statuses(report)
    assert {gate: statuses[gate] for gate in expected} == expected
    assert not report.passed


def test_non_finite_output_fails_bounded_outputs_and_stops_the_route(tmp_path: Path) -> None:
    source = UNSATURATED.replace(
        "throttle = self.speed_kp * (tick.target_speed_mps - tick.speed_mps)",
        "throttle = float('nan')",
    )
    submission = _submission(tmp_path / "submission", source)

    report = run_gates(
        submission,
        _straight_plan(),
        base_config=config_from_dict({}),
        release_root=tmp_path,
    )

    results = {result.gate: result for result in report.results}
    assert results[BOUNDED].status is GateStatus.FAILED
    assert "throttle_brake must be a finite number" in results[BOUNDED].detail
    assert results[RUNS].status is GateStatus.FAILED
    assert "straight: the run stopped" in results[RUNS].detail


TREE = "nothing outside submission/ was modified"


def _release(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "src" / "foundation.py").write_text("VALUE = 1\n")
    (root / "configs").mkdir()
    (root / "configs" / "default.yaml").write_text("simulator: {}\n")
    _submission(root / "submission", "class Controller(:\n")
    write_release_manifest(root)
    return root


def _tree_gate(root: Path) -> tuple[GateStatus, str]:
    report = run_gates(
        root / "submission",
        _straight_plan(),
        base_config=config_from_dict({}),
        release_root=root,
    )
    [result] = [result for result in report.results if result.gate == TREE]
    return result.status, result.detail


def test_release_changed_only_inside_submission_passes_the_tree_gate(tmp_path: Path) -> None:
    root = _release(tmp_path / "release")
    (root / "submission" / "agent.yaml").write_text("controller: {}\n")
    (root / "tmp").mkdir()
    (root / "tmp" / "events.jsonl").write_text("{}\n")

    assert _tree_gate(root) == (GateStatus.PASSED, "2 shipped files unchanged")


def test_release_edited_outside_submission_fails_the_tree_gate(tmp_path: Path) -> None:
    root = _release(tmp_path / "release")
    (root / "src" / "foundation.py").write_text("VALUE = 2\n")
    (root / "configs" / "default.yaml").unlink()

    assert _tree_gate(root) == (
        GateStatus.FAILED,
        "configs/default.yaml deleted; src/foundation.py modified",
    )


def test_proportional_only_submission_drives_for_half_the_score(tmp_path: Path) -> None:
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)

    score = run_gates(
        submission,
        _straight_plan(),
        base_config=config_from_dict({}),
        release_root=tmp_path,
    ).score

    # On the starting gains it arrives 11 steps behind par and about 0.54 m/s off
    # its target, but no implementation check finds an integral or derivative term.
    assert score is not None
    assert [check.passed for check in score.checks] == [False] * 6
    assert score.implementation == 0.0
    [straight] = score.scenarios
    assert (straight.steps, straight.par_steps) == (145, 134)
    assert straight.credit == pytest.approx(0.535, abs=0.001)
    assert score.driving == pytest.approx(53.5, abs=0.1)
    assert score.total == pytest.approx(26.8, abs=0.1)


def test_drive_route_scores_a_controller_as_the_gate_report_does(tmp_path: Path) -> None:
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)
    plan = _straight_plan()
    config = config_from_dict({})
    report = run_gates(submission, plan, base_config=config, release_root=tmp_path)

    [scenario] = plan.scenarios
    route = drive_route(load_controller(submission), scenario, config=config)

    assert report.score is not None
    assert route == report.score.scenarios[0]


SLOW = WELL_BEHAVED.replace(
    "self.speed_kp = settings.speed_pid.kp",
    "self.speed_kp = 0.03",
)


def test_slow_submission_earns_driving_credit_only_for_holding_its_lane(
    tmp_path: Path,
) -> None:
    submission = _submission(tmp_path / "submission", SLOW)

    score = run_gates(
        submission,
        _straight_plan(),
        base_config=config_from_dict({}),
        release_root=tmp_path,
    ).score

    # Past 1.3 times par and well over 0.6 m/s below its target once launched,
    # it keeps only the lateral share of the driving credit.
    assert score is not None
    [straight] = score.scenarios
    assert straight.steps > 1.5 * straight.par_steps
    assert straight.speed_error_mps > 1.5
    assert straight.lateral_error_m == pytest.approx(0.0, abs=0.01)
    assert score.driving == pytest.approx(20.0)


def test_route_that_never_completes_costs_its_driving_credit_and_no_gate(
    tmp_path: Path,
) -> None:
    # Where routes are scored, an incomplete route is how the submission drove:
    # it earns nothing, and the implementation checks are still marked.
    submission = _submission(tmp_path / "submission", STATIONARY)

    report = run_gates(
        submission,
        _straight_plan(),
        base_config=config_from_dict({}),
        release_root=tmp_path,
    )

    results = {result.gate: result for result in report.results}
    assert report.passed
    assert results[RUNS].status is GateStatus.PASSED
    assert "straight: did not arrive within 600 steps" in results[RUNS].detail
    assert report.score is not None
    [straight] = report.score.scenarios
    assert straight.credit == 0.0
    assert straight.problem == "did not arrive within 600 steps"
    assert report.score.driving == 0.0


def test_route_that_never_completes_fails_the_gate_where_routes_are_not_scored(
    tmp_path: Path,
) -> None:
    submission = _submission(tmp_path / "submission", STATIONARY)
    route = GateScenario(scenario_id="straight", map="S", seed=0, horizon=600)
    plan = GatePlan("unscored", "public", (route,), no_score="Marked from hidden runs.")

    report = run_gates(
        submission, plan, base_config=config_from_dict({}), release_root=tmp_path
    )

    results = {result.gate: result for result in report.results}
    assert not report.passed
    assert results[ROUTE].status is GateStatus.FAILED
    assert results[ROUTE].detail == "straight: did not arrive within 600 steps"
    assert RUNS not in results


def test_submission_that_does_not_load_is_not_scored(tmp_path: Path) -> None:
    submission = _submission(tmp_path / "submission", "class Controller(:\n")

    report = run_gates(
        submission,
        _straight_plan(),
        base_config=config_from_dict({}),
        release_root=tmp_path,
    )

    assert report.score is None
    assert report.to_dict()["score"] is None


def test_running_a_shipped_notebook_passes_the_tree_gate(tmp_path: Path) -> None:
    root = tmp_path / "release"
    notebook = root / "course" / "assignment-1" / "lab.ipynb"
    notebook.parent.mkdir(parents=True)
    notebook.write_text('{"cells": []}\n')
    _release(root)
    # Running a notebook rewrites it with outputs, next to a checkpoint copy.
    notebook.write_text('{"cells": [{"outputs": ["plot"]}]}\n')
    checkpoints = notebook.parent / ".ipynb_checkpoints"
    checkpoints.mkdir()
    (checkpoints / "lab-checkpoint.ipynb").write_text('{"cells": []}\n')

    assert _tree_gate(root) == (GateStatus.PASSED, "2 shipped files unchanged")


VALID = "valid observations emitted"
UNBLOCKED = "control loop never blocked"
OBSERVATION = """
from metadrive_starter.vla.observation import Observation

class ObservationBuilder:
    def __init__(self, settings):
        pass

    def build(self, request):
        return Observation(frame=request.frame, prompt="\\n".join(request.contract.lines))
"""
ARBITRATION = """
class Arbiter:
    def __init__(self, settings):
        pass

    def begin_episode(self):
        pass

    def arbitrate(self, request):
        return request.assessment

    def review(self, request):
        return True
"""


def _vla_base(*, request_timeout_s: float = 1.0) -> AppConfig:
    """The fixture-provider demo as a gate base configuration: the model is asked."""
    config = load_config(PROJECT_ROOT / "configs" / "demo-vla-fixture.yaml")
    config.event_log = EventLogSettings()
    config.vla = replace(
        config.vla,
        request_timeout_s=request_timeout_s,
        fixture=replace(
            config.vla.fixture,
            path=str(PROJECT_ROOT / config.vla.fixture.path),
        ),
    )
    return config


def _short_plan() -> GatePlan:
    # Long enough for several model requests; the route itself is not judged here.
    return GatePlan(
        "test-observation-gates",
        "public",
        (GateScenario(scenario_id="straight", map="S", seed=0, horizon=30, par_steps=30),),
    )


def _observation_gates(
    tmp_path: Path,
    observation_source: str,
    *,
    request_timeout_s: float = 1.0,
    arbitration_source: str = ARBITRATION,
) -> dict[str, GateResult]:
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)
    (submission / "observation.py").write_text(observation_source)
    (submission / "arbitration.py").write_text(arbitration_source)
    report = run_gates(
        submission,
        _short_plan(),
        base_config=_vla_base(request_timeout_s=request_timeout_s),
        release_root=tmp_path,
    )
    return {result.gate: result for result in report.results}


def test_observation_that_carries_the_contract_passes_the_observation_gates(
    tmp_path: Path,
) -> None:
    results = _observation_gates(tmp_path, OBSERVATION)

    assert results[LOADS].status is GateStatus.PASSED
    assert results[VALID].status is GateStatus.PASSED
    assert results[VALID].detail.endswith(
        "observations built, each with a valid frame and the output contract"
    )
    assert results[UNBLOCKED].status is GateStatus.PASSED
    assert "within the 1 s request timeout" in results[UNBLOCKED].detail


@pytest.mark.needs("fixtures")
def test_observation_without_the_contract_passes_the_gates_with_a_notice(
    tmp_path: Path,
) -> None:
    source = OBSERVATION.replace(
        '"\\n".join(request.contract.lines)',
        '"\\n".join(request.contract.lines[:3])',
    )

    with pytest.warns(OutputContractWarning):
        results = _observation_gates(tmp_path, source)

    assert results[VALID].status is GateStatus.PASSED
    detail = results[VALID].detail
    assert "each with a valid frame;" in detail
    assert "of them the prompt lacks output-contract lines" in detail
    assert repr(OUTPUT_CONTRACT.format) in detail
    assert results[UNBLOCKED].status is GateStatus.PASSED


@pytest.mark.needs("fixtures")
def test_observation_that_raises_fails_the_valid_observation_gate(tmp_path: Path) -> None:
    source = OBSERVATION.replace(
        "        return Observation(",
        "        raise RuntimeError('no font')\n        return Observation(",
    )

    results = _observation_gates(tmp_path, source)

    assert results[VALID].status is GateStatus.FAILED
    assert "build(request) raised RuntimeError: no font" in results[VALID].detail


@pytest.mark.needs("fixtures")
def test_observation_that_outlasts_the_request_timeout_is_abandoned_and_fails(
    tmp_path: Path,
) -> None:
    started = tmp_path / "build-started"
    returned = tmp_path / "build-returned"
    source = OBSERVATION.replace(
        "    def build(self, request):\n",
        "    def build(self, request):\n"
        "        import pathlib, time\n"
        f"        pathlib.Path({str(started)!r}).touch()\n"
        "        time.sleep(600.0)\n"
        f"        pathlib.Path({str(returned)!r}).touch()\n",
    )

    results = _observation_gates(tmp_path, source, request_timeout_s=0.2)

    # The gates returned while the first build was still running: it was
    # abandoned, not awaited, and no later build was started.
    assert started.exists()
    assert not returned.exists()
    assert results[UNBLOCKED].status is GateStatus.FAILED
    assert results[UNBLOCKED].detail == (
        "straight: build(request) had not returned after 0.2 s, the request timeout"
    )
    # The vehicle drove on without the model rather than stopping the run.
    assert "the run stopped" not in results[RUNS].detail


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        ("raise ValueError('bad settings')", "raised ValueError: bad settings"),
        (
            "import time; time.sleep(30.0)",
            "ObservationBuilder(settings) had not returned after 0.2 s, the request timeout",
        ),
    ],
    ids=["raises", "hangs"],
)
def test_observation_builder_that_cannot_be_built_fails_to_load(
    tmp_path: Path,
    constructor: str,
    message: str,
) -> None:
    source = OBSERVATION.replace("        pass\n", f"        {constructor}\n", 1)

    results = _observation_gates(tmp_path, source, request_timeout_s=0.2)

    assert results[LOADS].status is GateStatus.FAILED
    assert "observation.py" in results[LOADS].detail
    assert message in results[LOADS].detail
    assert results[VALID].status is GateStatus.NOT_RUN
    assert results[UNBLOCKED].status is GateStatus.NOT_RUN


def test_submission_without_observation_file_fails_to_load_when_the_model_is_asked(
    tmp_path: Path,
) -> None:
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)

    report = run_gates(
        submission,
        _short_plan(),
        base_config=_vla_base(),
        release_root=tmp_path,
    )

    results = {result.gate: result for result in report.results}
    assert results[LOADS].status is GateStatus.FAILED
    assert "observation.py not found" in results[LOADS].detail


MONOTONE = "monotone restriction preserved"


def _arbitration_gates(tmp_path: Path, arbitration_source: str) -> dict[str, GateResult]:
    return _observation_gates(tmp_path, OBSERVATION, arbitration_source=arbitration_source)


def _arbiter(answer: str, keep: str = "True") -> str:
    """An arbiter whose arbitrate returns ``answer`` and whose review returns
    ``keep``, both expressions in ``request``."""
    return (
        "from dataclasses import replace\n"
        "from metadrive_starter.vla import ArbitrationError, HighLevelAction\n"
        "def refuse(message):\n"
        "    raise ArbitrationError(message)\n"
        + ARBITRATION.replace("return request.assessment", f"return {answer}").replace(
            "return True", f"return {keep}"
        )
    )


@pytest.mark.parametrize(
    ("answer", "keep"),
    [
        ("request.assessment", "True"),
        ("replace(request.assessment, proposed_target_speed_mps=4.0, confidence=0.6)", "True"),
        ("replace(request.assessment, proposed_action=HighLevelAction.REQUEST_FALLBACK)", "True"),
        ("request.assessment", "False"),
    ],
    ids=["endorses", "restricts", "declines", "ends-every-command"],
)
def test_arbiter_within_the_proposal_passes_the_model_gates(
    tmp_path: Path,
    answer: str,
    keep: str,
) -> None:
    results = _arbitration_gates(tmp_path, _arbiter(answer, keep))

    assert results[LOADS].status is GateStatus.PASSED
    assert results[MONOTONE].status is GateStatus.PASSED
    assert results[MONOTONE].detail.endswith(
        "ticks reviewed, none granting more authority than the model proposed"
    )
    assert results[UNBLOCKED].status is GateStatus.PASSED
    assert "on the control loop within the 0.1 s control step" in results[UNBLOCKED].detail


@pytest.mark.parametrize(
    ("answer", "keep", "problem"),
    [
        (
            "replace(request.assessment, proposed_target_speed_mps=9.0)",
            "True",
            "proposed_target_speed_mps raised from 8 to 9 m/s",
        ),
        (
            "replace(request.assessment, proposed_action=HighLevelAction.STOP)",
            "True",
            "proposed_action changed from KEEP_LANE to STOP; "
            "only REQUEST_FALLBACK may replace it",
        ),
        ("None", "True", "arbitrate(request) must return a VLAAssessment; got a NoneType"),
        ("1 / 0", "True", "arbitrate(request) raised ZeroDivisionError: division by zero"),
        # The gates' own refusal is an ArbitrationError too; a team's never passes for one.
        ("refuse('no')", "True", "arbitrate(request) raised ArbitrationError: no"),
        ("request.assessment", "None", "review(request) must return True or False; got None"),
    ],
    ids=[
        "faster",
        "another-action",
        "not-an-assessment",
        "raises",
        "raises-arbitration-error",
        "review-without-answer",
    ],
)
@pytest.mark.needs("fixtures")
def test_arbiter_beyond_the_proposal_fails_the_monotone_gate_and_the_run_drives_on(
    tmp_path: Path,
    answer: str,
    keep: str,
    problem: str,
) -> None:
    results = _arbitration_gates(tmp_path, _arbiter(answer, keep))

    assert results[MONOTONE].status is GateStatus.FAILED
    assert results[MONOTONE].detail.startswith("straight: ")
    assert f"could not stand; the first: {problem}" in results[MONOTONE].detail
    # DriveBench declined each such answer itself rather than stopping the run.
    assert "the run stopped" not in results[RUNS].detail
    assert results[UNBLOCKED].status is GateStatus.PASSED


@pytest.mark.needs("fixtures")
def test_arbiter_that_outlasts_the_control_step_is_abandoned_and_fails(
    tmp_path: Path,
) -> None:
    started = tmp_path / "arbitrate-started"
    source = ARBITRATION.replace(
        "    def arbitrate(self, request):\n",
        "    def arbitrate(self, request):\n"
        "        import pathlib, time\n"
        f"        pathlib.Path({str(started)!r}).touch()\n"
        "        time.sleep(600.0)\n",
    )

    results = _arbitration_gates(tmp_path, source)

    assert started.exists()
    assert results[UNBLOCKED].status is GateStatus.FAILED
    assert results[UNBLOCKED].detail == (
        "straight: arbitrate(request) had not returned after 0.1 s, the control step"
    )
    # Every later assessment was declined without calling the arbiter again.
    assert results[MONOTONE].status is GateStatus.PASSED
    assert "the run stopped" not in results[RUNS].detail


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        ("raise ValueError('bad settings')", "raised ValueError: bad settings"),
        (
            "import time; time.sleep(30.0)",
            "Arbiter(settings) had not returned after 0.2 s, the request timeout",
        ),
    ],
    ids=["raises", "hangs"],
)
def test_arbiter_that_cannot_be_built_fails_to_load(
    tmp_path: Path,
    constructor: str,
    message: str,
) -> None:
    source = ARBITRATION.replace("        pass\n", f"        {constructor}\n", 1)

    results = _observation_gates(
        tmp_path, OBSERVATION, arbitration_source=source, request_timeout_s=0.2
    )

    assert results[LOADS].status is GateStatus.FAILED
    assert "arbitration.py" in results[LOADS].detail
    assert message in results[LOADS].detail
    assert results[MONOTONE].status is GateStatus.NOT_RUN


def test_submission_without_arbitration_file_fails_to_load_when_the_model_is_asked(
    tmp_path: Path,
) -> None:
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)
    (submission / "observation.py").write_text(OBSERVATION)

    report = run_gates(
        submission,
        _short_plan(),
        base_config=_vla_base(),
        release_root=tmp_path,
    )

    results = {result.gate: result for result in report.results}
    assert results[LOADS].status is GateStatus.FAILED
    assert "arbitration.py not found" in results[LOADS].detail
    assert results[MONOTONE].status is GateStatus.NOT_RUN


def test_assignment_1_gates_neither_load_nor_check_an_arbiter(tmp_path: Path) -> None:
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)
    (submission / "arbitration.py").write_text("raise RuntimeError('never imported')")

    report = run_gates(
        submission,
        _short_plan(),
        base_config=config_from_dict({}),
        release_root=tmp_path,
    )

    results = {result.gate: result for result in report.results}
    assert results[LOADS].status is GateStatus.PASSED
    assert MONOTONE not in results


# Fault scenarios: driven under injected faults, judged only on whether one broke the run.


def test_public_assignment_2_gates_add_one_fault_scenario_per_kind_to_the_routes() -> None:
    assignment_1 = load_gate_plan(PROJECT_ROOT / "configs" / "gates-assignment-1.yaml")
    plan = load_gate_plan(PROJECT_ROOT / "configs" / "gates-assignment-2.yaml")

    routes = [scenario for scenario in plan.scenarios if not scenario.is_fault_scenario]
    faulted = [scenario for scenario in plan.scenarios if scenario.is_fault_scenario]
    assert (plan.plan_id, plan.visibility) == ("drivebench-gates-assignment-2-public", "public")
    assert [(route.scenario_id, route.map, route.seed, route.horizon) for route in routes] == [
        (route.scenario_id, route.map, route.seed, route.horizon)
        for route in assignment_1.scenarios
    ]
    # Assignment 1's score would mean nothing under Assignment 2, so there is no par.
    assert plan.no_score is not None and "course/assignment-2/rubric.md" in plan.no_score
    assert all(route.par_steps is None for route in routes)
    assert sorted(fault.kind for scenario in faulted for fault in scenario.faults) == sorted(
        FaultKind
    )
    lidar = next(scenario for scenario in faulted if scenario.scenario_id == "lidar-dropout")
    # A LiDAR dropout acts only on a LiDAR safety scene.
    assert lidar.safety_source == "lidar"


def test_fault_scenario_parses_its_world_and_schedule(tmp_path: Path) -> None:
    path = tmp_path / "gates.yaml"
    path.write_text(
        "version: 1\nid: g\nvisibility: public\nscenarios:\n"
        "  - id: dark\n    map: S\n    seed: 0\n    horizon: 200\n"
        "    safety_source: lidar\n    stopped_vehicle_ahead_m: 45\n"
        "    faults:\n"
        "      - {id: lidar-1, kind: lidar_dropout, start_step: 30, duration_steps: 20}\n"
        "      - {id: late-1, kind: provider_latency, start_step: 0, duration_steps: 5,"
        " latency_s: 0.5}\n"
    )

    [scenario] = load_gate_plan(path).scenarios

    assert scenario == GateScenario(
        scenario_id="dark",
        map="S",
        seed=0,
        horizon=200,
        stopped_vehicle_ahead_m=45,
        safety_source="lidar",
        faults=(
            FaultSpec("lidar-1", FaultKind.LIDAR_DROPOUT, 30, 20),
            FaultSpec("late-1", FaultKind.PROVIDER_LATENCY, 0, 5, latency_s=0.5),
        ),
    )
    assert scenario.par_steps is None


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        (
            "{id: a, map: S, seed: 0, horizon: 100, faults: "
            "[{id: f, kind: camera_dropout, start_step: 100, duration_steps: 5}]}",
            "fault 'f' starts after the horizon",
        ),
        (
            "{id: a, map: S, seed: 0, horizon: 100, faults: "
            "[{id: f, kind: camera_dropout, start_step: 1}]}",
            "missing field 'duration_steps'",
        ),
        (
            "{id: a, map: S, seed: 0, horizon: 100, faults: "
            "[{id: f, kind: solar_flare, start_step: 1, duration_steps: 5}]}",
            "fault kind must be one of",
        ),
        (
            "{id: a, map: S, seed: 0, horizon: 100, par_steps: 50, safety_source: radar}",
            "safety_source must be 'oracle' or 'lidar'",
        ),
    ],
    ids=["after-horizon", "incomplete", "unknown-kind", "unknown-safety-source"],
)
def test_malformed_fault_scenario_is_rejected(
    tmp_path: Path,
    scenario: str,
    message: str,
) -> None:
    path = tmp_path / "gates.yaml"
    path.write_text(f"version: 1\nid: g\nvisibility: public\nscenarios:\n  - {scenario}\n")

    with pytest.raises(ValueError, match=message):
        load_gate_plan(path)


def _fault_plan() -> GatePlan:
    return GatePlan(
        "test-fault-gates",
        "public",
        (
            GateScenario(scenario_id="straight", map="S", seed=0, horizon=30, par_steps=30),
            GateScenario(
                scenario_id="camera-dark",
                map="S",
                seed=0,
                horizon=40,
                faults=(FaultSpec("camera-1", FaultKind.CAMERA_DROPOUT, 5, 20),),
            ),
        ),
    )


def test_fault_scenarios_need_a_base_configuration_that_asks_the_model() -> None:
    with pytest.raises(ValueError, match="enables vla"):
        check_gate_plan(_fault_plan(), config_from_dict({}))
    check_gate_plan(_fault_plan(), _vla_base())


def _fault_gates(tmp_path: Path, controller_source: str) -> GateReport:
    submission = _submission(tmp_path / "submission", controller_source)
    (submission / "observation.py").write_text(OBSERVATION)
    (submission / "arbitration.py").write_text(ARBITRATION)
    return run_gates(submission, _fault_plan(), base_config=_vla_base(), release_root=tmp_path)


@pytest.mark.needs("fixtures")
def test_a_fault_scenario_is_judged_apart_from_the_routes_and_the_score(
    tmp_path: Path,
) -> None:
    report = _fault_gates(tmp_path, WELL_BEHAVED)

    results = {result.gate: result for result in report.results}
    assert results[FAULTS_SURVIVED].status is GateStatus.PASSED
    assert results[FAULTS_SURVIVED].detail == (
        "1 fault scenario driven without a crash, an off-road exit, or a stopped run, "
        "under camera_dropout"
    )
    # The route gate and the score see only the route; the fault scenario need not arrive.
    assert results[RUNS].detail == (
        "1 route driven without an error; 1 route did not complete, so earned no "
        "driving credit: straight: did not arrive within 30 steps"
    )
    assert report.score is not None
    assert [scenario.scenario_id for scenario in report.score.scenarios] == ["straight"]


FULL_LOCK = WELL_BEHAVED.replace(
    "return (max(-1.0, min(1.0, steering)), max(-1.0, min(1.0, throttle)))",
    "return (1.0, 1.0)",
)


@pytest.mark.needs("fixtures")
def test_a_fault_scenario_that_leaves_the_road_fails_the_fault_gate(tmp_path: Path) -> None:
    report = _fault_gates(tmp_path, FULL_LOCK)

    results = {result.gate: result for result in report.results}
    assert results[FAULTS_SURVIVED].status is GateStatus.FAILED
    assert results[FAULTS_SURVIVED].detail == "camera-dark: left the road"


def test_a_run_of_routes_alone_does_not_run_the_fault_gate(tmp_path: Path) -> None:
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)
    (submission / "observation.py").write_text(OBSERVATION)
    (submission / "arbitration.py").write_text(ARBITRATION)

    report = run_gates(
        submission,
        _fault_plan(),
        base_config=_vla_base(),
        release_root=tmp_path,
        scenario_ids=["straight"],
    )

    results = {result.gate: result for result in report.results}
    assert results[FAULTS_SURVIVED].status is GateStatus.NOT_RUN
    assert results[FAULTS_SURVIVED].detail == "no fault scenario was driven"


def test_a_plan_that_declares_no_score_reports_its_reason_instead(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    submission = _submission(tmp_path / "submission", WELL_BEHAVED)
    route = GateScenario(scenario_id="straight", map="S", seed=0, horizon=30)
    plan = GatePlan("unscored", "public", (route,), no_score="Marked from hidden runs.")

    report = run_gates(
        submission, plan, base_config=config_from_dict({}), release_root=tmp_path
    )

    assert report.score is None
    assert report.to_dict()["no_score"] == "Marked from hidden runs."
    _print_gate_report(report)
    assert capsys.readouterr().out.endswith("\nNo score: Marked from hidden runs.\n")


def test_a_scored_plan_needs_par_for_every_route() -> None:
    route = GateScenario(scenario_id="straight", map="S", seed=0, horizon=30)

    with pytest.raises(ValueError, match="'straight' needs par_steps"):
        GatePlan("scored", "public", (route,))
    with pytest.raises(ValueError, match="no_score must be text"):
        GatePlan("blank", "public", (route,), no_score="  ")
