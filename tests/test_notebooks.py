import json
import re
import time
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

from metadrive_starter.cli import main
from metadrive_starter.config import EventLogSettings, load_config
from metadrive_starter.faults import FaultKind, FaultSpec
from metadrive_starter.gates import (
    CONTROL_LOOP_UNBLOCKED,
    MONOTONE_RESTRICTION,
    VALID_OBSERVATIONS,
)
from metadrive_starter.recorded_probes import INTERPRETATION_CATALOG, load_recorded_probes
from metadrive_starter.simulation import run_simulation
from metadrive_starter.vla import declined


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_NOTEBOOKS = sorted(
    path
    for path in (PROJECT_ROOT / "course").glob("*/*.ipynb")
    if path.parent.name != "instructor"
)
PID_LAB = PROJECT_ROOT / "course" / "assignment-1" / "pid-lab.ipynb"
OBSERVATION_INSPECTOR = PROJECT_ROOT / "course" / "assignment-2" / "observation-inspector.ipynb"
PROBE_ANALYSIS = PROJECT_ROOT / "course" / "assignment-2" / "probe-analysis.ipynb"
REQUEST_TIMING = PROJECT_ROOT / "course" / "assignment-3" / "request-timing.ipynb"
# A student release ships Assignment 2, and the recorded probes its notebooks
# read, only once Assignment 2 is released.
ASSIGNMENT_2 = pytest.mark.needs("course/assignment-2", "fixtures")


def _code_cells(notebook: Path) -> list[dict]:
    cells = json.loads(notebook.read_text())["cells"]
    return [cell for cell in cells if cell["cell_type"] == "code"]


def _markdown(notebook: Path) -> str:
    cells = json.loads(notebook.read_text())["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown")


def _run(notebook: Path, edit=lambda source: source) -> dict[str, object]:
    """Run every code cell in order, as Jupyter would, and return the namespace.

    ``edit`` may change a cell's source first, as a team editing the cell would.
    """
    matplotlib.use("Agg")
    namespace: dict[str, object] = {"__name__": "__main__"}
    with warnings.catch_warnings():
        # The Agg backend cannot show figures; drawing them is enough.
        warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
        for cell in _code_cells(notebook):
            exec(compile(edit("".join(cell["source"])), str(notebook), "exec"), namespace)
    plt.close("all")
    return namespace


def test_course_ships_its_notebooks() -> None:
    assert PID_LAB in COURSE_NOTEBOOKS
    if OBSERVATION_INSPECTOR.parent.is_dir():
        assert {OBSERVATION_INSPECTOR, PROBE_ANALYSIS} <= set(COURSE_NOTEBOOKS)
    if REQUEST_TIMING.parent.is_dir():
        assert REQUEST_TIMING in COURSE_NOTEBOOKS


# The tests marked instructor in this file pin what the release ships:
# notebooks without outputs, the PID lab starting from the shipped controller
# with every task still to do, the shipped observation and arbiter as they are
# before a team's work, and the shipped agent.yaml's toggles. That work changes
# exactly that, so they run only where course/instructor/ is, never on a team's
# machine.
@pytest.mark.parametrize(
    "notebook",
    [
        pytest.param(path, marks=ASSIGNMENT_2)
        if path.parent == OBSERVATION_INSPECTOR.parent
        else path
        for path in COURSE_NOTEBOOKS
    ],
    ids=lambda path: path.stem,
)
@pytest.mark.instructor
def test_course_notebook_runs_and_ships_without_outputs(notebook: Path) -> None:
    _run(notebook)

    assert [
        (cell["execution_count"], cell["outputs"]) for cell in _code_cells(notebook)
    ] == [(None, [])] * len(_code_cells(notebook))


SHIPPED_CONTROLLER = (PROJECT_ROOT / "submission" / "controller.py").read_text()
HANDICAP = "self.speed_loop = PID(replace(settings.speed_pid, kp=0.03))"


@pytest.mark.instructor
def test_pid_lab_starts_from_a_copy_of_the_shipped_controller() -> None:
    # Teams copy the cell back below the docstring of submission/controller.py.
    below_docstring = SHIPPED_CONTROLLER[SHIPPED_CONTROLLER.index('"""', 3) + 3 :]
    copies = [
        "".join(cell["source"])
        for cell in _code_cells(PID_LAB)
        if "class Controller" in "".join(cell["source"])
    ]

    assert [copy.strip("\n") for copy in copies] == [below_docstring.strip("\n")]


def test_pid_lab_drives_its_own_controller_or_the_submission() -> None:
    namespace = _run(PID_LAB)
    build, settings = namespace["build"], namespace["settings"]

    assert type(build(settings)).__module__ == "__main__"
    namespace["DRIVE"] = "submission"
    assert type(build(settings)).__module__ == "submission.controller"
    # The shipped notebook and the shipped submission are the same controller.
    assert namespace["copied"] is True


@pytest.mark.instructor
def test_pid_lab_shows_a_team_every_task_still_to_do() -> None:
    namespace = _run(PID_LAB)

    # The shipped controller passes no implementation check, and every task
    # cell names what it found.
    assert namespace["passed"] == 0
    assert [result.passed for result in namespace["results"]] == [False] * 6


@pytest.mark.instructor
def test_pid_lab_notices_work_not_yet_copied_into_the_submission() -> None:
    assert HANDICAP in SHIPPED_CONTROLLER

    namespace = _run(
        PID_LAB,
        edit=lambda source: source.replace(HANDICAP, "self.speed_loop = PID(settings.speed_pid)"),
    )

    assert namespace["copied"] is False


@ASSIGNMENT_2
@pytest.mark.instructor
def test_observation_inspector_builds_the_shipped_observation_through_the_seam() -> None:
    namespace = _run(OBSERVATION_INSPECTOR)

    builder = namespace["ObservationBuilder"](namespace["config"].observation)
    assert type(builder).__module__ == "submission.observation"
    probes = namespace["probes"]
    assert len(probes) == 8
    # The shipped observation.py is DriveBench's original observation less the
    # line stating the number bounds, which a TODO replaces.
    bounds = "target_speed_mps must be non-negative; confidence must be between 0 and 1."
    for probe in probes.values():
        shipped, original = namespace["build"](probe), namespace["original"](probe)
        assert shipped.frame == original.frame
        assert shipped.prompt.splitlines() == [
            line for line in original.prompt.splitlines() if line != bounds
        ]
        assert bounds in original.prompt.splitlines()


@ASSIGNMENT_2
def test_probe_analysis_scores_the_example_in_opposite_directions() -> None:
    namespace = _run(PROBE_ANALYSIS)

    runs = {run["name"]: run for run in namespace["runs"]}
    agreement = namespace["agreement"]
    stopped = runs["demo-vla-semantic-stopped-vehicle"]["scenes"]["stopped-vehicle"]
    occluded = runs["demo-vla-semantic-partial-occlusion"]["scenes"]["partial-occlusion"]
    assert agreement("stopped-vehicle", stopped).agrees
    assert stopped["outcome"]["semantic_evaluation"]["tactical_action_passed"] is False
    assert occluded["outcome"]["semantic_evaluation"]["perception_passed"] is True
    assert not agreement("partial-occlusion", occluded).agrees


@ASSIGNMENT_2
@pytest.mark.instructor
def test_observation_inspector_builds_the_public_interpretation_scenes_as_grading_does() -> None:
    namespace = _run(OBSERVATION_INSPECTOR)

    probes = namespace["interpretation_probes"]
    assert len(probes) == 8
    assert namespace["INTERPRETATION_SCENE"] in probes
    for probe in probes.values():
        prompt = namespace["build"](probe, namespace["interpretation_config"]).prompt
        # Which lanes exist, but not whether each is clear, and no object.
        assert '"available"' in prompt
        assert '"observed_clear"' not in prompt
        assert '"objects"' not in prompt


@ASSIGNMENT_2
@pytest.mark.instructor
def test_observation_inspector_ends_each_task_with_the_gates_checks() -> None:
    stub = (PROJECT_ROOT / "submission" / "observation.py").read_text()
    todos = [
        " ".join(todo.split())
        for todo in re.findall(r"`# TODO: ([^`]+)`", _markdown(OBSERVATION_INSPECTOR))
    ]
    task_cells = [
        cell
        for cell in _code_cells(OBSERVATION_INSPECTOR)
        if "".join(cell["source"]).endswith("check()")
    ]

    # Each task names a TODO line the shipped stub has, and its cell ends with the check.
    assert len(todos) == len(task_cells) == 3
    assert all(f"# TODO: {todo}" in stub for todo in todos)

    namespace = _run(OBSERVATION_INSPECTOR)

    assert [(gate, passed) for gate, passed, _ in namespace["observation_checks"]()] == [
        (VALID_OBSERVATIONS, True),
        (CONTROL_LOOP_UNBLOCKED, True),
    ]


@ASSIGNMENT_2
def test_observation_inspector_checks_an_observation_as_the_gates_do(capsys) -> None:
    namespace = _run(OBSERVATION_INSPECTOR)
    shipped = namespace["ObservationBuilder"]

    class Redated:
        def __init__(self, settings):
            self.builder = shipped(settings)

        def build(self, request):
            observation = self.builder.build(request)
            frame = replace(observation.frame, timestamp_s=observation.frame.timestamp_s + 1.0)
            return replace(observation, frame=frame)

    class WithoutContract(Redated):
        def build(self, request):
            return replace(self.builder.build(request), prompt="Describe the road.")

    capsys.readouterr()
    namespace["ObservationBuilder"] = Redated
    namespace["check"]()
    redated = capsys.readouterr().out
    namespace["ObservationBuilder"] = WithoutContract
    namespace["check"]()
    without_contract = capsys.readouterr().out

    # Each of the eight scenes is built for Qwen and for SmolVLM, as the gates build it.
    assert f"NOT YET  {VALID_OBSERVATIONS}: 16 of 16 observations could not be sent" in redated
    assert "must keep the captured frame's timestamp" in redated
    # The output contract is a recommendation: reported, never failed.
    assert f"PASS     {VALID_OBSERVATIONS}: 16 observations built" in without_contract
    assert "in 16 of them the prompt lacks output-contract lines" in without_contract


@ASSIGNMENT_2
@pytest.mark.instructor
def test_probe_analysis_puts_every_answer_through_the_submitted_arbiter(capsys) -> None:
    namespace = _run(PROBE_ANALYSIS)
    output = capsys.readouterr().out

    arbiter = namespace["Arbiter"](namespace["settings"])
    assert type(arbiter).__module__ == "submission.arbitration"
    rows = [row for _, rows in namespace["arbitrated"] for row in rows]
    # Both recorded answers on all eight recorded scenes; the shipped arbiter
    # endorses each, and the gates' checks pass.
    assert len(rows) == 16
    assert {row["decision"].outcome.value for row in rows} == {"endorsed"}
    assert f"PASS     {MONOTONE_RESTRICTION}: 16 assessments arbitrated" in output
    assert f"PASS     {CONTROL_LOOP_UNBLOCKED}: every arbiter call returned" in output


@ASSIGNMENT_2
def test_probe_analysis_checks_an_arbiter_as_the_gates_do() -> None:
    namespace = _run(PROBE_ANALYSIS)

    class GrantsSpeed:
        def __init__(self, settings):
            pass

        def begin_episode(self):
            pass

        def arbitrate(self, request):
            speed = request.assessment.proposed_target_speed_mps + 1.0
            return replace(request.assessment, proposed_target_speed_mps=speed)

        def review(self, request):
            return True

    namespace["Arbiter"] = GrantsSpeed
    namespace["CONTROL_STEP_S"] = -1.0
    run = namespace["runs"][0]
    monotone, unblocked = namespace["arbiter_checks"]([(run, namespace["arbitrate_run"](run))])

    assert monotone[:2] == (MONOTONE_RESTRICTION, False)
    assert "proposed_target_speed_mps raised from 9.72 to 10.72 m/s" in monotone[2]
    assert unblocked[:2] == (CONTROL_LOOP_UNBLOCKED, False)


@ASSIGNMENT_2
def test_probe_analysis_gives_the_arbiter_the_scene_measured_now() -> None:
    # Every recorded answer is older than the floor's scene freshness limit, and
    # the live loop hands the arbiter the current scene, so the notebook must
    # too, in arbitration and in every review, or a reasonable freshness check
    # declines everything here and nothing at grading.
    namespace = _run(PROBE_ANALYSIS)
    stale_reviews = []

    class DeclinesStaleScenes:
        def __init__(self, settings):
            self.limit_s = settings.scene_stale_after_s

        def begin_episode(self):
            pass

        def arbitrate(self, request):
            if request.now_s - request.scene.timestamp_s > self.limit_s:
                return declined(request.assessment)
            return request.assessment

        def review(self, request):
            if request.now_s - request.scene.timestamp_s > self.limit_s:
                stale_reviews.append(request)
            return True

    namespace["Arbiter"] = DeclinesStaleScenes
    rows = [row for run in namespace["runs"] for row in namespace["arbitrate_run"](run)]

    assert len(rows) == 16
    assert {row["decision"].outcome.value for row in rows} == {"endorsed"}
    assert stale_reviews == []


def _scene_outcome(confidence: float | None, model_id: str = "drivebench-smolvlm-256m-q8-0") -> dict:
    """A probe scene's metadata: decoded at ``confidence``, or refused when it is None."""
    if confidence is None:
        return {"outcome": {"status": "failure", "error_category": "VLAResponseError"}}
    return {
        "outcome": {
            "status": "success",
            "model_id": model_id,
            "parsed_assessment": {"confidence": confidence},
        }
    }


@ASSIGNMENT_2
def test_probe_analysis_counts_smolvlm_answers_as_marking_does(capsys) -> None:
    namespace = _run(PROBE_ANALYSIS)
    output = capsys.readouterr().out
    answers = namespace["smolvlm_answers"]

    # The example replays recorded answers, so no run asked SmolVLM.
    assert "None of these runs asked SmolVLM" in output
    floor = namespace["smolvlm_floor"]
    assert floor == 0.5
    run = {
        "scenes": {
            "at-the-floor": _scene_outcome(floor),
            "above": _scene_outcome(0.9),
            "below": _scene_outcome(0.0),
            "refused": _scene_outcome(None),
            "unanswered": {
                "outcome": {"status": "failure", "error_category": "ModelTimeoutError"}
            },
        }
    }
    # Decoded and at the floor or above; the refused one counts, the unanswered not.
    assert len(answers(run)) == 4
    assert namespace["accepted"](answers(run)) == (2, 3)
    assert answers({"scenes": {"a": _scene_outcome(0.9, model_id="qwen")}}) == []


@ASSIGNMENT_2
def test_probe_analysis_pools_hazards_as_marking_does(capsys) -> None:
    namespace = _run(PROBE_ANALYSIS)
    output = capsys.readouterr().out
    hazard_counts, pooled = namespace["hazard_counts"], namespace["pooled"]
    hazard = namespace["VLAHazard"]
    kinds, places, risk = (
        namespace["HazardType"],
        namespace["RelativeLocation"],
        namespace["RiskLevel"],
    )
    ahead = hazard(kinds.VEHICLE, places.FRONT, risk.HIGH)
    person = hazard(kinds.PEDESTRIAN, places.FRONT, risk.HIGH)

    # A found vehicle, then an invented one on the empty road.
    assert hazard_counts([ahead], "stopped-vehicle") == (1, 1, 1, 1)
    assert hazard_counts([ahead], "clear-straight") == (0, 0, 0, 1)
    figures = pooled(
        [hazard_counts([ahead], "stopped-vehicle"), hazard_counts([ahead], "clear-straight")]
    )
    assert (figures["recall"], figures["precision"]) == (1.0, 0.5)
    assert figures["f1"] == pytest.approx(2 / 3)
    # Nothing claimed and nothing found is no F1 at all.
    assert pooled([hazard_counts([], "stopped-vehicle")])["f1"] == 0.0
    # A public interpretation scene is scored against its truth, as marking does.
    assert hazard_counts([person], "public-single-pedestrian-ego-13m") == (1, 1, 1, 1)
    assert hazard_counts([ahead], "public-single-pedestrian-ego-13m") == (0, 1, 0, 1)
    # An answer that does not decode counts, claiming nothing; one never answered does not.
    run = {
        "scenes": {
            "clear-straight": {**_scene_outcome(None), "scenario": namespace["recorded_scenarios"]["clear-straight"]},
            "curved-road": {
                "scenario": namespace["recorded_scenarios"]["curved-road"],
                "outcome": {"status": "failure", "error_category": "ModelTimeoutError"},
            },
            "an-unrecorded-scene": {**_scene_outcome(0.9), "scenario": {}},
        }
    }
    assert namespace["scored_scenes"](run) == {"clear-straight": []}
    assert "the constant answer" in output


@ASSIGNMENT_2
def test_probe_analysis_reads_a_team_probe_of_the_public_interpretation_scenes(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "probe-interpretation-01"
    status = main(
        [
            "probe",
            "--config",
            str(PROJECT_ROOT / "configs" / "demo-vla-semantic-stopped-vehicle.yaml"),
            "--submission",
            str(PROJECT_ROOT / "submission"),
            "--infer",
            "--replay-from",
            str(INTERPRETATION_CATALOG),
            "--output-dir",
            str(output),
        ]
    )
    assert status == 0, capsys.readouterr().err
    capsys.readouterr()

    namespace = _run(
        PROBE_ANALYSIS,
        edit=lambda source: source.replace("PROBE_RUNS = []", f"PROBE_RUNS = [{str(output)!r}]"),
    )

    printed = capsys.readouterr().out
    claimed = namespace["scored_scenes"](namespace["runs"][0])
    assert set(claimed) == {probe.scenario_id for probe in load_recorded_probes(INTERPRETATION_CATALOG)}
    assert "probe-interpretation-01: 8 scenes" in printed
    # The recorded answer reports one vehicle ahead on all eight: one scene holds one.
    assert "recall 0.100 (1/10), precision 0.125 (1/8)" in printed


@ASSIGNMENT_2
def test_probe_analysis_never_pools_runs_that_differ(capsys) -> None:
    namespace = _run(PROBE_ANALYSIS)
    output = capsys.readouterr().out
    stopped, occluded = namespace["runs"]

    # The example's two runs replay different recorded answers: each is reported
    # apart, with no median or range over the two (#31). Reruns pool.
    assert namespace["groups"] == [[stopped], [occluded]]
    assert "median" not in output and "range" not in output
    assert namespace["rerun_groups"]([stopped, occluded, stopped]) == [
        [stopped, stopped],
        [occluded],
    ]


# Assignment 3: the request timing notebook.


class _Standstill:
    """A controller that neither steers nor drives: the event log is what is tested."""

    def update(self, tick) -> tuple[float, float]:
        return (0.0, 0.0)

    def reset_speed_control(self) -> None:
        pass


class _Every:
    """A request policy that asks once every ``period_s``, whatever the road does."""

    def __init__(self, period_s: float) -> None:
        self.period_s = period_s

    def begin_episode(self) -> None:
        pass

    def should_request(self, tick) -> bool:
        return tick.last_requested_at_s is None or tick.now_s - tick.last_requested_at_s >= self.period_s - 1e-9


@pytest.mark.instructor
def test_request_timing_ends_each_task_with_its_check() -> None:
    stub = (PROJECT_ROOT / "submission" / "request_policy.py").read_text()
    todos = [
        " ".join(todo.split())
        for todo in re.findall(r"`# TODO: ([^`]+)`", _markdown(REQUEST_TIMING))
    ]
    task_cells = [
        "".join(cell["source"]).splitlines()[-1]
        for cell in _code_cells(REQUEST_TIMING)
        if re.fullmatch(r"check\(\d\)", "".join(cell["source"]).splitlines()[-1])
    ]

    # Each task names TODO lines the shipped stub has, and its cell ends with its check.
    assert task_cells == ["check(1)", "check(2)", "check(3)", "check(4)"]
    assert len(todos) == 6
    assert all(f"# TODO: {todo}" in stub for todo in todos)


@pytest.mark.instructor
def test_request_timing_shows_the_shipped_timer_every_task_but_the_first_still_to_do(capsys) -> None:
    namespace = _run(REQUEST_TIMING)
    capsys.readouterr()

    # Asking whenever DriveBench may reacts quickly, and wastes, spends through
    # faults, and runs out of budget.
    assert [namespace["check"](task) for task in (1, 2, 3, 4)] == [True, False, False, False]
    printed = capsys.readouterr().out
    assert "PASS     asks when the scene changes: every event asked about within 1.5 s" in printed
    assert "NOT YET  holds while the command holds: 23 of 33 requests wasted" in printed
    assert "NOT YET  backs off under the outage" in printed
    assert "NOT YET  keeps a reserve of the budget: 0 of 15 requests left" in printed
    assert f"PASS     {MONOTONE_RESTRICTION}" in printed
    assert f"PASS     {CONTROL_LOOP_UNBLOCKED}" in printed
    assert type(namespace["RequestPolicy"](namespace["settings"])).__module__ == "submission.request_policy"


def test_request_timing_drives_at_the_timing_the_brief_states() -> None:
    namespace = _run(REQUEST_TIMING)
    settings = namespace["RequestPolicySettings"].from_vla(namespace["drive_config"](60).vla)

    assert (settings.action_horizon_s, settings.request_timeout_s, settings.request_budget) == (4.0, 6.0, 60)
    assert namespace["drive_config"](60).command_validation.maximum_command_age_s == 2.5
    assert namespace["ANSWER_LATENCY_S"] == 1.4
    brief = (REQUEST_TIMING.parent / "README.md").read_text()
    assert "a command holds for 4.0 s" in " ".join(brief.split())
    assert "an answer more than 2.5 s old when it arrives is refused" in " ".join(brief.split())


def test_request_timing_counts_waste_as_the_brief_states() -> None:
    namespace = _run(REQUEST_TIMING)
    drive, wasted = namespace["drive"], namespace["wasted"]

    # Asking as soon as each answer arrives, 1.4 s in, leaves 2.6 s of a 4 s
    # command: more than the 2.5 s age limit, so on the empty road it is waste.
    eager = drive(policy=_Every(1.4))
    reasons = wasted(eager)
    early = [request for request in eager.requests if request.sent_s < 8.0]
    assert [request.sent_s for request in early[:3]] == [0.0, 1.4, 2.8]
    assert early[0].request_id not in reasons
    assert reasons[early[1].request_id] == "2.6 s left of the command, nothing changed"
    # 1.5 s in, the request falls in the command's last 2.5 s: never waste.
    patient = drive(policy=_Every(1.5))
    assert not [request for request in patient.requests if request.request_id in wasted(patient)]
    # During a fault, one request every 5 s wastes nothing, and more often does.
    outage = namespace["FaultWindow"](namespace["FaultKind"].PROVIDER_OUTAGE, 14.0, 24.0)
    assert any("in a fault" in reason for reason in wasted(drive(policy=_Every(1.0), faults=[outage])).values())
    assert not any("in a fault" in reason for reason in wasted(drive(policy=_Every(5.0), faults=[outage])).values())


def test_request_timing_finds_each_event_and_the_delay_to_the_next_request() -> None:
    namespace = _run(REQUEST_TIMING)
    found = namespace["reactions"](namespace["drive"](policy=_Every(1.5)))

    assert [label for _, label, _ in found][:2] == ["stopped-car within 50 m", "left lane occupied"]
    assert all(delay_s is not None and delay_s <= 1.5 for _, _, delay_s in found)
    assert namespace["falling"](1.0, 1.0, 3.0) == 1.0
    assert namespace["falling"](2.0, 1.0, 3.0) == 0.5
    assert namespace["falling"](None, 1.0, 3.0) == 0.0


def test_request_timing_checks_a_policy_as_the_gates_do() -> None:
    namespace = _run(REQUEST_TIMING)

    class Yes(_Every):
        def should_request(self, tick):
            return "yes"

    class Slow(_Every):
        def should_request(self, tick):
            time.sleep(0.15)
            return False

    monotone, unblocked = namespace["gate_checks"]([namespace["drive"](policy=Yes(1.0))])
    assert monotone[:2] == (MONOTONE_RESTRICTION, False)
    assert "should_request(tick) must return True or False; got 'yes'" in monotone[2]
    assert unblocked[:2] == (CONTROL_LOOP_UNBLOCKED, True)
    namespace["DURATION_S"] = 0.3
    _, unblocked = namespace["gate_checks"]([namespace["drive"](policy=Slow(1.0))])
    assert unblocked[:2] == (CONTROL_LOOP_UNBLOCKED, False)


def test_request_timing_reads_a_closed_loop_event_log(tmp_path: Path, capsys) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-vla-fixture.yaml")
    config.simulator.horizon = 60
    config.vla = replace(
        config.vla,
        fixture=replace(config.vla.fixture, path=str(PROJECT_ROOT / config.vla.fixture.path)),
    )
    config.faults = (FaultSpec("quota-1", FaultKind.PROVIDER_QUOTA, start_step=20, duration_steps=20),)
    log = tmp_path / "events.jsonl"
    config.event_log = EventLogSettings(enabled=True, path=str(log), scenario_id="request-timing")
    run_simulation(
        config,
        controller_factory=lambda settings: _Standstill(),
        request_policy_factory=lambda settings: _Every(0.5),
    )

    namespace = _run(
        REQUEST_TIMING,
        edit=lambda source: source.replace("EVENT_LOGS = []", f"EVENT_LOGS = [{str(log)!r}]"),
    )
    printed = capsys.readouterr().out
    (records,) = namespace["runs_in_event_log"](log)
    line = namespace["drive_from_event_log"](records)

    assert line["action_horizon_s"] == 2.0
    # The fixture demo is unpaced, and the log still gives the pace it held.
    assert line["asked_factor"] is None and line["achieved_factor"] > 0.0
    assert "not paced" in printed
    (fault,) = line["faults"]
    assert fault[0] == "provider_quota" and fault[1] == pytest.approx(2.0)
    refused = [request for request in line["requests"] if request["refused"]]
    answered = [request for request in line["requests"] if not request["refused"]]
    assert refused and all(fault[1] <= request["sent_s"] < fault[2] for request in refused)
    assert answered and all(request["authority"] is not None for request in answered if request["collected_s"])
    assert f"== events.jsonl: {len(line['requests'])} requests" in printed

    # A second run into the same file is appended: each run is read on its own.
    again = [
        json.dumps({**json.loads(text), "run_id": "second-run"})
        for text in log.read_text().splitlines()
    ]
    log.write_text(log.read_text() + "\n".join(again) + "\n")
    first, second = namespace["runs_in_event_log"](log)
    assert namespace["drive_from_event_log"](first) == line
    assert namespace["drive_from_event_log"](second) == line
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the figures have no window to show in
        for number, run in enumerate((first, second), start=1):
            namespace["describe_log"](namespace["drive_from_event_log"](run), f"events.jsonl, run {number} of 2")
    assert f"== events.jsonl, run 2 of 2: {len(line['requests'])} requests" in capsys.readouterr().out
