import json
import re
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

from metadrive_starter.cli import main
from metadrive_starter.gates import (
    CONTROL_LOOP_UNBLOCKED,
    MONOTONE_RESTRICTION,
    VALID_OBSERVATIONS,
)
from metadrive_starter.recorded_probes import INTERPRETATION_CATALOG, load_recorded_probes
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
def test_course_notebook_runs_and_ships_without_outputs(notebook: Path) -> None:
    _run(notebook)

    assert [
        (cell["execution_count"], cell["outputs"]) for cell in _code_cells(notebook)
    ] == [(None, [])] * len(_code_cells(notebook))


SHIPPED_CONTROLLER = (PROJECT_ROOT / "submission" / "controller.py").read_text()
HANDICAP = "self.speed_loop = PID(replace(settings.speed_pid, kp=0.03))"


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


def test_pid_lab_shows_a_team_every_task_still_to_do() -> None:
    namespace = _run(PID_LAB)

    # The shipped controller passes no implementation check, and every task
    # cell names what it found.
    assert namespace["passed"] == 0
    assert [result.passed for result in namespace["results"]] == [False] * 6


def test_pid_lab_notices_work_not_yet_copied_into_the_submission() -> None:
    assert HANDICAP in SHIPPED_CONTROLLER

    namespace = _run(
        PID_LAB,
        edit=lambda source: source.replace(HANDICAP, "self.speed_loop = PID(settings.speed_pid)"),
    )

    assert namespace["copied"] is False


@ASSIGNMENT_2
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
