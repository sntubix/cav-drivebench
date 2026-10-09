import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_NOTEBOOKS = sorted(
    path
    for path in (PROJECT_ROOT / "course").glob("*/*.ipynb")
    if path.parent.name != "instructor"
)
PID_LAB = PROJECT_ROOT / "course" / "assignment-1" / "pid-lab.ipynb"
OBSERVATION_INSPECTOR = PROJECT_ROOT / "course" / "assignment-2" / "observation-inspector.ipynb"
PROBE_ANALYSIS = PROJECT_ROOT / "course" / "assignment-2" / "probe-analysis.ipynb"
# A student release ships Assignment 2 only once it is released.
ASSIGNMENT_2 = pytest.mark.skipif(
    not OBSERVATION_INSPECTOR.parent.is_dir(), reason="Assignment 2 is not released yet"
)


def _code_cells(notebook: Path) -> list[dict]:
    cells = json.loads(notebook.read_text())["cells"]
    return [cell for cell in cells if cell["cell_type"] == "code"]


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


@pytest.mark.parametrize("notebook", COURSE_NOTEBOOKS, ids=lambda path: path.stem)
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
    # The shipped observation.py is DriveBench's original observation.
    for probe in probes.values():
        assert namespace["build"](probe) == namespace["original"](probe)


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
