import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from metadrive_starter.config import config_from_dict
from metadrive_starter import release
from metadrive_starter.gates import GatePlan, GateScenario, GateStatus, run_gates
from metadrive_starter.release import ReleaseError, build_releases


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="needs git to export a commit"
)

REFERENCE = "class Controller:\n    pass  # the answer\n"
# Built at run time, so that this file never looks like it holds a secret.
PRIVATE_KEY = "-----BEGIN " + "PRIVATE KEY-----\nMIIE\n"


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=DriveBench",
            "-c",
            "user.email=drivebench@example.com",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout


def _repository(root: Path, extra: dict[str, str] | None = None) -> Path:
    files = {
        "src/foundation.py": "VALUE = 1\n",
        "configs/default.yaml": "simulator: {}\n",
        "configs/gates-public.yaml": "version: 1\nvisibility: public\n",
        "course/assignment-1/lab.ipynb": '{"cells": []}\n',
        "course/instructor/hidden.yaml": "version: 1\nvisibility: hidden\n",
        "course/instructor/reference-submission/controller.py": REFERENCE,
        "submission/controller.py": "class Controller(:\n",
        **(extra or {}),
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "scripts").mkdir()
    (root / "scripts" / "check.sh").write_text("#!/bin/sh\n")
    (root / "scripts" / "check.sh").chmod(0o755)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "release")
    return root


def _members(archive: Path) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(archive) as bundle:
        return {member.name: member for member in bundle.getmembers()}


def _unpack(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if member.isfile():
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.extractfile(member).read())
    return destination / archive.name.removesuffix(".tar.gz")


def test_student_release_leaves_out_course_instructor(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    commit = _git(repository, "rev-parse", "HEAD").strip()

    archive = build_releases(repository, tmp_path / "dist").student

    root = f"drivebench-student-{commit[:12]}"
    assert archive == tmp_path / "dist" / f"{root}.tar.gz"
    members = _members(archive)
    assert sorted(name for name, member in members.items() if member.isfile()) == [
        f"{root}/configs/default.yaml",
        f"{root}/configs/gates-public.yaml",
        f"{root}/course/assignment-1/lab.ipynb",
        f"{root}/release-manifest.json",
        f"{root}/scripts/check.sh",
        f"{root}/src/foundation.py",
        f"{root}/submission/controller.py",
    ]
    assert members[f"{root}/scripts/check.sh"].mode == 0o755
    assert members[f"{root}/src/foundation.py"].mode == 0o644


def test_instructor_release_is_the_whole_commit(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    commit = _git(repository, "rev-parse", "HEAD").strip()

    archive = build_releases(repository, tmp_path / "dist").instructor

    root = f"drivebench-instructor-{commit[:12]}"
    assert archive == tmp_path / "dist" / f"{root}.tar.gz"
    release = _unpack(archive, tmp_path / "unpacked")
    shipped = sorted(
        path.relative_to(release).as_posix() for path in release.rglob("*") if path.is_file()
    )
    assert shipped == sorted(
        [*_git(repository, "ls-files").split(), "release-manifest.json"]
    )
    manifest = json.loads((release / "release-manifest.json").read_text())
    assert "course/instructor/hidden.yaml" in manifest["files"]


def test_student_release_manifest_passes_the_tree_gate_once_unpacked(
    tmp_path: Path,
) -> None:
    archive = build_releases(_repository(tmp_path / "repository"), tmp_path).student

    release = _unpack(archive, tmp_path / "unpacked")

    manifest = json.loads((release / "release-manifest.json").read_text())
    # Neither submission/ nor notebooks are in the manifest.
    assert sorted(manifest["files"]) == [
        "configs/default.yaml",
        "configs/gates-public.yaml",
        "scripts/check.sh",
        "src/foundation.py",
    ]
    report = run_gates(
        release / "submission",
        GatePlan(
            "release",
            "public",
            (GateScenario(scenario_id="s", map="S", seed=0, horizon=10, par_steps=10),),
        ),
        base_config=config_from_dict({}),
        release_root=release,
    )
    [tree] = [result for result in report.results if result.gate.startswith("nothing outside")]
    assert (tree.status, tree.detail) == (GateStatus.PASSED, "4 shipped files unchanged")


def test_student_release_ships_only_committed_files(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    (repository / "src" / "foundation.py").write_text("VALUE = 2\n")
    (repository / "notes.txt").write_text("uncommitted\n")

    release = _unpack(build_releases(repository, tmp_path).student, tmp_path / "unpacked")

    assert (release / "src" / "foundation.py").read_text() == "VALUE = 1\n"
    assert not (release / "notes.txt").exists()


def test_releases_are_reproducible_and_never_overwritten(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")

    first = build_releases(repository, tmp_path / "first")
    second = build_releases(repository, tmp_path / "second")

    assert first.student.read_bytes() == second.student.read_bytes()
    assert first.instructor.read_bytes() == second.instructor.read_bytes()
    # One existing archive stops both, so a pair always comes from one build.
    second.student.unlink()
    with pytest.raises(ReleaseError, match="already exists"):
        build_releases(repository, tmp_path / "second")
    assert not second.student.exists()


@pytest.mark.parametrize("failing_archive", ["student", "instructor"])
def test_releases_leave_no_partial_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_archive: str,
) -> None:
    repository = _repository(tmp_path / "repository")
    real_add = release._add

    def disk_full(bundle, path, name, mtime) -> None:
        if name.startswith(f"drivebench-{failing_archive}-"):
            raise OSError("No space left on device")
        real_add(bundle, path, name, mtime)

    monkeypatch.setattr(release, "_add", disk_full)
    with pytest.raises(OSError, match="No space left"):
        build_releases(repository, tmp_path / "dist")

    assert list((tmp_path / "dist").iterdir()) == []


@pytest.mark.parametrize(
    ("name", "text", "problem"),
    [
        ("docs/controller.py", REFERENCE, "is a copy of course/instructor/reference-submission/controller.py"),
        ("configs/gates-final.yaml", "version: 1\nvisibility: hidden\n", "is a hidden manifest"),
        ("configs/race.json", '{"visibility": "hidden"}\n', "is a hidden manifest"),
        ("docs/notes.md", PRIVATE_KEY, "contains a private key"),
        (
            "configs/account.json",
            '{"type": "service_' + 'account", "project_id": "course"}\n',
            "contains a Google credential file",
        ),
        (".env", "GOOGLE_CLOUD_PROJECT=course\n", "is a credential file"),
        # Instructors get course/instructor/, but no release carries a credential.
        ("course/instructor/grader.pem", PRIVATE_KEY, "is a credential file"),
    ],
)
def test_releases_refuse_withheld_material_and_credentials(
    tmp_path: Path,
    name: str,
    text: str,
    problem: str,
) -> None:
    repository = _repository(tmp_path / "repository", {name: text})

    with pytest.raises(ReleaseError, match="refusing to release") as error:
        build_releases(repository, tmp_path / "dist")

    assert f"{name} {problem}" in str(error.value)
    assert not (tmp_path / "dist").exists()


def test_student_release_names_a_missing_revision(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")

    with pytest.raises(ReleaseError, match="rev-parse failed"):
        build_releases(repository, tmp_path, revision="no-such-tag")


MAINTAINER_DOCUMENTS = {
    "README.md": "# DriveBench\n",
    "CONTEXT.md": "# Vocabulary\n",
    "AGENTS.md": "# For maintainers\n",
    "CLAUDE.md": "See AGENTS.md.\n",
    "CONTRIBUTING.md": "# Contributing\n",
    "WORK_NOTES.md": "# Notes\n",
    "proposal.pdf": "%PDF-1.4\n",
    "docs/getting-started.md": "# Getting started\n",
    "docs/adr/0001-decision.md": "# A decision\n",
    "docs/agents/domain.md": "# Domain docs\n",
}


def test_student_release_leaves_out_maintainer_documents(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository", MAINTAINER_DOCUMENTS)

    archives = build_releases(repository, tmp_path / "dist")

    student = _unpack(archives.student, tmp_path / "student")
    shipped = {path.relative_to(student).as_posix() for path in student.rglob("*.*")}
    assert {"README.md", "CONTEXT.md", "docs/getting-started.md"} <= shipped
    for name in MAINTAINER_DOCUMENTS.keys() - {
        "README.md", "CONTEXT.md", "docs/getting-started.md"
    }:
        assert name not in shipped
    instructor = _unpack(archives.instructor, tmp_path / "instructor")
    assert all((instructor / name).is_file() for name in MAINTAINER_DOCUMENTS)


def test_releases_refuse_a_student_document_linking_to_a_withheld_file(
    tmp_path: Path,
) -> None:
    readme = (
        "Start with [getting started](docs/getting-started.md#setup), see\n"
        "[MetaDrive](https://example.com/metadrive) or [above](#top), and read\n"
        "the [contributing guide](CONTRIBUTING.md) and [ADR-0001](docs/adr/0001-decision.md).\n"
    )
    repository = _repository(
        tmp_path / "repository", {**MAINTAINER_DOCUMENTS, "README.md": readme}
    )

    with pytest.raises(ReleaseError, match="refusing to release") as error:
        build_releases(repository, tmp_path / "dist")

    assert str(error.value).splitlines()[1:] == [
        "  README.md links to CONTRIBUTING.md, which the student release leaves out",
        "  README.md links to docs/adr/0001-decision.md, which the student release leaves out",
    ]


def _student_repository(path: Path) -> Path:
    # The maintainers' clone of the student repository, on main as documented.
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "symbolic-ref", "HEAD", "refs/heads/main")
    return path


def _change(repository: Path, name: str, text: str | None) -> None:
    if text is None:
        (repository / name).unlink()
    else:
        (repository / name).parent.mkdir(parents=True, exist_ok=True)
        (repository / name).write_text(text)
    _git(repository, "add", "-A")
    _git(repository, "commit", "-q", "-m", f"change {name}")


def _publish(repository: Path, student: Path) -> str | None:
    # The author identity comes from the test's git configuration.
    _git(student, "config", "user.name", "DriveBench")
    _git(student, "config", "user.email", "drivebench@example.com")
    _git(student, "config", "commit.gpgsign", "false")
    return release.publish_student_release(repository, student)


def test_each_release_is_one_more_commit_on_the_student_history(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository", {"AGENTS.md": "# For maintainers\n"})
    student = _student_repository(tmp_path / "student")

    first = _publish(repository, student)
    _change(repository, "src/foundation.py", "VALUE = 2\n")
    _change(repository, "submission/request_policy.py", "class RequestPolicy:\n    pass\n")
    second = _publish(repository, student)

    assert first is not None and second is not None
    assert _git(student, "rev-list", "--parents", "-n", "1", second).split() == [second, first]
    tracked = set(_git(student, "ls-files").split())
    assert "submission/request_policy.py" in tracked
    assert "release-manifest.json" in tracked
    assert not [name for name in tracked if name.startswith("course/instructor")]
    assert "AGENTS.md" not in tracked
    assert (student / "src" / "foundation.py").read_text() == "VALUE = 2\n"
    assert (student / "scripts" / "check.sh").stat().st_mode & 0o111
    message = _git(student, "log", "-1", "--format=%B")
    assert message.startswith(f"DriveBench release {_git(repository, 'rev-parse', 'HEAD')[:12]}")
    # Releasing the same commit again changes nothing.
    assert _publish(repository, student) is None


def test_a_team_pulls_the_next_release_into_its_own_repository(tmp_path: Path) -> None:
    # The commands docs/getting-started.md gives teams, step for step.
    repository = _repository(tmp_path / "repository")
    student = _student_repository(tmp_path / "student")
    _publish(repository, student)
    own = tmp_path / "own.git"
    _git(tmp_path, "init", "-q", "--bare", str(own))
    team = tmp_path / "team"
    _git(tmp_path, "clone", "-q", str(student), str(team))
    _git(team, "remote", "rename", "origin", "upstream")
    _git(team, "remote", "add", "origin", str(own))
    _git(team, "push", "-q", "-u", "origin", "main")
    _change(team, "submission/controller.py", "class Controller:\n    pass  # ours\n")
    _git(team, "push", "-q")

    _change(repository, "src/foundation.py", "VALUE = 2\n")
    _change(repository, "submission/observation.py", "class ObservationBuilder:\n    pass\n")
    _publish(repository, student)
    _git(team, "pull", "-q", "--no-rebase", "--no-edit", "upstream", "main")
    _git(team, "push", "-q")

    assert (team / "submission" / "controller.py").read_text().endswith("# ours\n")
    assert (team / "src" / "foundation.py").read_text() == "VALUE = 2\n"
    assert (team / "submission" / "observation.py").is_file()
    # The team's own repository holds both its work and the release.
    assert _git(own, "rev-parse", "main") == _git(team, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    ("name", "text", "problem"),
    [
        (
            "submission/controller.py",
            "class Controller:\n    pass  # fixed\n",
            "submission/controller.py would change",
        ),
        ("submission/controller.py", None, "submission/controller.py would disappear"),
        (
            "course/assignment-1/lab.ipynb",
            '{"cells": [], "fixed": true}\n',
            "course/assignment-1/lab.ipynb would change",
        ),
    ],
)
def test_a_release_never_changes_a_shipped_file_teams_edit(
    tmp_path: Path, name: str, text: str | None, problem: str
) -> None:
    repository = _repository(tmp_path / "repository")
    student = _student_repository(tmp_path / "student")
    first = _publish(repository, student)
    _change(repository, name, text)

    with pytest.raises(ReleaseError, match="refusing to release") as error:
        _publish(repository, student)

    assert f"  {problem}" in str(error.value)
    assert _git(student, "rev-parse", "HEAD").strip() == first
    assert _git(student, "status", "--porcelain") == ""


def test_a_release_needs_a_clean_clone_of_the_student_repository(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    student = _student_repository(tmp_path / "student")
    _publish(repository, student)
    (student / "src" / "foundation.py").write_text("VALUE = 3\n")

    with pytest.raises(ReleaseError, match="uncommitted changes"):
        _publish(repository, student)
    with pytest.raises(ReleaseError, match="not a git repository"):
        release.publish_student_release(repository, tmp_path / "nowhere")


MARKED_REFERENCE = (
    "class Controller:\n"
    "    # >>> solution\n"
    "    GAIN = 0.3\n"
    "    # <<<\n"
    "    # >>> stub\n"
    "    # GAIN = 0.03\n"
    "    # <<<\n"
)


def _with_stub_generator(stub: str) -> dict[str, str]:
    generator = Path(__file__).resolve().parents[1] / "course" / "instructor" / "make_stubs.py"
    return {
        "course/instructor/make_stubs.py": generator.read_text(),
        "course/instructor/reference-submission/controller.py": MARKED_REFERENCE,
        "submission/controller.py": stub,
    }


@pytest.mark.instructor
def test_releases_ship_the_stub_a_marked_reference_generates(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path / "repository",
        _with_stub_generator("class Controller:\n    GAIN = 0.03\n"),
    )

    student = _unpack(build_releases(repository, tmp_path / "dist").student, tmp_path / "s")

    assert (student / "submission" / "controller.py").read_text() == (
        "class Controller:\n    GAIN = 0.03\n"
    )


@pytest.mark.instructor
def test_releases_refuse_a_stub_its_marked_reference_no_longer_generates(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "repository",
        _with_stub_generator("class Controller:\n    GAIN = 0.05\n"),
    )

    with pytest.raises(ReleaseError, match="refusing to release") as error:
        build_releases(repository, tmp_path / "dist")

    assert "submission/controller.py is not what its marked reference generates" in str(
        error.value
    )


def test_a_release_can_be_staged_for_a_maintainer_to_commit(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    student = _student_repository(tmp_path / "student")
    commit = _git(repository, "rev-parse", "HEAD").strip()

    assert release.stage_student_release(repository, student) == commit

    assert _git(student, "rev-list", "--all") == ""
    staged = _git(student, "diff", "--cached", "--name-only").split()
    assert "submission/controller.py" in staged
    assert "release-manifest.json" in staged
    _git(student, "commit", "-q", "-m", release.release_message(commit))
    assert _git(student, "log", "-1", "--format=%s").strip() == f"DriveBench release {commit[:12]}"
    assert release.stage_student_release(repository, student) is None


HELD_ASSIGNMENT = {
    "course/instructor/release-plan.yaml": (
        "held:\n  - course/assignment-2\n  - submission/observation.py\n"
    ),
    "course/assignment-2/README.md": "# Assignment 2\n",
    "submission/observation.py": "class ObservationBuilder:\n    pass\n",
}


def test_held_paths_ship_once_the_plan_releases_them(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository", HELD_ASSIGNMENT)

    archives = build_releases(repository, tmp_path / "dist")
    student = _unpack(archives.student, tmp_path / "student-archive")
    instructor = _unpack(archives.instructor, tmp_path / "instructor-archive")
    assert not (student / "course" / "assignment-2").exists()
    assert not (student / "submission" / "observation.py").exists()
    assert (instructor / "course" / "assignment-2" / "README.md").is_file()

    published = _student_repository(tmp_path / "student")
    _publish(repository, published)
    team = tmp_path / "team"
    _git(tmp_path, "clone", "-q", str(published), str(team))
    _change(team, "submission/controller.py", "class Controller:\n    pass  # ours\n")
    _change(repository, "course/instructor/release-plan.yaml", "held: []\n")
    _publish(repository, published)
    _git(team, "pull", "-q", "--no-rebase", "--no-edit", "origin", "main")

    assert (team / "course" / "assignment-2" / "README.md").is_file()
    assert (team / "submission" / "observation.py").is_file()
    assert (team / "submission" / "controller.py").read_text().endswith("# ours\n")


def test_a_release_plan_may_hold_only_paths_in_the_commit(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path / "repository",
        {"course/instructor/release-plan.yaml": "held:\n  - course/assignment-9\n"},
    )

    with pytest.raises(ReleaseError, match="course/assignment-9 is not in the released commit"):
        build_releases(repository, tmp_path / "dist")
