from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_CHECK = PROJECT_ROOT / "scripts" / "platform-check.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_platform_check_help_and_syntax() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(PLATFORM_CHECK)],
        check=False,
        capture_output=True,
        text=True,
    )
    help_result = subprocess.run(
        [str(PLATFORM_CHECK), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert syntax.returncode == 0, syntax.stderr
    assert help_result.returncode == 0
    assert "G3a" in help_result.stdout
    assert "G3b" in help_result.stdout


def _fake_project(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A project holding only the script, with fake uname and docker on PATH."""
    project = tmp_path / "project"
    scripts = project / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(PLATFORM_CHECK, scripts / PLATFORM_CHECK.name)
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    _write_executable(
        fake_bin / "uname",
        """#!/usr/bin/env bash
case "${1-}" in
  -s) printf 'Linux\\n' ;;
  -r) printf '6.8.0-drivebench-test\\n' ;;
  -m) printf 'x86_64\\n' ;;
  *) printf 'Linux\\n' ;;
esac
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
if [[ "${1-}" == "--version" ]]; then
  printf 'Docker version test\\n'
  exit 0
fi
if [[ "${1-}" == "version" ]]; then
  printf '29.0-test\\n'
  exit 0
fi
if [[ "${1-}" == "compose" && "${2-}" == "version" ]]; then
  printf '2.99-test\\n'
  exit 0
fi
call="$*"
printf '%s\\n' "${call//$'\\n'/ }" >> "${DOCKER_CALLS}"

output_rel=""
for item in "$@"; do
  case "${item}" in
    PLATFORM_OUTPUT_REL=*) output_rel=${item#PLATFORM_OUTPUT_REL=} ;;
  esac
done
if [[ -n "${output_rel}" && "$*" == *"fixture/events.jsonl"* ]]; then
  mkdir -p -- "${PWD}/${output_rel}/fixture"
  printf '{"event":"fake"}\\n' > "${PWD}/${output_rel}/fixture/events.jsonl"
fi
exit 0
""",
    )

    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["DOCKER_CALLS"] = str(tmp_path / "docker-calls.log")
    return project, environment


def _check(project: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(project / "scripts" / PLATFORM_CHECK.name),
            "--output-dir",
            "artifacts/test-run",
            "--skip-build",
        ],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _statuses(project: Path) -> tuple[dict[str, object], dict[str, str]]:
    report = json.loads(
        (project / "artifacts/test-run/gates.json").read_text(encoding="utf-8")
    )
    return report, {gate["id"]: gate["status"] for gate in report["gates"]}


def test_platform_check_records_success_and_manual_gates_with_fake_docker(
    tmp_path: Path,
) -> None:
    project, environment = _fake_project(tmp_path)
    (project / "fixtures").mkdir()
    result = _check(project, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    report, statuses = _statuses(project)
    assert report["successful"] is True
    assert statuses["g0-platform"] == "passed"
    assert statuses["g1-build"] == "not_run"
    assert statuses["g1-assets"] == "passed"
    assert statuses["g2-tests"] == "passed"
    assert statuses["g2-gates"] == "passed"
    assert statuses["g4-gates"] == "passed"
    assert statuses["g3a-camera"] == "passed"
    assert statuses["g4-fixture"] == "passed"
    assert statuses["g4-replay"] == "passed"
    assert statuses["g3a-visual-review"] == "not_run"
    assert statuses["g3b-gui"] == "not_run"
    assert statuses["g5-local-vlm"] == "not_run"
    # A student release has no reference controller, so every run drives the
    # shipped submission.
    calls = (tmp_path / "docker-calls.log").read_text().splitlines()
    driving = [
        call
        for call in calls
        if "metadrive-starter run" in call
        or "metadrive-starter smoke" in call
        or "metadrive-starter gates" in call
    ]
    assert len(driving) == 5
    assert all("--submission submission" in call for call in driving)
    # The public gates once as Assignment 1 runs them, once through the model.
    gates = [call for call in driving if "metadrive-starter gates" in call]
    assert ["configs/demo-vla-fixture.yaml" in call for call in gates] == [False, True]

    repeated = _check(project, environment)
    assert repeated.returncode == 2
    assert "output already exists" in repeated.stderr


def test_platform_check_passes_a_release_without_fixtures_and_skips_the_model_gates(
    tmp_path: Path,
) -> None:
    # Student releases hold fixtures/ back until Assignment 2.
    project, environment = _fake_project(tmp_path)
    result = _check(project, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    report, statuses = _statuses(project)
    assert report["successful"] is True
    assert statuses["g2-gates"] == "passed"
    assert statuses["g3a-camera"] == "passed"
    for gate in ("g4-fixture", "g4-replay", "g4-gates"):
        assert statuses[gate] == "not_run"
        reason = (project / f"artifacts/test-run/logs/{gate}.log").read_text()
        assert "does not include fixtures/" in reason
    calls = (tmp_path / "docker-calls.log").read_text()
    assert "demo-vla-fixture" not in calls
