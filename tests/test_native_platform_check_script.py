from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_CHECK = PROJECT_ROOT / "scripts" / "native-platform-check.sh"
SUMMARY_SCRIPT = PROJECT_ROOT / "scripts" / "native_vlm_summary.py"
MODEL_NAME = "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ_NAME = "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
LLAMA_ARCHIVE_NAME = "llama-b10771-bin-ubuntu-x64.tar.gz"
LLAMA_ARCHIVE_SHA256 = "42bb60d6027c99ec05ff1a5ae441e45345a3fcccaf86ca655b4dc51b5519c814"
VULKAN_ARCHIVE_NAME = "llama-b10771-bin-ubuntu-vulkan-x64.tar.gz"
VULKAN_ARCHIVE_SHA256 = "35d98f9c60dcd6c1af2267990b33fcd981a011f8b902dcbb0d5f280bf807bd53"
VULKAN_DEVICES = (
    "Available devices:\n"
    "  Vulkan0: NVIDIA RTX PRO 6000 Blackwell (97887 MiB, 96000 MiB free)"
)
MODEL_SHA256 = "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"
MMPROJ_SHA256 = "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82"
SMOLVLM_MODEL_NAME = "SmolVLM-256M-Instruct-Q8_0.gguf"
SMOLVLM_MMPROJ_NAME = "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
SMOLVLM_MODEL_SHA256 = "2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65"
SMOLVLM_MMPROJ_SHA256 = "7e943f7c53f0382a6fc41b6ee0c2def63ba4fded9ab8ed039cc9e2ab905e0edd"
RUN_DIR = "artifacts/native-platform-validation/native-test-run"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_native_platform_check_help_and_syntax() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(NATIVE_CHECK)],
        check=False,
        capture_output=True,
        text=True,
    )
    help_result = subprocess.run(
        [str(NATIVE_CHECK), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert syntax.returncode == 0, syntax.stderr
    assert help_result.returncode == 0
    assert "native Ubuntu x86_64" in help_result.stdout
    assert "schema-off/on" in help_result.stdout
    assert "two clean llama-server rounds" in help_result.stdout


def _fake_native_project(
    tmp_path: Path,
) -> tuple[Path, list[str], dict[str, str], Path]:
    """A project whose native check runs against fake tools, servers, and models."""
    project = tmp_path / "project"
    scripts = project / "scripts"
    model_dir = project / "tmp" / "native-vlm"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(NATIVE_CHECK, scripts / NATIVE_CHECK.name)
    shutil.copy2(SUMMARY_SCRIPT, scripts / SUMMARY_SCRIPT.name)
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (project / "fixtures").mkdir()
    (model_dir / MODEL_NAME).write_text("fake model\n", encoding="utf-8")
    (model_dir / MMPROJ_NAME).write_text("fake projector\n", encoding="utf-8")
    (model_dir / SMOLVLM_MODEL_NAME).write_text("fake SmolVLM\n", encoding="utf-8")
    (model_dir / SMOLVLM_MMPROJ_NAME).write_text("fake SmolVLM projector\n", encoding="utf-8")

    os_release = tmp_path / "os-release"
    os_release.write_text(
        'ID=ubuntu\nPRETTY_NAME="Ubuntu 24.04 fake"\n',
        encoding="utf-8",
    )
    proc_version = tmp_path / "proc-version"
    proc_version.write_text("Linux version 6.8 test\n", encoding="utf-8")
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("0::/user.slice\n", encoding="utf-8")
    marker = tmp_path / "server-ready"

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
        fake_bin / "git",
        """#!/usr/bin/env bash
if [[ "${1-}" == "rev-parse" ]]; then
  printf '0123456789abcdef0123456789abcdef01234567\\n'
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "sha256sum",
        f"""#!/usr/bin/env bash
path=${{!#}}
case "${{path}}" in
  *{LLAMA_ARCHIVE_NAME}) digest={LLAMA_ARCHIVE_SHA256} ;;
  *{VULKAN_ARCHIVE_NAME}) digest={VULKAN_ARCHIVE_SHA256} ;;
  *{MODEL_NAME}) digest={MODEL_SHA256} ;;
  *{SMOLVLM_MMPROJ_NAME}) digest=${{FAKE_SMOLVLM_MMPROJ_SHA256:-{SMOLVLM_MMPROJ_SHA256}}} ;;
  *{SMOLVLM_MODEL_NAME}) digest={SMOLVLM_MODEL_SHA256} ;;
  *{MMPROJ_NAME}) digest={MMPROJ_SHA256} ;;
  *) digest=$(python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${{path}}") ;;
esac
printf '%s  %s\\n' "${{digest}}" "${{path}}"
""",
    )
    # Samples one fixed row for any process; the Docker image has no ps at all.
    _write_executable(
        fake_bin / "ps",
        """#!/usr/bin/env bash
pid=""
while (($# > 0)); do
  case "$1" in
    -p) pid=$2; shift 2 ;;
    *) shift ;;
  esac
done
kill -0 "${pid}" 2>/dev/null || exit 1
printf '%s 2048 4096 1.0 00:01\\n' "${pid}"
""",
    )
    _write_executable(
        fake_bin / "Xvfb",
        """#!/usr/bin/env bash
trap 'exit 0' TERM INT
while true; do sleep 1; done
""",
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
case "$*" in
  *driver_version*) printf '580.178.04\\n' ;;
  *) printf 'NVIDIA RTX PRO 6000 Blackwell, 97887 MiB\\n' ;;
esac
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/usr/bin/env bash
if [[ -f "${FAKE_SERVER_MARKER}" ]]; then
  printf '{"status":"ok"}\\n'
  exit 0
fi
exit 7
""",
    )
    llama_server = model_dir / "llama-server"
    _write_executable(
        llama_server,
        """#!/usr/bin/env bash
if [[ "${1-}" == "--version" ]]; then
  printf 'version: 0.3.0-dev (build 10771, commit 733905474)\\n'
  printf 'built with GNU 11.4.0 for Linux x86_64\\n'
  exit 0
fi
if [[ "${1-}" == "--list-devices" ]]; then
  printf '%b\\n' "${FAKE_LLAMA_DEVICES:-Available devices:}"
  exit 0
fi
touch "${FAKE_SERVER_MARKER}"
trap 'rm -f "${FAKE_SERVER_MARKER}"; exit 0' TERM INT EXIT
while true; do sleep 1; done
""",
    )
    for archive_name in (LLAMA_ARCHIVE_NAME, VULKAN_ARCHIVE_NAME):
        with tarfile.open(model_dir / archive_name, "w:gz") as archive:
            archive.add(llama_server, arcname="build/bin/llama-server")
    _write_executable(
        fake_bin / "time",
        """#!/usr/bin/env bash
output=""
while (($# > 0)); do
  case "$1" in
    -f) shift 2 ;;
    -o) output=$2; shift 2 ;;
    *) break ;;
  esac
done
"$@"
status=$?
printf 'wall_seconds=8.0\\nmax_rss_kib=4096\\n' > "${output}"
exit "${status}"
""",
    )
    _write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
if [[ "${1-}" == "--version" ]]; then
  printf 'uv 0.test\\n'
  exit 0
fi
if [[ "$*" == *"scripts/native_vlm_summary.py"* ]]; then
  shift 3
  exec python3 "$@"
fi
if [[ "$*" == *"scripts/native_render_check.py"* ]]; then
  mode=""
  output=""
  previous=""
  for item in "$@"; do
    if [[ "${previous}" == "--render-path" ]]; then mode=${item}; fi
    if [[ "${previous}" == "--output" ]]; then output=${item}; fi
    previous=${item}
  done
  valid=true
  if [[ " ${FAKE_RENDER_FAILS-} " == *" ${mode} "* ]]; then valid=false; fi
  mkdir -p -- "$(dirname -- "${output}")"
  printf '{"render_path":"%s","display":"%s","pipe":"fake-%s-pipe","valid":%s}\\n' \
    "${mode}" "${DISPLAY-}" "${mode}" "${valid}" > "${output}"
  [[ "${valid}" == true ]]
  exit
fi
if [[ "$*" == *"python -m pytest"* ]]; then
  printf 'DISPLAY=%s\\n' "${DISPLAY-}"
fi
if [[ -n "${FAKE_QWEN_STRUCTURED_FAILS-}" && "$*" == *"vla-probe-qwen3-vl-2b-llama-structured.yaml"* ]]; then
  exit 1
fi
probe_failed=false
if [[ -n "${FAKE_QWEN_PROBE_FAILS-}" && "$*" == *"vla-probe-qwen3-vl-2b-llama.yaml"* ]]; then
  probe_failed=true
fi

output_dir=""
event_log=""
previous=""
for item in "$@"; do
  if [[ "${previous}" == "--output-dir" ]]; then output_dir=${item}; fi
  if [[ "${previous}" == "--event-log" ]]; then event_log=${item}; fi
  previous=${item}
done
if [[ -n "${output_dir}" ]]; then
  mkdir -p -- "${output_dir}"
  printf '{"successful":true}\\n' > "${output_dir}/fake-result.json"
  if [[ "${probe_failed}" == true ]]; then
    printf '%s\\n' '{"successful":false,"scenarios":[{"scenario_id":"a","status":"success"},{"scenario_id":"b","status":"failure","error_category":"VLAAssessmentPayloadError"}]}' > "${output_dir}/summary.json"
  else
    printf '%s\\n' '{"successful":true,"scenarios":[{"scenario_id":"a","status":"success"},{"scenario_id":"b","status":"success"}]}' > "${output_dir}/summary.json"
  fi
fi
if [[ "${probe_failed}" == true ]]; then
  exit 1
fi
if [[ -n "${event_log}" ]]; then
  mkdir -p -- "$(dirname -- "${event_log}")"
  printf '{"event":"fake"}\\n' > "${event_log}"
fi
if [[ "$*" == *"demo-vla-qwen3-vl-2b-llama.yaml"* ]]; then
  printf '%s\\n' '{"simulation_time_s":2.0,"crashed":false,"went_off_road":false,"vla_metrics":{"provider_requests_attempted":3,"responses_succeeded":2,"authority_steps":7,"runtime_errors":0}}'
elif [[ "$*" == *"demo-vla-smolvlm-llama.yaml"* ]]; then
  printf '%s\\n' '{"simulation_time_s":2.0,"crashed":false,"went_off_road":false,"vla_metrics":{"provider_requests_attempted":3,"responses_succeeded":3,"authority_steps":0,"runtime_errors":0}}'
elif [[ "$*" == *" replay "* ]]; then
  printf '%s\\n' '{"successful":true,"mismatches":[]}'
else
  printf '%s\\n' '{"successful":true}'
fi
exit 0
""",
    )

    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["FAKE_SERVER_MARKER"] = str(marker)
    environment["NATIVE_PLATFORM_OS_RELEASE"] = str(os_release)
    environment["NATIVE_PLATFORM_PROC_VERSION"] = str(proc_version)
    environment["NATIVE_PLATFORM_CGROUP"] = str(cgroup)
    environment["NATIVE_PLATFORM_CONTAINER_MARKER"] = str(tmp_path / "no-container")
    environment["NATIVE_PLATFORM_GNU_TIME"] = str(fake_bin / "time")
    environment["NATIVE_PLATFORM_DISPLAY"] = ":197"

    command = [
        str(scripts / NATIVE_CHECK.name),
        "--llama-archive",
        f"tmp/native-vlm/{LLAMA_ARCHIVE_NAME}",
        "--llama-server",
        "tmp/native-vlm/llama-server",
        "--model",
        f"tmp/native-vlm/{MODEL_NAME}",
        "--mmproj",
        f"tmp/native-vlm/{MMPROJ_NAME}",
        "--output-dir",
        RUN_DIR,
    ]
    return project, command, environment, llama_server


def _run(
    command: list[str],
    project: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _statuses(project: Path) -> tuple[dict[str, object], dict[str, str]]:
    report = json.loads((project / RUN_DIR / "gates.json").read_text(encoding="utf-8"))
    return report, {gate["id"]: gate["status"] for gate in report["gates"]}


def test_native_platform_check_records_two_successful_fake_vlm_rounds(
    tmp_path: Path,
) -> None:
    project, command, environment, llama_server = _fake_native_project(tmp_path)
    result = _run(command, project, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    report, statuses = _statuses(project)
    assert report["successful"] is True
    assert report["candidate"]["llama_release"] == "b10771"
    assert report["candidate"]["llama_variant"] == "cpu"
    assert report["render_path"] == "xvfb"
    assert statuses["n0-platform"] == "passed"
    assert statuses["n2-render"] == "passed"
    assert statuses["n2-tests"] == "passed"
    assert statuses["n2-gates"] == "passed"
    assert statuses["n4-gates"] == "passed"
    assert statuses["n7-smolvlm"] == "not_run"
    # Every command a team runs drives the shipped submission.
    for gate in ("n2-smoke", "n2-headless", "n2-gates", "n4-fixture", "n4-gates"):
        recorded = (project / RUN_DIR / "commands" / f"{gate}.txt").read_text()
        assert "--submission submission" in recorded, gate
    assert statuses["n3-probe-capture"] == "passed"
    assert statuses["n4-replay"] == "passed"
    assert statuses["n5-round-1"] == "passed"
    assert statuses["n5-round-2"] == "passed"
    assert statuses["n3-gui"] == "not_run"
    host = json.loads(
        (
            project / "artifacts/native-platform-validation/native-test-run/host.json"
        ).read_text(encoding="utf-8")
    )
    assert host["platform"] == "Linux"
    assert host["architecture"] == "x86_64"
    assert host["llama_release"] == "b10771"
    assert host["llama_version"] == (
        "version: 0.3.0-dev (build 10771, commit 733905474)\n"
        "built with GNU 11.4.0 for Linux x86_64"
    )
    assert host["llama_archive_sha256"] == LLAMA_ARCHIVE_SHA256
    assert host["model_sha256"] == MODEL_SHA256
    assert host["mmproj_sha256"] == MMPROJ_SHA256
    assert host["llama_variant"] == "cpu"
    assert host["gpu_driver"] == "580.178.04"
    for round_number in (1, 2):
        server = (
            project / RUN_DIR / "model" / f"round-{round_number}" / "server-command.txt"
        ).read_text()
        assert "--n-gpu-layers 0 --no-mmproj-offload" in server
        acceptance = json.loads(
            (
                project
                / "artifacts/native-platform-validation/native-test-run"
                / f"model/round-{round_number}/acceptance.json"
            ).read_text(encoding="utf-8")
        )
        assert acceptance["successful"] is True
        assert acceptance["authority_steps"] == 7

    repeated = subprocess.run(
        command,
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert repeated.returncode == 2
    assert "output already exists" in repeated.stderr

    llama_server.write_text(
        llama_server.read_text(encoding="utf-8") + "# changed after extraction\n",
        encoding="utf-8",
    )
    mismatch_command = [
        *command[:-1],
        "artifacts/native-platform-validation/server-mismatch",
    ]
    mismatch = subprocess.run(
        mismatch_command,
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert mismatch.returncode == 1
    mismatch_report = json.loads(
        (
            project
            / "artifacts/native-platform-validation/server-mismatch/gates.json"
        ).read_text(encoding="utf-8")
    )
    mismatch_statuses = {
        gate["id"]: gate["status"] for gate in mismatch_report["gates"]
    }
    assert mismatch_statuses["n0-platform"] == "failed"
    assert mismatch_statuses["n1-sync"] == "not_run"
    assert "does not match the executable" in (
        project
        / "artifacts/native-platform-validation/server-mismatch/logs/n0-platform.log"
    ).read_text(encoding="utf-8")


def test_native_platform_check_skips_the_fixture_gates_in_a_release_without_fixtures(
    tmp_path: Path,
) -> None:
    # Student releases hold fixtures/ back until Assignment 2.
    project, command, environment, _ = _fake_native_project(tmp_path)
    (project / "fixtures").rmdir()
    result = _run(command, project, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    report, statuses = _statuses(project)
    assert report["successful"] is True
    assert statuses["n2-gates"] == "passed"
    for gate in ("n4-fixture", "n4-replay", "n4-gates"):
        assert statuses[gate] == "not_run"
        reason = (project / RUN_DIR / "logs" / f"{gate}.log").read_text()
        assert "does not include fixtures/" in reason
    assert statuses["n5-round-1"] == "passed"


def test_native_platform_check_runs_the_smolvlm_experiment_as_a_team_would(
    tmp_path: Path,
) -> None:
    project, command, environment, _ = _fake_native_project(tmp_path)
    command = [
        *command[:-2],
        "--smolvlm-model",
        f"tmp/native-vlm/{SMOLVLM_MODEL_NAME}",
        "--smolvlm-mmproj",
        f"tmp/native-vlm/{SMOLVLM_MMPROJ_NAME}",
        *command[-2:],
    ]

    result = _run(command, project, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    report, statuses = _statuses(project)
    assert report["successful"] is True
    assert statuses["n7-smolvlm"] == "passed"
    round_dir = project / RUN_DIR / "model" / "smolvlm"
    server = (round_dir / "server-command.txt").read_text()
    assert SMOLVLM_MODEL_NAME in server and "--image-min-tokens" not in server
    assert "--submission submission" in (round_dir / "loop-command.txt").read_text()
    acceptance = json.loads((round_dir / "acceptance.json").read_text())
    assert acceptance["successful"] is True
    assert acceptance["authority_steps"] == 0
    assert acceptance["authority_required"] is False


def test_failed_smolvlm_experiment_never_fails_the_platform_check(tmp_path: Path) -> None:
    project, command, environment, _ = _fake_native_project(tmp_path)
    environment["FAKE_SMOLVLM_MMPROJ_SHA256"] = "0" * 64
    command = [
        *command[:-2],
        "--smolvlm-model",
        f"tmp/native-vlm/{SMOLVLM_MODEL_NAME}",
        "--smolvlm-mmproj",
        f"tmp/native-vlm/{SMOLVLM_MMPROJ_NAME}",
        *command[-2:],
    ]

    result = _run(command, project, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    report, statuses = _statuses(project)
    assert report["successful"] is True
    assert statuses["n7-smolvlm"] == "failed"
    assert "SmolVLM projector SHA-256 does not match" in (
        project / RUN_DIR / "logs" / "n7-smolvlm.log"
    ).read_text()


def test_smolvlm_experiment_runs_when_this_machine_cannot_run_qwen(
    tmp_path: Path,
) -> None:
    project, command, environment, _ = _fake_native_project(tmp_path)
    environment["FAKE_QWEN_STRUCTURED_FAILS"] = "1"
    command = [
        *command[:-2],
        "--smolvlm-model",
        f"tmp/native-vlm/{SMOLVLM_MODEL_NAME}",
        "--smolvlm-mmproj",
        f"tmp/native-vlm/{SMOLVLM_MMPROJ_NAME}",
        *command[-2:],
    ]

    result = _run(command, project, environment)

    assert result.returncode == 1
    report, statuses = _statuses(project)
    assert report["successful"] is False
    assert statuses["n5-round-1"] == "failed"
    assert statuses["n5-round-2"] == "not_run"
    assert statuses["n7-smolvlm"] == "passed"


def test_schema_off_decode_failures_are_evidence_and_never_fail_a_round(
    tmp_path: Path,
) -> None:
    project, command, environment, _ = _fake_native_project(tmp_path)
    environment["FAKE_QWEN_PROBE_FAILS"] = "1"

    result = _run(command, project, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    report, statuses = _statuses(project)
    assert report["successful"] is True
    assert statuses["n5-round-1"] == "passed"
    assert statuses["n5-round-2"] == "passed"
    for round_number in (1, 2):
        round_dir = project / RUN_DIR / "model" / f"round-{round_number}"
        acceptance = json.loads((round_dir / "acceptance.json").read_text())
        assert acceptance["successful"] is True
        assert acceptance["schema_off_probe"] == {
            "scenes": 2,
            "decoded": 1,
            "failure_categories": {"VLAAssessmentPayloadError": 1},
        }
        assert (round_dir / "probe-structured.log").exists()
    assert "schema-off probe exited 1; recorded as evidence only" in (
        project / RUN_DIR / "logs" / "n5-round-1.log"
    ).read_text()


def test_native_platform_check_needs_both_smolvlm_files(tmp_path: Path) -> None:
    project, command, environment, _ = _fake_native_project(tmp_path)

    result = _run(
        [*command, "--smolvlm-model", f"tmp/native-vlm/{SMOLVLM_MODEL_NAME}"],
        project,
        environment,
    )

    assert result.returncode == 2
    assert "--smolvlm-mmproj must be a non-empty path" in result.stderr


@pytest.mark.parametrize(
    ("xvfb", "render_fails", "render_path", "tests_display"),
    [
        ("Xvfb", "", "xvfb", ":197"),
        # The grading server: no Xvfb, and MetaDrive renders through EGL.
        ("drivebench-no-xvfb", "", "headless", ""),
        ("Xvfb", "xvfb", "headless", ""),
    ],
)
def test_native_platform_check_renders_on_xvfb_when_it_draws_and_headless_otherwise(
    tmp_path: Path,
    xvfb: str,
    render_fails: str,
    render_path: str,
    tests_display: str,
) -> None:
    project, command, environment, _ = _fake_native_project(tmp_path)
    environment["NATIVE_PLATFORM_XVFB"] = xvfb
    environment["FAKE_RENDER_FAILS"] = render_fails
    # A desktop session's display is never the one the evidence renders on.
    environment["DISPLAY"] = ":0"

    result = _run(command, project, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    report, statuses = _statuses(project)
    assert report["successful"] is True
    assert report["render_path"] == render_path
    assert statuses["n2-render"] == "passed"
    assert statuses["n5-round-2"] == "passed"
    record = json.loads((project / RUN_DIR / "render" / f"{render_path}.json").read_text())
    assert record["valid"] is True
    assert record["display"] == tests_display
    assert f"DISPLAY={tests_display}\n" in (project / RUN_DIR / "logs" / "n2-tests.log").read_text()


def test_native_platform_check_refuses_when_neither_render_path_draws_a_frame(
    tmp_path: Path,
) -> None:
    project, command, environment, _ = _fake_native_project(tmp_path)
    environment["FAKE_RENDER_FAILS"] = "xvfb headless"

    result = _run(command, project, environment)

    assert result.returncode == 1
    report, statuses = _statuses(project)
    assert report["successful"] is False
    assert report["render_path"] == "none"
    assert statuses["n2-render"] == "failed"
    assert statuses["n2-tests"] == "not_run"
    assert "neither a private Xvfb display nor headless rendering" in (
        project / RUN_DIR / "logs" / "n2-render.log"
    ).read_text()
    for render_path in ("xvfb", "headless"):
        record = json.loads((project / RUN_DIR / "render" / f"{render_path}.json").read_text())
        assert record["valid"] is False


def _vulkan_command(command: list[str]) -> list[str]:
    archive = command.index("--llama-archive") + 1
    return [
        *command[:archive],
        f"tmp/native-vlm/{VULKAN_ARCHIVE_NAME}",
        *command[archive + 1 : -2],
        "--vulkan",
        *command[-2:],
    ]


def test_native_platform_check_validates_the_vulkan_build_on_the_gpu(
    tmp_path: Path,
) -> None:
    project, command, environment, _ = _fake_native_project(tmp_path)
    environment["FAKE_LLAMA_DEVICES"] = VULKAN_DEVICES
    command = _vulkan_command(command)
    command[-2:-2] = [
        "--smolvlm-model",
        f"tmp/native-vlm/{SMOLVLM_MODEL_NAME}",
        "--smolvlm-mmproj",
        f"tmp/native-vlm/{SMOLVLM_MMPROJ_NAME}",
    ]

    result = _run(command, project, environment)

    assert result.returncode == 0, result.stdout + result.stderr
    report, statuses = _statuses(project)
    assert report["successful"] is True
    assert report["candidate"]["llama_variant"] == "vulkan"
    assert statuses["n7-smolvlm"] == "passed"
    for round_name in ("round-1", "round-2", "smolvlm"):
        server = (project / RUN_DIR / "model" / round_name / "server-command.txt").read_text()
        assert "--n-gpu-layers 99" in server, round_name
        assert "--no-mmproj-offload" not in server, round_name
    host = json.loads((project / RUN_DIR / "host.json").read_text())
    assert host["llama_variant"] == "vulkan"
    assert host["llama_archive_path"] == f"tmp/native-vlm/{VULKAN_ARCHIVE_NAME}"
    assert host["llama_archive_sha256"] == VULKAN_ARCHIVE_SHA256
    assert host["gpu"] == "NVIDIA RTX PRO 6000 Blackwell, 97887 MiB"
    assert host["gpu_driver"] == "580.178.04"
    assert "Vulkan0: NVIDIA RTX PRO 6000 Blackwell" in host["llama_devices"]


@pytest.mark.parametrize(
    ("archive", "devices", "refusal"),
    [
        (LLAMA_ARCHIVE_NAME, VULKAN_DEVICES, f"must be the accepted file: {VULKAN_ARCHIVE_NAME}"),
        (VULKAN_ARCHIVE_NAME, "Available devices:", "the Vulkan build lists no Vulkan device"),
    ],
)
def test_vulkan_validation_refuses_another_archive_or_a_build_without_a_gpu(
    tmp_path: Path,
    archive: str,
    devices: str,
    refusal: str,
) -> None:
    project, command, environment, _ = _fake_native_project(tmp_path)
    environment["FAKE_LLAMA_DEVICES"] = devices
    command = _vulkan_command(command)
    command[command.index("--llama-archive") + 1] = f"tmp/native-vlm/{archive}"

    result = _run(command, project, environment)

    assert result.returncode == 1
    _, statuses = _statuses(project)
    assert statuses["n0-platform"] == "failed"
    assert statuses["n1-sync"] == "not_run"
    assert refusal in (project / RUN_DIR / "logs" / "n0-platform.log").read_text()
