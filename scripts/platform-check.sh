#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_REL="artifacts/platform-validation/$(date -u +%Y%m%dT%H%M%SZ)"
SKIP_BUILD=false

usage() {
  cat <<'EOF'
Usage: scripts/platform-check.sh [--output-dir RELATIVE_PATH] [--skip-build]

Run DriveBench's automated Docker-on-Linux platform gates:
  G0   Linux/Docker/Compose inventory and configuration
  G1   image build and frozen lock validation
  G2   tests, dry-run smoke, a real headless route run, and the public
       Assignment 1 gates, all with the shipped submission
  G3a  offscreen RGB probe catalog under a private Xvfb display
  G4   fixture-backed VLA closed loop, decision replay, and the public gates
       through the shipped observation.py; not run without fixtures/

G3b rendered GUI/input, live local VLM, and live cloud checks remain explicit
separate gates. The command writes a new non-secret evidence bundle and never
overwrites an existing one.
EOF
}

die() {
  printf 'platform-check: %s\n' "$*" >&2
  exit 2
}

while (($# > 0)); do
  case "$1" in
    --output-dir)
      (($# >= 2)) || die "--output-dir requires a value"
      OUTPUT_REL="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "${OUTPUT_REL}" in
  ""|/*|..|../*|*/../*|*/..)
    die "--output-dir must be a non-empty path inside the repository without '..'"
    ;;
esac

cd -- "${REPO_ROOT}"

OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_REL}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  die "output already exists: ${OUTPUT_REL}"
fi

mkdir -p -- "${OUTPUT_DIR}/commands" "${OUTPUT_DIR}/logs"
RESULTS_NDJSON="${OUTPUT_DIR}/gates.ndjson"
: > "${RESULTS_NDJSON}"

json_escape() {
  local value=${1-}
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  value=${value//$'\r'/\\r}
  value=${value//$'\t'/\\t}
  printf '%s' "${value}"
}

command_text() {
  local item
  printf '%q' "$1"
  shift
  for item in "$@"; do
    printf ' %q' "${item}"
  done
  printf '\n'
}

record_gate() {
  local gate_id=$1
  local description=$2
  local status=$3
  local duration_s=$4
  local command_path=$5
  local log_path=$6

  printf '{"id":"%s","description":"%s","status":"%s","duration_s":%s,"command_file":"%s","log_file":"%s"}\n' \
    "$(json_escape "${gate_id}")" \
    "$(json_escape "${description}")" \
    "$(json_escape "${status}")" \
    "${duration_s}" \
    "$(json_escape "${command_path}")" \
    "$(json_escape "${log_path}")" \
    >> "${RESULTS_NDJSON}"
}

run_gate() {
  local gate_id=$1
  local description=$2
  shift 2

  local command_rel="commands/${gate_id}.txt"
  local log_rel="logs/${gate_id}.log"
  local started_at
  local finished_at
  local status
  local exit_code

  command_text "$@" > "${OUTPUT_DIR}/${command_rel}"
  printf '[%s] %s ... ' "${gate_id}" "${description}"
  started_at=$(date +%s)
  if "$@" > "${OUTPUT_DIR}/${log_rel}" 2>&1; then
    exit_code=0
    status=passed
    printf 'passed\n'
  else
    exit_code=$?
    status=failed
    printf 'failed (exit %s)\n' "${exit_code}"
  fi
  finished_at=$(date +%s)
  record_gate \
    "${gate_id}" \
    "${description}" \
    "${status}" \
    "$((finished_at - started_at))" \
    "${command_rel}" \
    "${log_rel}"
  return "${exit_code}"
}

record_not_run() {
  local gate_id=$1
  local description=$2
  local reason=$3
  local command_rel="commands/${gate_id}.txt"
  local log_rel="logs/${gate_id}.log"

  printf '%s\n' "${reason}" > "${OUTPUT_DIR}/${command_rel}"
  printf '%s\n' "${reason}" > "${OUTPUT_DIR}/${log_rel}"
  record_gate "${gate_id}" "${description}" not_run 0 "${command_rel}" "${log_rel}"
}

sha256_file() {
  local path=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "${path}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -- "${path}" | awk '{print $1}'
  else
    printf 'unavailable'
  fi
}

value_or_unavailable() {
  local value
  value=$("$@" 2>/dev/null) || value=unavailable
  [[ -n "${value}" ]] || value=unavailable
  printf '%s' "${value}"
}

write_host_inventory() {
  local distro=unavailable
  local wsl=false
  local git_commit
  local git_dirty=false

  if [[ -r /etc/os-release ]]; then
    distro=$(awk -F= '$1 == "PRETTY_NAME" {gsub(/^"|"$/, "", $2); print $2}' /etc/os-release)
    [[ -n "${distro}" ]] || distro=unavailable
  fi
  if grep -qi microsoft /proc/version 2>/dev/null; then
    wsl=true
  fi
  if ! git_commit=$(git rev-parse --verify HEAD 2>/dev/null); then
    git_commit=unavailable
  fi
  if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    git_dirty=true
  fi

  cat > "${OUTPUT_DIR}/host.json" <<EOF
{
  "schema_version": 1,
  "platform": "$(json_escape "$(value_or_unavailable uname -s)")",
  "distribution": "$(json_escape "${distro}")",
  "kernel": "$(json_escape "$(value_or_unavailable uname -r)")",
  "architecture": "$(json_escape "$(value_or_unavailable uname -m)")",
  "wsl": ${wsl},
  "cpu": "$(json_escape "$(value_or_unavailable sh -c "command -v lscpu >/dev/null 2>&1 && lscpu | awk -F: '/Model name/ {sub(/^[[:space:]]+/, \"\", \$2); print \$2; exit}'")")",
  "memory": "$(json_escape "$(value_or_unavailable sh -c "command -v free >/dev/null 2>&1 && free -h | awk '/^Mem:/ {print \$2}'")")",
  "gpu": "$(json_escape "$(value_or_unavailable sh -c "if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';' -; elif command -v lspci >/dev/null 2>&1; then lspci | grep -Ei 'VGA|3D|Display' | paste -sd ';' -; fi")")",
  "docker_client": "$(json_escape "$(value_or_unavailable docker --version)")",
  "docker_server": "$(json_escape "$(value_or_unavailable docker version --format '{{.Server.Version}}')")",
  "docker_compose": "$(json_escape "$(value_or_unavailable docker compose version --short)")",
  "git_commit": "$(json_escape "${git_commit}")",
  "git_dirty": ${git_dirty},
  "uv_lock_sha256": "$(json_escape "$(sha256_file uv.lock)")"
}
EOF
}

check_docker_linux() {
  [[ "$(uname -s)" == Linux ]] || {
    printf 'required host kernel: Linux; found: %s\n' "$(uname -s)"
    return 1
  }
  command -v docker >/dev/null 2>&1 || {
    printf 'docker CLI not found\n'
    return 1
  }
  docker info >/dev/null
  docker compose version
  docker compose config -q
}

finalize_summary() {
  local successful=$1
  {
    printf '{\n  "schema_version": 1,\n  "successful": %s,\n  "gates": [\n' "${successful}"
    awk 'NR > 1 {printf ",\n"} {printf "    %s", $0} END {printf "\n"}' "${RESULTS_NDJSON}"
    printf '  ]\n}\n'
  } > "${OUTPUT_DIR}/gates.json"
  rm -- "${RESULTS_NDJSON}"
}

write_host_inventory

overall_success=true
if ! run_gate g0-platform "Linux, Docker, Compose, and project configuration" check_docker_linux; then
  overall_success=false
  record_not_run g1-build "Build simulator image" "blocked by g0-platform"
  record_not_run g1-lock "Validate frozen dependency lock" "blocked by g0-platform"
  record_not_run g1-assets "Verify bundled MetaDrive assets" "blocked by g0-platform"
  record_not_run g2-tests "Run complete test suite" "blocked by g0-platform"
  record_not_run g2-smoke "Run CLI smoke wiring" "blocked by g0-platform"
  record_not_run g2-headless "Run real headless route" "blocked by g0-platform"
  record_not_run g2-gates "Run the public Assignment 1 gates" "blocked by g0-platform"
  record_not_run g3a-camera "Capture offscreen RGB catalog" "blocked by g0-platform"
  record_not_run g4-fixture "Run fixture-backed VLA loop" "blocked by g0-platform"
  record_not_run g4-replay "Replay fixture-backed decisions" "blocked by g0-platform"
  record_not_run g4-gates "Run the public gates through the shipped observation" \
    "blocked by g0-platform"
else
  if [[ "${SKIP_BUILD}" == true ]]; then
    record_not_run g1-build "Build simulator image" "skipped by --skip-build"
  elif ! run_gate g1-build "Build simulator image" docker compose build metadrive; then
    overall_success=false
  fi

  if ! run_gate g1-lock "Validate frozen dependency lock" \
    docker compose run --rm --entrypoint uv metadrive lock --check; then
    overall_success=false
  fi
  if ! run_gate g1-assets "Verify bundled MetaDrive assets" \
    docker compose run --rm metadrive python -c \
      'from metadrive.version import asset_version; assert asset_version() == "0.4.3"; print(asset_version())'; then
    overall_success=false
  fi
  if ! run_gate g2-tests "Run complete test suite" \
    docker compose run --rm \
      --entrypoint bash \
      -e DISPLAY=:97 \
      -e PLATFORM_OUTPUT_REL="${OUTPUT_REL}" \
      metadrive -c '
        set -euo pipefail
        Xvfb :97 -screen 0 1280x720x24 -nolisten tcp >"/app/${PLATFORM_OUTPUT_REL}/logs/g2-tests-xvfb.log" 2>&1 &
        xvfb_pid=$!
        trap '\''kill "${xvfb_pid}" 2>/dev/null || true; wait "${xvfb_pid}" 2>/dev/null || true'\'' EXIT
        sleep 1
        python -m pytest
      '; then
    overall_success=false
  fi
  if ! run_gate g2-smoke "Run CLI smoke wiring" \
    docker compose run --rm metadrive metadrive-starter smoke --dry-run --submission submission; then
    overall_success=false
  fi
  if ! run_gate g2-headless "Run real headless route" \
    docker compose run --rm metadrive metadrive-starter run \
      --config configs/demo-autopilot.yaml --submission submission --headless --steps 500; then
    overall_success=false
  fi
  if ! run_gate g2-gates "Run the public Assignment 1 gates" \
    docker compose run --rm metadrive metadrive-starter gates --submission submission --json; then
    overall_success=false
  fi

  mkdir -p -- "${OUTPUT_DIR}/camera"
  if ! run_gate g3a-camera "Capture offscreen RGB catalog" \
    docker compose run --rm \
      --entrypoint bash \
      -e DISPLAY=:99 \
      -e PLATFORM_OUTPUT_REL="${OUTPUT_REL}" \
      metadrive -c '
        set -euo pipefail
        Xvfb :99 -screen 0 1280x720x24 -nolisten tcp >"/app/${PLATFORM_OUTPUT_REL}/camera/xvfb.log" 2>&1 &
        xvfb_pid=$!
        trap '\''kill "${xvfb_pid}" 2>/dev/null || true; wait "${xvfb_pid}" 2>/dev/null || true'\'' EXIT
        sleep 1
        metadrive-starter probe --output-dir "/app/${PLATFORM_OUTPUT_REL}/camera/probe"
      '; then
    overall_success=false
  fi

  # Student releases include fixtures/ only from Assignment 2 on, together
  # with the observation.py these gates drive through.
  if [[ ! -d fixtures ]]; then
    record_not_run g4-fixture "Run fixture-backed VLA loop" \
      "skipped: this release does not include fixtures/"
    record_not_run g4-replay "Replay fixture-backed decisions" \
      "skipped: this release does not include fixtures/"
    record_not_run g4-gates "Run the public gates through the shipped observation" \
      "skipped: this release does not include fixtures/"
  else
    mkdir -p -- "${OUTPUT_DIR}/fixture"
    if ! run_gate g4-fixture "Run fixture-backed VLA loop" \
      docker compose run --rm \
        --entrypoint bash \
        -e DISPLAY=:98 \
        -e PLATFORM_OUTPUT_REL="${OUTPUT_REL}" \
        metadrive -c '
          set -euo pipefail
          Xvfb :98 -screen 0 1280x720x24 -nolisten tcp >"/app/${PLATFORM_OUTPUT_REL}/fixture/xvfb.log" 2>&1 &
          xvfb_pid=$!
          trap '\''kill "${xvfb_pid}" 2>/dev/null || true; wait "${xvfb_pid}" 2>/dev/null || true'\'' EXIT
          sleep 1
          metadrive-starter run \
            --config configs/demo-vla-fixture.yaml \
            --submission submission \
            --headless \
            --steps 100 \
            --event-log "/app/${PLATFORM_OUTPUT_REL}/fixture/events.jsonl"
        '; then
      overall_success=false
    fi
    if [[ -s "${OUTPUT_DIR}/fixture/events.jsonl" ]]; then
      if ! run_gate g4-replay "Replay fixture-backed decisions" \
        docker compose run --rm metadrive metadrive-starter replay \
          --event-log "${OUTPUT_REL}/fixture/events.jsonl"; then
        overall_success=false
      fi
    else
      overall_success=false
      record_not_run g4-replay "Replay fixture-backed decisions" \
        "blocked: g4-fixture did not produce fixture/events.jsonl"
    fi
    if ! run_gate g4-gates "Run the public gates through the shipped observation" \
      docker compose run --rm \
        --entrypoint bash \
        -e DISPLAY=:96 \
        -e PLATFORM_OUTPUT_REL="${OUTPUT_REL}" \
        metadrive -c '
          set -euo pipefail
          Xvfb :96 -screen 0 1280x720x24 -nolisten tcp >"/app/${PLATFORM_OUTPUT_REL}/logs/g4-gates-xvfb.log" 2>&1 &
          xvfb_pid=$!
          trap '\''kill "${xvfb_pid}" 2>/dev/null || true; wait "${xvfb_pid}" 2>/dev/null || true'\'' EXIT
          sleep 1
          metadrive-starter gates --submission submission \
            --config configs/demo-vla-fixture.yaml --json
        '; then
      overall_success=false
    fi
  fi
fi

record_not_run g3a-visual-review "Offscreen RGB visual review" \
  "manual gate: inspect camera/probe images for orientation, colour, and scene content"
record_not_run g3b-gui "Rendered window and keyboard input" \
  "manual gate: follow the rendered Linux/WSLg command in README.md"
record_not_run g5-local-vlm "Live local VLM" \
  "explicit gate: start the selected isolated server and run fixed probe plus closed loop"
record_not_run g6-cloud-vlm "Live cloud VLM" \
  "explicit billable gate: use reviewed fixtures, then staff-controlled capped execution"

finalize_summary "${overall_success}"

printf 'Evidence: %s\n' "${OUTPUT_REL}"
if [[ "${overall_success}" != true ]]; then
  exit 1
fi
