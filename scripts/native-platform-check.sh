#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_REL="artifacts/native-platform-validation/$(date -u +%Y%m%dT%H%M%SZ)"
LLAMA_ARCHIVE_REL=""
LLAMA_SERVER_REL=""
MODEL_REL=""
MMPROJ_REL=""
SMOLVLM_MODEL_REL=""
SMOLVLM_MMPROJ_REL=""

LLAMA_RELEASE="b10771"
LLAMA_BUILD="10771"
LLAMA_ARCHIVE_FILENAME="llama-b10771-bin-ubuntu-x64.tar.gz"
LLAMA_ARCHIVE_SHA256="42bb60d6027c99ec05ff1a5ae441e45345a3fcccaf86ca655b4dc51b5519c814"
MODEL_REVISION="d38d39f5972e27cd58023f9b1e9f994b0c85ca47"
MODEL_FILENAME="Qwen3VL-2B-Instruct-Q4_K_M.gguf"
MODEL_SHA256="089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"
MMPROJ_FILENAME="mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
MMPROJ_SHA256="f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82"
MODEL_ALIAS="drivebench-qwen3-vl-2b-q4-k-m"
SERVER_PORT=8000

# The optional Assignment 2 planning experiment: the files of the 3 September
# Linux container trial, ggml-org/SmolVLM-256M-Instruct-GGUF at this revision.
SMOLVLM_REVISION="b9e4379657e1450d04d02eec8e345667265b0a00"
SMOLVLM_MODEL_FILENAME="SmolVLM-256M-Instruct-Q8_0.gguf"
SMOLVLM_MODEL_SHA256="2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65"
SMOLVLM_MMPROJ_FILENAME="mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
SMOLVLM_MMPROJ_SHA256="7e943f7c53f0382a6fc41b6ee0c2def63ba4fded9ab8ed039cc9e2ab905e0edd"
SMOLVLM_ALIAS="drivebench-smolvlm-256m-q8-0"

OS_RELEASE_PATH="${NATIVE_PLATFORM_OS_RELEASE:-/etc/os-release}"
PROC_VERSION_PATH="${NATIVE_PLATFORM_PROC_VERSION:-/proc/version}"
CGROUP_PATH="${NATIVE_PLATFORM_CGROUP:-/proc/1/cgroup}"
CONTAINER_MARKER="${NATIVE_PLATFORM_CONTAINER_MARKER:-/.dockerenv}"
GNU_TIME="${NATIVE_PLATFORM_GNU_TIME:-/usr/bin/time}"
XVFB_DISPLAY="${NATIVE_PLATFORM_DISPLAY:-:97}"

SERVER_PID=""
SAMPLER_PID=""
XVFB_PID=""
LLAMA_ARCHIVE_ACTUAL_SHA256="unavailable"
LLAMA_SERVER_ACTUAL_SHA256="unavailable"
MODEL_ACTUAL_SHA256="unavailable"
MMPROJ_ACTUAL_SHA256="unavailable"

usage() {
  cat <<'EOF'
Usage: scripts/native-platform-check.sh \
  --llama-archive RELATIVE_PATH \
  --llama-server RELATIVE_PATH \
  --model RELATIVE_PATH \
  --mmproj RELATIVE_PATH \
  [--smolvlm-model RELATIVE_PATH --smolvlm-mmproj RELATIVE_PATH] \
  [--output-dir RELATIVE_PATH]

Run the pinned native Ubuntu x86_64 VLM acceptance sequence:
  N0  native Ubuntu/runtime/model identity and clean-checkout checks
  N1  frozen uv environment and lock validation
  N2  tests, dry-run smoke, a real native headless route, and the public
      Assignment 1 gates, all with the shipped submission as a team runs them
  N3  one fixed offscreen RGB probe capture under private Xvfb
  N4  fixture-backed VLA closed loop, deterministic replay, and the public
      gates through the shipped observation.py; not run without fixtures/
  N5  two clean llama-server rounds, each with schema-off/on probe replay
      (schema-off as evidence only), paced model authority, zero-mismatch
      replay, and resource evidence
  N7  optional, once N0-N4 pass, whatever N5's result: the Assignment 2
      planning experiment, SmolVLM-256M under the same llama-server with JSON
      Schema output. Its result is recorded but never changes the overall result.

The accepted candidate is llama.cpp b10771 with Qwen3-VL-2B-Instruct Q4_K_M
and its Q8_0 multimodal projector at the hashes documented in
docs/model-providers.md. Inputs must be paths inside the repository (normally
under ignored tmp/native-vlm/). The evidence directory is new and is never
overwritten. This command does not install system packages or download models.
EOF
}

die() {
  printf 'native-platform-check: %s\n' "$*" >&2
  exit 2
}

validate_relative_path() {
  local label=$1
  local value=$2
  case "${value}" in
    ""|/*|..|../*|*/../*|*/..)
      die "${label} must be a non-empty path inside the repository without '..'"
      ;;
  esac
}

while (($# > 0)); do
  case "$1" in
    --llama-archive)
      (($# >= 2)) || die "--llama-archive requires a value"
      LLAMA_ARCHIVE_REL=$2
      shift 2
      ;;
    --llama-server)
      (($# >= 2)) || die "--llama-server requires a value"
      LLAMA_SERVER_REL=$2
      shift 2
      ;;
    --model)
      (($# >= 2)) || die "--model requires a value"
      MODEL_REL=$2
      shift 2
      ;;
    --mmproj)
      (($# >= 2)) || die "--mmproj requires a value"
      MMPROJ_REL=$2
      shift 2
      ;;
    --smolvlm-model)
      (($# >= 2)) || die "--smolvlm-model requires a value"
      SMOLVLM_MODEL_REL=$2
      shift 2
      ;;
    --smolvlm-mmproj)
      (($# >= 2)) || die "--smolvlm-mmproj requires a value"
      SMOLVLM_MMPROJ_REL=$2
      shift 2
      ;;
    --output-dir)
      (($# >= 2)) || die "--output-dir requires a value"
      OUTPUT_REL=$2
      shift 2
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

validate_relative_path "--llama-archive" "${LLAMA_ARCHIVE_REL}"
validate_relative_path "--llama-server" "${LLAMA_SERVER_REL}"
validate_relative_path "--model" "${MODEL_REL}"
validate_relative_path "--mmproj" "${MMPROJ_REL}"
if [[ -n "${SMOLVLM_MODEL_REL}${SMOLVLM_MMPROJ_REL}" ]]; then
  validate_relative_path "--smolvlm-model" "${SMOLVLM_MODEL_REL}"
  validate_relative_path "--smolvlm-mmproj" "${SMOLVLM_MMPROJ_REL}"
fi
validate_relative_path "--output-dir" "${OUTPUT_REL}"
case "${OUTPUT_REL}" in
  artifacts/native-platform-validation/*) ;;
  *) die "--output-dir must be below artifacts/native-platform-validation/" ;;
esac

cd -- "${REPO_ROOT}"

LLAMA_ARCHIVE_PATH="${REPO_ROOT}/${LLAMA_ARCHIVE_REL}"
LLAMA_SERVER_PATH="${REPO_ROOT}/${LLAMA_SERVER_REL}"
MODEL_PATH="${REPO_ROOT}/${MODEL_REL}"
MMPROJ_PATH="${REPO_ROOT}/${MMPROJ_REL}"
SMOLVLM_MODEL_PATH="${REPO_ROOT}/${SMOLVLM_MODEL_REL}"
SMOLVLM_MMPROJ_PATH="${REPO_ROOT}/${SMOLVLM_MMPROJ_REL}"
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
  sha256sum -- "$1" | awk '{print $1}'
}

value_or_unavailable() {
  local value
  value=$("$@" 2>/dev/null) || value=unavailable
  [[ -n "${value}" ]] || value=unavailable
  printf '%s' "${value}"
}

os_release_value() {
  local key=$1
  [[ -r "${OS_RELEASE_PATH}" ]] || return 1
  awk -F= -v wanted="${key}" '$1 == wanted {gsub(/^"|"$/, "", $2); print $2; exit}' \
    "${OS_RELEASE_PATH}"
}

cpu_inventory() {
  command -v lscpu >/dev/null 2>&1 || return 1
  lscpu | awk -F: '/Model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}'
}

memory_inventory() {
  awk '/MemTotal/ {print $2; exit}' /proc/meminfo
}

gpu_inventory() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';' -
  elif command -v lspci >/dev/null 2>&1; then
    lspci | grep -Ei 'VGA|3D|Display' | paste -sd ';' -
  else
    return 1
  fi
}

verify_server_matches_archive() {
  local members
  local member_count
  local archive_copy="${OUTPUT_DIR}/llama-server.archive-copy"

  members=$(tar -tzf "${LLAMA_ARCHIVE_PATH}" | awk '/(^|\/)llama-server$/') || return 1
  member_count=$(awk 'NF {count++} END {print count + 0}' <<<"${members}")
  if [[ "${member_count}" != 1 ]]; then
    printf 'accepted archive must contain exactly one llama-server; found: %s\n' \
      "${member_count}"
    return 1
  fi
  if ! tar -xOzf "${LLAMA_ARCHIVE_PATH}" "${members}" > "${archive_copy}"; then
    rm -f -- "${archive_copy}"
    return 1
  fi
  if [[ "$(sha256_file "${archive_copy}")" != "${LLAMA_SERVER_ACTUAL_SHA256}" ]]; then
    printf 'llama-server does not match the executable in the accepted archive\n'
    rm -f -- "${archive_copy}"
    return 1
  fi
  rm -f -- "${archive_copy}"
}

write_host_inventory() {
  local git_commit
  local git_dirty=false
  local llama_version=unavailable

  git_commit=$(value_or_unavailable git rev-parse --verify HEAD)
  if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    git_dirty=true
  fi
  if [[ -x "${LLAMA_SERVER_PATH}" ]]; then
    llama_version=$("${LLAMA_SERVER_PATH}" --version 2>&1) || llama_version=unavailable
    [[ -n "${llama_version}" ]] || llama_version=unavailable
  fi
  LLAMA_ARCHIVE_ACTUAL_SHA256=$(value_or_unavailable sha256_file "${LLAMA_ARCHIVE_PATH}")
  LLAMA_SERVER_ACTUAL_SHA256=$(value_or_unavailable sha256_file "${LLAMA_SERVER_PATH}")
  MODEL_ACTUAL_SHA256=$(value_or_unavailable sha256_file "${MODEL_PATH}")
  MMPROJ_ACTUAL_SHA256=$(value_or_unavailable sha256_file "${MMPROJ_PATH}")

  cat > "${OUTPUT_DIR}/host.json" <<EOF
{
  "schema_version": 1,
  "platform": "$(json_escape "$(value_or_unavailable uname -s)")",
  "distribution": "$(json_escape "$(value_or_unavailable os_release_value PRETTY_NAME)")",
  "kernel": "$(json_escape "$(value_or_unavailable uname -r)")",
  "architecture": "$(json_escape "$(value_or_unavailable uname -m)")",
  "cpu": "$(json_escape "$(value_or_unavailable cpu_inventory)")",
  "memory_kib": "$(json_escape "$(value_or_unavailable memory_inventory)")",
  "gpu": "$(json_escape "$(value_or_unavailable gpu_inventory)")",
  "uv": "$(json_escape "$(value_or_unavailable uv --version)")",
  "git_commit": "$(json_escape "${git_commit}")",
  "git_dirty": ${git_dirty},
  "uv_lock_sha256": "$(json_escape "$(value_or_unavailable sha256_file uv.lock)")",
  "llama_release": "${LLAMA_RELEASE}",
  "llama_version": "$(json_escape "${llama_version}")",
  "llama_archive_path": "$(json_escape "${LLAMA_ARCHIVE_REL}")",
  "llama_archive_sha256": "$(json_escape "${LLAMA_ARCHIVE_ACTUAL_SHA256}")",
  "llama_archive_size_bytes": "$(json_escape "$(value_or_unavailable stat -c %s "${LLAMA_ARCHIVE_PATH}")")",
  "llama_server_path": "$(json_escape "${LLAMA_SERVER_REL}")",
  "llama_server_sha256": "$(json_escape "${LLAMA_SERVER_ACTUAL_SHA256}")",
  "model_revision": "${MODEL_REVISION}",
  "model_path": "$(json_escape "${MODEL_REL}")",
  "model_sha256": "$(json_escape "${MODEL_ACTUAL_SHA256}")",
  "model_size_bytes": "$(json_escape "$(value_or_unavailable stat -c %s "${MODEL_PATH}")")",
  "mmproj_path": "$(json_escape "${MMPROJ_REL}")",
  "mmproj_sha256": "$(json_escape "${MMPROJ_ACTUAL_SHA256}")",
  "mmproj_size_bytes": "$(json_escape "$(value_or_unavailable stat -c %s "${MMPROJ_PATH}")")"
}
EOF
}

check_native_platform() {
  local distro_id
  local llama_version

  [[ "$(uname -s)" == Linux ]] || {
    printf 'required host kernel: Linux; found: %s\n' "$(uname -s)"
    return 1
  }
  [[ "$(uname -m)" == x86_64 ]] || {
    printf 'required architecture: x86_64; found: %s\n' "$(uname -m)"
    return 1
  }
  distro_id=$(os_release_value ID) || {
    printf 'cannot read distribution identity from %s\n' "${OS_RELEASE_PATH}"
    return 1
  }
  [[ "${distro_id}" == ubuntu ]] || {
    printf 'required distribution: ubuntu; found: %s\n' "${distro_id}"
    return 1
  }
  [[ ! -e "${CONTAINER_MARKER}" ]] || {
    printf 'container marker present: %s\n' "${CONTAINER_MARKER}"
    return 1
  }
  ! grep -Eqi '(docker|containerd|kubepods|podman|lxc)' "${CGROUP_PATH}" 2>/dev/null || {
    printf 'container runtime detected in %s\n' "${CGROUP_PATH}"
    return 1
  }
  ! grep -qi microsoft "${PROC_VERSION_PATH}" 2>/dev/null || {
    printf 'WSL is not accepted as native Ubuntu evidence\n'
    return 1
  }
  for command_name in uv curl Xvfb sha256sum git ps tar; do
    command -v "${command_name}" >/dev/null 2>&1 || {
      printf 'required command not found: %s\n' "${command_name}"
      return 1
    }
  done
  [[ -x "${GNU_TIME}" ]] || {
    printf 'GNU time not executable: %s\n' "${GNU_TIME}"
    return 1
  }
  "${GNU_TIME}" -f 'wall_seconds=%e' -o /dev/null true || {
    printf 'GNU time check failed: %s\n' "${GNU_TIME}"
    return 1
  }
  [[ -f "${LLAMA_ARCHIVE_PATH}" && "$(basename -- "${LLAMA_ARCHIVE_PATH}")" == "${LLAMA_ARCHIVE_FILENAME}" ]] || {
    printf 'llama.cpp archive must be the accepted file: %s\n' "${LLAMA_ARCHIVE_FILENAME}"
    return 1
  }
  [[ "${LLAMA_ARCHIVE_ACTUAL_SHA256}" == "${LLAMA_ARCHIVE_SHA256}" ]] || {
    printf 'llama.cpp archive SHA-256 does not match the accepted artifact\n'
    return 1
  }
  [[ -x "${LLAMA_SERVER_PATH}" ]] || {
    printf 'llama-server is not executable: %s\n' "${LLAMA_SERVER_REL}"
    return 1
  }
  verify_server_matches_archive || return 1
  [[ -f "${MODEL_PATH}" && "$(basename -- "${MODEL_PATH}")" == "${MODEL_FILENAME}" ]] || {
    printf 'model must be the accepted file: %s\n' "${MODEL_FILENAME}"
    return 1
  }
  [[ -f "${MMPROJ_PATH}" && "$(basename -- "${MMPROJ_PATH}")" == "${MMPROJ_FILENAME}" ]] || {
    printf 'projector must be the accepted file: %s\n' "${MMPROJ_FILENAME}"
    return 1
  }
  [[ "${MODEL_ACTUAL_SHA256}" == "${MODEL_SHA256}" ]] || {
    printf 'model SHA-256 does not match the accepted artifact\n'
    return 1
  }
  [[ "${MMPROJ_ACTUAL_SHA256}" == "${MMPROJ_SHA256}" ]] || {
    printf 'projector SHA-256 does not match the accepted artifact\n'
    return 1
  }
  llama_version=$("${LLAMA_SERVER_PATH}" --version 2>&1) || {
    printf 'cannot read llama-server version\n'
    return 1
  }
  grep -Eq "(b${LLAMA_BUILD}|build[[:space:]]*(=[[:space:]]*)?${LLAMA_BUILD}|version:[[:space:]]*${LLAMA_BUILD})" \
    <<<"${llama_version}" || {
      printf 'llama-server must identify build %s; found: %s\n' \
        "${LLAMA_BUILD}" "${llama_version}"
      return 1
    }
  [[ -z "$(git status --porcelain)" ]] || {
    printf 'git checkout must be clean before native acceptance\n'
    git status --short
    return 1
  }
  if curl -sS --max-time 2 "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
    printf 'port %s already serves an HTTP health endpoint\n' "${SERVER_PORT}"
    return 1
  fi
  printf 'native Ubuntu candidate identity verified\n'
}

start_xvfb() {
  local display_number=${XVFB_DISPLAY#:}
  [[ ! -e "/tmp/.X11-unix/X${display_number}" ]] || {
    printf 'X display already exists: %s (override NATIVE_PLATFORM_DISPLAY)\n' \
      "${XVFB_DISPLAY}"
    return 1
  }
  Xvfb "${XVFB_DISPLAY}" -screen 0 1280x720x24 -nolisten tcp \
    > "${OUTPUT_DIR}/logs/xvfb.log" 2>&1 &
  XVFB_PID=$!
  export DISPLAY="${XVFB_DISPLAY}"
  sleep 1
  kill -0 "${XVFB_PID}" 2>/dev/null || {
    printf 'Xvfb failed to remain running\n'
    return 1
  }
  printf 'private Xvfb display ready: %s\n' "${DISPLAY}"
}

stop_server() {
  local exit_code=0
  if [[ -n "${SERVER_PID}" ]]; then
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      exit_code=1
    else
      kill "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
  if [[ -n "${SAMPLER_PID}" ]]; then
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
    SAMPLER_PID=""
  fi
  return "${exit_code}"
}

cleanup() {
  stop_server >/dev/null 2>&1 || true
  if [[ -n "${XVFB_PID}" ]]; then
    kill "${XVFB_PID}" 2>/dev/null || true
    wait "${XVFB_PID}" 2>/dev/null || true
    XVFB_PID=""
  fi
}
trap cleanup EXIT INT TERM

sample_server_resources() {
  local output_path=$1
  printf 'timestamp_utc,pid,rss_kib,vsz_kib,cpu_percent,elapsed\n' > "${output_path}"
  while kill -0 "${SERVER_PID}" 2>/dev/null; do
    local row
    row=$(ps -p "${SERVER_PID}" -o pid=,rss=,vsz=,%cpu=,etime= 2>/dev/null) || break
    if [[ -n "${row}" ]]; then
      local pid rss vsz cpu elapsed
      read -r pid rss vsz cpu elapsed <<<"${row}"
      printf '%s,%s,%s,%s,%s,%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${pid}" "${rss}" "${vsz}" "${cpu}" "${elapsed}" \
        >> "${output_path}"
    fi
    sleep 1
  done
}

# start_server ROUND_DIR MODEL MMPROJ ALIAS [MODEL-SPECIFIC ARGUMENTS...]
start_server() {
  local round_dir=$1
  local model_path=$2
  local mmproj_path=$3
  local alias=$4
  shift 4
  local server_log="${round_dir}/server.log"
  local resources="${round_dir}/server-resources.csv"
  local -a server_command=(
    "${LLAMA_SERVER_PATH}"
    -m "${model_path}"
    --mmproj "${mmproj_path}"
    --host 127.0.0.1
    --port "${SERVER_PORT}"
    --alias "${alias}"
    --ctx-size 4096
    --threads 4
    --n-gpu-layers 0
    --no-mmproj-offload
    --parallel 1
    "$@"
    --temp 0
    --seed 0
  )

  command_text "${server_command[@]}" > "${round_dir}/server-command.txt"
  "${server_command[@]}" > "${server_log}" 2>&1 &
  SERVER_PID=$!
  sample_server_resources "${resources}" &
  SAMPLER_PID=$!

  local attempt
  for attempt in $(seq 1 600); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${SERVER_PORT}/health" \
      > "${round_dir}/health.json" 2> "${round_dir}/health-errors.log"; then
      local sample_attempt
      for sample_attempt in $(seq 1 10); do
        [[ -f "${resources}" && "$(wc -l < "${resources}")" -ge 2 ]] && break
        sleep 1
      done
      printf 'llama-server became healthy after %s second(s)\n' "${attempt}"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      printf 'llama-server exited before becoming healthy\n'
      return 1
    fi
    sleep 1
  done
  printf 'llama-server did not become healthy within 600 seconds\n'
  return 1
}

run_native_vlm_round() {
  local round_number=$1
  local round_dir="${OUTPUT_DIR}/model/round-${round_number}"
  local status=0
  mkdir -p -- "${round_dir}"

  if ! start_server "${round_dir}" "${MODEL_PATH}" "${MMPROJ_PATH}" "${MODEL_ALIAS}" \
    --image-min-tokens 1024; then
    stop_server || true
    return 1
  fi

  local -a probe_unstructured=(
    uv run --frozen metadrive-starter probe
    --config configs/vla-probe-qwen3-vl-2b-llama.yaml
    --infer
    --replay-from "${OUTPUT_REL}/camera/probe"
    --output-dir "${OUTPUT_REL}/model/round-${round_number}/probe-unstructured"
  )
  command_text "${probe_unstructured[@]}" > "${round_dir}/probe-unstructured-command.txt"
  # Evidence only: decoding without the grammar's help measures prompt-following,
  # a quality measure, so it never stops the round (ADR-0003). acceptance.json
  # records how many scenes decoded.
  local schema_off_status=0
  "${probe_unstructured[@]}" > "${round_dir}/probe-unstructured.log" 2>&1 \
    || schema_off_status=$?
  printf 'schema-off probe exited %s; recorded as evidence only\n' "${schema_off_status}"

  if [[ "${status}" == 0 ]]; then
    local -a probe_structured=(
      uv run --frozen metadrive-starter probe
      --config configs/vla-probe-qwen3-vl-2b-llama-structured.yaml
      --infer
      --replay-from "${OUTPUT_REL}/camera/probe"
      --output-dir "${OUTPUT_REL}/model/round-${round_number}/probe-structured"
    )
    command_text "${probe_structured[@]}" > "${round_dir}/probe-structured-command.txt"
    "${probe_structured[@]}" > "${round_dir}/probe-structured.log" 2>&1 || status=1
  fi

  if [[ "${status}" == 0 ]]; then
    local -a loop_command=(
      uv run --frozen metadrive-starter run
      --config configs/demo-vla-qwen3-vl-2b-llama.yaml
      --headless
      --steps 100
      --event-log "${OUTPUT_REL}/model/round-${round_number}/events.jsonl"
    )
    command_text "${loop_command[@]}" > "${round_dir}/loop-command.txt"
    "${GNU_TIME}" \
      -f $'wall_seconds=%e\nmax_rss_kib=%M' \
      -o "${round_dir}/loop-time.txt" \
      "${loop_command[@]}" \
      > "${round_dir}/loop.log" 2>&1 || status=1
  fi

  if [[ "${status}" == 0 ]]; then
    local -a summary_command=(
      uv run --frozen python scripts/native_vlm_summary.py
      --run-log "${round_dir}/loop.log"
      --time-file "${round_dir}/loop-time.txt"
      --server-resources "${round_dir}/server-resources.csv"
      --output "${round_dir}/acceptance.json"
      --schema-off-probe "${round_dir}/probe-unstructured/summary.json"
    )
    command_text "${summary_command[@]}" > "${round_dir}/summary-command.txt"
    "${summary_command[@]}" > "${round_dir}/summary.log" 2>&1 || status=1
  fi

  if [[ "${status}" == 0 ]]; then
    local -a replay_command=(
      uv run --frozen metadrive-starter replay
      --event-log "${OUTPUT_REL}/model/round-${round_number}/events.jsonl"
    )
    command_text "${replay_command[@]}" > "${round_dir}/replay-command.txt"
    "${replay_command[@]}" > "${round_dir}/replay.log" 2>&1 || status=1
  fi

  if ! stop_server; then
    printf 'llama-server exited unexpectedly during round %s\n' "${round_number}"
    status=1
  fi
  return "${status}"
}

verify_pinned_file() {
  local label=$1
  local path=$2
  local filename=$3
  local expected_sha256=$4
  [[ -f "${path}" && "$(basename -- "${path}")" == "${filename}" ]] || {
    printf '%s must be the pinned file: %s\n' "${label}" "${filename}"
    return 1
  }
  [[ "$(sha256_file "${path}")" == "${expected_sha256}" ]] || {
    printf '%s SHA-256 does not match the pinned artifact\n' "${label}"
    return 1
  }
}

# Can SmolVLM serve Assignment 2 on native Linux? Under the pinned server and
# JSON Schema output, every probe scene must decode, and a team's closed loop
# must run without runtime errors and replay exactly. Model authority is
# recorded, not required: the safety floor is expected to reject SmolVLM's
# answers for low confidence.
run_smolvlm_round() {
  local round_rel="${OUTPUT_REL}/model/smolvlm"
  local round_dir="${OUTPUT_DIR}/model/smolvlm"
  local status=0
  mkdir -p -- "${round_dir}"

  verify_pinned_file "SmolVLM model" "${SMOLVLM_MODEL_PATH}" \
    "${SMOLVLM_MODEL_FILENAME}" "${SMOLVLM_MODEL_SHA256}" || return 1
  verify_pinned_file "SmolVLM projector" "${SMOLVLM_MMPROJ_PATH}" \
    "${SMOLVLM_MMPROJ_FILENAME}" "${SMOLVLM_MMPROJ_SHA256}" || return 1
  printf 'SmolVLM revision %s verified\n' "${SMOLVLM_REVISION}"
  if ! start_server "${round_dir}" "${SMOLVLM_MODEL_PATH}" "${SMOLVLM_MMPROJ_PATH}" \
    "${SMOLVLM_ALIAS}"; then
    stop_server || true
    return 1
  fi

  local -a probe_structured=(
    uv run --frozen metadrive-starter probe
    --config configs/vla-probe-smolvlm-llama-structured.yaml
    --infer
    --replay-from "${OUTPUT_REL}/camera/probe"
    --output-dir "${round_rel}/probe-structured"
  )
  command_text "${probe_structured[@]}" > "${round_dir}/probe-structured-command.txt"
  "${probe_structured[@]}" > "${round_dir}/probe-structured.log" 2>&1 || status=1

  if [[ "${status}" == 0 ]]; then
    local -a loop_command=(
      uv run --frozen metadrive-starter run
      --config configs/demo-vla-smolvlm-llama.yaml
      --submission submission
      --headless
      --steps 100
      --event-log "${round_rel}/events.jsonl"
    )
    command_text "${loop_command[@]}" > "${round_dir}/loop-command.txt"
    "${GNU_TIME}" \
      -f $'wall_seconds=%e\nmax_rss_kib=%M' \
      -o "${round_dir}/loop-time.txt" \
      "${loop_command[@]}" \
      > "${round_dir}/loop.log" 2>&1 || status=1
  fi

  if [[ "${status}" == 0 ]]; then
    local -a summary_command=(
      uv run --frozen python scripts/native_vlm_summary.py
      --run-log "${round_dir}/loop.log"
      --time-file "${round_dir}/loop-time.txt"
      --server-resources "${round_dir}/server-resources.csv"
      --output "${round_dir}/acceptance.json"
      --allow-zero-authority
    )
    command_text "${summary_command[@]}" > "${round_dir}/summary-command.txt"
    "${summary_command[@]}" > "${round_dir}/summary.log" 2>&1 || status=1
  fi

  if [[ "${status}" == 0 ]]; then
    local -a replay_command=(
      uv run --frozen metadrive-starter replay
      --event-log "${round_rel}/events.jsonl"
    )
    command_text "${replay_command[@]}" > "${round_dir}/replay-command.txt"
    "${replay_command[@]}" > "${round_dir}/replay.log" 2>&1 || status=1
  fi

  if ! stop_server; then
    printf 'llama-server exited unexpectedly during the SmolVLM round\n'
    status=1
  fi
  return "${status}"
}

finalize_summary() {
  local successful=$1
  {
    printf '{\n  "schema_version": 1,\n  "successful": %s,\n' "${successful}"
    printf '  "candidate": {"llama_release": "%s", "model_revision": "%s"},\n' \
      "${LLAMA_RELEASE}" "${MODEL_REVISION}"
    printf '  "gates": [\n'
    awk 'NR > 1 {printf ",\n"} {printf "    %s", $0} END {printf "\n"}' \
      "${RESULTS_NDJSON}"
    printf '  ]\n}\n'
  } > "${OUTPUT_DIR}/gates.json"
  rm -- "${RESULTS_NDJSON}"
}

write_host_inventory

overall_success=true
ready=true

required_gate() {
  local gate_id=$1
  local description=$2
  shift 2
  if [[ "${ready}" != true ]]; then
    record_not_run "${gate_id}" "${description}" "blocked by an earlier required gate"
  elif ! run_gate "${gate_id}" "${description}" "$@"; then
    ready=false
    overall_success=false
  fi
}

required_gate n0-platform "Verify native Ubuntu and pinned VLM identity" check_native_platform
required_gate n1-sync "Create the frozen native uv environment" uv sync --frozen --group dev
required_gate n1-lock "Validate the dependency lock" uv lock --check
required_gate n2-xvfb "Start a private native Xvfb display" start_xvfb
required_gate n2-tests "Run the complete native test suite" uv run --frozen python -m pytest
# Every command a team runs drives the shipped submission, as it would in a
# student release, which has no reference controller.
required_gate n2-smoke "Run native CLI smoke wiring" \
  uv run --frozen metadrive-starter smoke --dry-run --submission submission
required_gate n2-headless "Run the native deterministic headless route" \
  uv run --frozen metadrive-starter run \
    --config configs/demo-autopilot.yaml --submission submission --headless --steps 500
required_gate n2-gates "Run the public Assignment 1 gates" \
  uv run --frozen metadrive-starter gates --submission submission --json

mkdir -p -- "${OUTPUT_DIR}/camera"
required_gate n3-probe-capture "Capture the fixed native RGB probe catalog" \
  uv run --frozen metadrive-starter probe \
    --output-dir "${OUTPUT_REL}/camera/probe"

# Student releases include fixtures/ only from Assignment 2 on, together with
# the observation.py these gates drive through.
if [[ -d fixtures ]]; then
  mkdir -p -- "${OUTPUT_DIR}/fixture"
  required_gate n4-fixture "Run the native fixture-backed VLA loop" \
    uv run --frozen metadrive-starter run \
      --config configs/demo-vla-fixture.yaml \
      --submission submission \
      --headless \
      --steps 100 \
      --event-log "${OUTPUT_REL}/fixture/events.jsonl"
  required_gate n4-replay "Replay native fixture-backed decisions" \
    uv run --frozen metadrive-starter replay \
      --event-log "${OUTPUT_REL}/fixture/events.jsonl"
  required_gate n4-gates "Run the public gates through the shipped observation" \
    uv run --frozen metadrive-starter gates --submission submission \
      --config configs/demo-vla-fixture.yaml --json
else
  record_not_run n4-fixture "Run the native fixture-backed VLA loop" \
    "skipped: this release does not include fixtures/"
  record_not_run n4-replay "Replay native fixture-backed decisions" \
    "skipped: this release does not include fixtures/"
  record_not_run n4-gates "Run the public gates through the shipped observation" \
    "skipped: this release does not include fixtures/"
fi
# SmolVLM needs only what precedes the Qwen rounds, so whether this machine can
# run Qwen never hides its answer.
smolvlm_ready=${ready}

mkdir -p -- "${OUTPUT_DIR}/model"
required_gate n5-round-1 "Run pinned native VLM acceptance from a clean server start" \
  run_native_vlm_round 1
required_gate n5-round-2 "Repeat pinned native VLM acceptance from a clean server start" \
  run_native_vlm_round 2

SMOLVLM_DESCRIPTION="Assignment 2 experiment: SmolVLM-256M under llama-server with JSON Schema"
if [[ -z "${SMOLVLM_MODEL_REL}" ]]; then
  record_not_run n7-smolvlm "${SMOLVLM_DESCRIPTION}" \
    "not requested: pass --smolvlm-model and --smolvlm-mmproj"
elif [[ "${smolvlm_ready}" != true ]]; then
  record_not_run n7-smolvlm "${SMOLVLM_DESCRIPTION}" \
    "blocked by a required gate before the model rounds"
else
  # A planning experiment, not platform acceptance: recorded, never decisive.
  run_gate n7-smolvlm "${SMOLVLM_DESCRIPTION}" run_smolvlm_round || true
fi

record_not_run n3-visual-review "Offscreen RGB visual review" \
  "manual gate: inspect camera/probe images for orientation, colour, and scene content"
record_not_run n3-gui "Rendered window and keyboard input" \
  "out of scope: the native VLM acceptance profile is deliberately headless"
record_not_run n6-cloud-vlm "Live course cloud VLM" \
  "out of scope: requires instructor-provisioned credentials and billing controls"

finalize_summary "${overall_success}"

printf 'Evidence: %s\n' "${OUTPUT_REL}"
if [[ "${overall_success}" != true ]]; then
  exit 1
fi
