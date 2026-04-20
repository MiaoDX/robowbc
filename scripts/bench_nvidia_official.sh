#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY_PATH="${ROOT_DIR}/artifacts/benchmarks/nvidia/cases.json"
NORMALIZER="${ROOT_DIR}/scripts/normalize_nvidia_benchmarks.py"
DECOUPLED_HARNESS="${ROOT_DIR}/scripts/bench_nvidia_decoupled_official.py"
GEAR_SONIC_HARNESS_SRC="${ROOT_DIR}/scripts/bench_nvidia_gear_sonic_official.cpp"
GEAR_SONIC_HARNESS_BIN="${ROOT_DIR}/target/nvidia-bench/bench_nvidia_gear_sonic_official"
DEFAULT_OFFICIAL_REPO_DIR="/tmp/GR00T-WholeBodyControl"
DEFAULT_UPSTREAM_COMMIT="bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8"

usage() {
  cat <<'EOF'
Usage:
  scripts/bench_nvidia_official.sh --list-cases
  scripts/bench_nvidia_official.sh --case <case_id>
  scripts/bench_nvidia_official.sh --all

Options:
  --output-root <dir>   Directory for normalized artifacts
  --provider <name>     Provider label recorded in the artifact (default: cpu)
  --repo-dir <dir>      Pinned GR00T-WholeBodyControl checkout (default: /tmp/GR00T-WholeBodyControl)
  --samples <n>         Microbenchmark samples for official harnesses (default: 100)
  --ticks <n>           End-to-end loop ticks for official harnesses (default: 200)
  --control-frequency-hz <n>
                        End-to-end control frequency label for official harnesses (default: 50)
EOF
}

list_cases() {
  python3 - "$REGISTRY_PATH" <<'PY'
import json, pathlib, sys
registry = json.loads(pathlib.Path(sys.argv[1]).read_text())
for case in registry["cases"]:
    print(case["case_id"])
PY
}

OUTPUT_ROOT="${ROOT_DIR}/artifacts/benchmarks/nvidia/official"
PROVIDER="cpu"
OFFICIAL_REPO_DIR="${DEFAULT_OFFICIAL_REPO_DIR}"
DECOUPLED_WBC_MODEL_DIR="${DECOUPLED_WBC_MODEL_DIR:-${ROOT_DIR}/models/decoupled-wbc}"
GEAR_SONIC_MODEL_DIR="${GEAR_SONIC_MODEL_DIR:-${ROOT_DIR}/models/gear-sonic}"
OFFICIAL_SAMPLES=100
OFFICIAL_TICKS=200
CONTROL_FREQUENCY_HZ=50
MODE=""
CASE_ID=""
GEAR_SONIC_BUILD_FAILURE_REASON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list-cases)
      list_cases
      exit 0
      ;;
    --case)
      MODE="case"
      CASE_ID="$2"
      shift 2
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --repo-dir)
      OFFICIAL_REPO_DIR="$2"
      shift 2
      ;;
    --samples)
      OFFICIAL_SAMPLES="$2"
      shift 2
      ;;
    --ticks)
      OFFICIAL_TICKS="$2"
      shift 2
      ;;
    --control-frequency-hz)
      CONTROL_FREQUENCY_HZ="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODE}" ]]; then
  echo "[error] choose --case, --all, or --list-cases" >&2
  usage >&2
  exit 1
fi

ROBOWBC_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
UPSTREAM_COMMIT="${DEFAULT_UPSTREAM_COMMIT}"

ensure_official_source_checkout() {
  if [[ -d "${OFFICIAL_REPO_DIR}/.git" ]]; then
    UPSTREAM_COMMIT="$(git -C "${OFFICIAL_REPO_DIR}" rev-parse HEAD)"
    return 0
  fi

  rm -rf "${OFFICIAL_REPO_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
    https://github.com/NVlabs/GR00T-WholeBodyControl "${OFFICIAL_REPO_DIR}"
  UPSTREAM_COMMIT="$(git -C "${OFFICIAL_REPO_DIR}" rev-parse HEAD)"
}

emit_blocked() {
  local case_id="$1"
  local reason="$2"
  local output_file="${OUTPUT_ROOT}/${case_id//\//__}.json"
  local source_command
  source_command="scripts/bench_nvidia_official.sh --case ${case_id}"

  python3 "${NORMALIZER}" emit-blocked \
    --registry "${REGISTRY_PATH}" \
    --case-id "${case_id}" \
    --stack official_nvidia \
    --provider "${PROVIDER}" \
    --upstream-commit "${UPSTREAM_COMMIT}" \
    --robowbc-commit "${ROBOWBC_COMMIT}" \
    --source-command "${source_command}" \
    --reason "${reason}" \
    --output "${output_file}"
  echo "[blocked] ${case_id} -> ${output_file}"
}

normalize_manual_case() {
  local case_id="$1"
  local input_path="$2"
  local notes="$3"
  local source_command="$4"
  local output_file="${OUTPUT_ROOT}/${case_id//\//__}.json"

  python3 "${NORMALIZER}" normalize-samples \
    --registry "${REGISTRY_PATH}" \
    --case-id "${case_id}" \
    --stack official_nvidia \
    --provider "${PROVIDER}" \
    --upstream-commit "${UPSTREAM_COMMIT}" \
    --robowbc-commit "${ROBOWBC_COMMIT}" \
    --input "${input_path}" \
    --notes "${notes}" \
    --source-command "${source_command}" \
    --output "${output_file}"
  echo "[ok] ${case_id} -> ${output_file}"
}

have_decoupled_models() {
  [[ -f "${DECOUPLED_WBC_MODEL_DIR}/GR00T-WholeBodyControl-Walk.onnx" ]] &&
    [[ -f "${DECOUPLED_WBC_MODEL_DIR}/GR00T-WholeBodyControl-Balance.onnx" ]]
}

have_gear_sonic_models() {
  [[ -f "${GEAR_SONIC_MODEL_DIR}/planner_sonic.onnx" ]] &&
    [[ -f "${GEAR_SONIC_MODEL_DIR}/model_encoder.onnx" ]] &&
    [[ -f "${GEAR_SONIC_MODEL_DIR}/model_decoder.onnx" ]]
}

discover_onnxruntime_root() {
  find "${ROOT_DIR}/target/debug/build" -maxdepth 4 -type d -name 'onnxruntime-linux-x64-*' | sort | tail -n 1
}

ensure_gear_sonic_harness() {
  if [[ -x "${GEAR_SONIC_HARNESS_BIN}" && "${GEAR_SONIC_HARNESS_BIN}" -nt "${GEAR_SONIC_HARNESS_SRC}" ]]; then
    return 0
  fi

  local upstream_include="${OFFICIAL_REPO_DIR}/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include"
  if [[ ! -f "${upstream_include}/policy_parameters.hpp" ]] || [[ ! -f "${upstream_include}/robot_parameters.hpp" ]]; then
    GEAR_SONIC_BUILD_FAILURE_REASON="Pinned commit ${UPSTREAM_COMMIT} is missing the GEAR-Sonic C++ reference headers under ${upstream_include}; cannot build the official headless harness."
    return 1
  fi

  local ort_root
  ort_root="$(discover_onnxruntime_root)"
  if [[ -z "${ort_root}" ]]; then
    GEAR_SONIC_BUILD_FAILURE_REASON="ONNX Runtime headers/libs were not found under ${ROOT_DIR}/target/debug/build; run cargo build first so the C++ harness can link against the same ORT bundle."
    return 1
  fi

  local build_log="${OUTPUT_ROOT}/raw/gear_sonic_build.log"
  mkdir -p "$(dirname "${GEAR_SONIC_HARNESS_BIN}")" "$(dirname "${build_log}")"
  if ! c++ -std=c++20 -O3 \
      -I "${ort_root}/include" \
      -I "${upstream_include}" \
      "${GEAR_SONIC_HARNESS_SRC}" \
      -L "${ort_root}/lib" \
      -Wl,-rpath,"${ort_root}/lib" \
      -lonnxruntime \
      -lpthread \
      -o "${GEAR_SONIC_HARNESS_BIN}" >"${build_log}" 2>&1; then
    GEAR_SONIC_BUILD_FAILURE_REASON="Failed to compile ${GEAR_SONIC_HARNESS_SRC} against ${ort_root}; see ${build_log} for the compiler output."
    return 1
  fi

  return 0
}

run_decoupled_case() {
  local case_id="$1"
  local raw_output
  raw_output="${OUTPUT_ROOT}/raw/${case_id//\//__}.json"
  mkdir -p "$(dirname "${raw_output}")"
  local source_command
  source_command="python3 ${DECOUPLED_HARNESS} --case-id ${case_id} --repo-dir ${OFFICIAL_REPO_DIR} --model-dir ${DECOUPLED_WBC_MODEL_DIR} --robot-config ${ROOT_DIR}/configs/robots/unitree_g1.toml --samples ${OFFICIAL_SAMPLES} --ticks ${OFFICIAL_TICKS} --control-frequency-hz ${CONTROL_FREQUENCY_HZ}"

  python3 "${DECOUPLED_HARNESS}" \
    --case-id "${case_id}" \
    --repo-dir "${OFFICIAL_REPO_DIR}" \
    --model-dir "${DECOUPLED_WBC_MODEL_DIR}" \
    --robot-config "${ROOT_DIR}/configs/robots/unitree_g1.toml" \
    --samples "${OFFICIAL_SAMPLES}" \
    --ticks "${OFFICIAL_TICKS}" \
    --control-frequency-hz "${CONTROL_FREQUENCY_HZ}" \
    --output "${raw_output}"

  normalize_manual_case \
    "${case_id}" \
    "${raw_output}" \
    "Measured via upstream Decoupled WBC headless harness on the pinned source checkout." \
    "${source_command}"
}

run_gear_sonic_case() {
  local case_id="$1"
  local raw_output="${OUTPUT_ROOT}/raw/${case_id//\//__}.json"
  local source_command
  source_command="${GEAR_SONIC_HARNESS_BIN} --case-id ${case_id} --model-dir ${GEAR_SONIC_MODEL_DIR} --samples ${OFFICIAL_SAMPLES} --ticks ${OFFICIAL_TICKS} --control-frequency-hz ${CONTROL_FREQUENCY_HZ} --output ${raw_output}"

  "${GEAR_SONIC_HARNESS_BIN}" \
    --case-id "${case_id}" \
    --model-dir "${GEAR_SONIC_MODEL_DIR}" \
    --samples "${OFFICIAL_SAMPLES}" \
    --ticks "${OFFICIAL_TICKS}" \
    --control-frequency-hz "${CONTROL_FREQUENCY_HZ}" \
    --output "${raw_output}"

  normalize_manual_case \
    "${case_id}" \
    "${raw_output}" \
    "Measured via official GEAR-Sonic C++ ONNX Runtime harness on the pinned source checkout." \
    "${source_command}"
}

blocker_for_case() {
  local case_id="$1"
  case "${case_id}" in
    gear_sonic_velocity/*|gear_sonic_tracking/*|gear_sonic/end_to_end_cli_loop)
      if [[ -n "${GEAR_SONIC_BUILD_FAILURE_REASON}" ]]; then
        printf '%s' "${GEAR_SONIC_BUILD_FAILURE_REASON}"
      else
        printf '%s' "Official GEAR-Sonic checkpoints are missing under ${GEAR_SONIC_MODEL_DIR}; run scripts/download_gear_sonic_models.sh first."
      fi
      ;;
    decoupled_wbc/walk_predict|decoupled_wbc/balance_predict)
      printf '%s' "Official Decoupled WBC models are missing under ${DECOUPLED_WBC_MODEL_DIR}; run scripts/download_decoupled_wbc_models.sh first."
      ;;
    decoupled_wbc/end_to_end_cli_loop)
      printf '%s' "Official Decoupled WBC models are missing under ${DECOUPLED_WBC_MODEL_DIR}; run scripts/download_decoupled_wbc_models.sh first."
      ;;
    *)
      printf '%s' "No official-wrapper mapping has been defined for ${case_id}."
      ;;
  esac
}

run_case() {
  local case_id="$1"
  case "${case_id}" in
    gear_sonic_velocity/*|gear_sonic_tracking/*|gear_sonic/end_to_end_cli_loop)
      ensure_official_source_checkout
      if have_gear_sonic_models && ensure_gear_sonic_harness; then
        run_gear_sonic_case "${case_id}"
      else
        emit_blocked "${case_id}" "$(blocker_for_case "${case_id}")"
      fi
      ;;
    decoupled_wbc/walk_predict|decoupled_wbc/balance_predict|decoupled_wbc/end_to_end_cli_loop)
      ensure_official_source_checkout
      if have_decoupled_models; then
        run_decoupled_case "${case_id}"
      else
        emit_blocked "${case_id}" "$(blocker_for_case "${case_id}")"
      fi
      ;;
    *)
      emit_blocked "${case_id}" "$(blocker_for_case "${case_id}")"
      ;;
  esac
}

if [[ "${MODE}" == "case" ]]; then
  run_case "${CASE_ID}"
  exit 0
fi

while IFS= read -r case_id; do
  run_case "${case_id}"
done < <(list_cases)
