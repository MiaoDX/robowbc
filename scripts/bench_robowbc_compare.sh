#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY_PATH="${ROOT_DIR}/artifacts/benchmarks/nvidia/cases.json"
NORMALIZER="${ROOT_DIR}/scripts/normalize_nvidia_benchmarks.py"
DEFAULT_GEAR_SONIC_REVISION="cc80d505b7e055fd6ae26426ae8bfa0a74c26011"
DEFAULT_DECOUPLED_COMMIT="bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8"

usage() {
  cat <<'EOF'
Usage:
  scripts/bench_robowbc_compare.sh --list-cases
  scripts/bench_robowbc_compare.sh --case <case_id>
  scripts/bench_robowbc_compare.sh --all

Options:
  --output-root <dir>   Directory for normalized artifacts
  --provider <name>     Provider label recorded in the artifact (default: cpu)
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

OUTPUT_ROOT="${ROOT_DIR}/artifacts/benchmarks/nvidia/robowbc"
PROVIDER="cpu"
MODE=""
CASE_ID=""
GEAR_SONIC_MODEL_DIR="${GEAR_SONIC_MODEL_DIR:-${ROOT_DIR}/models/gear-sonic}"
DECOUPLED_WBC_MODEL_DIR="${DECOUPLED_WBC_MODEL_DIR:-${ROOT_DIR}/models/decoupled-wbc}"

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

sanitize_case_id() {
  local case_id="$1"
  printf '%s' "${case_id//\//__}"
}

read_revision_file() {
  local path="$1"
  local fallback="$2"
  if [[ -f "${path}" ]]; then
    tr -d '\n' < "${path}"
  else
    printf '%s' "${fallback}"
  fi
}

gear_sonic_revision() {
  read_revision_file "${GEAR_SONIC_MODEL_DIR}/REVISION" "${DEFAULT_GEAR_SONIC_REVISION}"
}

decoupled_revision() {
  read_revision_file "${DECOUPLED_WBC_MODEL_DIR}/REVISION" "${DEFAULT_DECOUPLED_COMMIT}"
}

have_gear_sonic_models() {
  [[ -f "${GEAR_SONIC_MODEL_DIR}/model_encoder.onnx" ]] &&
    [[ -f "${GEAR_SONIC_MODEL_DIR}/model_decoder.onnx" ]] &&
    [[ -f "${GEAR_SONIC_MODEL_DIR}/planner_sonic.onnx" ]]
}

have_decoupled_models() {
  [[ -f "${DECOUPLED_WBC_MODEL_DIR}/GR00T-WholeBodyControl-Walk.onnx" ]] &&
    [[ -f "${DECOUPLED_WBC_MODEL_DIR}/GR00T-WholeBodyControl-Balance.onnx" ]]
}

emit_blocked() {
  local case_id="$1"
  local upstream_commit="$2"
  local reason="$3"
  local output_file="${OUTPUT_ROOT}/$(sanitize_case_id "${case_id}").json"
  local source_command
  source_command="scripts/bench_robowbc_compare.sh --case ${case_id}"
  python3 "${NORMALIZER}" emit-blocked \
    --registry "${REGISTRY_PATH}" \
    --case-id "${case_id}" \
    --stack robowbc \
    --provider "${PROVIDER}" \
    --upstream-commit "${upstream_commit}" \
    --robowbc-commit "${ROBOWBC_COMMIT}" \
    --source-command "${source_command}" \
    --reason "${reason}" \
    --output "${output_file}"
  echo "[blocked] ${case_id} -> ${output_file}"
}

normalize_criterion_case() {
  local case_id="$1"
  local upstream_commit="$2"
  local source_command="$3"
  local output_file="${OUTPUT_ROOT}/$(sanitize_case_id "${case_id}").json"
  python3 "${NORMALIZER}" normalize-criterion \
    --registry "${REGISTRY_PATH}" \
    --case-id "${case_id}" \
    --stack robowbc \
    --provider "${PROVIDER}" \
    --upstream-commit "${upstream_commit}" \
    --robowbc-commit "${ROBOWBC_COMMIT}" \
    --criterion-root "${ROOT_DIR}/target/criterion" \
    --source-command "${source_command}" \
    --notes "Normalized from Criterion sample.json per-iteration timings." \
    --output "${output_file}"
  echo "[ok] ${case_id} -> ${output_file}"
}

run_microbench_case() {
  local case_id="$1"
  local filter="$2"
  local upstream_commit="$3"
  local env_name="$4"
  local env_value="$5"
  local source_command
  source_command="${env_name}=${env_value} cargo bench -p robowbc-ort --bench inference -- --output-format bencher '${filter}'"
  (
    cd "${ROOT_DIR}"
    env "${env_name}=${env_value}" cargo bench -p robowbc-ort --bench inference -- --output-format bencher "${filter}"
  )
  normalize_criterion_case "${case_id}" "${upstream_commit}" "${source_command}"
}

run_cli_case() {
  local case_id="$1"
  local config_path="$2"
  local upstream_commit="$3"
  local temp_dir
  temp_dir="$(mktemp -d)"
  local temp_config="${temp_dir}/$(basename "${config_path}")"
  local raw_report="${temp_dir}/report.json"
  cp "${ROOT_DIR}/${config_path}" "${temp_config}"
  printf '\n[report]\noutput_path = "%s"\nmax_frames = 200\n' "${raw_report}" >> "${temp_config}"
  local source_command
  source_command="cargo run -p robowbc-cli --bin robowbc -- run --config ${temp_config}"
  (
    cd "${ROOT_DIR}"
    cargo run -p robowbc-cli --bin robowbc -- run --config "${temp_config}"
  )
  python3 "${NORMALIZER}" normalize-run-report \
    --registry "${REGISTRY_PATH}" \
    --case-id "${case_id}" \
    --stack robowbc \
    --provider "${PROVIDER}" \
    --upstream-commit "${upstream_commit}" \
    --robowbc-commit "${ROBOWBC_COMMIT}" \
    --input "${raw_report}" \
    --source-command "${source_command}" \
    --notes "Normalized from robowbc-cli JSON run report." \
    --output "${OUTPUT_ROOT}/$(sanitize_case_id "${case_id}").json"
  rm -rf "${temp_dir}"
  echo "[ok] ${case_id} -> ${OUTPUT_ROOT}/$(sanitize_case_id "${case_id}").json"
}

run_case() {
  local case_id="$1"
  case "${case_id}" in
    gear_sonic_velocity/cold_start_tick)
      if have_gear_sonic_models; then
        run_microbench_case "${case_id}" "policy/gear_sonic_velocity/cold_start_tick" "$(gear_sonic_revision)" "GEAR_SONIC_MODEL_DIR" "${GEAR_SONIC_MODEL_DIR}"
      else
        emit_blocked "${case_id}" "$(gear_sonic_revision)" "GEAR-Sonic checkpoints not found under ${GEAR_SONIC_MODEL_DIR}; run scripts/download_gear_sonic_models.sh first."
      fi
      ;;
    gear_sonic_velocity/warm_steady_state_tick)
      if have_gear_sonic_models; then
        run_microbench_case "${case_id}" "policy/gear_sonic_velocity/warm_steady_state_tick" "$(gear_sonic_revision)" "GEAR_SONIC_MODEL_DIR" "${GEAR_SONIC_MODEL_DIR}"
      else
        emit_blocked "${case_id}" "$(gear_sonic_revision)" "GEAR-Sonic checkpoints not found under ${GEAR_SONIC_MODEL_DIR}; run scripts/download_gear_sonic_models.sh first."
      fi
      ;;
    gear_sonic_velocity/replan_tick)
      if have_gear_sonic_models; then
        run_microbench_case "${case_id}" "policy/gear_sonic_velocity/replan_tick" "$(gear_sonic_revision)" "GEAR_SONIC_MODEL_DIR" "${GEAR_SONIC_MODEL_DIR}"
      else
        emit_blocked "${case_id}" "$(gear_sonic_revision)" "GEAR-Sonic checkpoints not found under ${GEAR_SONIC_MODEL_DIR}; run scripts/download_gear_sonic_models.sh first."
      fi
      ;;
    gear_sonic_tracking/standing_placeholder_tick)
      if have_gear_sonic_models; then
        run_microbench_case "${case_id}" "policy/gear_sonic_tracking/standing_placeholder_tick" "$(gear_sonic_revision)" "GEAR_SONIC_MODEL_DIR" "${GEAR_SONIC_MODEL_DIR}"
      else
        emit_blocked "${case_id}" "$(gear_sonic_revision)" "GEAR-Sonic checkpoints not found under ${GEAR_SONIC_MODEL_DIR}; run scripts/download_gear_sonic_models.sh first."
      fi
      ;;
    decoupled_wbc/walk_predict)
      if have_decoupled_models; then
        run_microbench_case "${case_id}" "policy/decoupled_wbc/walk_predict" "$(decoupled_revision)" "DECOUPLED_WBC_MODEL_DIR" "${DECOUPLED_WBC_MODEL_DIR}"
      else
        emit_blocked "${case_id}" "$(decoupled_revision)" "Decoupled WBC checkpoints not found under ${DECOUPLED_WBC_MODEL_DIR}; run scripts/download_decoupled_wbc_models.sh first."
      fi
      ;;
    decoupled_wbc/balance_predict)
      if have_decoupled_models; then
        run_microbench_case "${case_id}" "policy/decoupled_wbc/balance_predict" "$(decoupled_revision)" "DECOUPLED_WBC_MODEL_DIR" "${DECOUPLED_WBC_MODEL_DIR}"
      else
        emit_blocked "${case_id}" "$(decoupled_revision)" "Decoupled WBC checkpoints not found under ${DECOUPLED_WBC_MODEL_DIR}; run scripts/download_decoupled_wbc_models.sh first."
      fi
      ;;
    gear_sonic/end_to_end_cli_loop)
      if have_gear_sonic_models; then
        run_cli_case "${case_id}" "configs/sonic_g1.toml" "$(gear_sonic_revision)"
      else
        emit_blocked "${case_id}" "$(gear_sonic_revision)" "GEAR-Sonic checkpoints not found under ${GEAR_SONIC_MODEL_DIR}; run scripts/download_gear_sonic_models.sh first."
      fi
      ;;
    decoupled_wbc/end_to_end_cli_loop)
      if have_decoupled_models; then
        run_cli_case "${case_id}" "configs/decoupled_g1.toml" "$(decoupled_revision)"
      else
        emit_blocked "${case_id}" "$(decoupled_revision)" "Decoupled WBC checkpoints not found under ${DECOUPLED_WBC_MODEL_DIR}; run scripts/download_decoupled_wbc_models.sh first."
      fi
      ;;
    *)
      emit_blocked "${case_id}" "unknown-upstream" "No RoboWBC benchmark mapping has been defined for ${case_id}."
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
