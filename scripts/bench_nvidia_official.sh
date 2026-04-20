#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY_PATH="${ROOT_DIR}/artifacts/benchmarks/nvidia/cases.json"
NORMALIZER="${ROOT_DIR}/scripts/normalize_nvidia_benchmarks.py"
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
MODE=""
CASE_ID=""

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
if [[ -d "${OFFICIAL_REPO_DIR}/.git" ]]; then
  UPSTREAM_COMMIT="$(git -C "${OFFICIAL_REPO_DIR}" rev-parse HEAD)"
else
  UPSTREAM_COMMIT="${DEFAULT_UPSTREAM_COMMIT}"
fi

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

blocker_for_case() {
  local case_id="$1"
  case "${case_id}" in
    gear_sonic_velocity/*|gear_sonic_tracking/*|gear_sonic/end_to_end_cli_loop)
      if [[ ! -d "${OFFICIAL_REPO_DIR}/.git" ]]; then
        printf '%s' "Pinned GR00T-WholeBodyControl checkout not found at ${OFFICIAL_REPO_DIR}; the official GEAR-Sonic seam lives in gear_sonic_deploy and needs a local checkout plus a dedicated non-interactive harness."
      else
        printf '%s' "Pinned commit ${UPSTREAM_COMMIT} exposes the GEAR-Sonic deployment path, but not a headless apples-to-apples benchmark seam for ${case_id}; benchmark theater is avoided until a dedicated harness exists."
      fi
      ;;
    decoupled_wbc/walk_predict|decoupled_wbc/balance_predict)
      if ! python3 - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
PY
      then
        printf '%s' "Official Decoupled WBC benchmarking is blocked in this environment because the pinned upstream Python stack expects torch plus the decoupled_wbc package layout."
      elif [[ ! -d "${OFFICIAL_REPO_DIR}/.git" ]]; then
        printf '%s' "Pinned GR00T-WholeBodyControl checkout not found at ${OFFICIAL_REPO_DIR}; the official Decoupled WBC policy code cannot be imported."
      else
        printf '%s' "The pinned upstream Decoupled policy is available, but this wrapper still blocks until a dedicated headless benchmark seam is added instead of sampling an approximate control-loop path."
      fi
      ;;
    decoupled_wbc/end_to_end_cli_loop)
      printf '%s' "Official Decoupled WBC end-to-end benchmarking is blocked because the upstream control loop is ROS- and device-oriented rather than a deterministic headless benchmark command."
      ;;
    *)
      printf '%s' "No official-wrapper mapping has been defined for ${case_id}."
      ;;
  esac
}

run_case() {
  local case_id="$1"
  emit_blocked "${case_id}" "$(blocker_for_case "${case_id}")"
}

if [[ "${MODE}" == "case" ]]; then
  run_case "${CASE_ID}"
  exit 0
fi

while IFS= read -r case_id; do
  run_case "${case_id}"
done < <(list_cases)
