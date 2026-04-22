#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT="${1:-models/gear-sonic/reference/example}"
shift || true

UPSTREAM_COMMIT="${GEAR_SONIC_REFERENCE_COMMIT:-bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8}"
BASE_MEDIA_URL="https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/${UPSTREAM_COMMIT}/gear_sonic_deploy/reference/example"
BASE_RAW_URL="https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/${UPSTREAM_COMMIT}/gear_sonic_deploy/reference/example"

if [[ $# -eq 0 ]]; then
  clips=(
    "macarena_001__A545"
  )
else
  clips=("$@")
fi

files=(
  "joint_pos.csv"
  "joint_vel.csv"
  "body_pos.csv"
  "body_quat.csv"
  "body_lin_vel.csv"
  "body_ang_vel.csv"
  "metadata.txt"
  "info.txt"
)

mkdir -p "${DEST_ROOT}"
printf '%s\n' "${UPSTREAM_COMMIT}" > "${DEST_ROOT}/UPSTREAM_COMMIT"
echo "[info] pinned GR00T-WholeBodyControl commit: ${UPSTREAM_COMMIT}"

for clip in "${clips[@]}"; do
  clip_dir="${DEST_ROOT}/${clip}"
  mkdir -p "${clip_dir}"
  echo "[clip] ${clip}"

  for file in "${files[@]}"; do
    target="${clip_dir}/${file}"
    if [[ -s "${target}" ]] && ! head -n 1 "${target}" | grep -q '^version https://git-lfs.github.com/spec/v1$'; then
      echo "  [cache] ${file} already present"
      continue
    fi

    case "${file}" in
      metadata.txt|info.txt)
        source_url="${BASE_RAW_URL}/${clip}/${file}"
        ;;
      *)
        source_url="${BASE_MEDIA_URL}/${clip}/${file}"
        ;;
    esac

    echo "  [download] ${file}"
    curl --fail --location --retry 3 --retry-delay 2 \
      "${source_url}" \
      --output "${target}"
    if [[ ! -s "${target}" ]]; then
      echo "  [error] downloaded file is empty: ${target}" >&2
      exit 1
    fi
  done
done

echo "Downloaded GEAR-Sonic reference motions to ${DEST_ROOT}."
