#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build_out"
JOBS="${MAX_JOBS:-$(nproc)}"

if [[ -z "${MEMFABRIC_HYBRID_HOME_PATH:-}" ]]; then
  echo "ERROR: MEMFABRIC_HYBRID_HOME_PATH must point to the installed wgm-dev-310p package" >&2
  exit 2
fi

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  CANN_PATH="${ASCEND_HOME_PATH}"
else
  CANN_PATH="${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
fi
if [[ ! -d "${CANN_PATH}" ]]; then
  echo "ERROR: CANN path does not exist: ${CANN_PATH}" >&2
  exit 2
fi

rm -rf "${BUILD_DIR}"
cmake --preset default \
  -DASCEND_CANN_PACKAGE_PATH="${CANN_PATH}" \
  -DMEMFABRIC_HYBRID_HOME_PATH="${MEMFABRIC_HYBRID_HOME_PATH}"

# CANN 9.1 custom-operator packaging is intentionally two-phase: first build
# binary deliverables, then assemble the RUN package from those artifacts.
cmake --build "${BUILD_DIR}" --target binary -j"${JOBS}"
cmake --build "${BUILD_DIR}" --target package -j"${JOBS}"

RUN_FILE="$(find "${BUILD_DIR}" -maxdepth 2 -type f -name 'custom_opp_*.run' -print -quit)"
if [[ -z "${RUN_FILE}" ]]; then
  echo "ERROR: CANN build completed without custom_opp_*.run" >&2
  exit 3
fi

echo "Built: ${RUN_FILE}"
