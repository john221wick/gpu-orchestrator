#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORCH_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOCKET_PATH="${GPU_SCHEDULER_SOCKET:-/tmp/gpu-scheduler.sock}"
PRINT_REQUEST="${GPU_SCHEDULER_PRINT_REQUEST:-0}"
JOB_JSON="${1:-${SCRIPT_DIR}/job_hf_personal_chat.json}"

if [[ ! -f "${JOB_JSON}" ]]; then
    echo "[gpu-orchestrator] Job JSON not found: ${JOB_JSON}" >&2
    exit 1
fi

if [[ ! -x "${ORCH_DIR}/gpu-submit" ]]; then
    make -C "${ORCH_DIR}" gpu-submit gpu-status
fi

submit_cmd=("${ORCH_DIR}/gpu-submit")
if [[ "${PRINT_REQUEST}" == "1" ]]; then
    submit_cmd+=("--print-request")
fi

"${submit_cmd[@]}" "${JOB_JSON}" --socket "${SOCKET_PATH}"
