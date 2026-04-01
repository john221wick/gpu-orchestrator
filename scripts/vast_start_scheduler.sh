#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORCH_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOCKET_PATH="${GPU_SCHEDULER_SOCKET:-/tmp/gpu-scheduler.sock}"
LOG_PATH="${GPU_SCHEDULER_LOG:-/tmp/gpu-scheduler-daemon.log}"

if [[ ! -x "${ORCH_DIR}/gpu-scheduler" ]]; then
    make -C "${ORCH_DIR}" real
fi

if [[ -S "${SOCKET_PATH}" ]]; then
    if "${ORCH_DIR}/gpu-status" --socket "${SOCKET_PATH}" >/dev/null 2>&1; then
        echo "[gpu-orchestrator] Scheduler already running on ${SOCKET_PATH}"
        exit 0
    fi

    echo "[gpu-orchestrator] Removing stale socket ${SOCKET_PATH}"
    rm -f "${SOCKET_PATH}"
fi

args=(
    --socket "${SOCKET_PATH}"
    --log "${LOG_PATH}"
)

if [[ "${GPU_SCHEDULER_DEBUG:-0}" == "1" ]]; then
    args+=(--debug)
fi

nohup "${ORCH_DIR}/gpu-scheduler" "${args[@]}" >/dev/null 2>&1 &
pid=$!
sleep 2

if [[ ! -S "${SOCKET_PATH}" ]]; then
    echo "[gpu-orchestrator] Scheduler failed to create ${SOCKET_PATH}" >&2
    echo "[gpu-orchestrator] Check ${LOG_PATH} for details." >&2
    exit 1
fi

echo "[gpu-orchestrator] gpu-scheduler started"
echo "[gpu-orchestrator] pid=${pid}"
echo "[gpu-orchestrator] socket=${SOCKET_PATH}"
echo "[gpu-orchestrator] log=${LOG_PATH}"
