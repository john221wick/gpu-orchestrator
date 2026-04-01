#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORCH_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOCKET_PATH="${GPU_SCHEDULER_SOCKET:-/tmp/gpu-scheduler.sock}"
PRINT_REQUEST="${GPU_SCHEDULER_PRINT_REQUEST:-0}"

if [[ ! -x "${ORCH_DIR}/gpu-submit" ]]; then
    make -C "${ORCH_DIR}" gpu-submit gpu-status
fi

submit_cmd=("${ORCH_DIR}/gpu-submit")
if [[ "${PRINT_REQUEST}" == "1" ]]; then
    submit_cmd+=("--print-request")
fi

if [[ $# -gt 0 && -f "${1}" ]]; then
    "${submit_cmd[@]}" "${1}" --socket "${SOCKET_PATH}"
    exit 0
fi

MODEL_NAME="${HF_INFERENCE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
ADAPTER_PATH="${HF_INFERENCE_ADAPTER_PATH:-outputs/vast-personal-chat}"
PROMPT_TEXT="${HF_INFERENCE_PROMPT:-Tell me one thing you know about Bhushan Bharat.}"
SYSTEM_PROMPT="${HF_INFERENCE_SYSTEM_PROMPT:-You are Bhushan personal AI assistant. Stay grounded in the fine-tuned data.}"
MAX_NEW_TOKENS="${HF_INFERENCE_MAX_NEW_TOKENS:-160}"
TEMPERATURE="${HF_INFERENCE_TEMPERATURE:-0.2}"
TOP_P="${HF_INFERENCE_TOP_P:-0.9}"
MIN_VRAM_GB="${HF_INFERENCE_MIN_VRAM_GB:-10}"

tmp_json="$(mktemp "${SCRIPT_DIR}/.tmp-gpu-hf-inference-XXXXXX")"
trap 'rm -f "${tmp_json}"' EXIT

python3 - "${tmp_json}" "${MODEL_NAME}" "${ADAPTER_PATH}" "${PROMPT_TEXT}" "${SYSTEM_PROMPT}" "${MAX_NEW_TOKENS}" "${TEMPERATURE}" "${TOP_P}" "${MIN_VRAM_GB}" <<'PY'
import json
import sys

out_path, model_name, adapter_path, prompt_text, system_prompt, max_new_tokens, temperature, top_p, min_vram_gb = sys.argv[1:]

payload = {
    "framework": "python",
    "job_type": "inference",
    "num_gpus": 1,
    "needs_peer": False,
    "min_vram_gb": int(min_vram_gb),
    "priority": 2,
    "script": "scripts/chat_personal_model.py",
    "args": [
        "--model-name", model_name,
        "--adapter-path", adapter_path,
        "--prompt", prompt_text,
        "--system-prompt", system_prompt,
        "--max-new-tokens", max_new_tokens,
        "--temperature", temperature,
        "--top-p", top_p,
        "--load-in-4bit",
    ],
    "working_dir": "../../hf-finetuner",
    "max_time_sec": 1800,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY

"${submit_cmd[@]}" "${tmp_json}" --socket "${SOCKET_PATH}"
