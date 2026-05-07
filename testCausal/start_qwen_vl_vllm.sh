#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ROOT="${QWEN_VL_MODEL_ROOT:-}"
VLLM_BIN="${QWEN_VL_VLLM_BIN:-$(command -v vllm || true)}"
LOG_DIR="${QWEN_VL_LOG_DIR:-${PROJECT_ROOT}/logs/vllm}"

MODEL_NAME=""
PORT="8000"
GPU_INDEX=""
GPU_MEMORY_UTILIZATION="0.70"
MAX_MODEL_LEN="8192"
LOG_FILE=""
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  start_qwen_vl_vllm.sh --model MODEL_DIR [options]

Options:
  --model MODEL_DIR                 Model directory name under --model-root, e.g. Qwen3-VL-4B-Instruct
  --model-root PATH                 Root directory containing model folders
  --vllm-bin PATH                   Path to the vLLM executable
  --gpu GPU_INDEX                  Use a specific GPU instead of auto-selecting the most free one
  --port PORT                      API server port (default: 8000)
  --gpu-memory-utilization VALUE   vLLM GPU memory utilization (default: 0.70)
  --max-model-len VALUE            vLLM max model length (default: 8192)
  --log-dir PATH                   Log directory (default: <repo>/logs/vllm)
  --log-file PATH                  Log file path (default: <log-dir>/<model>_port<port>.log)
  --dry-run                        Print the final command without launching
  --help                           Show this help message
EOF
}

fail() {
  echo "$1" >&2
  exit 1
}

select_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found. Use --gpu to specify a GPU manually."
  fi

  local selected
  selected="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n1 | cut -d, -f1 | tr -d ' ')"
  if [[ -z "${selected}" ]]; then
    fail "Unable to auto-select a GPU from nvidia-smi output."
  fi
  echo "${selected}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || fail "Missing value for --model"
      MODEL_NAME="$2"
      shift 2
      ;;
    --model-root)
      [[ $# -ge 2 ]] || fail "Missing value for --model-root"
      MODEL_ROOT="$2"
      shift 2
      ;;
    --vllm-bin)
      [[ $# -ge 2 ]] || fail "Missing value for --vllm-bin"
      VLLM_BIN="$2"
      shift 2
      ;;
    --gpu)
      [[ $# -ge 2 ]] || fail "Missing value for --gpu"
      GPU_INDEX="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || fail "Missing value for --port"
      PORT="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      [[ $# -ge 2 ]] || fail "Missing value for --gpu-memory-utilization"
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --max-model-len)
      [[ $# -ge 2 ]] || fail "Missing value for --max-model-len"
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --log-file)
      [[ $# -ge 2 ]] || fail "Missing value for --log-file"
      LOG_FILE="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || fail "Missing value for --log-dir"
      LOG_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

[[ -n "${MODEL_NAME}" ]] || fail "Missing required argument: --model"
[[ -n "${MODEL_ROOT}" ]] || fail "Model root is not set. Use --model-root or QWEN_VL_MODEL_ROOT."
[[ -n "${VLLM_BIN}" ]] || fail "vLLM binary is not set. Use --vllm-bin, QWEN_VL_VLLM_BIN, or make vllm available in PATH."
[[ -x "${VLLM_BIN}" ]] || fail "vLLM binary is not executable: ${VLLM_BIN}"

MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
[[ -d "${MODEL_PATH}" ]] || fail "Model directory does not exist: ${MODEL_PATH}"

if [[ -z "${GPU_INDEX}" ]]; then
  GPU_INDEX="$(select_gpu)"
fi

mkdir -p "${LOG_DIR}"
if [[ -z "${LOG_FILE}" ]]; then
  LOG_FILE="${LOG_DIR}/${MODEL_NAME}_port${PORT}.log"
fi

COMMAND=(
  "${VLLM_BIN}" serve "${MODEL_PATH}"
  --served-model-name "${MODEL_NAME}"
  --host 127.0.0.1
  --port "${PORT}"
  --trust-remote-code
  --dtype auto
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
)

echo "Selected GPU: ${GPU_INDEX}"
echo "Model path: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "Log file: ${LOG_FILE}"
echo "Command:"
printf 'CUDA_VISIBLE_DEVICES=%s ' "${GPU_INDEX}"
printf '%q ' "${COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

nohup env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${COMMAND[@]}" >"${LOG_FILE}" 2>&1 &
PID=$!

echo "Started vLLM with PID: ${PID}"
echo "Check server:"
echo "curl http://127.0.0.1:${PORT}/v1/models"
echo "Stop server:"
echo "kill ${PID}"
