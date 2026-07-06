#!/usr/bin/env bash
set -euo pipefail

# GPU selection. Example: "0" or "0,1".
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Model and server configuration.
export LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-/data/labshare/Param/llada/}"
export LLADA_MODEL_NAME="${LLADA_MODEL_NAME:-llada}"
export LLADA_HOST="${LLADA_HOST:-0.0.0.0}"
export LLADA_PORT="${LLADA_PORT:-7004}"
export LLADA_DEVICE="${LLADA_DEVICE:-cuda}"
export LLADA_DTYPE="${LLADA_DTYPE:-bfloat16}"

# Default generation parameters.
export LLADA_STEPS="${LLADA_STEPS:-128}"
export LLADA_GEN_LENGTH="${LLADA_GEN_LENGTH:-512}"
export LLADA_BLOCK_LENGTH="${LLADA_BLOCK_LENGTH:-512}"
export LLADA_TEMPERATURE="${LLADA_TEMPERATURE:-0}"
export LLADA_CFG_SCALE="${LLADA_CFG_SCALE:-0}"
export LLADA_REMASKING="${LLADA_REMASKING:-low_confidence}"
export LLADA_DEBUG_REQUESTS="${LLADA_DEBUG_REQUESTS:-true}"
export LLADA_DEBUG_FULL_REQUEST="${LLADA_DEBUG_FULL_REQUEST:-false}"
export LLADA_DEBUG_PREVIEW_CHARS="${LLADA_DEBUG_PREVIEW_CHARS:-500}"

cd "$(dirname "$0")/.."

if [[ ! -d "${LLADA_MODEL_PATH}" ]]; then
  echo "ERROR: LLADA_MODEL_PATH does not exist: ${LLADA_MODEL_PATH}" >&2
  echo "Set it to a local HuggingFace model directory that contains config.json, for example:" >&2
  echo "  LLADA_MODEL_PATH=/path/to/LLaDA-8B-Instruct ./llada_server/start_server.sh" >&2
  exit 1
fi

if [[ ! -f "${LLADA_MODEL_PATH}/config.json" ]]; then
  echo "ERROR: LLADA_MODEL_PATH is not a complete HuggingFace model directory: ${LLADA_MODEL_PATH}" >&2
  echo "Missing file: ${LLADA_MODEL_PATH}/config.json" >&2
  exit 1
fi

echo "Starting LLaDA server"
echo "  LLADA_MODEL_PATH=${LLADA_MODEL_PATH}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "  URL=http://${LLADA_HOST}:${LLADA_PORT}/v1"

python -m llada_server.server
