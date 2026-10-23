#!/usr/bin/env bash
# Serve the local Muse-Glimmer-30B (bf16 is ~60 GB, so it is quantised to fp8 on the fly to fit one 48 GB GPU).
# Uses the shared vLLM environment in /home/pza/venvs/llmserve (vLLM 0.32, knows the muse_glimmer architecture).
#   bash scripts/serve_muse.sh                  # GPU 2, port 8000
#   GPU=1 PORT=8001 bash scripts/serve_muse.sh
# Then:  qins run -c quick -s llm.enabled=true -s execution.mitigation.policy=llm -s model_spec.accept_llm=true
# Check which GPU is free first:  nvidia-smi --query-gpu=index,memory.used --format=csv
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-/home/pza/local_models/Muse-Glimmer-30B}"
NAME="${NAME:-muse-glimmer-30b}"
PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${GPU:-2}"
export PATH="$(dirname "${VLLM:-/home/pza/venvs/llmserve/bin/vllm}"):$PATH"   # vLLM JIT needs ninja from its own venv
VLLM="${VLLM:-/home/pza/venvs/llmserve/bin/vllm}"
exec "$VLLM" serve "$MODEL_DIR" \
  --served-model-name "$NAME" \
  --port "$PORT" \
  --quantization fp8_per_tensor \
  --max-model-len "${MAXLEN:-16384}" \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --seed 0
