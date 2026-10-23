#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server shared by all agents.
# vLLM is installed in its own environment (it pins torch/CUDA), e.g.:
#   uv venv .venv-vllm && uv pip install --python .venv-vllm vllm
# Then:  bash scripts/serve_vllm.sh            (defaults below)
#        MODEL=mistralai/Mistral-Small-24B-Instruct-2501 TP=2 bash scripts/serve_vllm.sh
# and run the pipeline with:  -s llm.enabled=true -s llm.model=$MODEL
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"   # already in ~/.cache/huggingface
PORT="${PORT:-8000}"
TP="${TP:-1}"                                        # tensor-parallel GPUs
VLLM="${VLLM:-.venv-vllm/bin/vllm}"
command -v "$VLLM" >/dev/null 2>&1 || VLLM=vllm
exec "$VLLM" serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --seed 0
