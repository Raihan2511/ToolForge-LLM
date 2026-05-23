#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  serve_vllm.sh — Launch vLLM OpenAI-compatible server on RunPod
#  Model : Fine-tuned Qwen 2.5 7B  (APIGen / xLAM tool-calling)
#  GPU   : Tested on 16 GB / 24 GB (A5000, A10, RTX 4090, etc.)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths (adjust if your layout differs) ──
# Typical RunPod environment puts everything in /workspace
MODEL_DIR="../merged"
CHAT_TEMPLATE="./template.jinja"

# ── Verify paths exist ──
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found at $MODEL_DIR"
    exit 1
fi
if [ ! -f "$CHAT_TEMPLATE" ]; then
    echo "ERROR: Chat template not found at $CHAT_TEMPLATE"
    exit 1
fi

echo "Starting vLLM server…"
echo "  Model       : $MODEL_DIR"
echo "  Template    : $CHAT_TEMPLATE"
echo "  Endpoint    : http://0.0.0.0:8000"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model              "$MODEL_DIR" \
    --served-model-name  qwen-tools \
    --chat-template      "$CHAT_TEMPLATE" \
    --host               0.0.0.0 \
    --port               8000 \
    --max-model-len      4096 \
    --gpu-memory-utilization 0.90 \
    --dtype              auto \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser xlam \
    "$@"
