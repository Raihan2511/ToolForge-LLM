#!/bin/bash
# setup.sh — Install Unsloth + all dependencies on RunPod
# Template : RunPod PyTorch 2.4.0 / CUDA 12.4
# GPU      : NVIDIA L40S 48GB
#
# IMPORTANT: Unsloth has strict version requirements.
# Do NOT do a generic pip upgrade first — it will break the install.
# Run this script exactly once after pod start.

set -e
echo "============================================"
echo " Unsloth Training Environment Setup"
echo " GPU: NVIDIA L40S 48GB"
echo "============================================"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git wget curl screen tmux rclone jq

# ── 2. Verify CUDA before installing anything ─────────────────────────────────
echo ""
echo "Checking CUDA..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}')"

# ---> FIX INSERTED HERE: Lock PyTorch ecosystem to stable versions <---
echo ""
echo "Locking PyTorch ecosystem to stable versions..."
pip uninstall -y torch torchvision torchaudio xformers unsloth unsloth-zoo torchao || true
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu124
# ----------------------------------------------------------------------

# ── 3. Unsloth installation ───────────────────────────────────────────────────
# Unsloth must be installed BEFORE transformers to avoid conflicts.
# We use the nightly build which supports Qwen2.5.
echo ""
echo "Installing Unsloth..."

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q

# ── 4. Core training dependencies ─────────────────────────────────────────────
# Pin versions that are confirmed working with Unsloth + Qwen2.5
pip install -q \
    "transformers>=4.45.0" \
    "datasets>=2.19.0" \
    "trl>=0.11.0" \
    "peft>=0.13.0" \
    "accelerate>=0.34.0" \
    "bitsandbytes>=0.44.0" \
    "xformers" \
    "triton" \
    "scipy" \
    "einops" \
    "sentencepiece" \
    "protobuf" \
    "tensorboard" \
    "pyyaml" \
    "huggingface_hub"

# ---> FIX INSERTED HERE: Remove conflicting torchao package <---
pip uninstall -y torchao || true
# ---------------------------------------------------------------

# ── 5. Verify Unsloth installed correctly ─────────────────────────────────────
echo ""
echo "Verifying Unsloth..."
python3 -c "
import unsloth
from unsloth import FastLanguageModel
print(f'  Unsloth version : {unsloth.__version__}')
print('  Unsloth OK ✓')
"

# ── 6. Verify Flash Attention (Unsloth installs its own optimized kernels) ─────
python3 -c "
import torch
print(f'  Flash Attn available: {torch.backends.cuda.flash_sdp_enabled()}')
print(f'  BF16 supported      : {torch.cuda.is_bf16_supported()}')
"

# ── 7. Create working directories ─────────────────────────────────────────────
mkdir -p /workspace/training/data
mkdir -p /workspace/training/runs
mkdir -p /workspace/training/logs
mkdir -p /workspace/training/merged

echo ""
echo "============================================"
echo "✅ Setup complete."
echo "   Next: bash download_data.sh"
echo "============================================"