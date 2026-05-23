#!/bin/bash
# launch.sh — One-command launcher
#
# Usage:
#   bash launch.sh                               # uses default config
#   bash launch.sh configs/qwen7b_unsloth.yaml   # uses specific config
#   bash launch.sh configs/qwen7b_unsloth.yaml --resume_from_checkpoint /path/to/checkpoint

set -e

CONFIG=${1:-"../../configs/qwen7b_unsloth.yaml"}

# ── Validate config ───────────────────────────────────────────────────────────
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config not found: $CONFIG"
    echo "Available configs:"
    ls ../../configs/*.yaml 2>/dev/null || echo "  (none)"
    exit 1
fi

# ── Check data ────────────────────────────────────────────────────────────────
TRAIN_FILE=$(python3 -c "
import yaml, os
cfg = yaml.safe_load(open('$CONFIG'))
print(os.path.expanduser(cfg['data']['train_file']))
")
VAL_FILE=$(python3 -c "
import yaml, os
cfg = yaml.safe_load(open('$CONFIG'))
print(os.path.expanduser(cfg['data']['val_file']))
")

for f in "$TRAIN_FILE" "$VAL_FILE"; do
    if [ ! -f "$f" ]; then
        echo "❌ Data file missing: $f"
        echo "   Run: bash download_data.sh"
        exit 1
    fi
done

# ── Check HuggingFace login (needed for Qwen weights) ────────────────────────
python3 -c "
from huggingface_hub import HfApi
try:
    HfApi().whoami()
    print('HuggingFace: logged in ✓')
except Exception:
    print('⚠️  Not logged in to HuggingFace.')
    print('   Run: huggingface-cli login')
"

# ── Print run summary ─────────────────────────────────────────────────────────
TRAIN_LINES=$(wc -l < "$TRAIN_FILE")
VAL_LINES=$(wc -l < "$VAL_FILE")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# FIX: Ensure logs are saved to the persistent volume, not the ephemeral ~ (root) disk
LOG_FILE=/workspace/training/logs/train_${TIMESTAMP}.log
mkdir -p /workspace/training/logs

echo ""
echo "============================================"
echo " Launching Training"
echo "============================================"
echo "  Config     : $CONFIG"
echo "  Train      : $TRAIN_LINES samples"
echo "  Val        : $VAL_LINES samples"
echo "  Log        : $LOG_FILE"
echo "============================================"
echo ""

# ── Run TensorBoard in background ────────────────────────────────────────────
OUTPUT_DIR=$(python3 -c "
import yaml, os
cfg = yaml.safe_load(open('$CONFIG'))
print(os.path.expanduser(cfg['training']['output_dir']))
")
tensorboard --logdir "$OUTPUT_DIR" --port 6006 --host 0.0.0.0 &>/dev/null &
TB_PID=$!
echo "TensorBoard running at :6006 (PID $TB_PID)"

# ── Train ─────────────────────────────────────────────────────────────────────
# Use screen so training survives SSH disconnects.
# FIX: explicitly pass the resolved config, then slice array to pass the rest of the arguments
python train.py "$CONFIG" "${@:2}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

kill $TB_PID 2>/dev/null || true

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Test  : python ../evaluation/evaluation_latest.py --adapter_path $OUTPUT_DIR/final_adapter"
    echo "  2. Merge : python merge_and_save.py --adapter_path $OUTPUT_DIR/final_adapter --output_dir /workspace/training/merged"
else
    echo "❌ Training exited with code $EXIT_CODE — check $LOG_FILE"
fi