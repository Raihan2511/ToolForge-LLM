#!/bin/bash
# download_data.sh — Transfer preprocessed JSONL files from Google Drive → RunPod
#
# Two options:
#   Option A: rclone  — full Drive access, no public link needed (recommended)
#   Option B: gdown   — requires making files publicly shareable

set -e
DATA_DIR=~/training/data
mkdir -p $DATA_DIR

echo "============================================"
echo " Data Transfer: Google Drive → RunPod"
echo "============================================"
echo ""
echo "  1) rclone  (Google OAuth — most secure)"
echo "  2) gdown   (public shareable link)"
echo ""
read -p "Choose [1/2]: " choice

if [ "$choice" == "1" ]; then
    # Configure rclone on first run
    if ! rclone listremotes | grep -q "gdrive:"; then
        echo ""
        echo "One-time Google Drive authorisation..."
        echo "Choose remote type: drive"
        echo "Leave client_id and secret blank"
        echo "Scope: 1 (full access)"
        echo "If headless (no browser on pod): choose n → it gives you a URL to open elsewhere"
        echo ""
        rclone config
    fi

    # ── Set this to your exact Drive folder name ──────────────────────────────
    GDRIVE_FOLDER="gdrive:toolcall_mixed_dataset"

    echo ""
    echo "Downloading from $GDRIVE_FOLDER ..."
    rclone copy "$GDRIVE_FOLDER/train.jsonl" $DATA_DIR --progress
    rclone copy "$GDRIVE_FOLDER/val.jsonl"   $DATA_DIR --progress

elif [ "$choice" == "2" ]; then
    pip install gdown -q
    echo ""
    echo "Paste the shareable link for train.jsonl"
    echo "(Drive → right-click → Share → Copy link)"
    read -p "train.jsonl link: " TRAIN_LINK
    read -p "val.jsonl link  : " VAL_LINK

    gdown --fuzzy "$TRAIN_LINK" -O $DATA_DIR/train.jsonl
    gdown --fuzzy "$VAL_LINK"   -O $DATA_DIR/val.jsonl
else
    echo "Invalid. Exiting."
    exit 1
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "Verifying files..."
for f in train.jsonl val.jsonl; do
    path=$DATA_DIR/$f
    if [ ! -f "$path" ]; then
        echo "❌ Missing: $path"
        exit 1
    fi
    lines=$(wc -l < "$path")
    size=$(du -h "$path" | cut -f1)
    echo "  ✓ $f — $lines lines — $size"
    head -1 "$path" | python3 -c "
import sys, json
d = json.load(sys.stdin)
src = d.get('source','?')
turns = len(d.get('conversation', []))
print(f'    JSON valid ✓  source={src}  turns={turns}')
"
done

echo ""
echo "✅ Data ready at $DATA_DIR"
echo "   Next: bash launch.sh"
