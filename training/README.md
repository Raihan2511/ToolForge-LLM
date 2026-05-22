# Qwen2.5-7B-Instruct Tool-Calling Fine-Tune — RunPod + Unsloth

## Hardware
- GPU  : NVIDIA L40S 48GB
- Pod  : RunPod PyTorch 2.4.0 (CUDA 12.4)
- Disk : 80GB+ recommended

## File Structure
```
~/training/
├── setup.sh                         # Step 1: install everything
├── download_data.sh                 # Step 2: pull data from Google Drive
├── launch.sh                        # Step 4: one-command train
│
├── configs/
│   └── qwen7b_unsloth.yaml          # All hyperparameters
│
├── src/
│   ├── dataset.py                   # JSONL loader + chat template formatting
│   └── callbacks.py                 # Format-collapse early warning
│
├── scripts/
│   ├── train.py                     # Main training script (Unsloth + SFTTrainer)
│   ├── test_inference.py            # 5-case inference test after training
│   └── merge_and_save.py           # Merge adapter → full model / GGUF
│
└── data/                            # Created by download_data.sh
    ├── train.jsonl
    └── val.jsonl
```

---

## Exact Commands (in order)

```bash
# 1. Upload this folder to RunPod, then SSH in and:
cd ~/training
bash setup.sh

# 2. HuggingFace login (needed for Qwen weights)
huggingface-cli login
# paste your token when prompted

# 3. Pull your preprocessed dataset from Google Drive
bash download_data.sh

# 4. Train (single L40S, ~4–6 hours for 3 epochs on 100k samples)
bash launch.sh

# 5. Test the fine-tuned model
python scripts/test_inference.py \
    --adapter_path ~/training/runs/qwen7b_unsloth/final_adapter

# 6. Merge adapter into base model (needed for vLLM / llama.cpp)
python scripts/merge_and_save.py \
    --adapter_path ~/training/runs/qwen7b_unsloth/final_adapter \
    --output_dir   ~/training/merged

# Optional: also export GGUF
python scripts/merge_and_save.py \
    --adapter_path ~/training/runs/qwen7b_unsloth/final_adapter \
    --output_dir   ~/training/merged \
    --export_gguf
```

---

## Monitor Training Live

TensorBoard is launched automatically by `launch.sh` on port 6006.
In RunPod: go to your pod → Connect → HTTP Service → port 6006.

Metrics to watch:
- `train/loss`     — should steadily decrease
- `eval/loss`      — should follow train loss; if it diverges → overfitting
- `train/grad_norm` — should stay under ~2.0; if it spikes → lower max_grad_norm

---

## Expected VRAM Usage (L40S 48GB)

| Phase             | VRAM       |
|-------------------|------------|
| Model load (4bit) | ~8 GB      |
| + LoRA adapters   | ~10 GB     |
| + Activations     | ~18 GB     |
| Peak during train | ~22–26 GB  |
| Headroom          | ~22 GB     |

You have significant headroom. If training is slow:
- Increase `per_device_train_batch_size` to 6 or 8
- Reduce `gradient_accumulation_steps` proportionally

---

## Key Design Decisions

**`train_on_responses_only`** (Unsloth's label masking)
Replaces `DataCollatorForCompletionOnlyLM`. Finds `<|im_start|>user\n` and
`<|im_start|>assistant\n` in each tokenized sample and masks everything
before the assistant turn with `-100`. The model only trains on what it
should generate — not on system prompts or user queries.

**`use_gradient_checkpointing: "unsloth"`**
Unsloth's custom backward pass — 30% less VRAM than standard gradient
checkpointing with no speed penalty (unlike standard which is ~20% slower).

**`packing: False`**
Required when using `train_on_responses_only`. Packing merges multiple
short samples into one sequence for efficiency, but this breaks the
response template search because the boundary tokens appear multiple times.

**`optim: "adamw_8bit"`**
Unsloth's own fused 8-bit AdamW. Faster and lower memory than
`paged_adamw_8bit` (which is designed for standard PEFT, not Unsloth).

---

## Troubleshooting

| Error | Fix |
|---|---|
| `CUDA out of memory` | Reduce `per_device_train_batch_size` to 2, increase `gradient_accumulation_steps` to 16 |
| `unsloth not found` | Re-run `bash setup.sh` — Unsloth may have failed silently |
| `No module named 'src'` | Run scripts from `~/training/`, not from inside `scripts/` |
| `adapter_config.json not found` | Training didn't finish — check the log file |
| Format collapse warning fires | Check `conv_rate` — if UltraChat negatives aren't working, verify `formatted_tools="[]"` in your Colab preprocessing |
