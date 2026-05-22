"""
scripts/train.py — Qwen2.5-7B-Instruct LoRA fine-tuning via Unsloth.

Usage:
    # Fresh training run:
    python scripts/train.py configs/qwen7b_unsloth.yaml

    # Resume from a checkpoint:
    python scripts/train.py configs/qwen7b_unsloth.yaml \
        --resume_from_checkpoint /workspace/training/runs/qwen7b_unsloth/checkpoint-1000

Why Unsloth over standard HF LoRA?
  - 2–4× faster training via hand-written Triton kernels (RoPE, RMSNorm, etc.)
  - ~40% less VRAM via smarter gradient checkpointing
  - No code changes needed vs standard PEFT — same API surface
  - Unsloth's QLoRA (4-bit base + bf16 adapters) is the default; you get
    near-full-precision adapter training on a quantised base for free

Key Qwen2.5-specific notes:
  - Chat format: ChatML (<|im_start|> / <|im_end|>)
  - Response template for label masking: "<|im_start|>assistant"  ← NO \n suffix
    The \n is tokenised separately from the special token string, so including
    it in response_part causes train_on_responses_only to miss the boundary on
    many samples → entire assistant turn gets masked → 0% valid output.
  - Tokenizer: tiktoken-based, pad_token must be set manually
  - output_dir uses /workspace (RunPod persistent volume), NOT ~/training/runs
"""

import os
import sys
import yaml
import logging
import argparse
import shutil
from pathlib import Path

# Add src/ to path so `from dataset import ...` resolves correctly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import train_on_responses_only

from dataset import build_dataset, load_jsonl
from callbacks import FormatCollapseCallback

# NOTE: EarlyStoppingCallback intentionally removed.
# It was firing after only 3 evals (~35% through epoch 1) because the loss
# plateau during warm-up looked like convergence. At epoch 0.35 the model is
# still actively learning. Monitor eval_loss via TensorBoard instead and stop
# manually if loss genuinely flatlines after epoch 1.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # ── Persistence: always use /workspace on RunPod, never ~ paths ──────────
    # /workspace is the only directory that survives a pod stop/restart.
    # We override whatever the yaml says and force the absolute /workspace path.
    # If you are NOT on RunPod (e.g. local machine), set env var
    # TRAINING_ROOT to override (e.g. TRAINING_ROOT=~/training).
    training_root = os.environ.get("TRAINING_ROOT", "/workspace/training")

    cfg["data"]["train_file"]      = os.path.join(training_root, "data", "train.jsonl")
    cfg["data"]["val_file"]        = os.path.join(training_root, "data", "val.jsonl")
    cfg["training"]["output_dir"]  = os.path.join(
        training_root, "runs", "qwen7b_unsloth"
    )

    logger.info(f"[Paths] train      : {cfg['data']['train_file']}")
    logger.info(f"[Paths] val        : {cfg['data']['val_file']}")
    logger.info(f"[Paths] output_dir : {cfg['training']['output_dir']}")

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Startup checks: catch path problems before we waste 10 minutes loading weights
# ──────────────────────────────────────────────────────────────────────────────

def preflight_checks(cfg: dict, resume_path: str | None) -> None:
    errors = []

    for label, key in [("train", "train_file"), ("val", "val_file")]:
        p = cfg["data"][key]
        if not os.path.isfile(p):
            errors.append(f"  ❌ {label} file not found: {p}")
        else:
            size_mb = os.path.getsize(p) / 1_048_576
            logger.info(f"  ✓ {label} file: {p}  ({size_mb:.1f} MB)")

    out = cfg["training"]["output_dir"]
    try:
        Path(out).mkdir(parents=True, exist_ok=True)
        test_file = Path(out) / ".write_test"
        test_file.touch()
        test_file.unlink()
        logger.info(f"  ✓ output_dir writable: {out}")
    except OSError as e:
        errors.append(f"  ❌ output_dir not writable: {out} — {e}")

    # If resuming, verify the checkpoint directory actually exists
    if resume_path is not None:
        if not os.path.isdir(resume_path):
            errors.append(
                f"  ❌ resume_from_checkpoint not found: {resume_path}\n"
                f"     Available checkpoints in {out}:\n"
                + "\n".join(
                    f"       {p}" for p in sorted(Path(out).glob("checkpoint-*"))
                )
            )
        else:
            # Confirm trainer_state.json exists — without it the Trainer can't
            # restore step count, optimizer state, or scheduler state
            state_file = Path(resume_path) / "trainer_state.json"
            if not state_file.exists():
                errors.append(
                    f"  ❌ {resume_path}/trainer_state.json missing — "
                    "checkpoint may be corrupt or incomplete"
                )
            else:
                import json
                with open(state_file) as f:
                    state = json.load(f)
                resumed_step = state.get("global_step", "?")
                logger.info(
                    f"  ✓ Checkpoint valid: {resume_path} "
                    f"(global_step={resumed_step})"
                )

    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Label-masking boundary verification
#
# FIX — Label Masking Desync
# ──────────────────────────
# Root cause: train_on_responses_only searches for the response_part string
# as a *token sequence* in the already-tokenised input_ids. Qwen2.5's tiktoken
# tokeniser splits "<|im_start|>assistant\n" into:
#
#     token("<|im_start|>")  +  token("assistant")  +  token("\n")
#
# When response_part is set to "<|im_start|>assistant\n", the library encodes
# that string and gets [im_start_id, assistant_id, newline_id]. The newline_id
# at the end of the search pattern must exactly match what follows "assistant"
# in every sample. If the template ever emits a different newline encoding
# (e.g. \r\n, or a space before \n) the pattern is not found, the boundary is
# missed, and the ENTIRE assistant turn is masked with -100 — the model sees
# no training signal at all, producing 0% valid output.
#
# Fix: set response_part = "<|im_start|>assistant" (no trailing \n).
# The library will find the token for <|im_start|> followed by the token for
# "assistant", then label everything AFTER that match (including the \n and
# the actual response) as the training target. This is robust regardless of
# how whitespace is tokenised.
# ──────────────────────────────────────────────────────────────────────────────

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART    = "<|im_start|>assistant"   # ← no \n — see comment above


def verify_label_masking(trainer, tokenizer) -> None:
    """
    Run one forward pass on a tiny synthetic sample and confirm that at least
    some tokens have labels != -100 (i.e. the response boundary was found).
    Exits with a clear error if masking is broken before wasting hours of GPU.
    """
    logger.info(f"\n{'─'*60}")
    logger.info("🏷️  Verifying label masking with synthetic sample...")

    synthetic = (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHello.<|im_end|>\n"
        f"{RESPONSE_PART}\nHi there!<|im_end|>"
    )
    logger.info(f"  Synthetic input (raw):\n    {synthetic!r}")

    enc       = tokenizer(synthetic, return_tensors="pt")
    input_ids = enc["input_ids"]
    logger.info(f"  Tokenized shape : {input_ids.shape}")
    logger.info(f"  Token IDs       : {input_ids[0].tolist()}")
    logger.info(f"  Decoded tokens  : {[tokenizer.decode([t]) for t in input_ids[0].tolist()]}")

    collated = trainer.data_collator([{
        "input_ids": input_ids[0].tolist(),
        "labels":    input_ids[0].tolist(),
    }])
    labels = collated["labels"]

    visible = (labels != -100).sum().item()
    total   = labels.numel()

    # Show which tokens are masked vs visible
    label_list = labels[0].tolist() if labels.dim() > 1 else labels.tolist()
    token_list = input_ids[0].tolist()
    logger.info(f"\n  Label masking breakdown (✓=visible, ✗=masked):")
    for i, (tok_id, lbl) in enumerate(zip(token_list, label_list)):
        tok_str = tokenizer.decode([tok_id])
        marker  = "✓" if lbl != -100 else "✗"
        logger.info(f"    [{marker}] pos={i:3d}  id={tok_id:6d}  label={lbl:6d}  token={tok_str!r}")

    if visible == 0:
        logger.error(
            "❌ LABEL MASKING BROKEN: 0 tokens have a training label after masking.\n"
            f"  response_part = {RESPONSE_PART!r}\n"
            "  The boundary token sequence was not found in the tokenised input.\n"
            "  Check that RESPONSE_PART matches the exact token(s) produced by\n"
            "  tokenizer.encode('<|im_start|>assistant')."
        )
        sys.exit(1)
    else:
        logger.info(
            f"\n  ✅ Label masking OK — {visible}/{total} tokens visible to loss "
            f"({visible/total:.1%})"
        )
    logger.info(f"{'─'*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Model + tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    mcfg = cfg["model"]
    lcfg = cfg["lora"]

    logger.info(f"Loading model: {mcfg['name']}")
    logger.info(f"  max_seq_length = {mcfg['max_seq_length']}")
    logger.info(f"  load_in_4bit   = {mcfg.get('load_in_4bit', True)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = mcfg["name"],
        max_seq_length = mcfg["max_seq_length"],
        load_in_4bit   = mcfg.get("load_in_4bit", True),
        dtype          = None,   # auto-detect; bf16 on L40S
    )

    # ── Tokenizer diagnostics ─────────────────────────────────────────────
    logger.info(f"\n{'─'*60}")
    logger.info(f"🔤 Tokenizer loaded")
    logger.info(f"  Vocab size     : {tokenizer.vocab_size:,}")
    logger.info(f"  Model max len  : {tokenizer.model_max_length:,}")
    logger.info(f"  pad_token      : {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    logger.info(f"  eos_token      : {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    logger.info(f"  bos_token      : {tokenizer.bos_token!r} (id={getattr(tokenizer, 'bos_token_id', None)})")

    # Qwen2.5 tokenizer has no pad token by default.
    # Right-padding is mandatory for causal LM — left-padding shifts the
    # response template offset and breaks the boundary search.
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"  ⚠️  pad_token was None → set to eos_token: {tokenizer.pad_token!r}")
    tokenizer.padding_side = "right"
    logger.info(f"  padding_side   : {tokenizer.padding_side}")

    # ── Verify critical token sequences ───────────────────────────────────
    response_ids = tokenizer.encode(RESPONSE_PART, add_special_tokens=False)
    instruction_ids = tokenizer.encode(INSTRUCTION_PART, add_special_tokens=False)
    logger.info(f"\n  Token IDs for label masking boundaries:")
    logger.info(f"    RESPONSE_PART    = {RESPONSE_PART!r}")
    logger.info(f"    → token_ids      = {response_ids}")
    logger.info(f"    → decoded back   = {tokenizer.decode(response_ids)!r}")
    logger.info(f"    INSTRUCTION_PART = {INSTRUCTION_PART!r}")
    logger.info(f"    → token_ids      = {instruction_ids}")
    logger.info(f"    → decoded back   = {tokenizer.decode(instruction_ids)!r}")
    logger.info(f"{'─'*60}")

    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = lcfg["r"],
        lora_alpha                 = lcfg["alpha"],
        lora_dropout               = lcfg.get("dropout", 0.05),
        bias                       = lcfg.get("bias", "none"),
        target_modules             = lcfg["target_modules"],
        use_gradient_checkpointing = lcfg.get("use_gradient_checkpointing", "unsloth"),
        random_state               = lcfg.get("random_state", 42),
        use_rslora                 = False,
    )

    # ── Model architecture summary ────────────────────────────────────────
    model.print_trainable_parameters()
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params     : {total_params:,}")
    logger.info(f"  Trainable params : {trainable:,} ({trainable/total_params:.2%})")
    logger.info(f"  Model dtype      : {next(model.parameters()).dtype}")
    logger.info(f"  Device           : {next(model.parameters()).device}")

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Training arguments
# ──────────────────────────────────────────────────────────────────────────────

def build_training_args(cfg: dict) -> TrainingArguments:
    t   = cfg["training"]
    out = cfg["training"]["output_dir"]

    return TrainingArguments(
        output_dir = out,

        # Batch
        per_device_train_batch_size = t["per_device_train_batch_size"],
        per_device_eval_batch_size  = t.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps = t["gradient_accumulation_steps"],

        # Schedule
        num_train_epochs   = t["num_epochs"],
        learning_rate      = t["learning_rate"],
        lr_scheduler_type  = t.get("lr_scheduler", "cosine"),
        warmup_ratio       = t.get("warmup_ratio", 0.05),
        weight_decay       = t.get("weight_decay", 0.01),
        max_grad_norm      = t.get("max_grad_norm", 1.0),

        # Precision
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),

        # Optimiser — Unsloth's fused 8-bit AdamW
        optim = t.get("optim", "adamw_8bit"),

        # Eval + save
        eval_strategy          = t.get("eval_strategy", "steps"),
        eval_steps             = t.get("eval_steps", 500),
        save_strategy          = t.get("save_strategy", "steps"),
        save_steps             = t.get("save_steps", 500),
        save_total_limit       = t.get("save_total_limit", 3),
        load_best_model_at_end = True,
        metric_for_best_model  = "eval_loss",
        greater_is_better      = False,

        # Logging
        logging_steps = t.get("logging_steps", 20),
        report_to     = t.get("report_to", "tensorboard"),

        # Misc
        seed                   = t.get("seed", 42),
        remove_unused_columns  = False,
        dataloader_num_workers = 4,
        dataloader_pin_memory  = True,
        #dataset_num_proc       = 4,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Path to YAML config file",
    )
    # CHANGE 1: Added --resume_from_checkpoint argument.
    # Default is None → fresh training run (existing behaviour unchanged).
    # When provided, trainer.train() restores optimizer state, scheduler state,
    # RNG state, and data loader position — no repeated samples, correct step count.
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help=(
            "Path to a checkpoint directory to resume training from. "
            "Example: /workspace/training/runs/qwen7b_unsloth/checkpoint-1000"
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    logger.info(f"Config        : {args.config}")
    logger.info(f"Model         : {cfg['model']['name']}")
    logger.info(f"4-bit         : {cfg['model'].get('load_in_4bit', True)}")
    logger.info(f"LoRA r        : {cfg['lora']['r']}  alpha: {cfg['lora']['alpha']}")
    logger.info(f"Epochs        : {cfg['training']['num_epochs']}")
    logger.info(f"response_part : {RESPONSE_PART!r}  (no trailing \\n — desync fix)")

    # Log resume state clearly so you can confirm it's picked up correctly
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from : {args.resume_from_checkpoint}")
    else:
        logger.info("Resuming from : None (fresh run)")

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    preflight_checks(cfg, args.resume_from_checkpoint)

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── Build datasets ────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_ds = build_dataset(cfg["data"]["train_file"], tokenizer)
    val_ds   = build_dataset(cfg["data"]["val_file"],   tokenizer)

    # Keep raw val dicts for the format-collapse callback
    val_raw = load_jsonl(cfg["data"]["val_file"])

    # ── Log dataset composition ───────────────────────────────────────────────
    sources = {}
    for s in val_raw:
        src = s.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    logger.info(f"Val set source breakdown: {sources}")
    for src, count in sources.items():
        pct = count / len(val_raw) * 100
        if src == "apigen" and pct > 80:
            logger.warning(
                f"⚠️  Val set is {pct:.0f}% apigen. "
                "FormatCollapseCallback will have few UltraChat negatives — "
                "add more UltraChat samples if conv_rate warnings appear."
            )
        if src == "ultrachat" and pct > 80:
            logger.warning(
                f"⚠️  Val set is {pct:.0f}% ultrachat. "
                "Tool-call format checks will have few apigen samples."
            )

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = build_training_args(cfg)

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_ds,
        eval_dataset       = val_ds,
        dataset_text_field = "text",
        max_seq_length     = cfg["model"]["max_seq_length"],
        dataset_num_proc   = 4,
        packing            = False,
        # packing=False is mandatory with train_on_responses_only.
        # Packing concatenates multiple samples into one sequence, which causes
        # the boundary token to appear multiple times — the masking logic then
        # marks the wrong spans as training targets.
        args               = training_args,
    )

    # ── Label masking: train on assistant responses only ──────────────────────
    # response_part = "<|im_start|>assistant" (no trailing \n)
    # See RESPONSE_PART comment above for the full explanation.
    trainer = train_on_responses_only(
        trainer,
        instruction_part = INSTRUCTION_PART,
        response_part    = RESPONSE_PART,
    )

    # ── Verify masking before wasting GPU time ────────────────────────────────
    verify_label_masking(trainer, tokenizer)

    # ── First-batch inspection ────────────────────────────────────────────────
    # Peek at the actual tensors the model will see on its first step.
    logger.info(f"\n{'─'*60}")
    logger.info("🔍 First-batch inspection (what the model actually sees):")
    try:
        dl      = trainer.get_train_dataloader()
        batch   = next(iter(dl))
        logger.info(f"  Batch keys       : {list(batch.keys())}")
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                logger.info(f"  {k:18s}: shape={v.shape}  dtype={v.dtype}  device={v.device}")

        # Show first sample's input_ids decoded
        if "input_ids" in batch:
            ids    = batch["input_ids"][0]
            n_pad  = (ids == tokenizer.pad_token_id).sum().item()
            n_real = ids.shape[0] - n_pad
            logger.info(f"  Sample 0: {n_real} real tokens + {n_pad} padding tokens")
            decoded = tokenizer.decode(ids[:80], skip_special_tokens=False)
            logger.info(f"  First 80 tokens decoded:\n    {decoded[:300]!r}")

        # Show label stats for first sample
        if "labels" in batch:
            lbl       = batch["labels"][0]
            n_masked  = (lbl == -100).sum().item()
            n_visible = lbl.shape[0] - n_masked
            logger.info(f"  Labels: {n_visible} visible (trained on), {n_masked} masked (-100)")
            pct = n_visible / lbl.shape[0] * 100
            if pct < 5:
                logger.warning(f"  ⚠️  Only {pct:.1f}% of tokens are visible — label masking may be too aggressive")
            elif pct > 80:
                logger.warning(f"  ⚠️  {pct:.1f}% of tokens visible — system/user turns may not be masked")
            else:
                logger.info(f"  ✅ Visible ratio looks healthy: {pct:.1f}%")
    except Exception as e:
        logger.warning(f"  ⚠️  Could not inspect first batch: {e}")
    logger.info(f"{'─'*60}")
    # Samples apigen + ultrachat from val every 2nd eval cycle.
    # Fires warnings if:
    #   - tool_rate < 50%  → model not learning JSON format
    #   - conv_rate < 50%  → model outputting JSON for conversational queries
    # Fixed: uses FastLanguageModel.for_inference/for_training instead of
    # model.eval()/model.train() — prevents KV cache shape crash during .generate()
    format_callback = FormatCollapseCallback(
        val_raw   = val_raw,
        tokenizer = tokenizer,
        model     = model,
    )
    trainer.add_callback(format_callback)

    # CHANGE 2: EarlyStoppingCallback removed.
    # It was firing at step 1000 (~35% through epoch 1) because the cosine
    # warmup plateau looked like convergence. Removed in favour of manual
    # monitoring via TensorBoard (port 6006 in RunPod).

    # ── Log GPU state before training ─────────────────────────────────────────
    gpu_stats        = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 1)
    max_memory       = round(gpu_stats.total_memory / 1024**3, 1)
    logger.info(f"GPU: {gpu_stats.name}  Total VRAM: {max_memory} GB")
    logger.info(f"VRAM reserved before training: {start_gpu_memory} GB")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training...")

    # CHANGE 3: Pass resume_from_checkpoint to trainer.train().
    # When None  → fresh run (existing behaviour, no change).
    # When a path → Trainer restores:
    #     - optimizer state  (AdamW momentum/variance)
    #     - LR scheduler state (cosine position)
    #     - RNG state (torch, numpy, python random)
    #     - data loader position (no repeated or skipped samples)
    #     - global_step counter (TensorBoard/logging continues from correct step)
    trainer_stats = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # ── Post-training GPU report ──────────────────────────────────────────────
    peak_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 1)
    runtime_min = trainer_stats.metrics["train_runtime"] / 60
    logger.info(f"Peak VRAM used : {peak_memory} GB / {max_memory} GB")
    logger.info(f"Training time  : {runtime_min:.1f} minutes")

    # ── Save final LoRA adapter ───────────────────────────────────────────────
    # Saved inside /workspace so it survives pod restart.
    adapter_dir = Path(cfg["training"]["output_dir"]) / "final_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    shutil.copy(args.config, adapter_dir / "training_config.yaml")
    logger.info(f"✅ Adapter saved → {adapter_dir}")

    logger.info("")
    logger.info("Next steps:")
    logger.info(
        f"  Evaluate : python scripts/evaluate.py "
        f"--adapter_path {adapter_dir} --skip_base"
    )
    logger.info(
        f"  Merge    : python scripts/merge_and_save.py "
        f"--adapter_path {adapter_dir} "
        f"--output_dir /workspace/training/merged"
    )


if __name__ == "__main__":
    main()