"""
src/callbacks.py
FormatCollapseCallback — monitors tool-call vs conversational output
format during training and fires warnings before training goes off the rails.

Checks every FORMAT_CHECK_EVERY_N_EVALS eval cycles:
  - Tool-call samples (source=apigen)    → must produce JSON with tool_calls key
  - Negative samples  (source=ultrachat) → must NOT produce JSON

Uses plain model.eval() / model.train() — NOT Unsloth's for_inference/for_training.

Why NOT for_inference/for_training inside a callback:
  FastLanguageModel.for_inference() is designed for cold model loading in a
  standalone inference script. When called mid-training inside an active Trainer
  loop it causes a KV cache shape mismatch: [1, 28, 1, 128] vs [1, 28, N, 128].
  Inside the Trainer loop Unsloth's kernels are already correctly initialised —
  plain model.eval() is all that is needed to safely call .generate().
  The try/finally guarantees model.train() always restores training mode even
  if an exception fires mid-check.
"""

import json
import random
import logging
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

# NOTE: No Unsloth import here intentionally.
# FastLanguageModel.for_inference() / for_training() caused a KV cache shape
# crash ([1,28,1,128] vs [1,28,N,128]) when called mid-training because
# Unsloth's kernel swap is designed for cold model loading, not for toggling
# inside an active training loop.
# Plain model.eval() / model.train() is correct here — inside the Trainer loop
# Unsloth's kernels are already initialised; eval()/train() only toggle
# Dropout/BatchNorm which is all we need for safe generate() during a callback.

logger = logging.getLogger(__name__)

FORMAT_CHECK_EVERY_N_EVALS = 1
N_SAMPLES_PER_TYPE         = 4


class FormatCollapseCallback(TrainerCallback):

    def __init__(self, val_raw: list[dict], tokenizer, model):
        self.tokenizer  = tokenizer
        self.model      = model
        self.eval_count = 0

        self.tool_samples = [s for s in val_raw if s.get("source") == "apigen"]
        self.conv_samples = [s for s in val_raw if s.get("source") == "ultrachat"]

        logger.info(
            f"FormatCollapseCallback: "
            f"{len(self.tool_samples)} tool samples, "
            f"{len(self.conv_samples)} conv samples available for format checking"
        )

        if not self.tool_samples:
            logger.warning(
                "⚠️  FormatCollapseCallback: no 'apigen' samples found in val_raw. "
                "tool_json rate will always show 0/0. "
                "Verify your val.jsonl has records with source='apigen'."
            )
        if not self.conv_samples:
            logger.warning(
                "⚠️  FormatCollapseCallback: no 'ultrachat' samples found in val_raw. "
                "conv_plain rate will always show 0/0. "
                "Verify your val.jsonl has records with source='ultrachat'."
            )

    def _build_prompt_turns(self, sample: dict) -> list[dict] | None:
        conversation = sample.get("conversation", [])
        if not conversation:
            return None
        turns = list(conversation)
        while turns and turns[-1]["role"] == "assistant":
            turns.pop()
        if not turns:
            return None
        return turns

    def _generate(self, prompt_turns: list[dict]) -> str | None:
        try:
            prompt = self.tokenizer.apply_chat_template(
                prompt_turns,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1900,
            ).to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Prevents KV cache shape crash when called
                                      # mid-training with Unsloth's training kernels
                                      # active. The training kernel initialises the
                                      # cache as [1,28,1,128] which fails to broadcast
                                      # against the full sequence [1,28,N,128].
                                      # Disabling the cache bypasses this entirely.
                )

            return self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

        except Exception as e:
            logger.warning(f"[FormatCheck] Generation failed: {e}")
            return None

    def _check_tool(self, generated: str) -> bool:
        try:
            parsed = json.loads(generated)
            calls  = parsed.get("tool_calls")
            return isinstance(calls, list) and len(calls) > 0
        except (json.JSONDecodeError, AttributeError):
            return False

    def _check_conv(self, generated: str) -> bool:
        try:
            json.loads(generated)
            return False   # Parsed as JSON → mode collapse
        except json.JSONDecodeError:
            return True    # Plain text → correct

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.eval_count += 1
        if self.eval_count % FORMAT_CHECK_EVERY_N_EVALS != 0:
            return

        # model.eval() disables Dropout for generation — safe inside the Trainer
        # loop because Unsloth's kernels are already correctly initialised.
        # model.train() in the finally block always restores training mode even
        # if an exception fires mid-check (e.g. OOM, malformed sample).
        logger.info(f"\n[FormatCheck] 🔄 Switching to eval mode (eval #{self.eval_count}, step={state.global_step})")
        try:
            self.model.eval()
            logger.info("[FormatCheck] ✅ model.eval() — running format checks")
            self._run_checks(state)
        finally:
            self.model.train()
            logger.info("[FormatCheck] 🔄 Switched back to train mode")

    def _run_checks(self, state: TrainerState) -> None:
        tool_ok, tool_total = 0, 0
        conv_ok, conv_total = 0, 0

        tool_batch = random.sample(
            self.tool_samples, min(N_SAMPLES_PER_TYPE, len(self.tool_samples))
        )
        conv_batch = random.sample(
            self.conv_samples, min(N_SAMPLES_PER_TYPE, len(self.conv_samples))
        )

        for idx, (sample, is_tool) in enumerate(
            [(s, True)  for s in tool_batch] +
            [(s, False) for s in conv_batch]
        ):
            label = "tool" if is_tool else "conv"
            prompt_turns = self._build_prompt_turns(sample)
            if prompt_turns is None:
                logger.info(f"  [{label} #{idx}] ⏭️  skipped — no valid prompt turns")
                continue

            generated = self._generate(prompt_turns)
            if generated is None:
                logger.info(f"  [{label} #{idx}] ⏭️  skipped — generation returned None")
                continue

            if is_tool:
                tool_total += 1
                passed = self._check_tool(generated)
                if passed:
                    tool_ok += 1
                    logger.info(f"  [tool #{idx}] ✅ PASS — {generated[:100]!r}")
                else:
                    logger.info(f"  [tool #{idx}] ❌ FAIL — {generated[:100]!r}")
            else:
                conv_total += 1
                passed = self._check_conv(generated)
                if passed:
                    conv_ok += 1
                    logger.info(f"  [conv #{idx}] ✅ PASS (plain text) — {generated[:100]!r}")
                else:
                    logger.info(f"  [conv #{idx}] ❌ FAIL (JSON collapse) — {generated[:100]!r}")

        tool_rate = tool_ok / tool_total if tool_total else 0.0
        conv_rate = conv_ok / conv_total if conv_total else 0.0

        logger.info(
            f"[FormatCheck step={state.global_step}] "
            f"tool_json={tool_rate:.0%} ({tool_ok}/{tool_total})  "
            f"conv_plain={conv_rate:.0%} ({conv_ok}/{conv_total})"
        )

        if tool_total == 0:
            logger.warning(
                "[FormatCheck] No apigen samples evaluated — "
                "check val_raw source labels."
            )
        elif tool_rate < 0.5:
            logger.warning(
                f"⚠️  TOOL FORMAT COLLAPSE at step {state.global_step}: "
                f"{tool_rate:.0%} ({tool_ok}/{tool_total}) tool-call samples "
                "produced valid JSON. Possible causes:\n"
                "  1. Label masking broken — check response_part token alignment\n"
                "  2. Not enough training yet — check if loss is still falling\n"
                "  3. LoRA rank too low — try r=64 if using lower"
            )

        if conv_total == 0:
            logger.warning(
                "[FormatCheck] No ultrachat samples evaluated — "
                "check val_raw source labels."
            )
        elif conv_rate < 0.5:
            logger.warning(
                f"⚠️  CONV MODE COLLAPSE at step {state.global_step}: "
                f"{conv_rate:.0%} ({conv_ok}/{conv_total}) conversational samples "
                "produced plain text. Model outputting JSON for empty-tools prompts.\n"
                "  → Increase UltraChat negative sample ratio in dataset"
            )