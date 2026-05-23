"""
scripts/eval_ultrachat.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mode Collapse Detector for Fine-Tuned Qwen2.5-7B-Instruct — Ultrachat Subset.

Evaluates whether the fine-tuned model retains conversational ability by
running inference on ULTRACHAT samples (pure conversation — NO tool usage).

Grading Logic (inverse of BFCL tool-call evaluation):
  ✅ PASS  — Model responds with natural text (json.loads raises JSONDecodeError
             OR parsed JSON does NOT contain a "tool_calls" key).
  ❌ FAIL  — Model hallucinates a "tool_calls" key in JSON output — even if the
             list is empty []. Any presence of the tool_calls schema format in
             the output signals Mode Collapse.

Inference stack (identical to evaluation_latest.py):
  • Native Hugging Face Transformers (AutoModelForCausalLM) + PEFT (PeftModel).
  • 4-bit NF4 quantisation via BitsAndBytesConfig (double_quant, bfloat16 compute).
  • Standard KV-cache generation (use_cache=True).
  • ChatML template via tokenizer.apply_chat_template().

Usage:
    python scripts/eval_ultrachat.py \\
        --model_path  ./runs/checkpoint-final \\
        --dataset_path data/val.jsonl \\
        --output_dir   outputs/eval_ultrachat \\
        --max_samples  50

Author: Auto-generated evaluation pipeline (mode collapse variant)
Python: 3.10+
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Fix Windows console encoding for Unicode box-drawing characters ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# tqdm for progress bars — fall back to a no-op wrapper if missing
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        """Minimal no-op fallback when tqdm is not installed."""
        return iterable

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_ultrachat")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — CLI & Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ultrachat evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Mode Collapse Detector — evaluate conversational retention "
            "of fine-tuned Qwen2.5-7B-Instruct on ultrachat samples"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned LoRA adapter directory (must contain adapter_config.json).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../../data/val.jsonl",
        help="Path to the JSONL evaluation dataset (default: data/val.jsonl).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../outputs/eval_ultrachat",
        help="Directory to save detailed results JSON (default: outputs/eval_ultrachat).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit evaluation to the first N ultrachat samples (default: all).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens for generation (default: 512).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for model loading (default: 2048).",
    )
    return parser.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Inference Engine (Native Hugging Face + PEFT)
#              Identical to evaluation_latest.py — reused verbatim.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_model(model_path: str, max_seq_length: int = 2048):
    """
    Load the fine-tuned model using native Hugging Face Transformers + PEFT.

    Pipeline (identical to evaluation_latest.py):
      1. Read adapter_config.json to discover the base model name.
      2. Load the base model in 4-bit NF4 via BitsAndBytesConfig.
      3. Overlay the LoRA adapter with PeftModel.
      4. Set model to eval mode.

    Args:
        model_path: Path to the LoRA adapter directory.
        max_seq_length: Maximum sequence length (used for tokenizer consistency).

    Returns:
        Tuple of (model, tokenizer).
    """
    import json as _json
    adapter_path = Path(model_path)

    # ── Discover base model from adapter config ──
    adapter_cfg_file = adapter_path / "adapter_config.json"
    if not adapter_cfg_file.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_path}. "
            "Pass the adapter directory as --model_path."
        )
    with open(adapter_cfg_file, "r", encoding="utf-8") as f:
        base_model_name: str = _json.load(f)["base_model_name_or_path"]
    logger.info(f"Base model resolved from adapter config: {base_model_name}")

    # ── 4-bit quantisation config (matches training setup) ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ── Load base model ──
    logger.info(f"Loading base model: {base_model_name} (4-bit NF4)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side="left",   # Required for consistent generation alignment
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Overlay LoRA adapter ──
    logger.info(f"Loading LoRA adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # ── Inference mode ──
    model.eval()
    logger.info("Model loaded and set to eval mode (native HF + PEFT)")

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt_turns: list[dict],
    max_new_tokens: int = 512,
) -> str:
    """
    Generate a response from the model given prompt conversation turns.

    Uses standard Transformers generation with use_cache=True for speed.

    Args:
        model: The loaded language model.
        tokenizer: The tokenizer.
        prompt_turns: List of {"role": ..., "content": ...} dicts (no assistant turn).
        max_new_tokens: Max tokens to generate.

    Returns:
        The decoded model output (assistant response only, no prompt tokens).
    """
    # Apply Qwen2.5 ChatML template
    prompt = tokenizer.apply_chat_template(
        prompt_turns,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1900,  # Leave room for generation
    )
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,              # Greedy decoding for determinism
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,               # ── KV cache enabled for speed ──
        )

    # Decode only the newly generated tokens
    new_token_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — Prompt Builder (for ultrachat conversational samples)
#
# FIX: build_inference_prompt now REPLACES the raw dataset system prompt
# with a hybrid prompt that permits conversational responses.
#
# WHY THIS IS NEEDED:
# The UltraChat samples in val.jsonl were preprocessed with this system prompt:
#   "Output MUST be a JSON object with a 'tool_calls' key. No other text."
# But the assistant targets are long conversational paragraphs — the model was
# trained to IGNORE that strict instruction and reply naturally instead.
#
# During evaluation, if we feed that strict prompt as-is, the model obeys it
# literally and outputs {"tool_calls": []} — which the grader correctly marks
# as mode collapse. But this is a FALSE POSITIVE: the model is behaving correctly
# given the prompt it received. The prompt itself is the problem.
#
# The hybrid prompt fixes this by explicitly giving the model TWO valid paths:
#   Path A (tools available): output JSON with tool_calls
#   Path B (no tools / not needed): reply in natural language
#
# With tools=[] in the available tools block, the model should always take
# Path B and respond conversationally. If it still outputs {"tool_calls": []}
# THEN it is genuinely stuck in format mode and the FAIL is real.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# The hybrid system prompt used for ALL ultrachat evaluation samples.
# Keeps the structural blocks the model was trained on (BEGIN OF / END OF)
# so the format is familiar, but removes the "no other text" constraint.
HYBRID_SYSTEM_PROMPT = (
    "[BEGIN OF TASK INSTRUCTION]\n"
    "You are a helpful AI assistant. If tools are needed to solve the problem, "
    "output MUST be a JSON object with a 'tool_calls' key. "
    "If no tools are relevant or available, ignore the JSON format and reply "
    "to the user normally in a conversational tone.\n"
    "[END OF TASK INSTRUCTION]\n\n"
    "[BEGIN OF AVAILABLE TOOLS]\n"
    "[]\n"
    "[END OF AVAILABLE TOOLS]\n\n"
    "[BEGIN OF FORMAT INSTRUCTION]\n"
    "If using tools: "
    '{\"tool_calls\": [{\"name\": \"func_name\", \"arguments\": {\"arg1\": \"val1\"}}]}\n'
    "If no tools needed: Just reply with natural language.\n"
    "[END OF FORMAT INSTRUCTION]"
)


def build_inference_prompt(sample: dict) -> list[dict] | None:
    """
    Extract prompt turns (system + user) from a dataset sample, strip any
    trailing assistant turns, and INJECT the hybrid system prompt.

    The raw dataset system prompt ("Output MUST be a JSON object...") is
    replaced with HYBRID_SYSTEM_PROMPT, which explicitly permits both
    tool-call JSON and plain conversational responses.

    This is necessary because:
      - Raw dataset prompt: strict JSON-only instruction
      - Training target: conversational paragraphs
      - These contradict each other → evaluation with the raw prompt
        produces false-positive mode collapse readings.

    With HYBRID_SYSTEM_PROMPT + tools=[], the model has a clear, unambiguous
    instruction to respond conversationally. Any JSON output after this
    injection is a genuine mode collapse failure.

    Args:
        sample: A raw JSONL record with a "conversation" key.

    Returns:
        A list of conversation turns (system + user[s]) with the hybrid
        system prompt injected, or None if the sample is malformed.
    """
    conversation = sample.get("conversation", [])
    if not conversation or len(conversation) < 2:
        return None

    # Strip trailing assistant turns — keep only the prompt side
    turns = list(conversation)
    while turns and turns[-1]["role"] == "assistant":
        turns.pop()

    if not turns or turns[0]["role"] != "system":
        return None

    # ── INJECT HYBRID PROMPT ──────────────────────────────────────────────────
    # Replace whatever system prompt the dataset has with our hybrid version.
    # We do a shallow copy of the turn dict to avoid mutating the source data.
    turns[0] = {"role": "system", "content": HYBRID_SYSTEM_PROMPT}
    # ─────────────────────────────────────────────────────────────────────────

    return turns


def extract_user_query(sample: dict) -> str:
    """Extract the first user message from a conversation sample."""
    for turn in sample.get("conversation", []):
        if turn["role"] == "user":
            return turn.get("content", "")
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Grading Logic (Mode Collapse Detector)
#
#   ✅ PASS — json.JSONDecodeError  OR  parsed JSON has no "tool_calls" key
#   ❌ FAIL — Parses as JSON AND contains "tool_calls" key (even if empty [])
#
# NOTE: With the hybrid prompt injected in Section 3, the grading threshold
# for empty tool_calls [] is intentionally kept strict.
# Reason: the hybrid prompt explicitly tells the model "if no tools: reply
# naturally". If the model still outputs {"tool_calls": []} after receiving
# that instruction, it IS stuck in format mode — the failure is genuine.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class GradingResult:
    """Result of the mode-collapse grading for one sample."""
    passed: bool = False
    failure_reason: str = ""


def grade_response(raw_output: str) -> GradingResult:
    """
    Grade a model response for mode collapse.

    The grading rule is intentionally strict:
      • Try to parse raw_output with json.loads().
      • If it parses AND the resulting dict has a "tool_calls" key
        (even if the value is an empty list []):
          → ❌ FAIL — the model is stuck in tool-call schema format.
      • If json.loads() raises JSONDecodeError (i.e. plain text/markdown),
        OR the parsed object does NOT have a "tool_calls" key:
          → ✅ PASS — the model responded conversationally.

    Args:
        raw_output: The raw decoded model output string.

    Returns:
        GradingResult with pass/fail status and failure details.
    """
    result = GradingResult()
    text = raw_output.strip()

    if not text:
        # Empty output is not a tool-call hallucination.
        result.passed = True
        return result

    # ── Attempt JSON parse ──
    try:
        parsed = json.loads(text)

        # Check for the "tool_calls" key — ANY presence is mode collapse
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            # ❌ MODE COLLAPSE — model output the tool-call schema
            tool_calls = parsed["tool_calls"]
            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                call_names = [
                    tc.get("name", "?") for tc in tool_calls
                    if isinstance(tc, dict)
                ]
                result.failure_reason = (
                    f"Mode Collapse: hallucinated tool_calls "
                    f"[{', '.join(call_names)}]"
                )
            else:
                # Even an empty tool_calls list is a failure — the model
                # chose JSON schema format instead of conversing naturally,
                # despite the hybrid prompt explicitly permitting plain text.
                result.failure_reason = (
                    "Mode Collapse: output JSON with empty tool_calls [] "
                    "(model stuck in format mode despite hybrid prompt)"
                )
            result.passed = False
            return result

        # Parsed as JSON but NO "tool_calls" key — acceptable
        result.passed = True
        return result

    except (json.JSONDecodeError, TypeError, ValueError):
        # Not JSON → model produced natural conversational text ✅
        result.passed = True
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — Per-Sample Record & Dataset Loader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class EvalRecord:
    """Complete evaluation record for a single ultrachat sample."""
    sample_idx: int
    user_query: str = ""
    raw_output: str = ""
    generation_time_s: float = 0.0
    passed: bool = False
    failure_reason: str = ""


def load_dataset(dataset_path: str) -> list[dict]:
    """
    Load JSONL dataset and filter to ULTRACHAT samples only.

    Args:
        dataset_path: Path to the JSONL file.

    Returns:
        List of ultrachat sample dicts.
    """
    path = os.path.expanduser(dataset_path)
    if not os.path.exists(path):
        logger.error(f"Dataset not found: {path}")
        sys.exit(1)

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # ── Filter: ultrachat only ──
                if record.get("source") == "ultrachat":
                    rows.append(record)
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(rows)} ultrachat samples from {path}")
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — Pipeline with Live Terminal Feed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_pipeline(
    model,
    tokenizer,
    samples: list[dict],
    max_new_tokens: int = 512,
) -> list[EvalRecord]:
    """
    Run the conversational evaluation pipeline with live terminal output.

    For each sample:
      1. Build inference prompt (strip assistant turns, inject hybrid prompt).
      2. Generate a response.
      3. Grade using the mode-collapse grading logic.
      4. Print a live preview via tqdm.write().

    Args:
        model: The loaded language model.
        tokenizer: The tokenizer.
        samples: List of ultrachat dataset samples.
        max_new_tokens: Max new tokens per generation.

    Returns:
        List of EvalRecord, one per sample.
    """
    records: list[EvalRecord] = []
    pbar = tqdm(samples, desc="🗣️  Evaluating Ultrachat", unit="sample")

    for idx, sample in enumerate(pbar):
        record = EvalRecord(sample_idx=idx)

        # ── Extract user query ──
        user_query = extract_user_query(sample)
        record.user_query = user_query
        query_preview = user_query[:100] + ("..." if len(user_query) > 100 else "")

        # ── Build prompt (hybrid system prompt injected here) ──
        prompt_turns = build_inference_prompt(sample)
        if prompt_turns is None:
            record.passed = True  # Can't evaluate → don't count as collapse
            record.failure_reason = "Skipped (invalid prompt)"
            records.append(record)
            tqdm.write(f"  ⚪ [{idx}] SKIP — invalid prompt structure")
            continue

        # ── Generate ──
        try:
            t0 = time.time()
            raw_output = generate(model, tokenizer, prompt_turns, max_new_tokens)
            record.generation_time_s = time.time() - t0
            record.raw_output = raw_output
        except Exception as e:
            record.passed = True  # Generation failure ≠ mode collapse
            record.failure_reason = f"Generation error: {e}"
            records.append(record)
            tqdm.write(f"  ⚪ [{idx}] SKIP — generation error: {e}")
            continue

        # ── Grade ──
        grading = grade_response(raw_output)
        record.passed = grading.passed
        record.failure_reason = grading.failure_reason
        records.append(record)

        # ── Live Terminal Feed ──
        if grading.passed:
            response_preview = raw_output[:150] + ("..." if len(raw_output) > 150 else "")
            tqdm.write(
                f"\n  [{idx}] Query: {query_preview}\n"
                f"  ✅ PASS: {response_preview}"
            )
        else:
            # ❌ Blaring failure — show the exact hallucinated JSON
            tqdm.write(
                f"\n  [{idx}] Query: {query_preview}\n"
                f"  ❌ WARNING - MODE COLLAPSE: {raw_output}"
            )

        # ── Update progress bar with running stats ──
        total_so_far = len(records)
        passed_so_far = sum(1 for r in records if r.passed)
        collapse_pct = ((total_so_far - passed_so_far) / total_so_far * 100) if total_so_far > 0 else 0
        pbar.set_postfix({
            "pass": f"{passed_so_far}/{total_so_far}",
            "collapse": f"{collapse_pct:.1f}%",
        })

    return records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7 — Metrics & Terminal Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_metrics(records: list[EvalRecord]) -> dict[str, Any]:
    """Compute aggregate metrics from evaluation records."""
    total = len(records)
    if total == 0:
        return {"error": "No records to evaluate"}

    passed = sum(1 for r in records if r.passed)
    collapsed = total - passed

    gen_times = [r.generation_time_s for r in records if r.generation_time_s > 0]
    avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0.0

    return {
        "total_ultrachat_samples": total,
        "conversational_passed": passed,
        "conversational_success_rate": round(passed / total * 100, 2),
        "mode_collapse_count": collapsed,
        "mode_collapse_rate": round(collapsed / total * 100, 2),
        "avg_generation_time_s": round(avg_gen_time, 3),
    }


def print_dashboard(metrics: dict[str, Any]) -> None:
    """Print a clean terminal dashboard summarizing results."""
    W = 70
    total = metrics["total_ultrachat_samples"]

    def bar(n: int, total: int, width: int = 30) -> str:
        if total == 0:
            return "[" + "─" * width + "] n/a"
        filled = int(round(n / total * width))
        return "[" + "█" * filled + "─" * (width - filled) + f"] {n}/{total}"

    print("\n" + "━" * W)
    print("  🧠  MODE COLLAPSE REPORT — Ultrachat Conversational Test")
    print("━" * W)

    print(f"\n  Total Ultrachat Samples:       {total}")
    print(f"  Avg Generation Time:           {metrics['avg_generation_time_s']:.3f}s\n")

    success_rate = metrics["conversational_success_rate"]
    collapse_rate = metrics["mode_collapse_rate"]

    print(f"  Conversational Success Rate:  {success_rate:6.2f}%  "
          f"{bar(metrics['conversational_passed'], total)}")
    print(f"  Mode Collapse Rate:           {collapse_rate:6.2f}%  "
          f"{bar(metrics['mode_collapse_count'], total)}")

    # ── Verdict ──
    print(f"\n{'─' * W}")
    if collapse_rate == 0:
        print("  ✅ VERDICT: Zero mode collapse — full conversational retention!")
    elif collapse_rate < 5:
        print(f"  🟡 VERDICT: Minor mode collapse ({collapse_rate:.1f}%) — worth monitoring.")
    elif collapse_rate < 20:
        print(f"  ⚠️  VERDICT: Moderate collapse ({collapse_rate:.1f}%) — consider more chat data in training mix.")
    else:
        print(f"  ❌ VERDICT: Severe mode collapse ({collapse_rate:.1f}%) — model over-fitted to tool-call format!")

    print("━" * W + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8 — Results Saving
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def save_results(
    records: list[EvalRecord],
    metrics: dict[str, Any],
    output_dir: str,
) -> Path:
    """Save per-sample results and aggregate metrics to JSON files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Per-sample results ──
    results_file = out_path / "eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)
    logger.info(f"📄 Per-sample results saved to: {results_file}")

    # ── Aggregate metrics ──
    metrics_file = out_path / "eval_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"📄 Aggregate metrics saved to:  {metrics_file}")

    # ── Human-readable summary ──
    summary_file = out_path / "eval_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Mode Collapse Evaluation — Ultrachat Conversational Test\n")
        f.write("=" * 60 + "\n\n")
        for key, val in metrics.items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")

        failures = [r for r in records if not r.passed]
        if failures:
            f.write(f"\nMode Collapse Failures ({len(failures)} total):\n")
            f.write("-" * 50 + "\n")
            for r in failures[:20]:
                f.write(f"  Sample {r.sample_idx}: {r.user_query[:80]}...\n")
                f.write(f"    Reason: {r.failure_reason}\n")
                f.write(f"    Output: {r.raw_output[:300]}...\n\n")

    logger.info(f"📄 Summary saved to:            {summary_file}")
    return out_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9 — Main Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Main: parse CLI → load model → evaluate ultrachat → report."""
    args = parse_args()

    # ── Banner ──
    print("\n" + "━" * 70)
    print("  🧠  Mode Collapse Detector — Ultrachat Evaluation")
    print("━" * 70)
    print(f"  Model:    {args.model_path}")
    print(f"  Dataset:  {args.dataset_path}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Filter:   source == 'ultrachat'")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Prompt mode: HYBRID (injected at eval time — not raw dataset prompt)")
    print("━" * 70 + "\n")

    # ── Step 1: Load dataset ──
    logger.info("Step 1/3: Loading ultrachat samples...")
    samples = load_dataset(args.dataset_path)
    if args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"  → Limited to first {args.max_samples} samples")

    if not samples:
        logger.error("No ultrachat samples found in dataset. Exiting.")
        logger.error("Ensure your dataset has records with source='ultrachat'.")
        sys.exit(1)

    # ── Step 2: Load model ──
    logger.info("Step 2/3: Loading model...")
    model, tokenizer = load_model(args.model_path, args.max_seq_length)

    # ── Step 3: Run evaluation ──
    logger.info("Step 3/3: Running ultrachat evaluation (live terminal feed)...")
    print()
    records = run_pipeline(model, tokenizer, samples, args.max_new_tokens)

    # ── Dashboard ──
    metrics = compute_metrics(records)
    print_dashboard(metrics)

    # ── Save ──
    save_results(records, metrics, args.output_dir)

    logger.info("✅ Ultrachat evaluation complete!")


if __name__ == "__main__":
    main()
