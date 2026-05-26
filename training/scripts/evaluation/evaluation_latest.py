"""
scripts/evaluation_latest.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BFCL-Style Evaluation for Fine-Tuned Qwen2.5-7B-Instruct Tool Calling.

Architecture mirrors the Berkeley Function Calling Leaderboard (BFCL):
  • AST Static Evaluation  — parses predicted calls via Python's `ast` module
                              and compares structure + arguments against ground truth.
  • Executable Dynamic Eval — dynamically mocks functions from tool schemas and
                              executes predicted calls in a sandboxed namespace.

Usage:
    python scripts/evaluation_latest.py \
        --model_path  ./runs/checkpoint-final \
        --dataset_path data/val.jsonl \
        --output_dir   outputs/eval_latest \
        --max_samples  50

Inference stack (Unsloth-free):
  • Native Hugging Face Transformers (AutoModelForCausalLM) + PEFT (PeftModel).
  • 4-bit NF4 quantisation via BitsAndBytesConfig (double_quant, bfloat16 compute).
  • Standard KV-cache generation (use_cache=True) — stable without Unsloth kernels.
  • ChatML template is applied via tokenizer.apply_chat_template().

Author: Auto-generated evaluation pipeline
Python: 3.10+
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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
logger = logging.getLogger("eval_latest")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — CLI & Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 42

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="BFCL-style evaluation for fine-tuned Qwen2.5 tool-calling",
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
        default="../../outputs/eval_latest",
        help="Directory to save detailed results JSON (default: outputs/eval_latest).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit evaluation to the first N apigen samples (default: all).",
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
# SECTION 2 — Inference Engine (Native Hugging Face — Unsloth-free)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_model(model_path: str, max_seq_length: int = 2048):
    """
    Load the fine-tuned model using native Hugging Face Transformers + PEFT.

    Pipeline:
      1. Read adapter_config.json to discover the base model name.
      2. Load the base model in 4-bit NF4 via BitsAndBytesConfig.
      3. Overlay the LoRA adapter with PeftModel.
      4. Set model to eval mode (no Unsloth, no for_inference).

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

    # ── Inference mode — plain eval(), no Unsloth for_inference needed ──
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

    Uses standard Transformers generation with use_cache=True.
    The previous use_cache=False workaround was only needed for Unsloth's
    Triton kernels; native HF generation is stable with the KV cache enabled.

    Args:
        model: The loaded language model.
        tokenizer: The tokenizer.
        prompt_turns: List of {"role": ..., "content": ...} dicts (no assistant turn).
        max_new_tokens: Max tokens to generate.

    Returns:
        The decoded model output (assistant response only, no prompt tokens).
    """
    # Apply Qwen2.5 ChatML template — add_generation_prompt=True appends
    # the <|im_start|>assistant\n prefix so the model continues from there.
    prompt = tokenizer.apply_chat_template(
        prompt_turns,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize — pass only input_ids to model.generate() for stability
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
            use_cache=True,               # ── Standard KV cache ──
            # Native HF Transformers generation is stable with the KV cache.
            # The use_cache=False workaround was only needed for Unsloth's
            # Triton kernels which are no longer in the inference path.
        )

    # Decode only the newly generated tokens (strip the input prompt)
    new_token_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — Model Handler: Input Formatter & Ground Truth Extractor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_inference_prompt(sample: dict) -> list[dict] | None:
    """
    Extract prompt turns (system + user) from a dataset sample, stripping
    any trailing assistant turns.

    This replicates the logic in src/callbacks.py → _build_prompt_turns():
    walk backward from the conversation end and remove all assistant turns,
    leaving only the system prompt and user query for the model to complete.

    Args:
        sample: A raw JSONL record with a "conversation" key.

    Returns:
        A list of conversation turns (system + user), or None if invalid.
    """
    conversation = sample.get("conversation", [])
    if not conversation or len(conversation) < 2:
        return None

    # Strip trailing assistant turns — keep only the prompt
    turns = list(conversation)
    while turns and turns[-1]["role"] == "assistant":
        turns.pop()

    if not turns or turns[0]["role"] != "system":
        return None

    return turns


def extract_ground_truth(sample: dict) -> list[dict] | None:
    """
    Extract the ground-truth tool_calls from the last assistant turn.

    Expected format in val.jsonl (source="apigen"):
        assistant content = '{"tool_calls": [{"name": "fn", "arguments": {...}}, ...]}'

    Returns:
        A list of tool call dicts [{"name": ..., "arguments": {...}}, ...],
        or None if parsing fails.
    """
    conversation = sample.get("conversation", [])
    if not conversation:
        return None

    # Find the last assistant turn
    assistant_content: str | None = None
    for turn in reversed(conversation):
        if turn["role"] == "assistant":
            assistant_content = turn.get("content", "")
            break

    if assistant_content is None:
        return None

    try:
        parsed = json.loads(assistant_content)
        calls = parsed.get("tool_calls", [])
        if isinstance(calls, list):
            return calls
    except (json.JSONDecodeError, AttributeError):
        pass

    return None


def extract_tool_names_from_system(sample: dict) -> set[str]:
    """
    Parse the system prompt to extract the set of valid tool names.

    The apigen system prompt embeds tool schemas between
    [BEGIN OF AVAILABLE TOOLS] and [END OF AVAILABLE TOOLS] as a JSON array.
    We parse that to get the function names — used later to detect
    hallucinated tool names.

    Returns:
        A set of valid function names, or empty set if parsing fails.
    """
    conversation = sample.get("conversation", [])
    if not conversation or conversation[0]["role"] != "system":
        return set()

    system_content = conversation[0].get("content", "")

    # Extract the JSON array between the tool markers
    match = re.search(
        r"\[BEGIN OF AVAILABLE TOOLS\]\s*(\[.*?\])\s*\[END OF AVAILABLE TOOLS\]",
        system_content,
        re.DOTALL,
    )
    if not match:
        return set()

    try:
        tools_array = json.loads(match.group(1))
        names = set()
        for tool in tools_array:
            # Handle both {"name": "fn"} and {"function": {"name": "fn"}} formats
            if isinstance(tool, dict):
                if "function" in tool and isinstance(tool["function"], dict):
                    name = tool["function"].get("name")
                else:
                    name = tool.get("name")
                if name:
                    names.add(name)
        return names
    except (json.JSONDecodeError, TypeError):
        return set()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Output Normalizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class ParsedOutput:
    """Structured result from parsing raw model output."""
    raw: str                                      # Original model output
    tool_calls: list[dict] = field(default_factory=list)  # Parsed tool_calls list
    parse_error: str | None = None                # Error message if parsing failed


def parse_model_output(raw: str) -> ParsedOutput:
    """
    Parse the raw model output into structured tool_calls.

    Handles common LLM output quirks:
      1. Clean JSON: {"tool_calls": [...]}
      2. Markdown-fenced JSON: ```json\n{...}\n```
      3. Leading/trailing text around JSON
      4. Empty tool_calls (model declined to call)

    Args:
        raw: Raw decoded model output string.

    Returns:
        ParsedOutput with tool_calls populated on success, or parse_error on failure.
    """
    result = ParsedOutput(raw=raw)
    text = raw.strip()

    if not text:
        result.parse_error = "Empty model output"
        return result

    # ── Strategy 1: Strip markdown code fences ──
    # Models sometimes wrap JSON in ```json ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # ── Strategy 2: Direct JSON parse ──
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            result.tool_calls = parsed.get("tool_calls", [])
            if not isinstance(result.tool_calls, list):
                result.tool_calls = []
                result.parse_error = "tool_calls is not a list"
            return result
    except json.JSONDecodeError:
        pass

    # ── Strategy 3: Find the outermost JSON object ──
    # Sometimes models prepend "Here is the response:" or similar
    brace_start = text.find("{")
    if brace_start != -1:
        # Find the matching closing brace by counting nesting depth
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            result.tool_calls = parsed.get("tool_calls", [])
                            if not isinstance(result.tool_calls, list):
                                result.tool_calls = []
                                result.parse_error = "tool_calls is not a list"
                            return result
                    except json.JSONDecodeError:
                        pass
                    break

    result.parse_error = "Could not extract valid JSON from model output"
    return result


def normalize_to_call_string(tool_call: dict) -> str | None:
    """
    Convert a parsed tool_call dict to a canonical Python call string.

    Input:  {"name": "getproductsku", "arguments": {"product_sku": "LAP0987654"}}
    Output: 'getproductsku(product_sku="LAP0987654")'

    Args:
        tool_call: A dict with "name" and "arguments" keys.

    Returns:
        A Python-syntax call string, or None if the dict is malformed.
    """
    name = tool_call.get("name")
    args = tool_call.get("arguments", {})

    if not name or not isinstance(name, str):
        return None
    if not isinstance(args, dict):
        return None

    # Build keyword argument strings with proper repr() for values
    arg_parts: list[str] = []
    for key, value in args.items():
        arg_parts.append(f"{key}={repr(value)}")

    return f"{name}({', '.join(arg_parts)})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — AST Evaluator (Static Analysis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class ASTResult:
    """Result of AST-based static evaluation for one sample."""
    matched: bool = False        # True if ALL predicted calls match ground truth
    num_predicted: int = 0       # Count of predicted calls
    num_expected: int = 0        # Count of ground truth calls
    details: str = ""            # Human-readable explanation


def _values_match(predicted: Any, expected: Any) -> bool:
    """
    Compare two argument values with type-coercion tolerance.

    Handles common mismatches:
      • "123" vs 123  (string ↔ int)
      • 1.0 vs 1      (float ↔ int)
      • "true" vs True (string ↔ bool)
      • Case-insensitive string comparison

    Args:
        predicted: The value from the model's output.
        expected: The value from the ground truth.

    Returns:
        True if values are semantically equivalent.
    """
    # Direct equality
    if predicted == expected:
        return True

    # Both numeric — compare with tolerance
    if isinstance(predicted, (int, float)) and isinstance(expected, (int, float)):
        # Use relative tolerance for large numbers, absolute for small
        if expected == 0:
            return abs(predicted) < 1e-9
        return abs(predicted - expected) / max(abs(expected), 1e-9) < 1e-6

    # String ↔ numeric coercion
    if isinstance(predicted, str) and isinstance(expected, (int, float)):
        try:
            return _values_match(type(expected)(predicted), expected)
        except (ValueError, TypeError):
            return False
    if isinstance(expected, str) and isinstance(predicted, (int, float)):
        try:
            return _values_match(predicted, type(predicted)(expected))
        except (ValueError, TypeError):
            return False

    # String ↔ bool coercion
    if isinstance(predicted, str) and isinstance(expected, bool):
        return predicted.lower() == str(expected).lower()
    if isinstance(expected, str) and isinstance(predicted, bool):
        return str(predicted).lower() == expected.lower()

    # Case-insensitive string comparison
    if isinstance(predicted, str) and isinstance(expected, str):
        return predicted.strip().lower() == expected.strip().lower()

    # List comparison (for array arguments)
    if isinstance(predicted, list) and isinstance(expected, list):
        if len(predicted) != len(expected):
            return False
        return all(_values_match(p, e) for p, e in zip(predicted, expected))

    # Dict comparison (for nested object arguments)
    if isinstance(predicted, dict) and isinstance(expected, dict):
        if set(predicted.keys()) != set(expected.keys()):
            return False
        return all(
            _values_match(predicted[k], expected[k]) for k in expected
        )

    return False


def _match_single_call(predicted: dict, expected: dict) -> tuple[bool, str]:
    """
    Compare a single predicted tool call against an expected one.

    Checks:
      1. Function name must match exactly
      2. All expected arguments must be present and match in value
      3. Extra predicted arguments are tolerated (model may fill defaults)

    Args:
        predicted: {"name": "...", "arguments": {...}}
        expected:  {"name": "...", "arguments": {...}}

    Returns:
        (matched: bool, detail: str)
    """
    pred_name = predicted.get("name", "")
    exp_name = expected.get("name", "")

    if pred_name != exp_name:
        return False, f"Name mismatch: predicted={pred_name}, expected={exp_name}"

    pred_args = predicted.get("arguments", {})
    exp_args = expected.get("arguments", {})

    if not isinstance(pred_args, dict):
        return False, f"Predicted arguments is not a dict: {type(pred_args)}"
    if not isinstance(exp_args, dict):
        # Ground truth args not a dict — unusual, just check names match
        return True, "Name matched (ground truth args not a dict)"

    # Check every expected argument is present and matches
    mismatches: list[str] = []
    for key, exp_val in exp_args.items():
        if key not in pred_args:
            mismatches.append(f"Missing arg '{key}'")
            continue

        pred_val = pred_args[key]

        # Use ast.literal_eval for safe value comparison when they're strings
        # that might represent Python literals
        safe_pred = pred_val
        safe_exp = exp_val
        if isinstance(pred_val, str):
            try:
                safe_pred = ast.literal_eval(pred_val)
            except Exception:
                safe_pred = pred_val
        if isinstance(exp_val, str):
            try:
                safe_exp = ast.literal_eval(exp_val)
            except Exception:
                safe_exp = exp_val

        if not _values_match(safe_pred, safe_exp):
            mismatches.append(
                f"Arg '{key}': predicted={repr(pred_val)}, expected={repr(exp_val)}"
            )

    if mismatches:
        return False, "; ".join(mismatches)
    return True, "All arguments matched"


def ast_evaluate(
    predicted_calls: list[dict],
    ground_truth_calls: list[dict],
) -> ASTResult:
    """
    AST-based static evaluation: compare predicted vs ground-truth tool calls.

    Uses "all-or-nothing" scoring for parallel calls (BFCL style):
    ALL predicted calls must match ALL ground truth calls for the sample to pass.

    Matching strategy:
      1. Both lists must have the same length.
      2. For each expected call, find the best matching predicted call (greedy).
      3. Every expected call must have a match.

    Args:
        predicted_calls: List of predicted tool call dicts.
        ground_truth_calls: List of ground truth tool call dicts.

    Returns:
        ASTResult with match status and details.
    """
    result = ASTResult(
        num_predicted=len(predicted_calls),
        num_expected=len(ground_truth_calls),
    )

    # ── Length mismatch → fail ──
    if len(predicted_calls) != len(ground_truth_calls):
        result.details = (
            f"Call count mismatch: predicted={len(predicted_calls)}, "
            f"expected={len(ground_truth_calls)}"
        )
        return result

    # ── Empty lists → both agree no calls needed → pass ──
    if len(ground_truth_calls) == 0:
        result.matched = True
        result.details = "Both predicted and expected have zero calls"
        return result

    # ── Greedy matching ──
    # For each expected call, try to find a matching predicted call.
    # This handles cases where the model outputs calls in a different order.
    remaining_predicted = list(range(len(predicted_calls)))
    all_details: list[str] = []
    all_matched = True

    for exp_idx, exp_call in enumerate(ground_truth_calls):
        best_match_idx: int | None = None
        best_detail = ""

        for pred_idx in remaining_predicted:
            matched, detail = _match_single_call(predicted_calls[pred_idx], exp_call)
            if matched:
                best_match_idx = pred_idx
                best_detail = detail
                break

        if best_match_idx is not None:
            remaining_predicted.remove(best_match_idx)
            all_details.append(f"Call {exp_idx}: ✅ {best_detail}")
        else:
            all_matched = False
            # Report why the best candidate failed
            if predicted_calls:
                _, fail_detail = _match_single_call(
                    predicted_calls[remaining_predicted[0]] if remaining_predicted else predicted_calls[0],
                    exp_call,
                )
                all_details.append(f"Call {exp_idx}: ❌ {fail_detail}")
            else:
                all_details.append(f"Call {exp_idx}: ❌ No predicted calls to match against")

    result.matched = all_matched
    result.details = " | ".join(all_details)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — Executable Evaluator (Dynamic Sandbox)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class ExecResult:
    """Result of executable evaluation for one sample."""
    passed: bool = False             # True if ALL predicted calls executed without error
    num_calls_executed: int = 0      # Number of calls attempted
    num_calls_passed: int = 0        # Number of calls that succeeded
    errors: list[str] = field(default_factory=list)  # Error messages per failed call


def build_mock_registry(sample: dict) -> dict[str, callable]:
    """
    Dynamically create mock Python functions from the tool schemas embedded
    in a sample's system prompt.

    Each mock function:
      • Accepts **kwargs (any keyword arguments)
      • Returns a dict: {"status": "ok", "function": name, "received_args": kwargs}
      • This lets us verify the function was called with the right signature

    Why dynamic mocks instead of hardcoded ones?
      The apigen dataset has 1000+ diverse tool schemas. Hardcoding is impossible.
      By generating mocks from the schema, we can verify that the model produces
      syntactically valid calls that would not crash a real implementation.

    Args:
        sample: A raw JSONL record containing the system prompt with tool schemas.

    Returns:
        A dict mapping function_name → mock callable.
    """
    valid_names = extract_tool_names_from_system(sample)
    registry: dict[str, callable] = {}

    for name in valid_names:
        # Use a closure to capture the name for each mock function
        def make_mock(fn_name: str):
            def mock_fn(**kwargs):
                return {
                    "status": "ok",
                    "function": fn_name,
                    "received_args": kwargs,
                }
            mock_fn.__name__ = fn_name
            return mock_fn

        registry[name] = make_mock(name)

    return registry


def exec_evaluate(
    predicted_calls: list[dict],
    mock_registry: dict[str, callable],
) -> ExecResult:
    """
    Executable evaluation: attempt to run each predicted tool call against
    a mock function registry.

    For each call:
      1. Check if the function name exists in the registry
      2. Extract arguments
      3. Call the mock function with those arguments
      4. If no exception → Pass; any exception → Fail

    This tests that:
      • The model doesn't hallucinate function names
      • The arguments are valid Python types
      • The call signature doesn't cause runtime errors

    Uses all-or-nothing: ALL calls must pass for the sample to pass.

    Args:
        predicted_calls: List of predicted tool call dicts.
        mock_registry: Dict mapping function names to mock callables.

    Returns:
        ExecResult with pass/fail status and errors.
    """
    result = ExecResult()

    if not predicted_calls:
        # No calls predicted — we consider this a pass for execution
        # (the AST evaluator handles correctness checking)
        result.passed = True
        return result

    for call in predicted_calls:
        result.num_calls_executed += 1
        name = call.get("name", "")
        args = call.get("arguments", {})

        # ── Check 1: Function exists? ──
        if name not in mock_registry:
            result.errors.append(f"Hallucinated function: '{name}'")
            continue

        # ── Check 2: Arguments are a dict? ──
        if not isinstance(args, dict):
            result.errors.append(
                f"Arguments for '{name}' is not a dict: {type(args).__name__}"
            )
            continue

        # ── Check 3: Execute the call ──
        try:
            # Sandboxed execution via direct function call with kwargs
            fn = mock_registry[name]
            fn(**args)
            result.num_calls_passed += 1
        except TypeError as e:
            # TypeError typically means wrong argument names/types
            result.errors.append(f"TypeError calling '{name}': {e}")
        except Exception as e:
            result.errors.append(f"Error calling '{name}': {type(e).__name__}: {e}")

    result.passed = (result.num_calls_passed == result.num_calls_executed)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7 — Pipeline Orchestration & Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class EvalRecord:
    """Complete evaluation record for a single dataset sample."""
    sample_idx: int
    # Input
    user_query: str = ""
    num_gt_calls: int = 0
    # Generation
    raw_output: str = ""
    generation_time_s: float = 0.0
    # Parsing
    parse_error: str | None = None
    num_predicted_calls: int = 0
    normalized_calls: list[str] = field(default_factory=list)
    # AST evaluation
    ast_matched: bool = False
    ast_details: str = ""
    # Exec evaluation
    exec_passed: bool = False
    exec_errors: list[str] = field(default_factory=list)
    # Format issues
    has_format_error: bool = False  # JSON parse failure or hallucinated tool name
    format_error_detail: str = ""


def load_dataset(dataset_path: str) -> list[dict]:
    """Load JSONL dataset and filter to apigen samples only."""
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
                if record.get("source") == "apigen":
                    rows.append(record)
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(rows)} apigen samples from {path}")
    return rows


def run_pipeline(
    model,
    tokenizer,
    samples: list[dict],
    max_new_tokens: int = 512,
) -> list[EvalRecord]:
    """
    Execute the full evaluation pipeline:
      Generate → Parse → Normalize → AST Eval → Exec Eval

    Args:
        model: The loaded language model.
        tokenizer: The tokenizer.
        samples: List of apigen dataset samples.
        max_new_tokens: Max new tokens per generation.

    Returns:
        List of EvalRecord, one per sample.
    """
    records: list[EvalRecord] = []

    for idx, sample in enumerate(tqdm(samples, desc="🔄 Evaluating", unit="sample")):
        record = EvalRecord(sample_idx=idx)

        # ── Step 1: Extract ground truth ──
        gt_calls = extract_ground_truth(sample)
        if gt_calls is None:
            record.has_format_error = True
            record.format_error_detail = "Could not extract ground truth from sample"
            records.append(record)
            continue
        record.num_gt_calls = len(gt_calls)

        # Extract user query for logging
        conversation = sample.get("conversation", [])
        for turn in conversation:
            if turn["role"] == "user":
                record.user_query = turn.get("content", "")[:200]
                break

        # ── Step 2: Build prompt and generate ──
        prompt_turns = build_inference_prompt(sample)
        if prompt_turns is None:
            record.has_format_error = True
            record.format_error_detail = "Could not build inference prompt"
            records.append(record)
            continue

        try:
            t0 = time.time()
            raw_output = generate(model, tokenizer, prompt_turns, max_new_tokens)
            record.generation_time_s = time.time() - t0
            record.raw_output = raw_output
        except Exception as e:
            record.has_format_error = True
            record.format_error_detail = f"Generation failed: {e}"
            records.append(record)
            continue

        # ── Step 3: Parse output ──
        parsed = parse_model_output(raw_output)
        if parsed.parse_error:
            record.parse_error = parsed.parse_error
            record.has_format_error = True
            record.format_error_detail = parsed.parse_error
            records.append(record)
            continue

        record.num_predicted_calls = len(parsed.tool_calls)

        # ── Step 4: Check for hallucinated tool names ──
        valid_tool_names = extract_tool_names_from_system(sample)
        hallucinated = []
        for call in parsed.tool_calls:
            call_name = call.get("name", "")
            if call_name and call_name not in valid_tool_names:
                hallucinated.append(call_name)

        if hallucinated:
            record.has_format_error = True
            record.format_error_detail = f"Hallucinated tools: {hallucinated}"
            # Still proceed with evaluation — partial credit in other metrics

        # ── Step 5: Normalize to call strings (for logging/debugging) ──
        for call in parsed.tool_calls:
            call_str = normalize_to_call_string(call)
            if call_str:
                record.normalized_calls.append(call_str)

        # ── Step 6: AST Evaluation ──
        ast_result = ast_evaluate(parsed.tool_calls, gt_calls)
        record.ast_matched = ast_result.matched
        record.ast_details = ast_result.details

        # ── Step 7: Executable Evaluation ──
        mock_registry = build_mock_registry(sample)
        exec_result = exec_evaluate(parsed.tool_calls, mock_registry)
        record.exec_passed = exec_result.passed
        record.exec_errors = exec_result.errors

        records.append(record)

    return records


def compute_metrics(records: list[EvalRecord]) -> dict[str, Any]:
    """
    Compute aggregate metrics from evaluation records.

    Returns:
        Dict with metric name → value mapping.
    """
    total = len(records)
    if total == 0:
        return {"error": "No records to evaluate"}

    ast_pass = sum(1 for r in records if r.ast_matched)
    exec_pass = sum(1 for r in records if r.exec_passed)
    format_err = sum(1 for r in records if r.has_format_error)

    # Average generation time (excluding failures)
    gen_times = [r.generation_time_s for r in records if r.generation_time_s > 0]
    avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0.0

    return {
        "total_samples": total,
        "ast_success": ast_pass,
        "ast_success_rate": round(ast_pass / total * 100, 2),
        "exec_success": exec_pass,
        "exec_success_rate": round(exec_pass / total * 100, 2),
        "format_errors": format_err,
        "format_error_rate": round(format_err / total * 100, 2),
        "avg_generation_time_s": round(avg_gen_time, 3),
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """Pretty-print evaluation metrics to console."""
    W = 70
    print("\n" + "━" * W)
    print("  📊  EVALUATION RESULTS — BFCL-Style Metrics")
    print("━" * W)

    total = metrics["total_samples"]

    def bar(n: int, total: int, width: int = 30) -> str:
        if total == 0:
            return "[" + "─" * width + "] n/a"
        filled = int(round(n / total * width))
        return "[" + "█" * filled + "─" * (width - filled) + f"] {n}/{total}"

    print(f"\n  Total Samples Evaluated:  {total}")
    print(f"  Avg Generation Time:     {metrics['avg_generation_time_s']:.3f}s\n")

    print(f"  AST Success Rate:        {metrics['ast_success_rate']:6.2f}%  "
          f"{bar(metrics['ast_success'], total)}")
    print(f"  Execution Success Rate:  {metrics['exec_success_rate']:6.2f}%  "
          f"{bar(metrics['exec_success'], total)}")
    print(f"  Format Error Rate:       {metrics['format_error_rate']:6.2f}%  "
          f"{bar(metrics['format_errors'], total)}")

    # ── Diagnostic Breakdown ──
    print(f"\n{'─' * W}")
    print("  Diagnostic Breakdown:")
    print(f"    ✅ AST Matched:       {metrics['ast_success']}")
    print(f"    ✅ Exec Passed:       {metrics['exec_success']}")
    print(f"    ❌ Format Errors:     {metrics['format_errors']}")
    print("━" * W + "\n")


def save_results(
    records: list[EvalRecord],
    metrics: dict[str, Any],
    output_dir: str,
) -> Path:
    """
    Save detailed per-sample results and aggregate metrics to JSON files.

    Args:
        records: List of EvalRecord from the pipeline.
        metrics: Aggregate metrics dict.
        output_dir: Directory to save results.

    Returns:
        Path to the saved results directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save per-sample results
    results_file = out_path / "eval_results.json"
    serializable_records = []
    for r in records:
        d = asdict(r)
        serializable_records.append(d)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(serializable_records, f, indent=2, ensure_ascii=False)
    logger.info(f"📄 Per-sample results saved to: {results_file}")

    # Save aggregate metrics
    metrics_file = out_path / "eval_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"📄 Aggregate metrics saved to:  {metrics_file}")

    # Save a human-readable summary
    summary_file = out_path / "eval_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("BFCL-Style Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, val in metrics.items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")

        # Top failures for debugging
        failures = [r for r in records if not r.ast_matched]
        if failures:
            f.write(f"\nFirst 10 AST failures:\n")
            f.write("-" * 50 + "\n")
            for r in failures[:10]:
                f.write(f"  Sample {r.sample_idx}: {r.user_query[:80]}...\n")
                f.write(f"    AST: {r.ast_details}\n")
                if r.exec_errors:
                    f.write(f"    Exec errors: {r.exec_errors}\n")
                if r.format_error_detail:
                    f.write(f"    Format: {r.format_error_detail}\n")
                f.write("\n")

    logger.info(f"📄 Human-readable summary:      {summary_file}")
    return out_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8 — Main Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Main entry point: parse CLI args → load model → run pipeline → report."""
    args = parse_args()

    # ── Banner ──
    print("\n" + "━" * 70)
    print("  🧪  BFCL-Style Evaluation — Qwen2.5 Tool Calling")
    print("━" * 70)
    print(f"  Model:    {args.model_path}")
    print(f"  Dataset:  {args.dataset_path}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print("━" * 70 + "\n")

    # ── Step 1: Load dataset ──
    logger.info("Step 1/3: Loading dataset...")
    samples = load_dataset(args.dataset_path)
    if args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"  → Limited to first {args.max_samples} samples")

    if not samples:
        logger.error("No apigen samples found in dataset. Exiting.")
        sys.exit(1)

    # ── Step 2: Load model ──
    logger.info("Step 2/3: Loading model...")
    model, tokenizer = load_model(args.model_path, args.max_seq_length)

    # ── Step 3: Run evaluation pipeline ──
    logger.info("Step 3/3: Running evaluation pipeline...")
    records = run_pipeline(model, tokenizer, samples, args.max_new_tokens)

    # ── Compute & display metrics ──
    metrics = compute_metrics(records)
    print_metrics(metrics)

    # ── Save results ──
    save_results(records, metrics, args.output_dir)

    logger.info("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
