"""
scripts/test_inference.py — Verify the fine-tuned model after training.

Loads the LoRA adapter via Unsloth and runs 5 test cases covering every
scenario the model must handle:

  1. Tool call required        → JSON with tool_calls
  2. Parallel tool calls       → JSON with 2 calls
  3. Conversational            → plain text (no tools in prompt)
  4. Empty tools (negative)    → plain text (tools=[], must NOT hallucinate)
  5. Tool present but not needed → empty tool_calls list or plain text

Usage:
    python scripts/test_inference.py \
        --adapter_path ~/training/runs/qwen7b_unsloth/final_adapter

    # Stream output token by token (easier to read long responses):
    python scripts/test_inference.py \
        --adapter_path ~/training/runs/qwen7b_unsloth/final_adapter \
        --stream
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from unsloth import FastLanguageModel


# ── Shared prompt templates (match training format exactly) ───────────────────

FORMAT_BLOCK = """[BEGIN OF FORMAT INSTRUCTION]
Output MUST be a JSON object with a "tool_calls" key. No other text.
If no call is needed, set tool_calls to an empty list.
Example:
{{"tool_calls": [{{"name": "func_name", "arguments": {{"arg1": "val1"}}}}]}}
[END OF FORMAT INSTRUCTION]"""

def make_system(tools_json: str, instruction: str) -> str:
    return (
        f"[BEGIN OF TASK INSTRUCTION]\n{instruction}\n[END OF TASK INSTRUCTION]\n\n"
        f"[BEGIN OF AVAILABLE TOOLS]\n{tools_json}\n[END OF AVAILABLE TOOLS]\n\n"
        + FORMAT_BLOCK
    )

INSTRUCTION = "You are a helpful assistant with access to external tools. Use them when necessary to answer the user's request."

# ── Tool definitions ──────────────────────────────────────────────────────────

WEATHER_TOOL = json.dumps([{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["city"]
        }
    }
}], indent=2)

SEARCH_TOOL = json.dumps([{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for up-to-date information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"}
            },
            "required": ["query"]
        }
    }
}], indent=2)

BOTH_TOOLS = json.dumps(
    json.loads(WEATHER_TOOL) + json.loads(SEARCH_TOOL),
    indent=2
)

# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name"    : "1 — Tool call (weather)",
        "expected": "json_tool_calls",
        "conversation": [
            {"role": "system",    "content": make_system(WEATHER_TOOL, INSTRUCTION)},
            {"role": "user",      "content": "What is the weather in Tokyo right now?"},
        ]
    },
    {
        "name"    : "2 — Parallel tool calls (weather + search)",
        "expected": "json_tool_calls",
        "conversation": [
            {"role": "system",    "content": make_system(BOTH_TOOLS, INSTRUCTION)},
            {"role": "user",      "content": "Check the weather in Paris and also search for top tourist attractions there."},
        ]
    },
    {
        "name"    : "3 — Conversational, no tools in system",
        "expected": "plain_text",
        "conversation": [
            {"role": "system",    "content": "You are a helpful, harmless, and honest general AI assistant."},
            {"role": "user",      "content": "Can you explain what a transformer neural network is?"},
        ]
    },
    {
        "name"    : "4 — Negative sample: empty tools array",
        "expected": "plain_or_empty_calls",
        "conversation": [
            {"role": "system",    "content": make_system("[]", INSTRUCTION)},
            {"role": "user",      "content": "What year did World War II end?"},
        ]
    },
    {
        "name"    : "5 — Tool present but not needed",
        "expected": "plain_or_empty_calls",
        "conversation": [
            {"role": "system",    "content": make_system(WEATHER_TOOL, INSTRUCTION)},
            {"role": "user",      "content": "What is 12 multiplied by 15?"},
        ]
    },
]


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate(output: str, expected: str) -> tuple[bool, str]:
    if expected == "json_tool_calls":
        try:
            parsed = json.loads(output)
            calls  = parsed.get("tool_calls", None)
            if isinstance(calls, list) and len(calls) > 0:
                names = [c.get("name", "?") for c in calls]
                return True, f"Valid JSON tool_calls → {names}"
            return False, f"JSON parsed but tool_calls empty or missing: {output[:120]}"
        except json.JSONDecodeError:
            return False, f"Not valid JSON: {output[:120]}"

    elif expected == "plain_text":
        try:
            json.loads(output)
            return False, f"FORMAT COLLAPSE — produced JSON for a conversational query"
        except json.JSONDecodeError:
            return True, f"Plain text ✓: {output[:120]}"

    elif expected == "plain_or_empty_calls":
        try:
            parsed = json.loads(output)
            calls  = parsed.get("tool_calls", None)
            if calls == [] or calls is None:
                return True, f"Empty tool_calls ✓ (correctly declined to call)"
            return False, f"Hallucinated tool call: {calls}"
        except json.JSONDecodeError:
            return True, f"Plain text ✓ (also acceptable): {output[:100]}"

    return False, f"Unknown expected: {expected}"


# ── Inference ─────────────────────────────────────────────────────────────────

def run(model, tokenizer, conversation: list[dict], max_new_tokens=256, stream=False) -> str:
    FastLanguageModel.for_inference(model)   # Switch to Unsloth inference mode

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1900).to("cuda")

    if stream:
        from transformers import TextStreamer
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )
        return ""   # streamer prints inline
    else:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)

    # Get base model name from saved adapter config
    with open(adapter_path / "adapter_config.json") as f:
        base_model = json.load(f)["base_model_name_or_path"]

    print(f"\nBase model : {base_model}")
    print(f"Adapter    : {adapter_path}")
    print(f"Streaming  : {args.stream}")

    # Load via Unsloth (same path as training — uses inference-optimised kernels)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = str(adapter_path),
        max_seq_length= args.max_seq_length,
        load_in_4bit  = True,
        dtype         = None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "=" * 70)
    print(f"INFERENCE TEST — {base_model}")
    print("=" * 70)

    passed = 0
    for test in TEST_CASES:
        print(f"\n{'─'*70}")
        print(f"[{test['name']}]")
        print(f"  User  : {test['conversation'][-1]['content']}")
        print(f"  Output:")

        output = run(model, tokenizer, test["conversation"], stream=args.stream)
        if not args.stream:
            print(f"  {output[:300]}")

        ok, note = evaluate(output, test["expected"])
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  Result: {status}  —  {note}")
        if ok:
            passed += 1

    print("\n" + "=" * 70)
    print(f"Score: {passed}/{len(TEST_CASES)} passed")
    if passed == len(TEST_CASES):
        print("✅ All tests passed.")
    elif passed >= 4:
        print("⚠️  Mostly passing — review failed cases above.")
    else:
        print("❌ Multiple failures — check training logs for format collapse.")
    print("=" * 70)


if __name__ == "__main__":
    main()
