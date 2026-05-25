#!/usr/bin/env python3
"""
chat_client.py -- Interactive CLI client for the vLLM-served Qwen 2.5 7B
tool-calling model.

Sends 100% clean, standard OpenAI API JSON payloads. All structural
formatting ([BEGIN OF TASK INSTRUCTION], <|im_start|>, etc.) is handled
exclusively by the server-side Jinja template.

Usage:
    Interactive mode : python chat_client.py
    Test mode        : python chat_client.py --test
    Dry-run test     : python chat_client.py --test --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

# --------------- Configuration ---------------

VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "https://jackie-instantly-musician-styles.trycloudflare.com/v1")
MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "qwen-tools")

# --------------- System Prompt ---------------

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "deployment" / "system_prompt.txt"


def load_system_prompt() -> str:
    """Load the raw system prompt from disk."""
    if not _SYSTEM_PROMPT_PATH.exists():
        print(f"[ERROR] System prompt not found at {_SYSTEM_PROMPT_PATH}")
        sys.exit(1)
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


# --------------- Tool Definitions ---------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Retrieve the current weather for a given location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "City and optional country, e.g. 'London, UK'."
                        ),
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": (
                            "Temperature unit. Defaults to celsius."
                        ),
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "Retrieve the stock price for a given ticker symbol, "
                "optionally filtered by a time frame."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": (
                            "Stock ticker symbol, e.g. 'RELIANCE.NS', 'AAPL'."
                        ),
                    },
                    "time_frame": {
                        "type": "string",
                        "enum": ["1d", "5d", "1mo", "3mo", "1y"],
                        "description": (
                            "Historical look-back window. "
                            "Defaults to '1d' (current / latest)."
                        ),
                    },
                },
                "required": ["ticker_symbol"],
            },
        },
    },
]

# --------------- Payload Construction ---------------


def build_payload(
    system_prompt: str,
    user_input: str,
    tools: list = None,
) -> dict:
    """Return the complete JSON-serialisable request body."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    }
    if tools:
        payload["tools"] = tools
    return payload


# --------------- Response Parsing ---------------

_SEPARATOR = "-" * 60


def print_response(response) -> None:
    """Pretty-print either tool calls or a text response."""
    choice = response.choices[0]
    message = choice.message

    # -- Tool calls --
    if message.tool_calls:
        print(f"\n{_SEPARATOR}")
        print(f"  [TOOL] Model chose to call {len(message.tool_calls)} tool(s):")
        print(_SEPARATOR)
        for i, tc in enumerate(message.tool_calls, 1):
            fn = tc.function
            try:
                args = json.loads(fn.arguments)
            except (json.JSONDecodeError, TypeError):
                args = fn.arguments
            print(f"\n  [{i}] {fn.name}")
            print(f"      Arguments: {json.dumps(args, indent=6)}")
        print()
        return

    # -- Raw text content (fallback: try parsing JSON tool_calls) --
    content: str = message.content or ""

    # The model was trained to output JSON with a "tool_calls" key.
    # If the server didn't parse it into structured tool_calls, try here.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            calls = parsed["tool_calls"]
            if isinstance(calls, list) and len(calls) > 0:
                print(f"\n{_SEPARATOR}")
                print(f"  [TOOL] Model returned {len(calls)} tool call(s) (raw JSON):")
                print(_SEPARATOR)
                for i, call in enumerate(calls, 1):
                    name = call.get("name", "unknown")
                    args = call.get("arguments", {})
                    print(f"\n  [{i}] {name}")
                    print(f"      Arguments: {json.dumps(args, indent=6)}")
                print()
                return
            # Empty tool_calls list → conversational response
    except (json.JSONDecodeError, TypeError):
        pass

    # -- Plain conversational reply --
    print(f"\n{_SEPARATOR}")
    print(f"  [CHAT] Assistant:")
    print(_SEPARATOR)
    print(f"\n  {content}\n")


# --------------- Payload Validation ---------------

_FORBIDDEN_TAGS = [
    "[BEGIN OF TASK INSTRUCTION]",
    "[END OF TASK INSTRUCTION]",
    "[BEGIN OF AVAILABLE TOOLS]",
    "[END OF AVAILABLE TOOLS]",
    "[BEGIN OF FORMAT INSTRUCTION]",
    "[END OF FORMAT INSTRUCTION]",
    "<|im_start|>",
    "<|im_end|>",
]


def validate_payload(payload: dict) -> list[str]:
    """Return a list of issues (empty = OK)."""
    issues: list[str] = []

    # Check message roles
    for msg in payload.get("messages", []):
        if msg["role"] not in ("system", "user", "assistant", "tool"):
            issues.append(f"Unexpected role: {msg['role']}")
        # Check for forbidden structural tags in content
        for tag in _FORBIDDEN_TAGS:
            if tag in (msg.get("content") or ""):
                issues.append(
                    f"Forbidden tag '{tag}' found in {msg['role']} message content"
                )

    # Check tools present
    if not payload.get("tools"):
        issues.append("Tools list is empty or missing")

    # Check system message exists
    roles = [m["role"] for m in payload.get("messages", [])]
    if "system" not in roles:
        issues.append("No system message in payload")

    return issues



# --------------- Test Mode ---------------

TEST_QUERIES = [
    {
        "label": "Query 1 — Should trigger get_stock_price tool",
        "input": "What is the current stock price of Reliance Industries?",
    },
    {
        "label": "Query 2 — Should trigger conversational response",
        "input": "Hello, how are you today?",
    },
]



def run_tests(*, dry_run: bool = False) -> None:
    """Execute predefined test queries and validate payloads."""
    system_prompt = load_system_prompt()
    client: OpenAI | None = None
    if not dry_run:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

    print("=" * 60)
    print("  CHAT CLIENT -- TEST MODE" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 60)
    print(f"  Base URL : {VLLM_BASE_URL}")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Sys prompt: {_SYSTEM_PROMPT_PATH}")
    print("=" * 60)

    all_passed = True

    for test in TEST_QUERIES:
        print(f"\n{'-' * 60}")
        print(f"  >> {test['label']}")
        print(f"  Input: \"{test['input']}\"")
        print("-" * 60)

        payload = build_payload(system_prompt, test["input"])

        # ── Print raw JSON payload ──
        print("\n  > Raw JSON payload being sent:\n")
        print(json.dumps(payload, indent=2))

        # ── Validate ──
        issues = validate_payload(payload)
        if issues:
            print("\n  [FAIL] VALIDATION FAILED:")
            for issue in issues:
                print(f"     • {issue}")
            all_passed = False
        else:
            print("\n  [PASS] Payload validation passed -- no forbidden tags, "
                  "structure is correct.")

        # -- Send request (unless dry run) --
        if not dry_run and client is not None:
            try:
                print("\n  ... Sending request to vLLM server...")
                response = client.chat.completions.create(**payload)
                print_response(response)
            except Exception as exc:
                print(f"\n  [ERROR] API call failed: {exc}")

    print("\n" + "=" * 60)
    if all_passed:
        print("  [PASS] ALL PAYLOAD VALIDATIONS PASSED")
    else:
        print("  [FAIL] SOME VALIDATIONS FAILED -- see above")
    print("=" * 60 + "\n")


# --------------- Interactive Loop ---------------


def interactive_loop() -> None:
    """Run the interactive chat CLI."""
    system_prompt = load_system_prompt()
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

    print("=" * 60)
    print("  CHAT CLIENT -- Interactive Mode")
    print("=" * 60)
    print(f"  Base URL : {VLLM_BASE_URL}")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        tool_keywords = [
            "weather",
            "stock",
            "price",
            "temperature",
        ]

        use_tools = any(
            kw in user_input.lower()
            for kw in tool_keywords
        )

        payload = build_payload(
            system_prompt,
            user_input,
            tools=TOOLS if use_tools else None,
        )

        try:
            response = client.chat.completions.create(**payload)
            print_response(response)
        except Exception as exc:
            print(f"\n  [ERROR] API call failed: {exc}\n")


# --------------- Main ---------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat client for vLLM-served Qwen 2.5 7B tool-calling model.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run predefined test queries instead of the interactive loop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="(Used with --test) Print and validate payloads without sending.",
    )
    args = parser.parse_args()

    if args.test:
        run_tests(dry_run=args.dry_run)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
