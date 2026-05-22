"""
scripts/compare_evals.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Side-by-Side Comparison of Base Model vs Fine-Tuned Model Evaluation Results.

Reads the eval_results.json output files from both evaluation pipelines
and prints a formatted comparison table showing:
  • AST Success Rate
  • Execution Success Rate
  • Format Error Rate
  • Absolute delta (Δ) between the two runs

Usage:
    python scripts/compare_evals.py
    python scripts/compare_evals.py \
        --base_results outputs/eval_base/eval_results.json \
        --ft_results   outputs/eval_latest/eval_results.json

Author: Auto-generated comparison utility
Python: 3.10+
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ── Fix Windows console encoding for Unicode box-drawing characters ──
# The default cp1252 codec on Windows cannot encode ━, █, etc.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare BFCL evaluation results: Base Qwen2.5-7B-Instruct "
            "vs Fine-Tuned model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base_results",
        type=str,
        default="outputs/eval_base/eval_results.json",
        help=(
            "Path to base model eval_results.json "
            "(default: outputs/eval_base/eval_results.json)."
        ),
    )
    parser.add_argument(
        "--ft_results",
        type=str,
        default="outputs/eval_latest/eval_results.json",
        help=(
            "Path to fine-tuned model eval_results.json "
            "(default: outputs/eval_latest/eval_results.json)."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metric Computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_rates(records: list[dict]) -> dict[str, float]:
    """
    Compute the three core metrics from a list of per-sample eval records.

    Each record is expected to have (at minimum):
      - ast_matched: bool
      - exec_passed: bool
      - has_format_error: bool

    Args:
        records: List of per-sample result dicts (from eval_results.json).

    Returns:
        Dict with keys: ast_success_rate, exec_success_rate, format_error_rate
        (all as percentages, e.g. 78.5).
    """
    total = len(records)
    if total == 0:
        return {
            "total_samples": 0,
            "ast_success_rate": 0.0,
            "exec_success_rate": 0.0,
            "format_error_rate": 0.0,
        }

    ast_pass = sum(1 for r in records if r.get("ast_matched", False))
    exec_pass = sum(1 for r in records if r.get("exec_passed", False))
    format_err = sum(1 for r in records if r.get("has_format_error", False))

    return {
        "total_samples": total,
        "ast_success_rate": round(ast_pass / total * 100, 2),
        "exec_success_rate": round(exec_pass / total * 100, 2),
        "format_error_rate": round(format_err / total * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# File Loading with Error Handling
# ─────────────────────────────────────────────────────────────────────────────
def load_results(file_path: str, label: str) -> list[dict] | None:
    """
    Load a JSON results file, returning None if it doesn't exist.

    Args:
        file_path: Path to the eval_results.json file.
        label: Human-readable label for error messages (e.g. "Base Model").

    Returns:
        Parsed list of record dicts, or None on error.
    """
    path = Path(file_path)

    if not path.exists():
        print(f"\n  ⚠️  {label} results not found: {path}")
        print(f"      → Run the corresponding evaluation script first.")
        if "base" in label.lower():
            print(f"      → Command: python scripts/eval_base_qwen.py")
        else:
            print(f"      → Command: python scripts/evaluation_latest.py --model_path <adapter_dir>")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"\n  ❌  {label} results file has unexpected format (expected JSON array)")
            return None
        return data
    except json.JSONDecodeError as e:
        print(f"\n  ❌  {label} results file is not valid JSON: {e}")
        return None
    except Exception as e:
        print(f"\n  ❌  Error reading {label} results: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Delta Formatting
# ─────────────────────────────────────────────────────────────────────────────
def format_delta(ft_val: float, base_val: float, invert: bool = False) -> str:
    """
    Format the absolute difference between fine-tuned and base model metrics.

    For success rates: positive delta = improvement (green ↑)
    For error rates:   negative delta = improvement (green ↓, inverted)

    Args:
        ft_val: Fine-tuned model metric value.
        base_val: Base model metric value.
        invert: If True, a negative delta is treated as an improvement
                (used for error rates where lower is better).

    Returns:
        Formatted delta string like "+45.2%" or "-10.5%".
    """
    delta = ft_val - base_val

    if abs(delta) < 0.005:
        return "  0.0%"

    sign = "+" if delta > 0 else ""
    indicator = ""

    # Determine if this delta is an improvement or regression
    if invert:
        # For error rates, a decrease is good
        indicator = " ↓" if delta < 0 else " ↑"
    else:
        # For success rates, an increase is good
        indicator = " ↑" if delta > 0 else " ↓"

    return f"{sign}{delta:.1f}%{indicator}"


# ─────────────────────────────────────────────────────────────────────────────
# Comparison Table Printer
# ─────────────────────────────────────────────────────────────────────────────
def print_comparison(
    base_rates: dict[str, Any],
    ft_rates: dict[str, Any],
) -> None:
    """
    Print a formatted side-by-side comparison table to the terminal.

    Args:
        base_rates: Metrics dict from the base model evaluation.
        ft_rates: Metrics dict from the fine-tuned model evaluation.
    """
    W = 78
    print("\n" + "━" * W)
    print("  📊  EVALUATION COMPARISON — Base vs Fine-Tuned Qwen2.5-7B-Instruct")
    print("━" * W)

    # ── Sample counts ──
    print(f"\n  Base Model Samples:       {base_rates['total_samples']}")
    print(f"  Fine-Tuned Model Samples: {ft_rates['total_samples']}")

    if base_rates["total_samples"] != ft_rates["total_samples"]:
        print("  ⚠️  Warning: Sample counts differ — comparison may not be apples-to-apples")

    # ── Table header ──
    print(f"\n  {'─' * (W - 4)}")
    header = f"  {'Metric':<26} {'Base Model':>12} {'Fine-Tuned':>12} {'Δ (Delta)':>14}"
    print(header)
    print(f"  {'─' * (W - 4)}")

    # ── AST Success Rate ──
    base_ast = base_rates["ast_success_rate"]
    ft_ast = ft_rates["ast_success_rate"]
    delta_ast = format_delta(ft_ast, base_ast, invert=False)
    print(f"  {'AST Success Rate':<26} {base_ast:>11.2f}% {ft_ast:>11.2f}% {delta_ast:>14}")

    # ── Execution Success Rate ──
    base_exec = base_rates["exec_success_rate"]
    ft_exec = ft_rates["exec_success_rate"]
    delta_exec = format_delta(ft_exec, base_exec, invert=False)
    print(f"  {'Execution Success Rate':<26} {base_exec:>11.2f}% {ft_exec:>11.2f}% {delta_exec:>14}")

    # ── Format Error Rate ──
    base_fmt = base_rates["format_error_rate"]
    ft_fmt = ft_rates["format_error_rate"]
    delta_fmt = format_delta(ft_fmt, base_fmt, invert=True)
    print(f"  {'Format Error Rate':<26} {base_fmt:>11.2f}% {ft_fmt:>11.2f}% {delta_fmt:>14}")

    # ── Footer ──
    print(f"  {'─' * (W - 4)}")

    # ── Verdict ──
    print()
    if ft_ast > base_ast and ft_exec > base_exec:
        print("  ✅ VERDICT: Fine-tuning IMPROVED the model on both AST and Execution metrics!")
    elif ft_ast > base_ast:
        print("  🟡 VERDICT: Fine-tuning improved AST accuracy but Execution metrics need review.")
    elif ft_exec > base_exec:
        print("  🟡 VERDICT: Fine-tuning improved Execution accuracy but AST metrics need review.")
    elif ft_ast == base_ast and ft_exec == base_exec:
        print("  🟡 VERDICT: No measurable difference between base and fine-tuned models.")
    else:
        print("  ❌ VERDICT: Fine-tuning may have regressed performance — investigate results.")

    if ft_fmt < base_fmt:
        print("  📝 Format Error Rate decreased — fine-tuned model produces better-structured output.")
    elif ft_fmt > base_fmt:
        print("  ⚠️  Format Error Rate increased — fine-tuned model has more parsing issues.")

    print("\n" + "━" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Main: load both result files → compute metrics → print comparison."""
    args = parse_args()

    # ── Banner ──
    print("\n" + "━" * 78)
    print("  🔬  Evaluation Comparison Utility")
    print("━" * 78)
    print(f"  Base results:       {args.base_results}")
    print(f"  Fine-tuned results: {args.ft_results}")
    print("━" * 78)

    # ── Load results (with graceful error handling) ──
    base_records = load_results(args.base_results, "Base Model")
    ft_records = load_results(args.ft_results, "Fine-Tuned Model")

    # ── Check if both files are available ──
    if base_records is None and ft_records is None:
        print("\n  ❌ Neither result file was found. Please run both evaluations first:")
        print("     1. python scripts/eval_base_qwen.py --dataset_path data/val.jsonl")
        print("     2. python scripts/evaluation_latest.py --model_path <adapter_dir>")
        sys.exit(1)

    if base_records is None:
        print("\n  ❌ Cannot compare without base model results. Exiting.")
        sys.exit(1)

    if ft_records is None:
        print("\n  ❌ Cannot compare without fine-tuned model results. Exiting.")
        sys.exit(1)

    # ── Compute metrics ──
    base_rates = compute_rates(base_records)
    ft_rates = compute_rates(ft_records)

    # ── Print comparison ──
    print_comparison(base_rates, ft_rates)


if __name__ == "__main__":
    main()
