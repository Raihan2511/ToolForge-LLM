"""
src/dataset.py
Load train.jsonl / val.jsonl and apply Qwen2.5's chat template.

Returns a HuggingFace Dataset with a single "text" column.
SFTTrainer (trl) consumes it via dataset_text_field="text".

Why apply_chat_template here instead of inside the trainer?
  Unsloth's SFTTrainer expects pre-formatted strings in the "text" field.
  Doing the formatting here also lets you inspect exactly what the model
  will see before training starts.
"""

import json
import os
from datasets import Dataset
from transformers import PreTrainedTokenizer


def load_jsonl(path: str) -> list[dict]:
    path = os.path.expanduser(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_sample(sample: dict, tokenizer: PreTrainedTokenizer) -> str | None:
    """
    Apply Qwen2.5's ChatML template to one conversation.

    Qwen2.5-Instruct ChatML format:
        <|im_start|>system
        ...system content...<|im_end|>
        <|im_start|>user
        ...user content...<|im_end|>
        <|im_start|>assistant
        ...assistant content...<|im_end|>

    add_generation_prompt=False:
        We include the full conversation including the final assistant turn.
        The label-masking in the collator will handle which tokens to train on.
    """
    conversation = sample.get("conversation", [])

    if len(conversation) < 3:
        return None
    if conversation[0]["role"] != "system":
        return None
    if conversation[-1]["role"] != "assistant":
        return None

    try:
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return None


def build_dataset(
    jsonl_path: str,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """Load JSONL → HF Dataset with "text" and "source" columns."""
    raw = load_jsonl(jsonl_path)

    records = []
    skipped = 0
    for sample in raw:
        text = format_sample(sample, tokenizer)
        if text is None:
            skipped += 1
            continue
        records.append({
            "text":   text,
            "source": sample.get("source", "unknown"),
        })

    print(f"  {os.path.basename(jsonl_path)}: "
          f"{len(records):,} loaded, {skipped:,} skipped")
    return Dataset.from_list(records)
