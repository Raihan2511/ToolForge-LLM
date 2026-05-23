"""
scripts/merge_and_save.py
Merge LoRA adapter weights into the base model via Unsloth, then save
in multiple formats.

Unsloth's merge is faster and more memory-efficient than standard PEFT
merge_and_unload() because it uses its own fused dequantisation pass.

Outputs:
  --output_dir/          → merged HuggingFace model (safetensors)
  --output_dir/gguf/     → GGUF quantised models (optional, for llama.cpp)

Usage:
    # Merge to HF safetensors
    python scripts/merge_and_save.py \
        --adapter_path ~/training/runs/qwen7b_unsloth/final_adapter \
        --output_dir   ~/training/merged

    # Also export GGUF (q4_k_m and q8_0)
    python scripts/merge_and_save.py \
        --adapter_path ~/training/runs/qwen7b_unsloth/final_adapter \
        --output_dir   ~/training/merged \
        --export_gguf

    # Push merged model to HF Hub
    python scripts/merge_and_save.py \
        --adapter_path ~/training/runs/qwen7b_unsloth/final_adapter \
        --output_dir   ~/training/merged \
        --push_to_hub  your-username/qwen7b-toolcalling \
        --hf_token     hf_xxxxx
"""

import argparse
import json
from pathlib import Path
from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--export_gguf",  action="store_true",
                        help="Also export GGUF quantised models")
    parser.add_argument("--push_to_hub",  default=None,
                        help="HF repo ID e.g. myuser/my-model")
    parser.add_argument("--hf_token",     default=None)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Adapter    : {adapter_path}")
    print(f"Output dir : {output_dir}")

    # Load via Unsloth
    print("\nLoading model + adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = str(adapter_path),
        max_seq_length= args.max_seq_length,
        load_in_4bit  = True,
        dtype         = None,
    )

    # ── Merge to 16-bit HF safetensors ───────────────────────────────────────
    print(f"\nMerging and saving to {output_dir} ...")
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method = "merged_16bit",
        # "merged_16bit" → dequantise 4-bit base + merge adapters → bf16 weights
        # Use "merged_4bit" if you want to keep 4-bit for space savings
        # Use "lora" if you just want to re-save the adapter without merging
    )
    print("✅ Merged model saved.")

    # ── Optional: export GGUF ─────────────────────────────────────────────────
    if args.export_gguf:
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(exist_ok=True)
        print(f"\nExporting GGUF models to {gguf_dir} ...")

        for quant in ["q4_k_m", "q8_0"]:
            print(f"  Quantising {quant}...")
            model.save_pretrained_gguf(
                str(gguf_dir / quant),
                tokenizer,
                quantization_method = quant,
            )
            print(f"  ✓ {quant} saved")

    # ── Optional: push to HF Hub ─────────────────────────────────────────────
    if args.push_to_hub:
        if args.hf_token:
            from huggingface_hub import login
            login(token=args.hf_token)

        print(f"\nPushing to Hub: {args.push_to_hub} ...")
        model.push_to_hub_merged(
            args.push_to_hub,
            tokenizer,
            save_method    = "merged_16bit",
            token          = args.hf_token,
        )
        print(f"✅ Pushed → https://huggingface.co/{args.push_to_hub}")

        if args.export_gguf:
            print(f"Pushing GGUF...")
            model.push_to_hub_gguf(
                args.push_to_hub,
                tokenizer,
                quantization_method = ["q4_k_m", "q8_0"],
                token = args.hf_token,
            )
            print(f"✅ GGUF pushed → https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
