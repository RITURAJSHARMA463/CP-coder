"""
Module 3: Merge LoRA Adapters & Export
=======================================
Merges the trained QLoRA adapter weights back into the base model,
then exports in one or more formats:
  1. Full 16-bit merged model → HuggingFace Hub
  2. GGUF quantised model     → HuggingFace Hub  (for llama.cpp / Ollama)

Usage
-----
  python 3_merge_and_export.py \
      --adapter_path outputs/cp_lora/lora_adapter \
      --base_model   Qwen/Qwen2.5-Coder-7B-Instruct \
      --hub_id       your-username/cp-coder-7b \
      --export_gguf
"""

import argparse
import os
from pathlib import Path

from unsloth import FastLanguageModel


# ============================================================
# 1. Load base model + adapter
# ============================================================

def load_merged_model(base_model: str, adapter_path: str, max_seq_length: int = 4096):
    """
    FastLanguageModel.from_pretrained loads the 4-bit base model.
    We then call .merge_and_unload() to fuse adapter weights back into
    the base weights and unload the PEFT wrapper — producing a standard
    HuggingFace model that can be saved/pushed normally.
    """
    print(f"[Merge] Loading base model '{base_model}' …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,    # Unsloth finds adapter_config.json here
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    print("[Merge] Fusing LoRA weights into base model …")
    # merge_and_unload: absorbs A·B into W, removes adapter overhead.
    # After this call the model behaves identically to a full-precision model.
    model = model.merge_and_unload()
    print("[Merge] Merge complete.")
    return model, tokenizer


# ============================================================
# 2. Save 16-bit merged model locally
# ============================================================

def save_16bit(model, tokenizer, output_dir: str):
    """
    Saves the merged model in bfloat16.
    This is the recommended format for continued training or serving
    via vLLM / HuggingFace Inference Endpoints.
    """
    out = Path(output_dir) / "merged_16bit"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[Save] Writing 16-bit merged model → {out} …")
    model.save_pretrained_merged(
        str(out),
        tokenizer,
        save_method="merged_16bit",  # bf16 full precision
    )
    print(f"[Save] 16-bit model saved → {out}")
    return str(out)


# ============================================================
# 3. Push 16-bit model to HuggingFace Hub
# ============================================================

def push_16bit_to_hub(model, tokenizer, hub_id: str):
    """
    Pushes the merged 16-bit model to the HF Hub.
    Requires HF_TOKEN env var (or `huggingface-cli login`).

    The resulting repo will be publicly accessible and compatible with
    AutoModelForCausalLM.from_pretrained() out of the box.
    """
    print(f"[Hub] Pushing 16-bit model → hf.co/{hub_id} …")
    model.push_to_hub_merged(
        hub_id,
        tokenizer,
        save_method="merged_16bit",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"[Hub] 16-bit model live at: https://huggingface.co/{hub_id}")


# ============================================================
# 4. Export to GGUF (llama.cpp compatible)
# ============================================================

def export_gguf(model, tokenizer, hub_id: str, quantisation: str = "q4_k_m"):
    """
    Converts the model to GGUF format using Unsloth's built-in exporter.
    GGUF models can be run with llama.cpp, Ollama, LM Studio, and Jan.

    Common quantisation options:
      q4_k_m  → 4-bit, medium quality  (~4 GB for 7B) ← recommended
      q5_k_m  → 5-bit, higher quality  (~5 GB for 7B)
      q8_0    → 8-bit, near-lossless   (~8 GB for 7B)
      f16     → full 16-bit (large, but lossless)
    """
    gguf_hub_id = f"{hub_id}-GGUF"
    print(f"[GGUF] Exporting '{quantisation}' GGUF → hf.co/{gguf_hub_id} …")

    model.push_to_hub_gguf(
        gguf_hub_id,
        tokenizer,
        quantization_method=quantisation,   # e.g. "q4_k_m"
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"[GGUF] GGUF model live at: https://huggingface.co/{gguf_hub_id}")


# ============================================================
# 5. Save 4-bit model (optional — for fast Unsloth reloading)
# ============================================================

def save_4bit(model, tokenizer, output_dir: str):
    """
    Saves the merged model in 4-bit NF4 format.
    Only Unsloth can reload this format efficiently.
    Useful for sharing with users who will continue training.
    """
    out = Path(output_dir) / "merged_4bit"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[Save] Writing 4-bit merged model → {out} …")
    model.save_pretrained_merged(
        str(out),
        tokenizer,
        save_method="merged_4bit_forced",
    )
    print(f"[Save] 4-bit model saved → {out}")
    return str(out)


# ============================================================
# Main
# ============================================================

def main(args):
    model, tokenizer = load_merged_model(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        max_seq_length=args.max_seq_length,
    )

    if args.save_local:
        save_16bit(model, tokenizer, args.local_dir)

    if args.hub_id:
        push_16bit_to_hub(model, tokenizer, args.hub_id)

        if args.export_gguf:
            export_gguf(model, tokenizer, args.hub_id, args.gguf_quant)

        if args.export_4bit:
            # Push 4-bit version under a separate repo name
            model.push_to_hub_merged(
                f"{args.hub_id}-4bit",
                tokenizer,
                save_method="merged_4bit_forced",
                token=os.environ.get("HF_TOKEN"),
            )
            print(f"[Hub] 4-bit model → https://huggingface.co/{args.hub_id}-4bit")

    print("[Done] All exports complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and export model.")
    parser.add_argument(
        "--adapter_path",
        required=True,
        help="Path to saved LoRA adapter directory (from Module 2).",
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Base model identifier (must match training).",
    )
    parser.add_argument(
        "--hub_id",
        default=None,
        help="HuggingFace Hub repo ID, e.g. 'yourname/cp-coder-7b'.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Max sequence length (must match training).",
    )
    parser.add_argument(
        "--save_local",
        action="store_true",
        help="Save 16-bit merged model to disk.",
    )
    parser.add_argument(
        "--local_dir",
        default="outputs/export",
        help="Directory for local saves.",
    )
    parser.add_argument(
        "--export_gguf",
        action="store_true",
        help="Also export GGUF model to Hub.",
    )
    parser.add_argument(
        "--gguf_quant",
        default="q4_k_m",
        help="GGUF quantisation level (default: q4_k_m).",
    )
    parser.add_argument(
        "--export_4bit",
        action="store_true",
        help="Also push 4-bit NF4 model to Hub.",
    )
    args = parser.parse_args()
    main(args)
