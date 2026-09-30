"""
Fine-Tuning Pipeline — Unsloth + QLoRA + SFTTrainer + W&B
=====================================================================
Trains Qwen2.5-Coder-7B (or DeepSeek-Coder-7B) on the ChatML .jsonl
produced by Module 1, optimised for a single consumer GPU (≤24 GB VRAM).

Key design decisions
--------------------
- 4-bit NF4 quantisation via bitsandbytes (via Unsloth) → ~7 GB base footprint
- QLoRA: rank-16 adapters injected into every linear projection
- Gradient checkpointing for memory-efficient backprop
- Packed sequences + flash-attention-2 (Unsloth enables both automatically)
- W&B tracks loss curves, learning-rate schedule, and GPU utilisation
"""

import os
import math
import argparse
import wandb

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer , SFTConfig
from transformers import TrainingArguments


# ============================================================
# 0. Configuration — edit these or pass via CLI
# ============================================================

DEFAULT_CONFIG = {
    # --- Model ---
    "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",

    # --- Data ---
    "data_path": "data/cp_chatml.jsonl",
    "val_split": 0.02,         
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
    "max_seq_length": 1024,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,

    "num_train_epochs": 3,
    "warmup_ratio": 10,       # 5 % of steps used for LR warm-up
    "weight_decay": 0.01,       # L2 regularisation on non-LoRA params
    "lr_scheduler_type": "cosine",  # cosine decay → smooth convergence

    # --- Precision ---
    "bf16": False,   # Use BF16 on Ampere+ GPUs (A100, 4090, etc.)
    "fp16": True,  # Fall back to FP16 if bf16 unsupported

    # --- Logging & Saving ---
    "output_dir": "outputs/cp_lora",
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 3,
    "eval_strategy": "steps",
    "eval_steps": 100,

    # --- W&B ---
    "wandb_project": "cp-coder-finetune",
    "wandb_run_name": "qwen2.5-coder-7b-qlora",
}


# ============================================================
# 1. Load base model + tokeniser via Unsloth
# ============================================================

def load_model_and_tokenizer(cfg: dict):
    """
    FastLanguageModel.from_pretrained wraps HuggingFace's AutoModel with:
      - 4-bit NF4 quantisation (QLoRA-ready)
      - Automatic flash-attention-2 if available
      - Fused kernels for faster forward/backward passes
    """
    print(f"[Model] Loading '{cfg['model_name']}' in 4-bit …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,         # auto-detect: bf16 on Ampere, fp16 otherwise
        load_in_4bit=True,  # QLoRA: base weights stored as 4-bit NF4
    )
    return model, tokenizer


# ============================================================
# 2. Attach LoRA adapters
# ============================================================

def attach_lora(model, cfg: dict):
    """
    Injects rank-decomposed adapter matrices (A·B) into every targeted
    linear layer.  Only adapter weights are trainable (~1-3 % of params).

    get_peft_model is Unsloth's patched version — it enables gradient
    checkpointing and fuses the LoRA ops for ~30 % speedup over vanilla PEFT.
    """
    print(f"[LoRA] rank={cfg['lora_r']}, alpha={cfg['lora_alpha']}, "
          f"targets={cfg['lora_target_modules']}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",                 # don't adapt bias terms
        use_gradient_checkpointing="unsloth",  # Unsloth's memory-efficient impl
        random_state=42,
        use_rslora=False,            # set True to try rank-stabilised LoRA
        loftq_config=None,           # set for LoftQ initialisation (alternative)
    )

    # Print trainable parameter count
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LoRA] Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f} %)")
    return model


# ============================================================
# 3. Load & preprocess the dataset
# ============================================================

def load_and_format_dataset(tokenizer, cfg: dict):
    """
    Reads the ChatML .jsonl, applies the tokenizer's chat template,
    and splits into train / validation.
    """
    print(f"[Data] Loading '{cfg['data_path']}' …")
    raw = load_dataset("json", data_files=cfg["data_path"], split="train")

    # Apply the model's native chat template (handles <|im_start|> tokens, etc.)
    def apply_template(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = raw.map(apply_template, batched=True, remove_columns=raw.column_names)

    # Train / validation split
    val_size  = max(1, math.floor(len(dataset) * cfg["val_split"]))
    split     = dataset.train_test_split(test_size=val_size, seed=42)
    print(f"[Data] Train: {len(split['train']):,} | Val: {len(split['test']):,}")
    return split["train"], split["test"]


# ============================================================
# 4. Build SFTTrainer
# ============================================================

def build_trainer(model, tokenizer, train_ds, eval_ds, cfg: dict):
    """
    SFTTrainer wraps HuggingFace Trainer with:
      - Automatic sequence packing (fills context window efficiently)
      - Dataset formatting via 'dataset_text_field'
      - Native integration with PEFT / LoRA models
    """
    # Use SFTConfig instead of TrainingArguments for modern trl versions
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],

        # --- Batch & gradient ---
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=True,  # re-compute activations to save VRAM

        # --- Optimiser ---
        optim="adamw_8bit",           # 8-bit Adam → ~50 % less optimiser memory
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],

        # --- Precision ---
        bf16=cfg["bf16"],
        fp16=cfg["fp16"],

        # --- Logging ---
        logging_dir=f"{cfg['output_dir']}/logs",
        logging_steps=cfg["logging_steps"],
        report_to="wandb",             # stream metrics to W&B

        # --- Evaluation & checkpointing ---
        eval_strategy=cfg["eval_strategy"], 
        eval_steps=cfg["eval_steps"],
        save_strategy=cfg["eval_strategy"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # --- Misc ---
        dataloader_num_workers=4,
        seed=42,
        # group_by_length=True,       <--- REMOVED this line because it conflicts with packing!
        max_grad_norm=1.0,            # gradient clipping for stability
        
        # --- SFT Specific Arguments (Moved here in trl >= 0.20.0) ---
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        dataset_num_proc=4,
        packing=True,                 # pack short sequences to fill context window
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,           
    )
    return trainer

# ============================================================
# 5. Main training loop
# ============================================================

def train(cfg: dict):
    # --- W&B initialisation ---
    wandb.init(
        project=cfg["wandb_project"],
        name=cfg["wandb_run_name"],
        config=cfg,
    )

    model, tokenizer = load_model_and_tokenizer(cfg)
    model            = attach_lora(model, cfg)
    train_ds, eval_ds = load_and_format_dataset(tokenizer, cfg)
    trainer          = build_trainer(model, tokenizer, train_ds, eval_ds, cfg)

    print("[Train] Starting fine-tuning …")
    trainer_stats = trainer.train()

    print(f"[Train] Finished. "
          f"Steps: {trainer_stats.global_step} | "
          f"Final loss: {trainer_stats.training_loss:.4f}")

    # Save LoRA adapter weights (NOT the merged model — see Module 3)
    adapter_path = f"{cfg['output_dir']}/lora_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[Train] LoRA adapter saved → {adapter_path}")

    wandb.finish()
    return model, tokenizer


# ============================================================
# CLI entry-point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CP coder model with Unsloth + QLoRA")
    parser.add_argument("--model_name",      default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--data_path",       default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--output_dir",      default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--lora_r",          type=int,   default=DEFAULT_CONFIG["lora_r"])
    parser.add_argument("--lora_alpha",      type=int,   default=DEFAULT_CONFIG["lora_alpha"])
    parser.add_argument("--learning_rate",   type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--num_epochs",      type=int,   default=DEFAULT_CONFIG["num_train_epochs"])
    parser.add_argument("--batch_size",      type=int,   default=DEFAULT_CONFIG["per_device_train_batch_size"])
    parser.add_argument("--grad_accum",      type=int,   default=DEFAULT_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--max_seq_length",  type=int,   default=DEFAULT_CONFIG["max_seq_length"])
    parser.add_argument("--wandb_project",   default=DEFAULT_CONFIG["wandb_project"])
    parser.add_argument("--wandb_run_name",  default=DEFAULT_CONFIG["wandb_run_name"])
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG}
    cfg.update({
        "model_name":                  args.model_name,
        "data_path":                   args.data_path,
        "output_dir":                  args.output_dir,
        "lora_r":                      args.lora_r,
        "lora_alpha":                  args.lora_alpha,
        "learning_rate":               args.learning_rate,
        "num_train_epochs":            args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_seq_length":              args.max_seq_length,
        "wandb_project":               args.wandb_project,
        "wandb_run_name":              args.wandb_run_name,
    })

    train(cfg)
