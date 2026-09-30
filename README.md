# CP Coder — Fine-Tuning Qwen2.5-Coder-7B for Competitive Programming

A complete pipeline to fine-tune a 7B code model on competitive programming
problems and deploy it as a streaming web app.

---

## Project Structure

```
cp_finetune/
├── 1_prepare_data.py      # Module 1 — Dataset → ChatML .jsonl
├── 2_train.py             # Module 2 — Unsloth + QLoRA fine-tuning
├── 3_merge_and_export.py  # Module 3 — Merge adapters + push to Hub
├── 4_app.py               # Module 4 — Gradio streaming UI
└── requirements.txt
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create a clean environment
conda create -n cp_coder python=3.11 -y
conda activate cp_coder

# Install PyTorch (adjust cu121 → your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth (includes flash-attn, bitsandbytes)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Login to HuggingFace & W&B

```bash
huggingface-cli login      # paste your HF write token
wandb login                # paste your W&B API key
```

### 3. Prepare Data (Module 1)

```bash
python 1_prepare_data.py \
    --dataset deepmind/code_contests \
    --split   train \
    --output  data/cp_chatml.jsonl \
    --max_examples 50000
```

This downloads the CodeContests dataset and writes a ChatML `.jsonl` file.
Each record contains:
- **System**: Expert CP persona
- **User**: Problem statement + constraints + sample I/O
- **Assistant**: `### Strategy & Complexity Analysis` + C++20 code

### 4. Fine-Tune (Module 2)

```bash
python 2_train.py \
    --model_name  Qwen/Qwen2.5-Coder-7B-Instruct \
    --data_path   data/cp_chatml.jsonl \
    --output_dir  outputs/cp_lora \
    --lora_r      16 \
    --lora_alpha  32 \
    --learning_rate 2e-4 \
    --num_epochs  3 \
    --batch_size  2 \
    --grad_accum  8
```

**VRAM estimates:**
| GPU       | Batch × Accum | Approx VRAM |
|-----------|--------------|-------------|
| RTX 4090  | 2 × 8        | ~18 GB      |
| A100 40GB | 4 × 4        | ~24 GB      |
| RTX 3090  | 1 × 16       | ~22 GB      |

Monitor training at **https://wandb.ai/your-project**.

### 5. Merge & Export (Module 3)

```bash
# Merge and push to Hub (16-bit + GGUF)
python 3_merge_and_export.py \
    --adapter_path outputs/cp_lora/lora_adapter \
    --base_model   Qwen/Qwen2.5-Coder-7B-Instruct \
    --hub_id       your-username/cp-coder-7b \
    --save_local \
    --export_gguf \
    --gguf_quant   q4_k_m
```

This creates two HuggingFace repos:
- `your-username/cp-coder-7b`      — full 16-bit model
- `your-username/cp-coder-7b-GGUF` — quantised GGUF (llama.cpp / Ollama ready)

### 6. Launch the Showcase UI (Module 4)

```bash
python 4_app.py --model_path your-username/cp-coder-7b
```

Gradio will print a public URL like:
```
Running on public URL: https://xxxx.gradio.live
```
Share this link directly with recruiters — no server setup required.

---

## Key Hyperparameter Reference

| Parameter              | Value | Effect |
|------------------------|-------|--------|
| `lora_r`               | 16    | Adapter rank — higher = more capacity, more VRAM |
| `lora_alpha`           | 32    | Scaling (≈ 2×r recommended) |
| `lora_dropout`         | 0.05  | Regularisation on adapter weights |
| `learning_rate`        | 2e-4  | QLoRA sweet-spot for 7B models |
| `gradient_accumulation`| 8     | Effective batch = physical_batch × accum |
| `warmup_ratio`         | 0.05  | 5 % of steps for LR ramp-up |
| `max_seq_length`       | 4096  | Covers most CP problem+solution pairs |

---

## Switching to DeepSeek-Coder-7B

Change `--model_name` in all scripts:
```bash
--model_name deepseek-ai/deepseek-coder-7b-instruct-v1.5
```
Everything else stays identical — Unsloth supports both architectures.

---

## Tips for Better Results

1. **Data quality > data quantity** — Filter for problems with high-quality editorial solutions.
2. **Increase `lora_r` to 32** if the model struggles with complex DP / graph problems.
3. **Use a teacher model** (GPT-4 / Claude) to generate the `### Strategy` section instead of heuristics.
4. **Curriculum training** — start with easy problems, progressively add harder ones.
5. **Evaluate on Codeforces** — use the CF API to submit generated solutions and track AC rate.
