"""
Module 4: Gradio Showcase UI
=============================
Loads the fine-tuned CP model and provides a streaming C++ solution
generator that recruiters / users can access via a public share link.

Usage
-----
  python 4_app.py --model_path your-username/cp-coder-7b
  # or point to a local directory:
  python 4_app.py --model_path outputs/export/merged_16bit

The Gradio UI streams the model's response token-by-token so the user
can see the solution being generated in real time.
"""

import argparse
import threading
from queue import Queue, Empty

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ============================================================
# Defaults
# ============================================================

DEFAULT_MODEL      = "your-username/cp-coder-7b"   # replace with your Hub ID
MAX_NEW_TOKENS     = 2048
TEMPERATURE        = 0.2    # low temperature → deterministic, correct code
TOP_P              = 0.9
REPETITION_PENALTY = 1.1    # mild penalty to avoid repeated include / using lines

SYSTEM_PROMPT = (
    "You are an elite Competitive Programmer with deep expertise in algorithms, "
    "data structures, and C++20. When given a problem, you:\n"
    "1. Analyse time and space complexity rigorously using Big-O notation.\n"
    "2. Choose the optimal algorithmic paradigm (DP, graphs, segment trees, etc.).\n"
    "3. Write clean, fast, and correct C++20 code using #include <bits/stdc++.h>.\n"
    "Never sacrifice correctness for brevity."
)

EXAMPLE_PROBLEM = """\
Given an array of N integers, find the length of the longest increasing subsequence (LIS).

Constraints:
  1 ≤ N ≤ 100,000
  -10^9 ≤ A[i] ≤ 10^9
  Time limit: 1 second

Sample Input:
  5
  3 1 4 1 5

Sample Output:
  3
"""

# ============================================================
# Model loading
# ============================================================

_model     = None
_tokenizer = None
_device    = None


def load_model(model_path: str):
    """
    Loads the fine-tuned model.
    Tries Unsloth first (fastest inference), falls back to HuggingFace.
    """
    global _model, _tokenizer, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[App] Loading model from '{model_path}' on {_device} …")

    try:
        # Prefer Unsloth for ~2× faster inference on GPU
        from unsloth import FastLanguageModel
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_model)  # enable Unsloth inference optimisations
        print("[App] Loaded via Unsloth (optimised inference).")
    except ImportError:
        print("[App] Unsloth not found — falling back to HuggingFace Transformers.")
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model     = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if _device == "cuda" else torch.float32,
            device_map="auto",
        )

    _model.eval()
    print("[App] Model ready.")


# ============================================================
# Inference — streaming generation
# ============================================================

def build_prompt(problem_statement: str) -> str:
    """Format the user's problem into the ChatML template."""
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": problem_statement.strip()},
    ]
    # Use the tokenizer's built-in chat template
    return _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def stream_solution(problem_statement: str, temperature: float, max_tokens: int):
    """
    Generator that yields incremental text chunks so Gradio can stream them.

    Uses HuggingFace's TextIteratorStreamer, which runs model.generate()
    in a background thread and puts tokens into a queue that we read here.
    """
    if not problem_statement.strip():
        yield "⚠️  Please enter a problem statement."
        return

    prompt  = build_prompt(problem_statement)
    inputs  = _tokenizer(prompt, return_tensors="pt").to(_device)

    # TextIteratorStreamer decodes tokens as they are generated and
    # puts them into an internal queue.  We iterate over it on the main
    # thread to feed Gradio's streaming response.
    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,          # don't echo the input back
        skip_special_tokens=True,
    )

    gen_kwargs = {
        **inputs,
        "streamer":           streamer,
        "max_new_tokens":     max_tokens,
        "temperature":        temperature,
        "top_p":              TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "do_sample":          temperature > 0.01,
        "pad_token_id":       _tokenizer.eos_token_id,
    }

    # Run generation in a daemon thread so the UI stays responsive
    thread = threading.Thread(target=_model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    # Accumulate and yield chunks
    partial = ""
    for chunk in streamer:
        partial += chunk
        yield partial  # Gradio receives incremental updates

    thread.join()


# ============================================================
# Gradio UI
# ============================================================

def build_ui():
    """Construct the Gradio Blocks interface."""

    with gr.Blocks(
        title="CP Coder — AI Competitive Programming Assistant",
        theme=gr.themes.Monochrome(),
        css="""
            /* Custom CSS for a clean, dark CP-terminal aesthetic */
            body { font-family: 'JetBrains Mono', 'Fira Code', monospace !important; }
            .gr-button-primary { background: #00c853 !important; color: #000 !important; }
            #output_box textarea {
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 0.85rem !important;
            }
            .title-text { text-align: center; color: #00c853; }
        """,
    ) as demo:

        # --- Header ---
        gr.Markdown(
            """
            # 🏆 CP Coder — AI Competitive Programming Assistant
            *Fine-tuned on competitive programming problems · C++20 · Strategy-first responses*
            ---
            """,
            elem_classes=["title-text"],
        )

        with gr.Row():
            # --- Left column: inputs ---
            with gr.Column(scale=1):
                problem_input = gr.Textbox(
                    label="📋 Problem Statement",
                    placeholder="Paste your competitive programming problem here…",
                    lines=18,
                    value=EXAMPLE_PROBLEM,
                    elem_id="problem_box",
                )

                with gr.Accordion("⚙️ Generation Settings", open=False):
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=TEMPERATURE,
                        step=0.05,
                        label="Temperature (lower = more deterministic)",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=256,
                        maximum=4096,
                        value=MAX_NEW_TOKENS,
                        step=128,
                        label="Max New Tokens",
                    )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Generate Solution", variant="primary", scale=3)
                    clear_btn  = gr.Button("🗑️ Clear", scale=1)

            # --- Right column: output ---
            with gr.Column(scale=1):
                output_box = gr.Textbox(
                    label="💡 Strategy & C++ Solution (streaming)",
                    lines=30,
                    interactive=False,
                    elem_id="output_box",
                    show_copy_button=True,
                )

        # --- Example problems ---
        gr.Markdown("### 📚 Quick Examples")
        gr.Examples(
            examples=[
                ["Given N, count the number of prime numbers less than or equal to N.\n"
                 "Constraints: 1 ≤ N ≤ 10^7\nTime limit: 1s"],
                ["Find the shortest path from node 1 to node N in a weighted directed graph.\n"
                 "Constraints: 1 ≤ N ≤ 100,000 | 1 ≤ M ≤ 500,000 | weights ≥ 0"],
                ["Given a string S, find the longest palindromic substring.\n"
                 "Constraints: 1 ≤ |S| ≤ 5,000"],
            ],
            inputs=[problem_input],
        )

        # --- Footer ---
        gr.Markdown(
            """
            ---
            *Model: Fine-tuned Qwen2.5-Coder-7B · Training: QLoRA via Unsloth · Framework: HuggingFace TRL*
            """,
            elem_classes=["title-text"],
        )

        # --- Event bindings ---
        submit_btn.click(
            fn=stream_solution,
            inputs=[problem_input, temperature_slider, max_tokens_slider],
            outputs=output_box,
            api_name="generate",   # exposes a REST endpoint at /api/generate
        )

        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[problem_input, output_box],
        )

        # Also trigger on Ctrl+Enter in the problem textarea
        problem_input.submit(
            fn=stream_solution,
            inputs=[problem_input, temperature_slider, max_tokens_slider],
            outputs=output_box,
        )

    return demo


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch CP Coder Gradio UI")
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL,
        help="Path to fine-tuned model (local dir or HF Hub ID).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port for Gradio server (default: 7860).",
    )
    parser.add_argument(
        "--no_share",
        action="store_true",
        help="Disable Gradio's public share link (enabled by default).",
    )
    args = parser.parse_args()

    # Load model before launching UI (avoids cold-start on first request)
    load_model(args.model_path)

    demo = build_ui()
    demo.queue(max_size=5)         # queue up to 5 concurrent requests
    demo.launch(
        server_name="0.0.0.0",    # listen on all interfaces
        server_port=args.port,
        share=not args.no_share,  # True → generates a public gradio.live link
        show_error=True,
        favicon_path=None,
    )
