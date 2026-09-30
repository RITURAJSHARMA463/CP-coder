"""
Module 1: Data Structuring Script
==================================
Converts raw Competitive Programming datasets (DeepMind APPS / CodeContests)
into ChatML-formatted .jsonl files for fine-tuning.

Each record follows the structure:
  System  → Expert CP persona
  User    → Problem statement + constraints
  Assistant → Strategy & Complexity Analysis + optimized C++20 code
"""

import json
import re
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# System prompt — defines the model's expert persona
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an elite Competitive Programmer with deep expertise in algorithms, "
    "data structures, and C++20. When given a problem, you:\n"
    "1. Analyse time and space complexity rigorously using Big-O notation.\n"
    "2. Choose the optimal algorithmic paradigm (DP, graphs, segment trees, etc.).\n"
    "3. Write clean, fast, and correct C++20 code using #include <bits/stdc++.h>.\n"
    "4. Prefer concise, idiomatic competitive-programming style.\n"
    "Never sacrifice correctness for brevity."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip excessive whitespace while preserving meaningful newlines."""
    if not text:
        return ""
    # Collapse more than 3 consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_user_message(problem: dict) -> str:
    """
    Construct the user turn from a problem dict.
    Supports both APPS and CodeContests field schemas.
    """
    parts: list[str] = []

    # --- Problem statement ---
    statement = (
        problem.get("question")          # APPS
        or problem.get("description")    # CodeContests
        or problem.get("problem_statement", "")
    )
    parts.append(clean_text(statement))

    # --- Input / Output spec (CodeContests) ---
    for field in ("input_from", "output_to", "time_limit", "memory_limit"):
        val = problem.get(field)
        if val:
            parts.append(f"**{field.replace('_', ' ').title()}:** {val}")

    # --- Constraints block ---
    constraints = problem.get("constraints") or problem.get("notes", "")
    if constraints:
        parts.append(f"\n**Constraints:**\n{clean_text(constraints)}")

    # --- Sample I/O ---
    inputs  = problem.get("input_output", {})  # APPS stores as JSON string
    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except json.JSONDecodeError:
            inputs = {}

    sample_inputs  = inputs.get("inputs",  [])
    sample_outputs = inputs.get("outputs", [])

    # CodeContests keeps samples in separate lists
    sample_inputs  = sample_inputs  or problem.get("public_tests", {}).get("input",  [])
    sample_outputs = sample_outputs or problem.get("public_tests", {}).get("output", [])

    if sample_inputs:
        parts.append("\n**Sample Input/Output:**")
        for i, (inp, out) in enumerate(zip(sample_inputs[:2], sample_outputs[:2]), 1):
            parts.append(f"Example {i}:\nInput:\n{inp}\nOutput:\n{out}")

    return "\n\n".join(filter(None, parts))


def build_assistant_message(solution: str, problem: dict) -> str:
    """
    Wrap a raw solution in the expected multi-step format.
    """
    solution = clean_text(solution)

    # Heuristic: infer algorithmic tags from dataset metadata when available
    tags = problem.get("cf_tags") or problem.get("tags") or []
    tag_hint = (", ".join(tags[:4]) + " approach") if tags else "optimal approach"

    # Placeholder reasoning
    reasoning = (
        f"We use a {tag_hint}.\n\n"
        "**Time Complexity:** $O(N \\log N)$ (adjust per problem).\n"
        "**Space Complexity:** $O(N)$ auxiliary.\n\n"
        "Key observations:\n"
        "- Read constraints carefully to determine feasible complexity.\n"
        "- Use appropriate data structures (e.g., segment tree, BFS, DP table).\n"
        "- Handle edge cases: empty input, single element, overflow with `long long`."
    )

    assistant = (
        f"### Strategy & Complexity Analysis\n\n"
        f"{reasoning}\n\n"
        f"### C++ Solution\n\n"
        f"```cpp\n"
        f"// C++20 | Competitive Programming Template\n"
        f"#include <bits/stdc++.h>\n"
        f"using namespace std;\n\n"
        f"{solution}\n"
        f"```"
    )
    return assistant


def extract_solutions(problem: dict) -> list[str]:
    """
    Pull accepted C++ solutions from either APPS or CodeContests schema.
    """
    # APPS: 'solutions' is a JSON-encoded list of strings
    raw = problem.get("solutions", "[]")
    if isinstance(raw, str):
        try:
            solutions = json.loads(raw)
        except json.JSONDecodeError:
            solutions = []
    elif isinstance(raw, list):
        solutions = raw
    else:
        solutions = []

    # CodeContests: solutions live under 'solutions' → {'language': [...], 'solution': [...]}
    if not solutions:
        sol_block = problem.get("solutions", {})
        if isinstance(sol_block, dict):
            langs = sol_block.get("language", [])
            codes = sol_block.get("solution", [])
            
            # THE FIX: In deepmind/code_contests, language is an integer where 2 == CPP
            solutions = [
                c for l, c in zip(langs, codes) 
                if (l == 2 or str(l).upper() in ("2", "CPP", "C++"))
            ]

    return [s for s in solutions if s and len(s) > 30]


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def format_example(problem: dict) -> dict | None:
    """Return a ChatML dict or None if the problem should be skipped."""
    solutions = extract_solutions(problem)
    if not solutions:
        return None  # skip problems with no known C++ solution

    solution = solutions[0]  # use the first accepted solution

    user_msg      = build_user_message(problem)
    assistant_msg = build_assistant_message(solution, problem)

    if len(user_msg) < 20:  # degenerate / corrupted entry
        return None

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def process_dataset(
    dataset_name: str,
    split: str,
    output_path: str,
    max_examples: int | None = None,
    difficulty_filter: list[str] | None = None,
) -> None:
    """
    Load a HuggingFace dataset, convert to ChatML .jsonl, and write to disk.
    """
    print(f"[1/3] Loading dataset '{dataset_name}' split='{split}' …")
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    print(f"[2/3] Converting {len(ds):,} problems → ChatML …")
    with output_path.open("w", encoding="utf-8") as fout:
        for problem in tqdm(ds, desc="Converting"):
            # Optional difficulty filter
            if difficulty_filter:
                diff = str(problem.get("difficulty", "")).lower()
                if not any(d in diff for d in difficulty_filter):
                    skipped += 1
                    continue

            record = format_example(problem)
            if record is None:
                skipped += 1
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if max_examples and written >= max_examples:
                break

    print(
        f"[3/3] Done. Written: {written:,} records | Skipped: {skipped:,} | "
        f"File: {output_path}"
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CP datasets to ChatML .jsonl for fine-tuning."
    )
    parser.add_argument(
        "--dataset",
        default="deepmind/code_contests",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--output",
        default="/content/drive/MyDrive/cp_coder/data/cp_chatml.jsonl",
        help="Output .jsonl file path",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=20000, 
        help="Maximum number of training examples to write",
    )
    parser.add_argument(
        "--difficulty",
        nargs="*",
        default=None,
        help="Filter by difficulty strings",
    )
    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset,
        split=args.split,
        output_path=args.output,
        max_examples=args.max_examples,
        difficulty_filter=args.difficulty,
    )