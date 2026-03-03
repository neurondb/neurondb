"""
Generate Needle-in-a-Haystack (S-NIAH) tasks: insert a passkey at a given depth
in a long context and ask the model to retrieve it.
Output: list of dicts with "context_ids", "passkey", "position", "context_length".
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


PASSKEY_TEMPLATE = "The passkey is {passkey}. Remember it."
QUESTION = "\nWhat is the passkey? Answer:"


def generate_passkey() -> str:
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8))


def generate_filler_tokens(tokenizer, length: int, seed: int) -> list[int]:
    """Generate filler content (repeated sentences) to reach target length."""
    filler = "This is filler text for the needle in a haystack evaluation. "
    enc = tokenizer(filler * (length // len(filler) + 1), add_special_tokens=False)
    ids = enc["input_ids"][:length]
    return ids


def build_niah_sample(
    tokenizer,
    context_length: int,
    passkey_position: int,  # token index where passkey block starts
    passkey: str | None = None,
    seed: int = 42,
) -> dict:
    """Build one NIAH sample: context of context_length tokens with passkey at passkey_position."""
    rng = random.Random(seed)
    passkey = passkey or generate_passkey()
    block = PASSKEY_TEMPLATE.format(passkey=passkey)
    block_ids = tokenizer(block, add_special_tokens=False)["input_ids"]
    question_ids = tokenizer(QUESTION, add_special_tokens=False)["input_ids"]
    filler_before_len = passkey_position
    filler_after_len = context_length - passkey_position - len(block_ids) - len(question_ids)
    if filler_after_len < 0:
        filler_after_len = 0
    filler_before = generate_filler_tokens(tokenizer, filler_before_len, seed)
    filler_after = generate_filler_tokens(tokenizer, filler_after_len, seed + 1)
    context_ids = filler_before + block_ids + filler_after + question_ids
    return {
        "context_ids": context_ids,
        "passkey": passkey,
        "position": passkey_position,
        "context_length": context_length,
        "answer_ids": tokenizer(passkey, add_special_tokens=False)["input_ids"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    parser.add_argument("--num_samples_per_length", type=int, default=20)
    parser.add_argument("--depths", type=str, default="0.25,0.5,0.75", help="Comma-separated fractions (0-1) for passkey depth")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--output", type=str, default="experiments/niah/tasks.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    depth_fracs = [float(x) for x in args.depths.split(",")]
    tasks = []
    for ctx_len in args.context_lengths:
        for _ in range(args.num_samples_per_length):
            depth = random.choice(depth_fracs)
            pos = int(ctx_len * depth)
            pos = max(0, min(ctx_len - 50, pos))
            sample = build_niah_sample(tokenizer, ctx_len, pos, seed=args.seed + len(tasks))
            tasks.append(sample)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(tasks, f, indent=0)
    print(f"Wrote {len(tasks)} NIAH tasks to {out}")


if __name__ == "__main__":
    main()
