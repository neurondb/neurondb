"""
Evaluate a trained Olympus-MAC (or compatible) model on Needle-in-a-Haystack tasks.
Loads generated tasks from JSON, runs forward pass, and checks if the model output
contains the correct passkey (exact match or next-token prediction).
Usage: python experiments/niah/run_niah.py --model_path runs/lm_small/last.pt --tasks experiments/niah/tasks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Project root
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from olympus import OlympusMAC
from transformers import AutoTokenizer


def load_model(ckpt_path: str, device: torch.device, config: dict | None = None) -> OlympusMAC:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    if config is None:
        config = {
            "vocab_size": 50257,
            "dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "max_len": 2048,
            "num_persistent": 4,
            "chunk_size": 32,
        }
    model = OlympusMAC(**config)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_sample(
    model: OlympusMAC,
    tokenizer,
    context_ids: list[int],
    passkey: str,
    device: torch.device,
    max_len: int = 512,
) -> bool:
    """Run model on context and check if decoded output contains passkey."""
    if len(context_ids) > max_len:
        context_ids = context_ids[-max_len:]
    x = torch.tensor([context_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, _ = model(x, labels=None, return_memory_loss=False)
    # Last position prediction
    next_logits = logits[0, -1]
    next_ids = next_logits.argmax(dim=-1).unsqueeze(0)
    decoded = tokenizer.decode(next_ids[0].tolist(), skip_special_tokens=True)
    return passkey.strip().upper() in decoded.upper() or decoded.strip().upper() in passkey.upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="experiments/niah/tasks.json")
    parser.add_argument("--context_lengths", type=int, nargs="+", default=None, help="Filter to these lengths only")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = load_model(args.model_path, device)

    with open(args.tasks) as f:
        tasks = json.load(f)
    if args.context_lengths:
        tasks = [t for t in tasks if t["context_length"] in args.context_lengths]
    if args.max_samples:
        tasks = tasks[: args.max_samples]

    correct = 0
    by_length = {}
    for t in tasks:
        ok = evaluate_sample(
            model, tokenizer,
            t["context_ids"], t["passkey"],
            device,
        )
        if ok:
            correct += 1
        L = t["context_length"]
        by_length.setdefault(L, {"correct": 0, "total": 0})
        by_length[L]["total"] += 1
        if ok:
            by_length[L]["correct"] += 1

    acc = correct / len(tasks) if tasks else 0
    print(f"Overall NIAH accuracy: {correct}/{len(tasks)} = {acc:.2%}")
    for L in sorted(by_length.keys()):
        c, n = by_length[L]["correct"], by_length[L]["total"]
        print(f"  Context {L}: {c}/{n} = {c/n:.2%}")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": len(tasks), "by_length": by_length}, f, indent=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
