"""Train / validation / test splits for JSONL training data."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Union

PathLike = Union[str, Path]


def create_splits(
    input_file: PathLike,
    output_dir: PathLike,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    input_path = Path(input_file)
    out = Path(output_dir)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    lines: List[str] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    rng = random.Random(seed)
    rng.shuffle(lines)

    n = len(lines)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_lines = lines[:n_train]
    val_lines = lines[n_train : n_train + n_val]
    test_lines = lines[n_train + n_val :]

    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "validation").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)

    def _write(sub: str, rows: List[str]) -> None:
        p = out / sub / "data.jsonl"
        with open(p, "w", encoding="utf-8") as wf:
            for row in rows:
                wf.write(row + "\n")

    _write("train", train_lines)
    _write("validation", val_lines)
    _write("test", test_lines)

    stats = {
        "train_size": len(train_lines),
        "val_size": len(val_lines),
        "test_size": len(test_lines),
        "total": n,
        "seed": seed,
    }
    with open(out / "split_stats.json", "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
