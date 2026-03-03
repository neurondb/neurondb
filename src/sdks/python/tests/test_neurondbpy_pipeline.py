"""Tests for neurondbpy.pipeline (split, preprocess helpers)."""

import json
from pathlib import Path

import pytest

from neurondbpy.pipeline.split import create_splits
from neurondbpy.pipeline.preprocess import merge_datasets


def test_create_splits(tmp_path: Path) -> None:
    input_file = tmp_path / "all.jsonl"
    with open(input_file, "w") as f:
        for i in range(100):
            f.write(json.dumps({"question": f"q{i}", "sql": f"SELECT {i}"}) + "\n")
    output_dir = tmp_path / "splits"
    create_splits(
        input_file=str(input_file),
        output_dir=str(output_dir),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )
    assert (output_dir / "train" / "data.jsonl").exists()
    assert (output_dir / "validation" / "data.jsonl").exists()
    assert (output_dir / "test" / "data.jsonl").exists()
    assert (output_dir / "split_stats.json").exists()
    with open(output_dir / "split_stats.json") as f:
        stats = json.load(f)
    assert stats["train_size"] == 80
    assert stats["val_size"] == 10
    assert stats["test_size"] == 10


def test_merge_datasets(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"x": 1}\n')
    b.write_text('{"x": 2}\n')
    out = tmp_path / "merged.jsonl"
    merge_datasets([a, b], out)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["x"] in (1, 2)
    assert json.loads(lines[1])["x"] in (1, 2)
