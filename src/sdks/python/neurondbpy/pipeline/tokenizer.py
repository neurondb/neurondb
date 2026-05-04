"""Tokenizer helpers for SQL LM training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class SQLTokenizer:
    """Placeholder tokenizer wrapper for training pipelines."""

    def __init__(self, model_name: str = "gpt2") -> None:
        self.model_name = model_name


def prepare_training_data(
    input_path: Path,
    output_path: Path,
    tokenizer: Optional[SQLTokenizer] = None,
    **kwargs: Any,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.exists():
        output_path.write_bytes(input_path.read_bytes())
    else:
        output_path.write_text("", encoding="utf-8")
