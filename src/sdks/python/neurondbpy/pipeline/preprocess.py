"""Preprocess and merge JSONL datasets."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]


def merge_datasets(sources: Iterable[PathLike], output: PathLike) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for src in sources:
            text = Path(src).read_text(encoding="utf-8")
            if text and not text.endswith("\n"):
                text += "\n"
            out.write(text)


def preprocess(input_dir: Path, output_dir: Path) -> None:
    """Copy or unify JSONL files from raw input into processed output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        return
    for p in sorted(input_dir.glob("*.jsonl")):
        dest = output_dir / p.name
        shutil.copy2(p, dest)
