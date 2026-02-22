"""Data loaders for language modeling: WikiText-103, LAMBADA."""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


def get_tokenizer(model_name: str = "gpt2"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


class LMChunkDataset(Dataset):
    """Causal LM dataset: chunks of consecutive tokens."""

    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
    ):
        self.block_size = block_size
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.length = (len(self.data) - 1) // block_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self.data[start:end]
        x = chunk[:-1]
        y = chunk[1:].clone()
        y[chunk[:-1] == 0] = -100  # padding
        return x, y


def load_wikitext103(
    tokenizer,
    max_length: int = 256,
    split: str = "train",
) -> list[int]:
    """Load WikiText-103 and tokenize; return flat list of token ids."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, trust_remote_code=True)
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].squeeze(0).tolist()
    if max_length > 0:
        ids = ids[: max_length * (len(ids) // max_length)]
    return ids


def load_lambada(
    tokenizer,
    split: str = "test",
) -> list[tuple[list[int], list[int]]]:
    """Load LAMBADA; return list of (context_ids, target_ids) for last-word prediction."""
    ds = load_dataset("lambada", split=split, trust_remote_code=True)
    out = []
    for ex in ds:
        text = ex["text"]
        words = text.split()
        if len(words) < 2:
            continue
        context = " ".join(words[:-1])
        target = words[-1]
        ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        tgt_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if ctx_ids and tgt_ids:
            out.append((ctx_ids, tgt_ids))
    return out


def get_lm_dataloader(
    token_ids: list[int],
    block_size: int,
    batch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    """Build DataLoader for causal LM from token list."""
    dataset = LMChunkDataset(token_ids, block_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
