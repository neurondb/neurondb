"""Training entrypoint (requires ml extras)."""

from __future__ import annotations


def run_train(**kwargs: object) -> None:
    raise NotImplementedError(
        "Training requires neurondbpy[ml] (torch, deepspeed, etc.). "
        "See docs for SQL LLM fine-tuning."
    )
