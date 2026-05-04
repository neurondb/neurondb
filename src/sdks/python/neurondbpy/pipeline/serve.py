"""Model server launcher (requires ml / vLLM stack)."""

from __future__ import annotations

import subprocess
from typing import Optional


def run_serve(
    *,
    model_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> "subprocess.Popen":
    raise NotImplementedError(
        "Serving requires neurondbpy[ml] and a local vLLM or compatible server. "
        f"Would bind {host}:{port} model_path={model_path!r}."
    )
