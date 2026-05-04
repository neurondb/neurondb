"""Download public SQL training datasets (optional / heavy)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def download_datasets(*, output_dir: Path, datasets: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "download_datasets: skipped (install neurondbpy[ml] and implement fetchers). "
        "Requested: %s -> %s",
        datasets,
        output_dir,
    )
