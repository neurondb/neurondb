"""Evaluate generated SQL vs ground truth (optional DB execution)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping


class SQLEvaluator:
    def __init__(self, db_config: Mapping[str, Any]) -> None:
        self._db_config = dict(db_config)

    def evaluate_dataset(self, predictions_path: str, ground_truth_path: str) -> Dict[str, Any]:
        preds: List[dict[str, Any]] = []
        truth: List[dict[str, Any]] = []
        with open(predictions_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    preds.append(json.loads(line))
        with open(ground_truth_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    truth.append(json.loads(line))
        exact = 0
        for p, t in zip(preds, truth):
            pg = (p.get("generated_sql") or p.get("sql") or "").strip()
            tg = (t.get("sql") or "").strip()
            if pg and pg == tg:
                exact += 1
        n = min(len(preds), len(truth))
        return {
            "num_pairs": n,
            "exact_match": exact,
            "exact_match_rate": exact / n if n else 0.0,
        }

    def close(self) -> None:
        pass
