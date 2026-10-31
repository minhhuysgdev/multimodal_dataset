#!/usr/bin/env python3
"""Shared utilities for Phase 5 pipeline evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path | str) -> Any:
    """Load JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path | str, data: Any):
    """Save JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_results(path: Path | str) -> list[dict[str, Any]]:
    """Load results from JSON (handles both list and {"results": [...]} formats)."""
    doc = load_json(path)
    if isinstance(doc, dict) and "results" in doc:
        return list(doc["results"])
    if isinstance(doc, list):
        return doc
    raise ValueError(f"Unsupported JSON shape in {path}")


@dataclass
class IntentPrediction:
    """Unified prediction schema."""
    sample_id: str
    sentence: str
    predicted_l1: str
    predicted_l2: str
    predicted_l3: str
    confidence: float = 0.0
    is_valid_taxonomy: bool = True
    api_calls: int = 1
    api_cost_usd: float = 0.0
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sentence": self.sentence,
            "predicted_l1": self.predicted_l1,
            "predicted_l2": self.predicted_l2,
            "predicted_l3": self.predicted_l3,
            "confidence": self.confidence,
            "is_valid_taxonomy": self.is_valid_taxonomy,
            "api_calls": self.api_calls,
            "api_cost_usd": self.api_cost_usd,
            "latency_seconds": self.latency_seconds,
            "metadata": self.metadata or {},
        }


class TaxonomyValidator:
    """Validate intent predictions against taxonomy."""
    
    def __init__(self, taxonomy_path: Path | str, edges_path: Path | str):
        self.taxonomy = pd.read_csv(taxonomy_path)
        self.edges = pd.read_csv(edges_path)
        self._build_index()
    
    def _build_index(self):
        """Build lookup structures."""
        # Valid L1 → L2 → L3 paths
        self.valid_paths = set()

        col_map = {
            "l1": ["level_1", "L1 Category", "l1"],
            "l2": ["level_2", "L2 Intent", "l2"],
            "l3": ["level_3", "L3 Specific Intent", "l3"],
        }

        def _pick(row, keys):
            for key in keys:
                val = row.get(key)
                if val is not None and str(val).strip():
                    return str(val).strip()
            return ""

        for _, row in self.taxonomy.iterrows():
            l1 = _pick(row, col_map["l1"])
            l2 = _pick(row, col_map["l2"])
            l3 = _pick(row, col_map["l3"])

            if l1 and l2 and l3:
                self.valid_paths.add((l1, l2, l3))
    
    def is_valid(self, l1: str, l2: str, l3: str) -> bool:
        """Check if (L1, L2, L3) is a valid path."""
        return (l1, l2, l3) in self.valid_paths
    
    def get_valid_l2_for_l1(self, l1: str) -> set[str]:
        """Get all valid L2 values for given L1."""
        return {l2 for l1_, l2, l3 in self.valid_paths if l1_ == l1}
    
    def get_valid_l3_for_l2(self, l2: str) -> set[str]:
        """Get all valid L3 values for given L2."""
        return {l3 for l1, l2_, l3 in self.valid_paths if l2_ == l2}


def compute_hierarchical_metrics(
    predictions: list[IntentPrediction],
    gold_data: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute hierarchical accuracy and F1 scores."""
    from sklearn.metrics import accuracy_score, f1_score
    
    # Build gold dict
    gold_dict = {}
    for row in gold_data:
        sid = row.get("sample_id")
        gold_dict[sid] = {
            "l1": str(row.get("gold_level_1") or ""),
            "l2": str(row.get("gold_level_2") or ""),
            "l3": str(row.get("gold_level_3") or ""),
        }
    
    # Match predictions to gold
    matched = []
    for pred in predictions:
        if pred.sample_id in gold_dict:
            gold = gold_dict[pred.sample_id]
            matched.append({
                "pred_l1": pred.predicted_l1,
                "pred_l2": pred.predicted_l2,
                "pred_l3": pred.predicted_l3,
                "gold_l1": gold["l1"],
                "gold_l2": gold["l2"],
                "gold_l3": gold["l3"],
            })
    
    if not matched:
        return {"error": "No matched predictions"}
    
    # Extract arrays
    pred_l1 = [m["pred_l1"] for m in matched]
    pred_l2 = [m["pred_l2"] for m in matched]
    pred_l3 = [m["pred_l3"] for m in matched]
    gold_l1 = [m["gold_l1"] for m in matched]
    gold_l2 = [m["gold_l2"] for m in matched]
    gold_l3 = [m["gold_l3"] for m in matched]
    
    # Full-path accuracy
    full_path_correct = sum(
        1 for m in matched
        if m["pred_l1"] == m["gold_l1"]
        and m["pred_l2"] == m["gold_l2"]
        and m["pred_l3"] == m["gold_l3"]
    )
    
    metrics = {
        "num_samples": len(matched),
        "accuracy_l1": accuracy_score(gold_l1, pred_l1),
        "accuracy_l2": accuracy_score(gold_l2, pred_l2),
        "accuracy_l3": accuracy_score(gold_l3, pred_l3),
        "macro_f1_l1": f1_score(gold_l1, pred_l1, average="macro", zero_division=0),
        "macro_f1_l2": f1_score(gold_l2, pred_l2, average="macro", zero_division=0),
        "macro_f1_l3": f1_score(gold_l3, pred_l3, average="macro", zero_division=0),
        "weighted_f1_l1": f1_score(gold_l1, pred_l1, average="weighted", zero_division=0),
        "weighted_f1_l2": f1_score(gold_l2, pred_l2, average="weighted", zero_division=0),
        "weighted_f1_l3": f1_score(gold_l3, pred_l3, average="weighted", zero_division=0),
        "full_path_accuracy": full_path_correct / len(matched),
    }
    
    return metrics


def estimate_api_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """Estimate API cost in USD."""
    # Pricing as of 2026 (update as needed)
    prices = {
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    }
    
    if model not in prices:
        return 0.0
    
    return (input_tokens * prices[model]["input"] +
            output_tokens * prices[model]["output"])


def create_prompt_hash(prompt: str) -> str:
    """Create hash of prompt for tracking versions."""
    return hashlib.md5(prompt.encode()).hexdigest()[:8]
