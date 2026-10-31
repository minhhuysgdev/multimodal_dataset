#!/usr/bin/env python3
"""Shared utilities for Phase 6 review-routing evaluation."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path | str) -> Any:
    """Load JSON file."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path | str, data: Any):
    """Save JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_results(path: Path | str) -> list[dict[str, Any]]:
    """Load results list from JSON."""
    doc = load_json(path)
    if isinstance(doc, dict) and "results" in doc:
        return list(doc["results"])
    if isinstance(doc, list):
        return doc
    raise ValueError(f"Unsupported JSON in {path}")


def select_for_review_random(
    predictions: list[dict[str, Any]],
    budget: float,
    seed: int = 42
) -> set[str]:
    """B1: Random selection."""
    rng = random.Random(seed)
    sample_ids = [p["sample_id"] for p in predictions]
    k = int(len(sample_ids) * budget)
    return set(rng.sample(sample_ids, k))


def select_for_review_confidence(
    predictions: list[dict[str, Any]],
    budget: float,
    threshold: float = None
) -> set[str]:
    """B2: Low confidence → review."""
    # Sort by confidence ascending
    sorted_preds = sorted(predictions, key=lambda x: x.get("confidence", 1.0))
    
    if threshold is not None:
        # Fixed threshold
        return {p["sample_id"] for p in sorted_preds if p.get("confidence", 1.0) < threshold}
    else:
        # Budget-based
        k = int(len(predictions) * budget)
        return {p["sample_id"] for p in sorted_preds[:k]}


def select_for_review_entropy(
    predictions_multi: list[list[dict[str, Any]]],  # K runs per sample
    budget: float
) -> set[str]:
    """B3: High entropy from K runs → review."""
    # Compute entropy for each sample
    sample_entropies = []
    
    # Group by sample_id
    by_sample = {}
    for run in predictions_multi:
        for pred in run:
            sid = pred["sample_id"]
            if sid not in by_sample:
                by_sample[sid] = []
            by_sample[sid].append(pred)
    
    for sid, preds in by_sample.items():
        # Extract L3 predictions
        l3_votes = [p.get("predicted_l3", "") for p in preds]
        
        # Compute entropy
        counts = Counter(l3_votes)
        total = len(l3_votes)
        probs = [count / total for count in counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        
        sample_entropies.append((sid, entropy))
    
    # Sort by entropy descending
    sample_entropies.sort(key=lambda x: x[1], reverse=True)
    
    # Select top budget%
    k = int(len(sample_entropies) * budget)
    return {sid for sid, _ in sample_entropies[:k]}


def apply_oracle_review(
    predictions: list[dict[str, Any]],
    review_ids: set[str],
    oracle_labels: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    """Apply oracle (ground truth) labels to reviewed samples."""
    corrected = []
    
    for pred in predictions:
        sid = pred["sample_id"]
        
        if sid in review_ids and sid in oracle_labels:
            # Replace with oracle
            oracle = oracle_labels[sid]
            pred = pred.copy()
            pred["predicted_l1"] = oracle["l1"]
            pred["predicted_l2"] = oracle["l2"]
            pred["predicted_l3"] = oracle["l3"]
            pred["was_reviewed"] = True
        else:
            pred = pred.copy()
            pred["was_reviewed"] = False
        
        corrected.append(pred)
    
    return corrected


def compute_quality_metrics(
    predictions: list[dict[str, Any]],
    gold_labels: dict[str, dict[str, str]]
) -> dict[str, float]:
    """Compute accuracy and F1."""
    from sklearn.metrics import accuracy_score, f1_score
    
    matched = []
    for pred in predictions:
        sid = pred["sample_id"]
        if sid in gold_labels:
            gold = gold_labels[sid]
            matched.append({
                "pred_l1": pred.get("predicted_l1", ""),
                "pred_l2": pred.get("predicted_l2", ""),
                "pred_l3": pred.get("predicted_l3", ""),
                "gold_l1": gold["l1"],
                "gold_l2": gold["l2"],
                "gold_l3": gold["l3"],
            })
    
    if not matched:
        return {"error": "No matched samples"}
    
    pred_l1 = [m["pred_l1"] for m in matched]
    pred_l2 = [m["pred_l2"] for m in matched]
    pred_l3 = [m["pred_l3"] for m in matched]
    gold_l1 = [m["gold_l1"] for m in matched]
    gold_l2 = [m["gold_l2"] for m in matched]
    gold_l3 = [m["gold_l3"] for m in matched]
    
    full_path_correct = sum(
        1 for m in matched
        if m["pred_l1"] == m["gold_l1"]
        and m["pred_l2"] == m["gold_l2"]
        and m["pred_l3"] == m["gold_l3"]
    )
    
    return {
        "num_samples": len(matched),
        "accuracy_l1": accuracy_score(gold_l1, pred_l1),
        "accuracy_l2": accuracy_score(gold_l2, pred_l2),
        "accuracy_l3": accuracy_score(gold_l3, pred_l3),
        "macro_f1_l1": f1_score(gold_l1, pred_l1, average="macro", zero_division=0),
        "macro_f1_l2": f1_score(gold_l2, pred_l2, average="macro", zero_division=0),
        "macro_f1_l3": f1_score(gold_l3, pred_l3, average="macro", zero_division=0),
        "full_path_accuracy": full_path_correct / len(matched),
    }
