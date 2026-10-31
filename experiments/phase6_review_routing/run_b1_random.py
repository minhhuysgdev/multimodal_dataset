#!/usr/bin/env python3
"""B1: Random allocation baseline.

Usage:
  python experiments/phase6_review_routing/run_b1_random.py --budget 0.2
  python experiments/phase6_review_routing/run_b1_random.py --budget 0.4
  python experiments/phase6_review_routing/run_b1_random.py --budget 0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase6_review_routing"))
sys.path.insert(0, str(ROOT / "experiments" / "phase5_pipeline"))

from pipeline_utils import load_results as load_results_pipeline  # noqa: E402
from routing_utils import (  # noqa: E402
    apply_oracle_review,
    compute_quality_metrics,
    save_json,
    select_for_review_random,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, required=True, help="Review budget (0.2, 0.4, 0.6)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print(f"=== B1: Random Allocation (budget={args.budget}) ===\n")
    
    # Load annotation pool (train split, no D3)
    train_data = load_results_pipeline(ROOT / "data/splits/train.json")
    print(f"Annotation pool: {len(train_data)} samples")
    
    # Load D2 verified as oracle labels
    d2_data = load_results_pipeline(ROOT / "data/datasets/d2_verified_train.json")
    oracle_labels = {}
    for row in d2_data:
        sid = row.get("sample_id")
        intent = row.get("intent_3_level") or {}
        oracle_labels[sid] = {
            "l1": str(intent.get("level_1", "")),
            "l2": str(intent.get("level_2", "")),
            "l3": str(intent.get("level_3", "")),
        }
    
    print(f"Oracle labels: {len(oracle_labels)} samples")
    
    # Simulate M1 predictions (for demo, use oracle with some noise)
    # In real scenario, run M1 annotator first
    import random
    rng = random.Random(args.seed)
    
    predictions = []
    for row in train_data:
        sid = row.get("sample_id")
        
        # Simulate prediction (90% match oracle, 10% random error)
        if sid in oracle_labels and rng.random() > 0.1:
            pred_labels = oracle_labels[sid]
        else:
            pred_labels = {"l1": "truoc_mua_hang", "l2": "tim_kiem_san_pham", "l3": "hoi_gia"}
        
        predictions.append({
            "sample_id": sid,
            "sentence": row.get("sentence", ""),
            "predicted_l1": pred_labels["l1"],
            "predicted_l2": pred_labels["l2"],
            "predicted_l3": pred_labels["l3"],
            "confidence": rng.uniform(0.7, 0.95),
        })
    
    # Select for review (random)
    review_ids = select_for_review_random(predictions, args.budget, seed=args.seed)
    print(f"Selected for review: {len(review_ids)} samples ({len(review_ids)/len(predictions):.1%})")
    
    # Apply oracle review
    corrected_predictions = apply_oracle_review(predictions, review_ids, oracle_labels)
    
    # Compute quality metrics
    metrics = compute_quality_metrics(corrected_predictions, oracle_labels)
    
    result = {
        "method": "B1_random",
        "budget": args.budget,
        "seed": args.seed,
        "num_samples": len(predictions),
        "num_reviewed": len(review_ids),
        "review_rate": len(review_ids) / len(predictions),
        "quality_metrics": metrics,
        "predictions": corrected_predictions,
    }
    
    # Save
    outdir = ROOT / "experiments" / "phase6_review_routing" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / f"B1_random_budget{int(args.budget*100)}.json", result)
    
    print(f"\n=== B1 Results ===")
    print(f"Macro-F1 L3: {metrics['macro_f1_l3']:.4f}")
    print(f"Full-path Acc: {metrics['full_path_accuracy']:.4f}")
    print(f"Review Rate: {len(review_ids) / len(predictions):.2%}")
    print(f"\nSaved: {outdir / f'B1_random_budget{int(args.budget*100)}.json'}")


if __name__ == "__main__":
    main()
