#!/usr/bin/env python3
"""M3: Full pipeline (R3 + GPT-4o-mini + GPT-4o verifier + guardrails).

Reuses existing labeling_pipeline.py logic.

Usage:
  export OPENAI_API_KEY="your-key"
  python experiments/phase5_pipeline/run_m3_full.py --limit 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data/code"))
sys.path.insert(0, str(ROOT / "experiments" / "phase5_pipeline"))

from pipeline_utils import (  # noqa: E402
    IntentPrediction,
    TaxonomyValidator,
    compute_hierarchical_metrics,
    load_results,
    save_json,
)

# Try importing labeling pipeline (if available)
try:
    from labeling_pipeline import label_single_utterance  # noqa: E402
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    if not HAS_PIPELINE:
        print("ERROR: labeling_pipeline.py not found or cannot be imported")
        print("This baseline requires the full pipeline code at data/code/labeling_pipeline.py")
        sys.exit(1)
    
    print("=== M3: Full Pipeline Baseline ===\n")
    
    d3_gold = load_results(ROOT / "data/gold/d3_human_gold.json")
    if args.limit > 0:
        d3_gold = d3_gold[:args.limit]
    
    print(f"Test samples: {len(d3_gold)}")
    
    # Configure pipeline
    os.environ["ENABLE_VERIFIER"] = "1"
    os.environ["VERIFIER_MODEL"] = "gpt-4o"
    os.environ["VERIFY_CONF_THRESHOLD"] = "0.85"
    
    taxonomy_validator = TaxonomyValidator(
        ROOT / "data/unified_intents.csv",
        ROOT / "data/intent_kb.intent_edges_28_05_2026.csv"
    )
    
    predictions = []
    invalid_count = 0
    needs_review_count = 0
    
    for i, row in enumerate(d3_gold):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(d3_gold)}")
        
        sentence = row.get("sentence", "")
        sample_id = row.get("sample_id", "")
        
        try:
            # Run full pipeline
            result = label_single_utterance(sentence)
            
            is_valid = taxonomy_validator.is_valid(
                result.get("level_1", ""),
                result.get("level_2", ""),
                result.get("level_3", "")
            )
            
            if not is_valid:
                invalid_count += 1
            
            if result.get("qa_status") == "needs_review":
                needs_review_count += 1
            
            pred = IntentPrediction(
                sample_id=sample_id,
                sentence=sentence,
                predicted_l1=result.get("level_1", ""),
                predicted_l2=result.get("level_2", ""),
                predicted_l3=result.get("level_3", ""),
                confidence=result.get("confidence", 0.0),
                is_valid_taxonomy=is_valid,
                api_calls=result.get("api_calls", 2),
                api_cost_usd=result.get("api_cost_usd", 0.0),
                latency_seconds=result.get("latency_seconds", 0.0),
                metadata={
                    "qa_status": result.get("qa_status"),
                    "verifier_action": result.get("verifier_action"),
                    "retrieval_method": "hybrid",
                }
            )
            predictions.append(pred)
        
        except Exception as e:
            print(f"  Error on sample {sample_id}: {e}")
            pred = IntentPrediction(
                sample_id=sample_id, sentence=sentence,
                predicted_l1="", predicted_l2="", predicted_l3="",
                confidence=0.0, is_valid_taxonomy=False,
                api_calls=0, api_cost_usd=0.0, latency_seconds=0.0,
                metadata={"error": str(e)}
            )
            predictions.append(pred)
    
    metrics = compute_hierarchical_metrics(predictions, d3_gold)
    total_cost = sum(p.api_cost_usd for p in predictions)
    
    result_doc = {
        "method": "M3_full_pipeline",
        "config": {
            "retrieval": "hybrid_r3",
            "annotator": "gpt-4o-mini",
            "verifier": "gpt-4o",
            "guardrails": "enabled",
        },
        "num_samples": len(predictions),
        "quality_metrics": metrics,
        "pipeline_metrics": {
            "total_cost_usd": total_cost,
            "avg_cost_per_sample": total_cost / len(predictions),
            "invalid_rate": invalid_count / len(predictions),
            "review_rate": needs_review_count / len(predictions),
            "approval_rate": 1.0 - (needs_review_count / len(predictions)),
        },
        "predictions": [p.to_dict() for p in predictions],
    }
    
    outdir = ROOT / "experiments" / "phase5_pipeline" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "M3_full_pipeline_results.json", result_doc)
    
    print(f"\n=== M3 Results ===")
    print(f"Macro-F1 L3: {metrics['macro_f1_l3']:.4f}")
    print(f"Full-path Acc: {metrics['full_path_accuracy']:.4f}")
    print(f"Review Rate: {needs_review_count / len(predictions):.2%}")
    print(f"Total Cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
