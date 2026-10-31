#!/usr/bin/env python3
"""M2-SR: Self-Refine baseline (GPT-4o-mini annotate → critique → revise).

Usage:
  export OPENAI_API_KEY="your-key"
  python experiments/phase5_pipeline/run_m2_sr_selfrefine.py --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase5_pipeline"))

from pipeline_utils import (  # noqa: E402
    IntentPrediction,
    TaxonomyValidator,
    compute_hierarchical_metrics,
    estimate_api_cost,
    load_results,
    save_json,
)


def self_refine_annotate(
    sentence: str,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Self-refine: annotate → critique → revise."""
    import openai
    
    # Step 1: Initial annotation
    initial_prompt = f"""Gán nhãn intent phân cấp 3 tầng cho câu sau:

Câu: "{sentence}"

Output JSON:
{{
  "level_1": "<L1>",
  "level_2": "<L2>",
  "level_3": "<L3>",
  "confidence": <0.0-1.0>
}}"""
    
    start_time = time.time()
    total_cost = 0.0
    
    try:
        # Initial annotation
        response1 = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": initial_prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        
        initial_result = json.loads(response1.choices[0].message.content.strip())
        usage1 = response1.get("usage", {})
        total_cost += estimate_api_cost(model, usage1.get("prompt_tokens", 0), usage1.get("completion_tokens", 0))
        
        # Step 2: Critique
        critique_prompt = f"""Câu gốc: "{sentence}"

Nhãn đã gán:
- L1: {initial_result.get("level_1")}
- L2: {initial_result.get("level_2")}
- L3: {initial_result.get("level_3")}

Phân tích: Nhãn này có đúng không? Có cần chỉnh sửa không?

Output JSON:
{{
  "needs_revision": <true/false>,
  "reasoning": "<lý do ngắn gọn>"
}}"""
        
        response2 = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": critique_prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        
        critique_result = json.loads(response2.choices[0].message.content.strip())
        usage2 = response2.get("usage", {})
        total_cost += estimate_api_cost(model, usage2.get("prompt_tokens", 0), usage2.get("completion_tokens", 0))
        
        # Step 3: Revise if needed
        if critique_result.get("needs_revision"):
            revise_prompt = f"""Câu: "{sentence}"

Nhãn hiện tại:
- L1: {initial_result.get("level_1")}
- L2: {initial_result.get("level_2")}
- L3: {initial_result.get("level_3")}

Phản hồi: {critique_result.get("reasoning")}

Vui lòng chỉnh sửa nhãn. Output JSON:
{{
  "level_1": "<L1>",
  "level_2": "<L2>",
  "level_3": "<L3>",
  "confidence": <0.0-1.0>
}}"""
            
            response3 = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": revise_prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            
            final_result = json.loads(response3.choices[0].message.content.strip())
            usage3 = response3.get("usage", {})
            total_cost += estimate_api_cost(model, usage3.get("prompt_tokens", 0), usage3.get("completion_tokens", 0))
            api_calls = 3
        else:
            final_result = initial_result
            api_calls = 2
        
        latency = time.time() - start_time
        
        return {
            "level_1": str(final_result.get("level_1", "")),
            "level_2": str(final_result.get("level_2", "")),
            "level_3": str(final_result.get("level_3", "")),
            "confidence": float(final_result.get("confidence", 0.0)),
            "api_calls": api_calls,
            "api_cost_usd": total_cost,
            "latency_seconds": latency,
            "metadata": {
                "model": model,
                "needs_revision": critique_result.get("needs_revision"),
                "critique_reasoning": critique_result.get("reasoning"),
            }
        }
    
    except Exception as e:
        return {
            "level_1": "", "level_2": "", "level_3": "", "confidence": 0.0,
            "api_calls": 1, "api_cost_usd": total_cost, "latency_seconds": time.time() - start_time,
            "metadata": {"error": str(e)},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    print("=== M2-SR: Self-Refine Baseline ===\n")
    
    d3_gold = load_results(ROOT / "data/gold/d3_human_gold.json")
    if args.limit > 0:
        d3_gold = d3_gold[:args.limit]
    
    taxonomy_validator = TaxonomyValidator(
        ROOT / "data/unified_intents.csv",
        ROOT / "data/intent_kb.intent_edges_28_05_2026.csv"
    )
    
    predictions = []
    invalid_count = 0
    
    for i, row in enumerate(d3_gold):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(d3_gold)}")
        
        sentence = row.get("sentence", "")
        sample_id = row.get("sample_id", "")
        
        result = self_refine_annotate(sentence)
        
        is_valid = taxonomy_validator.is_valid(result["level_1"], result["level_2"], result["level_3"])
        if not is_valid:
            invalid_count += 1
        
        pred = IntentPrediction(
            sample_id=sample_id, sentence=sentence,
            predicted_l1=result["level_1"], predicted_l2=result["level_2"], predicted_l3=result["level_3"],
            confidence=result["confidence"], is_valid_taxonomy=is_valid,
            api_calls=result["api_calls"], api_cost_usd=result["api_cost_usd"],
            latency_seconds=result["latency_seconds"], metadata=result["metadata"],
        )
        predictions.append(pred)
    
    metrics = compute_hierarchical_metrics(predictions, d3_gold)
    total_cost = sum(p.api_cost_usd for p in predictions)
    total_calls = sum(p.api_calls for p in predictions)
    
    result_doc = {
        "method": "M2_SR_self_refine",
        "num_samples": len(predictions),
        "quality_metrics": metrics,
        "pipeline_metrics": {
            "total_cost_usd": total_cost,
            "avg_cost_per_sample": total_cost / len(predictions),
            "total_api_calls": total_calls,
            "avg_calls_per_sample": total_calls / len(predictions),
            "invalid_rate": invalid_count / len(predictions),
        },
        "predictions": [p.to_dict() for p in predictions],
    }
    
    outdir = ROOT / "experiments" / "phase5_pipeline" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "M2_SR_selfrefine_results.json", result_doc)
    
    print(f"\n=== M2-SR Results ===")
    print(f"Macro-F1 L3: {metrics['macro_f1_l3']:.4f}")
    print(f"Avg API calls: {total_calls / len(predictions):.2f}")
    print(f"Total Cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
