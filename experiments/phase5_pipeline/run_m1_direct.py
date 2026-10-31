#!/usr/bin/env python3
"""M1: Direct LLM baseline (GPT-4o-mini, zero-shot, no retrieval/verifier/guardrails).

Usage:
  export OPENAI_API_KEY="your-key"
  python experiments/phase5_pipeline/run_m1_direct.py
  
  # Or test on small subset:
  python experiments/phase5_pipeline/run_m1_direct.py --limit 10
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
    create_prompt_hash,
    estimate_api_cost,
    load_results,
    save_json,
)


def annotate_direct_llm(
    sentence: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Direct LLM annotation without retrieval."""
    import openai
    
    prompt = f"""Bạn là chuyên gia phân loại intent cho e-commerce tiếng Việt.

Gán nhãn phân cấp 3 tầng (L1 → L2 → L3) cho câu sau:

Câu: "{sentence}"

L1: Thời điểm (trước_mua_hang, sau_mua_hang)
L2: Chủ đề chính (vd: tim_kiem_san_pham, dat_hang, van_chuyen, hoan_tra, ...)
L3: Intent cụ thể (vd: hoi_gia, hoi_chat_lieu, hoi_ma_giam_gia, ...)

Output STRICTLY JSON format (không giải thích):
{{
  "level_1": "<L1 intent slug>",
  "level_2": "<L2 intent slug>",
  "level_3": "<L3 intent slug>",
  "confidence": <0.0-1.0>
}}"""
    
    start_time = time.time()
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200,
        )
        
        latency = time.time() - start_time
        
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        
        usage = response.get("usage", {})
        cost = estimate_api_cost(
            model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )
        
        return {
            "level_1": str(result.get("level_1", "")),
            "level_2": str(result.get("level_2", "")),
            "level_3": str(result.get("level_3", "")),
            "confidence": float(result.get("confidence", 0.0)),
            "api_calls": 1,
            "api_cost_usd": cost,
            "latency_seconds": latency,
            "metadata": {
                "model": model,
                "prompt_hash": create_prompt_hash(prompt),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        }
    
    except Exception as e:
        return {
            "level_1": "",
            "level_2": "",
            "level_3": "",
            "confidence": 0.0,
            "api_calls": 1,
            "api_cost_usd": 0.0,
            "latency_seconds": time.time() - start_time,
            "metadata": {"error": str(e)},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples for testing (0=all)")
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    print("=== M1: Direct LLM Baseline ===\n")
    print(f"Model: {args.model}\n")
    
    # Load test set
    d3_gold = load_results(ROOT / "data/gold/d3_human_gold.json")
    if args.limit > 0:
        d3_gold = d3_gold[:args.limit]
    print(f"Test samples: {len(d3_gold)}")
    
    # Load taxonomy validator
    taxonomy_validator = TaxonomyValidator(
        ROOT / "data/unified_intents.csv",
        ROOT / "data/intent_kb.intent_edges_28_05_2026.csv"
    )
    
    # Annotate
    predictions = []
    invalid_count = 0
    
    for i, row in enumerate(d3_gold):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(d3_gold)}")
        
        sentence = row.get("sentence", "")
        sample_id = row.get("sample_id", "")
        
        result = annotate_direct_llm(sentence, model=args.model)
        
        # Validate taxonomy
        is_valid = taxonomy_validator.is_valid(
            result["level_1"],
            result["level_2"],
            result["level_3"]
        )
        
        if not is_valid:
            invalid_count += 1
        
        pred = IntentPrediction(
            sample_id=sample_id,
            sentence=sentence,
            predicted_l1=result["level_1"],
            predicted_l2=result["level_2"],
            predicted_l3=result["level_3"],
            confidence=result["confidence"],
            is_valid_taxonomy=is_valid,
            api_calls=result["api_calls"],
            api_cost_usd=result["api_cost_usd"],
            latency_seconds=result["latency_seconds"],
            metadata=result["metadata"],
        )
        predictions.append(pred)
    
    # Compute metrics
    print("\n=== Evaluating ===")
    metrics = compute_hierarchical_metrics(predictions, d3_gold)
    
    # Pipeline metrics
    total_cost = sum(p.api_cost_usd for p in predictions)
    avg_latency = sum(p.latency_seconds for p in predictions) / len(predictions)
    invalid_rate = invalid_count / len(predictions)
    
    pipeline_metrics = {
        "total_api_cost_usd": total_cost,
        "avg_cost_per_sample": total_cost / len(predictions),
        "avg_latency_seconds": avg_latency,
        "invalid_taxonomy_rate": invalid_rate,
        "total_api_calls": sum(p.api_calls for p in predictions),
    }
    
    # Results
    result_doc = {
        "method": "M1_direct_llm",
        "model": args.model,
        "num_samples": len(predictions),
        "quality_metrics": metrics,
        "pipeline_metrics": pipeline_metrics,
        "predictions": [p.to_dict() for p in predictions],
    }
    
    # Save
    outdir = ROOT / "experiments" / "phase5_pipeline" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "M1_direct_llm_results.json", result_doc)
    
    # Print summary
    print("\n=== M1 Results ===")
    print(f"Quality Metrics:")
    print(f"  Accuracy L1: {metrics['accuracy_l1']:.4f}")
    print(f"  Accuracy L2: {metrics['accuracy_l2']:.4f}")
    print(f"  Accuracy L3: {metrics['accuracy_l3']:.4f}")
    print(f"  Macro-F1 L3: {metrics['macro_f1_l3']:.4f} ⭐")
    print(f"  Full-path Accuracy: {metrics['full_path_accuracy']:.4f}")
    print(f"\nPipeline Metrics:")
    print(f"  Invalid Taxonomy: {invalid_rate:.2%}")
    print(f"  Total Cost: ${total_cost:.4f}")
    print(f"  Avg Cost/Sample: ${pipeline_metrics['avg_cost_per_sample']:.6f}")
    print(f"  Avg Latency: {avg_latency:.2f}s")
    print(f"\nSaved: {outdir / 'M1_direct_llm_results.json'}")


if __name__ == "__main__":
    main()
