#!/usr/bin/env python3
"""M2: Hybrid RAG baseline (R3 candidates + GPT-4o-mini).

Usage:
  export OPENAI_API_KEY="your-key"
  python experiments/phase5_pipeline/run_m2_rag.py --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase2_retrieval"))
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
from retrieval_utils import HybridRetriever, intent_path  # noqa: E402


def annotate_with_rag(
    sentence: str,
    candidates: list[dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Annotate with RAG candidates."""
    import openai
    
    # Format candidates
    cand_text = []
    for i, cand in enumerate(candidates[:5], 1):  # Top-5
        l1, l2, l3 = intent_path(cand)
        cand_text.append(f"[{i}] L1={l1}, L2={l2}, L3={l3}\n    Câu tương tự: {cand.get('sentence', '')}")
    
    prompt = f"""Bạn là chuyên gia phân loại intent cho e-commerce tiếng Việt.

Gán nhãn phân cấp 3 tầng cho câu dưới đây, tham khảo các ví dụ tương tự:

Câu cần gán nhãn: "{sentence}"

Các ví dụ tương tự từ training set:
{chr(10).join(cand_text)}

Output STRICTLY JSON (không giải thích):
{{
  "level_1": "<L1>",
  "level_2": "<L2>",
  "level_3": "<L3>",
  "confidence": <0.0-1.0>
}}"""
    
    start_time = time.time()
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        
        latency = time.time() - start_time
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        
        usage = response.get("usage", {})
        cost = estimate_api_cost(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        
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
                "num_candidates": len(candidates),
                "retrieval_method": "hybrid_r3",
            }
        }
    except Exception as e:
        return {
            "level_1": "", "level_2": "", "level_3": "", "confidence": 0.0,
            "api_calls": 1, "api_cost_usd": 0.0, "latency_seconds": time.time() - start_time,
            "metadata": {"error": str(e)},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    print("=== M2: Hybrid RAG Baseline ===\n")
    
    # Load data
    d3_gold = load_results(ROOT / "data/gold/d3_human_gold.json")
    if args.limit > 0:
        d3_gold = d3_gold[:args.limit]
    
    d1_train = load_results(ROOT / "data/datasets/d1_annotator_train.json")
    print(f"Test: {len(d3_gold)}, Corpus: {len(d1_train)}")
    
    # Initialize retriever
    print("Initializing hybrid retriever...")
    retriever = HybridRetriever(d1_train)
    
    # Taxonomy validator
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
        
        # Retrieve
        retrieved = retriever.retrieve(sentence, k=40)
        candidates = [d1_train[idx] for idx, score in retrieved]
        
        # Annotate
        result = annotate_with_rag(sentence, candidates)
        
        # Validate
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
    
    # Evaluate
    metrics = compute_hierarchical_metrics(predictions, d3_gold)
    total_cost = sum(p.api_cost_usd for p in predictions)
    
    result_doc = {
        "method": "M2_hybrid_rag",
        "num_samples": len(predictions),
        "quality_metrics": metrics,
        "pipeline_metrics": {
            "total_cost_usd": total_cost,
            "avg_cost_per_sample": total_cost / len(predictions),
            "invalid_rate": invalid_count / len(predictions),
        },
        "predictions": [p.to_dict() for p in predictions],
    }
    
    # Save
    outdir = ROOT / "experiments" / "phase5_pipeline" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "M2_rag_results.json", result_doc)
    
    print(f"\n=== M2 Results ===")
    print(f"Macro-F1 L3: {metrics['macro_f1_l3']:.4f}")
    print(f"Full-path Acc: {metrics['full_path_accuracy']:.4f}")
    print(f"Invalid Rate: {invalid_count / len(predictions):.2%}")
    print(f"Total Cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
