#!/usr/bin/env python3
"""R4: TechniqueRAG-inspired retrieval with LLM reranking.

Retrieve top-K candidates → GPT-4o rerank by semantic fit + taxonomy → top-k exemplars

Usage:
  export OPENAI_API_KEY="your-key"
  python experiments/phase2_retrieval/evaluate_r4_reranking.py
  
  # Or with custom K/k:
  python experiments/phase2_retrieval/evaluate_r4_reranking.py --K 40 --k 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase2_retrieval"))

from retrieval_utils import (  # noqa: E402
    HybridRetriever,
    compute_retrieval_metrics,
    gold_intent_path,
    intent_path,
    load_results,
    save_json,
)


def rerank_with_llm(
    query: str,
    candidates: list[dict[str, Any]],
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> list[int]:
    """Rerank candidates using LLM reasoning about semantic fit and taxonomy.
    
    Returns reranked indices (into candidates list).
    """
    import openai
    
    # Build candidate descriptions
    candidate_texts = []
    for i, cand in enumerate(candidates):
        l1, l2, l3 = intent_path(cand)
        sent = cand.get("sentence", "")
        candidate_texts.append(f"[{i+1}] L1={l1}, L2={l2}, L3={l3}\n    Sentence: {sent}")
    
    prompt = f"""Bạn là chuyên gia phân loại intent cho e-commerce tiếng Việt.

Query cần gán nhãn: "{query}"

Các candidate intents (từ training set):
{chr(10).join(candidate_texts)}

Nhiệm vụ: Xếp hạng các candidates theo mức độ phù hợp ngữ nghĩa và taxonomy với query.

Xem xét:
1. Ngữ nghĩa: Query có ý định tương tự candidate không?
2. Taxonomy fit: L1/L2/L3 có hợp lý không?
3. Hierarchical consistency: L3 phải cụ thể hơn L2

Output format (chỉ JSON, không giải thích):
{{
  "ranked_indices": [<list of candidate indices 1-based, best first>],
  "reasoning": "<1-2 câu ngắn gọn>"
}}"""
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500,
        )
        
        import json
        result = json.loads(response.choices[0].message.content.strip())
        ranked_indices_1based = result.get("ranked_indices", [])
        
        # Convert to 0-based
        return [idx - 1 for idx in ranked_indices_1based if 0 < idx <= len(candidates)]
    
    except Exception as e:
        print(f"  Warning: LLM reranking failed: {e}")
        return list(range(len(candidates)))  # Fallback to original order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=40, help="Number of candidates to retrieve")
    parser.add_argument("--k", type=int, default=3, help="Number of top exemplars after reranking")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Reranking model")
    parser.add_argument("--limit", type=int, default=0, help="Limit queries for testing (0=all)")
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print("=== Phase 2: R4 TechniqueRAG-inspired Reranking ===\n")
    print(f"Config: K={args.K}, k={args.k}, model={args.model}\n")
    
    # Load corpus
    print("Loading corpus...")
    d1_train = load_results(ROOT / "data/datasets/d1_annotator_train.json")
    print(f"Corpus size: {len(d1_train)} samples")
    
    # Load test queries
    print("Loading D3 human gold...")
    d3_gold = load_results(ROOT / "data/gold/d3_human_gold.json")
    if args.limit > 0:
        d3_gold = d3_gold[:args.limit]
        print(f"Limited to {len(d3_gold)} queries for testing")
    else:
        print(f"Test queries: {len(d3_gold)} samples")
    
    # Initialize hybrid retriever
    print("\nInitializing hybrid retriever (R3)...")
    retriever = HybridRetriever(d1_train)
    
    # Evaluate with reranking
    print(f"\nReranking with {args.model}...")
    all_metrics = []
    query_results = []
    k_values = [1, 3, 5, 10]
    
    for i, query_row in enumerate(d3_gold):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(d3_gold)}")
        
        query = query_row.get("sentence", "")
        gold_path = gold_intent_path(query_row)
        
        # Step 1: Retrieve top-K with hybrid
        retrieved = retriever.retrieve(query, k=args.K)
        candidates = [d1_train[idx] for idx, score in retrieved]
        
        # Step 2: Rerank with LLM
        reranked_indices = rerank_with_llm(query, candidates, model=args.model)
        
        # Map back to corpus indices
        retrieved_corpus_indices = [retrieved[idx][0] for idx in reranked_indices if idx < len(retrieved)]
        
        # Step 3: Take top-k
        top_k_exemplars = retrieved_corpus_indices[:args.k]
        
        # Compute metrics on reranked order
        metrics = compute_retrieval_metrics(
            retrieved_corpus_indices,
            gold_path,
            d1_train,
            k_values=k_values
        )
        all_metrics.append(metrics)
        
        # Log
        query_results.append({
            "sample_id": query_row.get("sample_id"),
            "query": query,
            "gold_l1": gold_path[0],
            "gold_l2": gold_path[1],
            "gold_l3": gold_path[2],
            "initial_top_K": [
                {
                    "corpus_idx": idx,
                    "score": float(score),
                }
                for idx, score in retrieved[:args.K]
            ],
            "reranked_top_k": [
                {
                    "corpus_idx": idx,
                    "sample_id": d1_train[idx].get("sample_id"),
                }
                for idx in top_k_exemplars
            ],
            "metrics": metrics,
        })
    
    # Aggregate
    avg_metrics = {}
    for k in k_values:
        key = f"recall@{k}"
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    avg_metrics["mrr"] = sum(m["mrr"] for m in all_metrics) / len(all_metrics)
    avg_metrics["avg_taxonomy_coverage"] = sum(m["taxonomy_coverage"] for m in all_metrics) / len(all_metrics)
    
    result = {
        "method": "R4_techniquerag_reranking",
        "config": {
            "K": args.K,
            "k": args.k,
            "reranker_model": args.model,
            "retriever": "hybrid_r3",
        },
        "corpus_size": len(d1_train),
        "num_queries": len(d3_gold),
        "k_values": k_values,
        "avg_metrics": avg_metrics,
        "query_results": query_results,
    }
    
    print(f"\nR4 Results:")
    for k in k_values:
        print(f"  Recall@{k}: {avg_metrics[f'recall@{k}']:.4f}")
    print(f"  MRR: {avg_metrics['mrr']:.4f}")
    print(f"  Avg Taxonomy Coverage: {avg_metrics['avg_taxonomy_coverage']:.2f}")
    
    # Save
    outdir = ROOT / "experiments" / "phase2_retrieval" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "R4_techniquerag_results.json", result)
    print(f"\nSaved: {outdir / 'R4_techniquerag_results.json'}")


if __name__ == "__main__":
    main()
