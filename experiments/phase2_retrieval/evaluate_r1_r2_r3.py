#!/usr/bin/env python3
"""Evaluate retrieval methods R1-R3 on D3 human gold test set.

R1: Regex/token-based retrieval
R2: Dense embedding retrieval
R3: Hybrid (union of R1 + R2)

Usage:
  python experiments/phase2_retrieval/evaluate_r1_r2_r3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase2_retrieval"))

from retrieval_utils import (  # noqa: E402
    EmbeddingRetriever,
    HybridRetriever,
    RegexRetriever,
    compute_retrieval_metrics,
    gold_intent_path,
    load_results,
    save_json,
)


def main():
    print("=== Phase 2: Retrieval Evaluation (R1-R3) ===\n")
    
    # Load corpus (D1/D2 train - excluding D3)
    print("Loading corpus...")
    d1_train = load_results(ROOT / "data/datasets/d1_annotator_train.json")
    print(f"Corpus size: {len(d1_train)} samples")
    
    # Load test queries (D3 human gold)
    print("Loading D3 human gold queries...")
    d3_gold = load_results(ROOT / "data/gold/d3_human_gold.json")
    print(f"Test queries: {len(d3_gold)} samples\n")
    
    # Initialize retrievers
    print("Initializing retrievers...")
    print("  R1: Regex/token-based")
    r1 = RegexRetriever(d1_train)
    
    print("  R2: Dense embedding (this may take a while...)")
    r2 = EmbeddingRetriever(d1_train)
    
    print("  R3: Hybrid (R1 + R2 union)")
    r3 = HybridRetriever(d1_train)
    
    retrievers = {
        "R1_regex": r1,
        "R2_embedding": r2,
        "R3_hybrid": r3,
    }
    
    # Evaluate each retriever
    results = {}
    k_values = [1, 3, 5, 10]
    
    for name, retriever in retrievers.items():
        print(f"\nEvaluating {name}...")
        
        all_metrics = []
        query_results = []
        
        for i, query_row in enumerate(d3_gold):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(d3_gold)}")
            
            query = query_row.get("sentence", "")
            gold_path = gold_intent_path(query_row)
            
            # Retrieve
            retrieved = retriever.retrieve(query, k=40)
            retrieved_indices = [idx for idx, score in retrieved]
            
            # Compute metrics
            metrics = compute_retrieval_metrics(
                retrieved_indices,
                gold_path,
                d1_train,
                k_values=k_values
            )
            all_metrics.append(metrics)
            
            # Log detailed results
            query_results.append({
                "sample_id": query_row.get("sample_id"),
                "query": query,
                "gold_l1": gold_path[0],
                "gold_l2": gold_path[1],
                "gold_l3": gold_path[2],
                "retrieved_top10": [
                    {
                        "corpus_idx": idx,
                        "score": float(score),
                        "sample_id": d1_train[idx].get("sample_id"),
                    }
                    for idx, score in retrieved[:10]
                ],
                "metrics": metrics,
            })
        
        # Aggregate metrics
        avg_metrics = {}
        for k in k_values:
            key = f"recall@{k}"
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        avg_metrics["mrr"] = sum(m["mrr"] for m in all_metrics) / len(all_metrics)
        avg_metrics["avg_taxonomy_coverage"] = sum(m["taxonomy_coverage"] for m in all_metrics) / len(all_metrics)
        
        results[name] = {
            "method": name,
            "corpus_size": len(d1_train),
            "num_queries": len(d3_gold),
            "k_values": k_values,
            "avg_metrics": avg_metrics,
            "query_results": query_results,
        }
        
        print(f"\n{name} Results:")
        for k in k_values:
            print(f"  Recall@{k}: {avg_metrics[f'recall@{k}']:.4f}")
        print(f"  MRR: {avg_metrics['mrr']:.4f}")
        print(f"  Avg Taxonomy Coverage: {avg_metrics['avg_taxonomy_coverage']:.2f}")
    
    # Save results
    outdir = ROOT / "experiments" / "phase2_retrieval" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    
    for name, result in results.items():
        save_json(outdir / f"{name}_results.json", result)
        print(f"\nSaved: {outdir / f'{name}_results.json'}")
    
    # Save summary
    summary = {
        "methods": list(results.keys()),
        "corpus": "d1_annotator_train (1454 samples)",
        "test_set": "d3_human_gold (300 samples)",
        "k_values": k_values,
        "summary_table": [
            {
                "method": name,
                **results[name]["avg_metrics"]
            }
            for name in results.keys()
        ]
    }
    save_json(outdir / "retrieval_summary.json", summary)
    print(f"\nSaved summary: {outdir / 'retrieval_summary.json'}")
    
    print("\n=== Evaluation Complete ===")
    print("\nSummary Table:")
    print(f"{'Method':<15} " + " ".join(f"R@{k:<2}" for k in k_values) + "  MRR    Coverage")
    for name in results.keys():
        metrics = results[name]["avg_metrics"]
        recalls = " ".join(f"{metrics[f'recall@{k}']:.3f}" for k in k_values)
        print(f"{name:<15} {recalls}  {metrics['mrr']:.3f}  {metrics['avg_taxonomy_coverage']:.1f}")


if __name__ == "__main__":
    main()
