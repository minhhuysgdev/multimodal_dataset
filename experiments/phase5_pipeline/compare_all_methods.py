#!/usr/bin/env python3
"""Compare all Phase 5 pipeline baselines.

Usage:
  python experiments/phase5_pipeline/compare_all_methods.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase5_pipeline"))

from pipeline_utils import load_json  # noqa: E402


def main():
    print("=== Phase 5: Pipeline Baseline Comparison ===\n")
    
    results_dir = ROOT / "experiments" / "phase5_pipeline" / "results"
    
    methods = [
        ("M1_direct_llm_results.json", "M1: Direct LLM"),
        ("M2_rag_results.json", "M2: Hybrid RAG"),
        ("M2_SR_selfrefine_results.json", "M2-SR: Self-Refine"),
        ("M3_full_pipeline_results.json", "M3: Full Pipeline"),
    ]
    
    print(f"{'Method':<25} {'Acc-L3':<8} {'F1-L3':<8} {'F-Path':<8} {'Invalid%':<9} {'Cost$':<8}")
    print("-" * 85)
    
    all_data = []
    
    for filename, label in methods:
        filepath = results_dir / filename
        
        if not filepath.exists():
            print(f"{label:<25} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<9} {'N/A':<8}")
            continue
        
        data = load_json(filepath)
        qm = data.get("quality_metrics", {})
        pm = data.get("pipeline_metrics", {})
        
        acc_l3 = qm.get("accuracy_l3", 0.0)
        f1_l3 = qm.get("macro_f1_l3", 0.0)
        full_path = qm.get("full_path_accuracy", 0.0)
        invalid = pm.get("invalid_rate", pm.get("invalid_taxonomy_rate", 0.0))
        cost = pm.get("total_cost_usd", pm.get("total_api_cost_usd", 0.0))
        
        print(f"{label:<25} {acc_l3:<8.4f} {f1_l3:<8.4f} {full_path:<8.4f} {invalid*100:<9.2f} ${cost:<7.4f}")
        
        all_data.append({
            "method": label,
            "accuracy_l3": acc_l3,
            "macro_f1_l3": f1_l3,
            "full_path_accuracy": full_path,
            "invalid_rate": invalid,
            "total_cost_usd": cost,
        })
    
    # Save summary
    from pipeline_utils import save_json
    summary = {
        "comparison": all_data,
        "notes": "All methods evaluated on D3 human gold (300 samples)"
    }
    save_json(results_dir / "comparison_summary.json", summary)
    
    print(f"\nSummary saved: {results_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
