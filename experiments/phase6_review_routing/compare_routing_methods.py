#!/usr/bin/env python3
"""Compare all review-routing methods.

Usage:
  python experiments/phase6_review_routing/compare_routing_methods.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase6_review_routing"))

from routing_utils import load_json, save_json  # noqa: E402


def main():
    print("=== Phase 6: Review-Routing Comparison ===\n")
    
    results_dir = ROOT / "experiments" / "phase6_review_routing" / "results"
    
    budgets = [20, 40, 60]
    methods = ["B1_random", "B2_confidence"]
    
    all_comparisons = []
    
    for budget in budgets:
        print(f"\n--- Budget: {budget}% ---")
        print(f"{'Method':<20} {'F1-L3':<8} {'Full-Path':<10} {'Review%':<9}")
        print("-" * 50)
        
        for method in methods:
            filename = f"{method}_budget{budget}.json"
            filepath = results_dir / filename
            
            if not filepath.exists():
                print(f"{method:<20} {'N/A':<8} {'N/A':<10} {'N/A':<9}")
                continue
            
            data = load_json(filepath)
            qm = data.get("quality_metrics", {})
            
            f1_l3 = qm.get("macro_f1_l3", 0.0)
            full_path = qm.get("full_path_accuracy", 0.0)
            review_rate = data.get("review_rate", 0.0)
            
            print(f"{method:<20} {f1_l3:<8.4f} {full_path:<10.4f} {review_rate*100:<9.1f}")
            
            all_comparisons.append({
                "method": method,
                "budget": budget,
                "macro_f1_l3": f1_l3,
                "full_path_accuracy": full_path,
                "review_rate": review_rate,
            })
    
    # Save
    summary = {"comparisons": all_comparisons}
    save_json(results_dir / "routing_comparison_summary.json", summary)
    print(f"\nSummary saved: {results_dir / 'routing_comparison_summary.json'}")


if __name__ == "__main__":
    main()
