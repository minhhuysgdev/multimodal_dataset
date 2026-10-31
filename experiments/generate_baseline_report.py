#!/usr/bin/env python3
"""Generate summary tables and figures for all baseline evaluations.

Usage:
  python experiments/generate_baseline_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "phase5_pipeline"))

from pipeline_utils import load_json, save_json  # noqa: E402


def collect_retrieval_results():
    """Collect Phase 2 retrieval results."""
    results_dir = ROOT / "experiments" / "phase2_retrieval" / "results"
    
    if not results_dir.exists():
        return None
    
    summary_file = results_dir / "retrieval_summary.json"
    if summary_file.exists():
        return load_json(summary_file)
    
    return None


def collect_pipeline_results():
    """Collect Phase 5 pipeline results."""
    results_dir = ROOT / "experiments" / "phase5_pipeline" / "results"
    
    if not results_dir.exists():
        return None
    
    comparison_file = results_dir / "comparison_summary.json"
    if comparison_file.exists():
        return load_json(comparison_file)
    
    return None


def collect_routing_results():
    """Collect Phase 6 routing results."""
    results_dir = ROOT / "experiments" / "phase6_review_routing" / "results"
    
    if not results_dir.exists():
        return None
    
    summary_file = results_dir / "routing_comparison_summary.json"
    if summary_file.exists():
        return load_json(summary_file)
    
    return None


def generate_summary_table():
    """Generate consolidated summary table."""
    print("=== Baseline Evaluation Summary ===\n")
    
    # Phase 2: Retrieval
    retrieval_data = collect_retrieval_results()
    if retrieval_data:
        print("## Phase 2: Retrieval Baselines")
        print(f"{'Method':<20} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'MRR':<8}")
        print("-" * 60)
        for method_data in retrieval_data.get("summary_table", []):
            method = method_data.get("method", "")
            r1 = method_data.get("recall@1", 0.0)
            r3 = method_data.get("recall@3", 0.0)
            r5 = method_data.get("recall@5", 0.0)
            mrr = method_data.get("mrr", 0.0)
            print(f"{method:<20} {r1:<8.4f} {r3:<8.4f} {r5:<8.4f} {mrr:<8.4f}")
        print()
    
    # Phase 5: Pipeline
    pipeline_data = collect_pipeline_results()
    if pipeline_data:
        print("## Phase 5: Annotation Pipeline Baselines")
        print(f"{'Method':<25} {'Acc-L3':<8} {'F1-L3':<8} {'F-Path':<8} {'Cost$':<8}")
        print("-" * 70)
        for method_data in pipeline_data.get("comparison", []):
            method = method_data.get("method", "")
            acc = method_data.get("accuracy_l3", 0.0)
            f1 = method_data.get("macro_f1_l3", 0.0)
            fpath = method_data.get("full_path_accuracy", 0.0)
            cost = method_data.get("total_cost_usd", 0.0)
            print(f"{method:<25} {acc:<8.4f} {f1:<8.4f} {fpath:<8.4f} ${cost:<7.2f}")
        print()
    
    # Phase 6: Routing
    routing_data = collect_routing_results()
    if routing_data:
        print("## Phase 6: Review-Routing Baselines")
        print(f"{'Method':<20} {'Budget':<8} {'F1-L3':<8} {'F-Path':<8} {'Review%':<9}")
        print("-" * 70)
        for comp in routing_data.get("comparisons", []):
            method = comp.get("method", "")
            budget = comp.get("budget", 0)
            f1 = comp.get("macro_f1_l3", 0.0)
            fpath = comp.get("full_path_accuracy", 0.0)
            review = comp.get("review_rate", 0.0)
            print(f"{method:<20} {budget:<8}% {f1:<8.4f} {fpath:<8.4f} {review*100:<9.1f}%")
        print()


def plot_quality_cost_curve():
    """Plot quality vs cost for pipeline baselines."""
    pipeline_data = collect_pipeline_results()
    
    if not pipeline_data:
        print("No pipeline data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    f1_scores = []
    costs = []
    
    for method_data in pipeline_data.get("comparison", []):
        methods.append(method_data.get("method", ""))
        f1_scores.append(method_data.get("macro_f1_l3", 0.0))
        costs.append(method_data.get("total_cost_usd", 0.0))
    
    ax.scatter(costs, f1_scores, s=100, alpha=0.7)
    
    for i, method in enumerate(methods):
        ax.annotate(method, (costs[i], f1_scores[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel("Total Cost (USD)", fontsize=12)
    ax.set_ylabel("Macro-F1 L3", fontsize=12)
    ax.set_title("Quality-Cost Trade-off: Pipeline Baselines", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    outdir = ROOT / "experiments" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "quality_cost_curve.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {outdir / 'quality_cost_curve.png'}")
    plt.close()


def plot_review_budget_curve():
    """Plot quality vs review budget for routing baselines."""
    routing_data = collect_routing_results()
    
    if not routing_data:
        print("No routing data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by method
    by_method = {}
    for comp in routing_data.get("comparisons", []):
        method = comp.get("method", "")
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(comp)
    
    for method, data_points in by_method.items():
        budgets = [d.get("budget", 0) for d in data_points]
        f1_scores = [d.get("macro_f1_l3", 0.0) for d in data_points]
        
        ax.plot(budgets, f1_scores, marker='o', label=method, linewidth=2)
    
    ax.set_xlabel("Review Budget (%)", fontsize=12)
    ax.set_ylabel("Macro-F1 L3", fontsize=12)
    ax.set_title("Quality vs Review Budget: Routing Strategies", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    outdir = ROOT / "experiments" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "review_budget_curve.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {outdir / 'review_budget_curve.png'}")
    plt.close()


def main():
    print("Generating baseline evaluation report...\n")
    
    # Summary table
    generate_summary_table()
    
    # Plots
    try:
        plot_quality_cost_curve()
        plot_review_budget_curve()
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    print("\n✅ Report generation complete!")


if __name__ == "__main__":
    main()
