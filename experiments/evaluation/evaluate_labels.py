#!/usr/bin/env python3
"""Evaluate label quality: D1 vs D3 and D2 vs D3 on human gold test set.

Requires D3 with filled gold_level_1/2/3 (or intent_3_level as gold).

Usage:
  python experiments/evaluation/evaluate_labels.py \\
    --d3-file data/gold/d3_human_gold.json \\
    --d1-source data/raw/hasaki/hasaki_labelled_clean.json \\
    --d2-source data/raw/hasaki/hasaki_labelled_approved_merged.json
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "evaluation"))

from utils import (  # noqa: E402
    compute_hierarchical_metrics,
    gold_levels,
    intent_levels,
    load_results,
    machine_levels,
    rows_with_gold,
    save_json,
)

DEFAULT_D3 = ROOT / "data/gold/d3_dataset.json"
DEFAULT_D1 = ROOT / "data/raw/hasaki/d1_dataset.json"
DEFAULT_D2 = ROOT / "data/d2_dataset.json"
DEFAULT_OUT = ROOT / "experiments/evaluation/results"


def compare_dataset(
    name: str,
    d3_rows: list,
    label_by_id: dict,
    get_pred,
) -> dict:
    true_paths = []
    pred_paths = []
    missing = 0
    for r in d3_rows:
        sid = r["sample_id"]
        if sid not in label_by_id:
            missing += 1
            continue
        true_paths.append(gold_levels(r))
        pred_paths.append(get_pred(label_by_id[sid]))

    metrics = compute_hierarchical_metrics(true_paths, pred_paths)
    metrics["dataset"] = name
    metrics["missing_predictions"] = missing
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d3-file", type=Path, default=DEFAULT_D3)
    parser.add_argument("--d1-source", type=Path, default=DEFAULT_D1)
    parser.add_argument("--d2-source", type=Path, default=DEFAULT_D2)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    d3_all = load_results(args.d3_file)
    d3_gold = rows_with_gold(d3_all)

    if not d3_gold:
        print(
            "ERROR: No annotated gold labels found in D3 file.\n"
            "Fill gold_level_1, gold_level_2, gold_level_3 in CSV/JSON first,\n"
            "then save as data/gold/d3_human_gold.json and re-run.",
            file=sys.stderr,
        )
        print(f"Template has {len(d3_all)} rows but 0 with complete gold labels.")
        sys.exit(1)

    d1_rows = load_results(args.d1_source)
    d2_rows = load_results(args.d2_source)
    d1_by_id = {r["sample_id"]: r for r in d1_rows if r.get("sample_id")}
    d2_by_id = {r["sample_id"]: r for r in d2_rows if r.get("sample_id")}

    d1_metrics = compare_dataset(
        "D1_annotator",
        d3_gold,
        d1_by_id,
        intent_levels,
    )
    d2_metrics = compare_dataset(
        "D2_verified",
        d3_gold,
        d2_by_id,
        intent_levels,
    )

    # Also compare machine labels stored in D3 template vs gold
    d3_machine_metrics = compare_dataset(
        "D3_machine_reference",
        d3_gold,
        {r["sample_id"]: r for r in d3_gold},
        machine_levels,
    )

    payload = {
        "d3_file": str(args.d3_file.resolve()),
        "n_d3_annotated": len(d3_gold),
        "metrics": {
            "D1_vs_human_gold": d1_metrics,
            "D2_vs_human_gold": d2_metrics,
            "D3_machine_vs_human_gold": d3_machine_metrics,
        },
        "summary_table": [
            {
                "dataset": "D1: GPT Annotator",
                "l3_accuracy": d1_metrics["l3_accuracy"],
                "l3_macro_f1": d1_metrics["l3_macro_f1"],
                "full_path_accuracy": d1_metrics["full_path_accuracy"],
            },
            {
                "dataset": "D2: GPT Verified",
                "l3_accuracy": d2_metrics["l3_accuracy"],
                "l3_macro_f1": d2_metrics["l3_macro_f1"],
                "full_path_accuracy": d2_metrics["full_path_accuracy"],
            },
        ],
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    save_json(args.outdir / "label_quality_metrics.json", payload)

    csv_path = args.outdir / "label_quality_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "l3_accuracy", "l3_macro_f1", "full_path_accuracy"],
        )
        writer.writeheader()
        writer.writerows(payload["summary_table"])

    print(f"D3 annotated samples: {len(d3_gold)}")
    print("\nLabel Quality (vs Human Gold):")
    print(f"{'Dataset':<25} {'L3 Acc':>8} {'Macro-F1':>10} {'Full-path':>10}")
    for row in payload["summary_table"]:
        print(
            f"{row['dataset']:<25} "
            f"{row['l3_accuracy']:>8.4f} "
            f"{row['l3_macro_f1']:>10.4f} "
            f"{row['full_path_accuracy']:>10.4f}"
        )
    print(f"\nWrote: {args.outdir / 'label_quality_metrics.json'}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
