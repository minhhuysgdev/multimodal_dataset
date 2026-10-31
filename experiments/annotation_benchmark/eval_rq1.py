#!/usr/bin/env python3
"""RQ1: Compare annotation quality vs D3 human gold.

Outputs for paper/report:
  - annotation_quality.csv / .json   (full metrics)
  - annotation_quality_paper.csv   (rounded, paper labels)
  - annotation_quality_paper.md      (markdown table, copy into report)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from common import (  # noqa: E402
    METHODS,
    RESULTS_DIR,
    annotation_output_path,
    eval_predictions_vs_d3,
    load_d3_gold,
    load_json,
    save_json,
)

# Paper-facing labels (defined here so report naming stays in one place)
PAPER_METHOD_LABEL = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "cot": "Chain-of-Thought",
    "freeal": "FreeAL",
    "candist": "CanDist",
    "verified_d2": "RVI-HUL (Ours)",
}

PAPER_METHOD_FULL_NAME = {
    "verified_d2": "Cross-Verified Hierarchical Intent Labeling",
}

# Baselines first, proposed method last
PAPER_METHOD_ORDER = [
    "zero_shot",
    "few_shot",
    "cot",
    "freeal",
    "candist",
    "verified_d2",
]


def _paper_label(method_key: str) -> str:
    return PAPER_METHOD_LABEL.get(method_key, method_key)


def _round4(value: float) -> float:
    return round(float(value), 4)


def _pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _build_row(method_key: str, metrics: dict) -> dict:
    return {
        "method": _paper_label(method_key),
        "method_full_name": PAPER_METHOD_FULL_NAME.get(method_key, ""),
        "method_key": method_key,
        "num_matched": metrics.get("n_samples", 0),
        "l1_accuracy": metrics.get("l1_accuracy", 0),
        "l2_accuracy": metrics.get("l2_accuracy", 0),
        "l3_accuracy": metrics.get("l3_accuracy", 0),
        "l3_macro_f1": metrics.get("l3_macro_f1", 0),
        "l3_weighted_f1": metrics.get("l3_weighted_f1", 0),
        "full_path_accuracy": metrics.get("full_path_accuracy", 0),
    }


def _to_paper_row(row: dict) -> dict:
    return {
        "Method": row["method"],
        "N": row["num_matched"],
        "L1 Acc": _pct(row["l1_accuracy"]),
        "L2 Acc": _pct(row["l2_accuracy"]),
        "L3 Acc": _pct(row["l3_accuracy"]),
        "L3 Macro-F1": _pct(row["l3_macro_f1"]),
        "Full-path Acc": _pct(row["full_path_accuracy"]),
        "method_key": row["method_key"],
        "l1_accuracy": _round4(row["l1_accuracy"]),
        "l2_accuracy": _round4(row["l2_accuracy"]),
        "l3_accuracy": _round4(row["l3_accuracy"]),
        "l3_macro_f1": _round4(row["l3_macro_f1"]),
        "l3_weighted_f1": _round4(row["l3_weighted_f1"]),
        "full_path_accuracy": _round4(row["full_path_accuracy"]),
    }


def _markdown_table(paper_rows: list[dict]) -> str:
    headers = ["Method", "N", "L1 Acc", "L2 Acc", "L3 Acc", "L3 Macro-F1", "Full-path Acc"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in paper_rows:
        lines.append(
            "| "
            + " | ".join(str(row[h]) for h in headers)
            + " |"
        )
    note = (
        "\n\n*RVI-HUL = Cross-Verified Hierarchical Intent Labeling (proposed). "
        "Evaluated on D3 human gold (N=300).*"
    )
    return "\n".join(lines) + note


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    d3_gold = load_d3_gold()
    rows_by_key: dict[str, dict] = {}

    for method in METHODS:
        path = annotation_output_path(method)
        if not path.exists():
            print(f"WARNING: missing {path}, skipping")
            continue
        doc = load_json(path)
        preds = doc.get("results", [])
        metrics = eval_predictions_vs_d3(preds, d3_gold)
        rows_by_key[method] = _build_row(method, metrics)

    rows = [rows_by_key[k] for k in PAPER_METHOD_ORDER if k in rows_by_key]

    for row in rows:
        print(
            f"{row['method']:<22} "
            f"L1={_pct(row['l1_accuracy']):>6} "
            f"L2={_pct(row['l2_accuracy']):>6} "
            f"L3={_pct(row['l3_accuracy']):>6} "
            f"Macro-F1={_pct(row['l3_macro_f1']):>6} "
            f"Full={_pct(row['full_path_accuracy']):>6}"
        )

    csv_path = args.outdir / "annotation_quality.csv"
    fieldnames = [
        "method", "method_full_name", "method_key", "num_matched",
        "l1_accuracy", "l2_accuracy", "l3_accuracy",
        "l3_macro_f1", "l3_weighted_f1", "full_path_accuracy",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_json(args.outdir / "annotation_quality.json", {"results": rows})

    paper_rows = [_to_paper_row(r) for r in rows]
    paper_csv_path = args.outdir / "annotation_quality_paper.csv"
    paper_fieldnames = [
        "Method", "N", "L1 Acc", "L2 Acc", "L3 Acc", "L3 Macro-F1", "Full-path Acc",
        "method_key", "l1_accuracy", "l2_accuracy", "l3_accuracy",
        "l3_macro_f1", "l3_weighted_f1", "full_path_accuracy",
    ]
    with paper_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=paper_fieldnames)
        writer.writeheader()
        writer.writerows(paper_rows)

    md_path = args.outdir / "annotation_quality_paper.md"
    md_text = _markdown_table(paper_rows)
    md_path.write_text(md_text, encoding="utf-8")

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {paper_csv_path}")
    print(f"Wrote: {md_path}")
    print("\n--- Markdown (copy into report) ---\n")
    print(md_text)


if __name__ == "__main__":
    main()
