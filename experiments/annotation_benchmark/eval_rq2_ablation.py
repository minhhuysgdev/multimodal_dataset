#!/usr/bin/env python3
"""RQ2: CV-HIL component ablation (variants A–D) vs D3 human gold.

Loads data/annotations/cvhil_{a,b,c,d}.json and writes:
  - results/rq2_ablation/ablation_quality.csv
  - results/rq2_ablation/ablation_quality.json
  - results/rq2_ablation/ablation_quality_paper.csv
  - results/rq2_ablation/ablation_quality_paper.md
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from common import (  # noqa: E402
    ANNOTATIONS_DIR,
    RESULTS_DIR,
    eval_predictions_vs_d3,
    load_d3_gold,
    load_json,
    save_json,
)

RQ2_VARIANTS = ["cvhil_a", "cvhil_b", "cvhil_c", "cvhil_d"]

PAPER_VARIANT_LABEL = {
    "cvhil_a": "A. LLM-only",
    "cvhil_b": "B. Retrieval-Augmented",
    "cvhil_c": "C. Retrieval + Verification",
    "cvhil_d": "D. Full CV-HIL",
}

PAPER_VARIANT_FULL_NAME = {
    "cvhil_d": "Cross-Verified Hierarchical Intent Labeling",
}


def _round4(value: float) -> float:
    return round(float(value), 4)


def _pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _build_row(variant_key: str, metrics: dict) -> dict:
    return {
        "method": PAPER_VARIANT_LABEL.get(variant_key, variant_key),
        "method_full_name": PAPER_VARIANT_FULL_NAME.get(variant_key, ""),
        "method_key": variant_key,
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
        "Variant": row["method"],
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
    headers = [
        "Variant",
        "N",
        "L1 Acc",
        "L2 Acc",
        "L3 Acc",
        "L3 Macro-F1",
        "Full-path Acc",
    ]
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
        "\n\n*RQ2 component ablation on D3 human gold. "
        "D (Full CV-HIL) reused from verified_d2 cache when available.*"
    )
    return "\n".join(lines) + note


def _delta_table(rows_by_key: dict[str, dict]) -> str:
    """Incremental gains A→B→C→D for L3 Macro-F1 and full-path."""
    order = RQ2_VARIANTS
    lines = ["", "## Incremental gains (Δ vs previous variant)", ""]
    lines.append("| Step | Δ L3 Macro-F1 | Δ Full-path Acc |")
    lines.append("| --- | --- | --- |")
    for i in range(1, len(order)):
        prev_k, cur_k = order[i - 1], order[i]
        if prev_k not in rows_by_key or cur_k not in rows_by_key:
            continue
        prev, cur = rows_by_key[prev_k], rows_by_key[cur_k]
        d_f1 = cur["l3_macro_f1"] - prev["l3_macro_f1"]
        d_full = cur["full_path_accuracy"] - prev["full_path_accuracy"]
        step = f"{PAPER_VARIANT_LABEL[prev_k]} → {PAPER_VARIANT_LABEL[cur_k]}"
        lines.append(
            f"| {step} | {_pct(d_f1)} | {_pct(d_full)} |"
        )
    return "\n".join(lines)


def annotation_path(variant_key: str, annotations_dir: Path) -> Path:
    return annotations_dir / f"{variant_key}.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=ANNOTATIONS_DIR,
        help="Directory containing cvhil_*.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=RESULTS_DIR / "rq2_ablation",
    )
    parser.add_argument(
        "--variants",
        default=",".join(RQ2_VARIANTS),
        help="Comma-separated variant keys",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    variant_keys = [v.strip() for v in args.variants.split(",") if v.strip()]
    d3_gold = load_d3_gold()
    rows_by_key: dict[str, dict] = {}

    for variant_key in variant_keys:
        path = annotation_path(variant_key, args.annotations_dir)
        if not path.exists():
            print(f"WARNING: missing {path}, skipping")
            continue
        doc = load_json(path)
        preds = doc.get("results", [])
        metrics = eval_predictions_vs_d3(preds, d3_gold)
        rows_by_key[variant_key] = _build_row(variant_key, metrics)

    rows = [rows_by_key[k] for k in RQ2_VARIANTS if k in rows_by_key]
    if not rows:
        raise SystemExit("No ablation annotation files found.")

    for row in rows:
        print(
            f"{row['method']:<32} "
            f"L1={_pct(row['l1_accuracy']):>6} "
            f"L2={_pct(row['l2_accuracy']):>6} "
            f"L3={_pct(row['l3_accuracy']):>6} "
            f"Macro-F1={_pct(row['l3_macro_f1']):>6} "
            f"Full={_pct(row['full_path_accuracy']):>6}"
        )

    csv_path = args.outdir / "ablation_quality.csv"
    fieldnames = [
        "method",
        "method_full_name",
        "method_key",
        "num_matched",
        "l1_accuracy",
        "l2_accuracy",
        "l3_accuracy",
        "l3_macro_f1",
        "l3_weighted_f1",
        "full_path_accuracy",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.outdir / "ablation_quality.json"
    save_json(json_path, {"variants": rows})

    paper_rows = [_to_paper_row(r) for r in rows]
    paper_csv = args.outdir / "ablation_quality_paper.csv"
    paper_fields = [
        "Variant",
        "N",
        "L1 Acc",
        "L2 Acc",
        "L3 Acc",
        "L3 Macro-F1",
        "Full-path Acc",
        "method_key",
        "l1_accuracy",
        "l2_accuracy",
        "l3_accuracy",
        "l3_macro_f1",
        "l3_weighted_f1",
        "full_path_accuracy",
    ]
    with paper_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=paper_fields)
        writer.writeheader()
        writer.writerows(paper_rows)

    md_path = args.outdir / "ablation_quality_paper.md"
    md_body = _markdown_table(paper_rows) + _delta_table(rows_by_key)
    md_path.write_text(md_body, encoding="utf-8")

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {paper_csv}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
