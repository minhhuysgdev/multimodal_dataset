#!/usr/bin/env python3
"""RQ3: Annotation cost analysis across methods."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from common import (  # noqa: E402
    METHODS,
    METHOD_DISPLAY,
    RESULTS_DIR,
    annotation_output_path,
    load_json,
    save_json,
)


def human_review_required(method: str, row: dict) -> bool:
    if method == "verified_d2":
        meta = row.get("metadata") or {}
        return bool(meta.get("verifier_called"))
    return False


def summarize_method(method: str) -> dict:
    path = annotation_output_path(method)
    if not path.exists():
        return {"method_key": method, "error": "missing"}

    doc = load_json(path)
    rows = doc.get("results", [])
    total_calls = sum(r.get("api_calls", 0) for r in rows)
    total_in = sum(r.get("input_tokens", 0) for r in rows)
    total_out = sum(r.get("output_tokens", 0) for r in rows)
    total_cost = sum(r.get("cost_usd", 0.0) for r in rows)
    review_count = sum(1 for r in rows if human_review_required(method, r))

    post_processing = "SLM_distillation" if doc.get("distillation") else "none"

    return {
        "method": METHOD_DISPLAY.get(method, method),
        "method_key": method,
        "samples_annotated": len(rows),
        "llm_calls": total_calls,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "estimated_cost_usd": round(total_cost, 4),
        "avg_cost_per_sample": round(total_cost / len(rows), 6) if rows else 0,
        "human_review_samples": review_count,
        "human_review_rate": round(review_count / len(rows), 4) if rows else 0,
        "post_processing": post_processing,
        "model": doc.get("model", ""),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in METHODS:
        row = summarize_method(method)
        if "error" not in row:
            rows.append(row)
            print(
                f"{row['method']:<28} samples={row['samples_annotated']} "
                f"calls={row['llm_calls']} cost=${row['estimated_cost_usd']:.4f}"
            )
        else:
            print(f"WARNING: {method} annotations missing")

    csv_path = args.outdir / "annotation_cost.csv"
    fieldnames = [
        "method", "method_key", "samples_annotated", "llm_calls",
        "input_tokens", "output_tokens", "estimated_cost_usd",
        "avg_cost_per_sample", "human_review_samples", "human_review_rate",
        "post_processing", "model",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_json(args.outdir / "annotation_cost.json", {"results": rows})
    print(f"\nWrote: {csv_path}")


if __name__ == "__main__":
    main()
