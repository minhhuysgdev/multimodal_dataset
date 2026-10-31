#!/usr/bin/env python3
"""Phase 3: Build training datasets from annotations (exclude D3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from common import (  # noqa: E402
    METHODS,
    DATASETS_DIR,
    TRAIN_POOL_IDS,
    annotation_output_path,
    load_id_set,
    load_json,
    save_json,
)


def build_dataset_for_method(method: str, train_pool_ids: set[str]) -> dict:
    ann_path = annotation_output_path(method)
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotations: {ann_path}")

    doc = load_json(ann_path)
    annotations = doc.get("results", [])
    ann_by_id = {r["sample_id"]: r for r in annotations}

    results = []
    missing = 0
    d3_leak = 0

    for sid in sorted(train_pool_ids):
        if sid not in ann_by_id:
            missing += 1
            continue
        row = ann_by_id[sid]
        label = row.get("predicted_label") or {}
        results.append({
            "sample_id": sid,
            "sentence": row.get("sentence", ""),
            "intent_3_level": {
                "level_1": label.get("l1", ""),
                "level_2": label.get("l2", ""),
                "level_3": label.get("l3", ""),
            },
            "confidence": row.get("confidence", 0.0),
            "annotation_method": method,
        })

    # Safety check: no D3 ids
    d3_in_results = [r for r in results if r["sample_id"] not in train_pool_ids]
    d3_leak = len(d3_in_results)

    out_name = f"{method}_dataset.json"
    payload = {
        "dataset": out_name,
        "annotation_method": method,
        "do_not_use_for_training": False,
        "excludes_d3_gold": True,
        "num_samples": len(results),
        "missing_annotations": missing,
        "d3_leak_count": d3_leak,
        "results": results,
    }
    save_json(DATASETS_DIR / out_name, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS + ["all"], default="all")
    args = parser.parse_args()

    train_pool_ids = load_id_set(TRAIN_POOL_IDS)
    if not train_pool_ids:
        raise RuntimeError(f"Missing train pool ids: {TRAIN_POOL_IDS}")

    methods = METHODS if args.method == "all" else [args.method]
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    for method in methods:
        ann_path = annotation_output_path(method)
        if not ann_path.exists():
            print(f"WARNING: missing {ann_path}, skipping")
            continue
        payload = build_dataset_for_method(method, train_pool_ids)
        print(
            f"{method}: {payload['num_samples']} samples "
            f"(missing={payload['missing_annotations']}, d3_leak={payload['d3_leak_count']})"
        )


if __name__ == "__main__":
    main()
