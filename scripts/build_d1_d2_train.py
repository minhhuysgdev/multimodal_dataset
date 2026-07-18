#!/usr/bin/env python3
"""Build D1_train and D2_train matched sets, excluding D3 human gold IDs.

D1: annotator-only labels from d1_dataset.json
D2: verified approved labels from d2_dataset.json
Both use the same sample_id pool (D2 approved minus D3).

Usage:
  python scripts/build_d1_d2_train.py
  python scripts/build_d1_d2_train.py --d3-file data/gold/d3_dataset.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_D1 = ROOT / "data/raw/hasaki/d1_dataset.json"
DEFAULT_D2 = ROOT / "data/d2_dataset.json"
DEFAULT_D3 = ROOT / "data/gold/d3_dataset.json"
DEFAULT_OUTDIR = ROOT / "data/datasets"


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, dict) and "results" in doc:
        return list(doc["results"])
    if isinstance(doc, list):
        return doc
    raise ValueError(f"Unsupported JSON shape in {path}")


def intent_levels(row: dict[str, Any]) -> tuple[str, str, str]:
    intent = row.get("intent_3_level") or {}
    return (
        str(intent.get("level_1") or ""),
        str(intent.get("level_2") or ""),
        str(intent.get("level_3") or ""),
    )


def slim_row(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    l1, l2, l3 = intent_levels(row)
    return {
        "sample_id": row["sample_id"],
        "sentence": row.get("sentence", ""),
        "category": row.get("category", ""),
        "intent_3_level": {"level_1": l1, "level_2": l2, "level_3": l3},
        "confidence": row.get("confidence"),
        "qa_status": row.get("qa_status"),
        "label_source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-source", type=Path, default=DEFAULT_D1)
    parser.add_argument("--d2-source", type=Path, default=DEFAULT_D2)
    parser.add_argument("--d3-file", type=Path, default=DEFAULT_D3)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    d1_rows = load_results(args.d1_source)
    d2_rows = load_results(args.d2_source)
    d3_rows = load_results(args.d3_file)

    d3_ids = {r["sample_id"] for r in d3_rows if r.get("sample_id")}
    d1_by_id = {r["sample_id"]: r for r in d1_rows if r.get("sample_id")}
    d2_approved = [r for r in d2_rows if r.get("sample_id") and r.get("qa_status") == "approved"]
    d2_by_id = {r["sample_id"]: r for r in d2_approved}

    matched_ids = sorted(set(d2_by_id.keys()) - d3_ids)
    leakage = set(matched_ids) & d3_ids
    if leakage:
        raise RuntimeError(f"D3 leakage in train pool: {len(leakage)} IDs")
    d1_train = [slim_row(d1_by_id[sid], source="D1_annotator") for sid in matched_ids if sid in d1_by_id]
    d2_train = [slim_row(d2_by_id[sid], source="D2_verified") for sid in matched_ids if sid in d2_by_id]
    missing_in_d1 = [sid for sid in matched_ids if sid not in d1_by_id]

    meta = {
        "d1_source": str(args.d1_source.resolve()),
        "d2_source": str(args.d2_source.resolve()),
        "d3_file": str(args.d3_file.resolve()),
        "d3_excluded_count": len(d3_ids),
        "d2_total": len(d2_rows),
        "d2_approved_total": len(d2_by_id),
        "matched_train_ids": len(matched_ids),
        "d1_train_count": len(d1_train),
        "d2_train_count": len(d2_train),
        "missing_in_d1": missing_in_d1,
        "d1_l3_classes": len({intent_levels(r)[2] for r in d1_train}),
        "d2_l3_classes": len({intent_levels(r)[2] for r in d2_train}),
        "d1_l1_distribution": dict(Counter(intent_levels(r)[0] for r in d1_train)),
        "d2_l1_distribution": dict(Counter(intent_levels(r)[0] for r in d2_train)),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "d1_annotator_train.json": {
            "dataset": "D1_annotator_train",
            "description": "GPT-4o-mini annotator labels, matched IDs, D3 excluded",
            "meta": meta,
            "results": d1_train,
        },
        "d2_verified_train.json": {
            "dataset": "D2_verified_train",
            "description": "Verified approved labels, matched IDs, D3 excluded",
            "meta": meta,
            "results": d2_train,
        },
        "dataset_train_meta.json": meta,
    }

    for name, payload in outputs.items():
        path = args.outdir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        n = len(payload.get("results", []))
        print(f"Wrote: {path} ({n if n else 'meta'})")

    print(f"D3 excluded: {len(d3_ids)}")
    print(f"D1_train: {len(d1_train)}, D2_train: {len(d2_train)}")
    if missing_in_d1:
        print(f"Warning: {len(missing_in_d1)} IDs in D2 but missing in D1")


if __name__ == "__main__":
    main()
