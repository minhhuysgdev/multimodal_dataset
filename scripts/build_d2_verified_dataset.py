#!/usr/bin/env python3
"""Build D2 verified-approved dataset.

Modes:
  partial (default) — legacy merge: clean approved (not in reverify queue)
                      + human-review reverified approved (99 samples)
  full              — from full reverify export: qa_status == approved only
                      (requires hasaki_labelled_full_verified.json from notebook §8.7)

Usage:
  # After running notebook §8.7 full reverify:
  python scripts/build_d2_verified_dataset.py --mode full

  # Legacy partial (current 1702 = 1603 + 99):
  python scripts/build_d2_verified_dataset.py --mode partial

Outputs:
  data/datasets/d2_verified_approved.json
  data/datasets/d2_verified_approved.csv
  data/datasets/d2_build_meta.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN = ROOT / "data/raw/hasaki/hasaki_labelled_clean.json"
DEFAULT_REVERIFIED = ROOT / "data/audit/hasaki_labelled_clean_human_review_reverified (1).json"
DEFAULT_FULL_VERIFIED = ROOT / "data/raw/hasaki/hasaki_labelled_full_verified.json"
DEFAULT_OUTDIR = ROOT / "data/datasets"

EXCLUDED_STATUSES = {
    "approved_auto_new_label",
    "needs_review",
    "pending_new_label_review",
    "pending_new_label_invalid_slug",
    "no_intent",
    "reverify_error",
}


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


def build_d2_partial(
    clean_rows: list[dict[str, Any]],
    reverify_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reverify_ids = {r["sample_id"] for r in reverify_rows if r.get("sample_id")}

    from_clean = [
        r for r in clean_rows
        if r.get("qa_status") == "approved"
        and r.get("sample_id")
        and r["sample_id"] not in reverify_ids
    ]
    from_reverified = [
        r for r in reverify_rows
        if r.get("qa_status") == "approved" and r.get("sample_id")
    ]

    by_id: dict[str, dict] = {}
    for r in from_clean:
        by_id[r["sample_id"]] = {**r, "d2_source": "clean_approved"}
    for r in from_reverified:
        by_id[r["sample_id"]] = {**r, "d2_source": "reverified_approved"}

    merged = list(by_id.values())
    meta = {
        "mode": "partial",
        "merge_rule": (
            "clean approved (not in reverify queue) + reverified approved; "
            "reverified wins on overlap"
        ),
        "num_from_clean_approved": len(from_clean),
        "num_from_reverified_approved": len(from_reverified),
        "reverify_queue_size": len(reverify_ids),
        "verifier_coverage": "partial — only 250 human-review queue reverified",
    }
    return merged, meta


def build_d2_full(full_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    approved = [
        {**r, "d2_source": "full_verified_approved"}
        for r in full_rows
        if r.get("qa_status") == "approved" and r.get("sample_id")
    ]
    verified_count = sum(1 for r in full_rows if r.get("verifier_decision"))
    meta = {
        "mode": "full",
        "merge_rule": "qa_status == approved from full reverify export (§8.7)",
        "num_input_total": len(full_rows),
        "num_approved": len(approved),
        "verifier_coverage": "full — all samples passed through GPT-4o verifier",
        "num_with_verifier_decision": verified_count,
        "input_qa_status_distribution": dict(
            Counter(r.get("qa_status") for r in full_rows)
        ),
        "verifier_decision_distribution": dict(
            Counter(r.get("verifier_decision") for r in full_rows if r.get("verifier_decision"))
        ),
    }
    return approved, meta


def enrich_meta(meta: dict, rows: list[dict[str, Any]]) -> dict[str, Any]:
    meta.update({
        "excluded_statuses": sorted(EXCLUDED_STATUSES),
        "num_samples": len(rows),
        "qa_status_distribution": dict(Counter(r.get("qa_status") for r in rows)),
        "d2_source_distribution": dict(Counter(r.get("d2_source") for r in rows)),
        "unique_l1": len({intent_levels(r)[0] for r in rows}),
        "unique_l2": len({intent_levels(r)[1] for r in rows}),
        "unique_l3": len({intent_levels(r)[2] for r in rows}),
        "l1_distribution": dict(Counter(intent_levels(r)[0] for r in rows)),
    })
    return meta


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id", "sentence", "category",
        "level_1", "level_2", "level_3",
        "confidence", "qa_status", "d2_source",
        "verifier_decision",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            l1, l2, l3 = intent_levels(r)
            writer.writerow({
                "sample_id": r["sample_id"],
                "sentence": r.get("sentence", ""),
                "category": r.get("category", ""),
                "level_1": l1, "level_2": l2, "level_3": l3,
                "confidence": r.get("confidence"),
                "qa_status": r.get("qa_status"),
                "d2_source": r.get("d2_source", ""),
                "verifier_decision": r.get("verifier_decision", ""),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["partial", "full"],
        default="partial",
        help="partial=legacy 1603+99; full=from hasaki_labelled_full_verified.json",
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--reverified", type=Path, default=DEFAULT_REVERIFIED)
    parser.add_argument("--full-verified", type=Path, default=DEFAULT_FULL_VERIFIED)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    if args.mode == "full":
        if not args.full_verified.exists():
            print(
                f"ERROR: {args.full_verified} not found.\n"
                "Run notebook intent_labeling_gpt4.ipynb §8.7 (full reverify) first.",
                file=sys.stderr,
            )
            sys.exit(1)
        full_rows = load_results(args.full_verified)
        merged, meta = build_d2_full(full_rows)
        meta["full_verified_source"] = str(args.full_verified.resolve())
    else:
        clean_rows = load_results(args.clean)
        reverify_rows = load_results(args.reverified)
        merged, meta = build_d2_partial(clean_rows, reverify_rows)
        meta["clean_source"] = str(args.clean.resolve())
        meta["reverified_source"] = str(args.reverified.resolve())

    meta = enrich_meta(meta, merged)
    args.outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": "D2_verified_approved",
        "description": "Verified + guardrail-approved hierarchical intent labels",
        "meta": meta,
        "results": merged,
    }

    json_path = args.outdir / "d2_verified_approved.json"
    csv_path = args.outdir / "d2_verified_approved.csv"
    meta_path = args.outdir / "d2_build_meta.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    export_csv(csv_path, merged)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Mode: {args.mode}")
    print(f"D2 samples: {len(merged)}")
    if args.mode == "partial":
        print(f"  from clean approved: {meta.get('num_from_clean_approved')}")
        print(f"  from reverified approved: {meta.get('num_from_reverified_approved')}")
    else:
        print(f"  from full verified input: {meta.get('num_input_total')}")
        print(f"  approved after full verify: {meta.get('num_approved')}")
        print(f"  verifier decisions: {meta.get('verifier_decision_distribution')}")
    print(f"L1/L2/L3: {meta['unique_l1']}/{meta['unique_l2']}/{meta['unique_l3']}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
