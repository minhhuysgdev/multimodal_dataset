#!/usr/bin/env python3
"""Validate the refined BANKING77 taxonomy against the baseline inventory."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "banking77_hierarchy.csv"
REFINED = ROOT / "banking77_taxonomy_refined.csv"

EXPECTED_L1 = {
    "card",
    "cash_withdrawal",
    "top_up",
    "transfer",
    "refund_and_charges",
    "account_and_access",
    "currency_exchange",
    "identity_verification",
}

L3_RENAMES: dict[str, str] = {}

COLUMNS = ["label_id", "L3", "L2", "L1", "flag", "rationale"]


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{path.name}: columns {reader.fieldnames} != {COLUMNS}")
        return list(reader)


def main() -> int:
    errors: list[str] = []
    baseline = load(BASELINE)
    refined = load(REFINED)

    if len(refined) != 77:
        errors.append(f"expected 77 rows, found {len(refined)}")

    ids = []
    l3_names = []
    l2_to_l1: dict[str, set[str]] = defaultdict(set)
    l1_to_l2: dict[str, set[str]] = defaultdict(set)
    l2_counts: dict[str, int] = defaultdict(int)

    for row in refined:
        try:
            label_id = int(row["label_id"])
        except ValueError:
            errors.append(f"non-integer label_id: {row['label_id']!r}")
            continue
        l1, l2, l3 = row["L1"].strip(), row["L2"].strip(), row["L3"].strip()
        ids.append(label_id)
        l3_names.append(l3)
        if not l1 or not l2 or not l3:
            errors.append(f"label_id {label_id} is missing L1, L2, or L3")
        l2_to_l1[l2].add(l1)
        l1_to_l2[l1].add(l2)
        l2_counts[l2] += 1

    if len(set(ids)) != len(ids):
        errors.append("duplicate label_id values")
    if set(ids) != set(range(77)):
        missing = sorted(set(range(77)) - set(ids))
        extra = sorted(set(ids) - set(range(77)))
        errors.append(f"label_id coverage failed; missing={missing} extra={extra}")
    if len(set(l3_names)) != len(l3_names):
        errors.append("duplicate L3 names")

    baseline_by_id = {int(row["label_id"]): row for row in baseline}
    refined_by_id = {int(row["label_id"]): row for row in refined if row["label_id"].isdigit()}
    for label_id, source in baseline_by_id.items():
        if label_id not in refined_by_id:
            errors.append(f"missing baseline label_id {label_id}")
            continue
        expected_l3 = L3_RENAMES.get(source["L3"], source["L3"])
        if refined_by_id[label_id]["L3"] != expected_l3:
            errors.append(
                f"label_id {label_id}: L3 {refined_by_id[label_id]['L3']!r} != {expected_l3!r}"
            )
        if refined_by_id[label_id]["L1"] != source["L1"]:
            errors.append(
                f"label_id {label_id}: L1 changed from {source['L1']} to {refined_by_id[label_id]['L1']}"
            )

    if set(l1_to_l2) != EXPECTED_L1:
        errors.append(
            f"L1 set {sorted(l1_to_l2)} != {sorted(EXPECTED_L1)}"
        )
    for l1, families in l1_to_l2.items():
        if not families:
            errors.append(f"orphan L1: {l1}")
    for l2, parents in l2_to_l1.items():
        if len(parents) != 1:
            errors.append(f"L2 {l2} maps to multiple L1 values: {sorted(parents)}")
        if l2_counts[l2] < 1:
            errors.append(f"orphan L2: {l2}")

    print(f"rows: {len(refined)}")
    print(f"unique L3: {len(set(l3_names))}")
    print(f"unique label_id: {len(set(ids))}")
    print(f"L1: {len(l1_to_l2)} {sorted(l1_to_l2)}")
    print(f"L2: {len(l2_to_l1)}")
    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
