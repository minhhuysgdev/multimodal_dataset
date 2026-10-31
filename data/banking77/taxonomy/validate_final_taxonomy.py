#!/usr/bin/env python3
"""Compare the final BANKING77 taxonomy with the refined baseline."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFINED = ROOT / "banking77_taxonomy_refined.csv"
FINAL = ROOT / "banking77_taxonomy_final.csv"
COLUMNS = ["label_id", "L3", "L2", "L1", "flag", "rationale"]

RENAMES = {
    "card_issuance": "card_lifecycle_and_issuance",
    "withdrawal_discrepancy": "withdrawal_issue",
}
ALLOWED_L3 = {
    "card_issuance": {
        "activate_my_card",
        "card_about_to_expire",
        "card_arrival",
        "card_delivery_estimate",
        "card_linking",
        "get_physical_card",
        "getting_spare_card",
        "order_physical_card",
    },
    "withdrawal_discrepancy": {
        "cash_withdrawal_not_recognised",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    },
}


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{path.name}: columns {reader.fieldnames} != {COLUMNS}")
        return list(reader)


def main() -> int:
    errors: list[str] = []
    refined = load(REFINED)
    final = load(FINAL)
    if len(refined) != 77 or len(final) != 77:
        errors.append(f"row counts refined={len(refined)} final={len(final)}")

    refined_by_id = {}
    for row in refined:
        refined_by_id[int(row["label_id"])] = row
    final_by_id = {}
    for index, row in enumerate(final):
        try:
            label_id = int(row["label_id"])
        except ValueError:
            errors.append(f"row {index}: bad label_id {row['label_id']!r}")
            continue
        if label_id in final_by_id:
            errors.append(f"duplicate label_id {label_id}")
        final_by_id[label_id] = row

    if set(final_by_id) != set(range(77)):
        errors.append("final label_id set is not 0-76")
    if set(refined_by_id) != set(range(77)):
        errors.append("refined label_id set is not 0-76")

    modified = []
    unchanged = 0
    rename_counts = {new: 0 for new in RENAMES.values()}
    l2_to_l1: dict[str, set[str]] = defaultdict(set)
    l3_names: list[str] = []

    for label_id in range(77):
        if label_id not in refined_by_id or label_id not in final_by_id:
            continue
        before = refined_by_id[label_id]
        after = final_by_id[label_id]
        l3_names.append(after["L3"])
        l2_to_l1[after["L2"]].add(after["L1"])
        if not after["L1"] or not after["L2"] or not after["L3"]:
            errors.append(f"label_id {label_id} missing L1, L2, or L3")

        diffs = [column for column in COLUMNS if before[column] != after[column]]
        if not diffs:
            unchanged += 1
            continue
        if diffs != ["L2"]:
            errors.append(f"label_id {label_id} changed columns {diffs}")
            continue
        old_l2, new_l2 = before["L2"], after["L2"]
        expected = RENAMES.get(old_l2)
        allowed = ALLOWED_L3.get(old_l2, set())
        if expected is None or new_l2 != expected or after["L3"] not in allowed:
            errors.append(
                f"label_id {label_id} unexpected L2 change {old_l2!r} -> {new_l2!r}"
            )
            continue
        if before["L1"] != after["L1"] or before["L3"] != after["L3"]:
            errors.append(f"label_id {label_id} changed L1 or L3")
            continue
        modified.append(label_id)
        rename_counts[new_l2] += 1

    expected_members = sum(len(names) for names in ALLOWED_L3.values())
    for old_l2, new_l2 in RENAMES.items():
        source_rows = [row for row in refined if row["L2"] == old_l2]
        source_names = {row["L3"] for row in source_rows}
        if source_names != ALLOWED_L3[old_l2]:
            errors.append(f"refined {old_l2} members {sorted(source_names)} differ from spec")
        if any(row["L2"] == old_l2 for row in final):
            errors.append(f"old L2 name still present: {old_l2}")
        if rename_counts[new_l2] != len(ALLOWED_L3[old_l2]):
            errors.append(
                f"{new_l2}: renamed {rename_counts[new_l2]}, expected {len(ALLOWED_L3[old_l2])}"
            )

    l1_values = {parent for parents in l2_to_l1.values() for parent in parents}
    if len(set(l3_names)) != 77:
        errors.append(f"unique L3 count is {len(set(l3_names))}")
    if len(l1_values) != 8:
        errors.append(f"L1 count is {len(l1_values)}: {sorted(l1_values)}")
    if len(l2_to_l1) != 23:
        errors.append(f"L2 count is {len(l2_to_l1)}")
    for l2, parents in l2_to_l1.items():
        if len(parents) != 1:
            errors.append(f"L2 {l2} has L1 parents {sorted(parents)}")
    if len(modified) != expected_members:
        errors.append(f"modified rows {len(modified)} != {expected_members}")
    if unchanged != 77 - expected_members:
        errors.append(f"unchanged rows {unchanged}")

    print("card_lifecycle_and_issuance rows:", rename_counts["card_lifecycle_and_issuance"])
    print("withdrawal_issue rows:", rename_counts["withdrawal_issue"])
    print("modified rows:", len(modified))
    print("unchanged rows:", unchanged)
    print("L1:", len(l1_values))
    print("L2:", len(l2_to_l1))
    print("L3:", len(set(l3_names)))
    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
