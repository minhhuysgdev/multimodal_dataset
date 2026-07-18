#!/usr/bin/env python3
"""
Build data/unified_intents.csv from MongoDB export (intent_nodes CSV).

- Merges level=intent rows by normalized (L1, L2, L3).
- Keeps existing INT-* when a catalog row exists for the triple.
- Assigns INT-133+ for new triples (formerly AUTO-only).
- Normalizes L1: truoc ban / sau ban -> truoc_mua_hang / sau_mua_hang.

Usage:
  python scripts/build_unified_intents_from_mongodb_export.py
  python scripts/build_unified_intents_from_mongodb_export.py --nodes path/to/intent_nodes.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_NODES = REPO / "data/intent_kb.intent_nodes_27_05_2026.csv"
OUT_CSV = REPO / "data/unified_intents.csv"
BACKUP_CSV = REPO / "data/unified_intents.csv.bak"
MAP_CSV = REPO / "data/audit/intent_id_migration_map.csv"

CSV_HEADER = [
    "Intent Name",
    "Confidence Level",
    "Description",
    "Detection Signals",
    "Domain",
    "Examples",
    "L1 Category",
    "L2 Intent",
    "L3 Specific Intent",
    "Product Category",
    "Ma Intent",
    "Logic category san pham",
]

L1_NORM = {
    "truoc_ban": "truoc_mua_hang",
    "truoc mua hang": "truoc_mua_hang",
    "truoc_mua_hang": "truoc_mua_hang",
    "trước_bán": "truoc_mua_hang",
    "trước bán": "truoc_mua_hang",
    "truoc bán": "truoc_mua_hang",
    "sau_ban": "sau_mua_hang",
    "sau mua hang": "sau_mua_hang",
    "sau_mua_hang": "sau_mua_hang",
    "sau bán": "sau_mua_hang",
}


def norm_l1(raw: str) -> str:
    s = (raw or "").strip().lower().replace(" ", "_")
    return L1_NORM.get(s, (raw or "").strip())


def norm_confidence(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return "Cao"
    if s.replace(".", "", 1).isdigit():
        return "Cao" if float(s) >= 0.7 else "Trung binh"
    return s


def int_sort_key(ma: str) -> int:
    m = re.match(r"INT-(\d+)$", ma or "")
    return int(m.group(1)) if m else 10**9


def pick_name(row: dict) -> str:
    name = (row.get("name") or "").strip()
    if name and name not in {(row.get("l3") or ""), (row.get("l2") or "")}:
        return name
    l3 = (row.get("l3") or "").strip()
    return l3.replace("_", " ").title() if l3 else name or "Intent"


def merge_text(*parts: str, sep: str = " | ", max_len: int = 2000) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        p = (p or "").strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    text = sep.join(out)
    return text[:max_len] if len(text) > max_len else text


def row_priority(_id: str) -> tuple[int, int]:
    if _id.startswith("INT-"):
        return (0, int_sort_key(_id))
    if _id.startswith("AUTO-"):
        return (1, 0)
    return (2, 0)


def build_unified(nodes_path: Path) -> tuple[list[dict], list[dict]]:
    with open(nodes_path, newline="", encoding="utf-8") as f:
        intents = [r for r in csv.DictReader(f) if r.get("level") == "intent"]

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in intents:
        l1 = norm_l1(r.get("l1", ""))
        l2 = (r.get("l2") or "").strip()
        l3 = (r.get("l3") or "").strip()
        if not (l1 and l2 and l3):
            continue
        groups[(l1, l2, l3)].append({**r, "l1": l1, "l2": l2, "l3": l3})

    # Reserve Ma Intent for catalog INT rows
    used_int_ids: set[str] = set()
    merged_rows: list[dict] = []
    id_map: list[dict] = []

    for key, members in sorted(groups.items()):
        members.sort(key=lambda r: row_priority(r["_id"]))
        primary = members[0]
        l1, l2, l3 = key

        catalog_int = next((m["_id"] for m in members if m["_id"].startswith("INT-")), "")
        if catalog_int:
            ma_intent = catalog_int
        else:
            ma_intent = ""  # assign below

        descriptions = [m.get("description", "") for m in members]
        signals = [m.get("detection_signals", "") for m in members]
        examples = [m.get("example", "") for m in members]
        logic = [m.get("category_logic", "") for m in members]

        row_out = {
            "Intent Name": pick_name(primary),
            "Confidence Level": norm_confidence(primary.get("confidence", "")),
            "Description": merge_text(*descriptions, sep="\n"),
            "Detection Signals": merge_text(*signals, sep=" | "),
            "Domain": (primary.get("domain") or "My pham").strip(),
            "Examples": merge_text(*examples, sep=" | "),
            "L1 Category": l1,
            "L2 Intent": l2,
            "L3 Specific Intent": l3,
            "Product Category": (primary.get("product_category") or "").strip(),
            "Ma Intent": ma_intent,
            "Logic category san pham": merge_text(*logic, sep=" "),
        }
        merged_rows.append(row_out)

        for m in members:
            id_map.append(
                {
                    "old_id": m["_id"],
                    "new_ma_intent": ma_intent or "(pending)",
                    "l1": l1,
                    "l2": l2,
                    "l3": l3,
                    "merged_into_primary": m["_id"] == primary["_id"],
                }
            )

    # Assign new INT-* for triples without catalog id
    next_num = 133
    for row in merged_rows:
        if row["Ma Intent"]:
            used_int_ids.add(row["Ma Intent"])
            continue
        while f"INT-{next_num}" in used_int_ids:
            next_num += 1
        row["Ma Intent"] = f"INT-{next_num}"
        used_int_ids.add(row["Ma Intent"])
        next_num += 1

    # Fix map pending
    ma_by_triple = {
        (r["L1 Category"], r["L2 Intent"], r["L3 Specific Intent"]): r["Ma Intent"]
        for r in merged_rows
    }
    for entry in id_map:
        key = (entry["l1"], entry["l2"], entry["l3"])
        entry["new_ma_intent"] = ma_by_triple[key]

    merged_rows.sort(key=lambda r: int_sort_key(r["Ma Intent"]))
    return merged_rows, id_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if not args.nodes.exists():
        raise SystemExit(f"Không thấy file nodes: {args.nodes}")

    rows, id_map = build_unified(args.nodes)

    if OUT_CSV.exists() and not args.no_backup:
        shutil.copy2(OUT_CSV, BACKUP_CSV)
        print(f"Backup: {BACKUP_CSV}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)

    MAP_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MAP_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "old_id",
                "new_ma_intent",
                "l1",
                "l2",
                "l3",
                "merged_into_primary",
            ],
        )
        w.writeheader()
        w.writerows(id_map)

    catalog = sum(1 for r in rows if int_sort_key(r["Ma Intent"]) <= 132)
    new_ids = len(rows) - catalog
    auto_mapped = sum(1 for e in id_map if e["old_id"].startswith("AUTO-"))

    print(f"Wrote {len(rows)} intents -> {OUT_CSV}")
    print(f"  Catalog INT (<=132): {catalog}")
    print(f"  New INT (>=133):     {new_ids}")
    print(f"  AUTO rows mapped:    {auto_mapped}")
    print(f"  Migration map:       {MAP_CSV}")
    print(f"  Generated at:        {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
