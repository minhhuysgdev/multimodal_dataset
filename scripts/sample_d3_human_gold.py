#!/usr/bin/env python3
"""Sample D3 human gold test set from D2 verified-approved dataset.

Stratified sampling prioritises L2 coverage, then L3 diversity within each L2,
then fills remaining quota proportionally by L2 size.

Usage:
  python scripts/build_d2_verified_dataset.py
  python scripts/sample_d3_human_gold.py --n 300 --seed 42

Outputs:
  data/gold/d3_human_gold_to_annotate.json
  data/gold/d3_human_gold_to_annotate.csv
  data/gold/d3_sampling_meta.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/datasets/d2_verified_approved.json"
DEFAULT_OUTDIR = ROOT / "data/gold"


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


def sample_stratified(
    rows: list[dict[str, Any]],
    n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stratified sample with L2 coverage + L3 diversity."""
    rng = random.Random(seed)
    if n > len(rows):
        raise ValueError(f"Requested n={n} but only {len(rows)} rows available")

    by_l2: dict[str, list[dict]] = defaultdict(list)
    by_l3: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        l1, l2, l3 = intent_levels(r)
        by_l2[l2].append(r)
        by_l3[l3].append(r)

    selected: list[dict] = []
    selected_ids: set[str] = set()

    def pick_one(pool: list[dict]) -> dict | None:
        candidates = [r for r in pool if r["sample_id"] not in selected_ids]
        if not candidates:
            return None
        return rng.choice(candidates)

    l2_keys = sorted(by_l2.keys())
    for l2 in l2_keys:
        if len(selected) >= n:
            break
        if len(l2_keys) <= n:
            row = pick_one(by_l2[l2])
            if row:
                selected.append(row)
                selected_ids.add(row["sample_id"])

    l3_keys = sorted(by_l3.keys(), key=lambda k: len(by_l3[k]))
    for l3 in l3_keys:
        if len(selected) >= n:
            break
        if any(intent_levels(r)[2] == l3 for r in selected):
            continue
        row = pick_one(by_l3[l3])
        if row:
            selected.append(row)
            selected_ids.add(row["sample_id"])

    remaining = n - len(selected)
    if remaining > 0:
        total = len(rows)
        l2_weights = {l2: len(items) / total for l2, items in by_l2.items()}
        pool = [r for r in rows if r["sample_id"] not in selected_ids]
        rng.shuffle(pool)

        targets = {
            l2: max(0, int(round(remaining * w)))
            for l2, w in l2_weights.items()
        }
        while sum(targets.values()) < remaining:
            l2_max = max(l2_weights, key=l2_weights.get)
            targets[l2_max] += 1
        while sum(targets.values()) > remaining:
            l2_min = min(
                (l2 for l2 in targets if targets[l2] > 0),
                key=lambda x: targets[x],
            )
            targets[l2_min] -= 1

        for l2, quota in sorted(targets.items()):
            if quota <= 0:
                continue
            candidates = [
                r for r in by_l2[l2] if r["sample_id"] not in selected_ids
            ]
            rng.shuffle(candidates)
            for row in candidates[:quota]:
                if len(selected) >= n:
                    break
                selected.append(row)
                selected_ids.add(row["sample_id"])

    if len(selected) < n:
        pool = [r for r in rows if r["sample_id"] not in selected_ids]
        rng.shuffle(pool)
        for row in pool[: n - len(selected)]:
            selected.append(row)
            selected_ids.add(row["sample_id"])

    rng.shuffle(selected)

    l1_c = Counter(intent_levels(r)[0] for r in selected)
    l2_c = Counter(intent_levels(r)[1] for r in selected)
    l3_c = Counter(intent_levels(r)[2] for r in selected)

    meta = {
        "source": str(DEFAULT_INPUT),
        "n_requested": n,
        "n_sampled": len(selected),
        "seed": seed,
        "sampling_strategy": "L2_coverage + L3_diversity + proportional_L2_fill",
        "unique_l1": len(l1_c),
        "unique_l2": len(l2_c),
        "unique_l3": len(l3_c),
        "l1_distribution": dict(l1_c),
        "l2_distribution": dict(sorted(l2_c.items(), key=lambda x: -x[1])),
        "l3_count": len(l3_c),
        "source_pool_size": len(rows),
    }
    return selected, meta


def to_template_row(row: dict[str, Any]) -> dict[str, Any]:
    l1, l2, l3 = intent_levels(row)
    return {
        "sample_id": row["sample_id"],
        "sentence": row.get("sentence", ""),
        "category": row.get("category", ""),
        "machine_level_1": l1,
        "machine_level_2": l2,
        "machine_level_3": l3,
        "gold_level_1": "",
        "gold_level_2": "",
        "gold_level_3": "",
        "human_notes": "",
        "qa_status": row.get("qa_status", ""),
        "confidence": row.get("confidence"),
    }


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "sentence",
        "category",
        "machine_level_1",
        "machine_level_2",
        "machine_level_3",
        "gold_level_1",
        "gold_level_2",
        "gold_level_3",
        "human_notes",
        "qa_status",
        "confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(
            f"ERROR: {args.input} not found. Run build_d2_verified_dataset.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = load_results(args.input)
    rows = [r for r in rows if r.get("sample_id")]
    selected, meta = sample_stratified(rows, args.n, args.seed)
    meta["source"] = str(args.input.resolve())
    meta["d2_pipeline"] = True

    template_rows = [to_template_row(r) for r in selected]
    args.outdir.mkdir(parents=True, exist_ok=True)

    json_path = args.outdir / "d3_human_gold_to_annotate.json"
    csv_path = args.outdir / "d3_human_gold_to_annotate.csv"
    meta_path = args.outdir / "d3_sampling_meta.json"

    payload = {
        "dataset": "D3_human_gold_template",
        "purpose": "Human annotation — fill gold_level_1/2/3 columns",
        "do_not_use_for_training": True,
        "meta": meta,
        "results": template_rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    export_csv(csv_path, template_rows)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Sampled: {len(selected)} / pool {len(rows)}")
    print(f"L1: {meta['unique_l1']}, L2: {meta['unique_l2']}, L3: {meta['unique_l3']}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
