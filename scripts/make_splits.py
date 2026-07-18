#!/usr/bin/env python3
"""Create stratified train/val/test splits for the approved intent dataset.

Stratifies primarily by L3 intent. Classes with fewer than ``--min-stratify``
samples fall back to L2 as the stratification key (rare L3 singletons are
kept together under their L2 parent). Classes that still have a single sample
are assigned to train only.

Usage:
  python scripts/make_splits.py
  python scripts/make_splits.py \\
    --input data/raw/hasaki/hasaki_labelled_approved_merged.json \\
    --outdir data/splits \\
    --seed 42 \\
    --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

Optional:
  --exclude-ids data/gold/test_gold.json
      Drop sample_ids present in that file (human gold test) before splitting.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/hasaki/hasaki_labelled_approved_merged.json"
DEFAULT_OUTDIR = ROOT / "data/splits"


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, dict) and "results" in doc:
        return list(doc["results"])
    if isinstance(doc, list):
        return doc
    raise ValueError(f"Unsupported JSON shape in {path}")


def load_exclude_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    rows = load_results(path)
    return {r["sample_id"] for r in rows if r.get("sample_id")}


def intent_levels(row: dict[str, Any]) -> tuple[str, str, str]:
    intent = row.get("intent_3_level") or {}
    return (
        str(intent.get("level_1") or ""),
        str(intent.get("level_2") or ""),
        str(intent.get("level_3") or ""),
    )


def stratify_key(row: dict[str, Any], l3_counts: Counter, min_stratify: int) -> str:
    l1, l2, l3 = intent_levels(row)
    if l3 and l3_counts[l3] >= min_stratify:
        return f"L3::{l3}"
    if l2:
        return f"L2::{l2}"
    if l1:
        return f"L1::{l1}"
    return "UNK"


def split_group(
    items: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> tuple[list, list, list]:
    """Split one stratum into train/val/test.

    n == 1 → train only
    n == 2 → train + test
    n >= 3 → proportional split (at least 1 train; val/test get remainder)
    """
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    if n == 1:
        return items, [], []
    if n == 2:
        return [items[0]], [], [items[1]]

    n_train = max(1, int(round(n * train_ratio)))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    # Prefer at least one test when n >= 3 and ratios leave test empty
    if n_test == 0 and n - n_train >= 2:
        n_test = 1
        n_val = n - n_train - n_test
    elif n_test == 0 and n - n_train == 1:
        n_test = 1
        n_val = 0

    # Rebalance if rounding overflowed
    while n_train + n_val + n_test > n:
        if n_val > 0:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            n_train -= 1
    while n_train + n_val + n_test < n:
        n_train += 1

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def make_splits(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_stratify: int,
) -> tuple[list, list, list, dict]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    l3_counts: Counter = Counter(intent_levels(r)[2] for r in rows)
    groups: dict[str, list] = defaultdict(list)
    key_counts: Counter = Counter()
    for r in rows:
        key = stratify_key(r, l3_counts, min_stratify)
        groups[key].append(r)
        key_counts[key] += 1

    rng = random.Random(seed)
    train, val, test = [], [], []
    for key in sorted(groups.keys()):
        g_train, g_val, g_test = split_group(
            groups[key], train_ratio, val_ratio, rng
        )
        train.extend(g_train)
        val.extend(g_val)
        test.extend(g_test)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    meta = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "min_stratify": min_stratify,
        "num_strata": len(groups),
        "strata_key_counts": dict(sorted(key_counts.items(), key=lambda x: -x[1])),
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
        "num_total": len(rows),
        "l1_train": dict(Counter(intent_levels(r)[0] for r in train)),
        "l1_val": dict(Counter(intent_levels(r)[0] for r in val)),
        "l1_test": dict(Counter(intent_levels(r)[0] for r in test)),
        "l3_train": len({intent_levels(r)[2] for r in train}),
        "l3_val": len({intent_levels(r)[2] for r in val}),
        "l3_test": len({intent_levels(r)[2] for r in test}),
    }
    return train, val, test, meta


def dump_split(path: Path, rows: list[dict], split_name: str, meta: dict) -> None:
    payload = {
        "split": split_name,
        "num_samples": len(rows),
        "seed": meta["seed"],
        "results": rows,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--min-stratify",
        type=int,
        default=3,
        help="Min L3 count to stratify by L3; else fall back to L2",
    )
    parser.add_argument(
        "--exclude-ids",
        type=Path,
        default=None,
        help="JSON with results[].sample_id to exclude (e.g. human gold test)",
    )
    args = parser.parse_args()

    rows = load_results(args.input)
    exclude = load_exclude_ids(args.exclude_ids)
    if exclude:
        before = len(rows)
        rows = [r for r in rows if r.get("sample_id") not in exclude]
        print(f"Excluded {before - len(rows)} samples via --exclude-ids")

    # Keep only approved if present
    if any(r.get("qa_status") for r in rows):
        rows = [r for r in rows if r.get("qa_status") == "approved"]

    train, val, test, meta = make_splits(
        rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_stratify=args.min_stratify,
    )
    meta["source_input"] = str(args.input.resolve())
    meta["exclude_ids_file"] = str(args.exclude_ids.resolve()) if args.exclude_ids else None
    meta["num_excluded"] = len(exclude)

    args.outdir.mkdir(parents=True, exist_ok=True)
    dump_split(args.outdir / "train.json", train, "train", meta)
    dump_split(args.outdir / "val.json", val, "val", meta)
    dump_split(args.outdir / "test.json", test, "test", meta)
    with (args.outdir / "split_meta.json").open("w", encoding="utf-8") as f:
        # strata_key_counts can be large; keep it
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Input:  {args.input}")
    print(f"Outdir: {args.outdir}")
    print(f"Total:  {meta['num_total']}")
    print(f"Train:  {meta['num_train']}")
    print(f"Val:    {meta['num_val']}")
    print(f"Test:   {meta['num_test']}")
    print(f"Strata: {meta['num_strata']} (L3 if count>={args.min_stratify}, else L2)")
    print(f"L3 coverage train/val/test: {meta['l3_train']}/{meta['l3_val']}/{meta['l3_test']}")
    print(f"L1 train: {meta['l1_train']}")
    print(f"L1 val:   {meta['l1_val']}")
    print(f"L1 test:  {meta['l1_test']}")
    print(f"Wrote: train.json, val.json, test.json, split_meta.json")


if __name__ == "__main__":
    main()
