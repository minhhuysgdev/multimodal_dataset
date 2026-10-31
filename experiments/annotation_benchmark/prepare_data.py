#!/usr/bin/env python3
"""Phase 0: build annotation pool, D3 val/test split, train pool ids."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRELABEL = ROOT / "data/raw/hasaki/hasaki_prelabel.json"
D3_FILE = ROOT / "data/gold/d3_dataset.json"
OUT_DIR = ROOT / "data/annotations"
SPLITS_DIR = ROOT / "data/splits"


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_ids(path: Path, ids: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prelabel = load_json(PRELABEL)
    d3_doc = load_json(D3_FILE)
    d3_rows = d3_doc["results"] if isinstance(d3_doc, dict) else d3_doc
    d3_ids = {r["sample_id"] for r in d3_rows}

    pool = []
    for row in prelabel:
        pool.append({
            "sample_id": row["sample_id"],
            "sentence": row.get("sentence", ""),
            "category": row.get("category", ""),
            "product_name": row.get("product_name", ""),
        })

    pool_ids = {r["sample_id"] for r in pool}
    missing_d3 = d3_ids - pool_ids
    if missing_d3:
        raise RuntimeError(f"D3 ids missing from prelabel pool: {len(missing_d3)}")

    train_pool_ids = sorted(pool_ids - d3_ids)

    rng = random.Random(args.seed)
    d3_id_list = sorted(d3_ids)
    rng.shuffle(d3_id_list)
    val_ids = sorted(d3_id_list[:100])
    test_ids = sorted(d3_id_list[100:300])

    save_json(OUT_DIR / "pool.json", {
        "source": str(PRELABEL),
        "num_samples": len(pool),
        "num_d3_excluded_from_training": len(d3_ids),
        "num_train_pool": len(train_pool_ids),
        "results": pool,
    })
    save_ids(SPLITS_DIR / "d3_val_ids.txt", val_ids)
    save_ids(SPLITS_DIR / "d3_test_ids.txt", test_ids)
    save_ids(SPLITS_DIR / "train_pool_ids.txt", train_pool_ids)

    meta = {
        "seed": args.seed,
        "num_pool": len(pool),
        "num_d3": len(d3_ids),
        "num_d3_val": len(val_ids),
        "num_d3_test": len(test_ids),
        "num_train_pool": len(train_pool_ids),
        "d3_val_ids_file": str(SPLITS_DIR / "d3_val_ids.txt"),
        "d3_test_ids_file": str(SPLITS_DIR / "d3_test_ids.txt"),
        "train_pool_ids_file": str(SPLITS_DIR / "train_pool_ids.txt"),
        "pool_file": str(OUT_DIR / "pool.json"),
    }
    save_json(SPLITS_DIR / "d3_benchmark_split_meta.json", meta)

    print(f"Pool: {len(pool)} samples -> {OUT_DIR / 'pool.json'}")
    print(f"D3 val/test: {len(val_ids)}/{len(test_ids)}")
    print(f"Train pool (no D3): {len(train_pool_ids)}")
    print(f"Meta: {SPLITS_DIR / 'd3_benchmark_split_meta.json'}")


if __name__ == "__main__":
    main()
