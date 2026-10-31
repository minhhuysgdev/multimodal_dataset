#!/usr/bin/env python3
"""List pool samples not yet in d2_dataset.json (need live LLM annotation)."""
from __future__ import annotations

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from common import D2_DATASET_FILE, load_pool  # noqa: E402
from methods.verified_d2 import get_approved_cache  # noqa: E402


def main():
    pool = load_pool()
    cache = get_approved_cache()
    missing = [s for s in pool if s["sample_id"] not in cache]

    print(f"d2_dataset: {D2_DATASET_FILE}")
    print(f"pool={len(pool)} | in_d2={len(pool) - len(missing)} | missing={len(missing)}\n")

    for i, sample in enumerate(missing, 1):
        sid = sample["sample_id"]
        sentence = sample.get("sentence", "")
        preview = sentence[:120] + ("..." if len(sentence) > 120 else "")
        print(f"{i:2}. {sid}")
        print(f"    {preview}\n")

    out = Path(__file__).resolve().parents[2] / "data/splits/verified_d2_missing_ids.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(s["sample_id"] for s in missing) + ("\n" if missing else ""), encoding="utf-8")
    print(f"Saved IDs -> {out}")


if __name__ == "__main__":
    main()
