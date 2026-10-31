#!/usr/bin/env python3
"""Bootstrap verified_d2 annotations from d2_dataset.json (no LLM calls).

Copies 2011 labels already produced by the production D2 pipeline
(notebooks/intent_labeling_gpt4.ipynb). The remaining pool samples
(typically 32) must be annotated via run_annotation.py --method verified_d2.
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from common import D2_DATASET_FILE, load_pool, save_annotation_run  # noqa: E402
from methods.verified_d2 import get_approved_cache  # noqa: E402
from retrieval_utils import intent_path  # noqa: E402


def main():
    pool = load_pool()
    cache = get_approved_cache()
    results = []
    cached = 0
    missing_ids: list[str] = []

    for sample in pool:
        sid = sample["sample_id"]
        if sid in cache:
            row = cache[sid]
            l1, l2, l3 = intent_path(row)
            results.append({
                "sample_id": sid,
                "sentence": sample.get("sentence", ""),
                "predicted_label": {"l1": l1, "l2": l2, "l3": l3},
                "confidence": float(row.get("confidence", 0.9)),
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "metadata": {
                    "source": "d2_dataset_cache",
                    "qa_status": row.get("qa_status"),
                    "verifier_decision": row.get("verifier_decision"),
                },
            })
            cached += 1
        else:
            missing_ids.append(sid)

    save_annotation_run(
        "verified_d2",
        results,
        meta={
            "bootstrap": True,
            "cache_source": str(D2_DATASET_FILE),
            "cached_from_d2_dataset": cached,
            "missing_live_annotation": len(missing_ids),
        },
    )

    missing_path = Path(__file__).resolve().parents[2] / "data/splits/verified_d2_missing_ids.txt"
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.write_text("\n".join(sorted(missing_ids)) + ("\n" if missing_ids else ""), encoding="utf-8")

    print(f"verified_d2 bootstrap: cached={cached}, missing={len(missing_ids)}, total_written={len(results)}")
    print(f"Missing IDs: {missing_path}")
    if missing_ids:
        print("Next: python experiments/annotation_benchmark/run_annotation.py --method verified_d2")


if __name__ == "__main__":
    main()
