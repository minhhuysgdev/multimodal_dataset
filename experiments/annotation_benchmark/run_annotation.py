#!/usr/bin/env python3
"""Run LLM annotation methods on the full Hasaki prelabel pool.

Usage:
  export OPENAI_API_KEY="..."
  python experiments/annotation_benchmark/run_annotation.py --method zero_shot
  python experiments/annotation_benchmark/run_annotation.py --method all
  python experiments/annotation_benchmark/run_annotation.py --method cot --limit 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]
sys.path.insert(0, str(BENCH_DIR))

from common import (  # noqa: E402
    METHODS,
    annotation_output_path,
    load_existing_annotations,
    load_pool,
    save_annotation_run,
    LLMClient,
    TaxonomyIndex,
    DemoPool,
)
from methods import candist, cot, few_shot, freeal, verified_d2, zero_shot  # noqa: E402


METHOD_MODULES = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "cot": cot,
    "freeal": freeal,
    "candist": candist,
    "verified_d2": verified_d2,
}


def run_method(method: str, limit: int = 0, resume: bool = True, mock_llm: bool = False):
    if method not in METHOD_MODULES:
        raise ValueError(f"Unknown method: {method}. Choose from {METHODS}")

    if not mock_llm and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (use --mock-llm for offline pipeline test)")

    mod = METHOD_MODULES[method]
    pool = load_pool(limit=limit)
    existing = load_existing_annotations(method) if resume else {}
    client = LLMClient(mock=mock_llm)
    taxonomy = TaxonomyIndex()
    demo_pool = DemoPool(seed=42)
    approved_cache = verified_d2.get_approved_cache() if method == "verified_d2" else None

    print(f"=== Annotation: {method} ===")
    print(f"Pool size: {len(pool)} | Cached: {len(existing)} | Resume: {resume}")

    results = []
    start_all = time.time()
    new_count = 0

    for i, sample in enumerate(pool):
        sid = sample["sample_id"]
        if sid in existing:
            results.append(existing[sid])
            continue

        if i % 25 == 0:
            print(f"  Progress: {i}/{len(pool)} (new={new_count})")

        kwargs = {
            "sample": sample,
            "client": client,
            "taxonomy": taxonomy,
            "demo_pool": demo_pool,
        }
        if method == "verified_d2":
            kwargs["approved_cache"] = approved_cache

        result = mod.annotate_sample(**kwargs)
        row = result.to_dict()
        results.append(row)
        existing[sid] = row
        new_count += 1

        # Checkpoint every 50 new samples
        if new_count % 50 == 0:
            save_annotation_run(
                method,
                results,
                meta={
                    "model": client.model,
                    "checkpoint": True,
                    "completed": len(results),
                },
            )

    elapsed = time.time() - start_all
    total_calls = sum(r.get("api_calls", 0) for r in results)
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    total_in = sum(r.get("input_tokens", 0) for r in results)
    total_out = sum(r.get("output_tokens", 0) for r in results)

    meta = {
        "model": client.model,
        "num_samples": len(results),
        "new_annotations": new_count,
        "total_api_calls": total_calls,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": total_cost,
        "elapsed_seconds": elapsed,
    }
    save_annotation_run(method, results, meta=meta)

    print(f"Done: {method}")
    print(f"  Samples: {len(results)} | New: {new_count}")
    print(f"  API calls: {total_calls} | Cost: ${total_cost:.4f}")
    print(f"  Saved: {annotation_output_path(method)}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=METHODS + ["all"],
        help="Annotation method to run",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit pool size (0=all)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output")
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Offline deterministic mock LLM (pipeline testing only)",
    )
    args = parser.parse_args()

    if args.method == "all":
        for method in METHODS:
            run_method(
                method,
                limit=args.limit,
                resume=not args.no_resume,
                mock_llm=args.mock_llm,
            )
    else:
        run_method(
            args.method,
            limit=args.limit,
            resume=not args.no_resume,
            mock_llm=args.mock_llm,
        )


if __name__ == "__main__":
    main()
