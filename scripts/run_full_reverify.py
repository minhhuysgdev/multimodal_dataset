#!/usr/bin/env python3
"""Full reverify all labelled samples (GPT-4o verifier + guardrails, dry-run).

Standalone script — NOT part of intent_labeling_gpt4.ipynb.
Requires MongoDB + OPENAI_API_KEY (same as labeling notebook).

Usage:
  export MONGODB_URI='...'
  export OPENAI_API_KEY='...'
  python scripts/run_full_reverify.py

  python scripts/run_full_reverify.py --input data/raw/hasaki/hasaki_labelled_clean.json
  python scripts/run_full_reverify.py --limit 50          # smoke test
  python scripts/run_full_reverify.py --resume            # continue from checkpoint

Outputs:
  data/raw/hasaki/hasaki_labelled_full_verified.json
  data/raw/hasaki/hasaki_labelled_full_verified.checkpoint.json  (during run)

Then rebuild D2:
  python scripts/build_d2_verified_dataset.py --mode full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/hasaki/hasaki_labelled_clean.json"
DEFAULT_OUTPUT = ROOT / "data/raw/hasaki/hasaki_labelled_full_verified.json"
DEFAULT_CHECKPOINT = ROOT / "data/raw/hasaki/hasaki_labelled_full_verified.checkpoint.json"

_SAMPLE_FIELDS = ("sample_id", "sentence", "category", "product_id", "product_name", "brand")


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, dict) and "results" in doc:
        return list(doc["results"])
    if isinstance(doc, list):
        return doc
    raise ValueError(f"Unsupported JSON shape: {path}")


def intent_from_ann(ann_intent, fallback_intent=None) -> dict:
    fb = fallback_intent or {}
    l3_raw = (ann_intent or {}).get("level_3")
    if isinstance(l3_raw, list):
        l3 = l3_raw[0] if l3_raw else ""
    elif isinstance(l3_raw, str):
        l3 = l3_raw
    else:
        fb3 = fb.get("level_3")
        l3 = fb3[0] if isinstance(fb3, list) and fb3 else (fb3 or "")
    return {
        "level_1": (ann_intent or {}).get("level_1") or fb.get("level_1", ""),
        "level_2": (ann_intent or {}).get("level_2") or fb.get("level_2", ""),
        "level_3": l3 or "",
    }


def pred_from_labelled_item(item: dict) -> dict:
    intent = item.get("intent_3_level") or {}
    l3_val = intent.get("level_3")
    if isinstance(l3_val, list):
        l3_val = l3_val[0] if l3_val else ""
    return {
        "L1": intent.get("level_1", ""),
        "L2": intent.get("level_2", ""),
        "L3": l3_val or "",
        "confidence": float(item.get("confidence") or 0),
        "reasoning": item.get("reasoning", ""),
        "is_new_label": False,
    }


def save_checkpoint(path: Path, results: list, stats: dict, meta: dict) -> None:
    payload = {
        **meta,
        "stats": stats,
        "num_done": len(results),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic rename — tránh corrupt nếu bị kill


def load_checkpoint(path: Path) -> tuple[list, dict, set[str]]:
    if not path.exists():
        return [], {}, set()
    try:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
        results = doc.get("results", [])
        stats = doc.get("stats", {})
        done_ids = {r["sample_id"] for r in results if r.get("sample_id")}
        return results, stats, done_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"WARNING: checkpoint corrupt ({e}) — starting fresh", file=sys.stderr)
        return [], {}, set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit", type=int, default=0, help="Process only N samples (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint file")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.environ.setdefault("INTENT_REPO", str(ROOT))
    sys.path.insert(0, str(ROOT / "data" / "code"))

    try:
        import labeling_pipeline  # noqa: F401 — triggers mongo + openai init as side-effect
    except Exception as e:
        print(
            "ERROR: failed to load labeling_pipeline.py\n"
            "Need MONGODB_URI, OPENAI_API_KEY, and deps (pymongo, openai, sentence-transformers).\n"
            f"Detail: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    from labeling_pipeline import (  # noqa: E402
        ENABLE_VERIFIER,
        VERIFIER_MODEL,
        apply_verifier,
        db,
        enrich_sample,
        save_annotation_with_guardrails,
        union_retrieval,
    )

    all_items = load_results(args.input)
    all_items = [x for x in all_items if x.get("sample_id")]

    results, stats, done_ids = [], {}, set()
    if args.resume:
        results, stats, done_ids = load_checkpoint(args.checkpoint)
        print(f"Resumed: {len(results)} already done")

    todo = [x for x in all_items if x["sample_id"] not in done_ids]
    if args.limit > 0:
        todo = todo[: args.limit]

    print(f"Input: {args.input} ({len(all_items)} total)")
    print(f"To process: {len(todo)}")
    print(f"Verifier: {'ON' if ENABLE_VERIFIER else 'OFF'} | model={VERIFIER_MODEL} | force=True")
    print(f"Output: {args.output}")

    meta = {
        "source_file": str(args.input.resolve()),
        "reverify_scope": "full_all_samples",
        "reverify_config": {
            "enable_verifier": ENABLE_VERIFIER,
            "verifier_model": VERIFIER_MODEL,
            "force_verify": True,
            "script": "scripts/run_full_reverify.py",
        },
    }

    for idx, item in enumerate(todo):
        sample = enrich_sample({k: item.get(k, "") for k in _SAMPLE_FIELDS})
        intent_before = item.get("intent_3_level") or {}
        pred = pred_from_labelled_item(item)
        qa_status_before = item.get("qa_status")
        verify_meta = None
        ann = {}
        status = qa_status_before
        reverify_error = None

        try:
            cands = union_retrieval(
                db,
                sample.get("sentence", ""),
                category=sample.get("category"),
                sample=sample,
            )
            verify_meta = apply_verifier(sample, pred, cands, force=True)
            ann = save_annotation_with_guardrails(
                db, sample, pred, verify_meta=verify_meta, persist=False
            )
            status = ann.get("qa_status", status)
        except Exception as e:
            reverify_error = str(e)
            status = "reverify_error"

        stats[status] = stats.get(status, 0) + 1
        if verify_meta is not None:
            vkey = f"verify_{str(verify_meta.get('decision', '')).lower()}"
            stats[vkey] = stats.get(vkey, 0) + 1

        intent_after = intent_from_ann(ann.get("intent"), fallback_intent=intent_before)

        results.append({
            **item,
            "qa_status_before_reverify": qa_status_before,
            "intent_3_level_before_reverify": intent_before,
            "intent_3_level": intent_after,
            "confidence": ann.get("confidence", pred.get("confidence")),
            "qa_status": status,
            "reasoning": pred.get("reasoning", item.get("reasoning")),
            "verifier_decision": (verify_meta or {}).get("decision"),
            "verifier_reason": (verify_meta or {}).get("reason"),
            "confidence_before_verify": (verify_meta or {}).get("confidence_before_verify"),
            "reverify_error": reverify_error,
        })

        n_done = len(results)
        if (idx + 1) % args.checkpoint_every == 0 or idx == len(todo) - 1:
            save_checkpoint(args.checkpoint, results, stats, meta)
            pct = n_done / len(all_items) * 100
            print(f"  [{pct:.0f}%] checkpoint {n_done}/{len(all_items)} | this run {idx+1}/{len(todo)}")

    out_payload = {
        **meta,
        "num_samples_total": len(all_items),
        "num_samples": len(results),
        "stats": stats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print("\n=== Full reverify stats ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"\nWrote: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print("Next: python scripts/build_d2_verified_dataset.py --mode full")


if __name__ == "__main__":
    main()
