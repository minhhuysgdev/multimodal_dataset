#!/usr/bin/env python3
"""CanDist distillation stage (teacher LLM candidates -> PhoBERT student).

Faithful-ish adaptation of CanDist (ACL 2025): the LLM teacher emits candidate
label sets (partial labels); a PhoBERT student is distilled on those partial
labels with PRODEN-style progressive identification, then relabels the whole
pool. No back-translation.

Pipeline:
  1. python run_annotation.py --method candist        # teacher candidates
  2. python distill_candist.py                         # this script
     -> updates data/annotations/candist.json with distilled student labels
        (teacher pick + candidates preserved in metadata)

Usage:
  python experiments/annotation_benchmark/distill_candist.py \
      --model-name vinai/phobert-base --epochs 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]
sys.path.insert(0, str(BENCH_DIR))

from common import (  # noqa: E402
    TRAIN_POOL_IDS,
    annotation_output_path,
    load_id_set,
    load_json,
    save_annotation_run,
)


def _require_torch():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        return torch, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        print(
            "Missing dependencies. Install:\n"
            "  pip install torch transformers",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _pick_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _path_str(label: dict) -> str:
    return f"{label.get('l1', '')}::{label.get('l2', '')}::{label.get('l3', '')}"


def _split_path(path: str) -> tuple[str, str, str]:
    parts = path.split("::", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "", "", path


def _candidate_paths(row: dict) -> list[str]:
    meta = row.get("metadata") or {}
    cands = meta.get("candidates") or []
    paths = []
    for c in cands:
        p = _path_str(c)
        if all(_split_path(p)):
            paths.append(p)
    # Fallback to teacher pick / predicted label
    if not paths:
        p = _path_str(row.get("predicted_label") or {})
        if all(_split_path(p)):
            paths.append(p)
    # Dedup preserving order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="vinai/phobert-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch, AutoModelForSequenceClassification, AutoTokenizer = _require_torch()
    import numpy as np

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    teacher_path = annotation_output_path("candist")
    if not teacher_path.exists():
        raise FileNotFoundError(
            f"Missing {teacher_path}. Run: run_annotation.py --method candist first."
        )

    doc = load_json(teacher_path)
    rows = doc.get("results", [])
    by_id = {r["sample_id"]: r for r in rows}
    train_pool_ids = load_id_set(TRAIN_POOL_IDS)

    # Build label space from candidates over the train pool (no D3).
    label_set = set()
    train_items = []  # (sample_id, sentence, [candidate_paths])
    for sid in sorted(train_pool_ids):
        row = by_id.get(sid)
        if not row:
            continue
        cands = _candidate_paths(row)
        if not cands:
            continue
        train_items.append((sid, row.get("sentence", ""), cands))
        label_set.update(cands)

    labels = sorted(label_set)
    label2id = {l: i for i, l in enumerate(labels)}
    num_labels = len(labels)
    if num_labels == 0 or not train_items:
        raise RuntimeError("No candidate labels found; cannot distill.")

    print(f"Train samples: {len(train_items)} | Label space: {num_labels}")

    device = _pick_device(torch)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    ).to(device)

    # PRODEN confidence weights: uniform over candidate set per sample.
    n = len(train_items)
    confidence = torch.zeros(n, num_labels)
    cand_mask = torch.zeros(n, num_labels, dtype=torch.bool)
    texts = []
    for i, (_sid, sentence, cands) in enumerate(train_items):
        texts.append(sentence)
        idxs = [label2id[c] for c in cands]
        for j in idxs:
            cand_mask[i, j] = True
            confidence[i, j] = 1.0 / len(idxs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def encode(batch_texts):
        return tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )

    order = np.arange(n)
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        running = 0.0
        nb = 0
        update_conf = epoch >= args.warmup_epochs
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            batch_texts = [texts[k] for k in idx]
            enc = encode(batch_texts)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            logp = torch.log_softmax(logits, dim=-1)
            w = confidence[idx].to(device)
            # Weighted partial-label cross-entropy.
            loss = -(w * logp).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            nb += 1

            if update_conf:
                with torch.no_grad():
                    p = torch.softmax(logits, dim=-1).detach().cpu()
                    mask = cand_mask[idx]
                    p = p * mask
                    denom = p.sum(dim=1, keepdim=True)
                    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
                    confidence[idx] = p / denom
        print(f"  epoch {epoch + 1}/{args.epochs} loss={running / max(nb, 1):.4f} conf_update={update_conf}")

    # Predict distilled labels for the ENTIRE pool (incl. D3, which was excluded from training).
    model.eval()
    all_rows = rows
    distilled = 0
    with torch.no_grad():
        for start in range(0, len(all_rows), args.batch_size):
            chunk = all_rows[start : start + args.batch_size]
            enc = encode([r.get("sentence", "") for r in chunk])
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            confs = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
            for r, pid, cf in zip(chunk, pred_ids, confs):
                l1, l2, l3 = _split_path(labels[pid])
                meta = r.get("metadata") or {}
                meta.setdefault("teacher_pick", r.get("predicted_label"))
                meta["strategy"] = "candist_distilled"
                meta["student_model"] = args.model_name
                meta["student_confidence"] = float(cf)
                r["metadata"] = meta
                r["predicted_label"] = {"l1": l1, "l2": l2, "l3": l3}
                r["confidence"] = float(cf)
                distilled += 1

    meta = {k: v for k, v in doc.items() if k not in ("results",)}
    meta["distillation"] = {
        "student_model": args.model_name,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "label_space": num_labels,
        "train_samples": len(train_items),
        "method": "PRODEN_partial_label",
        "distilled_samples": distilled,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_annotation_run("candist", all_rows, meta=meta)
    print(f"Distilled {distilled} samples -> {teacher_path}")


if __name__ == "__main__":
    main()
