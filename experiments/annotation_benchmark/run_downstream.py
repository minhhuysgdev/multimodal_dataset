#!/usr/bin/env python3
"""RQ2: Downstream classification with ML + DL classifiers.

Classifiers:
  tfidf   - TF-IDF + LinearSVM
  phobert - PhoBERT-base (full fine-tune)
  xlmr    - XLM-RoBERTa-base (full fine-tune; multilingual RoBERTa)
  qwen3   - Qwen3-1.7B + LoRA
  mistral - Mistral-7B-v0.3 + LoRA

Decoding (L1/L2/L3):
  marginal (default) - sum path-class probs per level, then argmax
  argmax             - legacy: parse levels from best full path

Usage:
  python experiments/annotation_benchmark/run_downstream.py --method all
  python experiments/annotation_benchmark/run_downstream.py --method all \\
    --classifiers tfidf,phobert,xlmr,qwen3,mistral --epochs 5

  # Clean-train ablation for RVI-HUL:
  python .../run_downstream.py --method verified_d2_clean --classifiers tfidf

  # Chạy theo đợt, giữ kết quả cũ (--append):
  python .../run_downstream.py --method all --classifiers tfidf,phobert,xlmr --epochs 10
  python .../run_downstream.py --method all --classifiers qwen3 --epochs 8 --append
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(ROOT / "experiments" / "evaluation"))

from common import (  # noqa: E402
    METHODS,
    METHOD_DISPLAY,
    DATASETS_DIR,
    D3_FILE,
    D3_TEST_IDS,
    D3_VAL_IDS,
    RESULTS_DIR,
    load_id_set,
    load_results,
    save_json,
)
from utils import is_valid_intent_row, rows_with_gold  # noqa: E402
from train_tfidf_svm import train_and_eval as train_tfidf  # noqa: E402
from train_transformer import train_and_eval_transformer  # noqa: E402

DEFAULT_CLASSIFIERS = "tfidf,phobert"

# Downstream-only ablation: same clean train file, explicit key only
DOWNSTREAM_EXTRA_METHODS = ["verified_d2_clean"]
DOWNSTREAM_METHODS_ALL = METHODS  # --method all (no duplicate clean row)

DOWNSTREAM_DISPLAY = {
    **METHOD_DISPLAY,
    "verified_d2_clean": "RVI-HUL (approved only)",
}

METHOD_TRAIN_PATH = {
    "verified_d2": DATASETS_DIR / "d2_verified_train.json",
    "verified_d2_clean": DATASETS_DIR / "d2_verified_train.json",
}

TRANSFORMER_CLASSIFIERS: dict[str, dict[str, Any]] = {
    "phobert": {
        "model_name": "vinai/phobert-base",
        "display_name": "PhoBERT (vinai/phobert-base)",
        "use_lora": False,
        "batch_size": 8,
        "eval_batch_size": 8,
        "grad_accum": 1,
    },
    "xlmr": {
        "model_name": "FacebookAI/xlm-roberta-base",
        "display_name": "XLM-RoBERTa (FacebookAI/xlm-roberta-base)",
        "use_lora": False,
        "batch_size": 8,
        "eval_batch_size": 8,
        "grad_accum": 1,
    },
    "qwen3": {
        "model_name": "Qwen/Qwen3-1.7B",
        "display_name": "Qwen3-1.7B-LoRA (Qwen/Qwen3-1.7B)",
        "use_lora": True,
        "batch_size": 2,
        "eval_batch_size": 4,
        "grad_accum": 8,
        "lora_r": 8,
        "lora_alpha": 16,
    },
    "mistral": {
        "model_name": "mistralai/Mistral-7B-v0.3",
        "display_name": "Mistral-7B-LoRA (mistralai/Mistral-7B-v0.3)",
        "use_lora": True,
        "batch_size": 1,
        "eval_batch_size": 2,
        "grad_accum": 8,
        "lora_r": 8,
        "lora_alpha": 16,
    },
}

ALL_CLASSIFIER_KEYS = ["tfidf", *TRANSFORMER_CLASSIFIERS.keys()]


def train_path_for_method(method: str) -> Path:
    if method in METHOD_TRAIN_PATH:
        return METHOD_TRAIN_PATH[method]
    return DATASETS_DIR / f"{method}_dataset.json"


def filter_rows_by_ids(rows: list, id_set: set[str]) -> list:
    return [r for r in rows if r.get("sample_id") in id_set]


def parse_classifiers(raw: str) -> list[str]:
    keys = [k.strip().lower() for k in raw.split(",") if k.strip()]
    unknown = [k for k in keys if k not in ALL_CLASSIFIER_KEYS]
    if unknown:
        raise ValueError(
            f"Unknown classifiers: {unknown}. "
            f"Choose from: {', '.join(ALL_CLASSIFIER_KEYS)}"
        )
    return keys


RESULT_FIELDNAMES = [
    "training_dataset",
    "method_key",
    "classifier",
    "n_train",
    "n_train_invalid",
    "train_invalid_rate",
    "n_val",
    "n_test",
    "decoding",
    "use_l1_head",
    "lambda_l1",
    "l1_accuracy",
    "l2_accuracy",
    "l3_accuracy",
    "l3_macro_f1",
    "full_path_accuracy",
]


def _row_key(row: dict) -> tuple[str, str]:
    return (str(row.get("method_key", "")), str(row.get("classifier", "")))


def load_existing_results(outdir: Path) -> tuple[list[dict], list[str]]:
    """Load prior downstream rows + classifier keys from JSON (fallback CSV)."""
    json_path = outdir / "downstream_performance.json"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                rows = payload.get("results") or []
                clfs = payload.get("classifiers") or []
                if isinstance(rows, list):
                    return (
                        [r for r in rows if isinstance(r, dict)],
                        [str(c) for c in clfs],
                    )
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: cannot read {json_path}: {e}")

    csv_path = outdir / "downstream_performance.csv"
    if csv_path.exists():
        try:
            with csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            return rows, []
        except OSError as e:
            print(f"WARNING: cannot read {csv_path}: {e}")

    return [], []


def merge_results(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """Keep old rows; replace (method_key, classifier) when re-run."""
    by_key: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for row in existing:
        key = _row_key(row)
        if key not in by_key:
            order.append(key)
        by_key[key] = row
    for row in new_rows:
        key = _row_key(row)
        if key not in by_key:
            order.append(key)
        by_key[key] = row
    return [by_key[k] for k in order]


def write_downstream_outputs(
    outdir: Path,
    summary_rows: list[dict],
    classifier_keys: list[str],
    epochs: int,
    decoding: str,
    *,
    use_l1_head: bool = True,
    lambda_l1: float = 1.0,
) -> Path:
    csv_path = outdir / "downstream_performance.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    save_json(
        outdir / "downstream_performance.json",
        {
            "classifiers": classifier_keys,
            "epochs": epochs,
            "decoding": decoding,
            "use_l1_head": use_l1_head,
            "lambda_l1": lambda_l1,
            "results": summary_rows,
        },
    )
    return csv_path


def _result_row(
    method: str,
    classifier_name: str,
    result: dict,
    n_val: int,
    *,
    n_train_invalid: int = 0,
    train_invalid_rate: float = 0.0,
    decoding: str = "marginal",
    use_l1_head: bool = True,
    lambda_l1: float = 1.0,
) -> dict:
    is_tfidf = "TF-IDF" in classifier_name or classifier_name.lower() == "tfidf"
    row_use_l1 = result.get("use_l1_head", use_l1_head if not is_tfidf else True)
    row_lambda = result.get(
        "lambda_l1",
        lambda_l1 if not is_tfidf else 0.0,
    )
    return {
        "training_dataset": DOWNSTREAM_DISPLAY.get(method, method),
        "method_key": method,
        "classifier": classifier_name,
        "n_train": result.get("n_train", 0),
        "n_train_invalid": n_train_invalid,
        "train_invalid_rate": round(float(train_invalid_rate), 6),
        "n_val": n_val,
        "n_test": result.get("n_test", 0),
        "decoding": result.get("decoding", decoding),
        "use_l1_head": row_use_l1,
        "lambda_l1": row_lambda,
        "l1_accuracy": result.get("l1_accuracy", 0),
        "l2_accuracy": result.get("l2_accuracy", 0),
        "l3_accuracy": result.get("l3_accuracy", 0),
        "l3_macro_f1": result.get("l3_macro_f1", 0),
        "full_path_accuracy": result.get("full_path_accuracy", 0),
    }


def filter_valid_train(rows: list) -> tuple[list, int, float]:
    """Drop invalid taxonomy labels; return (valid_rows, n_invalid, invalid_rate)."""
    valid: list = []
    n_invalid = 0
    for r in rows:
        if is_valid_intent_row(r):
            valid.append(r)
        else:
            n_invalid += 1
    total = len(rows)
    rate = (n_invalid / total) if total else 0.0
    return valid, n_invalid, rate


def run_tfidf(
    method: str,
    train_rows: list,
    test_rows: list,
    n_val: int,
    *,
    n_train_invalid: int,
    train_invalid_rate: float,
    decoding: str,
    use_l1_head: bool,
    lambda_l1: float,
) -> dict:
    result = train_tfidf(
        train_rows,
        test_rows,
        train_name=f"{method}_tfidf_svm",
        decoding=decoding,
    )
    row = _result_row(
        method,
        "TF-IDF+SVM",
        result,
        n_val,
        n_train_invalid=n_train_invalid,
        train_invalid_rate=train_invalid_rate,
        decoding=decoding,
        use_l1_head=use_l1_head,
        lambda_l1=lambda_l1,
    )
    print(
        f"{method} TF-IDF: L1={row['l1_accuracy']:.4f} "
        f"L3={row['l3_accuracy']:.4f} Full={row['full_path_accuracy']:.4f}"
    )
    return row


def run_transformer(
    method: str,
    clf_key: str,
    cfg: dict[str, Any],
    train_rows: list,
    test_rows: list,
    n_val: int,
    epochs: int,
    outdir: Path,
    *,
    n_train_invalid: int,
    train_invalid_rate: float,
    decoding: str,
    use_l1_head: bool,
    lambda_l1: float,
) -> dict:
    result = train_and_eval_transformer(
        train_rows,
        test_rows,
        model_name=cfg["model_name"],
        train_name=f"{method}_{clf_key}",
        epochs=epochs,
        batch_size=cfg["batch_size"],
        output_dir=outdir / "models" / method / clf_key,
        use_lora=cfg.get("use_lora", False),
        lora_r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        gradient_accumulation_steps=cfg.get("grad_accum", 1),
        eval_batch_size=cfg.get("eval_batch_size"),
        decoding=decoding,
        use_l1_head=use_l1_head,
        lambda_l1=lambda_l1,
    )
    row = _result_row(
        method,
        cfg["display_name"],
        result,
        n_val,
        n_train_invalid=n_train_invalid,
        train_invalid_rate=train_invalid_rate,
        decoding=decoding,
        use_l1_head=use_l1_head,
        lambda_l1=lambda_l1,
    )
    print(
        f"{method} {clf_key}: L1={row['l1_accuracy']:.4f} "
        f"L3={row['l3_accuracy']:.4f} Full={row['full_path_accuracy']:.4f}"
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=DOWNSTREAM_METHODS_ALL + DOWNSTREAM_EXTRA_METHODS + ["all"],
        default="all",
    )
    parser.add_argument(
        "--classifiers",
        default=DEFAULT_CLASSIFIERS,
        help=f"Comma-separated list: {', '.join(ALL_CLASSIFIER_KEYS)}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs for transformer classifiers (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override batch size for all transformer classifiers (0=use per-model default)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=0,
        help="Override gradient accumulation for transformer classifiers "
        "(0=use per-model default)",
    )
    parser.add_argument(
        "--decoding",
        choices=["argmax", "marginal"],
        default="marginal",
        help="Level decoding for L1/L2/L3 (default: marginal)",
    )
    parser.add_argument(
        "--l1-head",
        action="store_true",
        default=True,
        help="Use dedicated L1 head for transformers (default: on)",
    )
    parser.add_argument(
        "--no-l1-head",
        action="store_true",
        help="Disable transformer L1 head (path-only)",
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=1.0,
        help="Weight for L1 auxiliary loss (transformers only)",
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Deprecated: same as --classifiers tfidf",
    )
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge new results into existing downstream_performance.* "
        "(replace same method_key+classifier; keep the rest)",
    )
    args = parser.parse_args()
    use_l1_head = not args.no_l1_head

    if args.skip_transformer:
        classifier_keys = ["tfidf"]
    else:
        classifier_keys = parse_classifiers(args.classifiers)

    val_ids = load_id_set(D3_VAL_IDS)
    test_ids = load_id_set(D3_TEST_IDS)
    d3_all = rows_with_gold(load_results(D3_FILE))
    val_rows = filter_rows_by_ids(d3_all, val_ids)
    test_rows = filter_rows_by_ids(d3_all, test_ids)

    if not val_rows or not test_rows:
        raise RuntimeError("D3 val/test split missing or empty")

    if args.method == "all":
        methods = DOWNSTREAM_METHODS_ALL
    else:
        methods = [args.method]
    args.outdir.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    existing_clfs: list[str] = []
    if args.append:
        existing_rows, existing_clfs = load_existing_results(args.outdir)
        print(f"Append mode: loaded {len(existing_rows)} existing result row(s)")

    print(f"Decoding: {args.decoding}")
    print(f"L1 head: {use_l1_head} (lambda_l1={args.lambda_l1})")
    summary_rows: list[dict] = []

    for method in methods:
        train_path = train_path_for_method(method)
        if not train_path.exists():
            print(f"WARNING: missing {train_path}, skip {method}")
            continue

        raw_rows = load_results(train_path)
        train_rows, n_invalid, invalid_rate = filter_valid_train(raw_rows)
        print(
            f"{method}: train={len(train_rows)} "
            f"(invalid dropped={n_invalid}, rate={invalid_rate:.4f}) "
            f"from {train_path.name}"
        )
        if not train_rows:
            print(f"WARNING: no valid train rows for {method}, skip")
            continue

        for clf_key in classifier_keys:
            if clf_key == "tfidf":
                summary_rows.append(
                    run_tfidf(
                        method,
                        train_rows,
                        test_rows,
                        len(val_rows),
                        n_train_invalid=n_invalid,
                        train_invalid_rate=invalid_rate,
                        decoding=args.decoding,
                        use_l1_head=use_l1_head,
                        lambda_l1=args.lambda_l1,
                    )
                )
                continue

            cfg = dict(TRANSFORMER_CLASSIFIERS[clf_key])
            if args.batch_size > 0:
                cfg["batch_size"] = args.batch_size
                cfg["eval_batch_size"] = max(1, args.batch_size)
            if args.grad_accum > 0:
                cfg["grad_accum"] = args.grad_accum

            summary_rows.append(
                run_transformer(
                    method,
                    clf_key,
                    cfg,
                    train_rows,
                    test_rows,
                    len(val_rows),
                    args.epochs,
                    args.outdir,
                    n_train_invalid=n_invalid,
                    train_invalid_rate=invalid_rate,
                    decoding=args.decoding,
                    use_l1_head=use_l1_head,
                    lambda_l1=args.lambda_l1,
                )
            )

    if args.append:
        summary_rows = merge_results(existing_rows, summary_rows)
        seen: set[str] = set()
        merged_clfs: list[str] = []
        for c in [*existing_clfs, *classifier_keys]:
            if c not in seen:
                seen.add(c)
                merged_clfs.append(c)
        classifier_keys = merged_clfs

    csv_path = write_downstream_outputs(
        args.outdir,
        summary_rows,
        classifier_keys,
        args.epochs,
        args.decoding,
        use_l1_head=use_l1_head,
        lambda_l1=args.lambda_l1,
    )
    print(f"\nWrote: {csv_path} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
