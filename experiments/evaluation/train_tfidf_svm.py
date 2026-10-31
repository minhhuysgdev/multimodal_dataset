#!/usr/bin/env python3
"""Train TF-IDF + Linear SVM on D1_train / D2_train, evaluate on D3 human gold.

Usage:
  python experiments/evaluation/train_tfidf_svm.py --train-set D1
  python experiments/evaluation/train_tfidf_svm.py --train-set D2
  python experiments/evaluation/train_tfidf_svm.py --train-set both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "evaluation"))

from utils import (  # noqa: E402
    compute_hierarchical_metrics,
    decode_from_class_probs,
    gold_levels,
    intent_levels,
    load_results,
    path_label,
    rows_with_gold,
    save_json,
    softmax_rows,
)

DEFAULT_D1_TRAIN = ROOT / "data/datasets/d1_annotator_train.json"
DEFAULT_D2_TRAIN = ROOT / "data/datasets/d2_verified_train.json"  # from build_d1_d2_train.py
DEFAULT_D3 = ROOT / "data/gold/d3_dataset.json"
DEFAULT_OUT = ROOT / "experiments/evaluation/results"


def train_and_eval(
    train_rows: list,
    test_rows: list,
    *,
    train_name: str,
    decoding: str = "marginal",
) -> dict:
    import numpy as np

    X_train = [r["sentence"] for r in train_rows]
    y_train = [path_label(*intent_levels(r)) for r in train_rows]
    y_train_l1 = [intent_levels(r)[0] for r in train_rows]

    X_test = [r["sentence"] for r in test_rows]
    y_true_paths = [gold_levels(r) for r in test_rows]
    y_true = [path_label(*p) for p in y_true_paths]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
        ("clf", LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    # Path-level scores → probs (for optional L2/L3 marginal). OvR scores are
    # poorly calibrated with 100s of rare classes, so L1 uses a dedicated head.
    scores = pipe.decision_function(X_test)
    class_labels = list(pipe.named_steps["clf"].classes_)
    if getattr(scores, "ndim", 1) == 1:
        scores = np.column_stack([-scores, scores])
        if len(class_labels) != 2:
            raise RuntimeError("Unexpected binary decision_function shape")
    probs = softmax_rows(scores)
    level_preds, argmax_paths = decode_from_class_probs(
        probs, class_labels, decoding=decoding
    )

    if decoding == "marginal":
        # Dedicated L1 classifier on the same TF-IDF features (2-way, stable).
        X_tr = pipe.named_steps["tfidf"].transform(X_train)
        X_te = pipe.named_steps["tfidf"].transform(X_test)
        l1_clf = LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)
        l1_clf.fit(X_tr, y_train_l1)
        l1_pred = list(l1_clf.predict(X_te))
        # Keep L2/L3 from argmax path (SVM path-marginal is unreliable);
        # still record decoding=marginal because L1 uses the hierarchical head.
        level_preds = [
            (l1_pred[i], argmax_paths[i][1], argmax_paths[i][2])
            for i in range(len(argmax_paths))
        ]

    y_pred_str = [path_label(*p) for p in argmax_paths]
    hier = compute_hierarchical_metrics(
        y_true_paths,
        level_preds,
        full_path_preds=argmax_paths,
    )
    report = classification_report(y_true, y_pred_str, zero_division=0, output_dict=True)

    return {
        "train_set": train_name,
        "model": "TF-IDF + LinearSVC",
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "decoding": decoding,
        "l3_accuracy": hier["l3_accuracy"],
        "l3_macro_f1": hier["l3_macro_f1"],
        "full_path_accuracy": hier["full_path_accuracy"],
        "l1_accuracy": hier["l1_accuracy"],
        "l2_accuracy": hier["l2_accuracy"],
        "classification_report": report,
        "pipeline": pipe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-set",
        choices=["D1", "D2", "both"],
        default="both",
        help="Which training set(s) to evaluate",
    )
    parser.add_argument("--d1-train", type=Path, default=DEFAULT_D1_TRAIN)
    parser.add_argument("--d2-train", type=Path, default=DEFAULT_D2_TRAIN)
    parser.add_argument("--d3-file", type=Path, default=DEFAULT_D3)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    d3_all = load_results(args.d3_file)
    d3_gold = rows_with_gold(d3_all)
    if not d3_gold:
        print(
            "ERROR: D3 has no annotated gold labels. Annotate first, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    results = []
    models_dir = args.outdir / "models" / "tfidf_svm"
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.train_set in ("D1", "both"):
        d1_train = load_results(args.d1_train)
        if not d1_train:
            print(f"WARNING: {args.d1_train} empty — run build_d1_d2_train.py first")
        else:
            r = train_and_eval(d1_train, d3_gold, train_name="D1_annotator_train")
            if args.save_models:
                joblib.dump(r.pop("pipeline"), models_dir / "d1_tfidf_svm.joblib")
            else:
                r.pop("pipeline")
            results.append(r)

    if args.train_set in ("D2", "both"):
        d2_train = load_results(args.d2_train)
        if not d2_train:
            print(f"WARNING: {args.d2_train} empty — run build_d1_d2_train.py first")
        else:
            r = train_and_eval(d2_train, d3_gold, train_name="D2_verified_train")
            if args.save_models:
                joblib.dump(r.pop("pipeline"), models_dir / "d2_tfidf_svm.joblib")
            else:
                r.pop("pipeline")
            results.append(r)

    summary = []
    for r in results:
        summary.append({
            "train_set": r["train_set"],
            "model": r["model"],
            "l3_accuracy": r["l3_accuracy"],
            "l3_macro_f1": r["l3_macro_f1"],
            "full_path_accuracy": r["full_path_accuracy"],
        })
        print(
            f"{r['train_set']}: L3 Acc={r['l3_accuracy']:.4f}, "
            f"Macro-F1={r['l3_macro_f1']:.4f}, "
            f"Full-path={r['full_path_accuracy']:.4f}"
        )

    payload = {"d3_file": str(args.d3_file), "results": results, "summary": summary}
    save_json(args.outdir / "tfidf_svm_metrics.json", payload)
    print(f"Wrote: {args.outdir / 'tfidf_svm_metrics.json'}")


if __name__ == "__main__":
    main()
