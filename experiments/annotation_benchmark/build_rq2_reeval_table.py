#!/usr/bin/env python3
"""Build ranked RQ2 paper tables from downstream_v2_reeval CSV/JSON."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from common import save_json  # noqa: E402

DEFAULT_IN = BENCH_DIR / "results" / "downstream_v2_reeval" / "downstream_performance.csv"
METRIC_COLS = [
    "l1_accuracy",
    "l2_accuracy",
    "l3_accuracy",
    "l3_macro_f1",
    "full_path_accuracy",
]
CLASSIFIER_ORDER = [
    "TF-IDF+SVM",
    "PhoBERT",
    "XLM-RoBERTa",
    "Qwen3",
    "Mistral",
]


def _clf_sort_key(name: str) -> tuple[int, str]:
    for i, token in enumerate(CLASSIFIER_ORDER):
        if token.lower() in name.lower():
            return (i, name)
    return (99, name)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rank_rows(rows: list[dict], metric: str) -> list[dict]:
    def score(row: dict) -> float:
        try:
            return float(row.get(metric, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    return sorted(rows, key=score, reverse=True)


def group_by_classifier(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("classifier", "unknown")].append(row)
    return dict(groups)


def build_summary(rows: list[dict]) -> dict:
    by_clf = group_by_classifier(rows)
    ranked: dict[str, dict] = {}
    for clf, clf_rows in sorted(by_clf.items(), key=lambda kv: _clf_sort_key(kv[0])):
        ranked[clf] = {
            metric: [
                {
                    "method_key": r.get("method_key"),
                    "training_dataset": r.get("training_dataset"),
                    metric: float(r.get(metric, 0) or 0),
                    "n_train": int(float(r.get("n_train", 0) or 0)),
                }
                for r in rank_rows(clf_rows, metric)[:6]
            ]
            for metric in METRIC_COLS
        }
    return {
        "n_rows": len(rows),
        "classifiers": sorted(by_clf.keys(), key=_clf_sort_key),
        "ranked_by_classifier": ranked,
        "rows": rows,
    }


def to_markdown(summary: dict) -> str:
    lines = ["# RQ2 downstream v2 re-eval", "", f"Rows: {summary['n_rows']}", ""]
    for clf in summary["classifiers"]:
        lines.append(f"## {clf}")
        lines.append("")
        lines.append(
            "| Rank | Method | L1 | L2 | L3 | Macro-F1 | Full-path | n_train |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        primary = summary["ranked_by_classifier"][clf]["l3_accuracy"]
        for i, entry in enumerate(primary, 1):
            row = next(
                (
                    r
                    for r in summary["rows"]
                    if r.get("method_key") == entry["method_key"]
                    and r.get("classifier") == clf
                ),
                {},
            )
            lines.append(
                f"| {i} | {entry.get('training_dataset', entry.get('method_key'))} | "
                f"{float(row.get('l1_accuracy', 0)):.3f} | "
                f"{float(row.get('l2_accuracy', 0)):.3f} | "
                f"{float(row.get('l3_accuracy', 0)):.3f} | "
                f"{float(row.get('l3_macro_f1', 0)):.3f} | "
                f"{float(row.get('full_path_accuracy', 0)):.3f} | "
                f"{entry.get('n_train', '')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_IN,
        help="Input downstream_performance.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output dir (default: same dir as CSV)",
    )
    args = parser.parse_args()
    outdir = args.outdir or args.csv.parent
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.csv)
    if not rows:
        print(f"No rows in {args.csv}")
        raise SystemExit(1)

    summary = build_summary(rows)
    json_path = outdir / "rq2_reeval_summary.json"
    md_path = outdir / "rq2_reeval_table.md"
    save_json(json_path, summary)
    md_path.write_text(to_markdown(summary), encoding="utf-8")

    print(f"Rows: {len(rows)}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    for clf in summary["classifiers"]:
        top = summary["ranked_by_classifier"][clf]["l3_accuracy"][:1]
        if top:
            t = top[0]
            print(
                f"  {clf}: best L3={t['l3_accuracy']:.4f} "
                f"({t.get('method_key')})"
            )


if __name__ == "__main__":
    main()
