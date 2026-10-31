#!/usr/bin/env python3
"""Generate paper tables and figures from benchmark results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from common import RESULTS_DIR, load_json, save_json  # noqa: E402


def _load_csv_as_rows(path: Path) -> list[dict]:
    import csv

    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_figures(fig_dir: Path, rq1_rows, rq2_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping figures")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    if rq1_rows:
        methods = [r["method"] for r in rq1_rows]
        l3_acc = [float(r["l3_accuracy"]) for r in rq1_rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(methods, l3_acc, color="#4C72B0")
        ax.set_ylabel("L3 Accuracy")
        ax.set_title("RQ1: Annotation Quality vs D3 Gold")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        fig.savefig(fig_dir / "fig_rq1_l3_accuracy.png", dpi=150)
        plt.close(fig)

    if rq2_rows:
        phobert_rows = [r for r in rq2_rows if "PhoBERT" in r.get("classifier", "")]
        if phobert_rows:
            methods = [r["training_dataset"] for r in phobert_rows]
            full_acc = [float(r["full_path_accuracy"]) for r in phobert_rows]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(methods, full_acc, color="#55A868")
            ax.set_ylabel("Full-path Accuracy")
            ax.set_title("RQ2: Downstream PhoBERT on D3 Test")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            fig.savefig(fig_dir / "fig_rq2_downstream_fullpath.png", dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.outdir / "figures"

    rq1_rows = _load_csv_as_rows(args.outdir / "annotation_quality.csv")
    rq2_rows = _load_csv_as_rows(args.outdir / "downstream_performance.csv")
    rq3_rows = _load_csv_as_rows(args.outdir / "annotation_cost.csv")

    report = {
        "table1_rq1_annotation_quality": rq1_rows,
        "table2_rq2_downstream_performance": rq2_rows,
        "table3_rq3_annotation_cost": rq3_rows,
    }
    save_json(args.outdir / "benchmark_report.json", report)

    make_figures(fig_dir, rq1_rows, rq2_rows)

    print("=== Benchmark Report ===")
    print(f"RQ1 rows: {len(rq1_rows)}")
    print(f"RQ2 rows: {len(rq2_rows)}")
    print(f"RQ3 rows: {len(rq3_rows)}")
    print(f"Report: {args.outdir / 'benchmark_report.json'}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
