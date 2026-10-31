#!/usr/bin/env bash
# Hoàn thiện verified_d2 + chạy eval downstream còn lại
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f env.sh ]]; then
  # shellcheck disable=SC1091
  source env.sh
fi

echo "=== Bước 0: Kiểm tra 32 câu thiếu ==="
python experiments/annotation_benchmark/list_verified_d2_missing.py

echo ""
echo "=== Bước 1: Bootstrap 2011 câu từ d2_dataset.json (không gọi LLM) ==="
python experiments/annotation_benchmark/bootstrap_verified_d2.py

echo ""
echo "=== Bước 2: Annotate 32 câu còn lại (gpt-4o-mini + gpt-4o verifier) ==="
python experiments/annotation_benchmark/run_annotation.py --method verified_d2

echo ""
echo "=== Bước 3: RQ1 — chất lượng annotation trên D3 ==="
python experiments/annotation_benchmark/eval_rq1.py

echo ""
echo "=== Bước 4: Build training datasets ==="
python experiments/annotation_benchmark/build_datasets.py --method all

echo ""
echo "=== Bước 5: RQ2 — downstream (TF-IDF+SVM, PhoBERT) ==="
python experiments/annotation_benchmark/run_downstream.py --method all

echo ""
echo "=== Bước 6: RQ3 — cost analysis ==="
python experiments/annotation_benchmark/eval_rq3_cost.py

echo ""
echo "=== Bước 7: Report tổng hợp ==="
python experiments/annotation_benchmark/report.py

echo ""
echo "Done. verified_d2 hoàn chỉnh 2043/2043."
