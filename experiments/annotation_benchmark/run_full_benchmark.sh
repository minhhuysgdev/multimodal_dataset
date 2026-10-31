#!/usr/bin/env bash
# Full annotation benchmark pipeline
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f env.sh ]]; then
  # shellcheck disable=SC1091
  source env.sh
fi

echo "=== Phase 0: Data prep ==="
python experiments/annotation_benchmark/prepare_data.py

echo "=== Phase 1: Annotations (all methods) ==="
python experiments/annotation_benchmark/run_annotation.py --method all

echo "=== Phase 1b: CanDist distillation (PhoBERT student) ==="
python experiments/annotation_benchmark/distill_candist.py

echo "=== Phase 2: RQ1 ==="
python experiments/annotation_benchmark/eval_rq1.py

echo "=== Phase 3: Build datasets ==="
python experiments/annotation_benchmark/build_datasets.py --method all

echo "=== Phase 4: RQ2 downstream ==="
python experiments/annotation_benchmark/run_downstream.py --method all

echo "=== Phase 5: RQ3 cost ==="
python experiments/annotation_benchmark/eval_rq3_cost.py

echo "=== Report ==="
python experiments/annotation_benchmark/report.py

echo "Done."
