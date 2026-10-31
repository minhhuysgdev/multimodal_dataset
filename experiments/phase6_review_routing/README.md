# Phase 6: Review-Routing and Cost Baselines

So sánh các chiến lược phân bổ human review với cùng budget.

## Baselines

| ID | Method | Description |
|----|--------|-------------|
| B1 | Random | Random selection for human review |
| B2 | Confidence | Low confidence → review |
| B3 | Entropy (CoAnnotating) | K runs, majority vote, high entropy → review |
| B4 | Ours | Verifier + guardrails → review queue |

## Setup

Tất cả methods chạy trên cùng annotation pool (không chứa D3) với cùng review budget:
- b ∈ {20%, 40%, 60%}

Human labels: Oracle simulation (dùng ground truth từ D2 verified hoặc human annotation)

## Metrics

### Label Quality Post-Review
- Accuracy L1/L2/L3
- Macro-F1 L3 (primary)
- Full-path accuracy

### Review Efficiency
- Review rate (% samples sent to review)
- Human effort (# samples reviewed)
- API calls & cost

### Downstream Performance
- Train TF-IDF SVM classifier on labeled data
- Evaluate Macro-F1 on held-out test set

## Usage

### B1: Random Allocation

```bash
python experiments/phase6_review_routing/run_b1_random.py --budget 0.2
python experiments/phase6_review_routing/run_b1_random.py --budget 0.4
python experiments/phase6_review_routing/run_b1_random.py --budget 0.6
```

### B2: Confidence Threshold

```bash
python experiments/phase6_review_routing/run_b2_confidence.py --budget 0.2
```

### B3: Entropy (CoAnnotating)

```bash
python experiments/phase6_review_routing/run_b3_entropy.py --budget 0.2 --K 3
```

### B4: Ours (Framework)

```bash
python experiments/phase6_review_routing/run_b4_framework.py --budget 0.2
```

### Compare All

```bash
python experiments/phase6_review_routing/compare_routing_methods.py
```

## Data Sources

- **Annotation pool**: `data/splits/train.json` (989 samples, D3 excluded)
- **Oracle labels**: Ground truth from D2 verified or manual annotation
- **Test set for downstream**: `data/splits/test.json` (225 samples)

## Output Schema

```json
{
  "method": "B1_random",
  "budget": 0.2,
  "num_samples": 989,
  "num_reviewed": 198,
  "review_rate": 0.2,
  "quality_metrics": {
    "accuracy_l3": 0.95,
    "macro_f1_l3": 0.87,
    ...
  },
  "downstream_metrics": {
    "test_macro_f1": 0.82
  },
  "cost_metrics": {
    "total_api_cost_usd": 15.20,
    "total_api_calls": 1200
  }
}
```
