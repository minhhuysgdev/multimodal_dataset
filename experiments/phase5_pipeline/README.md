# Phase 5: Annotation Pipeline Baselines

Đánh giá chất lượng annotation với các cấu hình khác nhau của pipeline.

## Baselines

| ID | Method | Description |
|----|--------|-------------|
| M1 | Direct LLM | GPT-4o-mini, zero-shot, no retrieval/verifier/guardrails |
| M2 | Hybrid RAG | R3 candidates + GPT-4o-mini |
| M2-RR | TechniqueRAG-inspired | R4 top-k exemplars + GPT-4o-mini |
| M2-SR | Self-Refine | GPT-4o-mini annotate → self-critique → revise |
| M3 | Full Pipeline | R3 + GPT-4o-mini + GPT-4o verifier + taxonomy guardrails |

## Metrics

### Quality Metrics (L1/L2/L3)
- **Accuracy**: Exact match
- **Macro-F1**: Average F1 across classes (main metric for L3)
- **Weighted-F1**: F1 weighted by class frequency
- **Full-path Accuracy**: All L1+L2+L3 correct

### Pipeline Metrics
- **Invalid Rate**: % predictions không khớp taxonomy
- **Approval Rate**: % được verifier approve (M3 only)
- **Review Rate**: % cần human review (M3 only)
- **API Cost**: USD per sample
- **Latency**: seconds per sample

## Usage

### M1: Direct LLM

```bash
export OPENAI_API_KEY="your-key"
python experiments/phase5_pipeline/run_m1_direct.py
```

### M2: Hybrid RAG

```bash
python experiments/phase5_pipeline/run_m2_rag.py
```

### M2-RR: TechniqueRAG

```bash
python experiments/phase5_pipeline/run_m2_rr_techniquerag.py
```

### M2-SR: Self-Refine

```bash
python experiments/phase5_pipeline/run_m2_sr_selfrefine.py
```

### M3: Full Pipeline

```bash
python experiments/phase5_pipeline/run_m3_full.py
```

### Evaluate All

Compare tất cả baselines trên D3 human gold:

```bash
python experiments/phase5_pipeline/compare_all_methods.py
```

## Output Format

Mọi method đều output cùng schema:

```json
{
  "sample_id": "...",
  "sentence": "...",
  "predicted_l1": "...",
  "predicted_l2": "...",
  "predicted_l3": "...",
  "confidence": 0.95,
  "is_valid_taxonomy": true,
  "api_calls": 1,
  "api_cost_usd": 0.0012,
  "latency_seconds": 1.2,
  "metadata": {
    "retrieval_ids": [...],
    "verifier_action": "approve|revise|needs_review",
    ...
  }
}
```

## Data Sources

- **Test set**: `data/gold/d3_human_gold.json` (300 samples)
- **Retrieval corpus**: `data/datasets/d1_annotator_train.json` (1454 samples)
- **Taxonomy**: `data/unified_intents.csv`
- **Intent KB**: `data/intent_kb.intent_nodes_27_05_2026.csv`, `data/intent_kb.intent_edges_28_05_2026.csv`
