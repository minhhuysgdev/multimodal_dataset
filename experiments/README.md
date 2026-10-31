# Experiments Index

Chỉ mục notebook / script theo từng phase của
[`docs/research_roadmap.md`](../docs/research_roadmap.md).

Hướng paper: **Verification-Guided LLM Annotation Framework** (dataset construction & QA).  
Hướng retrieval representation cũ: [`docs/paper_plan.md`](../docs/paper_plan.md) — không dùng thư mục này.

---

## Shared data

| Artifact | Path | Status |
|----------|------|--------|
| Approved gold (merged) | `data/raw/hasaki/hasaki_labelled_approved_merged.json` | Ready (1702) |
| Clean labelled | `data/raw/hasaki/hasaki_labelled_clean.json` | Ready (2011) |
| Full verified (all samples) | `data/raw/hasaki/hasaki_labelled_full_verified.json` | **TODO** Phase 0 |
| Human gold test | `data/gold/test_gold.json` | **TODO** Phase 0 |
| Train/val/test splits | `data/splits/{train,val,test}.json` | Run `scripts/make_splits.py` |
| Taxonomy | `data/unified_intents.csv` | Ready |
| Intent graph | `data/intent_kb.intent_nodes_*.csv`, `intent_edges_*.csv` | Ready |

```bash
# Freeze splits from current approved gold
python scripts/make_splits.py \
  --input data/raw/hasaki/hasaki_labelled_approved_merged.json \
  --outdir data/splits \
  --seed 42

# After human gold exists, exclude those IDs from train/val/test pool:
# python scripts/make_splits.py --exclude-ids data/gold/test_gold.json
```

---

## Phase 0 — Data completion

| Item | Path | Status |
|------|------|--------|
| Labeling + verify pipeline | `data/code/intent_labeling_gpt4.ipynb` | Ready |
| Verifier design doc | `docs/verifier_cross_verification_plan.md` | Ready |
| Verifier report | `scripts/generate_verifier_report.py` | Ready |
| Full verify export | `data/raw/hasaki/hasaki_labelled_full_verified.json` | TODO |
| Dataset A/B construction | *(script TBD)* | TODO |
| Human gold test | `data/gold/test_gold.json` | TODO |

**Checklist:**

1. `VERIFY_ALL_SAMPLES=1` trên toàn bộ ~2000 mẫu
2. Export full-verified JSON (annotator label + verifier fields + `qa_status`)
3. Build Dataset A (annotator-only) / B (verified approved), same `sample_id`
4. Human-label 250–300 stratified samples → `data/gold/test_gold.json`
5. Freeze splits (exclude gold IDs)

---

## Phase 1 — Dataset analysis

| Item | Path | Status |
|------|------|--------|
| Evaluation notebook | `notebooks/dataset_evaluation.ipynb` | Partial (pointed at full, not approved) |
| Intent distribution | `data/audit/intent_distribution.csv` | Partial |
| Updated stats on approved / full-verified | `data/audit/` | TODO |
| Class distribution plots | `data/audit/*.png` | Partial |

**TODO:** Point notebook at `hasaki_labelled_approved_merged.json` (and later full-verified); export paper-ready tables.

---

## Phase 2 — Retrieval evaluation

| Item | Path | Status |
|------|------|--------|
| Retrieval / embedding source | `intent_graph_rag_colab.ipynb` | Ready (logic) |
| Eval harness | `experiments/phase2_retrieval/` | **TODO** |
| Metrics | Recall@{1,3,5,10}, MRR | — |

Methods: Regex-only | Embedding-only | Hybrid (union).

---

## Phase 3 — Classification benchmark

| Item | Path | Status |
|------|------|--------|
| Tasks L1 / L2 / L3 | `experiments/phase3_classification/` | **TODO** |
| Metrics | Accuracy, Macro-F1, Weighted-F1 | Primary: Macro-F1 |

---

## Phase 4 — Baselines

| Family | Models | Path | Status |
|--------|--------|------|--------|
| Classical | TF-IDF+SVM, FastText | `experiments/phase4_baselines/` | **TODO** |
| Transformer | PhoBERT-base/large, XLM-R-large | same | **TODO** |
| Embedding prototype | e5-large, BGE-M3, jina-v3 | same | **TODO** |
| LLM | gpt-4o-mini, gpt-4o (zero/few-shot) | same | **TODO** |

---

## Phase 5 — Pipeline evaluation

| Method | Pipeline | Path | Status |
|--------|----------|------|--------|
| M1 | Direct GPT-4o-mini | `experiments/phase5_pipeline/` | **TODO** |
| M2 | Hybrid retrieval + GPT-4o-mini | same | **TODO** |
| M3 | Retrieval + annotator + GPT-4o verifier | same | **TODO** |

Metrics: Accuracy, Macro-F1, Review Rate (vs `data/gold/test_gold.json`).

Reuse configs from `data/code/intent_labeling_gpt4.ipynb`.

---

## Phase 6 — Dataset quality (core evidence)

| Item | Spec | Path | Status |
|------|------|------|--------|
| Dataset A | Annotator-only labels | `data/datasets/dataset_a_raw.json` | **TODO** |
| Dataset B | Verifier-approved labels | `data/datasets/dataset_b_verified.json` | **TODO** |
| Classifier | PhoBERT-base (same seed/hparams) | `experiments/phase6_dataset_quality/` | **TODO** |
| Eval | Human gold test | `data/gold/test_gold.json` | **TODO** |

Hypothesis: Macro-F1(B) > Macro-F1(A).

---

## Ablation

| Config | Components | Path | Status |
|--------|------------|------|--------|
| Abl-1 | Annotation only | `experiments/ablation/` | **TODO** |
| Abl-2 | + Retrieval | same | **TODO** |
| Abl-3 | + Verification | same | **TODO** |
| Abl-4 | + Guardrails (full) | same | **TODO** |

Metrics: Macro-F1, Approval Rate, Human Review Rate.

---

## Conventions

- Fixed seed: **42** (unless noted in `split_meta.json`).
- Primary metric: **Macro-F1** at L3 (also report L1/L2).
- Never train on `sample_id` present in `data/gold/test_gold.json`.
- Log model names, prompt versions, and `VERIFY_*` flags in each experiment's `meta.json`.
