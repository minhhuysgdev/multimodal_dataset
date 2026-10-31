# Cross-Verified Hierarchical Intent Labeling for Vietnamese E-commerce Dataset Construction

> Paper plan (Scopus-oriented).  
> Framework name in code/results: **RVI-HUL** / **Cross-Verified Hierarchical Intent Labeling (CV-HIL)**.  
> Related: [`research_roadmap.md`](research_roadmap.md), [`experiments/annotation_benchmark/README.md`](../experiments/annotation_benchmark/README.md).

---

## 1. Positioning

| Not this paper | This paper |
|----------------|------------|
| New SOTA intent classifier architecture | LLM annotation framework for hierarchical intents |
| Retrieval-representation paper (minimal prototypes) | Dataset construction with retrieval + verification + validation |
| Generic LLM labeling without structure | Controlled comparison + component ablation + downstream impact |

**One-sentence spine**

> We propose Cross-Verified Hierarchical Intent Labeling (CV-HIL), a retrieval-augmented LLM annotation framework with cross-model verification and taxonomy-aware validation, and show that it improves Vietnamese e-commerce intent annotation quality and yields better training data for hierarchical intent classification.

**Core pipeline**

```text
Hybrid Candidate Retrieval
→ Lightweight LLM Annotation (e.g., GPT-4o-mini)
→ Stronger-model Cross-Verification (e.g., GPT-4o RETAIN/REVISE)
→ Taxonomy-aware Validation / Guardrails
→ Train-ready labels (+ human review queue when needed)
```

---

## 2. Research Questions (updated)

| ID | Question | What it answers |
|----|----------|-----------------|
| **RQ1** | How effectively does the proposed framework annotate Vietnamese e-commerce queries compared with prompting-based, LLM-assisted, and retrieval-augmented methods? | End-to-end **annotation quality** vs external baselines |
| **RQ2** | How do candidate retrieval, cross-model verification, and taxonomy-aware validation contribute to annotation quality? | **Component ablation** inside CV-HIL |
| **RQ3** | Do the generated annotations provide better training data for downstream hierarchical intent classification? | **Downstream utility** of pseudo-labels |

### Evidence mapping

| RQ | Primary evidence | Existing assets |
|----|------------------|-----------------|
| **RQ1** | Per-level Accuracy / Macro-F1 / full-path Accuracy of each annotation method vs **D3 human gold** (300) | `eval_rq1.py` → `results/annotation_quality.csv` |
| **RQ2** | Same gold metrics on incremental CV-HIL variants A→D (table below) | Needs dedicated ablation runs (not the same as RQ1 baseline table) |
| **RQ3** | Train identical classifiers on each method’s train labels; evaluate on D3 test | `run_downstream.py` → `results/downstream_performance*.csv` |

**Important distinction**

- RQ1 compares **CV-HIL vs other annotation methods** (zero-shot, few-shot, CoT, FreeAL, CanDist, …).
- RQ2 ablates **components inside CV-HIL** (retrieval / verification / validation).
- RQ3 asks whether better annotations → better **supervised** L1/L2/L3 classifiers.

---

## 3. Method: CV-HIL (proposed)

### 3.1 Components

1. **Candidate retrieval** — hybrid regex + dense retrieval over intent prototypes/taxonomy; narrows L3 search space for the annotator.
2. **LLM annotation** — lightweight model proposes an L1–L2–L3 path (and confidence/rationale).
3. **Cross-model verification** — stronger model RETAIN/REVISE the proposed path given query, candidates, and annotator output.
4. **Taxonomy-aware validation** — guardrails for valid path, slug consistency, hierarchy constraints; assign `qa_status` (approved / needs_review / …).

### 3.2 RQ2 ablation variants

| Variant | Retrieval | Verification | Validation | Role |
| ------- | --------- | ------------ | ---------- | ---- |
| **A. LLM-only** | ❌ | ❌ | ❌ | Direct LLM path prediction (no retrieval/verify/validate) |
| **B. Retrieval-Augmented** | ✅ | ❌ | ❌ | + hybrid candidates |
| **C. Retrieval + Verification** | ✅ | ✅ | ❌ | + cross-model RETAIN/REVISE |
| **D. Full CV-HIL** | ✅ | ✅ | ✅ | + taxonomy validation / guardrails |

**Interpretation plan**

- A → B: value of candidate retrieval  
- B → C: value of cross-model verification  
- C → D: value of taxonomy-aware validation  

Primary metrics for RQ2: L1/L2/L3 accuracy, L3 macro-F1, full-path accuracy on D3 gold (same protocol as RQ1). Optionally report approval / review rates if validation changes routing.

---

## 4. Baselines (RQ1)

Prompting / LLM-assisted / retrieval-augmented family used in the annotation benchmark:

| Key | Display | Family |
|-----|---------|--------|
| `zero_shot` | Zero-shot | Prompting-based |
| `few_shot` | Few-shot | Prompting-based |
| `cot` | Chain-of-Thought | Prompting-based |
| `freeal` | FreeAL (LLM-only) | LLM-assisted annotation |
| `candist` | CanDist (LLM-only) | LLM-assisted (+ optional student distill) |
| `verified_d2` | CV-HIL / RVI-HUL (Ours) | Full proposed framework (= variant **D**) |

Notes:

- FreeAL/CanDist in the current benchmark are **LLM-only** adaptations (no full SLM self-training loop unless separately distilled).
- Ours train set for clean downstream: `data/datasets/d2_verified_train.json` (~1453–1454 valid rows after invalid drop).

---

## 5. Data protocol

| Split | Size | Use |
|-------|-----:|-----|
| Annotation pool | 2043 | Method labeling |
| Train pool (excl. D3) | 1743 ids | Pseudo-label training sets |
| D3 human gold | 300 | RQ1/RQ2 label quality; RQ3 eval |
| D3 val / test | 100 / 200 | Downstream split |

Hard constraints:

- **D3 never enters training** for RQ3.
- Same classifier hyperparameters across annotation methods.
- Report L1, L2, L3, full-path; emphasize L3 Macro-F1 (long-tail).

---

## 6. Experimental plan

### 6.1 RQ1 — Annotation effectiveness vs baselines

1. Run / load annotations for all methods on the pool.  
2. Evaluate against D3 gold (`eval_rq1.py`).  
3. Main table: method × (L1 Acc, L2 Acc, L3 Acc, L3 Macro-F1, Full-path Acc).

**Claim shape:** Full CV-HIL outperforms prompting and LLM-assisted baselines on fine-grained (L2/L3/path) metrics.

### 6.2 RQ2 — Component contributions (A–D)

1. Produce four annotation sets with component flags as in §3.2 (same pool, same models where applicable).  
2. Evaluate each variant on D3 with the same metrics as RQ1.  
3. Optional cost/review side metrics if validation changes `qa_status` mass.

**Claim shape:** Gains are incremental; retrieval and verification drive most L2/L3 gains; validation mainly enforces taxonomy consistency / review routing (measure both quality and review rate).

### 6.3 RQ3 — Downstream hierarchical classification

1. Build train sets from each method (`build_datasets.py`).  
2. Train classifiers under a fixed protocol, e.g.:
   - TF-IDF + LinearSVM  
   - PhoBERT-base  
   - XLM-RoBERTa-base  
   - (optional) Qwen3/Mistral LoRA  
3. Decoding: **marginal** level decoding (default); transformers may use L1 auxiliary head (`--l1-head`, `--lambda-l1`).  
4. Evaluate on D3 test; report L1/L2/L3 Acc, L3 Macro-F1, full-path Acc.

**Claim shape:** Labels from Full CV-HIL yield stronger downstream models than labels from prompting / LLM-assisted baselines under the same classifier.

Runner: `experiments/annotation_benchmark/run_downstream.py`.

---

## 7. Metrics

### Annotation quality (RQ1, RQ2)

- Accuracy @ L1 / L2 / L3  
- Macro-F1 @ L3 (primary fine-grained)  
- Full-path accuracy  
- (Optional) weighted-F1 @ L3  

### Downstream (RQ3)

- Same hierarchy metrics on D3 test  
- Always state classifier + epochs + decoding + whether L1 head is used  

### Secondary (optional, paper appendix)

- Annotation cost / token usage (`eval_rq3_cost.py` in benchmark — rename carefully in paper prose to avoid clashing with RQ3-downstream)  
- Human review / approval rates under validation  

---

## 8. Contributions (draft)

1. **CV-HIL framework** for Vietnamese hierarchical (L1–L2–L3) e-commerce intent annotation: hybrid retrieval, LLM annotation, cross-model verification, taxonomy-aware validation.  
2. **RQ1 empirical comparison** against prompting-based and LLM-assisted annotation methods on a human gold set.  
3. **RQ2 controlled ablation** isolating retrieval, verification, and validation.  
4. **RQ3 downstream study** showing that CV-HIL labels improve hierarchical intent classifiers trained under a matched protocol.

---

## 9. Paper outline

1. Introduction — noisy Vietnamese e-commerce queries; cost of hierarchical labeling; limits of single-pass LLM annotation  
2. Related Work — LLM annotation, retrieval-augmented labeling, verification/self-correction, hierarchical intent / text classification  
3. Method — CV-HIL components; formalize A–D variants  
4. Experimental Setup — data, baselines, metrics, no D3 leakage  
5. Results  
   - 5.1 RQ1 main comparison  
   - 5.2 RQ2 ablation A–D  
   - 5.3 RQ3 downstream  
6. Analysis — error types (L1 OK / L3 fail), when verification revises, long-tail L3  
7. Conclusion & limitations  

---

## 10. Status vs codebase (working checklist)

| Item | Status | Notes |
|------|--------|-------|
| RQ1 baselines + Ours on D3 | ✅ largely done | `annotation_quality.csv` (Ours strong on L2/L3/path) |
| RQ2 variants A–D | 🔄 in progress | Notebook [`notebooks/rq2_cvhil_ablation.ipynb`](../notebooks/rq2_cvhil_ablation.ipynb); pipeline flags in [`data/code/labeling_pipeline.py`](../data/code/labeling_pipeline.py) (`annotate_cvhil_ablation`, `CVHIL_ABLATION_VARIANTS`); eval [`experiments/annotation_benchmark/eval_rq2_ablation.py`](../experiments/annotation_benchmark/eval_rq2_ablation.py) → `results/rq2_ablation/`; D reuse `verified_d2` → `cvhil_d.json`; **chạy A/B/C trên D3 (300)** |
| RQ3 downstream multi-method | ✅ partial | Main `downstream_performance.csv`; L1-head reeval for Ours in `downstream_l1_reeval/` — re-run baselines under same decoding/L1-head for fair RQ3 table |
| Cost / review analysis | ⬜ optional | Keep out of main RQ numbering if RQ3 stays downstream-focused |

**RQ2 runbook (local / Colab)**

```bash
# Sau khi chạy notebook §5 (hoặc script tương đương):
python experiments/annotation_benchmark/eval_rq2_ablation.py \
  --outdir experiments/annotation_benchmark/results/rq2_ablation
```

Outputs: `ablation_quality.csv`, `ablation_quality_paper.md` (kèm bảng Δ A→B→C→D).

---

## 11. Naming consistency

| Context | Preferred name |
|---------|----------------|
| Paper title / method | Cross-Verified Hierarchical Intent Labeling (**CV-HIL**) |
| Code method key | `verified_d2` |
| Short display in tables | CV-HIL (Ours) or RVI-HUL (Ours) — pick **one** for the camera-ready paper |
| Ablation D | Full CV-HIL |

---

## Final claim paragraph (draft)

```text
We present CV-HIL, a cross-verified hierarchical intent labeling framework for Vietnamese e-commerce queries. Relative to prompting-based and LLM-assisted annotators, CV-HIL achieves higher agreement with human gold labels at fine-grained levels. Ablating candidate retrieval, cross-model verification, and taxonomy-aware validation shows how each stage contributes to annotation quality. Finally, classifiers trained on CV-HIL labels outperform those trained on alternative pseudo-labels under a matched downstream protocol, indicating that the framework produces more useful training data for hierarchical intent classification.
```
