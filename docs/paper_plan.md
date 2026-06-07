# Dataset Build Plan — ViEcomIntent: Vietnamese E-Commerce Intent Dataset

> Mục tiêu: Xây dựng dataset intent tiếng Việt cho e-commerce mỹ phẩm,  
> đủ tiêu chuẩn công bố tại hội nghị/tạp chí khoa học (ACL/EMNLP/COLING/WWW).

---

## 0. Tổng quan

| Hạng mục | Mục tiêu |
|---|---|
| Tên dataset | **ViEcomIntent** (Vietnamese E-Commerce Intent Dataset) |
| Miền | Mỹ phẩm / skincare (Hasaki.vn) |
| Ngôn ngữ | Tiếng Việt |
| Taxonomy | 2 L1 × ~12 L2 × 50+ L3 |
| Tổng mẫu target | **5,000 câu hỏi** (annotated, quality-controlled) |
| Split | Train 70% / Val 10% / Test 20% |
| Annotation method | Pipeline bán tự động (notebook hiện tại) + Claude loop harness |
| Tiêu chuẩn | Inter-annotator agreement ≥ 0.75 (Cohen's κ) |

---

## 1. Hiện trạng (what we have)

```
intent_nodes   : 181 nodes (L1/L2/L3 taxonomy đã build)
intent_edges   : 182 edges (graph structure)
hasaki_prelabel: ~5,000 câu hỏi thô từ product page Q&A
Pipeline       : notebook intent_labeling_mongodb_qwen.ipynb
  ├── Graph-aware embedding (paraphrase-multilingual-MiniLM-L12-v2)
  ├── Union retrieval (regex + semantic, top-12 candidates)
  ├── LLM labeling (GPT-4o-mini / Qwen2.5-7B)
  ├── Taxonomy validation + guardrail slug
  └── QA status: auto_labeled / needs_review / rejected
```

**Gap để đủ công bố:**
- Chưa có test set được verify thủ công (ground truth)
- Chưa có inter-annotator agreement
- Chưa có baseline classifier chạy trên dataset
- Chưa có data card / dataset paper chuẩn

---

## 2. Kiến trúc Dataset

### 2.1. Split chiến lược

```
Tổng 5,000 câu
├── TRAIN  3,500 câu (70%) — auto_labeled + reviewed
│   ├── auto_labeled (conf ≥ 0.85): ~2,800 câu
│   └── needs_review (reviewed thủ công): ~700 câu
├── VAL    500 câu (10%) — PHẢI verify thủ công 100%
│   └── Dùng để tune threshold / ablation
└── TEST   1,000 câu (20%) — PHẢI verify thủ công 100%
    ├── 500 câu: in-domain (từ Hasaki Q&A)
    └── 500 câu: synthetic diversity (Claude loop harness)
```

### 2.2. Phân phối nhãn (target)

- Mỗi L3 intent: tối thiểu **30 mẫu** trong train
- Mỗi L3 intent: tối thiểu **10 mẫu** trong test
- Tỷ lệ before_sale / after_sale: ~60% / 40%
- Không có intent nào chiếm > 8% tổng dataset (balance)

### 2.3. Schema mẫu (mỗi row)

```json
{
  "id": "VI-ECOM-00001",
  "question": "Sản phẩm này có rẻ hơn bên Shopee không?",
  "question_normalized": "san pham nay co re hon ben shopee khong",
  "source": "hasaki_faq | synthetic_claude | synthetic_paraphrase",
  "L1": "truoc_mua_hang",
  "L2": "gia_so_sanh",
  "L3": "compare_price_platform",
  "taxonomy_path": "truoc_mua_hang/gia_so_sanh/compare_price_platform",
  "annotation_method": "auto_labeled | human_reviewed | claude_harness",
  "confidence": 0.92,
  "annotator_1": "L3_label",
  "annotator_2": "L3_label",
  "agreement": true,
  "split": "train | val | test",
  "created_at": "2025-01-01T00:00:00Z"
}
```

---

## 3. Claude Loop Harness — Thiết kế

> Dùng Claude API để sinh câu hỏi đa dạng + verify nhãn cho test set.

### 3.1. Mục đích của Claude harness

```
Task 1 — GENERATE: Sinh câu hỏi đa dạng cho intent thiếu mẫu
Task 2 — VERIFY  : Review nhãn từ pipeline (thay human annotator 1)
Task 3 — PARAPHRASE: Tạo biến thể paraphrase cho data augmentation
Task 4 — ADVERSARIAL: Sinh câu khó, gần nghĩa nhưng khác intent
```

### 3.2. Prompt Template — Task GENERATE

```python
GENERATE_PROMPT = """
Bạn là annotator chuyên về e-commerce mỹ phẩm tiếng Việt.

Nhiệm vụ: Sinh {n} câu hỏi THỰC TẾ mà khách hàng có thể hỏi trên Hasaki.vn,
thuộc intent sau:

Intent path: {taxonomy_path}
Mô tả: {description}
Tín hiệu nhận diện: {detection_signals}
Ví dụ hiện có: {existing_examples}

Yêu cầu:
- Câu hỏi tiếng Việt tự nhiên, như người thật gõ (có thể viết tắt, thiếu dấu)
- ĐA DẠNG: không lặp cấu trúc, không lặp từ khóa
- KHÔNG dùng lại ví dụ có sẵn
- Độ dài: 5–30 từ
- Bao gồm: câu ngắn, câu dài, câu có emoji, câu viết không dấu

Trả về JSON array:
[
  {{"question": "...", "style": "formal|informal|no_accent|with_emoji"}},
  ...
]
"""
```

### 3.3. Prompt Template — Task VERIFY

```python
VERIFY_PROMPT = """
Bạn là expert reviewer cho dataset intent classification tiếng Việt.

Cho câu hỏi sau và nhãn được gán tự động, hãy xác nhận hoặc sửa lại:

Câu hỏi: "{question}"
Nhãn gán tự động:
  L1: {predicted_L1}
  L2: {predicted_L2}
  L3: {predicted_L3}
  Confidence: {confidence}

Taxonomy có sẵn (L3 options trong L2={predicted_L2}):
{sibling_intents_with_descriptions}

Trả về JSON:
{{
  "verdict": "correct | wrong | ambiguous",
  "corrected_L1": "...",  // nếu wrong
  "corrected_L2": "...",  // nếu wrong
  "corrected_L3": "...",  // nếu wrong
  "reason": "giải thích ngắn gọn",
  "confidence_review": 0.0–1.0
}}
"""
```

### 3.4. Prompt Template — Task ADVERSARIAL

```python
ADVERSARIAL_PROMPT = """
Sinh {n} câu hỏi "khó" cho bài toán intent classification:
Câu hỏi phải TRÔNG GIỐNG intent {source_intent} nhưng thực ra thuộc {target_intent}.

Source intent: {source_path} — {source_description}
Target intent: {target_path} — {target_description}

Ví dụ source: {source_examples}
Ví dụ target: {target_examples}

Mục tiêu: Tạo test case thách thức, dễ nhầm lẫn giữa 2 intent gần nhau.
Trả về JSON array với field: question, correct_intent (target), confusion_reason
"""
```

### 3.5. Loop Harness Flow

```python
# Pseudocode — chạy song song với notebook hiện tại

def claude_harness_loop(db, target_per_intent=30):
    """
    Loop qua từng intent L3, kiểm tra số mẫu,
    gọi Claude để bổ sung nếu thiếu.
    """
    for intent in db.intent_nodes.find({"level": "intent"}):
        current_count = db.labeled_examples.count_documents({
            "intent.level_3": intent["l3"],
            "qa_status": {"$in": ["auto_labeled", "human_reviewed"]}
        })

        # Phase 1: Generate nếu thiếu mẫu
        if current_count < target_per_intent:
            n_needed = target_per_intent - current_count
            generated = claude_generate(intent, n=n_needed)
            save_to_db(generated, source="synthetic_claude", split="train")

        # Phase 2: Verify needs_review samples
        pending = db.labeled_examples.find({
            "intent.level_3": intent["l3"],
            "qa_status": "needs_review"
        })
        for sample in pending:
            verdict = claude_verify(sample, intent)
            update_qa_status(sample, verdict)

    # Phase 3: Adversarial pairs
    confusable_pairs = find_confusable_intent_pairs(db)
    for src, tgt in confusable_pairs:
        adversarial = claude_adversarial(src, tgt, n=5)
        save_to_db(adversarial, source="synthetic_adversarial", split="test")
```

---

## 4. Quy trình Build theo Phase

### Phase 1 — Data Audit (Tuần 1)

**Mục tiêu:** Biết chính xác mình có gì, thiếu gì.

```
[ ] Export toàn bộ intent_annotations từ MongoDB
[ ] Vẽ distribution plot: số mẫu per L3 intent
[ ] Xác định các intent thiếu (< 30 mẫu train)
[ ] Xác định các intent thừa (> 200 mẫu) → cần down-sample
[ ] Tính tỷ lệ auto_labeled / needs_review / rejected hiện tại
[ ] Tạo spreadsheet tracking: intent_id | count | status | gap
```

**Output:** `data/audit/intent_distribution.csv`

---

### Phase 2 — Test Set Construction (Tuần 1–2)

> Test set phải SẠCH NHẤT — đây là thứ reviewer xem đầu tiên.

```
[ ] Chọn ngẫu nhiên 1,000 câu từ hasaki_prelabel.json
    └── Stratified by L2 (tỷ lệ theo distribution thực)
[ ] Chạy pipeline hiện tại để lấy predicted label
[ ] Gọi Claude VERIFY loop cho toàn bộ 1,000 câu
[ ] Human review 200 câu random (spot-check)
[ ] Tính Cohen's κ giữa Claude verdict vs human spot-check
    └── Target: κ ≥ 0.75 (substantial agreement)
[ ] Mark 1,000 câu này là split="test", lock khỏi training
[ ] Thêm 500 câu synthetic adversarial (Claude harness Phase 3)
```

**Output:** `data/splits/test_v1.jsonl`

---

### Phase 3 — Val Set Construction (Tuần 2)

```
[ ] Chọn 500 câu từ pool còn lại (chưa vào test)
    └── Stratified, ưu tiên needs_review samples
[ ] Claude VERIFY toàn bộ 500 câu
[ ] Human spot-check 100 câu
[ ] Tính κ
[ ] Mark split="val"
```

**Output:** `data/splits/val_v1.jsonl`

---

### Phase 4 — Train Set Augmentation (Tuần 2–3)

```
[ ] Dùng pool còn lại (~3,500 câu) làm train base
[ ] Chạy Claude GENERATE cho các intent < 30 mẫu
    └── Sinh tối đa 2x so với số mẫu thật (không quá 50% synthetic)
[ ] Chạy Claude PARAPHRASE cho train để tăng diversity
[ ] Verify nhanh bằng Claude (confidence filter > 0.85)
[ ] Down-sample intent > 200 mẫu về 150 (random seed = 42)
[ ] Final train size: ~3,500 câu thật + ~500 synthetic
```

**Output:** `data/splits/train_v1.jsonl`

---

### Phase 5 — Quality Check & Statistics (Tuần 3)

```
[ ] Tính Inter-Annotator Agreement (IAA)
    └── Cohen's κ tại L3 level
    └── Fleiss' κ nếu có 3+ annotators
[ ] Vẽ confusion matrix giữa Claude predict vs human
[ ] Tính class imbalance ratio (max_class / min_class)
[ ] Check: không có câu nào xuất hiện ở 2 splits khác nhau
[ ] Check: không có câu duplicate trong cùng split
[ ] Tạo data card (xem Phase 6)
```

**Output:** `data/quality/iaa_report.json`, `data/quality/stats.csv`

---

### Phase 6 — Baseline Experiments (Tuần 3–4)

> Phải có ít nhất 1 baseline để chứng minh dataset "useful".

```
Baseline 1: Zero-shot với paraphrase-multilingual-MiniLM-L12-v2
  └── Cosine similarity → top-1 intent (no fine-tuning)
  └── Report: Accuracy@1, Accuracy@3, Macro-F1

Baseline 2: Fine-tuned PhoBERT-base trên train set
  └── Multi-class classification tại L3 level
  └── Report: Precision, Recall, F1 per L1/L2/L3

Baseline 3: Pipeline hiện tại (GPT-4o-mini + retrieval)
  └── Chạy trên test set, dùng kết quả verify làm ground truth
  └── Report: Auto-labeled accuracy, Pipeline F1

Baseline 4 (optional): GPT-4o zero-shot prompt
  └── Dùng làm upper-bound reference
```

**Output:** `experiments/baseline_results.csv`

---

### Phase 7 — Data Card & Release (Tuần 4)

```
[ ] Viết Data Card theo HuggingFace standard:
    ├── Dataset summary
    ├── Source & collection method
    ├── Taxonomy description
    ├── Annotation process
    ├── Quality metrics (IAA, coverage)
    ├── Intended use & limitations
    ├── License (đề xuất: CC BY 4.0)
    └── Citation BibTeX
[ ] Upload lên HuggingFace Datasets (public)
    └── URL: huggingface.co/datasets/[username]/ViEcomIntent
[ ] Tạo GitHub repo với README + notebook demo
[ ] Viết Dataset Paper (~4 pages, format ACL)
```

---

## 5. File Structure

```
ViEcomIntent/
├── README.md                    # Data card
├── LICENSE                      # CC BY 4.0
│
├── data/
│   ├── raw/
│   │   └── hasaki_prelabel.json # Nguồn gốc (không publish nếu có PII)
│   ├── splits/
│   │   ├── train_v1.jsonl       # 3,500 mẫu
│   │   ├── val_v1.jsonl         # 500 mẫu
│   │   └── test_v1.jsonl        # 1,000 mẫu (500 real + 500 synthetic)
│   ├── taxonomy/
│   │   ├── intent_nodes.json    # 181 nodes
│   │   └── intent_edges.json    # 182 edges
│   └── quality/
│       ├── iaa_report.json      # Cohen's κ per L2
│       ├── stats.csv            # Distribution stats
│       └── confusion_matrix.png
│
├── src/
│   ├── build_dataset.py         # Script chạy toàn bộ pipeline
│   ├── claude_harness.py        # Claude API loop (generate/verify/adversarial)
│   ├── split_dataset.py         # Tạo train/val/test
│   └── compute_iaa.py           # Tính Cohen's κ
│
├── notebooks/
│   ├── intent_labeling_mongodb_qwen.ipynb  # Pipeline hiện tại
│   ├── baseline_phobert.ipynb   # Fine-tune PhoBERT
│   └── analysis.ipynb           # Distribution & quality plots
│
└── experiments/
    └── baseline_results.csv
```

---

## 6. Tiêu chuẩn nộp bài báo

### Venue đề xuất (theo độ khó tăng dần)

| Venue | Deadline | Scope | Chance |
|---|---|---|---|
| **PACLIC 2025** | ~Aug 2025 | SE Asia NLP, dataset OK | ⭐⭐⭐ |
| **SoICT 2025** | ~Sep 2025 | Hội nghị Việt Nam, phù hợp | ⭐⭐⭐⭐ |
| **COLING 2026** | ~Oct 2025 | Dataset track | ⭐⭐⭐ |
| **ACL 2026 Findings** | ~Feb 2026 | Dataset + system | ⭐⭐ |
| **EMNLP 2025** | ~Jun 2025 | Rất cạnh tranh | ⭐ |

### Checklist tối thiểu để submit

```
[ ] Dataset có train/val/test split rõ ràng
[ ] IAA (Cohen's κ) ≥ 0.70 tại L3 level
[ ] Tối thiểu 3,000 mẫu annotated
[ ] Ít nhất 2 baseline experiments với số liệu
[ ] Data card đầy đủ (source, license, limitations)
[ ] Public release (HuggingFace hoặc GitHub)
[ ] Không có PII (câu hỏi không chứa tên người dùng)
[ ] Paper ≥ 4 pages mô tả dataset + experiments
```

---

## 7. Timeline

```
Tuần 1 (hiện tại)
├── Phase 1: Data audit — biết số mẫu per intent
└── Phase 2 (bắt đầu): Xây test set, chạy Claude VERIFY loop

Tuần 2
├── Phase 2 (hoàn thành): Lock test set 1,000 câu
└── Phase 3: Xây val set 500 câu

Tuần 3
├── Phase 4: Augment train set
└── Phase 5: Quality check, tính IAA

Tuần 4
├── Phase 6: Chạy baseline experiments
└── Phase 7: Data card + HuggingFace upload

Tuần 5–6
└── Viết paper, submit
```

---

## 8. Claude Harness — Implementation Checklist

```python
# src/claude_harness.py — cần implement các functions sau:

def claude_generate(intent_node, n=10, model="claude-sonnet-4-20250514"):
    """Sinh n câu hỏi cho 1 intent node."""
    pass

def claude_verify(sample, intent_node, siblings, model="claude-sonnet-4-20250514"):
    """Verify nhãn của 1 sample. Trả về verdict JSON."""
    pass

def claude_paraphrase(question, n=3):
    """Sinh n paraphrase cho 1 câu hỏi."""
    pass

def claude_adversarial(src_intent, tgt_intent, n=5):
    """Sinh n câu khó, trông giống src nhưng thực ra là tgt."""
    pass

def run_verify_loop(db, split="val", batch_size=50):
    """Chạy VERIFY loop cho toàn bộ một split."""
    pass

def run_generate_loop(db, target_per_intent=30):
    """Chạy GENERATE loop cho các intent thiếu mẫu."""
    pass

def compute_cohen_kappa(human_labels, claude_labels):
    """Tính Cohen's κ giữa human và Claude."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(human_labels, claude_labels)
```

---

## 9. Rủi ro và Mitigation

| Rủi ro | Mitigation |
|---|---|
| Claude sinh câu không tự nhiên | Spot-check 10% mẫu synthetic, loại nếu κ < 0.7 |
| Dataset bị imbalanced nặng | Down-sample + up-sample cứng theo cap 150 / floor 30 |
| Test set bị contaminate với train | Hash dedup trước khi split |
| Không đủ mẫu after_sale | Viết thêm câu hỏi after_sale từ kinh nghiệm domain |
| Hasaki không cho phép publish | Chỉ publish câu hỏi (không publish câu trả lời/URL gốc) |
| IAA thấp do taxonomy mơ hồ | Review lại description của các intent dễ nhầm, làm rõ boundary |

---

## 10. Số liệu kỳ vọng cho paper

```
Dataset Statistics (target):
  Total samples:        5,000
  Train / Val / Test:   3,500 / 500 / 1,000
  L1 categories:        2
  L2 groups:            ~12
  L3 intents:           50+
  Avg question length:  12.3 words
  Synthetic ratio:      ~15% (750 câu Claude-generated)
  IAA (Cohen's κ):      ≥ 0.75

Baseline Results (target):
  Zero-shot MiniLM:     Acc@1 ~65%, Macro-F1 ~60%
  Fine-tuned PhoBERT:   Acc@1 ~82%, Macro-F1 ~79%
  Pipeline (GPT+RAG):   Acc@1 ~88%, Macro-F1 ~85%
```

---

*Plan version 1.0 — cập nhật sau Phase 1 audit*