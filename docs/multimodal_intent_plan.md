# ViEcomIntent: Continual Graph-Aware Intent Modeling for Vietnamese E-Commerce

> Muc tieu: xay dung huong model tiep theo cho bo du lieu ViEcomIntent theo bai toan phan loai intent phan cap L1/L2/L3, **khong su dung hinh anh san pham**. Trong tam la text-only modeling, graph-aware semantic retrieval, va continual learning de xu ly nhan moi ma khong can retrain model moi lan taxonomy thay doi.

---

## 0. Dinh huong moi

Dataset hien tai da du on dinh de chuyen sang phase model. Huong tiep theo:

1. **Khong multimodal:** bo hoan toan anh san pham, image embedding, visual caption, CLIP/vision LLM.
2. **Text-only hierarchical intent classification:** input chinh la `sentence`; `category`, `product_name`, `brand` chi la context phu.
3. **Graph-aware semantic retrieval:** intent graph la bo nho mo rong cua he thong, dung de lay candidate L1/L2/L3 va ho tro label moi.
4. **Continual learning without immediate retraining:** khi co intent moi, cap nhat graph/taxonomy va cho phep auto-approve dua tren reasoning + confidence + guardrail thay vi phai retrain model ngay.
5. **Human-in-the-loop:** cac status `needs_review` va `pending_new_label_review` duoc dua cho human review; pipeline RAG/model dung o `qa_status`.

Ten paper/model de xuat:

**ViEcomIntent: A Hierarchical Vietnamese E-Commerce Intent Dataset with Continual Graph-Aware Semantic Retrieval**

---

## 1. Du lieu va taxonomy hien tai

Nguon chinh:

- `data/raw/hasaki/hasaki_prelabel.json`: cau hoi tho.
- `data/raw/hasaki/hasaki_labelled_clean.json`: nhan da qua GPT-4o-mini + guardrail.
- `data/unified_intents.csv`: taxonomy L1/L2/L3.
- MongoDB graph:
  - `intent_nodes`
  - `intent_edges`

Trang thai labelled hien tai:

```json
{
  "approved": 1603,
  "approved_auto_new_label": 158,
  "needs_review": 81,
  "pending_new_label_review": 169
}
```

Y nghia:

- `approved`: co the dua vao train/eval.
- `approved_auto_new_label`: co the dua vao train/eval sau khi kiem tra nhanh taxonomy.
- `needs_review`: human review truoc khi dua vao gold set.
- `pending_new_label_review`: human quyet dinh map ve intent cu hay them intent moi.

---

## 2. Bai toan model

### 2.1. Input

```text
sentence + category + product_name + brand
```

Trong do:

- `sentence`: tin hieu chinh.
- `category/product_name/brand`: context phu de giam nham domain, khong duoc de model overfit vao ten san pham.

### 2.2. Output

Model du doan intent phan cap:

```json
{
  "level_1": "truoc_mua_hang",
  "level_2": "thong_tin_san_pham",
  "level_3": "cong_dung_san_pham",
  "confidence": 0.91
}
```

Rang buoc:

- L2 phai thuoc L1.
- L3 phai thuoc L2.
- Neu confidence thap hoac taxonomy invalid -> `needs_review`.
- Neu model de xuat label moi -> `pending_new_label_review` truoc khi human chap nhan vao graph.

---

## 3. Pipeline tong the

```text
Raw QA sample
  -> text normalization
  -> graph-aware semantic retrieval (top-K intent candidates)
  -> model prediction
  -> taxonomy guardrail
  -> confidence + reasoning gate
  -> qa_status
      - approved
      - approved_auto_new_label
      - needs_review
      - pending_new_label_review
  -> human review queue (chi 2 status can review)
  -> update taxonomy / graph if needed
  -> continual graph memory
```

Khac voi multimodal plan cu:

- Khong download anh.
- Khong caption anh.
- Khong image embedding.
- Khong fusion text-image.
- Khong dung product image lam input.

---

## 4. Graph-aware semantic retrieval

### 4.1. Vai tro

Intent graph dong vai tro nhu **external memory** cho model:

- Luu taxonomy L1/L2/L3.
- Luu mo ta, detection signals, examples.
- Luu embedding cua moi intent node.
- Truy hoi top-K intent candidates cho moi cau hoi.

### 4.2. Retrieval input

```text
query = sentence + optional category/product_name/brand
```

### 4.3. Retrieval output

```json
[
  {
    "intent_id": "INT-123",
    "l1": "truoc_mua_hang",
    "l2": "thong_tin_san_pham",
    "l3": "cong_dung_san_pham",
    "semantic_score": 0.82
  }
]
```

### 4.4. Cach dung trong model

Co 3 muc do:

1. **Candidate-only retrieval:** top-K candidate duoc dua vao prompt/feature.
2. **Feature retrieval:** lay embedding cua candidate intent lam feature them cho classifier.
3. **Decision retrieval:** neu classifier confidence thap, dung nearest intent trong graph de snap ve label hop le.

---

## 5. Continual learning design

### 5.1. Muc tieu

Khong retrain model moi lan xuat hien intent moi. Thay vao do:

1. Human hoac LLM pipeline de xuat label moi.
2. Guardrail kiem tra L1/L2/L3 hop le.
3. Neu confidence cao va reasoning tot -> auto-approve co dieu kien.
4. Them label moi vao `unified_intents.csv` va MongoDB graph.
5. Embed node moi.
6. Retrieval se thay label moi trong cac lan predict sau.

### 5.2. Auto-approval cho label moi

Mot label moi chi duoc auto-approve khi thoa cac dieu kien:

| Dieu kien | Mo ta |
|---|---|
| Confidence cao | `confidence >= MIN_CONF_AUTO_ADD_NEW_LABEL` |
| Reasoning ro | Giai thich phai noi dung y dinh cau hoi, khong chi lap lai label |
| Taxonomy valid | L1 dung tap cho phep; L2/L3 la slug hop le |
| Semantic support | Cau hoi gan voi candidate hoac gan voi cluster intent tuong ung |
| Khong duplicate | L3 moi khong trung/gan trung voi intent cu |

Trang thai:

- Pass tat ca -> `approved_auto_new_label`
- Khong chac -> `pending_new_label_review`
- Sai slug/taxonomy -> `pending_new_label_invalid_slug`

### 5.3. Vong lap continual learning

```text
New labelled data
  -> detect new / uncertain labels
  -> human review only if needed
  -> update taxonomy graph
  -> re-embed intent nodes
  -> retrieval sees new labels immediately
  -> periodic model retraining (weekly/monthly), not per new label
```

### 5.4. Khi nao moi retrain model?

Retrain theo dot, khong retrain tuc thoi:

- Khi so label moi du lon, vi du >= 30-50 examples.
- Khi L3 macro-F1 giam tren validation set.
- Khi distribution drift theo category/san pham moi.
- Khi human review sua qua nhieu mau cung mot cluster.

---

## 6. Baselines

Theo setup paper, baseline gom hai nhom chinh.

### 6.1. RoBERTa multi-head classifier

Mo ta:

> We fine-tune RoBERTa-base (Liu et al., 2019) with multiple classification heads. This adaptation allows the model to simultaneously categorize utterances across multiple dimensions.

Thiet ke:

```text
Encoder: RoBERTa-base / XLM-RoBERTa-base
Input: sentence + [SEP] category + [SEP] product_name + [SEP] brand

Shared encoder output [CLS]
  -> L1 classification head
  -> L2 classification head
  -> L3 classification head
```

Loss:

```text
loss = CE(L1) + CE(L2) + CE(L3)
```

Bien the:

- Flat L3 classification only.
- Multi-head L1/L2/L3.
- Hierarchical masked decoding: chi cho phep L2/L3 hop le theo parent.

Muc tieu:

- Baseline supervised text-only.
- De so sanh voi custom PhoBERT + retrieval.

### 6.2. Mistral sequence classification

Mo ta:

> We fine-tune Mistral-7B-v0.3 with a sequence classification head. Instead of directly generating output sequences, the model predicts intent classes through classification heads.

Thiet ke:

```text
Encoder/decoder backbone: Mistral-7B-v0.3
Head: sequence classification / pooled final token classifier
Output:
  - L1 class
  - L2 class
  - L3 class
```

Ly do khong dung generation:

- Giam hallucination slug.
- De tinh confidence/logits.
- De enforce label space co dinh.
- Cong bang hon khi so sanh voi RoBERTa/PhoBERT classifier.

Han che:

- Chi phi fine-tune cao hon.
- Can GPU manh hoac LoRA/QLoRA.
- Kho cap nhat label moi neu classifier head co label space co dinh.

---

## 7. Model custom: PhoBERT Graph-Aware Semantic Retrieval

### 7.1. Ten model de xuat

**PhoBERT-GAR: PhoBERT with Graph-Aware Retrieval for Hierarchical Vietnamese E-Commerce Intent Classification**

### 7.2. Y tuong

Ket hop:

1. **PhoBERT text encoder** cho cau hoi tieng Viet.
2. **Graph-aware retrieval** lay top-K intent candidates.
3. **Candidate-aware scoring** de chon L1/L2/L3 hop le.
4. **Continual graph memory** de label moi co the duoc retrieval thay ngay sau khi update graph.

### 7.3. Input representation

```text
[CLS] sentence [SEP] category [SEP] product_name [SEP] brand
```

PhoBERT encode thanh `h_query`.

Moi intent candidate co text:

```text
L1: ... | L2: ... | L3: ... | description: ... | signals: ... | examples: ...
```

Encode thanh `h_intent`.

### 7.4. Scoring

```text
score(intent_i | query) =
  alpha * classifier_score(L3_i)
  + beta * cosine(h_query, h_intent_i)
  + gamma * retrieval_score_i
```

Output label:

```text
argmax over top-K valid intent candidates
```

Neu khong candidate nao du tot:

- `needs_review`, neu cau hoi gan intent cu nhung confidence thap.
- `pending_new_label_review`, neu model/retrieval goi y label moi.

### 7.5. Kien truc

```text
Query text
  -> PhoBERT encoder
  -> query embedding

Intent graph
  -> intent text encoder / cached intent embeddings
  -> top-K retrieval candidates

query embedding + candidate embeddings
  -> candidate scorer
  -> taxonomy-constrained prediction
  -> confidence
  -> qa_status
```

### 7.6. Training objective

Ket hop 3 loss:

```text
L = L_classification + lambda1 * L_contrastive + lambda2 * L_hierarchy
```

Trong do:

- `L_classification`: cross-entropy cho L1/L2/L3.
- `L_contrastive`: day query gan intent dung, xa intent sai.
- `L_hierarchy`: phat prediction khong hop le theo parent-child.

### 7.7. Vi sao phu hop continual learning?

Classifier truyen thong bi khoa boi label head co dinh. PhoBERT-GAR co loi the:

- Label moi duoc them vao graph nhu mot candidate moi.
- Neu embedding intent moi tot, retrieval co the lay ra ngay.
- Model scoring candidate, khong nhat thiet phai mo rong classifier head moi lan.
- Retrain dinh ky giup consolidate label moi, nhung khong chan pipeline van hanh.

---

## 8. Experimental setup

### 8.1. Dataset split

De tranh leakage:

- Split theo `product_id` neu co.
- Neu khong, split theo `sample_id` va kiem tra overlap sentence gan trung.

De xuat:

| Split | Ty le |
|---|---:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Chi dua vao gold set:

- `approved`
- `approved_auto_new_label` sau khi kiem nhanh
- human-approved tu `needs_review`
- human-approved tu `pending_new_label_review`

Khong dua vao train:

- `skipped_ambiguous`
- `pending_new_label_invalid_slug`
- item chua human review trong `needs_review`/`pending_new_label_review`

### 8.2. Metrics

Bao cao theo cap:

- L1 Accuracy / Macro-F1
- L2 Accuracy / Macro-F1
- L3 Accuracy / Macro-F1
- Hierarchical exact match
- Path consistency rate

Bao cao cho retrieval:

- Recall@K cua gold L3 trong top-K candidates.
- MRR@K.
- Candidate coverage theo L1/L2/L3.

Bao cao cho continual learning:

- New-label acceptance precision.
- Human review reduction rate.
- Time-to-adapt: so buoc can thiet de label moi duoc retrieval thay.
- Performance truoc/sau khi update graph.

### 8.3. Baseline table

| Model | Retrieval | Continual label support | Output |
|---|---|---|---|
| RoBERTa multi-head | No | No | L1/L2/L3 heads |
| Mistral sequence classification | No | No | L1/L2/L3 heads |
| PhoBERT classifier | No | Limited | L1/L2/L3 heads |
| PhoBERT + retrieval features | Yes | Partial | L1/L2/L3 |
| **PhoBERT-GAR** | **Yes** | **Yes** | Candidate-constrained L1/L2/L3 |

---

## 9. Milestones

### Milestone 1 - Freeze dataset v1

Output:

- `hasaki_labelled_clean.json`
- human review queue cho:
  - `needs_review`
  - `pending_new_label_review`
- final gold split train/val/test

Done khi:

- Moi item train/test co L1/L2/L3 hop le.
- Taxonomy graph dong bo voi `unified_intents.csv`.
- Khong con item invalid slug trong gold set.

### Milestone 2 - Retrieval benchmark

Output:

- Recall@K, MRR@K cua graph-aware retrieval.
- Error analysis cho cac case gold intent khong nam trong top-K.

Done khi:

- Chon duoc `top_k` mac dinh.
- Biet retrieval loi do missing taxonomy hay embedding yeu.

### Milestone 3 - Baseline supervised models

Train:

- RoBERTa/XLM-R multi-head.
- Mistral-7B-v0.3 sequence classification (LoRA/QLoRA neu can).

Output:

- Bang metric L1/L2/L3.
- Confusion theo L3.
- Error cases theo before-sale / after-sale.

### Milestone 4 - PhoBERT-GAR

Build:

- PhoBERT query encoder.
- Intent candidate encoder/cache.
- Candidate scorer.
- Taxonomy-constrained decoding.

Output:

- Metric vs baselines.
- Ablation:
  - without retrieval
  - regex-only retrieval
  - semantic-only retrieval
  - graph-aware retrieval

### Milestone 5 - Continual learning simulation

Setup:

1. An mot phan L3 intent khoi train.
2. Them lai cac intent do vao graph sau training.
3. Kiem tra model co retrieve va classify duoc nhan moi khong.

Metrics:

- New-label Recall@K.
- New-label classification accuracy.
- Human review reduction.
- False auto-approval rate.

---

## 10. Paper positioning

Dong gop chinh:

1. **Dataset:** bo du lieu intent tieng Viet cho e-commerce voi taxonomy L1/L2/L3.
2. **Labeling pipeline:** GPT-4o-mini + graph-aware retrieval + guardrails + human review statuses.
3. **Graph-aware retrieval:** intent graph giup giam hallucination va tang taxonomy consistency.
4. **Continual learning:** label moi duoc them vao graph va retrieval thay ngay, khong can retrain lien tuc.
5. **Model:** PhoBERT-GAR ket hop PhoBERT voi graph-aware semantic retrieval cho hierarchical intent classification.

Thong diep can nhan manh:

- Khong phai multimodal.
- Khong phai chi LLM labeling.
- Trong tam la **hierarchical intent dataset + graph-aware retrieval + continual adaptation**.

---

## 11. Next actions

1. Chot gold set sau human review `needs_review` va `pending_new_label_review`.
2. Tao split train/val/test.
3. Benchmark retrieval Recall@K.
4. Train RoBERTa/XLM-R multi-head baseline.
5. Train Mistral sequence classification baseline neu co GPU.
6. Build PhoBERT-GAR candidate scorer.
7. Chay continual learning simulation voi label moi.
8. Viet bang ket qua va error analysis cho paper.
