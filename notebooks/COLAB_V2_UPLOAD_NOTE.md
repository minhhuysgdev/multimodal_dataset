# Colab v2 — Upload checklist (1 lần)

Sync **cùng cấu trúc thư mục** lên Drive (`/content/drive/MyDrive/intent_kb/` hoặc folder bạn đặt trong notebook).

Không cần upload cả repo. Chỉ các path dưới đây là đủ để chạy `downstream_colab_pro_v2.ipynb`.

---

## A. Notebook

```
notebooks/downstream_colab_pro_v2.ipynb
```

---

## B. Code (bắt buộc — đây là phần hay thiếu trên Drive)

```
experiments/annotation_benchmark/colab_downstream_v2.py   # runner §4
experiments/annotation_benchmark/run_downstream.py        # --l1-head, --decoding, verified_d2→d2_verified_train
experiments/annotation_benchmark/common.py                # lazy retrieval; METHODS

experiments/evaluation/utils.py                           # marginal decode + is_valid_intent_row
experiments/evaluation/train_tfidf_svm.py                 # L1 head
experiments/evaluation/train_transformer.py               # softmax + marginal
experiments/evaluation/lora_utils.py

experiments/annotation_benchmark/build_rq2_reeval_table.py  # paper table from reeval CSV

experiments/phase5_pipeline/pipeline_utils.py             # common.py import TaxonomyValidator
```

> Nếu thiếu `pipeline_utils.py` → import `common.py` fail.  
> `verified_d2` downstream train = `d2_verified_train.json` (1454 rows, không dùng 1737-row file).

### RQ2 v2 Colab (không Qwen)

Notebook `downstream_colab_pro_v2.ipynb`: `RUN_STAGE_A=True` → 18 jobs (tfidf+phobert+xlmr).

```bash
python experiments/annotation_benchmark/colab_downstream_v2.py --preset rq2_light --l1-head
# Optional Mistral: --preset rq2_mistral --append
python experiments/annotation_benchmark/build_rq2_reeval_table.py
```

---

## C. Data train + gold (bắt buộc)

```
data/datasets/zero_shot_dataset.json
data/datasets/few_shot_dataset.json
data/datasets/cot_dataset.json
data/datasets/freeal_dataset.json
data/datasets/candist_dataset.json
data/datasets/verified_d2_dataset.json      # RVI-HUL full (~1743)
data/datasets/d2_verified_train.json        # RVI-HUL clean (1454) ← ablation verified_d2_clean

data/gold/d3_dataset.json

data/splits/d3_val_ids.txt
data/splits/d3_test_ids.txt
```

---

## D. Nên có (tránh lỗi nếu code đụng taxonomy)

```
data/unified_intents.csv
data/intent_kb.intent_edges_28_05_2026.csv
```

---

## E. Không bắt buộc cho downstream v2

- `data/d2_dataset.json`, Mongo dump, embedding cache  
- `experiments/phase2_retrieval/retrieval_utils.py` (chỉ cần nếu chạy annotation RAG)  
- Kết quả cũ `experiments/annotation_benchmark/results/*` (optional: giữ để `--append`)

---

## Kiểm tra nhanh trên Colab (sau mount)

```bash
!python experiments/annotation_benchmark/colab_downstream_v2.py --check-upload
```

In `✅/❌` từng file. Chỉ chạy train khi tất cả mục **B + C** đều ✅.

---

## Gợi ý upload

1. Zip đúng các path trên (giữ relative path) → giải nén vào `intent_kb/`.  
2. Hoặc `rclone` / Drive desktop sync cả folder `experiments/` + `data/datasets/` + `data/gold/` + `data/splits/`.  
3. Mỗi lần sửa code local: **ưu tiên re-upload nhóm B** (code), data chỉ khi đổi nhãn.
