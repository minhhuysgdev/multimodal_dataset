# Annotation Benchmark

So sánh 6 phương pháp annotation LLM (5 baselines + RVI-HUL proposed) trên pool Hasaki (2043 câu), đánh giá bằng D3 gold (300).

## Setup

```bash
cp env.sh.example env.sh
# Điền OPENAI_API_KEY
source env.sh
pip install -r requirements.txt
```

## Phase 0 — Data prep

```bash
python experiments/annotation_benchmark/prepare_data.py
```

Tạo:
- `data/annotations/pool.json` (2043)
- `data/splits/d3_val_ids.txt` (100)
- `data/splits/d3_test_ids.txt` (200)
- `data/splits/train_pool_ids.txt` (1743, loại D3)

## Phase 1 — Annotation

```bash
# Một method
python experiments/annotation_benchmark/run_annotation.py --method zero_shot

# Tất cả method (resume/checkpoint mỗi 50 mẫu)
python experiments/annotation_benchmark/run_annotation.py --method all

# Bootstrap verified_d2 từ d2_dataset.json (2011 câu, không gọi LLM)
python experiments/annotation_benchmark/bootstrap_verified_d2.py

# Xem 32 câu pool chưa có trong d2_dataset
python experiments/annotation_benchmark/list_verified_d2_missing.py

# Annotate 32 câu còn lại (resume tự động)
python experiments/annotation_benchmark/run_annotation.py --method verified_d2

# Hoặc chạy full pipeline còn lại một lần:
bash experiments/annotation_benchmark/finish_verified_d2.sh
```

Output: `data/annotations/{method}.json`

Methods: `zero_shot`, `few_shot`, `cot`, `freeal`, `candist`, `verified_d2` (6 baselines + RVI-HUL proposed)

### CanDist distillation (bắt buộc cho CanDist)

CanDist gồm 2 bước: teacher LLM sinh candidate labels, rồi student PhoBERT distill (PRODEN partial-label). Sau khi chạy `--method candist`, chạy tiếp:

```bash
python experiments/annotation_benchmark/distill_candist.py --model-name vinai/phobert-base --epochs 5
```

Script này ghi đè nhãn cuối trong `data/annotations/candist.json` bằng dự đoán của student (candidate + teacher pick vẫn lưu trong `metadata`). Student chỉ train trên `train_pool` (1743, đã loại D3), rồi relabel toàn bộ pool — không leak D3.

## Phase 2–5

```bash
python experiments/annotation_benchmark/eval_rq1.py
python experiments/annotation_benchmark/build_datasets.py --method all

# RQ2 downstream — mặc định TF-IDF + PhoBERT (5 epochs)
python experiments/annotation_benchmark/run_downstream.py --method all

# Thêm XLM-RoBERTa, Qwen3-1.7B-LoRA, Mistral-7B-LoRA (Mac M3 Pro)
python experiments/annotation_benchmark/run_downstream.py --method all \
  --classifiers tfidf,phobert,xlmr,qwen3,mistral --epochs 5

# Chỉ ML nhanh
python experiments/annotation_benchmark/run_downstream.py --method all --classifiers tfidf

# Smoke 1 method + 2 DL models
python experiments/annotation_benchmark/run_downstream.py --method verified_d2 \
  --classifiers xlmr,qwen3 --epochs 1

# Chạy theo đợt, giữ kết quả cũ (--append)
python experiments/annotation_benchmark/run_downstream.py --method all \
  --classifiers tfidf,phobert,xlmr --epochs 10
python experiments/annotation_benchmark/run_downstream.py --method all \
  --classifiers qwen3 --epochs 8 --append
python experiments/annotation_benchmark/run_downstream.py --method all \
  --classifiers mistral --epochs 8 --append

python experiments/annotation_benchmark/eval_rq3_cost.py
python experiments/annotation_benchmark/report.py
```

Hoặc chạy full:

```bash
bash experiments/annotation_benchmark/run_full_benchmark.sh
```

## Lưu ý quan trọng

- **D3 không bao giờ vào training dataset** (`train_pool_ids` loại 300 gold).
- FreeAL/CanDist ở đây là bản **LLM-only** (không có vòng SLM distillation).
- **RVI-HUL** (*Cross-Verified Hierarchical Intent Labeling*, proposed): key `verified_d2`. Tái sử dụng cache từ `data/d2_dataset.json` (pipeline production); 32 câu còn lại gọi LLM qua `verified_d2.py`.
- Dùng `--mock-llm` chỉ để test pipeline offline, **không dùng cho paper**.

## RQ2 classifiers (downstream)

| Key | Model | Train trên M3 |
|---|---|---|
| `tfidf` | TF-IDF + LinearSVM | CPU, vài phút |
| `phobert` | `vinai/phobert-base` | Full FT, batch 8 |
| `xlmr` | `FacebookAI/xlm-roberta-base` (multilingual RoBERTa) | Full FT, batch 8 |
| `qwen3` | `Qwen/Qwen3-1.7B` + LoRA | MPS, batch 2, grad_accum 8 |
| `mistral` | `mistralai/Mistral-7B-v0.3` + LoRA | MPS, batch 1, grad_accum 8 |

Chọn qua `--classifiers` (comma-separated). Qwen3/Mistral dùng LoRA (PEFT), không full fine-tune.

Dependencies thêm: `peft`, `accelerate`, `transformers>=4.51` (Qwen3).
