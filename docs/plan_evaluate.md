---
name: baseline-evaluation
overview: Lập kế hoạch triển khai và đánh giá các baseline retrieval, annotation, verification và review-routing cho framework gán nhãn intent phân cấp tiếng Việt. Kế hoạch tái sử dụng D1/D2/D3, taxonomy, split cố định và các harness đánh giá sẵn có; mọi kết quả được so sánh trên human gold độc lập.
todos:
  - id: freeze-evaluation-data
    content: Hoàn tất D1/D2/D3, loại D3 IDs và freeze split/cấu hình tái lập.
    status: pending
  - id: build-retrieval-baselines
    content: Triển khai R1–R4 và evaluation retrieval với logging candidate-level.
    status: pending
  - id: build-pipeline-baselines
    content: Triển khai M1, M2, M2-RR, M2-SR và M3 với schema/metrics thống nhất.
    status: pending
  - id: evaluate-review-routing
    content: Chạy Random, Confidence, Entropy và routing của framework ở cùng budget review.
    status: pending
  - id: report-results
    content: Xuất bảng/đồ thị quality-cost, kiểm định paired và cập nhật roadmap/experiment index.
    status: pending
isProject: false
---

# Baseline Evaluation Plan

## Mục tiêu
Đánh giá liệu retrieval, LLM re-ranking, cross-model verification, guardrails và review routing lần lượt cải thiện chất lượng nhãn L1–L3 và giảm nhu cầu human review hay không.

## Điều kiện dữ liệu bắt buộc
- Hoàn tất full verification để có D1 (annotator-only) và D2 (verified) trên cùng `sample_id`.
- Hoàn thành D3 human gold, loại toàn bộ ID D3 khỏi corpus retrieval, train và validation.
- Freeze seed `42`, taxonomy tại [`data/unified_intents.csv`](../data/unified_intents.csv), version prompt/model và budget review trong metadata.
- Tái sử dụng kiểm tra leakage trong [`scripts/build_d1_d2_train.py`](../scripts/build_d1_d2_train.py) và metric hierarchy trong [`experiments/evaluation/evaluate_labels.py`](../experiments/evaluation/evaluate_labels.py).

## Phase 2: Retrieval benchmarks
Tạo harness tại `experiments/phase2_retrieval/` và dùng D3 hoặc validation có gold để đo Recall@1/3/5/10, MRR và candidate taxonomy coverage.

- R1: Regex/BM25 retrieval.
- R2: Dense embedding retrieval.
- R3: Hybrid union hiện hữu từ [`intent_graph_rag_colab.ipynb`](../intent_graph_rag_colab.ipynb).
- R4: TechniqueRAG-inspired: retrieve top-K exemplar `(sentence, L1/L2/L3)` chỉ từ train, GPT-4o re-rank theo semantic fit + taxonomy, giữ top-k exemplar.

Lưu retrieval IDs, scores, re-ranking rationale, top-k và cấu hình `K/k` cho từng query để audit.

## Phase 5: Annotation pipeline baselines
Tạo runner cấu hình tại `experiments/phase5_pipeline/`, sử dụng cùng utterance D3, cùng schema output và cùng metric L1/L2/L3.

- M1 Direct LLM: GPT-4o-mini, không retrieval/verifier/guardrail.
- M2 Hybrid RAG: candidate từ R3 + GPT-4o-mini.
- M2-RR TechniqueRAG-inspired: top-k exemplar đã rerank + GPT-4o-mini; không verifier/guardrail. Đây là baseline phỏng theo TechniqueRAG, không phải tái lập LoRA generator của bài gốc.
- M2-SR Self-Refine: GPT-4o-mini annotate rồi tự critique/revise; giữ ngân sách gọi model được log riêng.
- M3 Full: hybrid retrieval + GPT-4o-mini + GPT-4o verifier + taxonomy guardrails, tái sử dụng logic [`data/code/labeling_pipeline.py`](../data/code/labeling_pipeline.py).

Báo cáo Accuracy, Macro-F1 và Weighted-F1 ở từng L1/L2/L3, full-path accuracy, invalid-taxonomy rate, approval rate, review rate, latency và chi phí API.

## Phase 6: Review-routing và cost baselines
Trên một annotation pool không chứa D3, chạy các routing policy cùng budget human review `b ∈ {20%, 40%, 60%}`; human labels phải là nhãn thật hoặc ghi rõ là oracle simulation.

- B1 Random allocation.
- B2 Confidence threshold: confidence annotator thấp chuyển review.
- B3 CoAnnotating entropy: K prompt/runs, majority vote; entropy cao chuyển review.
- B4 Ours: verifier + guardrails tạo review queue.

Đo label quality sau review, review-rate, số mẫu/call LLM, human effort và Macro-F1 downstream khi train classifier trên tập nhãn nhận được. Dùng cùng matched-ID D1/D2 logic và loại D3 theo [`scripts/build_d1_d2_train.py`](../scripts/build_d1_d2_train.py).

## Phân tích và báo cáo
- Xuất mỗi run: predictions theo `sample_id`, config/prompt hash, seed, model, candidate IDs, timestamps, cost và lỗi parse.
- Dùng [`experiments/evaluation/evaluate_labels.py`](../experiments/evaluation/evaluate_labels.py) làm điểm chung để đối chiếu nhãn với D3 human gold.
- Vẽ quality–review-budget curve và cost–quality curve; kiểm tra khác biệt bằng paired bootstrap trên cùng ID D3.
- Tách kết quả head/tail L3 và phân tích lỗi: retrieval miss, reranker miss, verifier revise đúng/sai, guardrail rejection.

## Tài liệu paper
Cập nhật [`docs/research_roadmap.md`](../docs/research_roadmap.md) và [`experiments/README.md`](../experiments/README.md) để ghi rõ M2-RR, Self-Refine, Random/Confidence/Entropy routing, nguồn citation và phạm vi mỗi baseline.
