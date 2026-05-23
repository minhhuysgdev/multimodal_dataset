# Plan — Gán nhãn Gemini 2.5 Pro từ MongoDB và đối chiếu với GPT-4o-mini

> Mục tiêu: dùng **Gemini 2.5 Pro** như một annotator độc lập để gán nhãn lại dữ liệu thô từ `data/hasaki_prelabel.json`, truy hồi candidate intent từ MongoDB giống pipeline hiện tại trong `data/raw/intent_labeling_mongodb_qwen.ipynb`, sau đó đối chiếu với file nhãn GPT-4o-mini đã export ở `data/raw/hasaki/hasaki_labelled_clean.json`.

---

## 1. Bối cảnh hiện tại

Pipeline hiện tại trong notebook `data/raw/intent_labeling_mongodb_qwen.ipynb` đang thực hiện:

```text
MongoDB taxonomy graph
→ retrieve top-k intent candidates
→ GPT-4o-mini / Qwen label JSON
→ validate taxonomy + guardrail
→ export labelled JSON
```

File dữ liệu thô cần Gemini gán nhãn:

```text
data/hasaki_prelabel.json
```

File nhãn GPT-4o-mini hiện tại dùng để so sánh sau khi Gemini chạy xong:

```text
data/raw/hasaki/hasaki_labelled_clean.json
```

Schema chính của mỗi mẫu đầu vào thô:

```json
{
  "sample_id": "...",
  "product_id": "...",
  "product_name": "...",
  "brand": "...",
  "sentence": "...",
  "category": "...",
  "intent": {
    "level_1": "",
    "level_2": "",
    "level_3": []
  },
  "modality": "text",
  "source": "hasaki"
}
```

Trong plan này, nhãn hiện tại được gọi là:

```text
annotator_a = GPT-4o-mini/Qwen pipeline
```

Nhãn mới từ Gemini được gọi là:

```text
annotator_b = Gemini 2.5 Pro
```

---

## 2. Mục tiêu đánh giá

Không xem Gemini là ground truth tuyệt đối. Gemini được dùng như **LLM judge / independent annotator** để:

- Kiểm tra nhãn hiện tại có nhất quán với một model mạnh khác hay không.
- Phát hiện các mẫu có nhãn mơ hồ, sai taxonomy, hoặc cần review thủ công.
- Tính agreement giữa hai nguồn gán nhãn.
- Tạo tập `needs_human_review` cho các trường hợp bất đồng.
- Tăng độ tin cậy khi báo cáo chất lượng dataset trong paper.

---

## 3. Model đề xuất

### 3.1. Model chính

```text
gemini-2.5-pro
```

Lý do:

- Hiểu tiếng Việt tốt.
- Khả năng reasoning mạnh, phù hợp để phân biệt intent gần nghĩa.
- Khác provider với OpenAI, giúp giảm bias khi kiểm định nhãn.
- Phù hợp làm annotator thứ hai trong quy trình LLM-as-a-judge.

### 3.2. Model phụ nếu cần giảm chi phí

```text
gemini-2.5-flash
```

Chỉ nên dùng `Flash` cho dry-run, test prompt, hoặc gán nhãn sơ bộ. Khi tạo kết quả chính thức để báo cáo, ưu tiên `Gemini 2.5 Pro`.

---

## 4. Dữ liệu đầu vào

### 4.1. Nguồn dữ liệu thô cho Gemini

```text
data/hasaki_prelabel.json
```

Đây là nguồn chứa:

- `sample_id`
- `sentence`
- `category`
- `product_id`
- `product_name`
- `brand`
- `image`
- `modality`
- `source`

Gemini sẽ đọc dữ liệu từ file này và tự gán nhãn mới, không đọc nhãn GPT trong prompt.

### 4.2. Nguồn nhãn GPT-4o-mini để đối chiếu

```text
data/raw/hasaki/hasaki_labelled_clean.json
```

Đây là nguồn chứa nhãn đã export từ pipeline hiện tại:

- `sample_id`
- `sentence`
- `category`
- `intent_3_level`
- `confidence`
- `qa_status`
- `reasoning`

File này chỉ dùng ở bước join/so sánh sau khi Gemini đã gán nhãn xong. Không đưa nhãn GPT vào prompt Gemini để tránh confirmation bias.

### 4.3. Nguồn taxonomy/candidate intent

Ưu tiên dùng taxonomy graph trong MongoDB như notebook hiện tại để lấy candidate intent.

Các collection chính:

```text
intent_nodes
intent_edges
```

Gemini pipeline nên giữ cùng cơ chế:

```text
sample từ data/hasaki_prelabel.json
→ retrieve top-k candidate intent từ MongoDB
→ Gemini 2.5 Pro chọn/gợi ý nhãn
→ validate taxonomy + guardrail
→ export Gemini labelled JSON
```

Nếu cần chạy offline, dùng thêm:

```text
data/intent_hierachy.json
```

Lưu ý: taxonomy trong file JSON có thể khác format slug ASCII của pipeline hiện tại. Khi dùng để prompt Gemini, cần chuẩn hóa thành dạng:

```json
{
  "level_1": "truoc_mua_hang",
  "level_2": "thong_tin_san_pham",
  "level_3": "tem_chong_gia",
  "description": "...",
  "examples": []
}
```

---

## 5. Output mong muốn từ Gemini

Tạo file mới:

```text
data/raw/hasaki/hasaki_labelled_gemini_2_5_pro.json
```

Schema đề xuất:

```json
{
  "run_mode": "gemini_2_5_pro_mongodb_labeling",
  "source_input_file": "data/hasaki_prelabel.json",
  "comparison_label_file": "data/raw/hasaki/hasaki_labelled_clean.json",
  "model": "gemini-2.5-pro",
  "num_labelled": 2043,
  "results": [
    {
      "sample_id": "...",
      "product_id": "...",
      "product_name": "...",
      "brand": "...",
      "sentence": "...",
      "category": "...",
      "candidate_intents": [
        {
          "level_1": "truoc_mua_hang",
          "level_2": "...",
          "level_3": "...",
          "intent_id": "..."
        }
      ],
      "gemini_intent_3_level": {
        "level_1": "truoc_mua_hang",
        "level_2": "...",
        "level_3": "..."
      },
      "gemini_confidence": 0.92,
      "gemini_reasoning": "...",
      "gemini_is_new_label": false,
      "gemini_error": null
    }
  ]
}
```

Giữ `sample_id` giống `data/hasaki_prelabel.json` và `data/raw/hasaki/hasaki_labelled_clean.json` để join kết quả chính xác.

---

## 6. Prompt Gemini

Prompt nên giữ cùng triết lý với notebook hiện tại: `sentence` là tín hiệu chính, `category/product_name/brand` chỉ là context phụ.

Template đề xuất:

```text
Bạn là chuyên gia gán nhãn intent cho QA thương mại điện tử tiếng Việt trong miền mỹ phẩm/làm đẹp.

Nhiệm vụ:
Chọn intent phù hợp nhất cho câu hỏi của khách hàng dựa trên NGỮ NGHĨA của câu.

Quy tắc:
- Chỉ chọn 1 intent duy nhất.
- `sentence` là nguồn quyết định chính.
- `category`, `product_name`, `brand` chỉ dùng làm ngữ cảnh phụ.
- Không suy diễn intent đặc thù từ tên sản phẩm nếu câu hỏi không nhắc tới.
- L1 chỉ được là `truoc_mua_hang` hoặc `sau_mua_hang`.
- L2/L3 phải là slug lowercase ASCII, dùng underscore.
- Nếu không có intent nào khớp trong danh sách ứng viên, đề xuất nhãn mới và đặt `is_new_label=true`.
- Confidence ∈ [0,1].

Ứng viên taxonomy:
{candidate_intents}

Mẫu cần gán:
- sample_id: {sample_id}
- sentence: {sentence}
- category: {category}
- product_name: {product_name}
- brand: {brand}

Trả về DUY NHẤT JSON hợp lệ:
{
  "reasoning": "...",
  "confidence": 0.0,
  "L1": "...",
  "L2": "...",
  "L3": "...",
  "is_new_label": false
}
```

Khuyến nghị cấu hình:

```text
temperature = 0
top_p = 1
max_output_tokens = 512
response_mime_type = application/json
```

---

## 7. Chiến lược chạy Gemini

### 7.1. Dry-run

Chạy thử 50-100 mẫu từ `data/hasaki_prelabel.json`.

Có thể chọn mẫu dry-run theo hai cách:

- Chọn tuần tự 50-100 mẫu đầu tiên từ file thô để kiểm tra pipeline đọc dữ liệu, retrieve MongoDB và gọi Gemini.
- Chọn có chủ đích bằng cách join tạm với `hasaki_labelled_clean.json` để lấy nhóm `approved`, `needs_review`, confidence thấp, hoặc mẫu có khả năng sai taxonomy. Khi gọi Gemini vẫn chỉ truyền câu hỏi thô và candidate từ MongoDB, không truyền nhãn GPT.

Mục tiêu dry-run:

- Kiểm tra Gemini có trả JSON ổn định không.
- Kiểm tra slug có đúng format không.
- Kiểm tra candidate intent retrieve từ MongoDB có đủ tốt không.
- Kiểm tra validate taxonomy/guardrail có tương thích output Gemini không.

Quan trọng: không đưa nhãn GPT hiện tại vào prompt ở lần gán nhãn độc lập. Nếu đưa nhãn GPT vào, Gemini sẽ bị bias.

### 7.2. Full-run

Sau khi prompt ổn định:

- Chạy toàn bộ danh sách mẫu trong `data/hasaki_prelabel.json`.
- Với mỗi mẫu, truy hồi top-k candidate intent từ MongoDB giống notebook GPT-4o-mini/Qwen.
- Gọi Gemini 2.5 Pro để sinh JSON nhãn.
- Validate taxonomy + guardrail trước khi lưu.
- Resume theo batch để tránh mất tiến trình.
- Lưu checkpoint sau mỗi batch.
- Log lỗi parse JSON, timeout, safety block, empty response.

Batch size đề xuất:

```text
50-100 samples / batch
```

---

## 8. So sánh nhãn GPT-4o-mini và Gemini

Tạo file so sánh:

```text
data/audit/gpt4o_mini_vs_gemini_2_5_pro_label_agreement.csv
```

Mỗi row:

```csv
sample_id,sentence,category,gpt_l1,gpt_l2,gpt_l3,gpt_confidence,gpt_qa_status,gemini_l1,gemini_l2,gemini_l3,gemini_confidence,match_l1,match_l2,match_l3,agreement_level,review_bucket
```

### 8.1. Rule agreement

```text
match_l1 = gpt_l1 == gemini_l1
match_l2 = gpt_l2 == gemini_l2
match_l3 = gpt_l3 == gemini_l3
```

`agreement_level`:

```text
exact_l3_match     : L1, L2, L3 đều giống
partial_l2_match   : L1, L2 giống nhưng L3 khác
partial_l1_match   : chỉ L1 giống
no_match           : L1 khác
gemini_new_label   : Gemini đề xuất nhãn mới
error              : Gemini lỗi hoặc thiếu output
```

### 8.2. Review bucket

```text
high_confidence_keep
  - exact_l3_match
  - GPT confidence >= 0.80
  - Gemini confidence >= 0.80

medium_confidence_keep
  - exact_l3_match
  - một trong hai confidence < 0.80

needs_human_review
  - partial_l2_match
  - partial_l1_match
  - no_match
  - gemini_new_label
  - GPT qa_status in ["needs_review", "pending_new_label_review", "pending_new_label_invalid_slug", "rejected"]

critical_review
  - GPT confidence >= 0.90
  - Gemini confidence >= 0.90
  - nhưng L3 khác nhau
```

---

## 9. Chỉ số cần báo cáo

### 9.1. Agreement rate

```text
L1 agreement = số mẫu L1 giống / tổng mẫu hợp lệ
L2 agreement = số mẫu L2 giống / tổng mẫu hợp lệ
L3 agreement = số mẫu L3 giống / tổng mẫu hợp lệ
```

L3 agreement là chỉ số quan trọng nhất vì bài toán đang gán intent 3 cấp.

### 9.2. Cohen's Kappa

Tính theo từng cấp:

```text
Cohen's κ L1
Cohen's κ L2
Cohen's κ L3
```

Diễn giải:

```text
κ < 0.40  : agreement yếu
0.40-0.60 : trung bình
0.60-0.75 : khá
> 0.75    : tốt, có thể báo cáo là reliable
```

### 9.3. Confusion matrix

Tạo confusion matrix cho `level_3`:

```text
rows    = GPT-4o-mini L3
columns = Gemini 2.5 Pro L3
```

Mục tiêu:

- Tìm các cặp intent hay bị nhầm.
- Phát hiện taxonomy bị overlap.
- Ưu tiên review các cặp có tần suất bất đồng cao.

### 9.4. Breakdown theo `qa_status`

Báo cáo agreement riêng cho:

```text
approved
approved_auto_new_label
needs_review
pending_new_label_review
rejected
```

Kỳ vọng:

- `approved` nên có agreement cao.
- `needs_review` nên có agreement thấp hơn.
- Nếu `approved_auto_new_label` agreement thấp, cần kiểm tra lại rule auto-add nhãn mới.

---

## 10. Quyết định nhãn cuối cùng

Không tự động thay nhãn chỉ vì Gemini khác GPT. Nên dùng Gemini để chia luồng review.

### 10.1. Giữ nhãn GPT hiện tại

Giữ nhãn hiện tại nếu:

```text
GPT L3 == Gemini L3
AND GPT confidence >= 0.80
AND Gemini confidence >= 0.80
```

Set:

```text
final_label_source = "gpt_gemini_agreed"
final_review_status = "trusted"
```

### 10.2. Cần review thủ công

Đưa vào human review nếu:

```text
GPT L3 != Gemini L3
OR Gemini đề xuất nhãn mới
OR GPT qa_status không phải approved
OR một trong hai confidence < 0.70
```

Set:

```text
final_review_status = "needs_human_review"
```

### 10.3. Ưu tiên review cao

Ưu tiên cao nếu:

```text
GPT confidence >= 0.90
AND Gemini confidence >= 0.90
AND GPT L3 != Gemini L3
```

Đây là nhóm đáng chú ý vì hai model đều tự tin nhưng bất đồng, thường cho thấy:

- Câu hỏi đa ý định.
- Taxonomy overlap.
- Prompt chưa rõ.
- Một trong hai model suy diễn sai.

---

## 11. File đầu ra cuối cùng

### 11.1. Gemini raw output

```text
data/raw/hasaki/hasaki_labelled_gemini_2_5_pro.json
```

### 11.2. Agreement CSV

```text
data/audit/gpt4o_mini_vs_gemini_2_5_pro_label_agreement.csv
```

### 11.3. Review set

```text
data/audit/needs_review_gpt4o_mini_vs_gemini_2_5_pro.csv
```

### 11.4. Metrics summary

```text
data/audit/gemini_2_5_pro_agreement_metrics.json
```

Schema:

```json
{
  "num_samples": 2043,
  "num_valid_compared": 2043,
  "l1_agreement": 0.0,
  "l2_agreement": 0.0,
  "l3_agreement": 0.0,
  "cohen_kappa_l1": 0.0,
  "cohen_kappa_l2": 0.0,
  "cohen_kappa_l3": 0.0,
  "review_buckets": {
    "high_confidence_keep": 0,
    "medium_confidence_keep": 0,
    "needs_human_review": 0,
    "critical_review": 0
  }
}
```

---

## 12. Đề xuất thứ tự triển khai

1. Tạo notebook/script mới, ví dụ:

```text
data/raw/intent_labeling_gemini_2_5_pro_eval.ipynb
```

2. Load `data/hasaki_prelabel.json` làm nguồn mẫu thô cần Gemini gán nhãn.

3. Kết nối MongoDB và load các collection taxonomy giống notebook hiện tại:

```text
intent_nodes
intent_edges
```

4. Với mỗi sample, retrieve top-k candidate intent từ MongoDB.

5. Dựng prompt Gemini bằng `sentence`, `category`, `product_name`, `brand`, và candidate intent vừa retrieve.

6. Chạy dry-run 50-100 mẫu.

7. Kiểm tra output JSON, validate taxonomy, và sửa prompt nếu cần.

8. Chạy full-run và lưu:

```text
data/raw/hasaki/hasaki_labelled_gemini_2_5_pro.json
```

9. Load `data/raw/hasaki/hasaki_labelled_clean.json` làm nguồn nhãn GPT để so sánh.

10. Join GPT output và Gemini output theo `sample_id`.

11. Tính agreement, Cohen's Kappa, confusion matrix.

12. Export review set.

13. Review thủ công nhóm `critical_review` trước, sau đó đến `needs_human_review`.

---

## 13. Cách viết trong paper

Có thể mô tả ngắn gọn như sau:

```text
To improve annotation reliability, we employed Gemini 2.5 Pro as an independent LLM annotator over the same raw Hasaki utterances used by the GPT-4o-mini pipeline. For each sample, Gemini was provided with the utterance and taxonomy candidates retrieved from the MongoDB intent graph, but it was not shown the original GPT label to avoid confirmation bias. We then measured inter-annotator agreement at each taxonomy level using agreement rate and Cohen's Kappa. Samples with label disagreement, low confidence, or newly proposed labels were routed to a manual review queue.
```

Phiên bản tiếng Việt:

```text
Để tăng độ tin cậy của dữ liệu gán nhãn, chúng tôi sử dụng Gemini 2.5 Pro như một annotator độc lập trên cùng tập câu hỏi Hasaki thô đã được pipeline GPT-4o-mini xử lý. Với mỗi mẫu, Gemini được cung cấp câu hỏi và các candidate taxonomy truy hồi từ intent graph trong MongoDB, nhưng không được xem nhãn GPT trước đó để tránh thiên lệch xác nhận. Sau đó, chúng tôi đo mức đồng thuận ở từng cấp taxonomy bằng agreement rate và Cohen's Kappa. Các mẫu bất đồng nhãn, confidence thấp, hoặc có nhãn mới được đưa vào hàng đợi review thủ công.
```

---

## 14. Tiêu chí thành công

Plan này được xem là đạt nếu sau khi chạy Gemini:

- L3 agreement đạt tối thiểu `0.75`.
- Cohen's Kappa L3 đạt tối thiểu `0.60`, tốt nhất là `>= 0.75`.
- Tất cả mẫu bất đồng được export vào review set.
- Các nhóm intent hay nhầm được phát hiện qua confusion matrix.
- Dataset cuối cùng có trường audit rõ ràng: GPT label, Gemini label, agreement, review status.

