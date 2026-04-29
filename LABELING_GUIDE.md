## Quy Trình Annotation Chi Tiết (5 Bước)

**Model gán nhãn (local / GPU):** [Qwen2.5-14B-Instruct-GPTQ-Int8](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8) — mô hình instruction-tuned 14B, lượng tử GPTQ 8-bit, hỗ trợ đa ngôn ngữ (kể cả tiếng Việt), phù hợp JSON có cấu trúc. Cần `transformers` phiên bản đủ mới (theo model card: từ `4.37.0` trở lên để tránh lỗi `KeyError: 'qwen2'`).

---

### **Bước 1️⃣: Nhập JSON — Chuẩn bị dữ liệu**

**Input:** mỗi mẫu theo schema Hasaki (ví dụ `data/raw/hasaki/hasaki_prelabel.json`): câu hỏi người dùng kèm ngữ cảnh sản phẩm; trường `intent` ban đầu có thể để trống, sau bước gán nhãn sẽ điền `level_1`, `level_2`, `level_3`.

```json
{
  "sample_id": "bang-che-khyet-diem-judydoll-3-mau-01-che-phu-cao-2-7g-125592_q03",
  "product_id": "bang-che-khyet-diem-judydoll-3-mau-01-che-phu-cao-2-7g-125592",
  "product_name": "Bảng Che Khuyết Điểm Judydoll 3 Màu - 01 Che Phủ Cao 2.7g",
  "brand": "Judydoll",
  "category": "Trang điểm",
  "sentence": "Dùng có dễ bị mốc không ?",
  "intent": {
    "level_1": "",
    "level_2": "",
    "level_3": []
  },
  "source": "hasaki"
}
```

**Gợi ý trường bổ sung cho prompt (nếu có):** `image`, `modality`, `cross_modal_ref` — chỉ đưa vào prompt khi thực sự dùng đa phương thức.

**Mapping nhãn → file:** sau khi model trả JSON, ghi vào `intent.level_1` (L1: `truoc_ban` / `sau_ban` hoặc slug tương ứng taxonomy), `intent.level_2` (L2), `intent.level_3` (L3 — có thể là chuỗi hoặc mảng tùy quy ước dự án; với Hasaki hiện tại thường là một intent cụ thể).

✓ **Mục đích:** Chuẩn hóa input, đảm bảo đủ ngữ cảnh sản phẩm + câu cần gán nhãn.

---

### **Bước 2: Xử lý dữ liệu (tiền xử lý)**

- **Khử nhiễu (Denoising), đa cấp (hội thoại / câu / từ):**
  - *Cấp hội thoại:* Loại bỏ phản hồi thừa của CSKH nếu là log chat.
  - *Cấp câu:* Lọc câu vô nghĩa (chào hỏi, hệ thống, spam).
  - *Cấp từ:* Chuẩn hóa ký tự, bỏ nhiễu không cần cho phân loại (tùy policy).
- **Tăng cường (Augmentation):** TrivialAugment cho ảnh nếu pipeline có ảnh; pseudo-label cho mẫu chưa có nhãn sau khi đã lọc theo ngưỡng tin cậy.

Tham khảo: https://dl.acm.org/doi/epdf/10.1145/3701716.3718371

---

### **Bước 3️⃣: Tạo prompt — Hướng dẫn Qwen2.5-14B-Instruct**

**Nội dung prompt nên có:**

- Mô tả task phân loại L1 → L2 → L3 theo taxonomy nội bộ (`unified_intents.csv` / `intent_hierachy.json`).
- **Detection signals** rõ ràng cho `truoc_ban` / `sau_ban` (hoặc `before_sale` / `after_sale` nếu taxonomy dùng tiếng Anh).
- **Rules ưu tiên** (ví dụ: có mã đơn → `sau_ban`).
- **Ngữ cảnh domain:** mỹ phẩm / TMĐT Việt Nam; đưa `category`, `product_name`, `brand` vào prompt.
- **Few-shot:** 3–4 ví dụ, output đúng format JSON.
- **Output:** chỉ một khối JSON hợp lệ (Qwen2.5 hỗ trợ structured output / JSON tốt hơn; vẫn nên yêu cầu “chỉ trả về JSON, không markdown”).

**Ví dụ rule trong prompt:**

```
RULE 1: Nếu tin nhắn chứa mã đơn (ví dụ #ABC123) → sau_ban
RULE 2: Nếu khách hỏi "có gây dị ứng không" → truoc_ban
RULE 3: Nếu khách nói "nhận được hàng hư" → sau_ban
RULE 4: Khi mơ hồ, ưu tiên tone + hành động (mua vs khiếu nại)
```

**Chạy model (tham khảo model card):** dùng `AutoTokenizer` + `AutoModelForCausalLM.from_pretrained(..., torch_dtype="auto", device_map="auto")` và `apply_chat_template` cho định dạng chat. Chi tiết cài đặt và GPTQ xem [trang model](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8).

✓ **Mục đích:** Model hiểu rõ ý định và trả nhãn + độ tin cậy ổn định trên GPU đủ VRAM.

---

### **Bước 4️⃣: Annotation (Qwen) + QA**

**4a — Model trả nhãn:** parse JSON từ Qwen rồi map vào `intent`.

**Output mong đợi từ model:**

```json
{
  "L1": "truoc_ban",
  "L2": "chat_luong_san_pham",
  "L3": "ket_cau_san_pham_co_de_bi_moc",
  "confidence": 0.88,
  "reasoning": "Hỏi độ bám/kết cấu sau khi thoa, không có mã đơn, ngữ cảnh trang điểm"
}
```

**Ghi vào JSON Hasaki:**

```json
"intent": {
  "level_1": "truoc_ban",
  "level_2": "chat_luong_san_pham",
  "level_3": ["ket_cau_san_pham_co_de_bi_moc"]
}
```

(Nếu dự án giữ `level_3` là chuỗi đơn, có thể dùng một chuỗi thay vì mảng một phần tử.)

✓ **Mục đích:** Có nhãn có thể kiểm tra và đưa vào huấn luyện.

**4b — Kiểm tra chất lượng (QA) ngay sau khi gán**

**Mức 1 — Consistency**

- ✓ **Confidence ≥ 0.90** → Chấp nhận **khi nhãn đã tồn tại trong taxonomy MongoDB**
- ⚠️ **0.70–0.89** → Xem lại prompt / few-shot hoặc rule; có thể model lẫn L1
- ❌ **< 0.70** → Từ chối pseudo-label, gán lại hoặc sửa tay
- ✓ **Reasoning** khớp tín hiệu trong prompt
- Nếu nhãn **không có trong taxonomy** (nhãn mới): áp dụng bảng **Chế độ `adaptive_mode`** ở cuối tài liệu (ngưỡng auto-add mặc định **0.96**).

**Mức 2 — Domain**

- ✓ L1 trùng taxonomy
- ✓ L2 có trong danh mục (mỹ phẩm + TMĐT)
- ✓ L3 khớp L2 và `category` sản phẩm
- ❌ Sai → gán lại bằng prompt chỉnh sửa hoặc sửa thủ công

---

## 📊 Quy trình QA tóm tắt

1. **Confidence:** với nhãn **đã có** trong taxonomy: `≥ 0.90` pass; `0.70–0.89` cảnh báo; `< 0.70` loại. Với nhãn **mới** (chưa có trong MongoDB): xem bảng `adaptive_mode` (auto-add mặc định `≥ 0.96`).
2. **Domain:** L1/L2/L3 hợp taxonomy và ngữ cảnh `category` / `sentence`.
3. **Quyết định:** Approved → tập huấn luyện; Rejected → chỉnh sửa; Pending → duyệt nhãn mới.

---

## 💡 Tín hiệu nhận dạng: PRE-SALE vs POST-SALE

| Signal | PRE-SALE | POST-SALE |
| --- | --- | --- |
| **Mã đơn** | Không | Có (vd. #ABC123) |
| **Tone** | Hỏi tư vấn | Khiếu nại, lo lắng |
| **Thì** | Hiện tại (“em cần…”) | Quá khứ (“em đã nhận…”) |
| **Hành động** | Quyết định mua | Giải quyết vấn đề sau mua |
| **Từ khóa** | dị ứng, phù hợp, so sánh | nhận được, hư, hoàn tiền, giao muộn |

---

## 🎓 Ví dụ thực tế

### Ví dụ 1: Before-sale

```
Customer: "Dạ em có làn da dầu mụn, dùng CeraVe được không?
           Sản phẩm này có chứa paraben không ạ?"
```

- L1: `truoc_ban` (không mã đơn, tone hỏi tư vấn)
- L2: `tu_van_san_pham` / `allergen_safety` (tùy taxonomy)
- L3: quan tâm thành phần / paraben
- Confidence cao nếu rule và few-shot rõ

### Ví dụ 2: After-sale — lỗi chất lượng

```
Customer: "Mã đơn #ZJU2025001. Hôm qua em nhận serum này,
           mở ra thì nước chảy ra hết. Hàng bị vỡ rồi!"
```

- L1: `sau_ban`
- L2: khiếu nại / chất lượng
- L3: sản phẩm hỏng khi nhận
- QA Pass nếu confidence và reasoning khớp tín hiệu

### Ví dụ 3: Hasaki — câu hỏi trên trang sản phẩm (như `hasaki_prelabel.json`)

```
sentence: "Dùng có dễ bị mốc không ?"
category: "Trang điểm"
```

- L1: `truoc_ban` (hỏi trước mua, không mã đơn)
- L2/L3: nhóm hỏi hiệu năng / kết cấu sản phẩm (ví dụ finish, độ bám, có bị mốc/mốc lớp không) — căn chỉnh đúng slug trong `unified_intents.csv`.

---

### **Bước 5️⃣: Ghi nhãn vào MongoDB + đồng bộ taxonomy mới**

Sau khi model trả nhãn và QA pass, cần ghi kết quả vào MongoDB để truy vấn/gắn cờ review.

**Collection gợi ý:**

- `intent_annotations`: lưu bản ghi gán nhãn cho từng `sample_id`
- `intent_nodes`, `intent_edges`: graph taxonomy intent (đã có trong notebook `intent_graph_rag_colab.ipynb`)

**Schema gợi ý cho `intent_annotations`:**

```json
{
  "sample_id": "bang-che-khyet-diem-..._q03",
  "product_id": "bang-che-khyet-diem-...",
  "sentence": "Dùng có dễ bị mốc không ?",
  "model": "Qwen2.5-14B-Instruct-GPTQ-Int8",
  "intent": {
    "level_1": "truoc_ban",
    "level_2": "chat_luong_san_pham",
    "level_3": ["ket_cau_san_pham_co_de_bi_moc"]
  },
  "confidence": 0.88,
  "reasoning": "Hỏi trước mua, không có mã đơn",
  "qa_status": "approved",
  "source": "hasaki",
  "updated_at": "2026-04-29T22:59:00Z"
}
```

**Nếu phát sinh nhãn mới (L2/L3 chưa có trong taxonomy):**

1. Gắn cờ `new_label_pending_review = true` trong `intent_annotations`.
2. Sau khi reviewer duyệt, upsert nhãn mới vào graph (`intent_nodes`/`intent_edges`).
3. Đồng bộ lại nguồn taxonomy (`unified_intents.csv` và/hoặc `intent_hierachy.json`) để lần chạy sau nhất quán.

Notebook `data/intent_graph_rag_colab.ipynb` đã được bổ sung cell helper để upsert nhãn mới vào graph.

**Notebook chạy gán nhãn end-to-end (MongoDB + Qwen + guardrail + batch Hasaki):** `intent_labeling_mongodb_qwen.ipynb` (ở thư mục gốc repo). Script tái tạo notebook: `scripts/build_labeling_notebook.py`.

**Nguyên tắc chống gán nhãn bậy bạ (bắt buộc):**

- Chỉ chấp nhận nhãn nếu `L1/L2/L3` đều tồn tại trong `intent_nodes` (MongoDB graph).
- Nếu model trả nhãn ngoài taxonomy: không approve trực tiếp, lưu `qa_status = pending_new_label_review`.
- Chỉ auto-thêm nhãn mới khi bật cờ quản trị (ví dụ `allow_auto_add_new_label=true`) **và** confidence vượt ngưỡng cao — **mặc định dự án: `>= 0.96`** (xem bảng dưới).
- Mọi nhãn mới phải được lưu log trong `intent_annotations` để audit.

---

## Chế độ `adaptive_mode`: khác taxonomy + confidence cao → tự thêm

Khi bật `adaptive_mode` (cùng `allow_auto_add_new_label=true`), hệ thống **không bắt buộc** nhãn phải có sẵn trong MongoDB trước khi gán; nếu model trả bộ `(L1, L2, L3)` **chưa có** trong graph thì xử lý theo **ngưỡng confidence** (do model trả về, thường trong `[0, 1]`).

**Bảng ngưỡng khuyến nghị (mặc định pipeline):**

| Tình huống | Điều kiện confidence | Hành động |
| --- | --- | --- |
| Nhãn **đã có** trong taxonomy MongoDB | `≥ 0.90` | `approved` — dùng để huấn luyện / downstream |
| Nhãn **đã có** trong taxonomy | `0.70 – 0.89` | `needs_review` — xem lại prompt, few-shot, hoặc rule |
| Nhãn **đã có** trong taxonomy | `< 0.70` | `rejected` — gán lại hoặc sửa tay |
| Nhãn **chưa có** (khác taxonomy), muốn **tự thêm** vào graph | `≥ 0.96` **và** `allow_auto_add_new_label=true` | `approved_auto_new_label` — upsert node/edge (taxonomy mở rộng có kiểm soát) |
| Nhãn **chưa có**, chưa đủ tin để auto-add | `0.90 – 0.95` | `pending_new_label_review` — **không** tự thêm; chờ người duyệt |
| Nhãn **chưa có**, confidence thấp | `< 0.90` | `rejected` hoặc `pending_new_label_review` — không tự thêm |

**Vì sao tách ngưỡng “nhãn mới” cao hơn (0.96)?** Nhãn lệch taxonomy làm **phình lớp nhãn** và khó rollback; ngưỡng cao hơn một bậc so với nhãn đã có giúp giảm pseudo-label nhiễu (tương tự tinh thần lọc pseudo-label bằng ngưỡng trong học bán giám sát — xem FixMatch bên dưới).

**Điều kiện phụ (khuyến nghị bật thêm):** chỉ auto-add khi `reasoning` không rỗng, không vi phạm rule cứng (ví dụ có mã đơn mà L1 lại `truoc_ban`), và (tuỳ chọn) độ dài câu đủ để không phân loại trên câu quá ngắn.

**Tham số trong code (notebook):** có thể đặt `AUTO_ADD_NEW_LABEL_THRESHOLD = 0.96`, `min_conf_auto_approve = 0.90`, `min_conf_allow_new_label = 0.96` để đồng bộ với bảng trên.

---

## Tài liệu tham khảo (ưu tiên nguồn có DOI / chính thức)

1. **Pseudo-label & ngưỡng confidence (học bán giám sát):** Sohn et al., *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence* (NeurIPS 2020). Dùng ngưỡng τ trên xác suất lớp để chấp nhận pseudo-label, tránh nhiễu — áp dụng tinh thần tương tự cho pipeline “chỉ tin pseudo-label / nhãn mới khi confidence đủ cao”. https://arxiv.org/abs/2001.07685  

2. **Hiệu chỉnh xác suất (tránh tin sai “confidence” của mạng):** Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017). Nếu sau này dùng softmax thật thay vì confidence do LLM tự báo, nên hiệu chỉnh trước khi áp ngưỡng cố định. https://arxiv.org/abs/1706.04599  

3. **RAG / truy hồi taxonomy trước khi sinh nhãn:** Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020). Cơ chế “lấy ứng viên từ kho (ở đây MongoDB) rồi mới sinh/ chọn nhãn” bám định hướng RAG. https://arxiv.org/abs/2005.11401  

4. **MongoDB — cập nhật có `upsert` (thêm nhãn / cạnh idempotent):** MongoDB Manual, `updateOne` / `upsert`. https://www.mongodb.com/docs/manual/reference/method/db.collection.updateOne/  

5. **Model gán nhãn (Qwen2.5):** Model card & quickstart — [Qwen2.5-14B-Instruct-GPTQ-Int8](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8).  

6. **Bài phương pháp pipeline ban đầu (denoise / augmentation):** NLPCC / ACM (đã trích trong Bước 2). https://dl.acm.org/doi/epdf/10.1145/3701716.3718371  

*Ghi chú:* Ngưỡng 0.90 / 0.96 là **mặc định vận hành** của dự án; có thể tinh chỉnh sau khi đo precision/recall trên tập dev có nhãn vàng.
