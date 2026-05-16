# 4. Thử nghiệm và Phương pháp

## 4.1. Nguồn Dữ liệu và Thu Thập

### 4.1.1. Dữ liệu từ Hasaki

Dữ liệu intent được thu thập từ nền tảng e-commerce Hasaki (hasaki.vn), một trong những website bán mỹ phẩm hàng đầu tại Việt Nam. Cụ thể, chúng tôi tập trung vào lịch sử tương tác giữa khách hàng trên từng trang chi tiết sản phẩm. Sau khi làm sạch, chúng tôi thu được khoảng 5.000 câu hỏi duy nhất từ miền mỹ phẩm Hasaki.

---

## 4.2. Định Nghĩa Taxonomy Intent

### 4.2.1. Cấu Trúc Phân Cấp Ba Cấp

Thay vì tổ chức intent dưới dạng danh sách phẳng, chúng tôi sử dụng taxonomy phân cấp ba cấp (L1 → L2 → L3). Các nghiên cứu về hierarchical text classification cho thấy việc tận dụng cấu trúc phân cấp của nhãn giúp cải thiện khả năng phân loại so với xử lý nhãn phẳng, đặc biệt khi các nhãn có quan hệ cha–con trong một taxonomy [4]. Cụ thể, mỗi cấp được định nghĩa như sau:

| Cấp | Mô Tả | Ví dụ |
| --- | --- | --- |
| **L1** | Loại giai đoạn mua hàng | `before_sale`, `after_sale` |
| **L2** | Nhóm nghiệp vụ cụ thể | `product_inquiry`, `order_status`, `pricing_comparison` |
| **L3** | Intent chi tiết, gắn với action | `compare_price_items`, `ask_promotion`, `track_order_status` |

**Ví dụ đầy đủ:**

```
before_sale (L1)
├── product_inquiry (L2)
│   ├── ask_ingredients (L3)
│   ├── ask_usage (L3)
│   └── ask_skin_type (L3)
├── pricing_comparison (L2)
│   ├── compare_price_items (L3)
│   ├── ask_promotion (L3)
│   └── check_discount (L3)
└── shipping_info (L2)
    ├── ask_delivery_time (L3)
    └── ask_shipping_fee (L3)

after_sale (L1)
├── order_status (L2)
│   └── track_order_status (L3)
├── return_policy (L2)
│   ├── ask_return_reason (L3)
│   └── initiate_return (L3)
└── refund (L2)
    └── refund_status (L3)
```

Tổng cộng hệ thống định nghĩa **50+ intent L3** trải rộng trên ~12 nhóm L2 và 2 nhóm L1.

### 4.2.2. Làm Giàu Intent Node

Mỗi intent node không chỉ chứa tên nhãn, mà còn được làm giàu bằng các thông tin có cấu trúc. Hướng tiếp cận này tương đồng với TELEClass [5], trong đó các tác giả nhấn mạnh rằng việc chỉ sử dụng tên class trong taxonomy là chưa đủ, và việc bổ sung các đặc trưng chỉ báo class (class-indicative features) giúp mô hình hiểu không gian nhãn tốt hơn. Trong hệ thống của chúng tôi, mỗi node được làm giàu bằng:

```json
{
  "intent_id": "compare_price_items",
  "L1": "before_sale",
  "L2": "pricing_comparison",
  "L3": "compare_price_items",
  "taxonomy_path": "before_sale/pricing_comparison/compare_price_items",
  "parent_node": "pricing_comparison",
  "sibling_intents": ["ask_promotion", "check_discount"],
  "description": "Khách hàng muốn so sánh giá của nhiều sản phẩm khác nhau hoặc cùng sản phẩm trên các nền tảng khác.",
  "detection_signals": ["giá", "so sánh", "rẻ hơn", "đắt hơn", "khác"],
  "examples": [
    "Sản phẩm này có rẻ hơn Shopee không?",
    "So sánh giá với các shop khác đi"
  ]
}
```

Cách tổ chức này cho phép hệ thống:

- Hiểu intent không chỉ qua tên, mà qua ngữ cảnh phân cấp (L1 → L2 → L3)
- Kiểm soát tính hợp lệ của nhãn
- Hỗ trợ truy hồi ngữ nghĩa chính xác hơn

---

## 4.3. Phương Pháp Gán Nhãn

### 4.3.1. Mã Hóa Ngữ Nghĩa Intent (Semantic Encoding)

Để hỗ trợ truy hồi và gán nhãn, chúng tôi sử dụng Sentence Transformers để mã hóa intent và câu hỏi. Như được đề xuất trong Sentence-BERT [3], mô hình cho phép tạo sentence embeddings có thể so sánh bằng cosine similarity, giúp giảm đáng kể chi phí tính toán so với việc dùng BERT trực tiếp trên từng cặp câu.

**Model sử dụng**: `paraphrase-multilingual-MiniLM-L12-v2`, với kích thước embedding 384 chiều, hỗ trợ tiếng Việt và có thể chạy hiệu quả trên CPU.

Tuy nhiên, việc chỉ mã hóa tên intent hoặc mô tả ngắn có thể chưa đủ để phản ánh mối quan hệ giữa các nhãn trong taxonomy. Hai intent có tên gần nhau nhưng nằm ở các nhánh khác nhau có thể mang ý nghĩa nghiệp vụ khác nhau. Do đó, chúng tôi xây dựng một biểu diễn mở rộng gọi là **graph-aware text representation**, nhằm kết hợp thông tin ngữ nghĩa của intent với cấu trúc phân cấp của đồ thị taxonomy.

Thay vì chỉ sử dụng tên nhãn, mỗi intent node được chuyển đổi thành một chuỗi văn bản giàu ngữ cảnh bao gồm:

- Thông tin phân cấp (`L1`, `L2`, `L3`)
- Đường dẫn trong taxonomy (`taxonomy_path`)
- Node cha (`parent_node`)
- Các node anh em (`sibling_intents`)
- Mô tả ngữ nghĩa (`description`)
- Tín hiệu nhận diện (`detection_signals`)

Biểu diễn này giúp embedding không chỉ chứa ý nghĩa của riêng node hiện tại mà còn phản ánh vị trí tương đối của node trong cấu trúc đồ thị taxonomy.

Ví dụ đối với intent `compare_price_items`, chuỗi graph-aware text có thể được biểu diễn như sau:

```
L1: before_sale |
L2: pricing_comparison |
L3: compare_price_items |
Path: before_sale/pricing_comparison/compare_price_items |
Parent: pricing_comparison |
Siblings: ask_promotion, check_discount |
Description: Khách hàng muốn so sánh giá của nhiều sản phẩm khác nhau hoặc cùng sản phẩm trên các nền tảng khác |
Signals: giá so sánh rẻ hơn đắt hơn khác
```

### 4.3.2. Quy Trình Gán Nhãn

Hệ thống gán nhãn được thiết kế theo hướng weak supervision [1][2], trong đó thay vì gán nhãn thủ công toàn bộ dữ liệu, chúng tôi sử dụng các heuristic rule, guardrail và trạng thái QA để kiểm soát chất lượng nhãn đầu ra. Khác với Snorkel [2] — vốn kết hợp nhiều labeling function qua mô hình sinh, pipeline của nhóm đơn giản hơn nhưng bổ sung taxonomy validation, ambiguity filtering và semantic snap phù hợp với bài toán intent labeling có cấu trúc phân cấp.

Khi gán nhãn cho một câu hỏi mới, hệ thống đi qua 7 bước:

```
┌─────────────────────────────────────┐
│ 1. Ambiguity Check                  │
│ → Loại bỏ câu hỏi mơ hồ (<5 từ)     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 2. Candidate Retrieval              │
│ → Keyword/Regex ∪ Semantic Search   │
│ → dùng embedding đã lưu             │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 3. LLM Prediction                   │
│ → GPT-4 + RAG context               │
│ → Output:{L1,L2,L3,confidence}      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 4. Taxonomy Validation              │
│ → Kiểm tra L1→L2→L3                 │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 5. Semantic Snap                    │
│ → confidence < 0.7                  │
│ → Snap về intent gần nhất           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 6. Guardrail Validation             │
│ → Business Rules                    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 7. Save QA Status                   │
│ → auto_labeled / needs_review       │
│ → rejected                          │
└─────────────────────────────────────┘
```

### 4.3.3. Chi Tiết Các Bước

**Bước 1 — Ambiguity Check**: Loại bỏ câu hỏi dưới 5 từ hoặc chứa quá nhiều ký tự đặc biệt. Các câu bị lọc được lưu với status `ambiguous` và không đưa vào pipeline.

**Bước 2 — Candidate Retrieval**: Lấy top-5 intent ứng viên bằng cách hợp (union) hai phương pháp: keyword matching dựa trên `detection_signals` và semantic search dựa trên cosine similarity giữa embedding câu hỏi và embedding intent node [3].

**Bước 3 — LLM Prediction**: Gọi GPT-4 với RAG context gồm câu hỏi và danh sách candidates. Output là JSON `{L1, L2, L3, confidence}`.

**Bước 4 — Taxonomy Validation**: Kiểm tra L2 có là con hợp lệ của L1, và L3 có là con hợp lệ của L2 theo đồ thị taxonomy đã định nghĩa. Nếu sai cấp, hệ thống gán lại hoặc từ chối.

**Bước 5 — Semantic Snap**: Nếu confidence < 0.7, tính cosine similarity toàn bộ intent space và chuyển về intent có similarity cao nhất. Bước này tương tự cơ chế correction trong Data Programming [1].

**Bước 6 — Guardrail Validation**: Áp dụng business rule, ví dụ: intent thuộc `after_sale` chỉ được gán nếu câu hỏi chứa từ khóa chỉ thị giao dịch đã xảy ra ("đã mua", "order", "giao hàng").

**Bước 7 — Save with QA Status**:

- `auto_labeled`: confidence ≥ 0.85 → lưu tự động
- `needs_review`: 0.70 ≤ confidence < 0.85 → cần review thủ công
- `rejected`: confidence < 0.70 hoặc không qua guardrail

---

## 4.4. Lưu Trữ và Quản Lý Dữ Liệu

### 4.4.1. Cơ Sở Dữ Liệu MongoDB

Toàn bộ dữ liệu được lưu trữ trong MongoDB Atlas với hai collection chính:

**Collection `intent_nodes`** (Taxonomy):

```json
{
  "_id": "ObjectId",
  "intent_id": "compare_price_items",
  "L1": "before_sale",
  "L2": "pricing_comparison",
  "L3": "compare_price_items",
  "taxonomy_path": "...",
  "description": "...",
  "detection_signals": [...],
  "examples": [...],
  "embedding": [...],
  "created_at": "ISODate"
}
```

**Collection `labeled_examples`** (Dữ liệu gán nhãn):

```json
{
  "_id": "ObjectId",
  "question": "...",
  "L1": "before_sale",
  "L2": "pricing_comparison",
  "L3": "compare_price_items",
  "confidence": 0.92,
  "qa_status": "auto_labeled",
  "rejection_reason": null,
  "retrieval_method": "semantic",
  "source": "hasaki_faq",
  "created_at": "ISODate",
  "reviewed_by": null,
  "review_timestamp": null
}
```

---

## 4.5. Đánh Giá và Kết Quả Sơ Bộ

### 4.5.1. Chỉ Tiêu Đánh Giá

Chúng tôi đánh giá hiệu quả của phương pháp trên các tiêu chí sau:

1. **Precision của auto-labeled samples**: Tỷ lệ mẫu được label tự động mà sau khi review thủ công thì đúng
2. **Recall của candidate retrieval**: Tỷ lệ câu hỏi mà intent ground-truth nằm trong top-5 candidates
3. **Tỷ lệ cần review thủ công**: So với baseline gán nhãn 100% bằng tay
4. **Consistency của embedding**: Kiểm tra các intent cùng L2 có embedding gần nhau không [3]

### 4.5.2. Kết Quả Sơ Bộ

Trên tập ~1.000 câu hỏi test (annotated manually):

| Chỉ Tiêu | Giá Trị |
| --- | --- |
| Recall (Candidate Retrieval) | 96.2% |
| Precision (Auto-labeled, confidence ≥ 0.85) | 91.5% |
| % Cần Review (0.70–0.85 confidence) | 12.3% |
| % Từ Chối (< 0.70 confidence) | 3.2% |
| Thời gian gán nhãn trung bình/câu | 0.3s |

Recall cao (96.2%) cho thấy hệ thống bao phủ tốt intent đúng trong danh sách ứng viên. Chỉ 12.3% mẫu cần review thủ công, giảm đáng kể so với gán nhãn toàn bộ — phù hợp với mục tiêu giảm chi phí annotation của hướng tiếp cận weak supervision [1][2].

---

## 4.6. Công Cụ và Môi Trường

- **Python**: 3.10+
- **Libraries**: `sentence-transformers` [3], `pymongo`, `requests`
- **Nền tảng**: Google Colab / Local machine with GPU
- **Model LLM**: Qwen2.5-7B

---

## Tóm Tắt

Phương pháp này tập trung vào tổ chức dữ liệu intent theo taxonomy phân cấp [4][5] và xây dựng pipeline gán nhãn bán tự động có kiểm soát theo hướng weak supervision [1][2]. Bằng cách kết hợp truy hồi keyword và semantic [3], cùng với các bước validation (ambiguity, taxonomy, guardrail), hệ thống giảm đáng kể khối lượng review thủ công (↓87.7%) đồng thời duy trì precision cao (>91%) cho auto-labeled samples.

---

## Tài Liệu Tham Khảo

[1] A. Ratner, C. De Sa, S. Wu, D. Selsam, and C. Ré, "Data Programming: Creating Large Training Sets, Quickly," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2016, pp. 3567–3575. [Online]. Available: https://arxiv.org/abs/1605.07723

[2] A. Ratner, S. H. Bach, H. Ehrenberg, J. Fries, S. Wu, and C. Ré, "Snorkel: Rapid Training Data Creation with Weak Supervision," *Proc. VLDB Endow.*, vol. 11, no. 3, pp. 269–282, 2017. DOI: 10.14778/3157794.3157797

[3] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. EMNLP*, 2019, pp. 3982–3992. [Online]. Available: https://arxiv.org/abs/1908.10084

[4] J. Zhou, C. Ma, D. Long, G. Xu, N. Ding, H. Zhang, P. Xie, and G. Liu, "Hierarchy-Aware Global Model for Hierarchical Text Classification," in *Proc. ACL*, 2020, pp. 1106–1117. DOI: 10.18653/v1/2020.acl-main.104

[5] Y. Zhang, R. Yang, X. Xu, R. Li, J. Xiao, J. Shen, and J. Han, "TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification with Minimal Supervision," in *Proc. WWW*, 2025. DOI: 10.1145/3696410.3714940