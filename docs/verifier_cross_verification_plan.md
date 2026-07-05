# Verifier — Cross-Verification cho pipeline gán nhãn intent

Tài liệu mô tả bước **LLM cross-verification** (verifier) bổ sung vào pipeline gán nhãn trong [`data/code/intent_labeling_gpt4.ipynb`](../data/code/intent_labeling_gpt4.ipynb). Mục tiêu: không chỉ dựa vào `confidence` của annotator (`gpt-4o-mini`), mà cho một model mạnh hơn (**GPT-4o**) đọc lại nhãn + rationale + utterance để **retain** hoặc **revise**, trước khi qua guardrail.

**Liên quan:** pipeline tổng quan → [`intent_labeling_gpt4_pipeline.md`](intent_labeling_gpt4_pipeline.md) · taxonomy/guardrail → [`../LABELING_GUIDE.md`](../LABELING_GUIDE.md)

---

## 1. Vì sao cần verifier (không chỉ confidence score)

- `confidence` do LLM tự báo thường **overconfident** và không calibrate tốt → ngưỡng cứng (0.80/0.85) dễ giữ lại nhãn sai hoặc loại nhầm nhãn đúng.
- Verifier là một lần "đọc lại" độc lập (second opinion) bằng model mạnh hơn, tập trung vào **ngữ nghĩa** thay vì keyword/category.
- Biến pipeline từ *single-pass prediction* thành *multi-pass reasoning*: Annotator (nhanh, nhiều nhiễu) → Verifier (chỉnh, ổn định).

## 2. Vị trí trong pipeline

```
1 Ambiguity Check
2 Candidate Retrieval
3 LLM Prediction (gpt-4o-mini)        <- Annotator
3.5 LLM Verification (GPT-4o)         <- MỚI (khi conf thấp hoặc VERIFY_ALL_SAMPLES)
4 Taxonomy Validation                  ┐
5 Semantic Snap                        ├ guardrail (giữ nguyên)
6 Guardrail Validation                 ┘
7 Save QA
```

Verifier chèn **giữa** `predict_intent()` và `save_annotation_with_guardrails()` trong `run_batch()`. Verifier **không** enforce taxonomy — để guardrail xử lý như cũ.

## 3. Cấu hình (§2)

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `ENABLE_VERIFIER` | `True` | Bật/tắt verifier. Đặt `False` để về pipeline cũ. |
| `VERIFIER_MODEL` | `gpt-4o` | Model verifier (env `VERIFIER_MODEL`). |
| `VERIFY_CONF_THRESHOLD` | `0.85` | Chỉ gọi verifier khi `confidence < ngưỡng`. |
| `VERIFY_ALL_SAMPLES` | `False` | `True` = verify mọi mẫu (khớp diagram); `False` = chỉ khi conf < ngưỡng. |
| `VERIFY_MAX_TOKENS` | `256` | Giới hạn token mỗi lần verify. |

Lưu ý: `VERIFY_CONF_THRESHOLD` (khi nào verify) tách biệt với `MIN_CONF_APPROVE_EXISTING = 0.80` (khi nào guardrail approve). Verifier không đổi ngưỡng approve.

## 4. Verifier prompt (hierarchy-aware L1/L2/L3)

Input: `utterance`, quy tắc taxonomy, danh sách candidates (từ retrieval), nhãn dự đoán (L1/L2/L3), confidence, rationale.

Output JSON:

```json
{
  "decision": "RETAIN | REVISE",
  "final_L1": "...",
  "final_L2": "...",
  "final_L3": "...",
  "confidence_adjusted": 0.0,
  "reason": "giải thích ngắn gọn tiếng Việt"
}
```

Quy tắc taxonomy verifier phải tuân:
- `L1 ∈ {truoc_mua_hang, sau_mua_hang}`.
- L2/L3 là slug ASCII lowercase + underscore; L3 khác L2 và cụ thể hơn.
- Đánh giá theo NGỮ NGHĨA câu hỏi, không theo product_category/tên sản phẩm.
- RETAIN nếu nhãn đúng; REVISE + đề xuất bộ (L1, L2, L3) tốt hơn nếu sai.

## 5. Merge logic

```python
if decision == "REVISE":
    label = (final_L1, final_L2, final_L3)
    confidence = confidence_adjusted
else:  # RETAIN
    confidence = max(confidence_annotator, confidence_adjusted)
```

- Chỉ gọi verifier khi `confidence < VERIFY_CONF_THRESHOLD` (tiết kiệm cost).
- Bỏ qua verifier với câu `_ambiguous` (đã skip ở annotator).
- Sau merge, `pred` đi vào guardrail → quyết định `qa_status`.

## 6. Metadata lưu lại (phục vụ paper / audit)

Thêm vào annotation + file export:

| Field | Ý nghĩa |
|-------|---------|
| `verifier_decision` | `RETAIN` / `REVISE` / `null` (không verify) |
| `verifier_reason` | Giải thích của verifier |
| `confidence_before_verify` | Confidence annotator trước khi verify |

## 7. Mapping qa_status

Verifier chỉ chỉnh `pred` (label + confidence) trước guardrail; bảng `qa_status` giữ nguyên.

- Annotator conf `0.78` → verifier `RETAIN`, `conf_adjusted=0.86` → guardrail `approved`.
- Annotator conf `0.87` (≥ 0.85) → **không** gọi verifier → luồng cũ.
- Annotator conf `0.80` → verifier `REVISE` nhãn khác, `conf=0.83` → guardrail đánh giá lại nhãn mới.

## 8. Rủi ro / lưu ý

- Chi phí tăng do gọi thêm model: giảm bằng cách chỉ verify `conf < 0.85`. Ước lượng số mẫu từ phân phối confidence thực tế của `gpt-4o-mini`.
- `OPENAI_API_KEY` cần quyền gọi `gpt-4o` (hoặc model đặt trong `VERIFIER_MODEL`).
- Verifier có thể đề xuất slug mới → guardrail vẫn xử lý qua semantic snap / pending, không upsert ẩu (trừ khi `persist=False` trong reverify — không ghi graph).
- Quay lui an toàn: `ENABLE_VERIFIER = False`.

## 9. Reverify dữ liệu human review (§8.6 notebook)

Khi đã có file labelled (vd. `hasaki_labelled_clean.json`) và chỉ muốn verify lại **hàng đợi human review** (không chạy ~2000 mẫu), dùng cell §8.6 trong [`data/code/intent_labeling_gpt4.ipynb`](../data/code/intent_labeling_gpt4.ipynb):

- Lọc theo `HUMAN_REVIEW_QA_STATUSES` (`needs_review`, `pending_new_label_review`).
- **Không** gọi `predict_intent` (không annotate lại).
- `apply_verifier(..., force=True)` — bỏ qua ngưỡng `VERIFY_CONF_THRESHOLD` cho từng mẫu trong queue.
- `save_annotation_with_guardrails(..., persist=False)` — dry-run guardrail, không ghi MongoDB.
- Xuất `*_human_review_reverified.json` với `qa_status_before_reverify`, `intent_3_level_before_reverify`, metadata verifier.

Báo cáo: `python scripts/generate_verifier_report.py --input <file_human_review_reverified.json> --outdir data/audit`.

## 9b. Full reverify toàn bộ dataset (Colab notebook riêng)

**Không chạy trong notebook gán nhãn chính.**

Mở trên Google Colab: [`data/code/run_full_reverify_colab.ipynb`](../data/code/run_full_reverify_colab.ipynb)

1. Mount Drive → set `INTENT_REPO=/content/drive/MyDrive/intent_kb`
2. Nhập `MONGODB_URI`, `OPENAI_API_KEY`
3. Chạy full reverify (~2011 mẫu, checkpoint mỗi 50; `RESUME=True` nếu disconnect)
4. Output: `data/raw/hasaki/hasaki_labelled_full_verified.json`

Local CLI (tuỳ chọn): `python scripts/run_full_reverify.py`

Sau đó: `python scripts/build_d2_verified_dataset.py --mode full`

Lưu ý chi phí: ~2011 lần gọi GPT-4o.

## 10. Đánh giá (A/B)

Chạy thử subset Hasaki, so sánh trước/sau verifier:
- Tỷ lệ REVISE / RETAIN.
- Thay đổi phân phối `qa_status` (approved / needs_review / pending).
- Cost (số lần gọi verifier = số mẫu conf < 0.85).
- (Nếu có gold) thay đổi agreement / precision.
