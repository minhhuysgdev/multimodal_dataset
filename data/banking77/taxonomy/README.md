# BANKING77 hierarchical taxonomy (research-constructed)

Ontology **L1 → L2 → L3** cho bộ dữ liệu [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) (13.083 utterances, 77 flat intents, tiếng Anh). Đây **không** phải taxonomy chính thức của PolyAI; L3 giữ nguyên tên và `label_id` gốc (0–76).

## Quy ước tầng

| Tầng | Vai trò |
|------|---------|
| **BANKING** | Root ngữ cảnh miền; nằm **ngoài** L1–L2–L3 |
| **L1** | Nhóm dịch vụ / product line (card, transfer, top_up, …) |
| **L2** | Intent family trong cùng L1 (ví dụ `card_payment`, `transfer_status`) |
| **L3** | Intent gốc BANKING77 — **không đổi** định danh hay semantic |

Nguyên tắc bảo toàn: mỗi intent gốc có **đúng một** path `L1 → L2 → L3`; không gộp hay tách L3.

## Artifacts

| File | Mô tả |
|------|--------|
| [`banking77_hierarchy.csv`](banking77_hierarchy.csv) | 77 hàng: `label_id`, `L3`, `L2`, `L1`, `flag`, `rationale` |
| [`banking77_hierarchy.json`](banking77_hierarchy.json) | `flat`, `tree`, `coverage`, `statistics`, `audit` |
| [`build_banking77_hierarchy.py`](build_banking77_hierarchy.py) | Script tái tạo CSV/JSON và chạy audit |
| [`banking77_taxonomy_review.html`](banking77_taxonomy_review.html) | **Menu L1 → L2 → L3** — chọn từng tầng, tìm kiếm, lọc `flag` |
| [`TAXONOMY_REVIEW.md`](TAXONOMY_REVIEW.md) | Hướng dẫn review + Mermaid overview |
| [`diagrams/`](diagrams/) | Mermaid: overview + 8 sơ đồ chi tiết theo L1 |

Tái tạo:

```bash
python3 data/banking77/taxonomy/build_banking77_hierarchy.py
```

## Coverage (audit)

| Kiểm tra | Kết quả |
|----------|---------|
| Số L3 | 77 |
| `label_id` | 0–76, đủ, không trùng |
| Khớp tên L3 Hugging Face | Có |
| Path trùng lặp | 0 |
| Orphan intent | 0 |

### Phân bố L1

| L1 | Số L3 |
|----|------:|
| `card` | 27 |
| `top_up` | 13 |
| `transfer` | 10 |
| `cash_withdrawal` | 8 |
| `account_and_access` | 6 |
| `refund_and_charges` | 5 |
| `currency_exchange` | 4 |
| `identity_verification` | 4 |

**L2:** 21 families (mỗi L2 có từ 2 đến 8 L3; không còn L2 một con).

## L2 index (theo L1)

### `card`

- `card_issuance` — phát hành, giao thẻ, kích hoạt, liên kết, thẻ sắp hết hạn
- `virtual_card` — thẻ ảo / disposable
- `card_security` — PIN, mất cắp, compromised
- `card_function` — thẻ hoặc contactless không hoạt động
- `card_payment` — thanh toán thẻ: pending / declined / fee / FX
- `card_product_info` — chấp nhận thẻ, Visa/Mastercard, wallet

### `cash_withdrawal`

- `atm_operation` — ATM, pending/declined, thẻ bị nuốt, phí rút tiền
- `withdrawal_discrepancy` — không nhận diện, sai số tiền, sai tỷ giá khi rút

### `transfer`

- `transfer_setup` — beneficiary, nhận/chuyển vào tài khoản
- `transfer_status` — pending, failed, declined, cancel, recipient
- `transfer_info_and_fee` — thời gian, phí chuyển

### `top_up`

- `top_up_method` — auto, card, cash/cheque, thẻ/tiền tệ được chấp nhận, xác minh thẻ nạp
- `top_up_status` — pending/failed/reverted, số dư chưa cập nhật sau nạp
- `top_up_fee_and_limit` — phí và hạn mức nạp

### `currency_exchange`

Đổi tiền và loại tiền tệ (trước đây tên `fx_and_currency`; **fx** = foreign exchange).

- `exchange_info` — tỷ giá, đổi qua app, loại tiền hỗ trợ, phí đổi tiền

### `refund_and_charges`

- `refund` — yêu cầu / refund chưa hiện
- `unrecognised_or_extra_charge` — direct debit, extra charge, double charge

### `identity_verification`

- `kyc` — verify, why, unable, source of funds

### `account_and_access`

- `account_lifecycle` — sửa thông tin, đóng tài khoản
- `eligibility` — age limit, country support
- `app_access_security` — passcode, mất điện thoại

## Quyết định đã chốt (theo utterance)

Các intent từng có `flag` đã được đặt lại sau khi đọc câu mẫu PolyAI `train.csv`. Cột `flag` hiện để trống.

| L3 | Path đã chốt | Căn cứ utterance |
|----|----------------|------------------|
| `card_about_to_expire` | `card` / `card_issuance` | Hỏi cách lấy thẻ mới khi hết hạn, không phải thẻ hỏng |
| `exchange_charge` | `currency_exchange` / `exchange_info` | Phí đổi ngoại tệ, cùng family với tỷ giá |
| `balance_not_updated_after_bank_transfer` | `top_up` / `top_up_status` | Chuyển từ ngân hàng khác vào mà số dư chưa hiện |
| `card_swallowed` | `cash_withdrawal` / `atm_operation` | ATM nuốt thẻ |
| `cash_withdrawal_charge` | `cash_withdrawal` / `atm_operation` | Phí khi rút ATM, không phải sai số tiền |
| `supported_cards_and_currencies` | `top_up` / `top_up_method` | Thẻ / loại tiền dùng để nạp |
| `verify_top_up` | `top_up` / `top_up_method` | Mã xác minh thẻ dùng để nạp, không phải KYC |

## Trích dẫn

```bibtex
@inproceedings{Casanueva2020,
  author    = {I{\~{n}}igo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulic},
  title     = {Efficient Intent Detection with Dual Sentence Encoders},
  year      = {2020},
  booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020},
  url       = {https://arxiv.org/abs/2003.04807}
}
```

Dataset: [https://huggingface.co/datasets/PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) (CC BY 4.0).

## Phạm vi hiện tại

- Chưa gán L1/L2 lên từng utterance train/test.
- Chưa chạy human agreement trên taxonomy.
- Liên quan slide báo cáo: [`docs/slides/banking77_taxonomy_progress.tex`](../../../docs/slides/banking77_taxonomy_progress.tex).
