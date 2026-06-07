# Tái tạo `unified_intents.csv` và import lại MongoDB

## Đã làm (2026-05-28)

1. Script `scripts/build_unified_intents_from_mongodb_export.py` đọc `data/intent_kb.intent_nodes_27_05_2026.csv`.
2. Gộp `level=intent` theo `(L1, L2, L3)` đã chuẩn hóa (`truoc ban` → `truoc_mua_hang`).
3. Giữ `INT-1`…`INT-132` nếu catalog đã có; gán `INT-133`… cho triple mới (từ AUTO).
4. Output:
   - `data/unified_intents.csv` — **286** dòng intent
   - `data/unified_intents.csv.bak` — backup 109 dòng cũ
   - `data/audit/intent_id_migration_map.csv` — map `AUTO-*` / `INT-*` cũ → `Ma Intent` mới

## Import MongoDB (Colab)

1. Upload / sync `data/unified_intents.csv` lên Drive (`intent_kb/data/`).
2. Mở `intent_graph_rag_colab.ipynb`:
   - §2 MongoDB
   - §3 `CSV_PATH` trỏ đúng file
   - §5 **Đồng bộ graph** (drop `intent_nodes` + `intent_edges` → insert lại)
3. `intent_labeling_gpt4.ipynb` §7: `embed_and_store_intent_nodes(db, force=True)`.

## Dữ liệu đã gán nhãn

- File `hasaki_labelled_*.json` dùng **`level_1/2/3` slug** → **không cần label lại** nếu slug không đổi.
- `candidate_intents[].intent_id` có thể còn `AUTO-*` (metadata cũ).
- Collection `intent_annotations` **không** bị xóa khi chạy §5; field `semantic_snap_intent_id` có thể lệch → chỉ ảnh hưởng audit snap.

## Thống kê lần build

| | Số lượng |
|---|----------|
| Intent trong export MongoDB | 311 (109 INT + 202 AUTO) |
| Sau gộp triple | 286 |
| Catalog giữ INT ≤132 | 109 |
| Intent mới INT ≥133 | 177 |
