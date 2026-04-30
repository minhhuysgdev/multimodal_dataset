#!/usr/bin/env python3
"""Generate intent_labeling_mongodb_qwen.ipynb (run from repo root)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "intent_labeling_mongodb_qwen.ipynb"


def cell_md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def cell_code(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


HELPERS = r'''from datetime import datetime, timezone
import re


def upsert_new_label_to_graph(
    db,
    *,
    domain,
    l1,
    l2,
    l3,
    intent_id,
    intent_name,
    description="",
    confidence="",
    product_category="",
    detection_signals="",
    example="",
    category_logic="",
):
    col_nodes = db[COL_NODES]
    col_edges = db[COL_EDGES]
    domain_id = f"domain:{domain}"
    l1_id, l2_id, l3_id = f"l1:{l1}", f"l2:{l2}", f"l3:{l3}"
    now = datetime.now(timezone.utc)

    for node_id, level, name, extra in [
        ("ROOT", "root", "Intent Knowledge Base", {}),
        (domain_id, "domain", domain, {}),
        (l1_id, "l1", l1, {}),
        (l2_id, "l2", l2, {}),
        (l3_id, "l3", l3, {}),
        (
            intent_id,
            "intent",
            intent_name,
            {
                "description": description,
                "confidence": confidence,
                "domain": domain,
                "product_category": product_category,
                "l1": l1,
                "l2": l2,
                "l3": l3,
                "detection_signals": detection_signals,
                "example": example,
                "category_logic": category_logic,
            },
        ),
    ]:
        col_nodes.update_one(
            {"_id": node_id},
            {"$setOnInsert": {"_id": node_id, "level": level}, "$set": {"name": name, **extra, "updated_at": now}},
            upsert=True,
        )

    for src, dst in [
        ("ROOT", domain_id),
        (domain_id, l1_id),
        (l1_id, l2_id),
        (l2_id, l3_id),
        (l3_id, intent_id),
    ]:
        eid = f"{src}->{dst}"
        col_edges.update_one(
            {"_id": eid},
            {"$setOnInsert": {"_id": eid, "from": src, "to": dst}, "$set": {"updated_at": now}},
            upsert=True,
        )
    return {"intent_id": intent_id, "status": "upserted"}


def _collect_allowed_taxonomy(db):
    l1_set, l2_set, l3_set = set(), set(), set()
    for doc in db[COL_NODES].find({"level": "l1"}, {"name": 1}):
        if doc.get("name"):
            l1_set.add(doc["name"])
    for doc in db[COL_NODES].find({"level": "l2"}, {"name": 1}):
        if doc.get("name"):
            l2_set.add(doc["name"])
    for doc in db[COL_NODES].find({"level": "l3"}, {"name": 1}):
        if doc.get("name"):
            l3_set.add(doc["name"])
    return {"l1": l1_set, "l2": l2_set, "l3": l3_set}


def _is_label_allowed(pred, allowed):
    l1 = (pred.get("L1") or "").strip()
    l2 = (pred.get("L2") or "").strip()
    l3 = (pred.get("L3") or "").strip()
    return (l1 in allowed["l1"]) and (l2 in allowed["l2"]) and (l3 in allowed["l3"])


def save_annotation_with_guardrails(
    db,
    sample,
    pred,
    *,
    min_conf_auto_approve=MIN_CONF_APPROVE_EXISTING,
    min_conf_allow_new_label=MIN_CONF_AUTO_ADD_NEW_LABEL,
    allow_auto_add_new_label=ALLOW_AUTO_ADD_NEW_LABEL,
    model_name=MODEL_ID,
):
    col_ann = db["intent_annotations"]
    now = datetime.now(timezone.utc)
    allowed = _collect_allowed_taxonomy(db)
    confidence = float(pred.get("confidence") or 0)
    valid_existing = _is_label_allowed(pred, allowed)

    qa_status = "rejected"
    new_label_pending_review = False
    graph_upserted = False

    if valid_existing:
        qa_status = "approved" if confidence >= min_conf_auto_approve else "needs_review"
    else:
        new_label_pending_review = True
        qa_status = "pending_new_label_review"
        if allow_auto_add_new_label and confidence >= min_conf_allow_new_label:
            upsert_new_label_to_graph(
                db,
                domain=sample.get("domain") or infer_domain(sample) or "Unknown",
                l1=pred.get("L1", "").strip(),
                l2=pred.get("L2", "").strip(),
                l3=pred.get("L3", "").strip(),
                intent_id=sample.get("intent_id") or f"AUTO-{sample.get('sample_id', 'UNKNOWN')}",
                intent_name=pred.get("L3", "").strip() or "new_intent",
                description=pred.get("reasoning", ""),
                confidence=str(confidence),
                product_category=sample.get("category", ""),
                detection_signals=sample.get("sentence", ""),
                example=sample.get("sentence", ""),
                category_logic="auto_added_from_labeling_notebook",
            )
            graph_upserted = True
            qa_status = "approved_auto_new_label"

    ann_doc = {
        "sample_id": sample.get("sample_id"),
        "product_id": sample.get("product_id"),
        "sentence": sample.get("sentence", ""),
        "category": sample.get("category", ""),
        "model": model_name,
        "intent": {
            "level_1": pred.get("L1", "").strip(),
            "level_2": pred.get("L2", "").strip(),
            "level_3": [pred.get("L3", "").strip()] if pred.get("L3") else [],
        },
        "confidence": confidence,
        "reasoning": pred.get("reasoning", ""),
        "qa_status": qa_status,
        "new_label_pending_review": new_label_pending_review,
        "graph_upserted": graph_upserted,
        "source": sample.get("source", "hasaki"),
        "updated_at": now,
    }
    col_ann.update_one({"sample_id": ann_doc["sample_id"]}, {"$set": ann_doc}, upsert=True)
    return ann_doc


def get_candidate_l3_from_mongodb(db, text, category=None, top_k=10):
    tokens = [t.lower() for t in re.findall(r"\w+", text or "", flags=re.UNICODE) if len(t) >= 2]
    if not tokens:
        return []

    or_conditions = []
    for tok in tokens:
        rgx = {"$regex": re.escape(tok), "$options": "i"}
        or_conditions.extend(
            [
                {"name": rgx},
                {"description": rgx},
                {"detection_signals": rgx},
                {"example": rgx},
                {"l3": rgx},
            ]
        )

    query = {"level": "intent", "$or": or_conditions}
    docs = list(
        db[COL_NODES].find(
            query,
            {
                "_id": 1,
                "name": 1,
                "l1": 1,
                "l2": 1,
                "l3": 1,
                "description": 1,
                "detection_signals": 1,
                "example": 1,
                "product_category": 1,
            },
        )
    )

    scored = []
    token_set = set(tokens)
    for d in docs:
        blob = " ".join(
            [
                str(d.get("name", "")),
                str(d.get("description", "")),
                str(d.get("detection_signals", "")),
                str(d.get("example", "")),
                str(d.get("l3", "")),
                str(d.get("product_category", "")),
            ]
        ).lower()
        hit = sum(1 for t in token_set if t in blob)
        scored.append((hit, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k] if s > 0]


def build_retrieval_prompt(sample, candidates):
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. L1={c.get('l1')} | L2={c.get('l2')} | L3={c.get('l3')} | intent_id={c.get('_id')}"
        )
    taxonomy_text = "\n".join(lines) if lines else "KHONG TIM THAY CANDIDATE — van tra JSON, is_new_label=true neu khong khop"

    parts = [
        "Ban la bo gan nhan intent (e-commerce my pham / dien tu).",
        "Chi duoc chon L1, L2, L3 khop mot dong trong danh sach candidate.",
        "Neu khong co dong nao hop ly, dat is_new_label=true.",
        "",
        "Candidate (tu MongoDB):",
        taxonomy_text,
        "",
        "Ngu canh:",
        f"- sentence: {sample.get('sentence', '')}",
        f"- category: {sample.get('category', '')}",
        f"- product_name: {sample.get('product_name', '')}",
        f"- brand: {sample.get('brand', '')}",
        "",
        "Tra ve DUY NHAT mot JSON (khong markdown):",
        "{",
        '  "L1": "...",',
        '  "L2": "...",',
        '  "L3": "...",',
        '  "confidence": 0.0,',
        '  "reasoning": "...",',
        '  "is_new_label": false',
        "}",
    ]
    return "\n".join(parts)


def infer_domain(sample):
    cat = (sample.get("category") or "").lower()
    name = (sample.get("product_name") or "").lower()
    blob = cat + " " + name
    electronics_kw = (
        "laptop phone dien thoai tai nghe chuot ban phim camera smartwatch "
        "gaming console ram gpu cpu may anh dong ho thong minh"
    )
    if any(k in blob for k in electronics_kw.split()):
        return "Dien tu"
    return "My pham"


print("Helpers OK")
'''


def main():
    cells = [
        cell_md(
            """# Intent labeling — MongoDB + Qwen2.5 + guardrail

Pipeline: **truy hồi ứng viên từ MongoDB** → **Qwen sinh JSON** → **kiểm taxonomy + ngưỡng** → lưu `intent_annotations`.

Xem `LABELING_GUIDE.md` (auto-add nhãn mới mặc định **confidence ≥ 0.96**).

**Điều kiện:** graph đã có trong MongoDB (`intent_nodes` / `intent_edges`) — build từ `data/intent_graph_rag_colab.ipynb`.
"""
        ),
        cell_md("## 1. Cài đặt\n"),
        cell_code(
            """# %pip install -q pymongo 'pymongo[srv]' certifi transformers accelerate torch sentencepiece protobuf
import sys
print(sys.executable)
"""
        ),
        cell_md("## 2. Cấu hình\n"),
        cell_code(
            """import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("INTENT_REPO", ".")).resolve()
HASAKI_JSON = REPO_ROOT / "data/raw/hasaki/hasaki_prelabel.json"

DB_NAME = os.environ.get("MONGODB_DB", "intent_kb")
COL_NODES = "intent_nodes"
COL_EDGES = "intent_edges"
MONGODB_URI = os.environ.get("MONGODB_URI", "")

MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8")

MIN_CONF_APPROVE_EXISTING = 0.90
MIN_CONF_AUTO_ADD_NEW_LABEL = 0.96
ALLOW_AUTO_ADD_NEW_LABEL = False

BATCH_START = 0
BATCH_LIMIT = 5
TOP_K_CANDIDATES = 12
MAX_NEW_TOKENS = 384

print("HASAKI_JSON:", HASAKI_JSON.exists(), HASAKI_JSON)
print("MONGODB_URI set:", bool(MONGODB_URI))
"""
        ),
        cell_md("## 3. Kết nối MongoDB\n"),
        cell_code(
            """import certifi
from pymongo import MongoClient

if not MONGODB_URI:
    MONGODB_URI = input("Nhap MONGODB_URI: ").strip()

client = MongoClient(
    MONGODB_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=15000,
)
client.admin.command("ping")
db = client[DB_NAME]
print("OK db:", db.name)
print("intent_nodes:", db[COL_NODES].count_documents({}))
print("intent_edges:", db[COL_EDGES].count_documents({}))
"""
        ),
        cell_md("## 4. Helper: upsert graph, guardrail, retrieval\n"),
        cell_code(HELPERS),
        cell_md("## 5. Tải Qwen\n"),
        cell_code(
            """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
print("device", next(model.parameters()).device)
"""
        ),
        cell_md("## 6. Sinh nhãn + parse JSON\n"),
        cell_code(
            """import json as json_lib
import torch


def extract_json_object(text):
    text = (text or "").strip()
    if "```" in text:
        for chunk in text.split("```"):
            c = chunk.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c.startswith("{"):
                text = c
                break
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Khong tim thay JSON")
    return json_lib.loads(text[start : end + 1])


@torch.inference_mode()
def predict_intent(sample, top_k=None):
    if top_k is None:
        top_k = TOP_K_CANDIDATES
    cands = get_candidate_l3_from_mongodb(
        db, sample.get("sentence", ""), category=sample.get("category"), top_k=top_k
    )
    user_prompt = build_retrieval_prompt(sample, cands)
    messages = [
        {"role": "system", "content": "Chi tra ve JSON hop le, khong markdown."},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(gen, skip_special_tokens=True)
    pred = extract_json_object(raw)
    return pred, raw, cands


def enrich_sample(s):
    out = dict(s)
    out["domain"] = infer_domain(s)
    out["intent_id"] = f"AUTO-{s.get('sample_id', 'x')}"
    return out
"""
        ),
        cell_md("## 7. Thử một mẫu\n"),
        cell_code(
            """sample_demo = enrich_sample(
    {
        "sample_id": "demo_q",
        "product_id": "demo",
        "product_name": "Bang che khuyet diem",
        "brand": "Judydoll",
        "category": "Trang diem",
        "sentence": "Dung co de bi moc khong?",
        "source": "demo",
    }
)

pred, raw, cands = predict_intent(sample_demo)
print("candidates:", len(cands))
print(raw[:1200])
print("parsed:", pred)
ann = save_annotation_with_guardrails(db, sample_demo, pred)
print(ann["qa_status"], ann.get("graph_upserted"))
"""
        ),
        cell_md(
            """## 8. Batch Hasaki

`hasaki_prelabel.json` rất lớn — chỉ `json.load` khi đủ RAM; giữ `BATCH_LIMIT` nhỏ khi test.
"""
        ),
        cell_code(
            """def load_hasaki_slice(path: Path, start: int, limit: int):
    with open(path, "r", encoding="utf-8") as f:
        data = json_lib.load(f)
    return data[start : start + limit]


def run_batch(samples):
    stats = {}
    for s in samples:
        ex = enrich_sample(s)
        try:
            pred, raw, cands = predict_intent(ex)
            ann = save_annotation_with_guardrails(db, ex, pred)
            st = ann["qa_status"]
            stats[st] = stats.get(st, 0) + 1
            print(st, ex.get("sample_id"), pred.get("L3"), ann.get("confidence"))
        except Exception as e:
            stats["error"] = stats.get("error", 0) + 1
            print("ERROR", ex.get("sample_id"), e)
    return stats


if HASAKI_JSON.exists():
    batch = load_hasaki_slice(HASAKI_JSON, BATCH_START, BATCH_LIMIT)
    print("Batch size:", len(batch))
    print("Stats:", run_batch(batch))
else:
    print("Khong tim thay HASAKI_JSON")
"""
        ),
        cell_md(
            """## Ghi chú

- Index gợi ý: `db.intent_annotations.create_index("sample_id", unique=True)`.
- `ALLOW_AUTO_ADD_NEW_LABEL=True` + conf ≥ 0.96 → upsert nhãn mới vào graph.
"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }

    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
