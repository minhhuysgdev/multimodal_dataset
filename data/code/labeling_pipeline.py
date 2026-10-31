"""Shared labeling pipeline (extracted from intent_labeling_gpt4.ipynb).

Do NOT run full reverify in the main notebook — use scripts/run_full_reverify.py instead.

Requires env: MONGODB_URI, OPENAI_API_KEY
Optional: INTENT_REPO (defaults to repo root when run via script)
"""
from __future__ import annotations

import os
import getpass
from pathlib import Path

REPO_ROOT = Path(os.environ.get("INTENT_REPO", ".")).resolve()
HASAKI_JSON = REPO_ROOT / "data/hasaki_prelabel.json"
# Output (checkpoint + file labelled cuoi). Tren Colab REPO_ROOT da tro vao Drive,
# nen OUTPUT_DIR mac dinh cung nam tren Drive -> khong mat khi runtime disconnect.
# Co the override bang env INTENT_OUTPUT_DIR (vd. /content/drive/MyDrive/khac/outputs).
_out_env = (os.environ.get("INTENT_OUTPUT_DIR") or "").strip()
OUTPUT_DIR = Path(_out_env).resolve() if _out_env else (REPO_ROOT / "data" / "outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_NAME = os.environ.get("MONGODB_DB", "intent_kb")
COL_NODES = "intent_nodes"
COL_EDGES = "intent_edges"
# Khong hardcode URI: uu tien bien moi truong MONGODB_URI, neu trong thi nhap khi chay cell (an khi go)
MONGODB_URI = (os.environ.get("MONGODB_URI") or "").strip()
if not MONGODB_URI:
    MONGODB_URI = getpass.getpass("MONGODB_URI (dan vao roi Enter): ").strip()

# === LLM: OpenAI gpt-4o-mini (API) ===
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0"))
OPENAI_MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
# OPENAI_API_KEY: set env trước khi import (Colab/script) hoặc nhập getpass
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
if not OPENAI_API_KEY:
    OPENAI_API_KEY = getpass.getpass("OPENAI_API_KEY (dan vao roi Enter): ").strip()

MIN_CONF_APPROVE_EXISTING = 0.85
MIN_CONF_AUTO_ADD_NEW_LABEL = 0.80
ALLOW_AUTO_ADD_NEW_LABEL = True

# === Verifier (cross-verification, §6.5) ===
# Annotator: gpt-4o-mini | Verifier: gpt-4o (model manh hon, doc lai nhan).
ENABLE_VERIFIER = bool(int(os.environ.get("ENABLE_VERIFIER", "1")))
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", "gpt-4o")
# VERIFY_ALL_SAMPLES=1: verify moi mau (khop diagram 2-step). Mac dinh 0: chi verify khi conf thap.
VERIFY_ALL_SAMPLES = bool(int(os.environ.get("VERIFY_ALL_SAMPLES", "0")))
# Chi goi verifier khi confidence < nguong nay (bo qua neu VERIFY_ALL_SAMPLES=1).
VERIFY_CONF_THRESHOLD = float(os.environ.get("VERIFY_CONF_THRESHOLD", "0.85"))
VERIFY_MAX_TOKENS = int(os.environ.get("VERIFY_MAX_TOKENS", "256"))

BATCH_START = 0
HASAKI_BATCH_SIZE = 200
# Resume chay full prelabel: index bat dau (0 = tu dau file)
HASAKI_RUN_OFFSET = 0
# Sau khi export JSON: xoa BATCH_OUTPUT khoi RAM (checkpoint van tren dia)
HASAKI_FREE_MEM_AFTER_EXPORT = True
BATCH_LIMIT = HASAKI_BATCH_SIZE
TOP_K_CANDIDATES = 5
MAX_NEW_TOKENS = 384

# Human review target statuses (xuất ra để người rà soát sau pipeline).
_human_review_env = os.environ.get(
    "HUMAN_REVIEW_QA_STATUSES", "needs_review,pending_new_label_review"
)
HUMAN_REVIEW_QA_STATUSES = {
    s.strip() for s in _human_review_env.split(",") if s.strip()
}

print("HASAKI_JSON:", HASAKI_JSON.exists(), HASAKI_JSON)
print("OUTPUT_DIR :", OUTPUT_DIR)
print("MONGODB_URI set:", bool(MONGODB_URI))
print("LLM model:", OPENAI_MODEL)
print("Verifier:", "ON" if ENABLE_VERIFIER else "OFF", "| model:", VERIFIER_MODEL,
      "| verify all:", VERIFY_ALL_SAMPLES,
      "| verify if conf <", VERIFY_CONF_THRESHOLD if not VERIFY_ALL_SAMPLES else "N/A")
print("Human review statuses:", sorted(HUMAN_REVIEW_QA_STATUSES))


import certifi
from pymongo import MongoClient

if not MONGODB_URI:
    raise ValueError(
        "Chua co MONGODB_URI. Dat bien moi truong MONGODB_URI hoac chay lai cell cau hinh de nhap."
    )

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


# LLM client — OpenAI gpt-4o-mini (chạy cell này trước §6 / §8)
from openai import OpenAI

openai_client = OpenAI(api_key=OPENAI_API_KEY)
print(f"✅ OpenAI API model: {OPENAI_MODEL}")


from datetime import datetime, timezone
import json
import re
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ngưỡng snap nhãn intent có sẵn qua cosine vs embedding đã enrich (đối xứng với retrieval)
SEMANTIC_SNAP_TO_INTENT_THRESHOLD = 0.85


# Global semantic encoder - lazy loaded
_semantic_model = None

def get_semantic_model():
    """Lazy load sentence transformer model"""
    global _semantic_model
    if _semantic_model is None:
        print("Loading paraphrase-multilingual-MiniLM-L12-v2...")
        _semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _semantic_model


def build_graph_aware_intent_text(intent_doc, db):
    """Build graph-aware description for intent node embedding
    
    Args:
        intent_doc: intent node document from MongoDB
        db: MongoDB database connection
    
    Returns:
        str: graph-aware text for embedding
    """
    # Extract path components
    l1 = intent_doc.get('l1', '')
    l2 = intent_doc.get('l2', '') 
    l3 = intent_doc.get('l3', '')
    description = intent_doc.get('description', '')
    detection_signals = intent_doc.get('detection_signals', '')
    example = intent_doc.get('example', '')
    
    # Build full path
    full_path = f"{l1}.{l2}.{l3}"
    
    # Get sibling intents (same L2, limit 5)
    siblings_cursor = db[COL_NODES].find(
        {'level': 'intent', 'l2': l2, '_id': {'$ne': intent_doc['_id']}}, 
        {'l3': 1}
    ).limit(5)
    siblings = [doc.get('l3', '') for doc in siblings_cursor if doc.get('l3')]
    siblings_text = ', '.join(siblings) if siblings else 'None'
    
    # Construct graph-aware text
    graph_text = f"""Intent: {full_path}.
Parent: {l2}.
Siblings: {siblings_text}.
Description: {description}
Signals: {detection_signals}
Example: {example}"""
    
    return graph_text


def embed_and_store_intent_nodes(db, *, force=False):
    """Embedding enrichment cho toan bo intent nodes.

    - force=False (mac dinh): bo qua node da co `embedding` + `graph_aware_text`.
    - force=True: re-embed TAT CA (dung sau khi rebuild graph tu CSV moi,
      doi sentence-transformer model, hoac sua build_graph_aware_intent_text).
    """
    model = get_semantic_model()
    col_nodes = db[COL_NODES]

    intent_nodes = list(col_nodes.find({'level': 'intent'}))
    print(f"Found {len(intent_nodes)} intent nodes to process (force={force})")

    updated_count = 0
    skipped_count = 0
    for intent_doc in intent_nodes:
        if not force and 'embedding' in intent_doc and 'graph_aware_text' in intent_doc:
            skipped_count += 1
            continue

        graph_text = build_graph_aware_intent_text(intent_doc, db)

        # Generate embedding
        embedding = model.encode([graph_text])[0]
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Store in MongoDB
        col_nodes.update_one(
            {'_id': intent_doc['_id']},
            {
                '$set': {
                    'graph_aware_text': graph_text,
                    'embedding': embedding.tolist(),  # Convert to list for MongoDB
                    'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
                    'embedding_updated_at': datetime.now(timezone.utc)
                }
            }
        )
        updated_count += 1

        if updated_count % 20 == 0:
            print(f"Processed {updated_count} nodes...")

    print(f"✅ Embedded {updated_count} | ⏭️ Skipped {skipped_count} (đã có embedding) | Total {len(intent_nodes)}")
    if skipped_count == len(intent_nodes) and not force:
        print("   ⚠️ Tất cả node đã có embedding cũ. Sau khi đổi taxonomy/CSV → gọi `embed_and_store_intent_nodes(db, force=True)` để rebuild.")
    return updated_count


def semantic_retrieval(db, query_text, top_k=10, min_similarity=0.3):
    """Semantic retrieval using graph-aware embeddings
    
    Args:
        db: MongoDB database connection
        query_text: user query sentence
        top_k: number of candidates to return
        min_similarity: minimum cosine similarity threshold
    
    Returns:
        list: intent documents with similarity scores
    """
    model = get_semantic_model()
    col_nodes = db[COL_NODES]
    
    # Encode query
    query_embedding = model.encode([query_text])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Find intent nodes with embeddings
    intent_nodes = list(col_nodes.find(
        {'level': 'intent', 'embedding': {'$exists': True}},
        {
            '_id': 1, 'name': 1, 'l1': 1, 'l2': 1, 'l3': 1,
            'description': 1, 'detection_signals': 1, 'example': 1,
            'product_category': 1, 'embedding': 1
        }
    ))

    if not intent_nodes:
        print("Warning: No intent nodes with embeddings found")
        return []

    # Calculate similarities
    scored_candidates = []
    for intent_doc in intent_nodes:
        stored_embedding = np.array(intent_doc['embedding'])

        # Compute cosine similarity
        similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
        
        if similarity >= min_similarity:
            # Add similarity score to document
            intent_doc['semantic_score'] = float(similarity)
            scored_candidates.append(intent_doc)
    
    # Sort by similarity score descending
    scored_candidates.sort(key=lambda x: x['semantic_score'], reverse=True)
    
    return scored_candidates[:top_k]


def union_retrieval(db, text, category=None, top_k_regex=8, top_k_semantic=6, sample=None):
    """UNION of regex and semantic retrieval with deduplication
    
    Args:
        db: MongoDB database connection
        text: query text
        category: product category (optional)
        top_k_regex: max regex candidates
        top_k_semantic: max semantic candidates
        sample: sample data for context matching
    
    Returns:
        list: deduplicated union of candidates
    """
    # Get regex candidates (existing logic)
    regex_candidates = get_candidate_l3_from_mongodb(db, text, category, top_k_regex, sample)
    
    # Get semantic candidates (new logic)
    semantic_candidates = semantic_retrieval(db, text, top_k_semantic)
    
    # Deduplicate by _id, preserving document structure
    seen_ids = set()
    union_candidates = []
    
    # Add regex candidates first (preserve original scoring)
    for candidate in regex_candidates:
        if candidate['_id'] not in seen_ids:
            candidate['retrieval_method'] = 'regex'
            union_candidates.append(candidate)
            seen_ids.add(candidate['_id'])
    
    # Add semantic candidates
    for candidate in semantic_candidates:
        if candidate['_id'] not in seen_ids:
            candidate['retrieval_method'] = 'semantic'
            union_candidates.append(candidate)
            seen_ids.add(candidate['_id'])
    
    # Limit total candidates
    max_total = max(top_k_regex + top_k_semantic, TOP_K_CANDIDATES)
    return union_candidates[:max_total]


def find_nearest_existing_intent(db, pred_text, threshold=SEMANTIC_SNAP_TO_INTENT_THRESHOLD):
    """Embed prediction path/slug; cosine vs intent node embeddings. Returns (node, sim) or (None, best_sim)."""
    text = (pred_text or "").strip()
    if not text:
        return None, -1.0
    model = get_semantic_model()
    q = np.asarray(model.encode([text])[0], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return None, -1.0
    q = q / n
    nodes = list(
        db[COL_NODES].find(
            {"level": "intent", "embedding": {"$exists": True}},
            {"_id": 1, "l1": 1, "l2": 1, "l3": 1, "embedding": 1},
        )
    )
    best, best_sim = None, -1.0
    for intent_doc in nodes:
        ev = np.asarray(intent_doc["embedding"], dtype=np.float64)
        en = np.linalg.norm(ev)
        if en < 1e-9:
            continue
        sim = float(np.dot(q, ev / en))
        if sim > best_sim:
            best_sim, best = sim, intent_doc
    if best is not None and best_sim >= threshold:
        return best, best_sim
    return None, best_sim


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


def _norm_label(s):
    s = unicodedata.normalize("NFC", (s or "").strip())
    return re.sub(r"\s+", " ", s).lower()


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

    # norm -> chuỗi canonical trong Mongo (tranh trùng key: lấy deterministic theo sort)
    def _norm_to_canon(name_set):
        m = {}
        for name in sorted(name_set):
            k = _norm_label(name)
            if k and k not in m:
                m[k] = name
        return m

    return {
        "l1": l1_set,
        "l2": l2_set,
        "l3": l3_set,
        "norm_canon_l1": _norm_to_canon(l1_set),
        "norm_canon_l2": _norm_to_canon(l2_set),
        "norm_canon_l3": _norm_to_canon(l3_set),
    }


def _snap_pred_to_canonical_taxonomy(pred, allowed):
    """Ghi đè L1/L2/L3 bằng tên đúng trong graph nếu normalize khớp một nhãn có sẵn."""
    for lvl, key in [("l1", "L1"), ("l2", "L2"), ("l3", "L3")]:
        raw = pred.get(key)
        if raw is None:
            continue
        raw = str(raw).strip()
        if not raw:
            continue
        cmap = allowed.get(f"norm_canon_{lvl}", {})
        canon = cmap.get(_norm_label(raw))
        if canon:
            pred[key] = canon
    # KHÔNG ép L1 sang 'truoc_mua_hang' khi model trả slug domain/ngành làm L1
    # ('my pham', 'trang diem', 'gia_so_sanh', 'san_pham_so_sanh', ...).
    # Để pred giữ nguyên để _validate_pred_for_auto_create chặn upsert
    # và rơi vào pending_new_label_invalid_slug cho human review.


def _is_label_allowed(pred, allowed):
    n1, n2, n3 = (
        _norm_label(pred.get("L1")),
        _norm_label(pred.get("L2")),
        _norm_label(pred.get("L3")),
    )
    nl1 = {_norm_label(x) for x in allowed["l1"]}
    nl2 = {_norm_label(x) for x in allowed["l2"]}
    nl3 = {_norm_label(x) for x in allowed["l3"]}
    return (n1 in nl1) and (n2 in nl2) and (n3 in nl3)


# === Guardrail: validate L1/L2/L3 trước khi tạo intent mới trong graph ===
# Chỉ cho phép L1 ∈ {truoc_mua_hang, sau_mua_hang}; L2/L3 phải là slug ASCII lowercase + underscore.
# Reject các trường hợp model nhầm slug L2 (gia_so_sanh, ...) hoặc domain (My pham, Trang diem) vào L1.

_ALLOWED_L1_NORM_TO_CANON = {
    "truoc_mua_hang": "truoc_mua_hang",
    "truoc mua hang": "truoc_mua_hang",
    "truoc_ban": "truoc_mua_hang",
    "truoc ban": "truoc_mua_hang",
    "trước mua hàng": "truoc_mua_hang",
    "trước bán": "truoc_mua_hang",
    "trước ban": "truoc_mua_hang",
    "truoc bán": "truoc_mua_hang",
    "sau_mua_hang": "sau_mua_hang",
    "sau mua hang": "sau_mua_hang",
    "sau_ban": "sau_mua_hang",
    "sau ban": "sau_mua_hang",
    "sau mua hàng": "sau_mua_hang",
    "sau bán": "sau_mua_hang",
}

_L2_SLUGS_OFTEN_CONFUSED_AS_L1 = {
    "gia_so_sanh", "san_pham_so_sanh", "thiet_bi_tu_van", "chat_luong_loi",
    "di_ung_an_toan", "ton_kho_kiem_tra", "uu_dai_yeu_cau", "khieu_nai_dich_vu",
    "trang_thai_van_chuyen", "thay_doi_don_hang", "doi_tra", "hoan_tien_yeu_cau",
    "hong_vo_san_pham", "giao_hang_loi", "van_de_giao_hang", "thong_tin_san_pham",
    "tinh_nang_truy_van", "thong_sos_truy_van",
}

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]+$")


def _validate_pred_for_auto_create(pred):
    """Kiểm tra (L1, L2, L3) trước khi upsert intent mới vào graph.

    Trả về (valid: bool, reason: str, normalized: dict).
    Quy tắc:
    - L1 phải normalize được về 'truoc_mua_hang' hoặc 'sau_mua_hang'
    - L2 / L3: slug ASCII lowercase + underscore, không trùng nhau, không rỗng
    - L2 không được là một trong các slug L2 phổ biến mà model hay nhầm lên L1
    """
    out = {"L1": "", "L2": "", "L3": ""}
    l1_raw = (pred.get("L1") or "").strip()
    l2_raw = (pred.get("L2") or "").strip()
    l3_raw = (pred.get("L3") or "").strip()

    if not (l1_raw and l2_raw and l3_raw):
        return False, "empty_l1_l2_l3", out

    l1_canon = _ALLOWED_L1_NORM_TO_CANON.get(_norm_label(l1_raw))
    if not l1_canon:
        return False, f"l1_must_be_truoc_or_sau_mua_hang_got:{l1_raw!r}", out
    out["L1"] = l1_canon

    if not _SLUG_RE.match(l2_raw):
        return False, f"l2_invalid_slug:{l2_raw!r}", out
    if l2_raw in _ALLOWED_L1_NORM_TO_CANON:
        return False, f"l2_looks_like_l1:{l2_raw!r}", out
    out["L2"] = l2_raw

    if not _SLUG_RE.match(l3_raw):
        return False, f"l3_invalid_slug:{l3_raw!r}", out
    if l3_raw == l2_raw:
        return False, "l3_eq_l2", out
    if l3_raw in _L2_SLUGS_OFTEN_CONFUSED_AS_L1 or l3_raw in _ALLOWED_L1_NORM_TO_CANON:
        return False, f"l3_looks_like_higher_level:{l3_raw!r}", out
    out["L3"] = l3_raw

    return True, "", out


def is_sentence_too_ambiguous_to_annotate(sentence, min_length=6, min_meaningful_words=1):
    """Check if sentence is too short/ambiguous to reliably annotate
    
    Args:
        sentence: input sentence text
        min_length: minimum character length (excluding spaces)
        min_meaningful_words: minimum meaningful words (length >= 2)
    
    Returns:
        tuple: (is_ambiguous: bool, reason: str)
    """
    if not sentence or not sentence.strip():
        return True, "empty_sentence"
    
    text = sentence.strip()
    
    # Check character length (excluding spaces) - reduced from 8 to 6
    char_count = len(re.sub(r'\s+', '', text))
    if char_count < min_length:
        return True, f"too_short_chars_{char_count}"
    
    # Extract meaningful words (length >= 2, more lenient)
    words = re.findall(r'\w+', text.lower(), flags=re.UNICODE)
    # Reduced stop words list - keep common meaningful words
    stop_words = {'có', 'thế', 'này', 'kia', 'được', 'như'}
    meaningful_words = [w for w in words if len(w) >= 2 and w not in stop_words]
    
    # Special handling for price/money questions  
    money_words = {'tiền', 'giá', 'bao', 'nhiều', 'cost', 'price'}
    product_words = {'sản', 'phẩm', 'sp', 'hàng', 'tốt', 'xấu', 'quality'}
    
    has_money_context = any(w in ' '.join(meaningful_words) for w in money_words)
    has_product_context = any(w in ' '.join(meaningful_words) for w in product_words)
    
    # Allow price questions even if short
    if has_money_context or has_product_context:
        min_meaningful_words = 1
    
    if len(meaningful_words) < min_meaningful_words:
        return True, f"too_few_meaningful_words_{len(meaningful_words)}"
    
    # Check if sentence is just question words without ANY context
    question_only_patterns = [
        r'^(có\s*)?(không\s*)?(gì\s*\?*\s*)$',  # "Gì?"
        r'^(sao\s*\?*\s*)$',                    # "Sao"
        r'^(nào\s*\?*\s*)$',                    # "Nào?"
        r'^(đâu\s*\?*\s*)$'                     # "Đâu?"
    ]
    
    text_normalized = re.sub(r'\s+', ' ', text.lower().strip())
    for pattern in question_only_patterns:
        if re.match(pattern, text_normalized):
            return True, "question_word_only"
    
    # Check for single word + punctuation patterns (very short)
    if len(words) == 1 and len(text) <= 4:
        return True, "single_word_too_short"
    
    return False, ""


def save_annotation_with_guardrails(
    db,
    sample,
    pred,
    *,
    min_conf_auto_approve=MIN_CONF_APPROVE_EXISTING,
    min_conf_allow_new_label=MIN_CONF_AUTO_ADD_NEW_LABEL,
    allow_auto_add_new_label=ALLOW_AUTO_ADD_NEW_LABEL,
    model_name=OPENAI_MODEL,
    verify_meta=None,
    persist=True,
):
    col_ann = db["intent_annotations"]
    now = datetime.now(timezone.utc)
    
    # STEP 1: Check if sentence is too ambiguous to annotate
    sentence = sample.get("sentence", "").strip()
    is_ambiguous, ambiguous_reason = is_sentence_too_ambiguous_to_annotate(sentence)
    
    if is_ambiguous:
        # Skip annotation for ambiguous sentences
        ann_doc = {
            "sample_id": sample.get("sample_id"),
            "product_id": sample.get("product_id"),
            "sentence": sentence,
            "category": sample.get("category", ""),
            "model": model_name,
            "intent": {
                "level_1": "",
                "level_2": "",
                "level_3": [],
            },
            "confidence": 0.0,
            "reasoning": f"Câu quá ngắn/mơ hồ để annotate: {ambiguous_reason}",
            "qa_status": "skipped_ambiguous",
            "ambiguous_reason": ambiguous_reason,
            "new_label_pending_review": False,
            "graph_upserted": False,
            "semantic_snap_similarity": None,
            "semantic_snap_intent_id": None,
            "taxonomy_semantically_snapped": False,
            "verifier_decision": None,
            "verifier_reason": None,
            "confidence_before_verify": None,
            "source": sample.get("source", "hasaki"),
            "updated_at": now,
        }
        if persist:
            col_ann.update_one({"sample_id": ann_doc["sample_id"]}, {"$set": ann_doc}, upsert=True)
        return ann_doc
    
    # STEP 2: Continue with normal annotation logic
    allowed = _collect_allowed_taxonomy(db)
    # Snap chuỗi L1/L2/L3 → tên trong graph (truoc_mua_hang vs trước ban, underscore vs space, v.v.)
    _snap_pred_to_canonical_taxonomy(pred, allowed)
    confidence = float(pred.get("confidence") or 0)
    valid_existing = _is_label_allowed(pred, allowed)

    semantic_snap_similarity = None
    semantic_snap_intent_id = None
    used_semantic_snap = False
    auto_create_rejected_reason = None

    # Nếu triple không khớp taxonomy: embed đường dẫn/slug prediction, snap vào intent gần nhất trong graph (đối xứng với embeddings đã có)
    if not valid_existing:
        parts = []
        for p in (pred.get("L1"), pred.get("L2"), pred.get("L3")):
            if p is not None and str(p).strip():
                parts.append(str(p).strip())
        slug_blob = ".".join(parts)
        l3_piece = (pred.get("L3") or "").strip()
        reasoning_snip = ((pred.get("reasoning") or "").strip())[:280]
        pred_text_for_snap = slug_blob or l3_piece or reasoning_snip
        sentence_ctx = (sample.get("sentence") or "").strip()
        cat_ctx = (sample.get("category") or "").strip()
        extra_ctx = (" ".join(x for x in (sentence_ctx, cat_ctx) if x)).strip()
        # Triple rỗng hoặc slug không đủ: dùng câu hỏi thật để embed nearest intent (khớp case pending L1/L2/L3 trống)
        if not pred_text_for_snap and extra_ctx:
            pred_text_for_snap = extra_ctx[:512]
        elif extra_ctx and extra_ctx[:80] not in pred_text_for_snap:
            pred_text_for_snap = f"{pred_text_for_snap}\n{extra_ctx}".strip()[:768]

        nearest, nearest_sim = find_nearest_existing_intent(db, pred_text_for_snap)
        semantic_snap_similarity = float(nearest_sim)

        if nearest is not None:
            semantic_snap_intent_id = nearest.get("_id")
            pred["L1"] = nearest.get("l1") or ""
            pred["L2"] = nearest.get("l2") or ""
            pred["L3"] = nearest.get("l3") or ""
            _snap_pred_to_canonical_taxonomy(pred, allowed)
            valid_existing = _is_label_allowed(pred, allowed)
            used_semantic_snap = valid_existing

    qa_status = "rejected"
    new_label_pending_review = False
    graph_upserted = False

    if valid_existing:
        # Snap semantic: không phụ thuộc ngưỡng confidence gpt-4o-mini để thoát pending
        if used_semantic_snap:
            qa_status = "approved_snapped"
            new_label_pending_review = False
        else:
            qa_status = "approved" if confidence >= min_conf_auto_approve else "needs_review"
    else:
        new_label_pending_review = True
        qa_status = "pending_new_label_review"
        auto_create_rejected_reason = None
        if allow_auto_add_new_label and confidence >= min_conf_allow_new_label:
            valid_for_create, reject_reason, norm = _validate_pred_for_auto_create(pred)
            if valid_for_create:
                # Sử dụng L1/L2/L3 đã chuẩn hóa (truoc_mua_hang / sau_mua_hang + slug sạch)
                if persist:
                    upsert_new_label_to_graph(
                        db,
                        domain=sample.get("domain") or infer_domain(sample) or "Unknown",
                        l1=norm["L1"],
                        l2=norm["L2"],
                        l3=norm["L3"],
                        intent_id=sample.get("intent_id") or f"AUTO-{sample.get('sample_id', 'UNKNOWN')}",
                        intent_name=norm["L3"] or "new_intent",
                        description=pred.get("reasoning", ""),
                        confidence=str(confidence),
                        product_category=sample.get("category", ""),
                        detection_signals=sample.get("sentence", ""),
                        example=sample.get("sentence", ""),
                        category_logic="auto_added_from_labeling_notebook",
                    )
                    graph_upserted = True
                # Cập nhật pred để annotation lưu đúng slug đã normalize
                pred["L1"], pred["L2"], pred["L3"] = norm["L1"], norm["L2"], norm["L3"]
                qa_status = "approved_auto_new_label"
                new_label_pending_review = False
            else:
                # Conf cao nhưng L1/L2/L3 sai rule → giữ pending để human review,
                # KHÔNG upsert vào graph (tránh sinh node l1:gia_so_sanh, l1:trước bán, ...)
                auto_create_rejected_reason = reject_reason
                qa_status = "pending_new_label_invalid_slug"

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
        "auto_create_rejected_reason": auto_create_rejected_reason,
        "semantic_snap_similarity": semantic_snap_similarity,
        "semantic_snap_intent_id": semantic_snap_intent_id,
        "taxonomy_semantically_snapped": used_semantic_snap,
        "verifier_decision": (verify_meta or {}).get("decision"),
        "verifier_reason": (verify_meta or {}).get("reason"),
        "confidence_before_verify": (verify_meta or {}).get("confidence_before_verify"),
        "source": sample.get("source", "hasaki"),
        "updated_at": now,
    }
    if persist:
        col_ann.update_one({"sample_id": ann_doc["sample_id"]}, {"$set": ann_doc}, upsert=True)
    return ann_doc


def _intent_slug_blob(d):
    parts = [d.get("l1"), d.get("l2"), d.get("l3"), d.get("name")]
    return " ".join(str(p or "") for p in parts).lower()


_ELECTRONICS_SLUG_MARKERS = (
    "dien_thoai",
    "laptop",
    "may_anh",
    "smartwatch",
    "tai_nghe",
    "iphone",
    "macbook",
    "gpu",
    "chuot",
)


def _candidate_looks_electronics_intent(d):
    s = _intent_slug_blob(d)
    return any(m in s for m in _ELECTRONICS_SLUG_MARKERS)


def _sentence_mentions_electronics(text):
    t = (text or "").lower()
    hints = (
        "dien thoai",
        "điện thoại",
        "laptop",
        "iphone",
        "macbook",
        "may anh",
        "máy ảnh",
        "tai nghe",
        "chuột",
        "ban phim",
        "bàn phím",
        "smartwatch",
        "card man hinh",
    )
    return any(h in t for h in hints)


def _category_match_bonus(sample, d):
    if not sample:
        return 0
    cat = (sample.get("category") or "").lower()
    pc = (d.get("product_category") or "").lower()
    if not cat or not pc:
        return 0
    cat_tokens = [w for w in re.findall(r"\w+", cat, flags=re.UNICODE) if len(w) >= 3]
    hit = sum(1 for w in cat_tokens if w in pc)
    # CHU Y: category chi dung lam tie-breaker, KHONG duoc lan at token match cua cau hoi.
    return min(1, hit)


def get_candidate_l3_from_mongodb(db, text, category=None, top_k=10, sample=None):
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

    beauty_ctx = bool(sample) and infer_domain(sample) == "My pham"

    scored = []
    token_set = set(tokens)
    for d in docs:
        if beauty_ctx and _candidate_looks_electronics_intent(d) and not _sentence_mentions_electronics(text):
            continue

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
        if hit <= 0:
            continue
        cat_bonus = _category_match_bonus(sample, d) if sample else 0
        # Ngu nghia (token hits trong sentence/desc/signals) van la chu yeu;
        # category chi cong them <= 0.2 de tie-break giua 2 candidate ngang diem.
        scored.append((hit + 0.2 * cat_bonus, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    pool = [d for _, d in scored[: max(top_k * 4, top_k + 8)]]

    seen = set()
    uniq = []
    for d in pool:
        key = (d.get("l1"), d.get("l2"), d.get("l3"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
        if len(uniq) >= top_k:
            break
    return uniq


def build_retrieval_prompt(sample, candidates):
    vertical = infer_domain(sample)
    lines = []
    for i, c in enumerate(candidates, 1):
        pc = c.get("product_category") or ""
        lines.append(
            f"{i}. L1={c.get('l1')} | L2={c.get('l2')} | L3={c.get('l3')} | product_category={pc!r} | intent_id={c.get('_id')}"
        )
    taxonomy_text = (
        "\n".join(lines)
        if lines
        else "KHÔNG CÓ ỨNG VIÊN nào từ MongoDB. Bạn vẫn PHẢI đề xuất bộ (L1, L2, L3) hợp lệ theo quy tắc bên dưới và đặt is_new_label=true."
    )

    parts = [
        "Bạn là chuyên gia gán nhãn intent cho QA thương mại điện tử (mỹ phẩm / làm đẹp và điện tử).",
        "Nhiệm vụ: chọn intent dựa trên NGỮ NGHĨA câu hỏi của khách, KHÔNG dựa vào category/tên sản phẩm.",
        f"Ngành ước lượng (chỉ dùng để loại trừ slug ngành sai, KHÔNG dùng để chọn intent): **{vertical}**.",
        "",
        "QUY TẮC TAXONOMY (BẮT BUỘC):",
        "- L1 ∈ {'truoc_mua_hang', 'sau_mua_hang'} — viết đúng underscore + lowercase ASCII, KHÔNG khoảng trắng, KHÔNG dấu tiếng Việt.",
        "  + truoc_mua_hang: khách hỏi TRƯỚC khi mua (tư vấn, thành phần, công dụng, giá, khuyến mãi, tồn kho, bảo quản, an toàn da, phù hợp đối tượng).",
        "  + sau_mua_hang: khách đã/đang sở hữu (giao hàng, đổi trả, hoàn tiền, sản phẩm lỗi/hư khi nhận, khiếu nại dịch vụ, có mã đơn).",
        "- L2, L3 là slug ASCII lowercase + underscore (vd: di_ung_an_toan, chong_anh_sang_xanh). KHÔNG dấu, KHÔNG khoảng trắng.",
        "- L3 phải KHÁC L2 và mô tả ý định cụ thể hơn L2.",
        "",
        "CÁCH SUY LUẬN (BẮT BUỘC THEO THỨ TỰ NÀY):",
        "  1. Đọc kỹ NGỮ NGHĨA câu hỏi của khách (sentence) — xác định ý định chính: khách đang HỎI GÌ, MUỐN BIẾT GÌ.",
        "  2. Tìm L3 nào MÔ TẢ ĐÚNG ý định đó trong danh sách ứng viên (so khớp description / detection_signals / examples — KHÔNG so với product_category).",
        "  3. Suy ra L2 từ L3 đã chọn (sao chép từ ứng viên).",
        "  4. Quyết định L1 dựa vào tone/từ khóa câu (mã đơn / 'đã nhận' → sau_mua_hang; 'có … không' / 'phù hợp' → truoc_mua_hang).",
        "  5. CHỈ SAU CÙNG dùng category/product_name để LOẠI TRỪ slug ngành rõ ràng sai (vd. dien_thoai_* / laptop_* khi sản phẩm là mỹ phẩm).",
        "",
        "QUY TẮC CHỌN NHÃN:",
        "- KHÔNG được chọn intent chỉ vì product_category trùng với category của mẫu — phải có ngữ nghĩa khớp với câu hỏi.",
        "- KHÔNG suy diễn intent từ tên sản phẩm khi câu hỏi không nhắc tới (vd. câu chỉ hỏi 'giá bao nhiêu' thì chọn intent về giá, KHÔNG chọn intent đặc thù của sản phẩm).",
        "- Nếu CÓ dòng ứng viên phù hợp ngữ nghĩa: copy CHÍNH XÁC L1/L2/L3 từ dòng đó, đặt is_new_label=false.",
        "- Nếu KHÔNG có dòng nào khớp ngữ nghĩa câu hỏi: đặt is_new_label=true VÀ tự đề xuất bộ (L1, L2, L3) hợp lệ theo quy tắc taxonomy. KHÔNG ĐƯỢC để L1/L2/L3 rỗng — kể cả khi confidence thấp.",
        "- Tránh slug điện tử (dien_thoai_*, laptop_*, may_anh_*, tai_nghe_*…) khi câu KHÔNG nhắc rõ thiết bị điện tử.",
        "- Confidence ∈ [0,1]: ≥0.90 rất chắc (ngữ nghĩa khớp 1-1), 0.70–0.89 hợp lý, <0.70 không chắc. KHÔNG trả 0.0 nếu đã đề xuất được L1/L2/L3.",
        "",
        "ỨNG VIÊN (MongoDB):",
        taxonomy_text,
        "",
        "NGỮ CẢNH ĐẦU VÀO (sentence là CHÍNH; category/product_name/brand chỉ là context phụ — chỉ dùng để loại trừ ngành sai):",
        f"- Câu (sentence) [CHÍNH]: {sample.get('sentence', '')}",
        f"- Danh mục (category) [phụ]: {sample.get('category', '')}",
        f"- Tên sản phẩm (product_name) [phụ]: {sample.get('product_name', '')}",
        f"- Thương hiệu (brand) [phụ]: {sample.get('brand', '')}",
        "",
        "Trả về DUY NHẤT một đối tượng JSON hợp lệ (KHÔNG markdown, KHÔNG text ngoài JSON).",
        "Thứ tự khóa — bắt buộc: viết reasoning trước (1–3 câu tiếng Việt), sau đó confidence và nhãn:",
        "{",
        '  "reasoning": "...",',
        '  "confidence": 0.0,',
        '  "L1": "...",',
        '  "L2": "...",',
        '  "L3": "...",',
        '  "is_new_label": false',
        "}",
    ]
    return "\n".join(parts)

_HASAKI_BEAUTY_CATEGORIES_CACHE = None  # (frozenset, mtime_ns) sau khi load thanh cong


def get_hasaki_beauty_categories():
    """Doc unique `category` tu HASAKI_JSON; cache theo mtime file (doi file -> doc lai)."""
    global _HASAKI_BEAUTY_CATEGORIES_CACHE
    path = HASAKI_JSON
    mtime = path.stat().st_mtime_ns if path.exists() else -1
    if isinstance(_HASAKI_BEAUTY_CATEGORIES_CACHE, tuple) and len(_HASAKI_BEAUTY_CATEGORIES_CACHE) == 2:
        cached, mt = _HASAKI_BEAUTY_CATEGORIES_CACHE
        if mt == mtime:
            return cached
    cats = set()
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        c = (row.get("category") or "").strip()
                        if c:
                            cats.add(c)
    except Exception:
        pass
    if not cats:
        cats = {
            "Sức khỏe làm đẹp",
            "Trang điểm",
            "Mỹ phẩm high end",
        }
    fs = frozenset(cats)
    _HASAKI_BEAUTY_CATEGORIES_CACHE = (fs, mtime)
    return fs


def clear_hasaki_beauty_category_cache():
    """Xoa cache catalog category (sau khi doi HASAKI_JSON hoac can ep doc lai)."""
    global _HASAKI_BEAUTY_CATEGORIES_CACHE
    _HASAKI_BEAUTY_CATEGORIES_CACHE = None


def infer_domain(sample):
    cat_raw = (sample.get("category") or "").strip()
    cat = cat_raw.lower()
    name = (sample.get("product_name") or "").lower()
    blob = f"{cat} {name}"

    if cat_raw and cat_raw in get_hasaki_beauty_categories():
        return "My pham"

    electronics_kw = (
        "laptop phone dien thoai tai nghe chuot ban phim camera smartwatch "
        "gaming console ram gpu cpu may anh dong ho thong minh "
        "tablet iphone tivi tv"
    )
    if any(k in blob for k in electronics_kw.split()):
        return "Dien tu"

    cat_hints = (
        "may tinh",
        "dien tu",
        "phu kien dien",
        "dien thoai",
        "dong ho thong minh",
    )
    if any(h in cat for h in cat_hints):
        return "Dien tu"

    return "My pham"


print("Helpers OK")


import json as json_lib
import re
import time


SYSTEM_PROMPT = (
    "Bạn chỉ trả về một đoạn JSON hợp lệ, không markdown, không text ngoài JSON. "
    "Tuân thủ thứ tự khóa trong yêu cầu: reasoning trước, sau đó confidence, L1, L2, L3, is_new_label."
)


def _call_llm(messages, max_new_tokens=None, model=None):
    """Gọi OpenAI Chat Completions, trả raw text (JSON string).

    model=None -> dùng OPENAI_MODEL (annotator). Truyền model khác (vd VERIFIER_MODEL)
    để gọi verifier mà không phá Phase 1.
    """
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    if model is None:
        model = OPENAI_MODEL
    if openai_client is None:
        raise RuntimeError(
            "openai_client chưa khởi tạo — chạy lại cell §5 (LLM client setup)."
        )
    last_err = None
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=OPENAI_TEMPERATURE,
                max_tokens=max_new_tokens,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"OpenAI call thất bại sau {OPENAI_MAX_RETRIES} lần: {last_err}")



def extract_json_object(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    if "```" in text:
        for chunk in text.split("```"):
            c = chunk.strip()
            if c.lower().startswith("json"):
                c = c[4:].strip()
            if c.startswith("{"):
                text = c
                break
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Khong tim thay JSON")
    return json_lib.loads(text[start : end + 1])


def build_llm_only_prompt(sample):
    """Direct LLM annotation without candidate retrieval (CV-HIL ablation variant A)."""
    vertical = infer_domain(sample)
    parts = [
        "Bạn là chuyên gia gán nhãn intent cho QA thương mại điện tử (mỹ phẩm / làm đẹp và điện tử).",
        "Nhiệm vụ: chọn intent dựa trên NGỮ NGHĨA câu hỏi của khách, KHÔNG dựa vào category/tên sản phẩm.",
        f"Ngành ước lượng (chỉ dùng để loại trừ slug ngành sai, KHÔNG dùng để chọn intent): **{vertical}**.",
        "",
        "QUY TẮC TAXONOMY (BẮT BUỘC):",
        "- L1 ∈ {'truoc_mua_hang', 'sau_mua_hang'} — viết đúng underscore + lowercase ASCII, KHÔNG khoảng trắng, KHÔNG dấu tiếng Việt.",
        "  + truoc_mua_hang: khách hỏi TRƯỚC khi mua (tư vấn, thành phần, công dụng, giá, khuyến mãi, tồn kho, bảo quản, an toàn da, phù hợp đối tượng).",
        "  + sau_mua_hang: khách đã/đang sở hữu (giao hàng, đổi trả, hoàn tiền, sản phẩm lỗi/hư khi nhận, khiếu nại dịch vụ, có mã đơn).",
        "- L2, L3 là slug ASCII lowercase + underscore (vd: di_ung_an_toan, chong_anh_sang_xanh). KHÔNG dấu, KHÔNG khoảng trắng.",
        "- L3 phải KHÁC L2 và mô tả ý định cụ thể hơn L2.",
        "",
        "CÁCH SUY LUẬN:",
        "  1. Đọc kỹ NGỮ NGHĨA câu hỏi của khách (sentence).",
        "  2. Quyết định L1 dựa vào tone/từ khóa câu.",
        "  3. Chọn L2/L3 slug hợp lệ mô tả đúng ý định.",
        "  4. CHỈ SAU CÙNG dùng category/product_name để LOẠI TRỪ slug ngành rõ ràng sai.",
        "",
        "KHÔNG có danh sách ứng viên từ retrieval — bạn tự đề xuất bộ (L1, L2, L3) hợp lệ.",
        "Confidence ∈ [0,1]. KHÔNG trả 0.0 nếu đã đề xuất được L1/L2/L3.",
        "",
        "NGỮ CẢNH ĐẦU VÀO (sentence là CHÍNH):",
        f"- Câu (sentence) [CHÍNH]: {sample.get('sentence', '')}",
        f"- Danh mục (category) [phụ]: {sample.get('category', '')}",
        f"- Tên sản phẩm (product_name) [phụ]: {sample.get('product_name', '')}",
        f"- Thương hiệu (brand) [phụ]: {sample.get('brand', '')}",
        "",
        "Trả về DUY NHẤT một đối tượng JSON hợp lệ (KHÔNG markdown, KHÔNG text ngoài JSON):",
        "{",
        '  "reasoning": "...",',
        '  "confidence": 0.0,',
        '  "L1": "...",',
        '  "L2": "...",',
        '  "L3": "...",',
        '  "is_new_label": false',
        "}",
    ]
    return "\n".join(parts)


def predict_intent(sample, top_k=None, use_semantic=True, use_retrieval=True):
    """Enhanced predict_intent with graph-aware semantic retrieval and ambiguity filtering
    
    Args:
        sample: sample data dict
        top_k: max candidates (fallback to TOP_K_CANDIDATES)
        use_semantic: when use_retrieval=True, union retrieval (True) or regex only (False)
        use_retrieval: False = LLM-only prompt without candidates (CV-HIL ablation A)
    
    Returns:
        tuple: (prediction, raw_output, candidates)
    """
    if top_k is None:
        top_k = TOP_K_CANDIDATES
    
    # PRE-CHECK: Skip prediction if sentence is too ambiguous
    sentence = sample.get("sentence", "").strip()
    is_ambiguous, ambiguous_reason = is_sentence_too_ambiguous_to_annotate(sentence)
    
    if is_ambiguous:
        # Return dummy prediction for ambiguous sentences
        pred = {
            "reasoning": f"Câu quá ngắn/mơ hồ để dự đoán: {ambiguous_reason}",
            "confidence": 0.0,
            "L1": "",
            "L2": "",
            "L3": "",
            "is_new_label": False,
            "_ambiguous": True,
            "_ambiguous_reason": ambiguous_reason
        }
        return pred, f"SKIPPED_AMBIGUOUS: {ambiguous_reason}", []
    
    if not use_retrieval:
        cands = []
        user_prompt = build_llm_only_prompt(sample)
    elif use_semantic:
        cands = union_retrieval(
            db, 
            sample.get("sentence", ""), 
            category=sample.get("category"),
            top_k_regex=max(6, top_k//2),
            top_k_semantic=max(4, top_k//3),
            sample=sample
        )
        user_prompt = build_retrieval_prompt(sample, cands)
    else:
        cands = get_candidate_l3_from_mongodb(
            db, sample.get("sentence", ""), category=sample.get("category"), top_k=top_k, sample=sample
        )
        user_prompt = build_retrieval_prompt(sample, cands)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = _call_llm(messages, max_new_tokens=MAX_NEW_TOKENS)
    pred = extract_json_object(raw)
    return pred, raw, cands


def enrich_sample(s):
    out = dict(s)
    out["domain"] = infer_domain(s)
    out["intent_id"] = f"AUTO-{s.get('sample_id', 'x')}"
    return out


# === §6.5. Verifier (cross-verification, GPT-4o) ===
# Doc lai nhan annotator (gpt-4o-mini) bang VERIFIER_MODEL (mac dinh gpt-4o) -> RETAIN | REVISE.
# Verifier KHONG enforce taxonomy; de guardrail (§4) xu ly.

VERIFIER_SYSTEM_PROMPT = (
    "Bạn là model kiểm định nhãn intent cho QA thương mại điện tử (mỹ phẩm / điện tử). "
    "Bạn chỉ trả về một đoạn JSON hợp lệ, không markdown, không text ngoài JSON."
)


def build_verification_prompt(sample, pred, candidates):
    lines = []
    for i, c in enumerate(candidates or [], 1):
        pc = c.get("product_category") or ""
        lines.append(
            f"{i}. L1={c.get('l1')} | L2={c.get('l2')} | L3={c.get('l3')} | product_category={pc!r}"
        )
    taxonomy_text = "\n".join(lines) if lines else "KHÔNG CÓ ứng viên từ MongoDB."

    parts = [
        "Nhiệm vụ: KIỂM ĐỊNH bộ nhãn (L1, L2, L3) mà annotator đã gán cho câu hỏi của khách.",
        "Đánh giá dựa trên NGỮ NGHĨA câu hỏi, KHÔNG dựa vào category/tên sản phẩm.",
        "",
        "QUY TẮC TAXONOMY (BẮT BUỘC):",
        "- L1 ∈ {'truoc_mua_hang', 'sau_mua_hang'} — lowercase ASCII + underscore, không dấu, không khoảng trắng.",
        "  + truoc_mua_hang: hỏi TRƯỚC khi mua (tư vấn, thành phần, công dụng, giá, khuyến mãi, tồn kho, an toàn da, phù hợp đối tượng).",
        "  + sau_mua_hang: đã/đang sở hữu (giao hàng, đổi trả, hoàn tiền, sản phẩm lỗi khi nhận, khiếu nại, có mã đơn).",
        "- L2, L3 là slug ASCII lowercase + underscore. L3 phải KHÁC L2 và cụ thể hơn.",
        "",
        "CÁCH QUYẾT ĐỊNH:",
        "- Nếu nhãn annotator khớp đúng ngữ nghĩa câu hỏi → decision='RETAIN', giữ nguyên final_L1/L2/L3.",
        "- Nếu sai (sai L1 truoc/sau, sai slug, hiểu nhầm category) → decision='REVISE' và đề xuất bộ (L1, L2, L3) đúng hơn.",
        "- KHÔNG để final_L1/L2/L3 rỗng. confidence_adjusted ∈ [0,1] phản ánh độ tin sau khi kiểm định.",
        "",
        "ỨNG VIÊN (MongoDB):",
        taxonomy_text,
        "",
        "CÂU HỎI KHÁCH (sentence là CHÍNH; context phụ chỉ để loại trừ ngành sai):",
        f"- Câu (sentence): {sample.get('sentence', '')}",
        f"- Danh mục (category) [phụ]: {sample.get('category', '')}",
        f"- Tên sản phẩm (product_name) [phụ]: {sample.get('product_name', '')}",
        "",
        "NHÃN ANNOTATOR ĐỀ XUẤT (cần kiểm định):",
        f"- L1: {pred.get('L1', '')}",
        f"- L2: {pred.get('L2', '')}",
        f"- L3: {pred.get('L3', '')}",
        f"- confidence: {pred.get('confidence', '')}",
        f"- reasoning: {pred.get('reasoning', '')}",
        "",
        "Trả về DUY NHẤT một JSON hợp lệ theo đúng thứ tự khóa:",
        "{",
        '  "decision": "RETAIN",',
        '  "final_L1": "...",',
        '  "final_L2": "...",',
        '  "final_L3": "...",',
        '  "confidence_adjusted": 0.0,',
        '  "reason": "giải thích ngắn gọn tiếng Việt"',
        "}",
    ]
    return "\n".join(parts)


def verify_intent(sample, pred, candidates):
    """Goi verifier (VERIFIER_MODEL) doc lai pred. Tra dict chuan hoa.

    Output keys: decision (RETAIN|REVISE), final_L1/L2/L3, confidence_adjusted, reason.
    Neu verifier loi/parse fail -> fallback RETAIN giu nguyen pred (an toan).
    """
    user_prompt = build_verification_prompt(sample, pred, candidates)
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = _call_llm(messages, max_new_tokens=VERIFY_MAX_TOKENS, model=VERIFIER_MODEL)
        v = extract_json_object(raw)
    except Exception as e:
        return {
            "decision": "RETAIN",
            "final_L1": pred.get("L1", ""),
            "final_L2": pred.get("L2", ""),
            "final_L3": pred.get("L3", ""),
            "confidence_adjusted": float(pred.get("confidence") or 0),
            "reason": f"verifier_error: {e}",
        }

    decision = str(v.get("decision") or "RETAIN").strip().upper()
    if decision not in ("RETAIN", "REVISE"):
        decision = "RETAIN"
    try:
        conf_adj = float(v.get("confidence_adjusted"))
    except (TypeError, ValueError):
        conf_adj = float(pred.get("confidence") or 0)
    conf_adj = max(0.0, min(1.0, conf_adj))

    if decision == "REVISE":
        f1 = (v.get("final_L1") or "").strip()
        f2 = (v.get("final_L2") or "").strip()
        f3 = (v.get("final_L3") or "").strip()
        # Verifier thieu nhan -> coi nhu RETAIN de tranh lam rong pred.
        if not (f1 and f2 and f3):
            decision = "RETAIN"
            f1, f2, f3 = pred.get("L1", ""), pred.get("L2", ""), pred.get("L3", "")
    else:
        f1, f2, f3 = pred.get("L1", ""), pred.get("L2", ""), pred.get("L3", "")

    return {
        "decision": decision,
        "final_L1": f1,
        "final_L2": f2,
        "final_L3": f3,
        "confidence_adjusted": conf_adj,
        "reason": (v.get("reason") or "").strip(),
    }


def apply_verifier(sample, pred, candidates, *, force=False, enable_verifier=None):
    """Goi verifier neu can va merge ket qua vao pred (in-place).

    Tra ve verify_meta (dict) hoac None neu khong verify.
    Chi verify khi: ENABLE_VERIFIER, khong _ambiguous, va (force hoac VERIFY_ALL_SAMPLES hoac confidence < VERIFY_CONF_THRESHOLD).

    enable_verifier: None = dung bien moi truong ENABLE_VERIFIER; True/False override cho ablation.
    """
    verifier_on = ENABLE_VERIFIER if enable_verifier is None else bool(enable_verifier)
    if not verifier_on:
        return None
    if pred.get("_ambiguous"):
        return None
    conf0 = float(pred.get("confidence") or 0)
    if not force and not VERIFY_ALL_SAMPLES and conf0 >= VERIFY_CONF_THRESHOLD:
        return None

    v = verify_intent(sample, pred, candidates)
    v["confidence_before_verify"] = conf0
    if v["decision"] == "REVISE":
        pred["L1"], pred["L2"], pred["L3"] = v["final_L1"], v["final_L2"], v["final_L3"]
        pred["confidence"] = v["confidence_adjusted"]
        if v.get("reason"):
            pred["reasoning"] = (pred.get("reasoning", "") + " | VERIFIER: " + v["reason"]).strip()
    else:  # RETAIN
        pred["confidence"] = max(conf0, v["confidence_adjusted"])
    return v


# Pipeline dừng ở nhãn + qa_status. Các mẫu `needs_review` và
# `pending_new_label_review` được human review ngoài notebook.


def _pred_levels(pred: dict) -> tuple[str, str, str]:
    return (
        str(pred.get("L1") or "").strip(),
        str(pred.get("L2") or "").strip(),
        str(pred.get("L3") or "").strip(),
    )


def _ann_doc_to_predicted_label(ann_doc: dict) -> dict[str, str]:
    intent = ann_doc.get("intent") or {}
    l3_raw = intent.get("level_3") or []
    if isinstance(l3_raw, list):
        l3 = str(l3_raw[0] if l3_raw else "").strip()
    else:
        l3 = str(l3_raw or "").strip()
    return {
        "l1": str(intent.get("level_1") or "").strip(),
        "l2": str(intent.get("level_2") or "").strip(),
        "l3": l3,
    }


def finalize_prediction(
    sample,
    pred,
    *,
    enable_validation: bool = True,
    verify_meta=None,
    persist: bool = False,
):
    """Apply taxonomy guardrails or return raw pred (CV-HIL ablation).

    enable_validation=True (default production): save_annotation_with_guardrails.
    enable_validation=False: skip snap/qa_status; return benchmark-compatible dict.
    """
    if enable_validation:
        ann = save_annotation_with_guardrails(
            db,
            sample,
            pred,
            verify_meta=verify_meta,
            persist=persist,
        )
        label = _ann_doc_to_predicted_label(ann)
        return {
            "predicted_label": label,
            "confidence": float(ann.get("confidence") or 0),
            "qa_status": ann.get("qa_status"),
            "ann_doc": ann,
        }

    l1, l2, l3 = _pred_levels(pred)
    return {
        "predicted_label": {"l1": l1, "l2": l2, "l3": l3},
        "confidence": float(pred.get("confidence") or 0),
        "qa_status": None,
        "ann_doc": None,
    }


def annotate_cvhil_ablation(
    sample,
    *,
    variant: str = "cvhil_a",
    use_retrieval: bool = True,
    enable_verifier: bool = False,
    enable_validation: bool = False,
    force_verify: bool = True,
):
    """Run one CV-HIL ablation variant on a single sample.

    Returns a benchmark-compatible row for eval_predictions_vs_d3.
    """
    pred, raw, cands = predict_intent(
        sample,
        use_retrieval=use_retrieval,
        use_semantic=True,
    )
    label_after_annotate = _pred_levels(pred)

    verify_meta = None
    if enable_verifier and not pred.get("_ambiguous"):
        verify_meta = apply_verifier(
            sample,
            pred,
            cands,
            force=force_verify,
            enable_verifier=True,
        )
    label_after_verify = _pred_levels(pred)

    finalized = finalize_prediction(
        sample,
        pred,
        enable_validation=enable_validation,
        verify_meta=verify_meta,
        persist=False,
    )

    candidate_paths = [
        {
            "l1": c.get("l1", ""),
            "l2": c.get("l2", ""),
            "l3": c.get("l3", ""),
        }
        for c in (cands or [])
    ]

    return {
        "sample_id": sample.get("sample_id"),
        "sentence": sample.get("sentence", ""),
        "predicted_label": finalized["predicted_label"],
        "confidence": finalized["confidence"],
        "api_calls": 1 + (1 if verify_meta is not None else 0),
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "metadata": {
            "variant": variant,
            "use_retrieval": use_retrieval,
            "enable_verifier": enable_verifier,
            "enable_validation": enable_validation,
            "label_after_annotate": {
                "l1": label_after_annotate[0],
                "l2": label_after_annotate[1],
                "l3": label_after_annotate[2],
            },
            "label_after_verify": {
                "l1": label_after_verify[0],
                "l2": label_after_verify[1],
                "l3": label_after_verify[2],
            },
            "verifier_decision": (verify_meta or {}).get("decision"),
            "verifier_reason": (verify_meta or {}).get("reason"),
            "qa_status": finalized.get("qa_status"),
            "num_candidates": len(cands or []),
            "candidate_paths": candidate_paths[:10],
            "raw_annotator_snippet": (raw or "")[:500],
        },
    }


CVHIL_ABLATION_VARIANTS = {
    "cvhil_a": {
        "display_name": "A. LLM-only",
        "use_retrieval": False,
        "enable_verifier": False,
        "enable_validation": False,
    },
    "cvhil_b": {
        "display_name": "B. Retrieval-Augmented",
        "use_retrieval": True,
        "enable_verifier": False,
        "enable_validation": False,
    },
    "cvhil_c": {
        "display_name": "C. Retrieval + Verification",
        "use_retrieval": True,
        "enable_verifier": True,
        "enable_validation": False,
    },
    "cvhil_d": {
        "display_name": "D. Full CV-HIL",
        "use_retrieval": True,
        "enable_verifier": True,
        "enable_validation": True,
        "reuse_from": "verified_d2",
    },
}
