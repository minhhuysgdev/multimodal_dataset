"""RVI-HUL: Cross-Verified Hierarchical Intent Labeling (annotator + verifier)."""
from __future__ import annotations

import os
from typing import Any

from common import (
    D2_DATASET_FILE,
    AnnotationResult,
    DemoPool,
    LLMClient,
    LLMUsage,
    TaxonomyIndex,
    load_results,
    make_annotation_result,
    merge_usage,
    parse_label_response,
)
from retrieval_utils import intent_path


VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "gpt-4o")
VERIFY_CONF_THRESHOLD = float(os.getenv("VERIFY_CONF_THRESHOLD", "0.85"))


def _load_d2_cache() -> dict[str, dict[str, Any]]:
    """Load labels from production D2 pipeline output (d2_dataset.json)."""
    rows = load_results(D2_DATASET_FILE)
    cache = {}
    for r in rows:
        sid = r.get("sample_id")
        if not sid:
            continue
        l1, l2, l3 = intent_path(r)
        if all([l1, l2, l3]):
            cache[sid] = r
    return cache


def build_annotator_prompt(
    sentence: str,
    taxonomy: TaxonomyIndex,
    demos: list[dict[str, Any]],
) -> str:
    demo_block = DemoPool.format_demo_block(demos)
    return f"""Bạn là annotator chính (gpt-4o-mini) trong pipeline Cross-Verified Hierarchical Intent Labeling (RVI-HUL).

{taxonomy.compact_taxonomy_text()}

Ví dụ tham khảo (RAG):
{demo_block}

Câu: "{sentence}"

Trả về JSON:
{{
  "reasoning": "...",
  "level_1": "...",
  "level_2": "...",
  "level_3": "...",
  "confidence": 0.0
}}"""


def build_verifier_prompt(
    sentence: str,
    pred: dict[str, str],
    confidence: float,
    reasoning: str,
) -> str:
    return f"""Bạn là verifier kiểm định nhãn intent e-commerce tiếng Việt.

Câu khách: "{sentence}"

Nhãn annotator đề xuất:
- L1: {pred.get('l1', '')}
- L2: {pred.get('l2', '')}
- L3: {pred.get('l3', '')}
- confidence: {confidence}
- reasoning: {reasoning}

Quyết định RETAIN (giữ) hoặc REVISE (sửa).
Trả về JSON:
{{
  "decision": "RETAIN",
  "level_1": "...",
  "level_2": "...",
  "level_3": "...",
  "confidence": 0.0,
  "reason": "..."
}}"""


def annotate_sample(
    sample: dict[str, Any],
    client: LLMClient,
    taxonomy: TaxonomyIndex,
    demo_pool: DemoPool,
    approved_cache: dict[str, dict[str, Any]] | None = None,
) -> AnnotationResult:
    sid = sample["sample_id"]
    if approved_cache and sid in approved_cache:
        cached = approved_cache[sid]
        l1, l2, l3 = intent_path(cached)
        return make_annotation_result(
            sample, l1, l2, l3,
            float(cached.get("confidence", 0.9)),
            LLMUsage(),
            metadata={
                "strategy": "verified_d2",
                "source": "d2_dataset_cache",
                "verifier_called": False,
            },
        )

    sentence = sample.get("sentence", "")
    demos = demo_pool.retrieve_demos(sentence, k=5)
    prompt = build_annotator_prompt(sentence, taxonomy, demos)
    data, usage = client.chat_json([{"role": "user", "content": prompt}], max_tokens=512)
    l1, l2, l3, conf = parse_label_response(data)
    reasoning = str(data.get("reasoning", ""))
    verifier_called = False
    decision = "SKIPPED"

    if conf < VERIFY_CONF_THRESHOLD and all([l1, l2, l3]):
        verifier_called = True
        vprompt = build_verifier_prompt(
            sentence,
            {"l1": l1, "l2": l2, "l3": l3},
            conf,
            reasoning,
        )
        vdata, vusage = client.chat_json(
            [{"role": "user", "content": vprompt}],
            model=VERIFIER_MODEL,
            max_tokens=512,
        )
        usage = merge_usage(usage, vusage)
        decision = str(vdata.get("decision", "RETAIN")).upper()
        if decision == "REVISE":
            l1, l2, l3, conf = parse_label_response(vdata)

    return make_annotation_result(
        sample, l1, l2, l3, conf, usage,
        metadata={
            "strategy": "verified_d2",
            "source": "live_pipeline",
            "verifier_called": verifier_called,
            "verifier_decision": decision,
            "annotator_model": client.model,
            "verifier_model": VERIFIER_MODEL if verifier_called else None,
        },
    )


def get_approved_cache() -> dict[str, dict[str, Any]]:
    """Backward-compatible alias: D2 cache from d2_dataset.json."""
    return _load_d2_cache()
