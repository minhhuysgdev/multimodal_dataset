"""CanDist LLM-only: candidate generation + selection."""
from __future__ import annotations

from typing import Any

from common import (
    AnnotationResult,
    DemoPool,
    LLMClient,
    LLMUsage,
    TaxonomyIndex,
    make_annotation_result,
    merge_usage,
    parse_label_response,
)


def build_candist_prompt(
    sentence: str,
    taxonomy: TaxonomyIndex,
    demos: list[dict[str, Any]],
) -> str:
    demo_block = DemoPool.format_demo_block(demos)
    return f"""Bạn là annotator theo phương pháp CanDist (candidate prompting).

{taxonomy.compact_taxonomy_text()}

Few-shot demos:
{demo_block}

Câu: "{sentence}"

Bước 1: Liệt kê 3-5 bộ nhãn ứng viên (candidate intents) phù hợp nhất.
Bước 2: Chọn 1 bộ nhãn cuối cùng từ các ứng viên.

Trả về JSON:
{{
  "candidates": [
    {{"level_1": "...", "level_2": "...", "level_3": "...", "score": 0.0}}
  ],
  "level_1": "...",
  "level_2": "...",
  "level_3": "...",
  "confidence": 0.0
}}"""


def _normalize_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract clean candidate label sets (dedup, keep only full L1/L2/L3)."""
    out = []
    seen = set()
    for c in data.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        l1, l2, l3, score = parse_label_response(c)
        if not all([l1, l2, l3]):
            continue
        key = (l1, l2, l3)
        if key in seen:
            continue
        seen.add(key)
        out.append({"l1": l1, "l2": l2, "l3": l3, "score": score})
    return out


def _pick_from_candidates(data: dict[str, Any]) -> tuple[str, str, str, float]:
    l1, l2, l3, conf = parse_label_response(data)
    if all([l1, l2, l3]):
        return l1, l2, l3, conf

    candidates = data.get("candidates") or []
    if not candidates:
        return l1, l2, l3, conf

    best = max(
        candidates,
        key=lambda c: float(c.get("score", 0) if isinstance(c, dict) else 0),
    )
    if isinstance(best, dict):
        return parse_label_response(best)
    return l1, l2, l3, conf


def annotate_sample(
    sample: dict[str, Any],
    client: LLMClient,
    taxonomy: TaxonomyIndex,
    demo_pool: DemoPool,
) -> AnnotationResult:
    sentence = sample.get("sentence", "")
    demos = demo_pool.retrieve_demos(sentence, k=5)
    prompt = build_candist_prompt(sentence, taxonomy, demos)
    data, usage = client.chat_json([{"role": "user", "content": prompt}], max_tokens=768)
    l1, l2, l3, conf = _pick_from_candidates(data)
    candidates = _normalize_candidates(data)
    # Guarantee the teacher pick is part of the candidate set for distillation.
    if all([l1, l2, l3]) and not any(
        c["l1"] == l1 and c["l2"] == l2 and c["l3"] == l3 for c in candidates
    ):
        candidates.append({"l1": l1, "l2": l2, "l3": l3, "score": conf})
    return make_annotation_result(
        sample, l1, l2, l3, conf, usage,
        metadata={
            "strategy": "candist_teacher",
            "num_candidates": len(candidates),
            "candidates": candidates,
            "teacher_pick": {"l1": l1, "l2": l2, "l3": l3},
            "num_demos": len(demos),
        },
    )
