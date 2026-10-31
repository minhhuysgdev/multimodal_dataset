"""FreeAL LLM-only: initial labeling + refinery pass."""
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
from retrieval_utils import intent_path


def build_freeal_prompt(
    sentence: str,
    taxonomy: TaxonomyIndex,
    demos: list[dict[str, Any]],
    *,
    refinery: bool = False,
) -> str:
    demo_block = DemoPool.format_demo_block(demos)
    mode = "REFINERY (kiểm tra và sửa nhãn nếu cần)" if refinery else "INITIAL"
    return f"""Bạn là annotator trong vòng active learning (FreeAL, {mode}).

{taxonomy.compact_taxonomy_text()}

Các ví dụ được chọn theo embedding similarity:
{demo_block}

Câu cần gán nhãn: "{sentence}"

Nếu câu quá mơ hồ, trả về confidence thấp (<0.5).
Trả về JSON:
{{
  "level_1": "...",
  "level_2": "...",
  "level_3": "...",
  "confidence": 0.0,
  "reasoning": "..."
}}"""


def annotate_sample(
    sample: dict[str, Any],
    client: LLMClient,
    taxonomy: TaxonomyIndex,
    demo_pool: DemoPool,
) -> AnnotationResult:
    sentence = sample.get("sentence", "")
    demos = demo_pool.retrieve_demos(sentence, k=5)

    # Initial pass
    prompt1 = build_freeal_prompt(sentence, taxonomy, demos, refinery=False)
    data1, usage1 = client.chat_json([{"role": "user", "content": prompt1}], max_tokens=512)
    l1_1, l2_1, l3_1, conf1 = parse_label_response(data1)

    total_usage = usage1
    final_l1, final_l2, final_l3, final_conf = l1_1, l2_1, l3_1, conf1
    refinery_applied = False

    # Refinery pass for low-confidence or empty labels
    if conf1 < 0.85 or not all([l1_1, l2_1, l3_1]):
        prompt2 = build_freeal_prompt(sentence, taxonomy, demos, refinery=True)
        hint = f"\nNhãn initial: L1={l1_1}, L2={l2_1}, L3={l3_1}, confidence={conf1}"
        data2, usage2 = client.chat_json(
            [{"role": "user", "content": prompt2 + hint}],
            max_tokens=512,
        )
        total_usage = merge_usage(total_usage, usage2)
        l1_2, l2_2, l3_2, conf2 = parse_label_response(data2)
        if all([l1_2, l2_2, l3_2]):
            final_l1, final_l2, final_l3, final_conf = l1_2, l2_2, l3_2, conf2
            refinery_applied = True

    return make_annotation_result(
        sample, final_l1, final_l2, final_l3, final_conf, total_usage,
        metadata={
            "strategy": "freeal_llm_only",
            "refinery_applied": refinery_applied,
            "num_demos": len(demos),
        },
    )
