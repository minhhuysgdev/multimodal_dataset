"""Chain-of-Thought prompting baseline."""
from __future__ import annotations

from typing import Any

from common import (
    AnnotationResult,
    DemoPool,
    LLMClient,
    TaxonomyIndex,
    make_annotation_result,
    parse_label_response,
)


def build_cot_prompt(sentence: str, taxonomy: TaxonomyIndex) -> str:
    return f"""Bạn là chuyên gia phân loại intent cho e-commerce tiếng Việt.

{taxonomy.compact_taxonomy_text()}

Phân tích câu sau theo Chain-of-Thought:
1) Xác định khách đang ở giai đoạn trước hay sau mua hàng (L1)
2) Xác định chủ đề chính (L2)
3) Chọn intent cụ thể nhất (L3)

Câu: "{sentence}"

Trả về JSON (có reasoning nhưng chỉ dùng để suy luận):
{{
  "reasoning": "...",
  "level_1": "...",`
  "level_2": "...",
  "level_3": "...",
  "confidence": 0.0
}}"""


def annotate_sample(
    sample: dict[str, Any],
    client: LLMClient,
    taxonomy: TaxonomyIndex,
    demo_pool: DemoPool | None = None,
) -> AnnotationResult:
    prompt = build_cot_prompt(sample.get("sentence", ""), taxonomy)
    data, usage = client.chat_json([{"role": "user", "content": prompt}], max_tokens=768)
    l1, l2, l3, conf = parse_label_response(data)
    return make_annotation_result(
        sample, l1, l2, l3, conf, usage,
        metadata={
            "strategy": "cot",
            "reasoning": data.get("reasoning", ""),
        },
    )
