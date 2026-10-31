"""Zero-shot prompting baseline."""
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


def build_zero_shot_prompt(sentence: str, taxonomy: TaxonomyIndex) -> str:
    return f"""Bạn là chuyên gia phân loại intent cho e-commerce tiếng Việt.

{taxonomy.compact_taxonomy_text()}

Gán nhãn phân cấp L1/L2/L3 cho câu sau (zero-shot, không ví dụ):

Câu: "{sentence}"

Trả về JSON:
{{
  "level_1": "...",
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
    prompt = build_zero_shot_prompt(sample.get("sentence", ""), taxonomy)
    data, usage = client.chat_json([{"role": "user", "content": prompt}])
    l1, l2, l3, conf = parse_label_response(data)
    return make_annotation_result(
        sample, l1, l2, l3, conf, usage,
        metadata={"strategy": "zero_shot"},
    )
