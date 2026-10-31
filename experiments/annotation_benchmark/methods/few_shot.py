"""Few-shot prompting baseline with fixed demonstrations."""
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


def build_few_shot_prompt(
    sentence: str,
    taxonomy: TaxonomyIndex,
    demos: list[dict[str, Any]],
) -> str:
    demo_block = DemoPool.format_demo_block(demos)
    return f"""Bạn là chuyên gia phân loại intent cho e-commerce tiếng Việt.

{taxonomy.compact_taxonomy_text()}

Dưới đây là các ví dụ mẫu cố định (few-shot):

{demo_block}

Bây giờ gán nhãn cho câu mới:

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
    demo_pool: DemoPool,
) -> AnnotationResult:
    demos = demo_pool.get_fixed_demos()
    prompt = build_few_shot_prompt(sample.get("sentence", ""), taxonomy, demos)
    data, usage = client.chat_json([{"role": "user", "content": prompt}])
    l1, l2, l3, conf = parse_label_response(data)
    return make_annotation_result(
        sample, l1, l2, l3, conf, usage,
        metadata={"strategy": "few_shot", "num_demos": len(demos)},
    )
