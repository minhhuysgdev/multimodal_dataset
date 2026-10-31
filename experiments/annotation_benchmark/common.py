"""Shared utilities for annotation benchmark."""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "phase5_pipeline"))
sys.path.insert(0, str(ROOT / "experiments" / "phase2_retrieval"))
sys.path.insert(0, str(ROOT / "experiments" / "evaluation"))

from pipeline_utils import TaxonomyValidator, estimate_api_cost, save_json, load_json  # noqa: E402
from utils import (  # noqa: E402
    compute_hierarchical_metrics,
    gold_levels,
    intent_levels,
    load_results,
    path_label,
)

# retrieval_utils (sentence_transformers) is OPTIONAL — only needed for DemoPool /
# annotation methods. Downstream classification must work without it.


def intent_path(row: dict[str, Any]) -> tuple[str, str, str]:
    """L1/L2/L3 path from a labelled row (no sentence-transformers dependency)."""
    return intent_levels(row)


def _get_embedding_retriever_cls():
    """Lazy import EmbeddingRetriever (requires sentence_transformers)."""
    try:
        from retrieval_utils import EmbeddingRetriever  # noqa: WPS440
    except ImportError as e:
        raise ImportError(
            "sentence_transformers is required for retrieval/demo features. "
            "Install: pip install sentence-transformers\n"
            f"Original error: {e}"
        ) from e
    return EmbeddingRetriever


# Paths
POOL_FILE = ROOT / "data/annotations/pool.json"
D3_FILE = ROOT / "data/gold/d3_dataset.json"
APPROVED_FILE = ROOT / "data/raw/hasaki/hasaki_labelled_approved_merged.json"
D2_DATASET_FILE = ROOT / "data/d2_dataset.json"
TAXONOMY_FILE = ROOT / "data/unified_intents.csv"
EDGES_FILE = ROOT / "data/intent_kb.intent_edges_28_05_2026.csv"
ANNOTATIONS_DIR = ROOT / "data/annotations"
DATASETS_DIR = ROOT / "data/datasets"
RESULTS_DIR = ROOT / "experiments/annotation_benchmark/results"
D3_VAL_IDS = ROOT / "data/splits/d3_val_ids.txt"
D3_TEST_IDS = ROOT / "data/splits/d3_test_ids.txt"
TRAIN_POOL_IDS = ROOT / "data/splits/train_pool_ids.txt"

METHODS = [
    "zero_shot",
    "few_shot",
    "cot",
    "freeal",
    "candist",
    "verified_d2",
]

METHOD_DISPLAY = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "cot": "CoT",
    "freeal": "FreeAL (LLM-only)",
    "candist": "CanDist (LLM-only)",
    "verified_d2": "Cross-Verified Hierarchical Intent Labeling (Ours)",
}

# Acronym for paper/tables: RVI-HUL
METHOD_ACRONYM = {
    "verified_d2": "RVI-HUL",
}


def load_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_id_list(path: Path) -> list[str]:
    return sorted(load_id_set(path))


@dataclass
class LLMUsage:
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    models: list[str] = field(default_factory=list)

    def add(self, model: str, input_tokens: int, output_tokens: int, latency: float):
        self.api_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += estimate_api_cost(model, input_tokens, output_tokens)
        self.latency_seconds += latency
        self.models.append(model)


@dataclass
class AnnotationResult:
    sample_id: str
    sentence: str
    predicted_label: dict[str, str]
    confidence: float = 0.0
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sentence": self.sentence,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }


class LLMClient:
    """OpenAI 2.x client with retry and JSON output."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
        mock: bool = False,
    ):
        self.mock = mock
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(
            temperature if temperature is not None else os.getenv("OPENAI_TEMPERATURE", "0")
        )
        self.max_retries = int(
            max_retries if max_retries is not None else os.getenv("OPENAI_MAX_RETRIES", "3")
        )
        self.client = None
        if not self.mock:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set (use --mock-llm for offline pipeline test)")
            self.client = OpenAI(api_key=api_key)

    def _mock_response(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        import hashlib

        text = messages[-1]["content"]
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        paths = sorted(TaxonomyIndex().all_l3_paths())
        l1, l2, l3 = paths[h % len(paths)]
        return {
            "level_1": l1,
            "level_2": l2,
            "level_3": l3,
            "confidence": 0.55 + (h % 40) / 100.0,
            "reasoning": "mock response for pipeline testing",
            "decision": "RETAIN",
        }

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], LLMUsage]:
        model = model or self.model
        if self.mock:
            data = self._mock_response(messages)
            usage = LLMUsage()
            usage.add(model, 500, 80, 0.01)
            return data, usage

        temp = self.temperature if temperature is None else temperature
        last_err = None
        start = time.time()
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                latency = time.time() - start
                content = resp.choices[0].message.content or "{}"
                usage = resp.usage
                in_tok = usage.prompt_tokens if usage else 0
                out_tok = usage.completion_tokens if usage else 0
                llm_usage = LLMUsage()
                llm_usage.add(model, in_tok, out_tok, latency)
                return extract_json_object(content), llm_usage
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} retries: {last_err}")


def extract_json_object(text: str) -> dict[str, Any]:
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
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


class TaxonomyIndex:
    """Load taxonomy and build prompt text + validator."""

    def __init__(self):
        self.validator = TaxonomyValidator(TAXONOMY_FILE, EDGES_FILE)
        self.df = pd.read_csv(TAXONOMY_FILE)
        self.l1_values = sorted({p[0] for p in self.validator.valid_paths})
        self._l2_by_l1: dict[str, list[str]] = {}
        self._l3_by_l2: dict[str, list[str]] = {}
        for l1, l2, l3 in sorted(self.validator.valid_paths):
            self._l2_by_l1.setdefault(l1, [])
            if l2 not in self._l2_by_l1[l1]:
                self._l2_by_l1[l1].append(l2)
            self._l3_by_l2.setdefault(l2, [])
            if l3 not in self._l3_by_l2[l2]:
                self._l3_by_l2[l2].append(l3)

    def is_valid(self, l1: str, l2: str, l3: str) -> bool:
        return self.validator.is_valid(l1, l2, l3)

    def compact_taxonomy_text(self, max_l3_per_l2: int = 8) -> str:
        lines = [
            "Taxonomy phân cấp (slug ASCII, lowercase, underscore):",
            f"L1: {', '.join(self.l1_values)}",
        ]
        for l1 in self.l1_values:
            lines.append(f"\n[{l1}]")
            for l2 in self._l2_by_l1.get(l1, [])[:20]:
                l3s = self._l3_by_l2.get(l2, [])[:max_l3_per_l2]
                l3_preview = ", ".join(l3s)
                if len(self._l3_by_l2.get(l2, [])) > max_l3_per_l2:
                    l3_preview += ", ..."
                lines.append(f"  L2={l2} -> L3: {l3_preview}")
        return "\n".join(lines)

    def all_l3_paths(self) -> list[tuple[str, str, str]]:
        return sorted(self.validator.valid_paths)


def parse_label_response(data: dict[str, Any]) -> tuple[str, str, str, float]:
    l1 = str(data.get("level_1") or data.get("l1") or data.get("L1") or "").strip()
    l2 = str(data.get("level_2") or data.get("l2") or data.get("L2") or "").strip()
    l3 = str(data.get("level_3") or data.get("l3") or data.get("L3") or "").strip()
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return l1, l2, l3, max(0.0, min(1.0, conf))


def make_annotation_result(
    sample: dict[str, Any],
    l1: str,
    l2: str,
    l3: str,
    confidence: float,
    usage: LLMUsage | None = None,
    metadata: dict[str, Any] | None = None,
) -> AnnotationResult:
    usage = usage or LLMUsage()
    return AnnotationResult(
        sample_id=sample["sample_id"],
        sentence=sample.get("sentence", ""),
        predicted_label={"l1": l1, "l2": l2, "l3": l3},
        confidence=confidence,
        api_calls=usage.api_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        metadata=metadata or {},
    )


def merge_usage(*usages: LLMUsage) -> LLMUsage:
    merged = LLMUsage()
    for u in usages:
        merged.api_calls += u.api_calls
        merged.input_tokens += u.input_tokens
        merged.output_tokens += u.output_tokens
        merged.cost_usd += u.cost_usd
        merged.latency_seconds += u.latency_seconds
        merged.models.extend(u.models)
    return merged


class DemoPool:
    """Fixed few-shot demos from approved data (excluding D3)."""

    def __init__(self, seed: int = 42, k_fixed: int = 5):
        self.seed = seed
        self.k_fixed = k_fixed
        self.d3_ids = load_id_set(D3_VAL_IDS) | load_id_set(D3_TEST_IDS)
        if not self.d3_ids:
            d3_rows = load_results(D3_FILE)
            self.d3_ids = {r["sample_id"] for r in d3_rows}
        approved = load_results(APPROVED_FILE)
        self.demos = [
            r for r in approved
            if r.get("sample_id") not in self.d3_ids
            and all(intent_path(r))
        ]
        self.fixed_demos = self._select_fixed_demos()
        self._retriever = None

    def _select_fixed_demos(self) -> list[dict[str, Any]]:
        rng = random.Random(self.seed)
        by_l1: dict[str, list] = {}
        for r in self.demos:
            l1 = intent_path(r)[0]
            by_l1.setdefault(l1, []).append(r)
        selected = []
        per_l1 = max(1, self.k_fixed // max(1, len(by_l1)))
        for l1, rows in sorted(by_l1.items()):
            rng.shuffle(rows)
            selected.extend(rows[:per_l1])
        rng.shuffle(selected)
        return selected[: self.k_fixed]

    def get_fixed_demos(self) -> list[dict[str, Any]]:
        return self.fixed_demos

    def get_retriever(self):
        if self._retriever is None:
            EmbeddingRetriever = _get_embedding_retriever_cls()
            self._retriever = EmbeddingRetriever(self.demos)
        return self._retriever

    def retrieve_demos(self, sentence: str, k: int = 5) -> list[dict[str, Any]]:
        retriever = self.get_retriever()
        hits = retriever.retrieve(sentence, k=k)
        return [self.demos[idx] for idx, _ in hits]

    @staticmethod
    def format_demo_block(demos: list[dict[str, Any]]) -> str:
        lines = []
        for i, d in enumerate(demos, 1):
            l1, l2, l3 = intent_path(d)
            lines.append(
                f"Ví dụ {i}:\n"
                f'  Câu: "{d.get("sentence", "")}"\n'
                f"  Nhãn: L1={l1}, L2={l2}, L3={l3}"
            )
        return "\n\n".join(lines)


def load_pool(limit: int = 0) -> list[dict[str, Any]]:
    doc = load_json(POOL_FILE)
    rows = doc["results"] if isinstance(doc, dict) else doc
    if limit > 0:
        return rows[:limit]
    return rows


def annotation_output_path(method: str) -> Path:
    return ANNOTATIONS_DIR / f"{method}.json"


def load_existing_annotations(method: str) -> dict[str, dict[str, Any]]:
    path = annotation_output_path(method)
    if not path.exists():
        return {}
    doc = load_json(path)
    rows = doc.get("results", doc if isinstance(doc, list) else [])
    return {r["sample_id"]: r for r in rows}


def save_annotation_run(
    method: str,
    results: list[dict[str, Any]],
    meta: dict[str, Any],
):
    payload = {
        "method": method,
        "display_name": METHOD_DISPLAY.get(method, method),
        "num_samples": len(results),
        **meta,
        "results": results,
    }
    save_json(annotation_output_path(method), payload)


def majority_vote_paths(paths: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    if not paths:
        return "", "", ""
    full = [path_label(*p) for p in paths if all(p)]
    if not full:
        return "", "", ""
    winner = Counter(full).most_common(1)[0][0]
    parts = winner.split("::", 2)
    return tuple(parts) if len(parts) == 3 else ("", "", winner)


def eval_predictions_vs_d3(
    predictions: list[dict[str, Any]],
    d3_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_by_id = {r["sample_id"]: r for r in d3_rows}
    true_paths = []
    pred_paths = []
    for pred in predictions:
        sid = pred["sample_id"]
        if sid not in gold_by_id:
            continue
        label = pred.get("predicted_label") or {}
        true_paths.append(gold_levels(gold_by_id[sid]))
        pred_paths.append((
            str(label.get("l1", "")),
            str(label.get("l2", "")),
            str(label.get("l3", "")),
        ))
    metrics = compute_hierarchical_metrics(true_paths, pred_paths)
    from sklearn.metrics import f1_score

    if true_paths:
        y_true = [p[2] for p in true_paths]
        y_pred = [p[2] for p in pred_paths]
        metrics["l3_weighted_f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    return metrics


def load_d3_gold() -> list[dict[str, Any]]:
    rows = load_results(D3_FILE)
    return [r for r in rows if all(gold_levels(r))]
