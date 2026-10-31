#!/usr/bin/env python3
"""Utility functions for retrieval evaluation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


def load_results(path: Path | str) -> list[dict[str, Any]]:
    """Load results from JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, dict) and "results" in doc:
        return list(doc["results"])
    if isinstance(doc, list):
        return doc
    raise ValueError(f"Unsupported JSON shape in {path}")


def intent_path(row: dict[str, Any]) -> tuple[str, str, str]:
    """Extract (L1, L2, L3) intent path from a row."""
    intent = row.get("intent_3_level") or {}
    return (
        str(intent.get("level_1") or ""),
        str(intent.get("level_2") or ""),
        str(intent.get("level_3") or ""),
    )


def gold_intent_path(row: dict[str, Any]) -> tuple[str, str, str]:
    """Extract gold intent path (for D3 human gold)."""
    # Try gold_level_* first, fallback to intent_3_level
    if row.get("gold_level_1"):
        return (
            str(row.get("gold_level_1") or ""),
            str(row.get("gold_level_2") or ""),
            str(row.get("gold_level_3") or ""),
        )
    return intent_path(row)


class RegexRetriever:
    """Regex/BM25-like retrieval based on keyword matching."""
    
    def __init__(self, corpus: list[dict[str, Any]]):
        self.corpus = corpus
        self._build_index()
    
    def _build_index(self):
        """Build inverted index for fast lookup."""
        self.index = {}
        for idx, item in enumerate(self.corpus):
            # Index by intent slugs and keywords from sentence
            l1, l2, l3 = intent_path(item)
            sentence = item.get("sentence", "").lower()
            
            # Tokenize sentence
            tokens = set(re.findall(r'\w+', sentence))
            
            for token in tokens:
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append(idx)
    
    def retrieve(self, query: str, k: int = 40) -> list[tuple[int, float]]:
        """Retrieve top-k corpus items by token overlap."""
        query_tokens = set(re.findall(r'\w+', query.lower()))
        
        # Score each corpus item by token overlap
        scores = {}
        for token in query_tokens:
            if token in self.index:
                for idx in self.index[token]:
                    scores[idx] = scores.get(idx, 0) + 1
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]


class EmbeddingRetriever:
    """Dense embedding retrieval."""
    
    def __init__(self, corpus: list[dict[str, Any]], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        self._build_index()
    
    def _build_index(self):
        """Encode all corpus sentences."""
        sentences = [item.get("sentence", "") for item in self.corpus]
        self.embeddings = self.model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)
    
    def retrieve(self, query: str, k: int = 40) -> list[tuple[int, float]]:
        """Retrieve top-k by cosine similarity."""
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Cosine similarity
        scores = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class HybridRetriever:
    """Union of regex and embedding retrieval."""
    
    def __init__(self, corpus: list[dict[str, Any]], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.regex_retriever = RegexRetriever(corpus)
        self.embedding_retriever = EmbeddingRetriever(corpus, model_name)
    
    def retrieve(self, query: str, k: int = 40) -> list[tuple[int, float]]:
        """Retrieve union of regex and embedding results."""
        regex_results = self.regex_retriever.retrieve(query, k=k)
        emb_results = self.embedding_retriever.retrieve(query, k=k)
        
        # Combine scores (max of both)
        combined = {}
        for idx, score in regex_results:
            combined[idx] = score
        
        for idx, score in emb_results:
            if idx in combined:
                combined[idx] = max(combined[idx], score)
            else:
                combined[idx] = score
        
        # Sort and return top-k
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]


def compute_retrieval_metrics(
    retrieved_indices: list[int],
    gold_path: tuple[str, str, str],
    corpus: list[dict[str, Any]],
    k_values: list[int] = [1, 3, 5, 10]
) -> dict[str, float]:
    """Compute Recall@k and MRR for a single query."""
    l1_gold, l2_gold, l3_gold = gold_path
    
    # Find rank of first correct match
    first_correct_rank = None
    for rank, idx in enumerate(retrieved_indices, start=1):
        l1, l2, l3 = intent_path(corpus[idx])
        if l3 == l3_gold:  # Match at L3 level
            first_correct_rank = rank
            break
    
    metrics = {}
    
    # Recall@k
    for k in k_values:
        top_k_indices = retrieved_indices[:k]
        found = any(
            intent_path(corpus[idx])[2] == l3_gold
            for idx in top_k_indices
        )
        metrics[f"recall@{k}"] = 1.0 if found else 0.0
    
    # MRR
    if first_correct_rank is not None:
        metrics["mrr"] = 1.0 / first_correct_rank
    else:
        metrics["mrr"] = 0.0
    
    # Taxonomy coverage (how many unique L3s in top-k)
    unique_l3s = set()
    for idx in retrieved_indices[:max(k_values)]:
        l3 = intent_path(corpus[idx])[2]
        if l3:
            unique_l3s.add(l3)
    metrics["taxonomy_coverage"] = len(unique_l3s)
    
    return metrics


def save_json(path: Path, data: Any):
    """Save JSON with proper encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
