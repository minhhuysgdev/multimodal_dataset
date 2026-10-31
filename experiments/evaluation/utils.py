"""Shared helpers for the D3 evaluation pipeline."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ALLOWED_L1 = {"truoc_mua_hang", "sau_mua_hang"}
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, dict) and "results" in doc:
        return list(doc["results"])
    if isinstance(doc, list):
        return doc
    raise ValueError(f"Unsupported JSON shape in {path}")


def intent_levels(row: dict[str, Any]) -> tuple[str, str, str]:
    intent = row.get("intent_3_level") or {}
    if not intent and row.get("gold_level_1"):
        return (
            str(row.get("gold_level_1") or ""),
            str(row.get("gold_level_2") or ""),
            str(row.get("gold_level_3") or ""),
        )
    return (
        str(intent.get("level_1") or ""),
        str(intent.get("level_2") or ""),
        str(intent.get("level_3") or ""),
    )


def gold_levels(row: dict[str, Any]) -> tuple[str, str, str]:
    """Human gold labels from D3 row (supports template + annotated formats)."""
    if row.get("gold_level_1"):
        return (
            str(row["gold_level_1"]),
            str(row.get("gold_level_2") or ""),
            str(row.get("gold_level_3") or ""),
        )
    intent = row.get("intent_3_level") or {}
    return (
        str(intent.get("level_1") or ""),
        str(intent.get("level_2") or ""),
        str(intent.get("level_3") or ""),
    )


def machine_levels(row: dict[str, Any]) -> tuple[str, str, str]:
    if row.get("machine_level_1"):
        return (
            str(row["machine_level_1"]),
            str(row.get("machine_level_2") or ""),
            str(row.get("machine_level_3") or ""),
        )
    return intent_levels(row)


def path_label(l1: str, l2: str, l3: str) -> str:
    return f"{l1}::{l2}::{l3}"


def parse_path_label(label: str) -> tuple[str, str, str]:
    parts = str(label).split("::", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "", "", str(label)


def is_valid_intent_row(row: dict[str, Any]) -> bool:
    """Guardrail-style check for train labels (L1 closed set + slug L2/L3)."""
    l1, l2, l3 = intent_levels(row)
    if not (l1 and l2 and l3):
        return False
    if l1 not in ALLOWED_L1:
        return False
    if not _SLUG_RE.match(l2) or not _SLUG_RE.match(l3):
        return False
    if l2 == l3:
        return False
    return True


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def decode_from_class_probs(
    probs: np.ndarray,
    class_labels: list[str],
    *,
    decoding: str = "marginal",
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Decode path-class probabilities into level preds + argmax full-path preds.

    decoding:
      - "argmax": L1/L2/L3 taken from the single best path (legacy behavior)
      - "marginal": L1/L2/L3 = argmax over summed probability mass per level
    Full-path predictions always use argmax path.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N, C), got shape {probs.shape}")
    if probs.shape[1] != len(class_labels):
        raise ValueError(
            f"probs columns ({probs.shape[1]}) != len(class_labels) ({len(class_labels)})"
        )

    argmax_ids = probs.argmax(axis=1)
    argmax_paths = [parse_path_label(class_labels[i]) for i in argmax_ids]

    if decoding == "argmax":
        return argmax_paths, argmax_paths
    if decoding != "marginal":
        raise ValueError(f"Unknown decoding={decoding!r}; use 'argmax' or 'marginal'")

    l1_idxs: dict[str, list[int]] = defaultdict(list)
    l2_idxs: dict[str, list[int]] = defaultdict(list)
    l3_idxs: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(class_labels):
        l1, l2, l3 = parse_path_label(lab)
        l1_idxs[l1].append(i)
        l2_idxs[l2].append(i)
        l3_idxs[l3].append(i)

    def _pick(row_probs: np.ndarray, groups: dict[str, list[int]]) -> str:
        best_k, best_s = "", -1.0
        for key, idxs in groups.items():
            score = float(row_probs[idxs].sum())
            if score > best_s:
                best_s, best_k = score, key
        return best_k

    level_preds = [
        (_pick(probs[n], l1_idxs), _pick(probs[n], l2_idxs), _pick(probs[n], l3_idxs))
        for n in range(probs.shape[0])
    ]
    return level_preds, argmax_paths


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    level: str = "L3",
) -> dict[str, float]:
    """Compute accuracy, macro-F1, and full-path accuracy for hierarchical labels."""
    if level == "path":
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "full_path_accuracy": float(acc),
            "n_samples": len(y_true),
        }

    # Single-level comparison
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "n_samples": len(y_true),
    }


def compute_hierarchical_metrics(
    true_paths: list[tuple[str, str, str]],
    pred_paths: list[tuple[str, str, str]],
    *,
    full_path_preds: list[tuple[str, str, str]] | None = None,
) -> dict[str, float]:
    """Level metrics from pred_paths; full-path Acc from full_path_preds (or pred_paths)."""
    l1_t, l2_t, l3_t = zip(*true_paths) if true_paths else ([], [], [])
    l1_p, l2_p, l3_p = zip(*pred_paths) if pred_paths else ([], [], [])

    path_preds = full_path_preds if full_path_preds is not None else pred_paths
    full_path = [path_label(a, b, c) for a, b, c in true_paths]
    pred_full = [path_label(a, b, c) for a, b, c in path_preds]

    return {
        "l1_accuracy": float(accuracy_score(l1_t, l1_p)),
        "l2_accuracy": float(accuracy_score(l2_t, l2_p)),
        "l3_accuracy": float(accuracy_score(l3_t, l3_p)),
        "l3_macro_f1": float(f1_score(l3_t, l3_p, average="macro", zero_division=0)),
        "full_path_accuracy": float(accuracy_score(full_path, pred_full)),
        "n_samples": len(true_paths),
    }


def rows_with_gold(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only rows where human gold L1/L2/L3 are filled."""
    out = []
    for r in rows:
        g = gold_levels(r)
        if all(g):
            out.append(r)
    return out


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
