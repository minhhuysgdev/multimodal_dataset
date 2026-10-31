#!/usr/bin/env python3
"""Fine-tune transformer encoders / LLMs for hierarchical intent classification.

Supports PhoBERT, XLM-RoBERTa (full FT) and Qwen3 / Mistral (LoRA) on CUDA/MPS/CPU.

Usage:
  python experiments/evaluation/train_transformer.py \\
    --model-name vinai/phobert-base --train-set D1

  python experiments/evaluation/train_transformer.py \\
    --model-name FacebookAI/xlm-roberta-base --train-set D2

  python experiments/evaluation/train_transformer.py \\
    --model-name Qwen/Qwen3-1.7B --use-lora --train-set D2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "evaluation"))

from lora_utils import apply_lora  # noqa: E402
from utils import (  # noqa: E402
    ALLOWED_L1,
    compute_hierarchical_metrics,
    decode_from_class_probs,
    gold_levels,
    intent_levels,
    load_results,
    path_label,
    rows_with_gold,
    save_json,
)

L1_LABELS = sorted(ALLOWED_L1)
L1_LABEL2ID = {label: i for i, label in enumerate(L1_LABELS)}
L1_ID2LABEL = {i: label for label, i in L1_LABEL2ID.items()}

DEFAULT_D1_TRAIN = ROOT / "data/datasets/d1_annotator_train.json"
DEFAULT_D2_TRAIN = ROOT / "data/datasets/d2_verified_train.json"
DEFAULT_D3 = ROOT / "data/gold/d3_dataset.json"
DEFAULT_OUT = ROOT / "experiments/evaluation/results"


def _require_torch():
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        return (
            torch,
            Dataset,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        print(
            "Missing dependencies. Install:\n"
            "  pip install torch transformers peft accelerate scikit-learn",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def get_device(torch):
    """Pick best available device (MPS for Apple Silicon)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_tokenizer(tokenizer, model, model_name: str = ""):
    """Ensure pad token exists (required for Qwen/Mistral batching).

    Decoder LLMs should use left padding so the last token carries the label.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    lower = (model_name or "").lower()
    if any(k in lower for k in ("qwen", "mistral", "llama", "gemma", "phi")):
        tokenizer.padding_side = "left"
    return tokenizer


def _needs_trust_remote_code(model_name: str) -> bool:
    lower = model_name.lower()
    return "qwen" in lower


class IntentDataset:
    """Mixin-style helpers; subclass torch.utils.data.Dataset in train_and_eval_transformer."""

    def __init__(
        self,
        texts,
        labels,
        label2id,
        tokenizer,
        max_length=128,
        *,
        l1_labels: list[int] | None = None,
    ):
        self.texts = texts
        self.labels = labels
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.l1_labels = l1_labels

    def __len__(self):
        return len(self.texts)

    def _encode_item(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = self.label2id[self.labels[idx]]
        if self.l1_labels is not None:
            item["l1_labels"] = int(self.l1_labels[idx])
        return item


def _resolve_backbone(model):
    """Return the transformer backbone module for pooling."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        inner = model.base_model.model
        if hasattr(inner, "roberta"):
            return inner.roberta
        if hasattr(inner, "bert"):
            return inner.bert
        if hasattr(inner, "model"):
            return inner.model
        return inner
    if hasattr(model, "roberta"):
        return model.roberta
    if hasattr(model, "bert"):
        return model.bert
    if hasattr(model, "model"):
        return model.model
    raise AttributeError("Cannot resolve transformer backbone for pooling")


def _pool_hidden(backbone_outputs, attention_mask, torch):
    if getattr(backbone_outputs, "pooler_output", None) is not None:
        return backbone_outputs.pooler_output
    last_hidden = backbone_outputs.last_hidden_state
    if attention_mask is None:
        return last_hidden[:, 0, :]
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _path_logits(model, enc, torch):
    """Forward pass returning path classification logits."""
    outputs = model(**enc)
    return outputs.logits


def _l1_logits(model, enc, torch):
    """Forward pass returning L1 head logits (requires model.l1_classifier)."""
    backbone = _resolve_backbone(model)
    backbone_outputs = backbone(
        input_ids=enc["input_ids"],
        attention_mask=enc.get("attention_mask"),
    )
    pooled = _pool_hidden(backbone_outputs, enc.get("attention_mask"), torch)
    return model.l1_classifier(pooled)


def _attach_l1_head(model, torch):
    hidden = model.config.hidden_size
    model.l1_classifier = torch.nn.Linear(hidden, len(L1_LABELS))
    return model


def predict_batched(
    model,
    tokenizer,
    texts: list[str],
    device,
    *,
    batch_size: int = 8,
    max_length: int = 128,
    return_probs: bool = False,
    use_l1_head: bool = False,
):
    """Run inference in batches to avoid OOM on large models.

    return_probs=False -> list[int] argmax class ids (legacy)
    return_probs=True  -> (path_probs, l1_pred_ids|None)
    """
    import torch

    model.eval()
    pred_ids: list[int] = []
    prob_chunks: list = []
    l1_pred_chunks: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = _path_logits(model, enc, torch)
            if return_probs:
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                prob_chunks.append(probs)
                if use_l1_head and hasattr(model, "l1_classifier"):
                    l1_logits = _l1_logits(model, enc, torch)
                    l1_pred_chunks.extend(l1_logits.argmax(dim=-1).cpu().tolist())
            else:
                pred_ids.extend(logits.argmax(dim=-1).cpu().tolist())
    if return_probs:
        import numpy as np

        path_probs = (
            np.concatenate(prob_chunks, axis=0) if prob_chunks else np.zeros((0, 0))
        )
        l1_preds = l1_pred_chunks if use_l1_head else None
        return path_probs, l1_preds
    return pred_ids


def _build_multitask_trainer(
    Trainer,
    *,
    lambda_l1: float,
    use_l1_head: bool,
    torch,
):
    class MultiTaskTrainer(Trainer):
        def __init__(self, *args, lambda_l1=1.0, use_l1_head=True, **kwargs):
            self.lambda_l1 = float(lambda_l1)
            self.use_l1_head = bool(use_l1_head)
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            l1_labels = inputs.pop("l1_labels", None)
            path_logits = _path_logits(model, inputs, torch)
            loss = torch.nn.functional.cross_entropy(path_logits, labels)
            if self.use_l1_head and l1_labels is not None:
                l1_logits = _l1_logits(model, inputs, torch)
                l1_loss = torch.nn.functional.cross_entropy(l1_logits, l1_labels)
                loss = loss + self.lambda_l1 * l1_loss
            return (loss, {"logits": path_logits}) if return_outputs else loss

    return MultiTaskTrainer


def train_and_eval_transformer(
    train_rows: list,
    test_rows: list,
    *,
    model_name: str,
    train_name: str,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    output_dir: Path | None = None,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    gradient_accumulation_steps: int = 1,
    eval_batch_size: int | None = None,
    decoding: str = "marginal",
    use_l1_head: bool = True,
    lambda_l1: float = 1.0,
) -> dict:
    torch, DatasetBase, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments = _require_torch()

    class IntentTorchDataset(DatasetBase, IntentDataset):
        def __getitem__(self, idx):
            return self._encode_item(idx)

    X_train = [r["sentence"] for r in train_rows]
    y_train = [path_label(*intent_levels(r)) for r in train_rows]
    y_train_l1 = [
        L1_LABEL2ID.get(intent_levels(r)[0], 0) for r in train_rows
    ]

    X_test = [r["sentence"] for r in test_rows]
    y_true_paths = [gold_levels(r) for r in test_rows]

    all_labels = sorted(set(y_train) | {path_label(*gold_levels(r)) for r in test_rows})
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    trust = _needs_trust_remote_code(model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    except Exception as e:
        raise RuntimeError(
            f"Không tải được tokenizer '{model_name}'. "
            f"Với Mistral (gated): đặt HF_TOKEN / huggingface-cli login. "
            f"Với Qwen3: cần transformers>=4.51. Chi tiết: {e}"
        ) from e

    device = get_device(torch)
    # float16 chỉ ổn định trên CUDA; MPS dùng float32 để tránh NaN/diverge
    dtype = torch.float32

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(all_labels),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=dtype,
            trust_remote_code=trust,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Không tải được model '{model_name}' cho sequence classification. "
            f"Kiểm tra: (1) HF token nếu model gated, "
            f"(2) transformers đủ mới cho Qwen3, "
            f"(3) đủ RAM/VRAM. Chi tiết: {e}"
        ) from e

    tokenizer = prepare_tokenizer(tokenizer, model, model_name)

    if use_l1_head:
        model = _attach_l1_head(model, torch)

    if use_lora:
        model = apply_lora(model, model_name, r=lora_r, alpha=lora_alpha)

    l1_for_ds = y_train_l1 if use_l1_head else None
    train_ds = IntentTorchDataset(
        X_train, y_train, label2id, tokenizer, max_length, l1_labels=l1_for_ds
    )

    run_dir = output_dir or Path(f"./runs/{train_name}_{model_name.replace('/', '_')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = device.type == "cuda"
    use_bf16 = False  # giữ False trên CUDA (dùng fp16); MPS dùng float32 ở trên
    if device.type == "mps":
        use_fp16 = False
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-5 if not use_lora else 1e-4,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=42,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=device.type == "cuda",
        remove_unused_columns=False,
    )

    TrainerCls = (
        _build_multitask_trainer(
            Trainer, lambda_l1=lambda_l1, use_l1_head=use_l1_head, torch=torch
        )
        if use_l1_head
        else Trainer
    )
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )
    if use_l1_head:
        trainer_kwargs["lambda_l1"] = lambda_l1
        trainer_kwargs["use_l1_head"] = use_l1_head
    try:
        trainer = TrainerCls(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = TrainerCls(**trainer_kwargs, tokenizer=tokenizer)
    trainer.train()

    model.to(device)
    infer_bs = eval_batch_size or batch_size
    probs, l1_pred_ids = predict_batched(
        model,
        tokenizer,
        X_test,
        device,
        batch_size=infer_bs,
        max_length=max_length,
        return_probs=True,
        use_l1_head=use_l1_head,
    )

    class_labels = [id2label[i] for i in range(len(id2label))]
    level_preds, argmax_paths = decode_from_class_probs(
        probs, class_labels, decoding=decoding
    )
    if use_l1_head and l1_pred_ids is not None:
        level_preds = [
            (L1_ID2LABEL.get(int(l1_pred_ids[i]), lp[0]), lp[1], lp[2])
            for i, lp in enumerate(level_preds)
        ]
    hier = compute_hierarchical_metrics(
        y_true_paths,
        level_preds,
        full_path_preds=argmax_paths,
    )

    return {
        "train_set": train_name,
        "model_name": model_name,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "num_labels": len(all_labels),
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_lora": use_lora,
        "decoding": decoding,
        "use_l1_head": use_l1_head,
        "lambda_l1": lambda_l1 if use_l1_head else 0.0,
        "device": str(device),
        "l3_accuracy": hier["l3_accuracy"],
        "l3_macro_f1": hier["l3_macro_f1"],
        "full_path_accuracy": hier["full_path_accuracy"],
        "l1_accuracy": hier["l1_accuracy"],
        "l2_accuracy": hier["l2_accuracy"],
        "output_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="vinai/phobert-base",
        help="HuggingFace model id",
    )
    parser.add_argument(
        "--train-set",
        choices=["D1", "D2", "both"],
        default="both",
    )
    parser.add_argument("--d1-train", type=Path, default=DEFAULT_D1_TRAIN)
    parser.add_argument("--d2-train", type=Path, default=DEFAULT_D2_TRAIN)
    parser.add_argument("--d3-file", type=Path, default=DEFAULT_D3)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument(
        "--decoding",
        choices=["argmax", "marginal"],
        default="marginal",
        help="Level decoding: marginal sums path probs per L1/L2/L3 (default)",
    )
    parser.add_argument(
        "--l1-head",
        action="store_true",
        default=True,
        help="Use dedicated L1 classification head (default: on)",
    )
    parser.add_argument(
        "--no-l1-head",
        action="store_true",
        help="Disable L1 head (path-only training)",
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=1.0,
        help="Weight for L1 auxiliary loss when --l1-head is enabled",
    )
    args = parser.parse_args()
    use_l1_head = not args.no_l1_head

    d3_all = load_results(args.d3_file)
    d3_gold = rows_with_gold(d3_all)
    if not d3_gold:
        print(
            "ERROR: D3 has no annotated gold labels. Annotate first, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    results = []
    runs_dir = args.outdir / "models" / "transformer"

    configs = []
    if args.train_set in ("D1", "both"):
        configs.append(("D1_annotator_train", args.d1_train))
    if args.train_set in ("D2", "both"):
        configs.append(("D2_verified_train", args.d2_train))

    for train_name, train_path in configs:
        train_rows = load_results(train_path)
        if not train_rows:
            print(f"WARNING: {train_path} empty — run build_d1_d2_train.py first")
            continue
        r = train_and_eval_transformer(
            train_rows,
            d3_gold,
            model_name=args.model_name,
            train_name=train_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            output_dir=runs_dir / f"{train_name}_{args.model_name.replace('/', '_')}",
            use_lora=args.use_lora,
            gradient_accumulation_steps=args.grad_accum,
            decoding=args.decoding,
            use_l1_head=use_l1_head,
            lambda_l1=args.lambda_l1,
        )
        results.append(r)
        print(
            f"{train_name} + {args.model_name}: "
            f"L3 Acc={r['l3_accuracy']:.4f}, "
            f"Macro-F1={r['l3_macro_f1']:.4f}, "
            f"Full-path={r['full_path_accuracy']:.4f}"
        )

    summary = [
        {
            "train_set": r["train_set"],
            "model_name": r["model_name"],
            "l3_accuracy": r["l3_accuracy"],
            "l3_macro_f1": r["l3_macro_f1"],
            "full_path_accuracy": r["full_path_accuracy"],
        }
        for r in results
    ]
    safe_name = args.model_name.replace("/", "_")
    out_path = args.outdir / f"transformer_{safe_name}_metrics.json"
    save_json(out_path, {"model_name": args.model_name, "results": results, "summary": summary})
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
