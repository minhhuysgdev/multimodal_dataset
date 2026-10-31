"""LoRA helpers for downstream transformer fine-tuning."""
from __future__ import annotations


def lora_target_modules(model_name: str) -> list[str]:
    """Return PEFT target module names for a HuggingFace model id."""
    name = model_name.lower()
    if "mistral" in name or "qwen" in name or "llama" in name:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    if "roberta" in name or "phobert" in name or "bert" in name:
        return ["query", "value"]
    return ["query", "value"]


def _is_decoder_llm(model_name: str) -> bool:
    name = model_name.lower()
    return any(k in name for k in ("qwen", "mistral", "llama", "gemma", "phi"))


def apply_lora(
    model,
    model_name: str,
    *,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
):
    """Wrap a sequence-classification model with LoRA adapters.

    For decoder LLMs (Qwen/Mistral/...), also train the classification head
    via modules_to_save — otherwise the randomly-init score layer stays frozen.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    modules_to_save = None
    candidates = []
    if hasattr(model, "l1_classifier"):
        candidates.append("l1_classifier")
    if _is_decoder_llm(model_name):
        # Only keep heads that actually exist on this model
        for name in ("score", "classifier"):
            if hasattr(model, name):
                candidates.append(name)
    modules_to_save = candidates or None

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=lora_target_modules(model_name),
        bias="none",
        modules_to_save=modules_to_save,
    )
    peft_model = get_peft_model(model, config)
    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass
    return peft_model
