"""PEFT (LoRA) injection, adapter save/load and merged export.

The single LoRA stack for the whole repo is HF **PEFT** (see the master plan's
"Unified LoRA strategy"): plain architectures (our ``P3D`` wrapper etc.) get
``get_peft_model(model, LoraConfig(...))`` directly on the module. This module is
the thin wrapper around that so the fine-tune script -- and later the Tadpole
(plan 03) and BaLoRA (plan 04) paths -- share one code path.

``peft`` is imported lazily inside each function so ``import neural_surrogates``
(and hence ``import neural_surrogates.finetuning``) stays free of the
transformers/accelerate stack peft pulls in; install it with
``pip install 'neural_surrogates[finetuning]'`` (or via the pixi dev env).

Key invariant for ESMDA compatibility: :func:`merge_to_state_dict` returns a
*plain, full* base-architecture state dict (LoRA folded into the base weights),
byte-loadable into a freshly instantiated architecture -- so a fine-tuned
``weights.pt`` is indistinguishable from a fully trained one and
``NeuralSurrogateForwardModel`` needs zero changes. Normalization buffers
(``state_mean/std``, ``param_mean/std``) are plain buffers on the base model,
untouched by PEFT and included in the merged export.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing peft eagerly
    from peft import PeftModel


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    target_modules: str | list[str],
    variant: str = "standard",
    modules_to_save: list[str] | None = None,
    balora_cfg: dict | None = None,
) -> "PeftModel":
    """Wrap ``model`` with LoRA adapters on ``target_modules`` and return it.

    Only ``lora_*`` (and any ``modules_to_save``) parameters end up trainable; the
    base weights are frozen by PEFT. At initialization the LoRA B-matrices are
    zero, so the wrapped model's outputs are byte-identical to ``model``'s.

    ``variant="balora"`` routes through the custom-dispatch layer from plan 04;
    it raises ``NotImplementedError`` here (this phase ships standard LoRA only).
    ``balora_cfg`` is accepted for forward-compatibility and ignored for now.
    """
    if variant == "balora":
        raise NotImplementedError(
            "The 'balora' LoRA variant is delivered in plan 04 "
            "(Bayesian LoRA); this phase supports variant='standard' only."
        )
    if variant != "standard":
        raise ValueError(f"variant must be 'standard' or 'balora', got {variant!r}")

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=target_modules,
        modules_to_save=list(modules_to_save) if modules_to_save else None,
        bias="none",
    )
    return get_peft_model(model, config)


def merge_to_state_dict(peft_model: "PeftModel") -> dict[str, torch.Tensor]:
    """Merge LoRA into the base weights and return a plain base state dict.

    Runs ``merge_and_unload`` on a **deepcopy** so ``peft_model`` itself stays
    intact (the fine-tune script still needs it to save the adapter afterwards).
    The result is a full state dict of the underlying architecture -- no
    ``base_layer`` / ``lora_*`` keys -- loadable into a freshly instantiated model.

    Cost: a ``deepcopy`` + merge on each call (``BaseTraining`` calls this on every
    val improvement). Cheap per epoch, non-trivial repeated allocation over a long
    run -- acceptable for short fine-tunes. Because the deepcopy happens on the
    training device (right after validation), it transiently roughly doubles model
    memory -- expect a ~1x model VRAM spike for the duration of the merge. Tensors
    are detached to **CPU** so the transient device-side deepcopy is freed on
    return (no GPU-memory retention via the returned dict) and the saved
    ``weights.pt`` is device-agnostic.
    """
    merged = copy.deepcopy(peft_model).merge_and_unload()
    return {k: v.detach().cpu() for k, v in merged.state_dict().items()}


def save_adapter(peft_model: "PeftModel", adapter_dir: str | Path) -> None:
    """Write the PEFT adapter (``adapter_model.safetensors`` + config) to disk."""
    peft_model.save_pretrained(str(adapter_dir))


def load_adapter(base_model: nn.Module, adapter_dir: str | Path) -> "PeftModel":
    """Re-attach a saved adapter to a freshly built base model."""
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, str(adapter_dir))
