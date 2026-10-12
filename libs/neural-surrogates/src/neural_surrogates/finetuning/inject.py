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
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
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
    fused_forward: bool = False,
) -> "PeftModel":
    """Wrap ``model`` with LoRA adapters on ``target_modules`` and return it.

    Only ``lora_*`` (and any ``modules_to_save``) parameters end up trainable; the
    base weights are frozen by PEFT. At initialization the LoRA B-matrices are
    zero, so the wrapped model's outputs are byte-identical to ``model``'s.

    ``variant="balora"`` routes through the custom-dispatch layer from plan 04;
    it raises ``NotImplementedError`` here (this phase ships standard LoRA only).
    ``balora_cfg`` is accepted for forward-compatibility and ignored for now.

    ``fused_forward=True`` runs every adapted Linear / Conv3d as one call with the
    merged weight ``W + scaling * B @ A`` (see :func:`_fused_lora_forward`): the
    same function, faster when the rank is not small against the layer widths.
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
    peft_model = get_peft_model(model, config)
    # PEFT casts each LoRA layer's input to the adapter weights' dtype (fp32)
    # before ``lora_A``, which under bf16 autocast casts it straight back: a
    # transient fp32 copy plus a second bf16 copy of the input kept for
    # backward, next to the base layer's. Without the cast both branches share
    # one bf16 input (bf16 -> fp32 -> bf16 is exact, so outputs and gradients
    # are bit-identical); in fp32 the cast is a no-op either way. The fused
    # forward below never casts; this covers PEFT's forward and its fallbacks.
    for module in peft_model.modules():
        if hasattr(module, "cast_input_dtype_enabled"):
            module.cast_input_dtype_enabled = False
    if fused_forward:
        from peft.tuners.lora import layer as lora_layer

        for module in peft_model.modules():
            if isinstance(module, (lora_layer.Linear, lora_layer.Conv3d)):
                module.forward = types.MethodType(_fused_lora_forward, module)
    return peft_model


def _fused_lora_forward(self: Any, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
    """A PEFT LoRA Linear / Conv3d forward as one call with the merged weight.

    PEFT computes ``base(x) + B(A(x)) * scaling``: three convolutions, one of them
    with the base kernel at rank output channels, plus elementwise passes over the
    output. This computes ``base_with_weight(x, W + scaling * B @ A)`` instead, the
    same function up to rounding. The merge runs in fp32 outside autocast; the
    gradients reach ``A`` and ``B`` through the merged weight. The cost moves from
    the low-rank branch to one full weight-gradient convolution, which wins when
    the rank is not small against the channel widths (and always without grad).

    Anything other than plain, unmerged LoRA (dropout, a variant, a LoRA bias,
    grouped convs, mixed adapter batches, ``fan_in_fan_out``) falls back to PEFT.
    """
    base = self.get_base_layer()
    adapters = [a for a in self.active_adapters if a in self.lora_A]
    if (
        args
        or kwargs
        or self.disable_adapters
        or self.merged
        or getattr(self, "fan_in_fan_out", False)
        or getattr(base, "groups", 1) != 1
        or any(
            a in self.lora_variant
            or self.lora_bias[a]
            or not isinstance(self.lora_dropout[a], nn.Identity)
            for a in adapters
        )
    ):
        return type(self).forward(self, x, *args, **kwargs)
    weight = base.weight
    with torch.autocast(x.device.type, enabled=False):
        for a in adapters:
            lora_a = self.lora_A[a].weight.flatten(1)  # (r, in * kernel)
            lora_b = self.lora_B[a].weight.flatten(1)  # (out, r)
            delta = (lora_b @ lora_a).view(weight.shape) * self.scaling[a]
            weight = weight + delta.to(weight.dtype)
    if isinstance(base, nn.Linear):
        return F.linear(x, weight, base.bias)
    return base._conv_forward(x, weight, base.bias)


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
