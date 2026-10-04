"""LoRA fine-tuning (plan 01): unit tests.

* **Unit** (``neural_surrogates.finetuning``): LoRA injection is a no-op at init
  (B=0), merging round-trips to a plain state dict byte-identical to the base at
  init, and only the adapter (+ ``modules_to_save``) parameters train. Run on P3D
  (the plan's focus); skipped if ``p3d_surrogate`` is absent.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from neural_surrogates import UNetConvNeXt
from neural_surrogates.finetuning import (
    all_adaptable_module_names,
    inject_lora,
    merge_to_state_dict,
    resolve_target_modules,
)

# --------------------------------------------------------------------------- #
# Unit tests on P3D (the plan's focus architecture).
# --------------------------------------------------------------------------- #


def _p3d(n_params: Any = 2) -> Any:
    p3d = pytest.importorskip("neural_surrogates.architectures.p3d")
    pytest.importorskip("p3d_surrogate")
    return p3d.P3D(
        n_state_channels=3,
        n_params=n_params,
        size="S",
        normalize=True,
        predict_residual=True,
        param_conditioning="native",
    )


def _p3d_inputs(b: Any = 2, grid: Any = (16, 16, 16)) -> Any:
    state = torch.randn(b, 3, *grid)
    params = torch.randn(b, 2)
    geom = (torch.rand(b, *grid) > 0.2).float()
    return state, params, geom


def test_inject_is_identity_at_init() -> None:
    """LoRA B=0 at init => the wrapped model's output equals the base's."""
    model = _p3d().eval()
    peft_model = inject_lora(
        copy.deepcopy(model),
        rank=8,
        alpha=16,
        target_modules=resolve_target_modules(model, preset="attention"),
    ).eval()
    state, params, geom = _p3d_inputs()
    with torch.no_grad():
        base = model(state, params, geom)
        wrapped = peft_model(state, params, geom)
    assert torch.allclose(base, wrapped, atol=1e-6)


def test_merge_round_trips_to_plain_state_dict() -> None:
    """merge_to_state_dict yields a plain base state dict, identical at init."""
    model = _p3d().eval()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    peft_model = inject_lora(
        copy.deepcopy(model),
        rank=8,
        alpha=16,
        target_modules=resolve_target_modules(model, preset="attention+conv"),
    )
    merged = merge_to_state_dict(peft_model)
    # loads strictly into a freshly instantiated architecture (no lora_ keys)
    fresh = _p3d()
    fresh.load_state_dict(merged)  # raises on any missing/unexpected key
    # B=0 at init => merged weights are byte-identical to the base
    for k, v in base_state.items():
        assert torch.allclose(merged[k], v, atol=1e-6), k


def test_merge_parity_at_nonzero_lora_b() -> None:
    """merge_to_state_dict reproduces the wrapped forward at *nonzero* LoRA B.

    The init round-trip (B=0) only proves the pass-through path; it never
    exercises peft's actual delta math -- the alpha/r scaling and the Conv3d
    weight fold. This repo has already hit two peft-0.19 conv-merge bugs, so
    assert the merged base weights reproduce the adapted forward end-to-end:
    perturb the trainable ``lora_B`` matrices to nonzero, then compare the
    wrapped model's output against a fresh base model loaded with the merged
    state dict, on a nontrivial input. The tolerance is loosened to ~1e-5 for
    the conv-fold arithmetic.
    """
    model = _p3d().eval()
    peft_model = inject_lora(
        copy.deepcopy(model),
        rank=8,
        alpha=16,
        target_modules=resolve_target_modules(model, preset="attention+conv"),
    )
    # Drive the adapter off the zero fixed point: bump every trainable lora_B
    # (zero-initialised by peft) so the merged delta is genuinely nonzero.
    with torch.no_grad():
        n_bumped = 0
        for n, p in peft_model.named_parameters():
            if "lora_B" in n and p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
                n_bumped += 1
    assert n_bumped, "no trainable lora_B parameters were perturbed"

    peft_model.eval()
    state, params, geom = _p3d_inputs()
    merged = merge_to_state_dict(peft_model)
    fresh = _p3d().eval()
    fresh.load_state_dict(merged)  # strict: plain base state dict, no lora_ keys
    with torch.no_grad():
        base = model(state, params, geom)
        wrapped = peft_model(state, params, geom)
        reloaded = fresh(state, params, geom)

    # sanity: nonzero B actually moved the output away from the base model
    assert not torch.allclose(
        wrapped, base, atol=1e-4
    ), "perturbation did not change the output; delta math is untested"
    # the folded base weights must reproduce the adapted (nonzero-B) forward
    assert torch.allclose(wrapped, reloaded, atol=1e-5)


def test_only_adapter_and_modules_to_save_train() -> None:
    """Only lora_* and the explicit modules_to_save head require grad."""
    model = _p3d().eval()
    peft_model = inject_lora(
        copy.deepcopy(model),
        rank=8,
        alpha=16,
        target_modules=resolve_target_modules(model, preset="attention"),
        modules_to_save=["param_to_scalar"],
    )
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable"
    for n in trainable:
        assert "lora_" in n or "param_to_scalar" in n, n
    # the frozen base weights (e.g. the qkv base_layer) must not train
    frozen = [
        n
        for n, p in peft_model.named_parameters()
        if not p.requires_grad and "attn.qkv.base_layer" in n
    ]
    assert frozen, "expected a frozen base qkv layer"


def test_merge_preserves_trained_modules_to_save_head() -> None:
    """merge_and_unload must fold a trained modules_to_save head back in.

    PEFT wraps a modules_to_save target in a ModulesToSaveWrapper; the merge has
    to unwrap it into a plain module carrying the *trained* weights (the riskiest
    merge path). Perturb the head's trainable copy, merge, load into a fresh
    architecture, and assert the head is the trained one — not the base.
    """
    model = _p3d().eval()  # n_params=2 => param_to_scalar (a Linear(2,1)) exists
    base_head = model.param_to_scalar.weight.detach().clone()
    peft_model = inject_lora(
        copy.deepcopy(model),
        rank=8,
        alpha=16,
        target_modules=resolve_target_modules(model, preset="attention"),
        modules_to_save=["param_to_scalar"],
    )
    # "Train" the head: bump its trainable (modules_to_save) copy by a constant.
    with torch.no_grad():
        n_bumped = 0
        for n, p in peft_model.named_parameters():
            if "param_to_scalar" in n and "weight" in n and p.requires_grad:
                p.add_(1.0)
                n_bumped += 1
    assert n_bumped == 1, "expected exactly one trainable param_to_scalar weight"

    merged = merge_to_state_dict(peft_model)
    fresh = _p3d()
    fresh.load_state_dict(merged)  # strict: no lora_/modules_to_save keys survive
    # the merged head must be the trained one (base + 1), not the base
    assert torch.allclose(fresh.param_to_scalar.weight, base_head + 1.0, atol=1e-6)
    assert not torch.allclose(fresh.param_to_scalar.weight, base_head, atol=1e-6)


# --------------------------------------------------------------------------- #
# Unit test on the generic enumerator (no P3D needed).
# --------------------------------------------------------------------------- #


def test_all_preset_skips_grouped_and_1x1_convs() -> None:
    """all_adaptable_module_names drops depthwise + 1x1x1 convs (unmergeable)."""
    model = UNetConvNeXt(
        n_state_channels=3,
        n_params=2,
        base_channels=4,
        channel_mults=(1, 2),
        depths=(1, 1),
        kernel_size=3,
        expansion=2,
    )
    names = set(all_adaptable_module_names(model))
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and (
            module.groups != 1 or all(k == 1 for k in module.kernel_size)
        ):
            assert name not in names, f"{name} should be excluded"
    # injecting on the resolved 'all' targets must not raise (grouped/1x1 excluded)
    targets = resolve_target_modules(model, preset="all")
    inject_lora(copy.deepcopy(model), rank=4, alpha=8, target_modules=targets)
