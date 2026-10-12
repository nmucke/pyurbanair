"""Activation-memory savings that keep the DFT's math: the trainer's
``activation_checkpoint_modules`` knob and LoRA layers sharing their input
with the base layer (no PEFT input cast).

On a tiny LoRA-wrapped ``TadpoleTimeStepper`` in the DFT setup (global spatial
mode, geometry branch, skip mixing), checkpointing the encoder/decoder must
leave the loss and the gradient of every trainable parameter -- LoRA adapters
of the frozen layers included -- as they are without it, on CPU in fp32 and
under bf16 autocast.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("einops")

from neural_surrogates import TadpoleTimeStepper, Trainer
from neural_surrogates.finetuning import (
    inject_lora,
    merge_to_state_dict,
    resolve_target_modules,
)
from neural_surrogates.training.base import checkpoint_submodules
from torch.utils.data import DataLoader

GRID = (16, 32, 16)
STAGES = [
    "dft.encoder.conv_encoder",
    "dft.encoder.transformer_encoder",
    "dft.decoder.transformer_decoder",
    "dft.decoder.conv_decoder",
]
BLOCKS = [
    "dft.encoder.conv_encoder.conv_encoder.blocks.*",
    "dft.decoder.conv_decoder.conv_decoder.blocks.*",
    "dft.encoder.transformer_encoder.transformer.encoder_levels.*",
]


def _peft_stepper(fused: bool = False) -> Any:
    """A tiny DFT: random stepper wired like ``train_dft.yaml``, LoRA with
    non-zero adapters on the frozen encoder/decoder, trainable new modules."""
    torch.manual_seed(0)
    model = TadpoleTimeStepper(
        n_state_channels=3,
        n_params=2,
        size="S",
        pretrained_ae_dir=None,
        skip_pretrained_load=True,
        spatial_mode="global",
        encoder_crop_size=16,
        encode_geometry=False,
        geometry_branch={"width": 8},
        skip_mixing={"width": 8, "levels": [4, 8]},
        sdf_clamp_cells=8,
        latent_type="mode",
    )
    model.requires_grad_(False)
    targets = resolve_target_modules(model, preset="tadpole_encdec")
    peft = inject_lora(
        model, rank=4, alpha=8, target_modules=targets, fused_forward=fused
    )
    for name, p in peft.named_parameters():
        if "lora_B" in name or "geom_proj" in name or "skip_mixing" in name:
            p.data.normal_(std=0.02)
        if any(k in name for k in (".subnetwork.", ".skip_mixing.", ".scales")):
            p.requires_grad_(True)
    return peft


def _inputs() -> tuple:
    g = torch.Generator().manual_seed(1)
    state = torch.randn(1, 3, *GRID, generator=g)
    params = torch.randn(1, 2, generator=g)
    geom = (torch.rand(1, *GRID, generator=g) > 0.2).float()
    return state, params, geom


def _trainer(model: Any, **kw: Any) -> Any:
    loader = DataLoader([0])
    return Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
        **kw,
    )


def _loss_and_grads(model: Any, amp: bool) -> tuple:
    """Two chained grad-carrying steps, as in the 2-step pushforward unroll."""
    model.zero_grad()
    state, params, geom = _inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        loss = model(model(state, params, geom), params, geom).float().square().mean()
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    return loss.detach(), grads


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("patterns", [STAGES, BLOCKS, ["dft.encoder", "dft.decoder"]])
def test_activation_checkpointing_keeps_loss_and_grads(
    patterns: list[str], amp: bool, fused: bool
) -> None:
    plain = _peft_stepper(fused)
    ckpt = copy.deepcopy(plain)
    _trainer(ckpt, activation_checkpoint_modules=patterns)
    wrapped = [
        n for n, m in ckpt.named_modules() if "_checkpoint_inner_forward" in vars(m)
    ]
    assert wrapped and all(n.startswith("base_model.model.dft.") for n in wrapped)
    assert list(ckpt.state_dict()) == list(plain.state_dict())

    loss0, grads0 = _loss_and_grads(plain, amp)
    loss1, grads1 = _loss_and_grads(ckpt, amp)
    assert any("lora_" in n and "encoder" in n for n in grads0)
    torch.testing.assert_close(loss1, loss0, atol=0, rtol=0)
    for n, g in grads0.items():
        if g is None:  # an adapter on a layer the forward never calls
            assert grads1[n] is None, n
            continue
        torch.testing.assert_close(grads1[n], g, atol=0, rtol=0)

    # Under no_grad (the pushforward prefix, validation) nothing is recomputed.
    with torch.no_grad():
        torch.testing.assert_close(ckpt(*_inputs()), plain(*_inputs()))
    # The merged weights a LoRA run saves are unaffected.
    merged, reference = merge_to_state_dict(ckpt), merge_to_state_dict(plain)
    assert list(merged) == list(reference)


def test_activation_checkpointing_wraps_the_fused_lora_forward() -> None:
    """Checkpointing a LoRA layer itself keeps ``lora.fused_forward``."""
    from neural_surrogates.finetuning.inject import _fused_lora_forward

    plain = _peft_stepper(fused=True)
    ckpt = copy.deepcopy(plain)
    leaves = [
        "dft.decoder.conv_decoder.conv_decoder.decompress",
        "dft.encoder.*.conv_1",
    ]
    for module in checkpoint_submodules(ckpt, leaves):
        assert module._checkpoint_inner_forward.__func__ is _fused_lora_forward
    loss0, grads0 = _loss_and_grads(plain, amp=True)
    loss1, grads1 = _loss_and_grads(ckpt, amp=True)
    torch.testing.assert_close(loss1, loss0, atol=0, rtol=0)
    for n, g in grads0.items():
        if g is not None:
            torch.testing.assert_close(grads1[n], g, atol=0, rtol=0)


def test_activation_checkpointing_off_wraps_nothing() -> None:
    model = _peft_stepper()
    _trainer(model)
    assert not [
        n for n, m in model.named_modules() if "_checkpoint_inner_forward" in vars(m)
    ]


def test_activation_checkpointing_skips_nested_and_must_match() -> None:
    model = _peft_stepper()
    wrapped = checkpoint_submodules(model, ["dft.encoder", "dft.encoder.*"])
    assert wrapped == [model.base_model.model.dft.encoder]
    with pytest.raises(ValueError, match="match no module"):
        checkpoint_submodules(model, ["dft.no_such_module.*"])


def test_lora_shares_its_input_with_the_base_layer() -> None:
    """No fp32 round trip of the LoRA input: one saved bf16 input per layer
    instead of two, and the same output as PEFT's default cast."""
    model = _peft_stepper()
    layers = [m for m in model.modules() if hasattr(m, "cast_input_dtype_enabled")]
    assert layers and not any(m.cast_input_dtype_enabled for m in layers)
    loss0, grads0 = _loss_and_grads(model, amp=True)
    for m in layers:
        m.cast_input_dtype_enabled = True
    loss1, grads1 = _loss_and_grads(model, amp=True)
    torch.testing.assert_close(loss1, loss0, atol=0, rtol=0)
    for n, g in grads0.items():
        if g is not None:
            torch.testing.assert_close(grads1[n], g, atol=0, rtol=0)


def _batch(k: int) -> dict:
    state, params, geom = _inputs()
    return {
        "state_n": state,
        "state_next": torch.roll(state, 1, dims=-1),
        "params_n": params[:, None].expand(1, k, 2).contiguous(),
        "geometry": geom,
    }


def _trainer_grads(model: Any, k: int, **kw: Any) -> dict:
    trainer = _trainer(model, grad_unroll_steps=2, **kw)
    model.zero_grad()
    trainer._forward(_batch(k)).backward()
    return {n: p.grad for n, p in model.named_parameters() if p.requires_grad}


def test_unroll_no_grad_modules_drops_only_their_early_terms() -> None:
    """The approximate-gradient knob: with two gradient-carrying steps, the
    encoder of the first one runs under no_grad. Every other gradient is
    unchanged; the encoder's adapters keep only the last step's term. With a
    single gradient step (horizon 1) it changes nothing."""
    plain = _peft_stepper()
    full = _trainer_grads(plain, k=2)
    approx = _trainer_grads(
        copy.deepcopy(plain), k=2, unroll_no_grad_modules=["dft.encoder"]
    )
    encoder = [n for n in full if ".dft.encoder." in n and full[n] is not None]
    assert encoder
    for n, g in full.items():
        if g is None:
            assert approx[n] is None, n
        elif n in encoder:
            assert approx[n] is not None and not torch.equal(approx[n], g), n
        else:
            torch.testing.assert_close(approx[n], g, atol=0, rtol=0)

    one = _trainer_grads(plain, k=1)
    one_approx = _trainer_grads(
        copy.deepcopy(plain), k=1, unroll_no_grad_modules=["dft.encoder"]
    )
    for n, g in one.items():
        if g is not None:
            torch.testing.assert_close(one_approx[n], g, atol=0, rtol=0)
