"""Regional ``torch.compile`` (the trainer's ``compile_modules`` knob).

On a tiny LoRA-wrapped ``TadpoleTimeStepper`` (the DFT setup), compiling only
the conv residual blocks of the frozen encoder/decoder must leave the forward
output, the gradients of every trainable parameter, the state-dict keys and the
merged LoRA weights as they are in eager mode, on two grid shapes. Without the
knob nothing is compiled. Runs inductor on CPU (needs a C++ compiler).
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
from neural_surrogates.training.base import compile_submodules
from torch.utils.data import DataLoader

BLOCKS = [
    "dft.encoder.conv_encoder.conv_encoder.blocks.*",
    "dft.decoder.conv_decoder.conv_decoder.blocks.*",
]
GRIDS = [(16, 16, 16), (32, 16, 16)]


def _peft_stepper() -> Any:
    """A tiny DFT: random stepper, LoRA on the encoder/decoder with non-zero
    adapters, and the subnetwork trained fully (as in ``_dft``)."""
    torch.manual_seed(0)
    model = TadpoleTimeStepper(
        n_state_channels=3,
        n_params=2,
        size="S",
        pretrained_ae_dir=None,
        skip_pretrained_load=True,
        encoder_crop_size=16,
        sdf_clamp_cells=8,
        latent_type="mode",
    )
    model.requires_grad_(False)
    targets = resolve_target_modules(model, preset="tadpole_encdec")
    peft = inject_lora(model, rank=4, alpha=8, target_modules=targets)
    for name, p in peft.named_parameters():
        if "lora_B" in name:
            p.data.normal_(std=0.02)
        if ".subnetwork." in name:
            p.requires_grad_(True)
    return peft


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


def _inputs(grid: tuple) -> tuple:
    g = torch.Generator().manual_seed(1)
    state = torch.randn(1, 3, *grid, generator=g)
    params = torch.randn(1, 2, generator=g)
    geom = (torch.rand(1, *grid, generator=g) > 0.2).float()
    return state, params, geom


def _compiled(model: Any) -> list[str]:
    return [n for n, m in model.named_modules() if m._compiled_call_impl is not None]


def test_compile_modules_matches_eager_with_lora() -> None:
    eager = _peft_stepper()
    compiled = copy.deepcopy(eager)
    _trainer(compiled, compile_modules=BLOCKS)

    # Only the blocks, under the PEFT prefix; children of a block are covered
    # by its graph and not compiled again.
    names = _compiled(compiled)
    assert names and all(n.startswith("base_model.model.dft.") for n in names)
    assert all(".blocks." in n and n.split(".blocks.")[1].isdigit() for n in names)
    assert list(compiled.state_dict()) == list(eager.state_dict())

    trainable = [n for n, p in eager.named_parameters() if p.requires_grad]
    assert any("lora_" in n for n in trainable)
    assert any(".subnetwork." in n for n in trainable)
    for grid in GRIDS:
        with torch.no_grad():
            torch.testing.assert_close(
                compiled(*_inputs(grid)), eager(*_inputs(grid)), atol=1e-4, rtol=1e-4
            )
        grads = []
        for model in (eager, compiled):
            model.zero_grad()
            model(*_inputs(grid)).square().mean().backward()
            grads.append(dict(model.named_parameters()))
        for n in trainable:
            torch.testing.assert_close(
                grads[1][n].grad, grads[0][n].grad, atol=1e-4, rtol=1e-3
            )

    # The merged weights a LoRA run saves are unaffected.
    merged, reference = merge_to_state_dict(compiled), merge_to_state_dict(eager)
    assert list(merged) == list(reference)
    for k in reference:
        torch.testing.assert_close(merged[k], reference[k])


def test_compile_modules_off_compiles_nothing() -> None:
    model = _peft_stepper()
    trainer = _trainer(model)
    assert trainer.model is model
    assert _compiled(model) == []


def test_compile_modules_pattern_must_match() -> None:
    with pytest.raises(ValueError, match="match no module"):
        compile_submodules(_peft_stepper(), ["dft.no_such_module.*"])


def test_pixel_shuffle_compiles_without_graph_break() -> None:
    """The vendored ``PixelShuffle3d`` builds its permutation without a tensor
    round trip, so it traces as one graph (``fullgraph``) and the axes are
    upstream's ``arange(ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]``."""
    from neural_surrogates.architectures._tadpole.architecture.p3d.modules import (
        PixelShuffle3d,
    )

    x = torch.randn(2, 16, 3, 4, 5)
    shuffle = PixelShuffle3d(2)
    view = x.view(2, 2, 2, 2, 2, 3, 4, 5)
    upstream = torch.arange(view.ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]
    expected = view.permute(upstream).reshape(2, 2, 6, 8, 10)
    torch.testing.assert_close(shuffle(x), expected, rtol=0, atol=0)
    compiled = torch.compile(shuffle, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)


def test_channels_last_converts_only_5d_tensors() -> None:
    """``channels_last=True`` on a model that also holds 4-D tensors (the DFT
    does): ``Module.to(memory_format=channels_last_3d)`` raises on those, the
    trainer converts only the 5-D ones, with unchanged values and outputs."""
    model = _peft_stepper()
    model.base_model.model.register_buffer("table4d", torch.randn(2, 3, 4, 5))
    with pytest.raises(RuntimeError, match="rank 5"):
        copy.deepcopy(model).to(memory_format=torch.channels_last_3d)
    reference = copy.deepcopy(model)
    trainer = _trainer(model, channels_last=True)
    conv = model.base_model.model.dft.encoder.conv_encoder.conv_encoder
    weight = conv.blocks[0].conv_1.base_layer.weight
    assert weight.is_contiguous(memory_format=torch.channels_last_3d)
    assert not weight.is_contiguous()
    assert trainer.model is model
    for a, b in zip(model.state_dict().values(), reference.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(
            model(*_inputs(GRIDS[0])),
            reference(*_inputs(GRIDS[0])),
            atol=1e-5,
            rtol=1e-5,
        )
