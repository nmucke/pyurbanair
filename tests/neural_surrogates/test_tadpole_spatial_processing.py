"""Real-model spatial policy, context, gradients and checkpoint contracts."""

# The isolated pre-commit mypy environment does not include pytest typings.
# mypy: disallow-untyped-decorators=false

from collections.abc import Iterator
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from neural_surrogates.architectures._tadpole.architecture.p3d.skip_wrapper import (
    KLP3DEncoderSkip,
    P3DDecoderSkip,
)
from neural_surrogates.architectures.tadpole_ae import TadpoleAE
from neural_surrogates.architectures.tadpole_stepper import TadpoleTimeStepper


@pytest.fixture(autouse=True)
def _cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def _ae(mode: str = "local", **kwargs: Any) -> TadpoleAE:
    model: TadpoleAE = TadpoleAE(
        n_state_channels=2,
        encoder_crop_size=16,
        spatial_mode=mode,
        latent_type="mode",
        **kwargs,
    ).eval()
    # Simulate a pretrained decoder so latent/context tests cannot pass vacuously.
    torch.nn.init.normal_(
        model.ae.decoder.transformer_decoder.final_layer.out_proj.weight, std=0.01
    )
    return model


def _stepper(mode: str, **kwargs: Any) -> TadpoleTimeStepper:
    model: TadpoleTimeStepper = TadpoleTimeStepper(
        n_state_channels=2,
        n_params=0,
        encoder_crop_size=16,
        spatial_mode=mode,
        subnetwork_cfg={"n_layers": 1, "hidden_size": 32, "num_heads": 2},
        **kwargs,
    ).eval()
    torch.nn.init.normal_(
        model.dft.decoder.transformer_decoder.transformer.final_layer.out_proj.weight,
        std=0.01,
    )
    return model


@pytest.mark.parametrize("mode", ["global", "halo"])
@pytest.mark.parametrize("branch", [False, True])
def test_ae_dft_shapes_init_parity_chunking_and_rebuild(
    mode: str, branch: bool
) -> None:
    options: dict[str, Any] = {"encode_geometry": not branch}
    if branch:
        options["geometry_branch"] = {"width": 4}
    ae = _ae(mode, **options)
    # Make branch projections active; zero-init would hide spatial misalignment.
    if branch:
        with torch.no_grad():
            for name, p in ae.named_parameters():
                if "geom_proj" in name or "geom_latent_proj" in name:
                    p.fill_(0.01)
    model = _stepper(mode, **options)
    model.dft.encoder = KLP3DEncoderSkip(ae.ae.encoder)
    model.dft.decoder = P3DDecoderSkip(ae.ae.decoder)
    if branch:
        assert model.geometry_branch is not None
        assert ae.geometry_branch is not None
        model.geometry_branch.load_state_dict(ae.geometry_branch.state_dict())
    state = torch.randn(1, 2, 33, 15, 17)
    geom = torch.ones(1, 33, 15, 17)
    geom[:, 10:13, 3:6, 2:5] = 0
    calls = []
    hook = model.dft.subnetwork.register_forward_pre_hook(
        lambda _, args: calls.append(args[0].shape)
    )
    with torch.no_grad():
        reference = ae(state, geom)
        result = model(state, None, geom)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        assert result.shape == state.shape
        assert torch.count_nonzero(result * (1 - geom.unsqueeze(1))) == 0
        assert len(calls) == 1
        assert tuple(calls[0][-3:]) == (3, 1, 2)
        torch.testing.assert_close(
            result, model._ae_reference_recon(state, geom), rtol=0, atol=0
        )
        model.dft.max_internal_batchsize = 1
        ae.ae.max_internal_batchsize = 1
        torch.testing.assert_close(
            model(state, None, geom), result, atol=5e-5, rtol=2e-4
        )
        torch.testing.assert_close(ae(state, geom), reference, atol=5e-5, rtol=2e-4)
        latent, skips = model.encode(state, geom)
        features = model._geom_branch_kwargs(state, geom, None).get("geom_feats")
        decoded = model.decode(latent, skips, features).reshape(1, -1, 48, 16, 32)
        torch.testing.assert_close(
            decoded[:, :2, :33, :15, :17] * geom.unsqueeze(1),
            result,
            atol=5e-5,
            rtol=2e-4,
        )
        rebuilt = _stepper(mode, **options)
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(
            rebuilt(state, None, geom), result, atol=5e-5, rtol=2e-4
        )
    hook.remove()


def test_global_is_one_whole_rectangular_field_per_channel() -> None:
    model = _ae("global", encode_geometry=True)
    state = torch.randn(2, 2, 17, 31, 15)
    geom = torch.ones(2, 17, 31, 15)
    with torch.no_grad():
        working = model._assemble_working_input(state, geom, None)
        padded, _ = model._pad_to_crop_multiple(working)
        assert padded.shape[-3:] == (32, 32, 16)
        folded = padded.flatten(0, 1).unsqueeze(1)
        latent = model.ae.encoder(folded, "mode")
        expected = model.ae.decoder(latent.contiguous()).reshape_as(padded)
        result, target = model(state, geom, working_space=True)
        torch.testing.assert_close(result, expected[..., :17, :31, :15], rtol=0, atol=0)
        torch.testing.assert_close(target, working)
        torch.testing.assert_close(model.encode(state, geom), latent, rtol=0, atol=0)
        decoded = model.decode(model.encode(state, geom))
        torch.testing.assert_close(decoded.reshape_as(padded), expected, rtol=0, atol=0)


def test_zero_halo_and_default_preserve_local_model_and_output() -> None:
    torch.manual_seed(8)
    default = TadpoleAE(2, encoder_crop_size=16, latent_type="mode")
    local = _ae("local")
    halo = _ae("halo", halo_size=0)
    local.load_state_dict(default.state_dict(), strict=True)
    halo.load_state_dict(default.state_dict(), strict=True)
    state = torch.randn(1, 2, 32, 16, 16)
    geom = torch.ones(1, 32, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(
            default(state, geom), local(state, geom), rtol=0, atol=0
        )
        torch.testing.assert_close(
            local(state, geom), halo(state, geom), atol=5e-5, rtol=2e-4
        )


def test_halo_has_encoder_and_decoder_context_across_tile_seams() -> None:
    local = _ae("local", encode_geometry=False)
    halo = _ae("halo", encode_geometry=False)
    halo.load_state_dict(local.state_dict())
    state = torch.randn(1, 2, 48, 16, 16)
    changed = state.clone()
    changed[:, :, 17:22] += 3
    geom = torch.ones(1, 48, 16, 16)
    with torch.no_grad():
        # The first tile is unchanged, but the second tile supplies halo context.
        zl = local.encode(state, geom)
        zlc = local.encode(changed, geom)
        torch.testing.assert_close(zl[0], zlc[0], rtol=0, atol=0)
        zh = halo.encode(state, geom)
        zhc = halo.encode(changed, geom)
        assert (zh[..., 0, :, :] - zhc[..., 0, :, :]).abs().max() > 1e-5
        changed_latent = zh.clone()
        changed_latent[..., 1, :, :] += 2
        decoded = halo.decode(zh)
        decoded_changed = halo.decode(changed_latent)
        assert (
            decoded[..., :16, :, :] - decoded_changed[..., :16, :, :]
        ).abs().max() > 1e-5


@pytest.mark.parametrize("mode", ["global", "halo"])
def test_ae_and_dft_gradients_and_inner_sampling_override(mode: str) -> None:
    ae = _ae(mode, encode_geometry=False)
    state = torch.randn(1, 2, 32, 16, 16)
    geom = torch.ones(1, 32, 16, 16)
    ae.latent_type = "sample"
    # Trainer validation overrides the inner module, not wrapper latent_type.
    ae.ae.latent_type = "mode"
    recon, kl = ae(state, geom, return_kl_element=True)
    (recon.square().mean() + 1e-3 * kl.mean()).backward()
    assert ae.ae.encoder.to_latent.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in ae.ae.decoder.parameters()
    )
    with torch.no_grad():
        torch.testing.assert_close(recon, ae(state, geom), rtol=0, atol=0)
    model = _stepper(mode, encode_geometry=False)
    result = model(state, None, geom)
    result.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.dft.subnetwork.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "scales" in n
    )


@pytest.mark.parametrize(
    "mode,halo", [("invalid", 16), ("halo", -16), ("halo", 3), ("halo", 16.5)]
)
def test_invalid_spatial_policy_fails(mode: str, halo: Any) -> None:
    with pytest.raises(ValueError, match="spatial_mode|halo_size"):
        _ae(mode, halo_size=halo)


@pytest.mark.parametrize("mode", ["global", "halo"])
@pytest.mark.parametrize("batch_tiles", [False, True])
def test_spatial_checkpoint_keeps_outputs_and_gradients(
    mode: str, batch_tiles: bool
) -> None:
    options: dict[str, Any] = {
        "encode_geometry": False,
        "geometry_branch": {"width": 4},
        "skip_mixing": {"width": 4},
        "spatial_batch_tiles": batch_tiles,
    }
    plain = _stepper(mode, **options)
    with torch.no_grad():  # make every path carry signal (zero-init hides bugs)
        for name, p in plain.named_parameters():
            if "scales" in name or "proj" in name or "skip_mixing" in name:
                p.fill_(0.05)
    plain.requires_grad_(True)  # encoder/decoder too, standing in for LoRA
    ckpt = _stepper(mode, spatial_checkpoint=True, **options)
    ckpt.load_state_dict(plain.state_dict(), strict=True)
    ckpt.requires_grad_(True)
    state = torch.randn(2, 2, 32, 16, 48)
    geom = torch.ones(2, 32, 16, 48)
    geom[:, 4:9, 3:7, 20:30] = 0
    calls = [0, 0]
    for i, m in enumerate((plain, ckpt)):
        for part in (m.dft.encoder, m.dft.decoder):
            part.register_forward_pre_hook(
                lambda *_, i=i: calls.__setitem__(i, calls[i] + 1)
            )
    out = [m(state, None, geom) for m in (plain, ckpt)]
    torch.testing.assert_close(out[1], out[0], rtol=0, atol=0)
    for o in out:
        o.square().mean().backward()
    assert calls[1] == 2 * calls[0] > 0  # every tile call was recomputed once
    grads = [dict(m.named_parameters()) for m in (plain, ckpt)]
    checked = 0
    for name, p in grads[0].items():
        q = grads[1][name]
        if p.grad is None:
            assert q.grad is None, name
            continue
        torch.testing.assert_close(q.grad, p.grad, rtol=1e-5, atol=1e-7, msg=name)
        checked += p.grad.abs().sum() > 0
    assert checked > 50
    with torch.no_grad():  # deployment: identical, no recompute bookkeeping
        torch.testing.assert_close(
            ckpt(state, None, geom), plain(state, None, geom), rtol=0, atol=0
        )


@pytest.mark.parametrize("mode", ["local", "halo"])
@pytest.mark.parametrize("limit", [None, 1, 4])
def test_batched_tiles_match_one_call_per_tile(mode: str, limit: Any) -> None:
    options: dict[str, Any] = {
        "encode_geometry": False,
        "geometry_branch": {"width": 4},
        "skip_mixing": {"width": 4},
    }
    # float64, and norm parameters skipped: the attention LayerNorm biases see
    # zero-padded window tokens, so their gradients are ~0 by cancellation and
    # differ by O(1) relative between any two batch layouts (also between
    # max_internal_batchsize chunkings of the unbatched path). DFT training never
    # trains them (LoRA targets linear/conv layers).
    plain = _stepper(mode, **options).double()
    with torch.no_grad():
        for name, p in plain.named_parameters():
            if "scales" in name or "proj" in name or "skip_mixing" in name:
                p.fill_(0.05)
    plain.requires_grad_(True)
    batched = _stepper(mode, spatial_batch_tiles=True, **options).double()
    batched.load_state_dict(plain.state_dict(), strict=True)
    batched.requires_grad_(True)
    for m in (plain, batched):
        m.dft.max_internal_batchsize = limit
    calls = [0, 0]
    for i, m in enumerate((plain, batched)):
        m.dft.encoder.register_forward_pre_hook(
            lambda *_, i=i: calls.__setitem__(i, calls[i] + 1)
        )
    state = torch.randn(1, 2, 32, 16, 64, dtype=torch.float64)
    geom = torch.ones(1, 32, 16, 64, dtype=torch.float64)
    geom[:, 4:9, 3:7, 20:30] = 0
    out = [m(state, None, geom) for m in (plain, batched)]
    torch.testing.assert_close(out[1], out[0], rtol=1e-10, atol=1e-12)
    assert calls[1] < calls[0]
    for o in out:
        o.square().mean().backward()
    named = dict(batched.named_parameters())
    grads = {n: p.grad for n, p in plain.named_parameters() if p.grad is not None}
    # Biases right before a norm have ~0 gradients too: compare on a global scale.
    scale = max(g.abs().max().item() for n, g in grads.items() if "norm" not in n)
    for name, g in grads.items():
        if "norm" not in name:
            torch.testing.assert_close(
                named[name].grad, g, rtol=1e-7, atol=1e-9 * scale, msg=name
            )
