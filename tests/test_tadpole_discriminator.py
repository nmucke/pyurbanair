"""Tadpole patch discriminator (optional GAN loss for AE pre-training): unit tests.

Pure model-side coverage of
:mod:`neural_surrogates.architectures.tadpole_discriminator`: input-channel
arithmetic for every ``encode_geometry`` x ``sdf_features`` combination, the
patch-logit shape + backprop, the pad-to-16 path, the validation errors, the two
hinge losses, the adaptive weight (including its fallbacks) and the
``last_layer`` resolution against a real ``TadpoleAE``.

Everything runs on tiny CPU grids (16-24 cells per axis, size ``"S"``): the
backbone downsamples by 16, so a 32^3 field already yields a 2^3 patch map --
enough to exercise every path without paying for a real training shape.

Gated with ``importorskip('diffusers')`` / ``importorskip('timm')`` -- the
vendored backbone's runtime deps -- so envs without them skip cleanly.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from neural_surrogates import TadpoleAE, TadpoleDiscriminator
from neural_surrogates.architectures.tadpole_discriminator import (
    adaptive_adv_weight,
    hinge_d_loss,
    hinge_g_loss,
    resolve_last_decoder_layer,
)

CROP = 16  # encoder_crop_size / backbone downsampling factor

N_STATE = 3


def _disc(
    encode_geometry: bool = True,
    sdf_features: bool | str = "none",
    **kw: Any,
) -> TadpoleDiscriminator:
    kw.setdefault("size", "S")
    return TadpoleDiscriminator(
        n_state_channels=N_STATE,
        encode_geometry=encode_geometry,
        sdf_features=sdf_features,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Channel arithmetic + construction validation.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "encode_geometry, sdf_features, expected",
    [
        (False, "none", N_STATE),  # state only
        (True, "none", N_STATE + 1),  # + mask
        (True, "sdf", N_STATE + 2),  # + mask + clamped sdf
        (True, "grad", N_STATE + 4),  # + mask + 3 gradient components
        (True, "both", N_STATE + 5),  # + mask + all 4
    ],
)
def test_input_channel_arithmetic(
    encode_geometry: bool, sdf_features: str, expected: int
) -> None:
    disc = _disc(encode_geometry=encode_geometry, sdf_features=sdf_features)
    assert disc.n_input_channels == expected
    # The critic must see the field whole -- no channel folding into the batch,
    # so the vendored backbone's conv stem takes all the channels at once.
    backbone = cast(Any, disc.disc).model[0]
    assert backbone.conv_encoder.in_channels == expected


def test_bad_size_raises() -> None:
    with pytest.raises(ValueError, match="size must be one of"):
        _disc(size="tiny")


def test_sdf_features_require_encode_geometry() -> None:
    with pytest.raises(ValueError, match="requires encode_geometry=True"):
        _disc(encode_geometry=False, sdf_features="both")


# --------------------------------------------------------------------------- #
# Forward pass.
# --------------------------------------------------------------------------- #


def test_forward_patch_logits_and_backprop() -> None:
    disc = _disc()
    x = torch.randn(2, disc.n_input_channels, 32, 32, 32)
    logits = disc(x)

    # Patch map, not a scalar: one logit per 16^3 receptive field.
    assert logits.shape == (2, 1, 2, 2, 2)
    assert torch.isfinite(logits).all()

    logits.mean().backward()
    grads = [p.grad for p in disc.parameters() if p.grad is not None]
    assert grads, "no discriminator parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads)


def test_forward_pads_non_multiple_of_16_grid() -> None:
    disc = _disc(sdf_features="sdf")
    x = torch.randn(1, disc.n_input_channels, 24, 16, 20)
    logits = disc(x)
    # 24 -> 32, 16 -> 16, 20 -> 32 after the internal zero-padding.
    assert logits.shape == (1, 1, 2, 1, 2)


def test_forward_wrong_channel_count_raises() -> None:
    disc = _disc()
    x = torch.randn(1, disc.n_input_channels + 1, 16, 16, 16)
    with pytest.raises(ValueError, match="expected 4 input channels"):
        disc(x)


def test_forward_wrong_rank_raises() -> None:
    disc = _disc(encode_geometry=False)
    with pytest.raises(ValueError, match="5-D"):
        disc(torch.randn(1, N_STATE, 16, 16))


# --------------------------------------------------------------------------- #
# Loss helpers.
# --------------------------------------------------------------------------- #


def test_hinge_d_loss_zero_beyond_the_margin() -> None:
    real = torch.full((2, 1, 2, 2, 2), 3.0)
    fake = torch.full((2, 1, 2, 2, 2), -3.0)
    assert hinge_d_loss(real, fake).item() == pytest.approx(0.0)

    # Perfectly uninformative critic (all logits 0): 0.5 * (1 + 1) = 1.
    zeros = torch.zeros(2, 1, 2, 2, 2)
    assert hinge_d_loss(zeros, zeros).item() == pytest.approx(1.0)

    # A critic fooled by the fakes pays on both sides.
    assert hinge_d_loss(-real, -fake).item() == pytest.approx(4.0)


def test_hinge_g_loss_decreases_as_fake_logits_rise() -> None:
    low = hinge_g_loss(torch.full((1, 1, 2, 2, 2), -1.0))
    high = hinge_g_loss(torch.full((1, 1, 2, 2, 2), 2.0))
    assert low.item() == pytest.approx(1.0)
    assert high.item() == pytest.approx(-2.0)
    assert high < low  # the generator always gains from higher fake logits


def test_adaptive_adv_weight_is_detached_finite_and_clamped() -> None:
    last_layer = torch.nn.Parameter(torch.randn(4, 2, 1, 1, 1))
    x = torch.randn(2, 2, 4, 4, 4)
    out = torch.nn.functional.conv3d(x, last_layer)

    recon_loss = (out**2).mean()
    adv_loss = -out.mean()
    w = adaptive_adv_weight(recon_loss, adv_loss, last_layer)

    assert w.shape == ()
    assert not w.requires_grad
    assert torch.isfinite(w)
    # Default ceiling is 1.0: the trainer scales this by its own `adv_weight`
    # (default 1e-4), which is the paper's maximum adversarial scale.
    assert 0.0 <= w.item() <= 1.0


def test_adaptive_adv_weight_respects_max_weight() -> None:
    last_layer = torch.nn.Parameter(torch.randn(4, 2, 1, 1, 1))
    x = torch.randn(2, 2, 4, 4, 4)
    out = torch.nn.functional.conv3d(x, last_layer)
    # A vanishing adversarial gradient would blow the ratio up; the clamp caps it
    # -- at the tunable ceiling, and at the default 1.0.
    big, tiny = (out**2).mean() * 1e6, -out.mean() * 1e-12
    assert adaptive_adv_weight(big, tiny, last_layer, max_weight=10.0).item() == (
        pytest.approx(10.0)
    )
    assert adaptive_adv_weight(big, tiny, last_layer).item() == pytest.approx(1.0)


def test_adaptive_adv_weight_falls_back_to_one_when_disconnected() -> None:
    last_layer = torch.nn.Parameter(torch.randn(4, 2, 1, 1, 1))
    other = torch.nn.Parameter(torch.randn(4, 2, 1, 1, 1))
    x = torch.randn(2, 2, 4, 4, 4)

    recon_loss = (torch.nn.functional.conv3d(x, last_layer) ** 2).mean()
    # Not a function of `last_layer` at all -> autograd returns None for it.
    adv_loss = -torch.nn.functional.conv3d(x, other).mean()

    w = adaptive_adv_weight(recon_loss, adv_loss, last_layer)
    assert w.item() == pytest.approx(1.0)
    assert not w.requires_grad

    # A fully detached loss (no graph) is the same neutral fallback, not a crash.
    w = adaptive_adv_weight(recon_loss, adv_loss.detach(), last_layer)
    assert w.item() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Generator last-layer resolution.
# --------------------------------------------------------------------------- #


def test_resolve_last_decoder_layer_on_tadpole_ae() -> None:
    ae = TadpoleAE(n_state_channels=N_STATE, size="S", encoder_crop_size=CROP)
    last_layer = resolve_last_decoder_layer(ae)

    assert isinstance(last_layer, torch.nn.Parameter)
    assert last_layer is ae.ae.decoder.conv_decoder.decompress.weight


def test_resolve_last_decoder_layer_returns_none_without_a_decoder() -> None:
    assert resolve_last_decoder_layer(torch.nn.Conv3d(1, 1, 1)) is None
