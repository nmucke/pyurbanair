"""make_periodic_y: circular y padding + y-periodic SDF at inference."""

# The isolated pre-commit mypy environment does not include pytest typings.
# mypy: disallow-untyped-decorators=false

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from neural_surrogates import (
    P3D,
    SimpleConv,
    UNetConvNeXt,
    make_periodic_y,
    sdf_features,
)
from torch import nn


def _rel_roll_error(f: Any, x: torch.Tensor, shift: int, dim: int = -2) -> float:
    """Relative error of f(roll(x)) against roll(f(x))."""
    with torch.no_grad():
        a = f(x).roll(shift, dim)
        b = f(x.roll(shift, dim))
    return float((a - b).norm() / a.norm())


def _conv_stack() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv3d(2, 4, 3, padding=1),
        nn.GELU(),
        nn.Conv3d(4, 4, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv3d(4, 4, (1, 5, 3), padding=(0, 2, 1)),
        nn.ConvTranspose3d(4, 2, 2, stride=2),
    ).eval()


def _mask(nz: int = 8, ny: int = 32, nx: int = 16) -> torch.Tensor:
    """Fluid mask with solid ground and a building at the y = 0 edge."""
    m = torch.ones(nz, ny, nx)
    m[0] = 0
    m[:4, :3, 5:9] = 0
    m[:6, 12:18, 2:6] = 0
    return m


def test_conv_stack_becomes_y_equivariant_only() -> None:
    model = _conv_stack()
    x = torch.randn(1, 2, 8, 32, 16)
    assert _rel_roll_error(model, x, 4) > 1e-2
    make_periodic_y(model)
    assert _rel_roll_error(model, x, 4) < 1e-6
    # z and x keep their zero padding.
    assert _rel_roll_error(model, x, 4, dim=-1) > 1e-2
    assert _rel_roll_error(model, x, 2, dim=-3) > 1e-2
    assert model[0].padding == (1, 0, 1)
    assert model[4].padding == (0, 0, 1)


def test_idempotent() -> None:
    model = _conv_stack()
    x = torch.randn(1, 2, 8, 32, 16)
    assert make_periodic_y(model) is model
    with torch.no_grad():
        once = model(x)
    hooks = [len(m._forward_pre_hooks) for m in model.modules()]
    make_periodic_y(model)
    assert [len(m._forward_pre_hooks) for m in model.modules()] == hooks
    with torch.no_grad():
        assert torch.equal(model(x), once)


@pytest.mark.parametrize(
    "layer",
    [
        nn.Conv3d(2, 2, 3, padding=1, padding_mode="reflect"),
        nn.Conv2d(2, 2, 3, padding=1),
        nn.ConvTranspose3d(2, 2, 3, padding=1),
    ],
)
def test_raises_on_unsupported_convs_and_leaves_model_untouched(
    layer: nn.Module,
) -> None:
    ok = nn.Conv3d(2, 2, 3, padding=1)
    model = nn.Sequential(ok, layer)
    with pytest.raises(ValueError, match="make_periodic_y"):
        make_periodic_y(model)
    assert ok.padding == (1, 1, 1) and not ok._forward_pre_hooks
    assert not hasattr(model, "periodic_y")


def test_own_periodic_y_switch() -> None:
    """A model with its own `periodic_y` (the baselines) is used as built."""
    on = nn.Conv3d(2, 2, 3, padding=1)
    on.periodic_y = True
    assert make_periodic_y(on) is on and on.padding == (1, 1, 1)
    off = nn.Conv3d(2, 2, 3, padding=1)
    off.periodic_y = False
    with pytest.raises(ValueError, match="non-periodic"):
        make_periodic_y(off)


def test_periodic_sdf_wraps_and_matches_interior() -> None:
    mask = _mask()
    plain = sdf_features(mask, clamp_cells=4.0)
    per = sdf_features(mask, clamp_cells=4.0, periodic_y=True)
    assert per.shape == plain.shape
    # Far from the y edges (more than the ground height) nothing changes ...
    torch.testing.assert_close(per[:, :, 8:-8], plain[:, :, 8:-8])
    # ... but next to the high-y edge the building across the seam is seen.
    assert not torch.allclose(per[:, :, -3:], plain[:, :, -3:])
    # Periodic features are y-shift-equivariant; the plain ones are not.
    for s in (5, 16):
        rolled = sdf_features(mask.roll(s, 1), clamp_cells=4.0, periodic_y=True)
        torch.testing.assert_close(rolled, per.roll(s, -2))
    rolled = sdf_features(mask.roll(5, 1), clamp_cells=4.0)
    assert not torch.allclose(rolled, plain.roll(5, -2))


def test_periodic_sdf_exact_without_ground() -> None:
    """No ground row: cells high up are far from every solid, so the wrap must
    reach the nearest periodic image (half the y extent), not just the clamp."""
    mask = torch.ones(40, 96, 8)
    mask[:10, 40:44, 2:5] = 0
    per = sdf_features(mask, clamp_cells=4.0, periodic_y=True)
    for s in (30, 48):
        rolled = sdf_features(mask.roll(s, 1), clamp_cells=4.0, periodic_y=True)
        torch.testing.assert_close(rolled, per.roll(s, -2))


def test_patched_model_pickles() -> None:
    import pickle

    model = make_periodic_y(_conv_stack())
    x = torch.randn(1, 2, 8, 32, 16)
    with torch.no_grad():
        torch.testing.assert_close(pickle.loads(pickle.dumps(model))(x), model(x))


def test_unet_and_lora_equivariant() -> None:
    torch.manual_seed(0)
    unet = UNetConvNeXt(
        2, 2, base_channels=4, channel_mults=[1, 2], depths=[1, 1]
    ).eval()
    state, params = torch.randn(1, 2, 8, 32, 16), torch.randn(1, 2)
    mask = torch.ones(1, 8, 32, 16)

    def f(x: torch.Tensor) -> torch.Tensor:
        return unet(x, params, mask)

    assert _rel_roll_error(f, state, 8) > 1e-2
    make_periodic_y(unet)
    assert _rel_roll_error(f, state, 8) < 1e-5

    peft = pytest.importorskip("peft")
    lora = _conv_stack()
    config = peft.LoraConfig(r=2, target_modules=["0", "2"], init_lora_weights=False)
    lora = peft.inject_adapter_in_model(config, lora).eval()
    x = torch.randn(1, 2, 8, 32, 16)
    assert _rel_roll_error(lora, x, 4) > 1e-2
    make_periodic_y(lora)
    assert _rel_roll_error(lora, x, 4) < 1e-5


def test_simple_conv_and_p3d_equivariant() -> None:
    pytest.importorskip("p3d_surrogate")
    torch.manual_seed(0)
    nz, ny, nx = 16, 32, 16
    mask = _mask(nz, ny, nx)
    params = torch.randn(1, 2)
    state = torch.randn(1, 2, nz, ny, nx)
    p3d = P3D(
        n_state_channels=2,
        n_params=2,
        size="S",
        window_size=2,
        partition_size=1,
        sdf_features="both",
    ).eval()
    for model in (SimpleConv(n_state_channels=2, n_params=2).eval(), p3d):

        def f(x: torch.Tensor, model: nn.Module = model) -> torch.Tensor:
            s = x[:, :2]
            return model(s, params, x[:, 2])  # geometry rides in channel 2

        x = torch.cat([state, mask[None, None]], dim=1)
        assert _rel_roll_error(f, x, 8) > 1e-2
        with torch.no_grad():
            model(state, params, mask[None])  # fills P3D's SDF cache
        make_periodic_y(model)
        assert _rel_roll_error(f, x, 8) < 1e-5
    # P3D now computes the periodic features.
    torch.testing.assert_close(
        p3d._sdf_features_for(mask[None])[0],
        sdf_features(mask, clamp_cells=p3d.sdf_clamp_cells, periodic_y=True),
    )


def _stepper(**kwargs: Any) -> Any:
    pytest.importorskip("diffusers")
    pytest.importorskip("timm")
    pytest.importorskip("einops")
    from neural_surrogates import TadpoleTimeStepper

    torch.manual_seed(7)
    model = TadpoleTimeStepper(
        n_state_channels=2,
        n_params=2,
        encoder_crop_size=16,
        sdf_features="both",
        encode_geometry=True,
        subnetwork_cfg={"n_layers": 1, "hidden_size": 32, "num_heads": 2},
        **kwargs,
    ).eval()
    # Simulate a pretrained decoder so the test cannot pass vacuously.
    torch.nn.init.normal_(
        model.dft.decoder.transformer_decoder.transformer.final_layer.out_proj.weight,
        std=0.01,
    )
    return model


def test_tadpole_stepper_equivariant() -> None:
    model = _stepper(spatial_mode="global")
    # The token windows span 64 cells in y at the coarsest level: a 64-cell
    # roll keeps every window aligned, so only the padding breaks equivariance.
    nz, ny, nx = 16, 128, 16
    mask = _mask(nz, ny, nx)
    params = torch.randn(1, 2)
    x = torch.cat([torch.randn(1, 2, nz, ny, nx), mask[None, None]], dim=1)

    def f(x: torch.Tensor) -> torch.Tensor:
        return model(x[:, :2], params, x[:, 2])

    assert _rel_roll_error(f, x, 64) > 1e-1
    make_periodic_y(model)
    assert _rel_roll_error(f, x, 64) < 1e-3


@pytest.mark.parametrize("mode", ["global", "local"])
def test_tadpole_ny_must_tile(mode: str) -> None:
    model = make_periodic_y(_stepper(spatial_mode=mode))
    ny = 48 if mode == "local" else 40  # local: one crop must span y
    mask = torch.ones(16, ny, 16)
    with pytest.raises(ValueError, match="periodic_y: ny"):
        model(torch.randn(1, 2, 16, ny, 16), torch.randn(1, 2), mask[None])


def test_tadpole_halo_raises() -> None:
    with pytest.raises(ValueError, match="halo"):
        make_periodic_y(_stepper(spatial_mode="halo", halo_size=16))
