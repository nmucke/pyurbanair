"""History conditioning (``num_history_steps``) across the next-step architectures.

Every next-step architecture takes the ``H`` past frames pre-flattened along the
channel axis, oldest first -- ``state`` is ``(B, H*C, nz, ny, nx)`` and the
newest frame is ``state[:, -C:]``. Only the input stem widens; the head still
emits ``C`` channels, the normalisation buffers stay length ``C`` (tiled ``H``
times at the input) and the residual base is the newest frame.

The must-hold invariant is the repo's no-op-when-absent rule: at ``H = 1`` the
state dict and the forward output are byte-identical to a model built without
the argument (mirrors ``test_unet_convnext_extra_channels.py`` /
``test_p3d_architecture.py``). ``DomainDecomposed`` and ``TadpoleTimeStepper``
accept the key (so Hydra instantiation never fails) but reject ``H > 1``.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import pytest

torch = pytest.importorskip("torch")

from neural_surrogates import UPT, SimpleConv, UNetConvNeXt

N_STATE = 3
N_PARAMS = 2
GRID = (8, 8, 8)
P3D_GRID = (16, 16, 16)  # P3D's U-structure downsamples 16x
HISTORY = 3


# -- builders ---------------------------------------------------------------
#
# ``build`` takes the history kwarg or omits it entirely, so the H=1 tests can
# compare "with the key" against "without the key"; ``stem`` reaches the input
# projection whose width the history widens.


class _Spec(NamedTuple):
    build: Callable[..., Any]
    stem: Callable[[Any], Any]
    grid: tuple[int, int, int]
    masks_output: bool
    normalizes: bool
    residual: bool


def _build_unet(**kw: Any) -> UNetConvNeXt:
    return UNetConvNeXt(
        N_STATE,
        N_PARAMS,
        base_channels=8,
        channel_mults=[1, 2],
        depths=[1, 1],
        normalize=True,
        residual=True,
        **kw,
    )


def _build_simple(**kw: Any) -> SimpleConv:
    return SimpleConv(N_STATE, N_PARAMS, **kw)


def _build_upt(**kw: Any) -> UPT:
    return UPT(
        N_STATE,
        N_PARAMS,
        dim=32,
        num_latent_tokens=16,
        num_supernodes=16,
        gnn_dim=16,
        enc_depth=1,
        approx_depth=1,
        dec_depth=1,
        num_heads=2,
        radius=2.5,
        max_degree=8,
        **kw,
    )


def _build_p3d(**kw: Any) -> Any:
    pytest.importorskip("p3d_surrogate")
    from neural_surrogates import P3D

    return P3D(N_STATE, N_PARAMS, size="S", window_size=2, partition_size=1, **kw)


_SPECS: dict[str, _Spec] = {
    "unet_convnext": _Spec(
        build=_build_unet,
        stem=lambda m: m.stem.weight,
        grid=GRID,
        masks_output=True,
        normalizes=True,
        residual=True,
    ),
    "simple_conv": _Spec(
        build=_build_simple,
        stem=lambda m: m.conv.weight,
        grid=GRID,
        masks_output=False,
        normalizes=False,
        residual=False,
    ),
    "p3d": _Spec(
        build=_build_p3d,
        stem=lambda m: m.net.encoder.feature_embed.weight,
        grid=P3D_GRID,
        masks_output=True,
        normalizes=True,
        residual=True,
    ),
    "upt": _Spec(
        build=_build_upt,
        stem=lambda m: m.encoder.supernode_pooling.input_proj.proj.weight,
        grid=GRID,
        masks_output=True,
        normalizes=True,
        residual=True,
    ),
}

ARCHS = sorted(_SPECS)


def _inputs(
    grid: tuple[int, int, int],
    history: int = 1,
    batch: int = 2,
    seed: int = 0,
    holes: bool = False,
) -> tuple[Any, Any, Any]:
    """``(state (B, H*C, *grid), params, geometry (B, 1, *grid))``.

    The geometry is shared across the batch (UPT's fast path assumes it), and
    with ``holes`` it carries obstacle cells so output masking is observable.
    """
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(batch, history * N_STATE, *grid, generator=g)
    params = torch.randn(batch, N_PARAMS, generator=g)
    if holes:
        mask = (torch.rand(1, 1, *grid, generator=g) > 0.3).float()
        geometry = mask.repeat(batch, 1, 1, 1, 1)
    else:
        geometry = torch.ones(batch, 1, *grid)
    return state, params, geometry


class _ZeroSampleNet(torch.nn.Module):
    """Stand-in for P3D's upstream net: emits a zero ``.sample``."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: Any, pde_parameters: Any = None) -> Any:
        zeros = x.new_zeros(x.shape[0], self.out_channels, *x.shape[-3:])
        return type("Out", (), {"sample": zeros})()


class _ZeroDecoder(torch.nn.Module):
    """Stand-in for UPT's perceiver decoder: emits zero per-point predictions."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, latent: Any, query_pos: Any, condition: Any = None) -> Any:
        return latent.new_zeros(latent.shape[0], query_pos.shape[1], self.out_channels)


def _zero_nonresidual_branch(name: str, model: Any) -> None:
    """Force the network branch to output exactly 0, leaving only the residual.

    Whatever survives is then the residual base, which must be the newest
    history frame.
    """
    if name == "unet_convnext":
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
    elif name == "p3d":
        model.net = _ZeroSampleNet(N_STATE)
    elif name == "upt":
        model.decoder = _ZeroDecoder(N_STATE)
    else:  # pragma: no cover - guarded by the residual flag
        raise AssertionError(f"{name} has no residual branch")


# -- (a) H = 1 is byte-identical to omitting the kwarg ----------------------


@pytest.mark.parametrize("name", ARCHS)
def test_h1_state_dict_identical(name: str) -> None:
    spec = _SPECS[name]
    torch.manual_seed(0)
    baseline = spec.build()
    torch.manual_seed(0)
    widened = spec.build(num_history_steps=1)

    assert widened.num_history_steps == 1
    assert widened.n_input_state_channels == N_STATE
    base_sd, hist_sd = baseline.state_dict(), widened.state_dict()
    assert set(base_sd) == set(hist_sd)
    assert {k: v.shape for k, v in base_sd.items()} == {
        k: v.shape for k, v in hist_sd.items()
    }
    # stem width unchanged -> existing checkpoints load
    assert spec.stem(baseline).shape == spec.stem(widened).shape
    widened.load_state_dict(base_sd)


@pytest.mark.parametrize("name", ARCHS)
def test_h1_forward_identical(name: str) -> None:
    spec = _SPECS[name]
    torch.manual_seed(0)
    baseline = spec.build().eval()
    torch.manual_seed(0)
    widened = spec.build(num_history_steps=1).eval()
    state, params, geometry = _inputs(spec.grid, holes=True)
    with torch.no_grad():
        a = baseline(state, params, geometry)
        b = widened(state, params, geometry)
    assert torch.equal(a, b)


# -- (b) H > 1 runs, is masked, and gradients flow --------------------------


@pytest.mark.parametrize("name", ARCHS)
def test_history_forward_shape_and_mask(name: str) -> None:
    spec = _SPECS[name]
    torch.manual_seed(0)
    model = spec.build(num_history_steps=HISTORY).eval()
    assert model.num_history_steps == HISTORY
    assert model.n_input_state_channels == HISTORY * N_STATE
    # the stem absorbs H*C state channels; the head still emits C
    assert spec.stem(model).shape[1] >= HISTORY * N_STATE

    state, params, geometry = _inputs(spec.grid, history=HISTORY, holes=True)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (state.shape[0], N_STATE, *spec.grid)
    assert torch.isfinite(out).all()
    if spec.masks_output:
        assert torch.equal(out, out * geometry)


@pytest.mark.parametrize("name", ARCHS)
def test_history_gradients_reach_stem(name: str) -> None:
    """The loss backprops into the stem, and the OLDEST frame gets a gradient
    (i.e. the history is genuinely consumed, not just padding)."""
    spec = _SPECS[name]
    torch.manual_seed(0)
    model = spec.build(num_history_steps=HISTORY)
    state, params, geometry = _inputs(spec.grid, history=HISTORY)
    state.requires_grad_(True)

    out = model(state, params, geometry)
    out.pow(2).mean().backward()

    stem_grad = spec.stem(model).grad
    assert stem_grad is not None
    assert torch.isfinite(stem_grad).all()
    assert stem_grad.abs().sum() > 0
    assert state.grad is not None
    assert state.grad[:, :N_STATE].abs().sum() > 0


# -- (c) the residual base is the NEWEST frame ------------------------------


@pytest.mark.parametrize("name", [n for n in ARCHS if _SPECS[n].residual])
def test_residual_base_is_last_history_frame(name: str) -> None:
    spec = _SPECS[name]
    torch.manual_seed(0)
    model = spec.build(num_history_steps=HISTORY).eval()
    _zero_nonresidual_branch(name, model)

    state, params, geometry = _inputs(spec.grid, history=HISTORY, holes=True)
    with torch.no_grad():
        out = model(state, params, geometry)

    last = state[:, -N_STATE:] * geometry
    first = state[:, :N_STATE] * geometry
    assert torch.allclose(out, last, atol=1e-6)
    # the frames really do differ, so this is a meaningful discrimination
    assert not torch.allclose(out, first, atol=1e-3)


# -- (d) normalisation stats stay length C ----------------------------------


@pytest.mark.parametrize("name", [n for n in ARCHS if _SPECS[n].normalizes])
def test_set_normalization_takes_length_c_stats_at_h3(name: str) -> None:
    spec = _SPECS[name]
    torch.manual_seed(0)
    model = spec.build(num_history_steps=HISTORY).eval()

    state_mean = torch.tensor([0.5, -0.25, 0.125])
    state_std = torch.tensor([2.0, 0.5, 4.0])
    param_mean = torch.zeros(N_PARAMS)
    param_std = torch.ones(N_PARAMS)
    model.set_normalization(state_mean, state_std, param_mean, param_std)

    # buffers stay length C (checkpoints / _normalization_signature depend on it)
    assert model.state_mean.shape == (N_STATE,)
    assert model.state_std.shape == (N_STATE,)
    assert torch.allclose(model.state_mean, state_mean)

    state, params, geometry = _inputs(spec.grid, history=HISTORY, holes=True)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (state.shape[0], N_STATE, *spec.grid)
    assert torch.isfinite(out).all()


# -- validation -------------------------------------------------------------


@pytest.mark.parametrize("name", ARCHS)
@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_history_raises(name: str, bad: int) -> None:
    with pytest.raises(ValueError):
        _SPECS[name].build(num_history_steps=bad)


# -- (e) wrappers that do not support history -------------------------------


def test_domain_decomposed_accepts_h1_and_rejects_more() -> None:
    from neural_surrogates import DomainDecomposed

    unet = "neural_surrogates.UNetConvNeXt"
    subnet = dict(_target_=unet, base_channels=8, channel_mults=[1, 2], depths=[1, 1])
    decomposition = dict(interior_size=8, halo=3, taper=2, coarsen_factor=2, n_pos=3)
    model = DomainDecomposed(
        N_STATE,
        N_PARAMS,
        decomposition=decomposition,
        fine_net=subnet,
        coarse_net=subnet,
        num_history_steps=1,
    )
    assert model.num_history_steps == 1
    # the key is NOT forwarded to the sub-nets (phase 1 keeps DD single-frame)
    assert model.fine_net.num_history_steps == 1
    with pytest.raises(NotImplementedError, match="num_history_steps"):
        DomainDecomposed(
            N_STATE,
            N_PARAMS,
            decomposition=decomposition,
            fine_net=subnet,
            coarse_net=subnet,
            num_history_steps=2,
        )


def test_tadpole_stepper_accepts_h1_and_rejects_more() -> None:
    pytest.importorskip("diffusers")
    pytest.importorskip("timm")
    pytest.importorskip("einops")
    from neural_surrogates import TadpoleTimeStepper

    def _build(history: int) -> TadpoleTimeStepper:
        return TadpoleTimeStepper(
            n_state_channels=N_STATE,
            n_params=N_PARAMS,
            size="S",
            pretrained_ae_dir=None,
            skip_pretrained_load=True,
            encoder_crop_size=16,
            sdf_clamp_cells=8,
            num_history_steps=history,
        )

    # H > 1 must fail BEFORE the (slow) DFT build.
    with pytest.raises(NotImplementedError, match="num_history_steps"):
        _build(2)
    model = _build(1)
    assert model.num_history_steps == 1
    assert model.n_input_state_channels == N_STATE
