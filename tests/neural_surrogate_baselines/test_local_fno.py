"""Local-FNO: patch tiling, the spectral layer, the stepper and its loss."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from neural_surrogate_baselines import LocalFNOStepper
from neural_surrogate_baselines.local_fno.patches import PatchGrid
from neural_surrogate_baselines.local_fno.spectral import SpectralConv3d
from neural_surrogate_baselines.losses import masked_rmse

NZ, NY, NX = 6, 20, 20


def _tiny(**overrides: Any) -> LocalFNOStepper:
    kwargs: dict[str, Any] = dict(
        n_state_channels=3,
        n_params=2,
        width=8,
        modes=(2, 4, 4),
        n_layers=2,
        patch_core=8,
        patch_overlap=2,
        sdf_clamp_cells=4,
    )
    kwargs.update(overrides)
    return LocalFNOStepper(**kwargs)


def _geometry(batch: int = 2) -> torch.Tensor:
    geometry = torch.ones(batch, NZ, NY, NX)
    geometry[:, :2, 7:13, 7:13] = 0.0
    return geometry


@pytest.mark.parametrize("periodic_y", [True, False])  # type: ignore[misc]
@pytest.mark.parametrize("periodic_x", [True, False])  # type: ignore[misc]
@pytest.mark.parametrize("ny,nx", [(20, 20), (19, 23), (8, 8), (5, 30)])  # type: ignore[misc]
def test_stitching_the_patches_restores_the_field(
    ny: int, nx: int, periodic_y: bool, periodic_x: bool
) -> None:
    grid = PatchGrid(
        ny, nx, core=8, overlap=2, periodic_y=periodic_y, periodic_x=periodic_x
    )
    field = torch.randn(2, 3, 4, ny, nx)
    patches = grid.extract(field)
    assert patches.shape == (2 * grid.n_patches, 3, 4, 12, 12)
    torch.testing.assert_close(grid.stitch(patches), field, rtol=0, atol=0)


def test_patch_overlap_is_the_neighbouring_data() -> None:
    """A patch's overlap is its neighbour's edge (or the wrapped/repeated edge)."""
    grid = PatchGrid(16, 16, core=8, overlap=2, periodic_y=True, periodic_x=False)
    field = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 1, 16, 16)
    patches = grid.extract(field)  # 2 x 2 patches, row-major
    first, right = patches[0, 0, 0], patches[1, 0, 0]
    # The right overlap of patch (0, 0) is the left of its core neighbour (0, 1).
    torch.testing.assert_close(first[2:10, 10:12], right[2:10, 2:4])
    # y is periodic: the top overlap of row 0 wraps to the last rows.
    torch.testing.assert_close(first[0:2, 2:10], field[0, 0, 0, 14:16, 0:8])
    # x is not: the left overlap of column 0 repeats column 0 and is invalid.
    torch.testing.assert_close(first[2:10, 0], field[0, 0, 0, 0:8, 0])
    valid = grid.valid(nz=1)[0, 0, 0]
    assert not valid[:, :2].any() and valid[:, 2:].all()


def test_valid_marks_only_real_cells() -> None:
    grid = PatchGrid(19, 23, core=8, overlap=2, periodic_y=False, periodic_x=False)
    valid = grid.valid(nz=3)
    assert valid.shape == (grid.n_patches, 1, 3, 12, 12)
    # The real cells covered by patch cores add up to the grid; each real cell
    # is also seen in the overlaps of its neighbours.
    cores = grid.stitch(valid.float().unsqueeze(0).flatten(0, 1))
    assert bool(cores.all())
    padded = grid.pad(torch.ones(1, 1, 3, 19, 23))
    assert padded.shape[-2:] == (3 * 8 + 4, 3 * 8 + 4)


def test_spectral_conv_is_translation_equivariant_on_periodic_grids() -> None:
    torch.manual_seed(0)
    conv = SpectralConv3d(3, 4, (2, 3, 3))
    x = torch.randn(1, 3, 6, 10, 12)
    shifted = torch.roll(x, shifts=(1, 2, 3), dims=(-3, -2, -1))
    torch.testing.assert_close(
        conv(shifted), torch.roll(conv(x), shifts=(1, 2, 3), dims=(-3, -2, -1))
    )


def test_spectral_conv_keeps_only_the_low_modes() -> None:
    """A wave above the kept modes passes as zero."""
    conv = SpectralConv3d(1, 1, (1, 2, 2))
    y = torch.arange(16, dtype=torch.float32)
    x = (
        torch.cos(2 * torch.pi * 5 * y / 16)
        .reshape(1, 1, 1, 16, 1)
        .expand(1, 1, 4, 16, 8)
    )
    torch.testing.assert_close(conv(x), torch.zeros_like(x), atol=1e-5, rtol=0)


def test_forward_shape_masking_and_any_grid() -> None:
    torch.manual_seed(0)
    model = _tiny()
    geometry = _geometry()
    out = model(torch.randn(2, 6, NZ, NY, NX), torch.randn(2, 2), geometry)
    assert out.shape == (2, 3, NZ, NY, NX)
    assert torch.all(out[:, :, :2, 7:13, 7:13] == 0)
    # Patches make the network grid-independent (domain_flexible).
    odd = model(
        torch.randn(1, 6, NZ, 13, 27), torch.randn(1, 2), torch.ones(1, NZ, 13, 27)
    )
    assert odd.shape == (1, 3, NZ, 13, 27)


def test_a_patch_core_sees_only_its_patch() -> None:
    """Local-FNO is local: input outside a patch (overlap included) leaves the
    patch's core prediction unchanged, input inside it does not."""
    torch.manual_seed(0)
    model = _tiny()
    state, params, geometry = (
        torch.randn(1, 6, NZ, NY, NX),
        torch.randn(1, 2),
        torch.ones(1, NZ, NY, NX),
    )
    base = model(state, params, geometry)
    # Patch (0, 0): core y, x in 0..7; with the overlap it reads y in -2..9
    # (wrapping to 18, 19) and x in 0..9 (x is not periodic).
    far = state.clone()
    far[..., 12:16, 13:17] += 5.0
    torch.testing.assert_close(
        model(far, params, geometry)[..., :8, :8], base[..., :8, :8]
    )
    near = state.clone()
    near[..., 18:20, 9:10] += 5.0  # in its wrapped y overlap and its x overlap
    assert not torch.allclose(
        model(near, params, geometry)[..., :8, :8], base[..., :8, :8]
    )


def test_sdf_features_are_cached_per_geometry() -> None:
    model = _tiny()
    geometry = _geometry()
    first = model._sdf_features_for(geometry)
    assert first.shape == (2, 1, NZ, NY, NX)
    assert model._sdf_features_for(geometry) is first
    assert model._sdf_features_for(geometry.clone()) is first
    # An in-place edit of the mask is not served from the cache.
    geometry[:, 4, 3, 3] = 0.0
    assert not torch.equal(model._sdf_features_for(geometry), first)


def test_normalization_round_trip() -> None:
    """With the network's output zeroed, the prediction is the state mean."""
    model = _tiny(n_params=0)
    model.set_normalization([1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [], [])
    assert model.state_std.tolist() == [2.0, 1.0, 1.0]
    for p in model.projection[-1].parameters():
        torch.nn.init.zeros_(p)
    out = model(
        torch.randn(1, 6, NZ, NY, NX), torch.zeros(1, 0), torch.ones(1, NZ, NY, NX)
    )
    torch.testing.assert_close(
        out, torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1, 1).expand_as(out)
    )


def test_paper_size() -> None:
    model = LocalFNOStepper(n_state_channels=3, n_params=2)
    # 4 layers x 4 corner blocks x 36 x 36 x 8 x 16 x 16 complex weights (two
    # reals each) dominate; the paper does not report a parameter count.
    spectral = sum(p.numel() for n, p in model.named_parameters() if "spectral" in n)
    assert spectral == 2 * 4 * 4 * 36 * 36 * 8 * 16 * 16


def test_dtype_casts() -> None:
    """The forward model casts the model with ``.to(dtype)``: the spectral
    weights survive, and every dtype runs end to end in that dtype."""
    torch.manual_seed(0)
    model = _tiny()
    state, params, geometry = (
        torch.randn(1, 6, NZ, NY, NX),
        torch.randn(1, 2),
        _geometry(1),
    )
    expected = model(state, params, geometry)
    torch.testing.assert_close(
        model.to(torch.float32)(state, params, geometry), expected
    )
    double = model.to(torch.float64)(state.double(), params.double(), geometry.double())
    assert double.dtype == torch.float64
    torch.testing.assert_close(double.float(), expected, rtol=1e-4, atol=1e-4)
    half = model.to(torch.bfloat16)(
        state.bfloat16(), params.bfloat16(), geometry.bfloat16()
    )
    assert half.dtype == torch.bfloat16 and torch.isfinite(half).all()


def test_spectral_conv_matches_a_numpy_low_pass() -> None:
    """With unit weights the layer is the low-pass filter of its kept modes,
    computed here independently with a full numpy FFT. z and y keep all their
    modes, so the filter is |k_x| < m_x (a real FFT keeps both signs)."""
    conv = SpectralConv3d(1, 1, (2, 3, 3))
    for w in conv.weights:
        torch.nn.init.zeros_(w)
        with torch.no_grad():
            w[..., 0] = 1.0
    nz, ny, nx = 4, 6, 12
    x = torch.randn(1, 1, nz, ny, nx)
    kx = np.fft.fftfreq(nx, 1 / nx).round().astype(int)
    expected = np.fft.ifftn(np.fft.fftn(x[0, 0].numpy()) * (np.abs(kx) < 3)).real
    torch.testing.assert_close(
        conv(x)[0, 0], torch.from_numpy(expected).float(), atol=1e-5, rtol=1e-4
    )


def test_masked_rmse() -> None:
    pred = torch.tensor([[1.0, 2.0, 10.0]])
    target = torch.tensor([[0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False]])
    torch.testing.assert_close(
        masked_rmse(pred, target, mask), torch.sqrt(torch.tensor(2.5))
    )
