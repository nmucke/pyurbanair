"""SSRollingUrbanNet: the Aurora adapter, SSGen, the rollout dataset and trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import xarray as xr
from aurora.model.film import AdaptiveLayerNorm
from neural_surrogate_baselines.datasets import RolloutTransitionDataset
from neural_surrogate_baselines.losses import spectral_loss
from neural_surrogate_baselines.ssrolling.model import SSRollingUrbanNet
from neural_surrogate_baselines.ssrolling.training import RolloutTrainer
from neural_surrogates.datasets.transition import transition_collate
from torch.utils.data import DataLoader

NZ, NY, NX = 6, 20, 20


def _tiny(**overrides: Any) -> SSRollingUrbanNet:
    kwargs: dict[str, Any] = dict(
        n_state_channels=3,
        n_params=2,
        num_history_steps=2,
        embed_dim=32,
        num_heads=2,
        # Two blocks per stage: Aurora shifts the windows of every second one.
        encoder_depths=(2, 2),
        encoder_num_heads=(2, 4),
        decoder_depths=(2, 2),
        decoder_num_heads=(4, 2),
        window_size=(2, 2, 2),
        ssgen_channels=16,
        n_levels=NZ,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    model = SSRollingUrbanNet(**kwargs)
    model.eval()
    # Aurora zero-initialises the adaptive-LayerNorm modulation, so at init its
    # Swin blocks are identities: neither the conditioning nor the windows
    # would show. Give them weights, as training would.
    for module in model.modules():
        if isinstance(module, AdaptiveLayerNorm):
            torch.nn.init.normal_(module.ln_modulation[-1].weight, std=0.2)
            torch.nn.init.normal_(module.ln_modulation[-1].bias, std=0.2)
    return model


def _geometry(batch: int = 2, ny: int = NY, nx: int = NX) -> torch.Tensor:
    geometry = torch.ones(batch, NZ, ny, nx)
    geometry[:, :2, 7:13, 7:13] = 0.0
    return geometry


def _inputs(
    batch: int = 2, ny: int = NY, nx: int = NX
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    state = torch.randn(batch, 6, NZ, ny, nx, generator=gen)
    params = torch.randn(batch, 2, generator=gen)
    return state, params, _geometry(batch, ny, nx)


@pytest.mark.parametrize("ssgen", [True, False])  # type: ignore[misc]
def test_predicts_the_next_state_zero_in_buildings(ssgen: bool) -> None:
    model = _tiny(ssgen=ssgen)
    assert model.num_history_steps == 2 and model.n_state_channels == 3
    assert model.domain_flexible and not model.compile_dynamic
    assert getattr(model, "n_geom_feature_channels", 0) == 0
    assert not hasattr(model, "_sdf_features")
    state, params, geometry = _inputs()
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (2, 3, NZ, NY, NX)
    assert torch.isfinite(out).all()
    solid = geometry == 0
    assert (out.permute(1, 0, 2, 3, 4)[:, solid] == 0).all()
    assert (out.permute(1, 0, 2, 3, 4)[:, ~solid] != 0).any()
    # The inflow parameters reach the prediction.
    with torch.no_grad():
        assert not torch.allclose(model(state, params + 1.0, geometry), out)


def test_ssgen_corrects_the_backbone() -> None:
    with_ssgen, without = _tiny(ssgen=True), _tiny(ssgen=False)
    without.load_state_dict(with_ssgen.state_dict(), strict=False)
    state, params, geometry = _inputs()
    with torch.no_grad():
        assert not torch.allclose(
            with_ssgen(state, params, geometry), without(state, params, geometry)
        )
    assert with_ssgen.ssgen is not None
    with pytest.raises(ValueError, match="n_levels"):
        _tiny(n_levels=NZ + 1)(state, params, geometry)


def test_paper_parameter_counts() -> None:
    """The papers' 451 M model and the size-matched 113 M arm."""
    paper = SSRollingUrbanNet(3, 2, ssgen=False, embed_dim=512, num_heads=16)
    aurora = sum(p.numel() for p in paper.aurora.parameters())
    total = sum(p.numel() for p in paper.parameters())
    print(f"embed_dim 512: Aurora {aurora:,}, with the inflow MLP {total:,}")
    assert aurora == pytest.approx(451.4e6, abs=0.2e6)
    del paper
    small = SSRollingUrbanNet(3, 2, ssgen=False, embed_dim=256, num_heads=8)
    aurora = sum(p.numel() for p in small.aurora.parameters())
    print(f"embed_dim 256: Aurora {aurora:,}")
    assert aurora == pytest.approx(113.3e6, abs=0.2e6)
    with_ssgen = SSRollingUrbanNet(3, 2, ssgen=True, embed_dim=256, num_heads=8)
    assert with_ssgen.ssgen is not None
    print(
        f"SSGen at n_levels 32: {sum(p.numel() for p in with_ssgen.ssgen.parameters()):,}"
    )


def test_aurora_takes_two_snapshots_and_one_wrapped_axis() -> None:
    with pytest.raises(ValueError, match="two input snapshots"):
        _tiny(num_history_steps=1)
    with pytest.raises(ValueError, match="one lateral axis"):
        _tiny(periodic_axes=("y", "x"))


def test_set_normalization_round_trip() -> None:
    """Statistics only rescale: physical inputs give the de-normalised
    prediction of the matching z-scored inputs."""
    identity, scaled = _tiny(ssgen=True), _tiny(ssgen=True)
    mean, std = np.array([2.0, -1.0, 0.5]), np.array([3.0, 0.5, 0.0])
    p_mean, p_std = np.array([10.0, 4.0]), np.array([2.0, 5.0])
    scaled.set_normalization(mean, std, p_mean, p_std)
    torch.testing.assert_close(scaled.state_std, torch.tensor([3.0, 0.5, 1.0]))
    torch.testing.assert_close(
        scaled.param_mean, torch.tensor(p_mean, dtype=torch.float32)
    )
    state, params, geometry = _inputs()
    m = scaled.state_mean.repeat(2).view(1, -1, 1, 1, 1)
    s = scaled.state_std.repeat(2).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        z_out = identity(state, params, geometry)
        out = scaled(
            state * s + m, params * scaled.param_std + scaled.param_mean, geometry
        )
    expected = z_out * scaled.state_std.view(1, -1, 1, 1, 1) + scaled.state_mean.view(
        1, -1, 1, 1, 1
    )
    torch.testing.assert_close(
        out, expected * geometry.unsqueeze(1), rtol=1e-4, atol=1e-4
    )
    # The statistics travel with the weights.
    restored = _tiny(ssgen=True)
    restored.load_state_dict(scaled.state_dict())
    torch.testing.assert_close(restored.state_std, scaled.state_std)


@pytest.mark.parametrize("periodic_axes", [(), ("y",), ("x",)])  # type: ignore[misc]
def test_pads_and_crops_grids_off_the_patch_size(
    periodic_axes: tuple[str, ...]
) -> None:
    model = _tiny(periodic_axes=periodic_axes)
    state, params, geometry = _inputs(ny=19, nx=22)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (2, 3, NZ, 19, 22)
    assert torch.isfinite(out).all()
    assert (out.permute(1, 0, 2, 3, 4)[:, geometry == 0] == 0).all()


def test_the_periodic_axis_is_aurora_s_wrapped_axis() -> None:
    """y periodic puts y on Aurora's wrapped W axis: on a square grid, the
    y-periodic model is the x-periodic one on the transposed fields. Without a
    periodic axis the wrap is off, which changes the prediction. (Aurora's
    wrap only connects the edges when the patches along W fill whole windows:
    16 cells are 4 patches, two windows of 2.)"""
    on_y = _tiny(ssgen=False, periodic_axes=("y",))
    on_x = _tiny(ssgen=False, periodic_axes=("x",))
    closed = _tiny(ssgen=False, periodic_axes=())
    on_x.load_state_dict(on_y.state_dict())
    closed.load_state_dict(on_y.state_dict())
    state, params, geometry = _inputs(ny=16, nx=16)
    swap = lambda t: t.transpose(-1, -2)  # noqa: E731
    with torch.no_grad():
        out_y = on_y(state, params, geometry)
        torch.testing.assert_close(
            out_y, swap(on_x(swap(state), params, swap(geometry)))
        )
        assert not torch.allclose(
            closed(state, params, geometry), on_x(state, params, geometry)
        )


def test_static_variables_need_one_geometry_per_batch() -> None:
    model = _tiny()
    state, params, geometry = _inputs()
    geometry[1, :2, 0:3, 0:3] = 0.0
    with pytest.raises(ValueError, match="same geometry"):
        model(state, params, geometry)


def test_aurora_normalisation_tables_are_only_extended() -> None:
    from aurora.normalisation import locations, scales

    locations_before, scales_before = dict(locations), dict(scales)
    model = _tiny()
    with torch.no_grad():
        model(*_inputs())
    assert {k: locations[k] for k in locations_before} == locations_before
    assert {k: scales[k] for k in scales_before} == scales_before
    assert all(k.startswith("nsb_") for k in set(locations) - set(locations_before))
    ours = {"nsb_height", "nsb_hstatic", "nsb_state0_1", "nsb_state2_6", "nsb_solid_6"}
    assert ours <= set(locations)
    nsb = [k for k in locations if k.startswith("nsb_")]
    assert all(locations[k] == 0.0 and scales[k] == 1.0 for k in nsb)


# --- rollout dataset and trainer -------------------------------------------

T_LEN = 7
GRID = 12


def _write_sample(root: Path, idx: int, seed: int) -> None:
    state_dir, param_dir = root / "state" / "train", root / "param" / "train"
    state_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dims = ("time", "z", "y", "x")
    shape = (T_LEN, NZ, GRID, GRID)
    data_vars: dict[str, Any] = {
        v: (dims, rng.standard_normal(shape).astype(np.float32)) for v in "uvw"
    }
    obstacle = np.zeros((NZ, GRID, GRID), dtype=np.float32)
    obstacle[0:2, 2:4, 2:4] = 1.0
    data_vars["blanking"] = (("z", "y", "x"), obstacle)
    xr.Dataset(data_vars).to_netcdf(state_dir / f"sample_{idx:04d}.nc")
    params = {
        "inflow_angle": (("time",), rng.uniform(-60, 60, T_LEN).astype(np.float32)),
        "velocity_magnitude": (("time",), rng.uniform(1, 5, T_LEN).astype(np.float32)),
    }
    xr.Dataset(params).to_netcdf(param_dir / f"sample_{idx:04d}.nc")


@pytest.fixture  # type: ignore[misc]
def corpus(tmp_path: Path) -> Path:
    for idx in range(2):
        _write_sample(tmp_path, idx, seed=idx)
    return tmp_path


def _dataset(root: Path, k: int) -> RolloutTransitionDataset:
    return RolloutTransitionDataset(
        root, "train", param_vars=None, num_history_steps=2, pushforward_steps=k
    )


@pytest.mark.parametrize("k", [1, 3])  # type: ignore[misc]
def test_rollout_dataset_adds_every_intermediate_target(corpus: Path, k: int) -> None:
    ds = _dataset(corpus, k)
    assert len(ds) == 2 * (T_LEN - k - 1)
    idx = len(ds) - 1
    traj, t = ds.sample_index[idx]
    item = ds[idx]
    with xr.open_dataset(ds._state_files[traj]) as raw:
        frames = torch.from_numpy(np.stack([raw[v].values for v in "uvw"], axis=1))
    assert item["state_targets"].shape == (k, 3, NZ, GRID, GRID)
    torch.testing.assert_close(item["state_targets"], frames[t + 1 : t + k + 1])
    torch.testing.assert_close(item["state_n"], frames[t - 1 : t + 1].flatten(0, 1))
    torch.testing.assert_close(item["state_next"], item["state_targets"][-1])
    # Every key of the plain dataset is unchanged.
    plain = super(RolloutTransitionDataset, ds).__getitem__(idx)
    for key, value in plain.items():
        torch.testing.assert_close(item[key], value)
    batch = transition_collate([ds[0], ds[1]])
    assert batch["state_targets"].shape == (2, k, 3, NZ, GRID, GRID)


def test_spectral_loss_is_a_scaled_mse() -> None:
    """Parseval: the backward-normalised spectral loss is ny * MSE."""
    pred, target = torch.randn(2, 3, NZ, 8, 5), torch.randn(2, 3, NZ, 8, 5)
    mse = torch.mean((pred - target) ** 2)
    torch.testing.assert_close(spectral_loss(pred, target, dim=-2), 8 * mse)
    torch.testing.assert_close(spectral_loss(pred, target, dim=-2, norm="ortho"), mse)


class _Linear(torch.nn.Module):
    """``a * newest + b * oldest + c * params``: a two-frame stepper."""

    num_history_steps = 2
    n_state_channels = 3

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.9))
        self.b = torch.nn.Parameter(torch.tensor(0.2))
        self.c = torch.nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = (
            self.a * state[:, 3:]
            + self.b * state[:, :3]
            + self.c * params.sum(1).view(-1, 1, 1, 1, 1)
        )
        return out * geometry.unsqueeze(1)


def _trainer(
    corpus: Path, model: torch.nn.Module, k: int, alpha: float = 1.0
) -> RolloutTrainer:
    loader = DataLoader(
        _dataset(corpus, k), batch_size=2, collate_fn=transition_collate
    )
    return RolloutTrainer(
        model,
        loader,
        loader,
        torch.optim.Adam(model.parameters()),
        torch.nn.MSELoss(),
        num_epochs=1,
        alpha=alpha,
    )


def test_rollout_loss_sums_every_step_with_gradients_through_all(corpus: Path) -> None:
    model = _Linear()
    trainer = _trainer(corpus, model, k=2, alpha=0.5)
    batch = next(iter(trainer.train_loader))
    loss = trainer._forward(batch)
    grads = torch.autograd.grad(loss, list(model.parameters()))

    # By hand: unroll two steps, sum MSE + 0.5 * L_spec over the fluid cells.
    geometry = batch["geometry"][0]
    fluid = geometry.bool()
    state = batch["state_n"]
    expected = mse_sum = torch.zeros(())
    for i in range(2):
        pred = model(
            state, batch["params_n"][:, i], geometry.expand(2, *geometry.shape)
        )
        target = batch["state_targets"][:, i]
        mse = torch.mean((pred[..., fluid] - target[..., fluid]) ** 2)
        spec = spectral_loss(pred * geometry, target * geometry, dim=-2)
        expected = expected + mse + 0.5 * spec
        mse_sum = mse_sum + mse
        state = torch.cat([state[:, 3:], pred], dim=1)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(trainer._aux_terms["mse"], mse_sum.detach())
    for got, want in zip(
        grads, torch.autograd.grad(expected, list(model.parameters()))
    ):
        torch.testing.assert_close(got, want)

    # Cutting the graph between the steps changes the gradient: it flows
    # through the first prediction into the second step's loss.
    pred0 = model(
        batch["state_n"], batch["params_n"][:, 0], geometry.expand(2, *geometry.shape)
    )
    cut = torch.cat([batch["state_n"][:, 3:], pred0.detach()], dim=1)
    pred1 = model(cut, batch["params_n"][:, 1], geometry.expand(2, *geometry.shape))
    target = batch["state_targets"]
    detached = sum(
        torch.mean((p[..., fluid] - target[:, i][..., fluid]) ** 2)
        + 0.5 * spectral_loss(p * geometry, target[:, i] * geometry, dim=-2)
        for i, p in enumerate((pred0, pred1))
    )
    cut_grads = torch.autograd.grad(detached, list(model.parameters()))
    assert not torch.allclose(cut_grads[0], grads[0])


def test_rollout_trainer_fits_the_aurora_stepper(corpus: Path) -> None:
    model = _tiny(n_params=2)
    model.train()
    trainer = _trainer(corpus, model, k=1)
    history = trainer.fit()
    assert np.isfinite(history["train"]).all() and np.isfinite(history["val"]).all()
    assert set(trainer._train_terms) == {"mse", "spec"}
