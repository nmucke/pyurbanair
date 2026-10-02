"""Autoencoder pre-training (plan 02): unit tests.

* **Unit** (``TadpoleAE`` / ``SnapshotDataset``): reconstruction shape, finite KL,
  padding round-trip on a non-divisible grid, ``encode_geometry`` on/off channel
  counts, and the snapshot dataset item shapes + shared-geometry collate.

Gated with ``importorskip('diffusers')`` / ``importorskip('timm')`` -- the
vendored autoencoder's runtime deps -- so envs without them skip cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from neural_surrogates import (
    AutoencoderTrainer,
    SnapshotDataset,
    TadpoleAE,
    snapshot_collate,
)
from torch import nn

STATE_VARS = ("u", "v", "w")


# --------------------------------------------------------------------------- #
# Unit tests on TadpoleAE.
# --------------------------------------------------------------------------- #

CROP = 16  # encoder_crop_size must be a multiple of 16 (encoder downsamples /16)


def _ae(encode_geometry: Any = True, sdf_features: Any = "none", **kw: Any) -> Any:
    kw.setdefault("encoder_crop_size", CROP)
    return TadpoleAE(
        n_state_channels=3,
        size="S",
        encode_geometry=encode_geometry,
        sdf_features=sdf_features,
        sdf_clamp_cells=8,
        **kw,
    )


def _inputs(b: Any = 1, grid: Any = (16, 16, 16)) -> Any:
    state = torch.randn(b, 3, *grid)
    geom = (torch.rand(b, *grid) > 0.2).float()
    return state, geom


def test_reconstruction_shape_and_kl_finite() -> None:
    ae = _ae().eval()
    ae.set_normalization([0, 0, 0], [1, 1, 1])
    state, geom = _inputs()
    with torch.no_grad():
        recon, kl = ae(state, geom, return_kl_element=True)
    assert recon.shape == state.shape
    assert torch.isfinite(kl).all()
    # obstacle cells are exactly zero in the physical reconstruction
    mask = geom.unsqueeze(1)
    assert torch.allclose(recon * (1 - mask), torch.zeros_like(recon))


def test_encoder_crop_size_must_be_multiple_of_16() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        _ae(encoder_crop_size=8)


@pytest.mark.parametrize(  # type: ignore[misc]
    "crop",
    [(16, 32), (16, 32, 8), (16, 32, 32, 64), (16, 32.0, 32)],
)
def test_anisotropic_encoder_crop_size_is_validated(crop: Any) -> None:
    with pytest.raises(ValueError, match="encoder_crop_size"):
        _ae(encoder_crop_size=crop)


def test_anisotropic_tiles_fold_across_the_full_batch() -> None:
    """Each axis uses its own tile extent without padding the thin z axis."""
    ae = _ae(
        encode_geometry=False,
        encoder_crop_size=(16, 32, 32),
        latent_type="mode",
    ).eval()
    state, geom = _inputs(b=2, grid=(16, 32, 64))
    working = ae._assemble_working_input(state, geom, None)
    padded, original = ae._pad_to_crop_multiple(working)
    assert original == (16, 32, 64)
    assert padded.shape[-3:] == (16, 32, 64)

    encoder_inputs = []
    hook = ae.ae.encoder.register_forward_pre_hook(
        lambda _, args: encoder_inputs.append(tuple(args[0].shape))
    )
    with torch.no_grad():
        recon = ae(state, geom)
    hook.remove()

    # B=2, C=3, tiles=1*1*2 are folded into one internal batch.
    assert encoder_inputs == [(12, 1, 16, 32, 32)]
    assert recon.shape == state.shape


def test_padding_round_trip_non_divisible_grid() -> None:
    """A grid not divisible by the crop size is padded internally and cropped
    back, so the reconstruction matches the (odd) input shape."""
    ae = _ae().eval()
    ae.set_normalization([0, 0, 0], [1, 1, 1])
    grid = (20, 24, 18)  # none divisible by 16
    state, geom = _inputs(grid=grid)
    with torch.no_grad():
        recon = ae(state, geom)
    assert recon.shape[2:] == grid


def test_encode_geometry_channel_counts() -> None:
    """Working-space recon/target channel counts track encode_geometry + SDF."""
    state, geom = _inputs()
    for eg, sdf, extra in [(False, "none", 0), (True, "none", 1), (True, "both", 5)]:
        ae = _ae(encode_geometry=eg, sdf_features=sdf).eval()
        ae.set_normalization([0, 0, 0], [1, 1, 1])
        with torch.no_grad():
            recon, target = ae(state, geom, working_space=True)
        assert recon.shape[1] == 3 + extra
        assert target.shape[1] == 3 + extra


def test_encode_decode_passthrough_runs() -> None:
    ae = _ae().eval()
    ae.set_normalization([0, 0, 0], [1, 1, 1])
    state, geom = _inputs()
    with torch.no_grad():
        latent = ae.encode(state, geom, latent_type="mode")
        decoded = ae.decode(latent)
    # decoder returns folded single-channel crops; just assert it runs + is finite
    assert torch.isfinite(decoded).all()


def test_normalization_round_trip() -> None:
    """`_denormalize_state(_normalize_state(x))` recovers x on fluid cells,
    locking down the z-score math independently of the random autoencoder."""
    ae = _ae()
    ae.set_normalization([1.0, -2.0, 0.5], [3.0, 0.5, 2.0])  # non-trivial stats
    state, geom = _inputs(b=2)
    mask = geom.unsqueeze(1)
    x_n = ae._normalize_state(state, mask)
    x_rec = ae._denormalize_state(x_n)
    fluid = mask.expand_as(state) > 0.5
    assert torch.allclose(x_rec[fluid], state[fluid], atol=1e-5)


def test_weight_round_trip_output_parity() -> None:
    """weights.pt reload reproduces outputs bit-for-bit, incl. the installed
    normalization buffers. `latent_type="mode"` makes the forward deterministic
    so a broken buffer save/load would change the output."""
    ae = _ae(latent_type="mode")
    ae.set_normalization([1.0, -2.0, 0.5], [3.0, 0.5, 2.0])
    ae.eval()
    state, geom = _inputs()
    with torch.no_grad():
        out = ae(state, geom)

    fresh = _ae(latent_type="mode")
    fresh.load_state_dict(ae.state_dict())  # carries the normalization buffers
    fresh.eval()
    with torch.no_grad():
        out_reloaded = fresh(state, geom)
    assert torch.allclose(out, out_reloaded, atol=1e-6)
    # and the normalization buffers actually travelled (not left at identity)
    assert torch.allclose(fresh.state_mean, torch.tensor([1.0, -2.0, 0.5]))
    assert torch.allclose(fresh.state_std, torch.tensor([3.0, 0.5, 2.0]))


# --------------------------------------------------------------------------- #
# Unit tests on SnapshotDataset.
# --------------------------------------------------------------------------- #

NZ, NY, NX, T = 16, 16, 16, 4


def _write_dataset(root: Path, *, splits: Any = None) -> None:
    splits = splits or {"train": 2, "val": 1}
    rng = np.random.default_rng(0)
    for split, n in splits.items():
        (root / "state" / split).mkdir(parents=True, exist_ok=True)
        blank = np.zeros((NZ, NY, NX), "f4")
        blank[0] = 1.0  # bottom layer is obstacle (blanking=1)
        for i in range(n):
            state: dict[str, tuple[tuple[str, ...], Any]] = {
                v: (
                    ("time", "z", "y", "x"),
                    rng.standard_normal((T, NZ, NY, NX)).astype("f4"),
                )
                for v in STATE_VARS
            }
            state["blanking"] = (("z", "y", "x"), blank)
            xr.Dataset(
                state,
                coords=dict(
                    time=np.arange(T) * 1.0,
                    z=np.arange(NZ),
                    y=np.arange(NY),
                    x=np.arange(NX),
                ),
            ).to_netcdf(root / "state" / split / f"sample_{i:04d}.nc")


def test_snapshot_dataset_items_and_collate(tmp_path: Any) -> None:
    root = tmp_path / "data"
    _write_dataset(root)
    ds = SnapshotDataset(root, "train", sdf_features="both", sdf_clamp_cells=8)
    # every (traj, t) is a sample: 2 trajectories x T time steps
    assert len(ds) == 2 * T
    item = ds[0]
    assert item["state"].shape == (3, NZ, NY, NX)
    assert item["geometry"].shape == (NZ, NY, NX)
    assert item["geom_features"].shape == (4, NZ, NY, NX)
    # single-geometry split shares one mask object -> collate ships it once
    batch = snapshot_collate([ds[0], ds[1], ds[2]])
    assert batch["state"].shape == (3, 3, NZ, NY, NX)
    assert batch["geometry"].shape == (1, NZ, NY, NX)
    assert batch["geom_features"].shape == (1, 4, NZ, NY, NX)


def test_snapshot_dataset_time_stride(tmp_path: Any) -> None:
    root = tmp_path / "data"
    _write_dataset(root)
    ds = SnapshotDataset(root, "train", time_stride=2)
    # ceil(T / 2) = 2 samples per trajectory
    assert len(ds) == 2 * 2


def test_snapshot_trajectory_batch_sampler(tmp_path: Any) -> None:
    """SnapshotDataset exposes `sample_index` + `grid_shape`, so the
    TrajectoryBatchSampler (multi-geometry batching) works over it."""
    from neural_surrogates import TrajectoryBatchSampler

    root = tmp_path / "data"
    _write_dataset(root)
    ds = SnapshotDataset(root, "train")
    sampler = TrajectoryBatchSampler(ds, batch_size=2, shuffle=False)
    batches = list(sampler)
    assert batches, "sampler yielded no batches"
    # every batch draws from a single trajectory (one grid/geometry)
    for batch in batches:
        trajs = {ds.sample_index[i][0] for i in batch}
        assert len(trajs) == 1, f"batch mixes trajectories {trajs}"


def test_snapshot_random_crop(tmp_path: Any) -> None:
    root = tmp_path / "data"
    _write_dataset(root)
    ds = SnapshotDataset(root, "train", random_crop_size=8, sdf_features="sdf")
    item = ds[0]
    assert item["state"].shape == (3, 8, 8, 8)
    assert item["geometry"].shape == (8, 8, 8)
    assert item["geom_features"].shape == (1, 8, 8, 8)
    # per-sample crops break geometry sharing -> collate stacks per-sample
    batch = snapshot_collate([ds[0], ds[1]])
    assert batch["geometry"].shape == (2, 8, 8, 8)


def test_snapshot_random_crop_equals_full_field_slice(tmp_path: Any) -> None:
    """The lazy-read crop (M5) must equal the corresponding slice of the
    full-field item for the same crop origin -- i.e. reading only the crop
    changes I/O, not values."""
    root = tmp_path / "data"
    _write_dataset(root)
    full = SnapshotDataset(root, "train", sdf_features="sdf")
    crop = SnapshotDataset(root, "train", random_crop_size=8, sdf_features="sdf")

    # Fix the origin: crop.__getitem__ draws one torch.randint per spatial dim.
    torch.manual_seed(1234)
    c_item = crop[0]
    torch.manual_seed(1234)
    sl_z, sl_y, sl_x = crop._crop_slices(tuple(full.geometry_for(0).shape))

    f_item = full[0]
    assert torch.equal(c_item["state"], f_item["state"][:, sl_z, sl_y, sl_x])
    assert torch.equal(c_item["geometry"], f_item["geometry"][sl_z, sl_y, sl_x])
    assert torch.equal(
        c_item["geom_features"], f_item["geom_features"][:, sl_z, sl_y, sl_x]
    )


# --------------------------------------------------------------------------- #
# AutoencoderTrainer device-geometry cache (M4).
# --------------------------------------------------------------------------- #


class _StubAE(nn.Module):
    """Minimal stand-in exposing the attributes AutoencoderTrainer reads."""

    n_state_channels = 3
    encode_geometry = True
    n_geom_feature_channels = 1

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))


def _make_ae_trainer(model: Any) -> Any:
    from torch.utils.data import DataLoader

    dummy = DataLoader([0, 1], batch_size=1)
    return AutoencoderTrainer(
        model=model,
        train_loader=dummy,
        val_loader=dummy,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
    )


def test_ae_trainer_geometry_cache_busts_on_change() -> None:
    """A shared geometry is uploaded once and reused across content-equal
    batches, but a genuinely different geometry refreshes the device cache; a
    per-sample (crop) batch bypasses the cache entirely."""
    torch.manual_seed(0)
    trainer = _make_ae_trainer(_StubAE())
    grid = (4, 4, 4)
    geom_a = (torch.rand(*grid) > 0.3).float()
    feat_a = torch.randn(1, *grid)
    batch_a = {
        "state": torch.randn(2, 3, *grid),
        "geometry": geom_a.unsqueeze(0),  # shared: (1, *grid)
        "geom_features": feat_a.unsqueeze(0),  # (1, C, *grid)
    }
    _, geometry, features = trainer._prepare_ae_batch(batch_a)
    assert geometry.shape == (2, *grid)
    assert features.shape == (2, 1, *grid)
    cached_geom = trainer._geometry
    torch.testing.assert_close(trainer._geometry, geom_a)

    # Content-equal but distinct object (what DataLoader workers produce):
    # revalidate without replacing the cached device tensors.
    batch_a2 = dict(batch_a)
    batch_a2["geometry"] = batch_a["geometry"].clone()
    batch_a2["geom_features"] = batch_a["geom_features"].clone()
    trainer._prepare_ae_batch(batch_a2)
    assert trainer._geometry is cached_geom

    # A different geometry busts the cache.
    geom_b = (torch.rand(*grid) > 0.7).float()
    assert not torch.equal(geom_a, geom_b)
    batch_b = {
        "state": torch.randn(2, 3, *grid),
        "geometry": geom_b.unsqueeze(0),
        "geom_features": torch.randn(1, *grid).unsqueeze(0),
    }
    trainer._prepare_ae_batch(batch_b)
    assert trainer._geometry is not cached_geom
    torch.testing.assert_close(trainer._geometry, geom_b)

    # A per-sample (crop) batch passes through without touching the cache.
    sentinel = trainer._geometry
    geom_crop = (torch.rand(2, *grid) > 0.5).float()
    batch_crop = {
        "state": torch.randn(2, 3, *grid),
        "geometry": geom_crop,  # (B, *grid), leading dim != 1
        "geom_features": torch.randn(2, 1, *grid),
    }
    _, geom_out, feat_out = trainer._prepare_ae_batch(batch_crop)
    assert geom_out.shape == (2, *grid)
    assert feat_out.shape == (2, 1, *grid)
    torch.testing.assert_close(geom_out.cpu(), geom_crop)
    assert trainer._geometry is sentinel  # cache untouched by the crop batch
