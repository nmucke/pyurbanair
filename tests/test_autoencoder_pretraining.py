"""Autoencoder pre-training (plan 02): unit + end-to-end smoke tests.

Two layers:

* **Unit** (``TadpoleAE`` / ``SnapshotDataset``): reconstruction shape, finite KL,
  padding round-trip on a non-divisible grid, ``encode_geometry`` on/off channel
  counts, and the snapshot dataset item shapes + shared-geometry collate.
* **End-to-end** (``pretrain_autoencoder.run``): compose the Hydra config, run 2
  epochs on a tiny fixture dataset, and assert the exported ``model_dir`` has the
  expected artifacts and that ``weights.pt`` reloads into a fresh ``TadpoleAE``.

Gated with ``importorskip('diffusers')`` / ``importorskip('timm')`` -- the
vendored autoencoder's runtime deps -- so envs without them skip cleanly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from neural_surrogates import SnapshotDataset, TadpoleAE, snapshot_collate

_WORKTREE = Path(__file__).resolve().parents[1]
_CONF = _WORKTREE / "conf"
_SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "pretrain_autoencoder.py"

STATE_VARS = ("u", "v", "w")


def _load_pretrain_run():
    spec = importlib.util.spec_from_file_location("pretrain_ae_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


# --------------------------------------------------------------------------- #
# Unit tests on TadpoleAE.
# --------------------------------------------------------------------------- #

CROP = 16  # encoder_crop_size must be a multiple of 16 (encoder downsamples /16)


def _ae(encode_geometry=True, sdf_features="none", **kw):
    kw.setdefault("encoder_crop_size", CROP)
    return TadpoleAE(
        n_state_channels=3,
        size="S",
        encode_geometry=encode_geometry,
        sdf_features=sdf_features,
        sdf_clamp_cells=8,
        **kw,
    )


def _inputs(b=1, grid=(16, 16, 16)):
    state = torch.randn(b, 3, *grid)
    geom = (torch.rand(b, *grid) > 0.2).float()
    return state, geom


def test_reconstruction_shape_and_kl_finite():
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


def test_encoder_crop_size_must_be_multiple_of_16():
    with pytest.raises(ValueError, match="multiple of 16"):
        _ae(encoder_crop_size=8)


def test_padding_round_trip_non_divisible_grid():
    """A grid not divisible by the crop size is padded internally and cropped
    back, so the reconstruction matches the (odd) input shape."""
    ae = _ae().eval()
    ae.set_normalization([0, 0, 0], [1, 1, 1])
    grid = (20, 24, 18)  # none divisible by 16
    state, geom = _inputs(grid=grid)
    with torch.no_grad():
        recon = ae(state, geom)
    assert recon.shape[2:] == grid


def test_encode_geometry_channel_counts():
    """Working-space recon/target channel counts track encode_geometry + SDF."""
    state, geom = _inputs()
    for eg, sdf, extra in [(False, "none", 0), (True, "none", 1), (True, "both", 5)]:
        ae = _ae(encode_geometry=eg, sdf_features=sdf).eval()
        ae.set_normalization([0, 0, 0], [1, 1, 1])
        with torch.no_grad():
            recon, target = ae(state, geom, working_space=True)
        assert recon.shape[1] == 3 + extra
        assert target.shape[1] == 3 + extra


def test_encode_decode_passthrough_runs():
    ae = _ae().eval()
    ae.set_normalization([0, 0, 0], [1, 1, 1])
    state, geom = _inputs()
    with torch.no_grad():
        latent = ae.encode(state, geom, latent_type="mode")
        decoded = ae.decode(latent)
    # decoder returns folded single-channel crops; just assert it runs + is finite
    assert torch.isfinite(decoded).all()


def test_normalization_round_trip():
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


def test_weight_round_trip_output_parity():
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


def _write_dataset(root: Path, *, splits=None) -> None:
    splits = splits or {"train": 2, "val": 1}
    rng = np.random.default_rng(0)
    for split, n in splits.items():
        (root / "state" / split).mkdir(parents=True, exist_ok=True)
        blank = np.zeros((NZ, NY, NX), "f4")
        blank[0] = 1.0  # bottom layer is obstacle (blanking=1)
        for i in range(n):
            state = {
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


def test_snapshot_dataset_items_and_collate(tmp_path):
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


def test_snapshot_dataset_time_stride(tmp_path):
    root = tmp_path / "data"
    _write_dataset(root)
    ds = SnapshotDataset(root, "train", time_stride=2)
    # ceil(T / 2) = 2 samples per trajectory
    assert len(ds) == 2 * 2


def test_snapshot_trajectory_batch_sampler(tmp_path):
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


def test_snapshot_random_crop(tmp_path):
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


# --------------------------------------------------------------------------- #
# End-to-end: compose pretrain_autoencoder.yaml + run the script.
# --------------------------------------------------------------------------- #


def _shrink_for_cpu(cfg) -> None:
    cfg.architecture.encoder_crop_size = CROP
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.trainer.num_epochs = 2
    cfg.trainer.device = "cpu"
    cfg.trainer.amp = False
    cfg.trainer.compile_model = False
    cfg.trainer.patience = None
    cfg.trainer.lr_warmup_epochs = None
    cfg.trainer.cudnn_benchmark = False
    cfg.trainer.tf32 = False
    cfg.trainer.resume = False


def test_pretrain_end_to_end(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)

    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        cfg = compose(
            config_name="neural_surrogate/pretrain_autoencoder",
            overrides=[
                f"dataset.root_dir={data_dir}",
                "model_name=tadpole_ae_test",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    _shrink_for_cpu(cfg)

    monkeypatch.chdir(tmp_path)
    _load_pretrain_run()(cfg)

    out = tmp_path / "model_weights" / "tadpole_ae_test"
    assert (out / "weights.pt").exists()
    assert (out / "encoder.pt").exists()
    assert (out / "decoder.pt").exists()
    assert (out / "config.yaml").exists()
    assert (out / "metrics.csv").exists()

    # the documented per-term breakdown lands in metrics.csv
    import csv

    with (out / "metrics.csv").open() as f:
        header = next(csv.reader(f))
    for col in ("train_recon", "train_geom", "train_kl", "val_recon", "val_kl"):
        assert col in header, f"missing metrics column {col!r} in {header}"

    saved = OmegaConf.load(out / "config.yaml")
    assert saved.architecture._target_.split(".")[-1] == "TadpoleAE"

    # weights.pt reloads into a fresh TadpoleAE (built from the saved config) and
    # the exported artifact runs — a finite, correctly-shaped reconstruction on a
    # fixed input (deterministic mode). Exact buffer save/load parity (original vs
    # reloaded) is locked down by test_weight_round_trip_output_parity.
    from hydra.utils import instantiate

    fresh = instantiate(saved.architecture, n_state_channels=len(STATE_VARS))
    fresh.load_state_dict(torch.load(out / "weights.pt"))
    assert isinstance(fresh, TadpoleAE)
    fresh.ae.latent_type = "mode"  # governs the AE's sampling -> deterministic
    fresh.eval()
    state = torch.randn(1, len(STATE_VARS), NZ, NY, NX)
    geom = (torch.rand(1, NZ, NY, NX) > 0.2).float()
    with torch.no_grad():
        recon = fresh(state, geom)
    assert recon.shape == state.shape
    assert torch.isfinite(recon).all()
