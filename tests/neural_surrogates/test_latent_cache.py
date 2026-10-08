"""Raw-latent cache for latent-generator training (``datasets/latent_cache.py``).

On CPU smoke shapes via the shared fixtures in ``_latent_generator_fixtures``:
the cache holds exactly what the frozen encoder gives every frame, its train
statistics match ``compute_latent_normalization``, it resumes and refuses like
the rechunk cache, a complete cache is validated without the lock (shared,
read-only), and :class:`LatentCacheDataset` batches through a loader.
"""

from __future__ import annotations

import fcntl
import json
import os
import pickle
import stat
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
    LatentCacheDataset,
    SnapshotDataset,
    SnapshotHistoryDataset,
    TadpoleLatentGenerator,
    TrajectoryBatchSampler,
    load_latent_stats,
    prepare_latent_cache,
)
from torch.utils.data import DataLoader, default_collate

from tests.neural_surrogates._latent_generator_fixtures import (
    HP,
    NET,
    PARAM_VARS,
    STATE_VARS,
    T_LEN,
    C,
    P,
    fixture_inputs,
    write_history_dataset,
)

SDF = {"fold": "both", "branch": "sdf"}


def _model(ae_dir: Path) -> TadpoleLatentGenerator:
    return TadpoleLatentGenerator(
        n_state_channels=C,
        n_params=P,
        param_history_steps=HP,
        pretrained_ae_dir=str(ae_dir),
        **NET,
    )


def _datasets(data_dir: Path, geometry: str) -> dict[str, SnapshotHistoryDataset]:
    return {
        split: SnapshotHistoryDataset(
            data_dir,
            split,
            state_vars=STATE_VARS,
            param_vars=PARAM_VARS,
            param_history_steps=HP,
            sdf_features=SDF[geometry],
            sdf_clamp_cells=8.0,
        )
        for split in ("train", "val")
    }


def _every_frame(data_dir: Path, split: str, geometry: str) -> SnapshotDataset:
    """Every saved frame of a split, as the history dataset reads it."""
    return SnapshotDataset(
        data_dir,
        split,
        state_vars=STATE_VARS,
        sdf_features=SDF[geometry],
        sdf_clamp_cells=8.0,
    )


def _batch(item: dict[str, Any]) -> tuple[Any, Any, Any]:
    features = item.get("geom_features")
    return (
        item["state"][None],
        item["geometry"][None],
        None if features is None else features[None],
    )


def _mtimes(root: Path, every: bool = False) -> dict[str, int]:
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns
        for p in root.rglob("*")
        # Preparing rewrites the manifest and the train stats.
        if p.is_file()
        and (
            every
            or p.name not in ("manifest.json", ".latent-cache.lock", "latent_stats.npz")
        )
    }


@pytest.fixture(scope="module", params=["fold", "branch"])
def prepared(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(request.param)
    ae_dir, data_dir = fixture_inputs(tmp, geometry=request.param)
    model = _model(ae_dir)
    datasets = _datasets(data_dir, request.param)
    root = prepare_latent_cache(model, datasets, tmp / "cache", device="cpu")
    return request.param, model, data_dir, datasets, root


def _rewrite_sources(data_dir: Path, dims: tuple[str, ...], step: int) -> None:
    """Store the state variables with ``dims`` order in zlib ``time`` chunks of
    ``step`` frames."""
    for path in sorted((data_dir / "state").glob("*/*.nc")):
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        ds = ds.transpose(*dims)
        sizes = ds["u"].sizes
        chunks = tuple(step if d == "time" else sizes[d] for d in dims)
        chunked = {"zlib": True, "chunksizes": chunks}
        ds.to_netcdf(path, encoding={v: chunked for v in STATE_VARS})


def test_cache_equals_the_frozen_encoder_on_every_frame(prepared):
    geometry, model, data_dir, _, root = prepared
    for split in ("train", "val"):
        frames = _every_frame(data_dir, split, geometry)
        latents = {}
        for idx, (traj, t) in enumerate(frames.sample_index):
            stem = frames._state_files[traj].stem
            if traj not in latents:
                latents[traj] = np.load(root / split / f"{stem}.npy")
                assert latents[traj].shape[0] == frames._traj_lengths[traj]
            state, mask, features = _batch(frames[idx])
            with model._autocast_off(torch.device("cpu")):
                z_raw, geom_raw, branch_cond, *_ = model._encode_raw(
                    state, mask, features
                )
                _, branch_alone, geom_alone = model._geometry_conditioning(
                    mask, features, torch.float32
                )
            assert torch.equal(torch.from_numpy(latents[traj][t]), z_raw[0])
            geom = torch.from_numpy(np.load(root / split / f"{stem}.geom.npy"))
            expected = branch_alone if geometry == "branch" else geom_alone
            assert (geom_raw is None) == (geometry == "branch")
            assert torch.equal(geom, expected[0])
            assert torch.equal(geom, (branch_cond if geom_raw is None else geom_raw)[0])


def test_time_chunked_sources_are_read_in_whole_chunks(prepared, tmp_path):
    """Sources stored in zlib time chunks (like the real corpora) are read in
    whole chunks and give the same latents."""
    geometry, model, _, _, root = prepared
    data_dir = write_history_dataset(tmp_path / "data")  # the fixture's data
    _rewrite_sources(data_dir, ("time", "z", "y", "x"), step=4)
    datasets = _datasets(data_dir, geometry)
    cache = prepare_latent_cache(model, datasets, tmp_path / "cache", device="cpu")
    for path in sorted(root.glob("*/*.npy")):
        np.testing.assert_array_equal(
            np.load(cache / path.relative_to(root)), np.load(path)
        )


def test_sources_with_time_not_first_give_the_same_latents(prepared, tmp_path):
    """``SnapshotDataset`` reads frames by name (``isel(time=t)``), so must the
    cache: a source whose ``time`` dim isn't first gives the same latents."""
    geometry, model, _, _, root = prepared
    data_dir = write_history_dataset(tmp_path / "data")  # the fixture's data
    _rewrite_sources(data_dir, ("z", "time", "y", "x"), step=4)
    datasets = _datasets(data_dir, geometry)
    frames = _every_frame(data_dir, "train", geometry)
    with xr.open_dataset(frames._state_files[0]) as ds:
        assert ds["u"].dims[0] == "z"
    cache = prepare_latent_cache(model, datasets, tmp_path / "cache", device="cpu")
    for path in sorted(root.glob("*/*.npy")):
        np.testing.assert_array_equal(
            np.load(cache / path.relative_to(root)), np.load(path)
        )


def test_train_stats_equal_compute_latent_normalization(prepared):
    geometry, model, data_dir, _, root = prepared
    mean, std = load_latent_stats(root)
    assert mean.dtype == std.dtype == np.float64
    assert mean.shape == (model.working_latent_dim,)
    frames = _every_frame(data_dir, "train", geometry)
    ref_mean, ref_std = model.compute_latent_normalization(
        _batch(frames[i]) for i in range(len(frames))
    )
    model.set_latent_normalization(mean, std)
    assert torch.allclose(model.latent_mean, ref_mean, rtol=1e-5, atol=1e-6)
    assert torch.allclose(model.latent_std, ref_std, rtol=1e-5, atol=1e-6)


def test_dataset_items_collate_and_batch_through_loaders(prepared):
    geometry, model, _, datasets, root = prepared
    base = datasets["train"]
    ds = LatentCacheDataset(root, base)
    assert len(ds) == len(base) and ds.sample_index == base.sample_index
    assert ds.grid_shape(0) == base.grid_shape(0)

    latent_grid = model.latent_grid_for(base.grid_shape(0))
    g = model.geom_cond_dim
    item = ds[0]
    traj, t = base.sample_index[0]
    assert item["latent"].dtype == torch.float32
    assert item["latent"].shape == (model.state_latent_dim, *latent_grid)
    assert item["geom"].shape == (g, *latent_grid)
    assert item["geom"] is ds[1]["geom"]  # one tensor per trajectory
    assert torch.equal(item["params_hist"], base.params_hist_for(traj, t))
    stem = base._state_files[traj].stem
    assert torch.equal(
        item["latent"], torch.from_numpy(np.load(root / "train" / f"{stem}.npy")[t])
    )
    # Workers get no open memory maps.
    assert ds._open and pickle.loads(pickle.dumps(ds))._open is None

    batch = default_collate([ds[0], ds[1]])
    assert batch["latent"].shape == (2, model.state_latent_dim, *latent_grid)
    assert batch["geom"].shape == (2, g, *latent_grid)
    assert batch["params_hist"].shape == (2, HP, P)
    loaders = [
        DataLoader(ds, batch_size=2, num_workers=0),
        DataLoader(
            ds,
            batch_sampler=TrajectoryBatchSampler(
                ds, batch_size=2, shuffle=True  # type: ignore[arg-type]
            ),
        ),
    ]
    for loader in loaders:
        seen = 0
        for batch in loader:
            assert batch["latent"].shape[1:] == (model.state_latent_dim, *latent_grid)
            assert batch["geom"].shape[0] == batch["latent"].shape[0]
            seen += batch["latent"].shape[0]
        assert seen == len(base)


def test_second_call_reuses_and_an_interrupted_file_is_redone(tmp_path):
    ae_dir, data_dir = fixture_inputs(tmp_path)
    model = _model(ae_dir)
    datasets = _datasets(data_dir, "fold")
    root = prepare_latent_cache(model, datasets, tmp_path / "cache", device="cpu")
    before = _mtimes(root)
    every = _mtimes(root, every=True)
    first = np.load(root / "train" / "sample_0001.npy")
    stats = load_latent_stats(root)
    # A complete cache is validated without writing anything.
    assert prepare_latent_cache(model, datasets, root, device="cpu") == root
    assert _mtimes(root, every=True) == every
    for new, old in zip(load_latent_stats(root), stats):
        np.testing.assert_array_equal(new, old)

    # Simulate an interruption while encoding train/sample_0001.
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["train/sample_0001"] = None
    manifest["complete"] = False
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "train" / "sample_0001.npy").write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        load_latent_stats(root)
    with pytest.raises(ValueError, match="incomplete"):
        LatentCacheDataset(root, datasets["train"])
    prepare_latent_cache(model, datasets, root, device="cpu")
    after = _mtimes(root)
    redone = {"train/sample_0001.npy", "train/sample_0001.geom.npy"}
    redone |= {"train/sample_0001.sums.npz"}
    assert {k for k in before if before[k] != after[k]} == redone
    np.testing.assert_array_equal(np.load(root / "train" / "sample_0001.npy"), first)
    for new, old in zip(load_latent_stats(root), stats):
        np.testing.assert_array_equal(new, old)

    # A completed file that vanished is an error, not a silent re-encode.
    (root / "val" / "sample_0000.geom.npy").unlink()
    with pytest.raises(ValueError, match="changed or is missing"):
        prepare_latent_cache(model, datasets, root, device="cpu")


def test_changed_ae_source_or_setting_is_refused(tmp_path):
    ae_dir, _ = fixture_inputs(tmp_path)
    data_dir = write_history_dataset(tmp_path / "own_data")
    model = _model(ae_dir)
    datasets = _datasets(data_dir, "fold")
    root = prepare_latent_cache(model, datasets, tmp_path / "cache", device="cpu")

    fingerprint = model.ae_fingerprint
    model.ae_fingerprint = None
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_latent_cache(model, datasets, root, device="cpu")
    model.ae_fingerprint = "0" * 64
    with pytest.raises(ValueError, match="new output_root"):
        prepare_latent_cache(model, datasets, root, device="cpu")
    model.ae_fingerprint = fingerprint

    ae_kwargs = model.ae_kwargs
    model.ae_kwargs = {**ae_kwargs, "halo_size": 99}
    with pytest.raises(ValueError, match="new output_root"):
        prepare_latent_cache(model, datasets, root, device="cpu")
    model.ae_kwargs = ae_kwargs
    other = _datasets(data_dir, "fold")
    other["val"].sdf_clamp_cells = 4.0
    with pytest.raises(ValueError, match="new output_root"):
        prepare_latent_cache(model, other, root, device="cpu")

    source = data_dir / "state" / "train" / "sample_0000.nc"
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    changed = _datasets(data_dir, "fold")
    with pytest.raises(ValueError, match="new output_root"):
        prepare_latent_cache(model, changed, root, device="cpu")
    with pytest.raises(ValueError, match="not built from this"):
        LatentCacheDataset(root, changed["train"])
    # The unchanged split still loads.
    LatentCacheDataset(root, changed["val"])


def test_complete_cache_validates_while_the_lock_is_held(tmp_path):
    """Training runs validating one complete cache don't contend for the lock;
    preparing an incomplete one still does."""
    ae_dir, data_dir = fixture_inputs(tmp_path)
    model = _model(ae_dir)
    datasets = _datasets(data_dir, "fold")
    root = prepare_latent_cache(model, datasets, tmp_path / "cache", device="cpu")
    with (root / ".latent-cache.lock").open("a") as other:  # another preparer
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert prepare_latent_cache(model, datasets, root, device="cpu") == root
        LatentCacheDataset(root, datasets["train"])
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["files"]["val/sample_0000"] = None
        manifest["complete"] = False
        (root / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(RuntimeError, match="Another process"):
            prepare_latent_cache(model, datasets, root, device="cpu")


def test_read_only_complete_cache_validates_and_loads(tmp_path):
    ae_dir, data_dir = fixture_inputs(tmp_path)
    model = _model(ae_dir)
    datasets = _datasets(data_dir, "fold")
    root = prepare_latent_cache(model, datasets, tmp_path / "cache", device="cpu")
    every = _mtimes(root, every=True)
    paths = [root, *root.rglob("*")]
    modes = {p: p.stat().st_mode for p in paths}
    try:
        for p in paths:
            p.chmod(modes[p] & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        assert prepare_latent_cache(model, datasets, root, device="cpu") == root
        load_latent_stats(root)
        ds = LatentCacheDataset(root, datasets["train"])
        assert ds[0]["latent"].shape[0] == model.state_latent_dim
    finally:
        for p in reversed(paths):
            p.chmod(modes[p])
    assert _mtimes(root, every=True) == every


def test_chmod_keeps_the_cache_but_a_rewrite_does_not(tmp_path):
    """Fingerprints are size + mtime: a permission change (ctime only) of a
    source or a cache file keeps the cache valid; rewriting a file doesn't."""
    ae_dir, _ = fixture_inputs(tmp_path)
    data_dir = write_history_dataset(tmp_path / "own_data")
    model = _model(ae_dir)
    datasets = _datasets(data_dir, "fold")
    root = prepare_latent_cache(model, datasets, tmp_path / "cache", device="cpu")
    every = _mtimes(root, every=True)
    source = data_dir / "state" / "train" / "sample_0000.nc"
    cached = root / "val" / "sample_0000.npy"
    for path in (source, cached):
        path.chmod(path.stat().st_mode & ~stat.S_IROTH)
    reread = _datasets(data_dir, "fold")
    assert prepare_latent_cache(model, reread, root, device="cpu") == root
    LatentCacheDataset(root, reread["train"])
    assert _mtimes(root, every=True) == every

    cached.write_bytes(cached.read_bytes())  # a rewrite: new mtime
    info = cached.stat()  # (explicitly: some filesystems keep whole seconds)
    os.utime(cached, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="changed or is missing"):
        prepare_latent_cache(model, reread, root, device="cpu")


def test_non_finite_latents_raise_right_after_their_trajectory(tmp_path, monkeypatch):
    """NaN latents of the val trajectory raise naming its file, before the
    cache is marked complete; the train trajectories stay completed."""
    ae_dir, data_dir = fixture_inputs(tmp_path)
    model = _model(ae_dir)
    datasets = _datasets(data_dir, "fold")
    encode = model._encode_raw
    calls = []

    def nan_for_val(*args, **kwargs):
        z_raw, *rest = encode(*args, **kwargs)
        calls.append(None)
        if len(calls) > 2 * T_LEN:  # past the two train trajectories
            z_raw = torch.full_like(z_raw, float("nan"))
        return (z_raw, *rest)

    monkeypatch.setattr(model, "_encode_raw", nan_for_val)
    root = tmp_path / "cache"
    with pytest.raises(ValueError, match="Non-finite latents.*val/sample_0000"):
        prepare_latent_cache(model, datasets, root, device="cpu")
    assert len(calls) == 2 * T_LEN + 1
    manifest = json.loads((root / "manifest.json").read_text())
    assert not manifest["complete"]
    assert manifest["files"]["val/sample_0000"] is None
    assert manifest["files"]["train/sample_0001"] is not None


def test_old_version_and_missing_stats_are_not_served(tmp_path):
    ae_dir, data_dir = fixture_inputs(tmp_path)
    model = _model(ae_dir)
    datasets = _datasets(data_dir, "fold")
    root = prepare_latent_cache(model, datasets, tmp_path / "cache", device="cpu")
    stats = root / "latent_stats.npz"
    stats.unlink()  # the locked path rebuilds it from the per-trajectory sums
    assert prepare_latent_cache(model, datasets, root, device="cpu") == root
    assert stats.is_file()
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["version"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="older version"):
        prepare_latent_cache(model, datasets, root, device="cpu")
