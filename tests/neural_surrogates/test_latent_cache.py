"""Raw-latent cache for latent-generator training (``datasets/latent_cache.py``).

On CPU smoke shapes via the shared fixtures in ``_latent_generator_fixtures``:
the cache holds exactly what the frozen encoder gives every frame, its train
statistics match ``compute_latent_normalization``, it resumes and refuses like
the rechunk cache, and :class:`LatentCacheDataset` batches through a loader.
"""

from __future__ import annotations

import json
import os
import pickle
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


def _mtimes(root: Path) -> dict[str, int]:
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns
        for p in root.rglob("*")
        # The manifest and the train stats are rewritten on every call.
        if p.is_file()
        and p.name not in ("manifest.json", ".latent-cache.lock", "latent_stats.npz")
    }


@pytest.fixture(scope="module", params=["fold", "branch"])
def prepared(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(request.param)
    ae_dir, data_dir = fixture_inputs(tmp, geometry=request.param)
    model = _model(ae_dir)
    datasets = _datasets(data_dir, request.param)
    root = prepare_latent_cache(model, datasets, tmp / "cache", device="cpu")
    return request.param, model, data_dir, datasets, root


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
    for path in sorted((data_dir / "state").glob("*/*.nc")):
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        chunked = {"zlib": True, "chunksizes": (4, *ds["u"].shape[1:])}
        ds.to_netcdf(path, encoding={v: chunked for v in STATE_VARS})
    datasets = _datasets(data_dir, geometry)
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
    first = np.load(root / "train" / "sample_0001.npy")
    stats = load_latent_stats(root)
    assert prepare_latent_cache(model, datasets, root, device="cpu") == root
    assert _mtimes(root) == before
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
