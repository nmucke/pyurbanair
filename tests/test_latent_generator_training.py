"""Plan 07 phase 2B: ``LatentFlowMatchingTrainer`` + ``train_latent_generator.py``.

End-to-end on CPU smoke shapes (AE size S, crop 16, grid 16x16x32, Hp=3, P=2,
2 epochs, batch 2) via the shared fixtures in ``_latent_generator_fixtures``:

* the export is complete and self-contained (``config.yaml`` stamped
  ``skip_pretrained_load: true`` + inline ``ae_kwargs`` + the ``generator``
  block; ``weights.pt`` reloads strictly WITHOUT the AE dir or the data and
  ``sample()`` runs on the reloaded model);
* the velocity net's zero-init output projection actually moves while the
  frozen AE's parameters stay byte-identical to the export;
* validation is deterministic (two passes agree), a shuffling val loader and
  an empty loader are rejected, and the optimizer may not hold AE parameters;
* resume continues from the next epoch with the latent statistics intact and
  the ``latent_stats.pt`` cache reused (no recomputation);
* the latent-attention budget and a ``state_vars`` mismatch against the AE
  fail before training.

Gated with ``importorskip`` on the vendored Tadpole runtime deps.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import xarray as xr

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from hydra.utils import instantiate
from neural_surrogates import (
    LatentFlowMatchingTrainer,
    SnapshotHistoryDataset,
    TadpoleLatentGenerator,
    TrajectoryBatchSampler,
    snapshot_history_collate,
)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from tests._latent_generator_fixtures import (
    DT,
    GRID,
    HP,
    NET,
    PARAM_VARS,
    STATE_VARS,
    C,
    P,
    compose_generator_cfg,
    fixture_inputs,
    load_run,
    train_tiny_generator,
    write_history_dataset,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(3)
    yield


def _metrics_rows(model_dir: Path) -> list[dict]:
    with (model_dir / "metrics.csv").open() as f:
        return list(csv.DictReader(f))


def _dataset(data_dir: Path, split: str, **kw: Any) -> SnapshotHistoryDataset:
    kwargs: dict[str, Any] = dict(
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        param_history_steps=HP,
        sdf_features="both",
        sdf_clamp_cells=8.0,
    )
    kwargs.update(kw)
    return SnapshotHistoryDataset(data_dir, split, **kwargs)


def _loader(ds, **kw) -> DataLoader:
    kw.setdefault("batch_size", 2)
    return DataLoader(ds, collate_fn=snapshot_history_collate, **kw)


def _model(ae_dir: Path, **kw) -> TadpoleLatentGenerator:
    return TadpoleLatentGenerator(
        n_state_channels=C,
        n_params=P,
        param_history_steps=HP,
        pretrained_ae_dir=str(ae_dir),
        **{**NET, **kw},
    )


def _trainer(model, train_loader, val_loader, **kw) -> LatentFlowMatchingTrainer:
    params = [p for p in model.parameters() if p.requires_grad]
    kwargs = dict(
        optimizer=torch.optim.AdamW(params, lr=1e-3),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
    )
    kwargs.update(kw)
    return LatentFlowMatchingTrainer(model, train_loader, val_loader, **kwargs)


# --------------------------------------------------------------------------- #
# (b) + (c) end-to-end smoke: artifacts, self-contained reload, real update.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spatial_mode,geometry", [("local", "fold"), ("halo", "branch")]
)
def test_train_end_to_end_exports_self_contained_generator(
    tmp_path, spatial_mode, geometry
):
    model_dir = train_tiny_generator(
        tmp_path, spatial_mode=spatial_mode, geometry=geometry
    )
    for name in ("config.yaml", "weights.pt", "checkpoint.pt", "metrics.csv"):
        assert (model_dir / name).exists(), name
    assert (model_dir / "latent_stats.pt").exists()
    rows = _metrics_rows(model_dir)
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert all(float(r["val_loss"]) > 0 for r in rows)

    saved = OmegaConf.load(model_dir / "config.yaml")
    arch = saved.architecture
    assert arch._target_.split(".")[-1] == "TadpoleLatentGenerator"
    assert arch.skip_pretrained_load is True
    assert arch.pretrained_ae_dir is None
    assert arch.ae_kwargs.pretrained == "none"
    assert arch.ae_kwargs.latent_type == "mode"
    assert arch.ae_kwargs.spatial_mode == spatial_mode
    assert arch.hidden_size == C * 256
    # Stamped alongside hidden_size: both shape the velocity net's tensors.
    assert arch.mlp_ratio == 4 and arch.normalize is True
    assert saved.dataset.sdf_features == ("both" if geometry == "fold" else "sdf")
    assert float(saved.dataset.sdf_clamp_cells) == 8.0

    gen = saved.generator
    schema = gen.physical_schema
    assert list(schema.state_vars) == list(STATE_VARS)
    assert list(schema.param_vars) == list(PARAM_VARS)
    assert schema.param_history_steps == HP
    assert schema.history_dt_seconds == DT
    assert dict(schema.units) == {
        "u": "m/s",
        "v": "m/s",
        "w": "m/s",
        "inflow_angle": "deg",
        "velocity_magnitude": "m/s",
    }
    assert "blanking" in schema.geometry_mask_convention
    assert list(schema.coordinate_order) == ["z", "y", "x"]
    nz, ny, nx = GRID
    grid = schema.grid
    assert (grid.nz, grid.ny, grid.nx) == (nz, ny, nx)
    assert (grid.dz, grid.dy, grid.dx) == (1.0, 1.0, 1.0)
    assert [list(b) for b in grid.bounds] == [
        [-0.5, nx - 0.5],
        [-0.5, ny - 0.5],
        [-0.5, nz - 0.5],
    ]
    supported: list[Any] = [
        OmegaConf.to_container(g) for g in schema.supported_geometries
    ]
    assert len(supported) == 1  # one unique train geometry
    assert supported[0]["shape"] == [nz, ny, nx]
    n_obstacle = (nz // 2) * (ny // 2 - ny // 4) * (nx // 2 - nx // 4)
    assert supported[0]["fluid_cells"] == nz * ny * nx - n_obstacle
    # The mask itself identifies the geometry, not just its cell count.
    from neural_surrogates.generative_spinup import geometry_fingerprint

    blank = xr.load_dataset(
        Path(saved.dataset.root_dir) / "state" / "train" / "sample_0000.nc"
    )["blanking"].values
    assert supported[0]["mask_sha256"] == geometry_fingerprint(1.0 - blank)
    assert schema.boundary_conditions == "synthetic test corpus"
    assert schema.constant_forcing_notes == ""
    assert len(gen.ae_fingerprint) == 64
    assert gen.sampling.num_steps == 2
    prov = gen.data_provenance
    assert prov.split == "train"
    assert prov.n_train == 2 * (6 - HP + 1) and prov.n_val == 6 - HP + 1
    assert prov.constant_prehistory is False
    assert prov.verified_prehistory is None  # nothing to verify with the flag off
    assert prov.training_data_config.time.output_frequency == DT
    assert gen.latent_stats.max_batches == 2 and gen.latent_stats.seed == 0

    # Self-contained reload: no AE dir, no data dir -- remove both first.
    import shutil

    ae_dir, data_dir = fixture_inputs(
        tmp_path, spatial_mode=spatial_mode, geometry=geometry
    )
    ae_export = torch.load(ae_dir / "weights.pt", map_location="cpu")
    shutil.rmtree(ae_dir)
    shutil.rmtree(data_dir)
    fresh = instantiate(
        saved.architecture, n_state_channels=len(STATE_VARS), n_params=len(PARAM_VARS)
    )
    assert fresh.ae_fingerprint is None
    weights = torch.load(model_dir / "weights.pt", map_location="cpu")
    fresh.load_state_dict(weights, strict=True)
    fresh.eval()
    assert bool(fresh.latent_stats_installed)
    assert gen.latent_stats.n_channels == fresh.working_latent_dim
    assert not torch.equal(fresh.latent_std, torch.ones_like(fresh.latent_std))
    assert not torch.equal(fresh.param_mean, torch.zeros_like(fresh.param_mean))

    # (c) the zero-init output projection really moved ...
    out_proj = fresh.velocity_net.seqmodel.out_proj.weight
    assert torch.count_nonzero(out_proj) > 0
    # ... and every frozen AE parameter is byte-identical to the export's.
    for name, p in fresh.ae.named_parameters():
        assert not p.requires_grad
        assert torch.equal(p, ae_export[name]), name
    assert torch.equal(fresh.ae.state_mean, ae_export["state_mean"])
    assert torch.equal(fresh.ae.state_std, ae_export["state_std"])

    # sample() runs on the reloaded model with the artifact's own step count.
    geom = torch.ones(2, *GRID)
    geom[:, :4, 2:6, 3:9] = 0.0
    feats = fresh.ae._sdf_features(geom)
    params_hist = torch.tensor([[[10.0, 3.0]] * HP, [[-5.0, 4.5]] * HP])
    out = fresh.sample(
        params_hist, geom, feats, generator=torch.Generator().manual_seed(0)
    )
    assert out.shape == (2, C, *GRID)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out * (1 - geom.unsqueeze(1))) == 0


# --------------------------------------------------------------------------- #
# (d) deterministic validation + rejections.
# --------------------------------------------------------------------------- #


def test_validation_is_deterministic_across_passes(tmp_path):
    ae_dir, data_dir = fixture_inputs(tmp_path)
    cfg = compose_generator_cfg(ae_dir, data_dir, tmp_path)
    trainer = load_run()(cfg)
    a = trainer._validate()
    b = trainer._validate()
    assert a == b and a > 0
    # A different val_seed gives a different (but again reproducible) draw.
    trainer.val_seed = 123
    c = trainer._validate()
    assert c != a and c == trainer._validate()
    # Training draws come from the global RNG and do not perturb validation.
    torch.manual_seed(999)
    trainer.val_seed = 0
    assert trainer._validate() == a


def test_trainer_rejects_shuffling_val_loader_and_empty_loaders(tmp_path):
    ae_dir, data_dir = fixture_inputs(tmp_path)
    model = _model(ae_dir)
    train_ds, val_ds = _dataset(data_dir, "train"), _dataset(data_dir, "val")
    ok_train, ok_val = _loader(train_ds, shuffle=True), _loader(val_ds)
    with pytest.raises(ValueError, match="shuffle"):
        _trainer(model, ok_train, _loader(val_ds, shuffle=True))
    shuffling = DataLoader(
        val_ds,
        batch_sampler=TrajectoryBatchSampler(
            val_ds, batch_size=2, shuffle=True  # type: ignore[arg-type]
        ),
        collate_fn=snapshot_history_collate,
    )
    with pytest.raises(ValueError, match="shuffle"):
        _trainer(model, ok_train, shuffling)
    empty = DataLoader(
        val_ds,
        batch_sampler=TrajectoryBatchSampler(
            val_ds, batch_size=100, shuffle=False, drop_last=True  # type: ignore[arg-type]
        ),
        collate_fn=snapshot_history_collate,
    )
    assert len(empty) == 0
    with pytest.raises(ValueError, match="no batches"):
        _trainer(model, ok_train, empty)
    with pytest.raises(ValueError, match="no batches"):
        _trainer(model, empty, ok_val)
    # The optimizer must only see trainable (velocity-net) parameters.
    with pytest.raises(ValueError, match="frozen-AE"):
        _trainer(
            model,
            ok_train,
            ok_val,
            optimizer=torch.optim.AdamW([p for p in model.parameters()], lr=1e-3),
        )
    # A well-formed trainer builds, and refuses to train without latent stats.
    trainer = _trainer(model, ok_train, ok_val)
    assert not bool(model.latent_stats_installed)
    with pytest.raises(RuntimeError, match="latent normalisation"):
        trainer.fit()


# --------------------------------------------------------------------------- #
# (e) resume: next epoch, latent stats intact, cache reused.
# --------------------------------------------------------------------------- #


def test_resume_continues_with_latent_stats_and_cached_stats(tmp_path, monkeypatch):
    model_dir = train_tiny_generator(tmp_path, values={"trainer.num_epochs": 1})
    assert [r["epoch"] for r in _metrics_rows(model_dir)] == ["1"]
    first = torch.load(model_dir / "weights.pt", map_location="cpu")
    cache = torch.load(model_dir / "latent_stats.pt", map_location="cpu")
    assert torch.equal(cache["mean"], first["latent_mean"])
    assert torch.equal(cache["std"], first["latent_std"])
    ckpt = torch.load(model_dir / "checkpoint.pt", map_location="cpu")
    assert ckpt["epoch"] == 0
    assert bool(ckpt["model"]["latent_stats_installed"])

    # Any recomputation on resume is a bug: the stats must come from the cache.
    def _no_recompute(*args, **kwargs):
        raise AssertionError("latent stats recomputed on resume")

    monkeypatch.setattr(
        TadpoleLatentGenerator, "compute_latent_normalization", _no_recompute
    )
    ae_dir, data_dir = fixture_inputs(tmp_path)
    cfg = compose_generator_cfg(
        ae_dir,
        data_dir,
        tmp_path,
        values={"trainer.resume": True, "trainer.num_epochs": 2},
    )
    trainer = load_run()(cfg)
    assert [r["epoch"] for r in _metrics_rows(model_dir)] == ["1", "2"]
    model = trainer._eager_model
    assert bool(model.latent_stats_installed)
    assert torch.equal(model.latent_mean, first["latent_mean"])
    assert torch.equal(model.latent_std, first["latent_std"])
    assert torch.load(model_dir / "checkpoint.pt", map_location="cpu")["epoch"] == 1

    # A checkpoint whose latent stats are missing is refused outright.
    ckpt = torch.load(model_dir / "checkpoint.pt", map_location="cpu")
    ckpt["model"]["latent_stats_installed"] = torch.tensor(False)
    torch.save(ckpt, model_dir / "checkpoint.pt")
    cfg = compose_generator_cfg(
        ae_dir,
        data_dir,
        tmp_path,
        values={"trainer.resume": True, "trainer.num_epochs": 3},
    )
    with pytest.raises(RuntimeError, match="latent_stats_installed"):
        load_run()(cfg)


# --------------------------------------------------------------------------- #
# (f) attention budget, (g) state_vars mismatch, required metadata.
# --------------------------------------------------------------------------- #


def test_attention_budget_rejects_before_training(tmp_path):
    ae_dir, data_dir = fixture_inputs(tmp_path)
    cfg = compose_generator_cfg(
        ae_dir, data_dir, tmp_path, values={"architecture.max_latent_tokens": 1}
    )
    with pytest.raises(ValueError, match="latent attention budget"):
        load_run()(cfg)
    assert not (tmp_path / "model_weights" / "latent_generator_test").exists()


def test_attention_budget_is_checked_on_the_val_split_too(tmp_path):
    """A val-only grid over the budget must fail up front, not at the first
    validation pass: the val loader forwards the same velocity net."""
    ae_dir, _ = fixture_inputs(tmp_path)
    data_dir = tmp_path / "data_big_val"
    write_history_dataset(data_dir, splits={"train": 2}, grid=GRID)
    write_history_dataset(data_dir, splits={"val": 1}, grid=(32, 16, 32), seed=7)
    # batch 2 x (1,1,2) = 4 latent tokens on train, x (2,1,2) = 8 on val.
    cfg = compose_generator_cfg(
        ae_dir, data_dir, tmp_path, values={"architecture.max_latent_tokens": 5}
    )
    with pytest.raises(ValueError, match="latent attention budget.*val split"):
        load_run()(cfg)
    assert not (tmp_path / "model_weights" / "latent_generator_test").exists()


def test_constant_prehistory_is_gated_on_corpus_provenance(tmp_path):
    """The repeated leading history is a claim about the DATA, so it is checked
    against the corpus' own constant-forcing spin-up."""
    ae_dir, data_dir = fixture_inputs(tmp_path)  # writes time.spinup_time: 0.0
    run = load_run()
    values: dict[str, Any] = {
        "dataset.constant_prehistory": True,
        "trainer.num_epochs": 1,
    }

    # (a) The spin-up is shorter than the plateau the flag invents.
    with pytest.raises(ValueError, match="at least as long as the repeated plateau"):
        run(compose_generator_cfg(ae_dir, data_dir, tmp_path, values=values))

    # (b) A corpus that records no spin-up at all cannot vouch for anything.
    data_cfg = OmegaConf.load(data_dir / "config.yaml")
    data_cfg.time.pop("spinup_time")
    OmegaConf.save(data_cfg, data_dir / "config.yaml")
    with pytest.raises(ValueError, match="time.spinup_time"):
        run(compose_generator_cfg(ae_dir, data_dir, tmp_path, values=values))

    # (c) An adequate spin-up trains, and the verdict is recorded.
    data_cfg.time.spinup_time = (HP - 1) * DT
    OmegaConf.save(data_cfg, data_dir / "config.yaml")
    run(compose_generator_cfg(ae_dir, data_dir, tmp_path, values=values))
    model_dir = tmp_path / "model_weights" / "latent_generator_test"
    prov = OmegaConf.load(model_dir / "config.yaml").generator.data_provenance
    assert prov.constant_prehistory is True
    assert prov.verified_prehistory.spinup_time == (HP - 1) * DT
    assert prov.verified_prehistory.required_seconds == (HP - 1) * DT
    assert prov.verified_prehistory.first_saved_time == 0.0
    # Anchors now start at t = 0 (Hp-1 more per trajectory).
    assert prov.n_train == 2 * 6


def test_transposed_corpus_is_rejected_before_export(tmp_path):
    """A corpus saved (x, y, z) would be trained and deployed on silently
    swapped axes -- every tensor here is positional."""
    ae_dir, _ = fixture_inputs(tmp_path)
    data_dir = tmp_path / "data_xyz"
    write_history_dataset(data_dir, grid=(16, 16, 16))
    for path in sorted((data_dir / "state").rglob("sample_*.nc")):
        xr.load_dataset(path).transpose("time", "x", "y", "z").to_netcdf(path)
    cfg = compose_generator_cfg(
        ae_dir, data_dir, tmp_path, values={"trainer.num_epochs": 1}
    )
    with pytest.raises(ValueError, match="coordinate_order"):
        load_run()(cfg)
    assert not (
        tmp_path / "model_weights" / "latent_generator_test" / "config.yaml"
    ).exists()


def test_state_vars_mismatch_with_ae_raises(tmp_path):
    ae_dir, data_dir = fixture_inputs(tmp_path)
    cfg = compose_generator_cfg(ae_dir, data_dir, tmp_path)
    cfg.dataset.state_vars = ["u", "v"]
    with pytest.raises(ValueError, match="state_vars mismatch"):
        load_run()(cfg)
    # An explicit SDF override that disagrees with the AE is refused too.
    cfg = compose_generator_cfg(
        ae_dir, data_dir, tmp_path, values={"dataset.sdf_features": "none"}
    )
    with pytest.raises(ValueError, match="sdf_features mismatch"):
        load_run()(cfg)


def test_required_inputs_fail_loud(tmp_path):
    ae_dir, data_dir = fixture_inputs(tmp_path)
    run = load_run()
    cfg = compose_generator_cfg(ae_dir, data_dir, tmp_path)
    cfg.physical_metadata.units.pop("velocity_magnitude")
    with pytest.raises(ValueError, match="units lacks"):
        run(cfg)
    cfg = compose_generator_cfg(ae_dir, data_dir, tmp_path)
    # The mask polarity is the deploy side's canonical constant, not free text.
    cfg.physical_metadata.geometry_mask_convention = "blanking: 0 = obstacle"
    with pytest.raises(ValueError, match="geometry_mask_convention must be exactly"):
        run(cfg)
    cfg = compose_generator_cfg(ae_dir, data_dir, tmp_path)
    cfg.physical_metadata.boundary_conditions = "???"
    with pytest.raises(ValueError, match="'\\?\\?\\?'"):
        run(cfg)
    cfg = compose_generator_cfg(ae_dir, data_dir, tmp_path)
    cfg.pretrained_ae_dir = "???"
    with pytest.raises(ValueError, match="pretrained_ae_dir"):
        run(cfg)
    cfg = compose_generator_cfg(ae_dir, data_dir, tmp_path)
    cfg.dataset.param_vars = "???"
    with pytest.raises(ValueError, match="param_vars"):
        run(cfg)
