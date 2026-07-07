"""Training-path wiring for the SDF geometry features.

Covers the pieces between the helper (:func:`neural_surrogates.sdf.sdf_features`,
tested in ``test_sdf_features.py``) and the model:

* :class:`TransitionDataset` ships ``geom_features`` iff built with
  ``sdf_features=True``, and items are byte-identical to today's otherwise;
* :func:`transition_collate` ships the features once as ``(1, 4, *grid)``;
* :class:`BaseTraining` fails loud when the model wants features the batch lacks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from neural_surrogates import Trainer
from neural_surrogates.datasets.transition import TransitionDataset, transition_collate
from torch.utils.data import DataLoader

NZ = NY = NX = 8
T_LEN = 4
STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")


def _write_sample(state_dir: Path, param_dir: Path, idx: int, seed: int) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dims = ("time", "z", "y", "x")
    shape = (T_LEN, NZ, NY, NX)
    data_vars = {
        v: (dims, rng.standard_normal(shape).astype(np.float32)) for v in STATE_VARS
    }
    obstacle = np.zeros((NZ, NY, NX), dtype=np.float32)
    obstacle[0:2, 2:4, 2:4] = 1.0
    data_vars["blanking"] = (dims, np.broadcast_to(obstacle, shape).copy())
    xr.Dataset(data_vars).to_netcdf(state_dir / f"sample_{idx:04d}.nc")
    pvars = {
        "inflow_angle": (("time",), rng.uniform(-60, 60, T_LEN).astype(np.float32)),
        "velocity_magnitude": (("time",), rng.uniform(1, 5, T_LEN).astype(np.float32)),
    }
    xr.Dataset(pvars).to_netcdf(param_dir / f"sample_{idx:04d}.nc")


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for idx in range(2):
        _write_sample(root / "state" / "train", root / "param" / "train", idx, 10 + idx)
    return root


def _make_ds(root: Path, **overrides) -> TransitionDataset:
    kwargs = dict(
        root_dir=root,
        split="train",
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        geometry_var="blanking",
        pushforward_steps=1,
    )
    kwargs.update(overrides)
    return TransitionDataset(**kwargs)


def test_dataset_ships_features_when_enabled(data_root: Path) -> None:
    ds = _make_ds(data_root, sdf_features=True, sdf_clamp_cells=8.0)
    item = ds[0]
    assert "geom_features" in item
    assert item["geom_features"].shape == (4, NZ, NY, NX)
    # Same shared tensor object across items (computed once at init).
    assert ds[1]["geom_features"] is ds[0]["geom_features"]
    # sdf_n channel matches the standalone helper on the mask.
    from neural_surrogates import sdf_features as sf

    torch.testing.assert_close(
        item["geom_features"], sf(ds.geometry_for(0), clamp_cells=8.0)
    )


def test_dataset_byte_identical_when_disabled(data_root: Path) -> None:
    ds = _make_ds(data_root)  # default sdf_features=False
    item = ds[0]
    assert "geom_features" not in item
    assert set(item) == {"state_n", "state_next", "params_n", "geometry"}
    assert ds._geom_features is None


def test_collate_ships_features_once(data_root: Path) -> None:
    ds = _make_ds(data_root, sdf_features=True)
    batch = transition_collate([ds[0], ds[1], ds[2]])
    assert batch["geometry"].shape == (1, NZ, NY, NX)
    assert batch["geom_features"].shape == (1, 4, NZ, NY, NX)
    assert batch["state_n"].shape == (3, len(STATE_VARS), NZ, NY, NX)


def test_collate_no_features_key_when_disabled(data_root: Path) -> None:
    ds = _make_ds(data_root)
    batch = transition_collate([ds[0], ds[1]])
    assert "geom_features" not in batch
    assert batch["geometry"].shape == (1, NZ, NY, NX)


# -- trainer guard ----------------------------------------------------------


class _FeatureHungryModel(torch.nn.Module):
    """Minimal model that advertises it needs SDF features."""

    n_geom_feature_channels = 4

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, state, params, geometry, geom_features=None):  # pragma: no cover
        return state


class _RecordingModel(torch.nn.Module):
    """Records the geom_features it is called with; returns the state as-is."""

    n_geom_feature_channels = 4

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.seen = None

    def forward(self, state, params, geometry, geom_features=None):
        self.seen = geom_features
        return state


def _dummy_loader() -> DataLoader:
    return DataLoader([0, 1], batch_size=1)


def _make_trainer(model) -> Trainer:
    return Trainer(
        model=model,
        train_loader=_dummy_loader(),
        val_loader=_dummy_loader(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
    )


def test_model_forward_threads_expanded_features(data_root: Path) -> None:
    """_prepare_batch caches the shared (4, *grid) features and _model_forward
    broadcasts them to (B, 4, *grid) for the model call."""
    model = _RecordingModel()
    trainer = _make_trainer(model)
    ds = _make_ds(data_root, sdf_features=True)
    batch = transition_collate([ds[0], ds[1], ds[2]])
    state, _, _, geometry = trainer._prepare_batch(batch)
    assert trainer._geom_features is not None
    assert trainer._geom_features.shape == (4, NZ, NY, NX)
    trainer._model_forward(state, torch.zeros(3, len(PARAM_VARS)), geometry)
    assert model.seen is not None
    assert model.seen.shape == (3, 4, NZ, NY, NX)


def test_trainer_raises_when_model_wants_features_but_batch_lacks_them() -> None:
    model = _FeatureHungryModel()
    trainer = Trainer(
        model=model,
        train_loader=_dummy_loader(),
        val_loader=_dummy_loader(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
    )
    batch = {
        "state_n": torch.zeros(2, 3, NZ, NY, NX),
        "state_next": torch.zeros(2, 3, NZ, NY, NX),
        "params_n": torch.zeros(2, 1, len(PARAM_VARS)),
        "geometry": torch.ones(1, NZ, NY, NX),  # no geom_features
    }
    with pytest.raises(ValueError, match="n_geom_feature_channels"):
        trainer._prepare_batch(batch)
