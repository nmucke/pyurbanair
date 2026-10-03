"""Plan 07 phase 2B: ``LatentFlowMatchingTrainer`` construction contract.

On CPU smoke shapes via the shared fixtures in ``_latent_generator_fixtures``:
a shuffling val loader and an empty loader are rejected, the optimizer may not
hold AE parameters, and training refuses to start without latent statistics.

Gated with ``importorskip`` on the vendored Tadpole runtime deps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from neural_surrogates import (
    LatentFlowMatchingTrainer,
    SnapshotHistoryDataset,
    TadpoleLatentGenerator,
    TrajectoryBatchSampler,
    snapshot_history_collate,
)
from torch.utils.data import DataLoader

from tests.neural_surrogates._latent_generator_fixtures import (
    HP,
    NET,
    PARAM_VARS,
    STATE_VARS,
    C,
    P,
    fixture_inputs,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(3)
    yield


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
# Trainer rejections.
# --------------------------------------------------------------------------- #


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
