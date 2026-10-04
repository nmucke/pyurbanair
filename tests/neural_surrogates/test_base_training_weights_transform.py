"""BaseTraining.weights_transform: best-val export must survive a resume.

The LoRA fine-tune trainer writes an always-valid *merged* ``weights.pt`` on every
val improvement (``weights_transform``) and, at the end of ``fit()``, restores the
best-val model in memory. The subtle bug this guards: on a **resumed** run whose
epochs never beat the pre-resume ``best_val``, the end-of-fit restore must still
put the best (not the last) epoch back — otherwise the caller's final
``save(merge(...))`` clobbers the on-disk best with worse weights.

This exercises ``BaseTraining`` directly with a scripted trainer (no peft/P3D
needed): the model mutates every epoch so best != last, and ``_validate`` returns
scripted losses so we control exactly when improvements happen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

torch = pytest.importorskip("torch")

from neural_surrogates import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _snapshot(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _make_trainer(
    model: nn.Module,
    weights_path: Path,
    num_epochs: int,
    val_scores: list[float | None],
    checkpoint_every: int = 1,
    weights_transform: Callable[[nn.Module], dict] | None = _snapshot,
) -> Trainer:
    """A Trainer whose train/val steps are scripted and deterministic.

    ``_train_epoch`` bumps every parameter by 1.0 (so each epoch's state is
    distinct — best != last), ``_validate`` yields the next scripted loss. A
    ``None`` in ``val_scores`` raises inside ``_validate`` to simulate a crash
    mid-run (after that epoch's train step, before any checkpoint).
    """
    ds = TensorDataset(torch.zeros(2, 2), torch.zeros(2, 2))
    loader = DataLoader(ds, batch_size=1)
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        loss_fn=nn.MSELoss(),
        num_epochs=num_epochs,
        device="cpu",
        weights_path=weights_path,
        # A stand-in for merge_to_state_dict: not None => the RAM-snapshot restore
        # path is taken (the on-disk weights.pt is treated as a non-reloadable
        # transformed form), which is exactly the path under test.
        weights_transform=weights_transform,
        resume=True,
        checkpoint_every=checkpoint_every,
        patience=None,
        amp=False,
        pushforward_epochs_per_step=None,
        lr_warmup_epochs=None,
    )
    scores = iter(val_scores)

    def fake_train_epoch() -> float:
        with torch.no_grad():
            for p in trainer._eager_model.parameters():
                p.add_(1.0)
        return 0.0

    def fake_validate() -> float:
        v = next(scores)
        if v is None:
            raise RuntimeError("simulated crash mid-run")
        return v

    trainer._train_epoch = fake_train_epoch  # type: ignore[method-assign]
    trainer._validate = fake_validate
    return trainer


def test_resume_exports_best_not_last(tmp_path: Path) -> None:
    """A resume that never improves must still leave the model at best-val."""
    weights_path = tmp_path / "weights.pt"

    torch.manual_seed(0)
    model1 = nn.Linear(2, 2)
    init = _snapshot(model1)  # best is reached after exactly one train step (+1)
    expected_best = {k: v + 1.0 for k, v in init.items()}

    # Run 1: epoch 1 improves to val=1.0 (best), epoch 2 worsens to 5.0.
    t1 = _make_trainer(model1, weights_path, num_epochs=2, val_scores=[1.0, 5.0])
    t1.fit()
    # end of run 1: model restored to best (init + 1), not last (init + 2)
    for k in expected_best:
        assert torch.allclose(model1.state_dict()[k], expected_best[k]), k
    assert t1.restored_best_weights

    # Run 2: fresh model, RESUME from checkpoint.pt. The best epoch (1) precedes
    # the resume; this run does one more epoch that does NOT improve (val=10.0).
    model2 = nn.Linear(2, 2)
    t2 = _make_trainer(model2, weights_path, num_epochs=3, val_scores=[10.0])
    t2.fit()

    # The bug: without the fix, best_state is None after resume, so end-of-fit
    # leaves model2 at the LAST epoch (init + 3). The fix restores best (init + 1)
    # from the checkpoint's persisted snapshot.
    assert t2.restored_best_weights
    for k in expected_best:
        assert torch.allclose(model2.state_dict()[k], expected_best[k]), (
            k,
            "resumed run left last-epoch weights instead of best-val",
        )

    # Simulate the fine-tune script's guarded step 7: only overwrite when the
    # best was restored. The exported weights must equal the best, not the last.
    if t2.restored_best_weights:
        assert t2.weights_transform is not None
        exported = t2.weights_transform(t2._eager_model)
        for k in expected_best:
            assert torch.allclose(exported[k], expected_best[k]), k


@pytest.mark.parametrize("transformed", [False, True])  # type: ignore[misc]
def test_resume_does_not_clobber_newer_on_disk_weights(
    tmp_path: Path, transformed: bool
) -> None:
    """M1: a resume from a STALE checkpoint must not clobber the newer weights.pt.

    weights.pt is rewritten on every val improvement, but the full checkpoint
    (carrying best_val / best_model_state) only every ``checkpoint_every`` epochs.
    So an improvement at a non-checkpoint epoch followed by a crash leaves an
    on-disk weights.pt that is BETTER than the last checkpoint's best. A resume
    that never re-beats it must neither restore the stale snapshot nor let the
    caller export it over the better on-disk weights.
    """
    weights_path = tmp_path / "weights.pt"
    best_val_path = tmp_path / "best_val.json"

    torch.manual_seed(1)
    model_a = nn.Linear(2, 2)
    transform = _snapshot if transformed else None

    # Run A: improve to val=8 (weights.pt = init+1, best_val.json = 8), then
    # plateau. checkpoint_every=1 => the checkpoint captures best_state = init+1,
    # best_val = 8. Ends cleanly (model left at init+2).
    ta = _make_trainer(
        model_a,
        weights_path,
        num_epochs=2,
        val_scores=[8.0, 9.0],
        checkpoint_every=1,
        weights_transform=transform,
    )
    ta.fit()
    assert json.loads(best_val_path.read_text())["best_val"] == 8.0

    # Run B: RESUME, improve to val=3 at a non-checkpoint epoch (checkpoint_every
    # huge), then crash before the next checkpoint. weights.pt + best_val.json
    # advance to the better epoch (val=3); the checkpoint stays stale (best_val=8).
    model_b = nn.Linear(2, 2)
    tb = _make_trainer(
        model_b,
        weights_path,
        num_epochs=4,
        val_scores=[3.0, None],
        checkpoint_every=99,
        weights_transform=transform,
    )
    with pytest.raises(RuntimeError):
        tb.fit()
    assert json.loads(best_val_path.read_text())["best_val"] == 3.0
    newer = {k: v.clone() for k, v in torch.load(weights_path).items()}

    # Run C: RESUME from the stale checkpoint (best_val=8, best_state=init+1) and
    # never beat val=3. The fix reads best_val.json (=3), adopts it as the
    # threshold and drops the stale snapshot => no restore, so the guarded caller
    # must keep the newer on-disk weights.pt untouched.
    model_c = nn.Linear(2, 2)
    tc = _make_trainer(
        model_c,
        weights_path,
        num_epochs=7,
        val_scores=[5.0] * 5,
        checkpoint_every=99,
        weights_transform=transform,
    )
    tc.fit()
    assert tc.restored_best_weights is (not transformed)
    assert tc.best_val == 3.0

    # weights.pt must still hold the NEWER (val=3) weights, not the stale best.
    on_disk = torch.load(weights_path)
    for k in newer:
        assert torch.allclose(on_disk[k], newer[k]), k
        if not transformed:
            assert torch.equal(model_c.state_dict()[k], newer[k])


def test_plain_resume_recovers_legacy_disk_best(tmp_path: Path) -> None:
    """An old plain run has no sidecar; score its newer weights once on resume."""
    weights_path = tmp_path / "weights.pt"
    original = nn.Linear(2, 2)
    first = _make_trainer(original, weights_path, 2, [0.5, 0.6], weights_transform=None)
    first.fit()
    interrupted = _make_trainer(
        nn.Linear(2, 2),
        weights_path,
        4,
        [0.4, None],
        checkpoint_every=99,
        weights_transform=None,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        interrupted.fit()
    saved_best = torch.load(weights_path)
    (tmp_path / "best_val.json").unlink()

    resumed = _make_trainer(
        nn.Linear(2, 2), weights_path, 3, [0.4, 0.45], weights_transform=None
    )
    seen = []
    validate = resumed._validate

    def record_validate() -> float:
        seen.append(_snapshot(resumed._eager_model))
        return float(validate())

    resumed._validate = record_validate
    resumed.fit()
    assert resumed.best_val == 0.4
    assert resumed.restored_best_weights
    assert json.loads((tmp_path / "best_val.json").read_text())["best_val"] == 0.4
    assert len(seen) == 2
    for k, v in saved_best.items():
        assert torch.equal(seen[0][k], v)
        assert torch.equal(torch.load(weights_path)[k], v)
        assert torch.equal(resumed._eager_model.state_dict()[k], v)


@pytest.mark.parametrize("warmup", [0, 2])  # type: ignore[misc]
@pytest.mark.parametrize("completed", [False, True])  # type: ignore[misc]
def test_extended_resume_keeps_cosine_lr_nonincreasing(
    tmp_path: Path, warmup: int, completed: bool
) -> None:
    """An extended run must not enter the rising half of the old cosine."""
    weights_path = tmp_path / "weights.pt"

    def configured(epochs: int, scores: list[float | None]) -> Trainer:
        trainer = _make_trainer(nn.Linear(2, 2), weights_path, epochs, scores)
        trainer.optimizer.param_groups[0]["lr"] = 0.1
        trainer.lr_warmup_epochs = warmup
        trainer.lr_warmup_start = 0.01
        trainer.lr_min = 0.001
        trainer.scheduler = trainer._build_lr_scheduler()
        return trainer

    first = configured(6, [1.0] * 6 if completed else [1.0] * 3 + [None])
    if completed:
        first.fit()
    else:
        with pytest.raises(RuntimeError, match="simulated crash"):
            first.fit()
    checkpoint = torch.load(tmp_path / "checkpoint.pt")
    checkpoint_lr = checkpoint["optimizer"]["param_groups"][0]["lr"]
    remaining = 10 - checkpoint["epoch"] - 1
    resumed = configured(10, [1.0] * remaining)
    train_epoch = resumed._train_epoch
    learning_rates: list[float] = []

    def record_epoch() -> float:
        learning_rates.append(resumed.optimizer.param_groups[0]["lr"])
        return float(train_epoch())

    resumed._train_epoch = record_epoch  # type: ignore[method-assign]
    resumed.fit()
    learning_rates.append(resumed.optimizer.param_groups[0]["lr"])
    assert learning_rates[0] == pytest.approx(checkpoint_lr)
    assert all(b <= a + 1e-12 for a, b in zip(learning_rates, learning_rates[1:]))
    assert learning_rates[-1] == pytest.approx(0.001)


def test_resume_checkpoint_metadata_mismatch_preserves_model(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.pt"
    first = _make_trainer(nn.Linear(2, 2), weights_path, 1, [0.5])
    first.checkpoint_metadata = {"param_vars": ["angle", "speed"]}
    first.fit()
    second = _make_trainer(nn.Linear(2, 2), weights_path, 2, [0.4])
    second.checkpoint_metadata = {"param_vars": ["speed", "angle"]}
    before = _snapshot(second._eager_model)
    with pytest.raises(ValueError, match="checkpoint metadata"):
        second.fit()
    for k, v in before.items():
        assert torch.equal(second._eager_model.state_dict()[k], v)
