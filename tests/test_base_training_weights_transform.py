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

import pytest

torch = pytest.importorskip("torch")

from neural_surrogates import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _snapshot(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _make_trainer(model, weights_path, num_epochs, val_scores):
    """A Trainer whose train/val steps are scripted and deterministic.

    ``_train_epoch`` bumps every parameter by 1.0 (so each epoch's state is
    distinct — best != last), ``_validate`` yields the next scripted loss.
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
        weights_transform=_snapshot,
        resume=True,
        checkpoint_every=1,
        patience=None,
        amp=False,
        pushforward_epochs_per_step=None,
        lr_warmup_epochs=None,
    )
    scores = iter(val_scores)

    def fake_train_epoch():
        with torch.no_grad():
            for p in trainer._eager_model.parameters():
                p.add_(1.0)
        return 0.0

    trainer._train_epoch = fake_train_epoch  # type: ignore[method-assign]
    trainer._validate = lambda: next(scores)  # type: ignore[method-assign]
    return trainer


def test_resume_exports_best_not_last(tmp_path):
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
        exported = t2.weights_transform(t2._eager_model)
        for k in expected_best:
            assert torch.allclose(exported[k], expected_best[k]), k
