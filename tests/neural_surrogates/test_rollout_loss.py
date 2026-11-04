"""``Trainer(rollout_loss=...)``: which steps of a K-step rollout the loss supervises.

* ``"final"`` (the default) is the historic pushforward loss: a ``no_grad``
  prefix, ``grad_unroll_steps`` gradient steps, the masked loss of the last
  prediction against ``state_next``;
* ``"sum"`` is the mean of every step's masked loss against ``state_targets``:
  the last ``g`` steps backpropagate through their chain, each earlier step
  through its own one-step loss from its detached input (step 1 from the true
  state), backpropagated at once during training;
* validation always uses ``"final"``; ``"sum"`` without targets fails loud.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from neural_surrogates import Trainer  # noqa: E402
from neural_surrogates.datasets.transition import transition_collate  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

NZ, NY, NX = 3, 4, 5
C = 3
P = 2


class _Linear(nn.Module):
    """``a * newest + b * oldest + c * params``, zero in the solid: a stepper
    with a nonlinear-in-parameters rollout (``a`` compounds over steps)."""

    n_state_channels = C

    def __init__(self, num_history_steps: int = 1) -> None:
        super().__init__()
        self.num_history_steps = num_history_steps
        self.a = nn.Parameter(torch.tensor(0.9))
        self.b = nn.Parameter(torch.tensor(0.2))
        self.c = nn.Parameter(torch.tensor(0.1))

    def forward(self, state: Any, params: Any, geometry: Any) -> Any:
        out = (
            self.a * state[:, -C:]
            + self.b * state[:, :C]
            + self.c * params.sum(1).view(-1, 1, 1, 1, 1)
        )
        return out * geometry.unsqueeze(1)


def _geometry() -> torch.Tensor:
    geometry = torch.ones(NZ, NY, NX)
    geometry[0, 1:3, 1:3] = 0.0
    return geometry


def _items(n: int, k: int, h: int = 1, seed: int = 0) -> list[dict]:
    gen = torch.Generator().manual_seed(seed)
    geometry = _geometry()  # one object: transition_collate ships it once
    items = []
    for _ in range(n):
        targets = torch.randn(k, C, NZ, NY, NX, generator=gen)
        items.append(
            {
                "state_n": torch.randn(h * C, NZ, NY, NX, generator=gen),
                "state_next": targets[-1],
                "state_targets": targets,
                "params_n": torch.randn(k, P, generator=gen),
                "geometry": geometry,
            }
        )
    return items


def _batch(k: int, h: int = 1, n: int = 2) -> dict:
    return transition_collate(_items(n, k, h))


def _trainer(model: nn.Module, items: list[dict] | None = None, **kwargs: Any):
    loader = DataLoader(
        items or _items(4, 3), batch_size=2, collate_fn=transition_collate
    )
    return Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=nn.MSELoss(),
        num_epochs=1,
        device="cpu",
        **kwargs,
    )


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fluid = _geometry().bool()
    return torch.mean((pred[..., fluid] - target[..., fluid]) ** 2)


def _advance(state: torch.Tensor, pred: torch.Tensor, h: int) -> torch.Tensor:
    return pred if h == 1 else torch.cat([state[:, C:], pred], dim=1)


def _reference(model: nn.Module, batch: dict, g: int, h: int, mode: str):
    """The two losses written out step by step."""
    state = batch["state_n"]
    params, targets = batch["params_n"], batch["state_targets"]
    geometry = batch["geometry"].expand(state.shape[0], NZ, NY, NX)
    K = params.shape[1]
    losses = []
    for i in range(K):
        early = i < K - g
        with torch.set_grad_enabled(mode == "sum" or not early):
            pred = model(state, params[:, i], geometry)
        if mode == "sum" or i == K - 1:
            losses.append(_masked_mse(pred, targets[:, i]))
        state = _advance(state, pred.detach() if early else pred, h)
    return torch.stack(losses).mean()


def _grads(model: nn.Module, loss: torch.Tensor) -> list[torch.Tensor]:
    return list(torch.autograd.grad(loss, list(model.parameters()), retain_graph=True))


@pytest.mark.parametrize(("k", "g", "h"), [(1, 2, 1), (3, 1, 1), (3, 2, 2)])
def test_final_is_the_default_and_unchanged(k: int, g: int, h: int) -> None:
    model = _Linear(h)
    batch = _batch(k, h)
    trainer = _trainer(model, grad_unroll_steps=g)
    assert trainer.rollout_loss == "final"
    loss = trainer._forward(batch)
    expected = _reference(model, batch, g, h, "final")
    torch.testing.assert_close(loss, expected)
    for got, want in zip(_grads(model, loss), _grads(model, expected)):
        torch.testing.assert_close(got, want)
    assert trainer._aux_terms is None


@pytest.mark.parametrize(("k", "g", "h"), [(1, 2, 1), (3, 1, 1), (4, 2, 2)])
def test_sum_is_the_mean_of_the_step_losses(k: int, g: int, h: int) -> None:
    model = _Linear(h)
    batch = _batch(k, h)
    trainer = _trainer(model, grad_unroll_steps=g, rollout_loss="sum")
    loss = trainer._forward(batch)  # no _backward_part: the whole graph is kept
    expected = _reference(model, batch, g, h, "sum")
    torch.testing.assert_close(loss, expected)
    for got, want in zip(_grads(model, loss), _grads(model, expected)):
        torch.testing.assert_close(got, want)
    terms = trainer._aux_terms
    assert terms is not None and list(terms) == [f"step_{i + 1}" for i in range(k)]
    torch.testing.assert_close(torch.stack(list(terms.values())).mean(), loss)


def test_sum_trains_the_first_step_from_the_true_state() -> None:
    """K=3, g=1: the first prediction, from ``state_n``, gets the gradient of
    its own loss, which the final-only loss does not give it."""
    k = 3
    batch = _batch(k)
    for mode in ("sum", "final"):
        model = _Linear()
        preds: list[torch.Tensor] = []
        model.register_forward_hook(lambda m, args, out: preds.append(out))
        trainer = _trainer(model, grad_unroll_steps=1, rollout_loss=mode)
        loss = trainer._forward(batch)
        first = preds[0]
        if mode == "final":  # the no_grad prefix
            assert not first.requires_grad
            continue
        first.retain_grad()
        loss.backward()
        # Only step 1's own loss reaches it (its prediction is fed on detached).
        x = first.detach().requires_grad_()
        (want,) = torch.autograd.grad(
            _masked_mse(x, batch["state_targets"][:, 0]) / k, [x]
        )
        assert want.abs().sum() > 0
        torch.testing.assert_close(first.grad, want)


def test_sum_backpropagates_each_early_step_at_once() -> None:
    """With ``_backward_part`` set (training) the K-g early losses are
    backpropagated inside the forward, already detached in the returned loss;
    the total gradient equals the whole-graph one."""
    k, g = 4, 2
    batch = _batch(k)
    reference = _Linear()
    expected = _grads(reference, _reference(reference, batch, g, 1, "sum"))

    model = _Linear()
    trainer = _trainer(model, grad_unroll_steps=g, rollout_loss="sum")
    parts: list[torch.Tensor] = []

    def backward_part(x: torch.Tensor) -> None:
        parts.append(x.detach())
        x.backward()

    trainer._backward_part = backward_part
    loss = trainer._forward(batch)
    assert len(parts) == k - g
    loss.backward()
    for p, want in zip(model.parameters(), expected):
        torch.testing.assert_close(p.grad, want)


def test_validation_keeps_the_final_loss() -> None:
    k = 3
    batch = _batch(k)
    model = _Linear()
    trainer = _trainer(model, rollout_loss="sum")
    trainer._forward(batch)  # a training batch leaves its per-step terms
    with torch.no_grad():
        val = trainer._forward(batch)
        expected = _reference(model, batch, 2, 1, "final")
    torch.testing.assert_close(val, expected)
    assert trainer._aux_terms is None


def test_sum_needs_the_targets() -> None:
    batch = _batch(3)
    del batch["state_targets"]
    trainer = _trainer(_Linear(), rollout_loss="sum")
    with pytest.raises(ValueError, match="RolloutTransitionDataset"):
        trainer._forward(batch)
    with torch.no_grad():  # validation needs only state_next
        trainer._forward(batch)


def test_unknown_rollout_loss_raises() -> None:
    with pytest.raises(ValueError, match="rollout_loss"):
        _trainer(_Linear(), rollout_loss="mean")


@pytest.mark.parametrize(
    ("accum", "group"),
    [({}, 1), ({"grad_accum_steps": 2}, 2), ({"grad_accum_samples": 4}, 2)],
)
def test_sum_epoch_takes_the_whole_graph_step(
    accum: dict, group: int, tmp_path: Path
) -> None:
    """A training epoch with the per-step backward makes the same update as
    backpropagating the whole-graph loss, under every accumulation mode; the
    per-step terms reach metrics.csv for training only."""
    k = 3
    items = _items(8, k)
    model = _Linear()
    trainer = _trainer(
        model,
        items,
        rollout_loss="sum",
        weights_path=tmp_path / "weights.pt",
        **accum,
    )
    reference = _Linear()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    loader = DataLoader(items, batch_size=2, collate_fn=transition_collate)
    for i, batch in enumerate(loader):
        if i % group == 0:
            optimizer.zero_grad()
        (_reference(reference, batch, 2, 1, "sum") / group).backward()
        if (i + 1) % group == 0:
            optimizer.step()

    trainer.fit()
    with torch.no_grad():
        for got, want in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(got, want)
    with open(tmp_path / "metrics.csv") as f:
        (row,) = list(csv.DictReader(f))
    assert {f"train_step_{i + 1}" for i in range(k)} <= row.keys()
    assert not any(key.startswith("val_step") for key in row)
