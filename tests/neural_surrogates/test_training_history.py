"""History-window (``num_history_steps``) wiring through the training stack.

Covers two of the places Task C touches:

* ``BaseTraining._advance_history`` -- the ring buffer that rolls the flattened
  ``(B, H*C, *grid)`` model input forward one frame per pushforward step. At the
  default ``H = 1`` it must return the prediction *itself* (the pre-history
  behaviour, byte-identical).
* ``BaseTraining._forward`` -- every model call inside a pushforward rollout
  must see the full window, not just the newest frame.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from neural_surrogates import Trainer  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

NZ = NY = NX = 4
C = 3  # state channels (u, v, w)
P = 2  # params per step


# --- unit-level fixtures ------------------------------------------------------ #
class _EchoModel(nn.Module):
    """Minimal stand-in for a history-aware architecture.

    Exposes the two attributes ``BaseTraining`` reads off the eager model
    (``num_history_steps`` / ``n_state_channels``), records the shape of every
    input it is handed, and returns a single ``(B, C, *grid)`` frame derived from
    the newest history frame -- so the trainer's masked loss and the gradient
    path are exercised without pulling in a real architecture.
    """

    def __init__(self, num_history_steps: int = 1, n_state_channels: int = C) -> None:
        super().__init__()
        self.num_history_steps = int(num_history_steps)
        self.n_state_channels = int(n_state_channels)
        self.n_input_state_channels = self.num_history_steps * self.n_state_channels
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.seen_shapes: list[tuple[int, ...]] = []

    def forward(self, state: Any, params: Any, geometry: Any, extra: Any = None) -> Any:
        self.seen_shapes.append(tuple(state.shape))
        return state[:, -self.n_state_channels :] * self.scale


class _BareModel(nn.Module):
    """A pre-history architecture: neither attribute is defined."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, state: Any, params: Any, geometry: Any, extra: Any = None) -> Any:
        return state * self.scale


def _dummy_loader() -> DataLoader:
    return DataLoader([0, 1], batch_size=1)


def _make_trainer(model: nn.Module) -> Trainer:
    return Trainer(
        model=model,
        train_loader=_dummy_loader(),
        val_loader=_dummy_loader(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
    )


def _batch(*, history: int, pushforward: int, batch: int = 2) -> dict:
    return {
        "state_n": torch.randn(batch, history * C, NZ, NY, NX),
        "state_next": torch.randn(batch, C, NZ, NY, NX),
        "params_n": torch.randn(batch, pushforward, P),
        "geometry": torch.ones(1, NZ, NY, NX),
    }


# --- (a) _advance_history ----------------------------------------------------- #
def test_trainer_reads_history_spec_off_the_eager_model() -> None:
    trainer = _make_trainer(_EchoModel(num_history_steps=3))
    assert trainer.num_history_steps == 3
    assert trainer.n_state_channels == C


def test_trainer_defaults_history_spec_for_pre_history_models() -> None:
    """An architecture predating the knob keeps the one-step behaviour."""
    trainer = _make_trainer(_BareModel())
    assert trainer.num_history_steps == 1
    assert trainer.n_state_channels is None


def test_advance_history_is_identity_on_the_prediction_at_h1() -> None:
    """H=1 must hand back the prediction object itself: the rollout then reduces
    to the pre-history ``state = self._model_forward(...)`` exactly."""
    trainer = _make_trainer(_EchoModel(num_history_steps=1))
    state = torch.randn(2, C, NZ, NY, NX)
    pred = torch.randn(2, C, NZ, NY, NX)
    assert trainer._advance_history(state, pred) is pred


def test_advance_history_shifts_the_window_oldest_first() -> None:
    trainer = _make_trainer(_EchoModel(num_history_steps=3))
    frames = [torch.randn(2, C, NZ, NY, NX) for _ in range(3)]
    pred = torch.randn(2, C, NZ, NY, NX)

    out = trainer._advance_history(torch.cat(frames, dim=1), pred)

    assert out.shape == (2, 3 * C, NZ, NY, NX)
    # Oldest frame dropped, the remaining two shift down, prediction appended
    # last -- so ``out[:, -C:]`` is always the newest frame.
    assert torch.equal(out[:, :C], frames[1])
    assert torch.equal(out[:, C : 2 * C], frames[2])
    assert torch.equal(out[:, -C:], pred)


def test_advance_history_raises_without_n_state_channels() -> None:
    model = _EchoModel(num_history_steps=2)
    trainer = _make_trainer(model)
    trainer.n_state_channels = None
    with pytest.raises(ValueError, match="n_state_channels"):
        trainer._advance_history(
            torch.randn(2, 2 * C, NZ, NY, NX), torch.randn(2, C, NZ, NY, NX)
        )


# --- (b) _forward under a pushforward rollout --------------------------------- #
def test_forward_feeds_the_full_window_at_every_pushforward_step() -> None:
    """K=3, H=2: all three model calls (2 rollout steps + the final loss step)
    must receive ``(B, H*C, *grid)``, i.e. the window is rolled, not replaced."""
    H, K, B = 2, 3, 2
    model = _EchoModel(num_history_steps=H)
    trainer = _make_trainer(model)

    loss = trainer._forward(_batch(history=H, pushforward=K, batch=B))

    assert loss.requires_grad
    assert len(model.seen_shapes) == K
    assert all(s == (B, H * C, NZ, NY, NX) for s in model.seen_shapes)


def test_forward_is_unchanged_at_h1() -> None:
    """The default path still feeds a single ``(B, C, *grid)`` frame per step."""
    K, B = 3, 2
    model = _EchoModel(num_history_steps=1)
    trainer = _make_trainer(model)

    trainer._forward(_batch(history=1, pushforward=K, batch=B))

    assert model.seen_shapes == [(B, C, NZ, NY, NX)] * K
