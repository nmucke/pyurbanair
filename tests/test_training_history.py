"""History-window (``num_history_steps``) wiring through the training stack.

Covers the three places Task C touches:

* ``BaseTraining._advance_history`` -- the ring buffer that rolls the flattened
  ``(B, H*C, *grid)`` model input forward one frame per pushforward step. At the
  default ``H = 1`` it must return the prediction *itself* (the pre-history
  behaviour, byte-identical).
* ``BaseTraining._forward`` -- every model call inside a pushforward rollout
  must see the full window, not just the newest frame.
* ``train_neural_surrogate.run(cfg)`` -- the dataset/architecture cross-check and
  the ``num_history_steps`` stamped into the saved ``config.yaml`` under BOTH
  nodes (the forward model rebuilds from ``architecture``, the fine-tune script
  reads ``dataset``). Plus the legacy path: a config carrying neither key must
  still run, at ``H = 1``.
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

from neural_surrogates import Trainer  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

NZ = NY = NX = 4
C = 3  # state channels (u, v, w)
P = 2  # params per step

_WORKTREE = Path(__file__).resolve().parents[1]
_CONF = _WORKTREE / "conf"
_SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "train_neural_surrogate.py"


# --- unit-level fixtures ------------------------------------------------------ #
class _EchoModel(torch.nn.Module):
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

    def forward(self, state, params, geometry, extra=None):
        self.seen_shapes.append(tuple(state.shape))
        return state[:, -self.n_state_channels :] * self.scale


class _BareModel(torch.nn.Module):
    """A pre-history architecture: neither attribute is defined."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, state, params, geometry, extra=None):
        return state * self.scale


def _dummy_loader() -> DataLoader:
    return DataLoader([0, 1], batch_size=1)


def _make_trainer(model: torch.nn.Module) -> Trainer:
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


# --- (c)/(d) end-to-end through the train script ------------------------------ #
def _load_run():
    spec = importlib.util.spec_from_file_location(
        "train_ns_history_under_test", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def _write_dataset(root: Path, *, nz=8, ny=8, nx=8, t=8) -> None:
    """Tiny pylbm-style dataset: u,v,w + blanking over (time,z,y,x), with
    matching per-time params. ``t`` is generous enough that a history window
    still leaves several (state_n, state_next) pairs per trajectory."""
    rng = np.random.default_rng(0)
    counts = {"train": 2, "val": 1}
    for split, n in counts.items():
        (root / "state" / split).mkdir(parents=True, exist_ok=True)
        (root / "param" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            blank = np.zeros((nz, ny, nx), "f4")
            blank[0] = 1.0  # one solid ground row
            state = {
                v: (
                    ("time", "z", "y", "x"),
                    rng.standard_normal((t, nz, ny, nx)).astype("f4"),
                )
                for v in ("u", "v", "w")
            }
            state["blanking"] = (("z", "y", "x"), blank)
            xr.Dataset(
                state,
                coords=dict(
                    time=np.arange(t) * 1.0,
                    z=np.arange(nz),
                    y=np.arange(ny),
                    x=np.arange(nx),
                ),
            ).to_netcdf(root / "state" / split / f"sample_{i:04d}.nc")
            xr.Dataset(
                {
                    "inflow_angle": (("time",), rng.standard_normal(t).astype("f4")),
                    "velocity_magnitude": (
                        ("time",),
                        (7 + rng.standard_normal(t)).astype("f4"),
                    ),
                },
                coords=dict(time=np.arange(t) * 1.0),
            ).to_netcdf(root / "param" / split / f"sample_{i:04d}.nc")


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        return compose(config_name="neural_surrogate/training", overrides=overrides)


def _smoke_cfg(data_dir: Path, model_name: str):
    """A 1-epoch CPU-fast standard-mode run on a tiny ConvNeXt-UNet."""
    cfg = _compose(
        [
            "neural_surrogate/mode@_global_=standard",
            "neural_surrogate/architectures@architecture=unet_convnext/tiny",
            # Pin the dataset SDF mode to the architecture default so the train
            # script's SDF cross-check is not what fails here.
            "dataset.sdf_features=none",
        ]
    )
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.root_dir = str(data_dir)
    cfg.dataset.pushforward_steps = 1
    cfg.model_name = model_name
    cfg.init_weights_path = None
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    if cfg.get("batch_sampler") is not None:
        cfg.batch_sampler.batch_size = 2
    cfg.trainer.num_epochs = 1
    cfg.trainer.device = "cpu"
    cfg.trainer.amp = False
    cfg.trainer.compile_model = False
    cfg.trainer.pushforward_epochs_per_step = None
    cfg.trainer.pushforward_start_steps = 1
    cfg.trainer.patience = None
    return cfg


def test_train_script_end_to_end_with_history(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)

    cfg = _smoke_cfg(data_dir, "history_h2_test")
    cfg.dataset.num_history_steps = 2
    cfg.architecture.num_history_steps = 2

    monkeypatch.chdir(tmp_path)
    _load_run()(cfg)

    out = tmp_path / "model_weights" / "history_h2_test"
    assert (out / "weights.pt").exists()
    saved = OmegaConf.load(out / "config.yaml")
    # Both nodes must carry the window: the forward model rebuilds the net from
    # `architecture` alone, the fine-tune script reads it back off `dataset`.
    assert saved.dataset.num_history_steps == 2
    assert saved.architecture.num_history_steps == 2


def test_train_script_rejects_history_mismatch(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)

    cfg = _smoke_cfg(data_dir, "history_mismatch_test")
    cfg.dataset.num_history_steps = 1
    cfg.architecture.num_history_steps = 2

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="history-window mismatch"):
        _load_run()(cfg)


def test_train_script_legacy_config_defaults_to_one_step(tmp_path, monkeypatch) -> None:
    """A config carrying the key on NEITHER node still trains, at H=1, and the
    saved config records the resolved default."""
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)

    cfg = _smoke_cfg(data_dir, "history_legacy_test")
    cfg.dataset.pop("num_history_steps", None)
    cfg.architecture.pop("num_history_steps", None)

    monkeypatch.chdir(tmp_path)
    _load_run()(cfg)

    out = tmp_path / "model_weights" / "history_legacy_test"
    assert (out / "weights.pt").exists()
    saved = OmegaConf.load(out / "config.yaml")
    assert saved.dataset.num_history_steps == 1
    assert saved.architecture.num_history_steps == 1
