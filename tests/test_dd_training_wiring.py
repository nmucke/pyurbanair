"""End-to-end wiring test for the domain-decomposed training paths.

Exercises the actual Hydra composition + ``train_neural_surrogate.run(cfg)``
for both the drop-in path (generic ``Trainer`` + ``MSELoss``) and the patch
path (``PatchTrainer`` + ``DomainDecompositionLoss``), proving the
``trainer``/``loss`` config groups and the ``domain_decomposed`` architecture
selection wire together and produce a checkpoint. The DD model is shrunk via
config overrides so the run is CPU-fast.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_WORKTREE = Path(__file__).resolve().parents[1]
_CONF = _WORKTREE / "conf"
_SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "train_neural_surrogate.py"


def _load_run():
    spec = importlib.util.spec_from_file_location("train_ns_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def _write_dataset(root: Path, *, nz=8, ny=8, nx=8, t=4) -> None:
    """Tiny pylbm-style dataset: u,v,w + blanking over (time,z,y,x), with
    matching per-time params. ny is divisible by the test interior_size (4)."""
    rng = np.random.default_rng(0)
    counts = {"train": 2, "val": 1}
    for split, n in counts.items():
        (root / "state" / split).mkdir(parents=True, exist_ok=True)
        (root / "param" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            blank = np.zeros((nz, ny, nx), "f4")
            blank[0] = 1.0  # one solid ground row
            state = {
                v: (("time", "z", "y", "x"),
                    rng.standard_normal((t, nz, ny, nx)).astype("f4"))
                for v in ("u", "v", "w")
            }
            state["blanking"] = (("z", "y", "x"), blank)
            xr.Dataset(
                state,
                coords=dict(time=np.arange(t) * 1.0, z=np.arange(nz),
                            y=np.arange(ny), x=np.arange(nx)),
            ).to_netcdf(root / "state" / split / f"sample_{i:04d}.nc")
            xr.Dataset(
                {
                    "inflow_angle": (("time",), rng.standard_normal(t).astype("f4")),
                    "velocity_magnitude": (
                        ("time",), (7 + rng.standard_normal(t)).astype("f4")),
                },
                coords=dict(time=np.arange(t) * 1.0),
            ).to_netcdf(root / "param" / split / f"sample_{i:04d}.nc")


def _shrink_dd(cfg) -> None:
    """Shrink the DD architecture + dataloader so the run is CPU-fast."""
    cfg.dataset.pushforward_steps = 1
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.architecture.decomposition.interior_size = 4
    cfg.architecture.decomposition.halo = 2
    cfg.architecture.decomposition.taper = 1
    cfg.architecture.decomposition.coarsen_factor = 2
    cfg.architecture.fine_net = dict(
        _target_="neural_surrogates.UNetConvNeXt",
        base_channels=4, channel_mults=[1, 2], depths=[1, 1],
        kernel_size=3, expansion=2, normalize=True, residual=True,
    )
    cfg.architecture.coarse_net = dict(
        _target_="neural_surrogates.UNetConvNeXt",
        base_channels=4, channel_mults=[1, 2], depths=[1, 1],
        normalize=True, residual=True,
    )
    cfg.trainer.num_epochs = 1
    cfg.trainer.device = "cpu"
    cfg.init_weights_path = None


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        return compose(config_name="neural_surrogate/training", overrides=overrides)


@pytest.mark.parametrize(
    "extra_overrides,model_name,expect_trainer,expect_loss",
    [
        # Each path selects its trainer + loss explicitly rather than relying on
        # the (mutable) training.yaml default, so the test is robust to whatever
        # default is currently checked in. The trainer class and the loss are
        # bundled into the `mode` group: standard -> Trainer + MSELoss,
        # domain_decomposition -> PatchTrainer + DomainDecompositionLoss.
        (
            ["neural_surrogate/mode@_global_=standard"],
            "dd_dropin_test",
            "Trainer",
            "MSELoss",
        ),
        (
            ["neural_surrogate/mode@_global_=domain_decomposition"],
            "dd_patch_test",
            "PatchTrainer",
            "DomainDecompositionLoss",
        ),
    ],
)
def test_dd_training_paths(
    tmp_path, monkeypatch, extra_overrides, model_name, expect_trainer, expect_loss
):
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)

    cfg = _compose(
        ["neural_surrogate/architectures@architecture=domain_decomposed/small"]
        + extra_overrides
    )
    OmegaConf.set_struct(cfg, False)
    _shrink_dd(cfg)
    cfg.dataset.root_dir = str(data_dir)
    cfg.model_name = model_name
    # Both trainers now default to GPU AMP + torch.compile + a pushforward
    # curriculum (PatchTrainer shares the generic Trainer's machinery via
    # BaseTraining); neutralise those for a 1-epoch CPU smoke run.
    cfg.trainer.amp = False
    cfg.trainer.compile_model = False
    cfg.trainer.pushforward_epochs_per_step = None
    cfg.trainer.pushforward_start_steps = 1
    cfg.trainer.patience = None

    assert cfg.trainer._target_.split(".")[-1] == expect_trainer
    assert cfg.loss._target_.split(".")[-1] == expect_loss

    monkeypatch.chdir(tmp_path)
    _load_run()(cfg)

    out = tmp_path / "model_weights" / model_name
    assert (out / "weights.pt").exists()
    saved = OmegaConf.load(out / "config.yaml")
    assert saved.trainer._target_.split(".")[-1] == expect_trainer
    assert saved.loss._target_.split(".")[-1] == expect_loss
    assert saved.architecture._target_.split(".")[-1] == "DomainDecomposed"
