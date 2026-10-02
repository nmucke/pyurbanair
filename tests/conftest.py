"""Shared helpers and fixtures for tests/ (see README.md).

The script tests (scripts/) use the real configs in configs/, made tiny by
an overlay from tests/configs/ (`+test=<name>`), e.g.

    cfg = compose("forward", "+test=forward", root=tmp_path)

Every run writes under `root`. The solver-free tests run the scripts on the
neural-surrogate backend: `training_data` writes a small synthetic dataset and
`trained` trains every surrogate on it once per session. Tests that run a
compiled CFD solver are marked `integration`.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from types import ModuleType
from typing import Any

# On macOS the native loader can abort if torch initializes before NumPy.
import numpy as np
import pytest
import xarray
from hydra import compose as hydra_compose
from hydra import initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# The library tests' fixtures (compose_test_cfg, ...) on the old frozen configs.
from tests.legacy.fixtures import (  # noqa: E402,F401
    _isolate_run_outputs,
    _limit_torch_test_threads,
    _restore_hydra_config_singleton,
    compose_module_cfg,
    compose_test_cfg,
    surrogate_model_dir_factory,
)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
TEST_CONFIGS = pathlib.Path(__file__).resolve().parent / "configs"


def compose(config_name: str, *overrides: str, root: pathlib.Path) -> DictConfig:
    """A configs/ config with every output and scratch dir under `root`."""
    paths = [
        f"paths.results_root={root / 'results'}",
        f"paths.scratch.local={root / 'scratch'}",
        f"paths.weights_dir={root / 'weights'}",
        f"paths.training_data_dir={root / 'training_data'}",
    ]
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return hydra_compose(
            config_name,
            overrides=[f"hydra.searchpath=[file://{TEST_CONFIGS}]", *paths, *overrides],
        )


def load_script(path: str) -> ModuleType:
    """Import a script (e.g. "scripts/run_forward.py") as a module."""
    file = REPO / path
    name = "_".join(file.relative_to(REPO).with_suffix("").parts)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, file)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def surrogate(session_root: pathlib.Path) -> list[str]:
    """Overrides that run the tiny trained surrogates (`trained` fixture)."""
    return [
        f"paths.weights_dir={session_root / 'weights'}",
        f"paths.training_data_dir={session_root / 'training_data'}",
    ]


# ---------------------------------------------------------------------------
# Session fixtures: synthetic training data and the surrogates trained on it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")  # type: ignore[misc]
def session_root(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return pathlib.Path(tmp_path_factory.mktemp("session"))


@pytest.fixture(scope="session")  # type: ignore[misc]
def training_data(session_root: pathlib.Path) -> pathlib.Path:
    """Synthetic training data in generate_data.py's layout, on the tiny grid.

    Smooth random u, v, w around a building block, 10 frames per trajectory:
    four train trajectories, one val and one test.
    """
    cfg = compose("forward", "+test=forward", root=session_root)
    out = session_root / "training_data" / "tiny"
    out.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create(
            {"case_name": cfg.case_name, "domain": cfg.domain, "time": cfg.time}
        ),
        out / "config.yaml",
        resolve=True,
    )
    nx, ny, nz = cfg.domain.nx, cfg.domain.ny, cfg.domain.nz
    (x0, x1), (y0, y1), (z0, z1) = cfg.domain.bounds
    coords = {
        "zt": z0 + (np.arange(nz) + 0.5) * (z1 - z0) / nz,
        "yt": y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny,
        "xt": x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx,
    }
    blanking = np.zeros((nz, ny, nx), dtype=np.int8)
    blanking[:2, 7:13, 7:13] = 1  # one building
    times = np.arange(10, dtype=float) * float(cfg.time.output_frequency)
    rng = np.random.default_rng(0)
    splits = {"train": 4, "val": 1, "test": 1}
    for split, count in splits.items():
        (out / "state" / split).mkdir(parents=True)
        (out / "param" / split).mkdir(parents=True)
        for i in range(count):
            angle = rng.normal(0.0, 10.0) + np.cumsum(rng.normal(0, 1, len(times)))
            speed = rng.normal(5.0, 0.5) + np.cumsum(rng.normal(0, 0.1, len(times)))
            fields = {}
            for c, name in enumerate(("u", "v", "w")):
                base = (speed if name == "u" else np.radians(angle) * speed)[
                    :, None, None, None
                ]
                noise = rng.normal(0, 0.3, (len(times), nz, ny, nx))
                field = (0.0 if name == "w" else base) * (1 + 0.2 * c) + noise
                fields[name] = (
                    ("time", "zt", "yt", "xt"),
                    np.where(blanking, 0.0, field).astype(np.float32),
                )
            state = xarray.Dataset(
                {**fields, "blanking": (("zt", "yt", "xt"), blanking)},
                coords={"time": times, **coords},
            )
            params = xarray.Dataset(
                {
                    "inflow_angle": ("time", angle),
                    "velocity_magnitude": ("time", speed),
                    "pressure_gradient_magnitude": 0.0041912,
                },
                coords={"time": times},
            )
            state.to_netcdf(out / "state" / split / f"sample_{i:04d}.nc")
            params.to_netcdf(out / "param" / split / f"sample_{i:04d}.nc")
    return out


TRAIN_ORDER = (
    "train_stepper",
    "finetune_stepper",
    "train_autoencoder",
    "train_latent_generator",
    "train_dft",
)


@pytest.fixture(scope="session")  # type: ignore[misc]
def trained(session_root: pathlib.Path, training_data: pathlib.Path) -> dict[str, Any]:
    """Train every surrogate once on the synthetic data (in dependency order).

    Returns {training config: its composed config}; the weights are in
    `<session_root>/weights/<name>/`.
    """
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    train = load_script("scripts/surrogate/train.py")
    configs = {}
    for name in TRAIN_ORDER:
        cfg = compose(f"surrogate/{name}", f"+test={name}", root=session_root)
        train.run(cfg)
        configs[name] = cfg
    return configs
