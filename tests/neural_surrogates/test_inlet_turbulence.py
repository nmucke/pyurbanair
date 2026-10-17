"""`InletTurbulence`: uDALES' inlet forcing replayed onto rollout predictions.

On a tiny synthetic corpus "generated" with inlet turbulence (1 m cells, 1 s
frames, 0.5 s driver records, a ground layer and one building): the footprint
equals the solver's own driver planes where a cell's inflow time falls on a
record, reaches U*dt/dx columns but never the building or obstacle cells, is
reproducible per seed, and has the requested rms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
pytest.importorskip("pyudales")

from neural_surrogates.datasets.transition import TransitionDataset
from neural_surrogates.inlet_turbulence import InletTurbulence

NZ, NY, NX, T_LEN = 6, 16, 12, 40
SPINUP, OUTPUT_FREQUENCY = 2.0, 1.0
FIRST_BUILT_COLUMN = 8
TURBULENCE = {
    "enabled": True,
    "intensity": 0.15,
    "length_scale_y": 2.0,
    "length_scale_z": 2.0,
    "length_scale_x": 4.0,
    "time_step": 0.5,
    "seed": None,
}


def _corpus(
    root: Path, speed: float = 3.0, turbulence: dict | None = TURBULENCE
) -> TransitionDataset:
    """One test trajectory at constant `speed` along x."""
    forward_model: dict = {
        "experiment_name": 999,
        "nudging_config": {"profile_config": {"type": "uniform"}},
    }
    if turbulence is not None:
        forward_model["inlet_turbulence"] = turbulence
    OmegaConf.save(
        {
            "model": {"forward_model": forward_model},
            "training_data": {"output_frequency": OUTPUT_FREQUENCY},
        },
        root / "config.yaml",
    )
    blanking = np.zeros((NZ, NY, NX), dtype=np.int8)
    blanking[0] = 1  # ground layer
    blanking[:3, 4:8, FIRST_BUILT_COLUMN:10] = 1
    times = np.arange(T_LEN) * OUTPUT_FREQUENCY
    state = xr.Dataset(
        {
            name: (("time", "zt", "yt", "xt"), np.zeros((T_LEN, NZ, NY, NX)))
            for name in ("u", "v", "w")
        }
        | {"blanking": (("zt", "yt", "xt"), blanking)},
        coords={
            "time": times,
            "zt": np.arange(NZ) + 0.5,
            "yt": np.arange(NY) + 0.5,
            "xt": np.arange(NX) + 0.5,
        },
        attrs={"spinup_time_s": SPINUP},
    )
    params = xr.Dataset(
        {
            "inflow_angle": ("time", np.zeros(T_LEN)),
            "velocity_magnitude": ("time", np.full(T_LEN, speed)),
        },
        coords={"time": times},
    )
    for kind, ds in (("state", state), ("param", params)):
        (root / kind / "test").mkdir(parents=True)
        ds.to_netcdf(root / kind / "test" / "sample_0000.nc")
    return TransitionDataset(root, "test")


def test_replays_the_solver_driver_planes(tmp_path: Path) -> None:
    """At U = 3 m/s, column 1's centre (1.5 m from the inlet face) holds the
    inflow of 0.5 s before the frame: exactly one driver record."""
    from pyudales.utils.inlet_turbulence_utils import (
        build_driver_planes,
        driver_time_grid,
    )

    from pyurbanair.utils.digital_filter import derive_seed

    footprint = InletTurbulence.from_dataset(_corpus(tmp_path), 0).footprint
    # As apply_inlet_turbulence builds them for this run (dtmax = 1 s).
    times = driver_time_grid(SPINUP + T_LEN * OUTPUT_FREQUENCY, 0.5, dtmax=1.0)
    kwargs = {
        "jtot": NY,
        "ktot": NZ,
        "zsize": float(NZ),
        "ylen": float(NY),
        "times": times,
        "start_index": 0,
        "inflow_angle": np.zeros(times.size),
        "velocity_magnitude": np.full(times.size, 3.0),
        "profile_config": {"type": "uniform"},
        "length_scale_y": 2.0,
        "length_scale_z": 2.0,
        "length_scale_x": 4.0,
        "seed": derive_seed("999"),
    }
    planes = build_driver_planes(intensity=0.15, **kwargs)
    means = build_driver_planes(intensity=0.0, **kwargs)
    u, v, w = (p - m for p, m in zip(planes, means))
    w = 0.5 * (w[:, 1:-1] + w[:, 2:])  # faces -> centres
    frames = np.arange(1, T_LEN)
    records = (SPINUP + OUTPUT_FREQUENCY * (frames + 1) - 0.5) / 0.5
    expected = np.stack([u[:, 1:-1, 1:-1], v[:, 1:-1, 1:-1], w[:, :, 1:-1]], axis=1)[
        records.astype(int)
    ]
    expected[:, :, 0] = 0.0  # the ground layer is masked
    np.testing.assert_allclose(footprint[1:, ..., 1], expected, atol=1e-5)
    assert not footprint[0].any()


def test_forces_only_the_free_inlet_columns(tmp_path: Path) -> None:
    turbulence = InletTurbulence.from_dataset(_corpus(tmp_path), 0)
    assert turbulence.footprint.shape == (T_LEN, 3, NZ, NY, 3)  # U*dt/dx
    pred = torch.randn(2, 3, NZ, NY, NX)
    before = pred.clone()
    out = turbulence(pred, 5)
    change = out - pred
    torch.testing.assert_close(change[0], change[1])
    assert change[..., :3].abs().amax(dim=(0, 1, 2, 3)).gt(0).all()
    assert not change[..., 3:].any()
    assert not change[:, :, 0].any()  # ground layer
    assert torch.equal(pred, before)  # input untouched


def test_never_reaches_the_buildings(tmp_path: Path) -> None:
    turbulence = InletTurbulence.from_dataset(_corpus(tmp_path, speed=30.0), 0)
    assert turbulence.footprint.shape[-1] == FIRST_BUILT_COLUMN


def test_seeds(tmp_path: Path) -> None:
    data = _corpus(tmp_path)
    truth = InletTurbulence.from_dataset(data, 0).footprint
    fresh = InletTurbulence.from_dataset(data, 0, seed=1).footprint
    assert torch.equal(fresh, InletTurbulence.from_dataset(data, 0, seed=1).footprint)
    assert not torch.allclose(fresh, truth)
    assert not torch.allclose(
        fresh, InletTurbulence.from_dataset(data, 0, seed=2).footprint
    )


def test_inlet_rms_matches_the_intensity(tmp_path: Path) -> None:
    footprint = InletTurbulence.from_dataset(_corpus(tmp_path), 0).footprint
    rms = footprint[1:, 0, 1:, :, 0].std().item()
    assert rms == pytest.approx(0.15 * 3.0, rel=0.25)


def test_rejects_a_corpus_without_inlet_turbulence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inlet_turbulence"):
        InletTurbulence.from_dataset(_corpus(tmp_path, turbulence=None), 0)
    disabled = tmp_path / "disabled"
    disabled.mkdir()
    data = _corpus(disabled, turbulence={**TURBULENCE, "enabled": False})
    with pytest.raises(ValueError, match="inlet_turbulence"):
        InletTurbulence.from_dataset(data, 0)


def test_rejects_a_state_without_spinup(tmp_path: Path) -> None:
    data = _corpus(tmp_path)
    path = data._state_files[0]
    state = xr.load_dataset(path)
    del state.attrs["spinup_time_s"]
    state.to_netcdf(path)
    with pytest.raises(ValueError, match="spinup_time_s"):
        InletTurbulence.from_dataset(data, 0)


def _edit_config(root: Path, key: str, value: object) -> None:
    cfg = OmegaConf.load(root / "config.yaml")
    OmegaConf.update(cfg, key, value, force_add=True)
    OmegaConf.save(cfg, root / "config.yaml")


def test_rejects_a_corpus_of_the_current_generator(tmp_path: Path) -> None:
    """generate_data.py writes time.output_frequency and stores no seed or
    record offset per sample."""
    data = _corpus(tmp_path)
    cfg = OmegaConf.load(tmp_path / "config.yaml")
    del cfg["training_data"]
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="training_data.output_frequency"):
        InletTurbulence.from_dataset(data, 0)


def test_rejects_a_corpus_without_experiment_name(tmp_path: Path) -> None:
    data = _corpus(tmp_path)
    _edit_config(tmp_path, "model.forward_model.experiment_name", None)
    with pytest.raises(ValueError, match="experiment_name"):
        InletTurbulence.from_dataset(data, 0)
    InletTurbulence.from_dataset(data, 0, seed=1)  # a fresh seed needs none


def test_reads_the_corpus_params_not_the_model_inputs(tmp_path: Path) -> None:
    data = _corpus(tmp_path)
    narrow = TransitionDataset(tmp_path, "test", param_vars=["velocity_magnitude"])
    torch.testing.assert_close(
        InletTurbulence.from_dataset(narrow, 0).footprint,
        InletTurbulence.from_dataset(data, 0).footprint,
    )


def test_knots_denser_than_frames_fall_back_to_the_frames(tmp_path: Path) -> None:
    data = _corpus(tmp_path)
    reference = InletTurbulence.from_dataset(data, 0).footprint
    _edit_config(tmp_path, "training_data.params_sampler.seconds_per_knot", 0.5)
    torch.testing.assert_close(
        InletTurbulence.from_dataset(data, 0).footprint, reference
    )


def test_warns_without_a_free_inlet_column(tmp_path: Path) -> None:
    data = _corpus(tmp_path)
    path = data._state_files[0]
    state = xr.load_dataset(path)
    state["blanking"][:3, 4:8, 0] = 1
    state.to_netcdf(path)
    data = TransitionDataset(tmp_path, "test")
    with pytest.warns(UserWarning, match="no building-free column"):
        turbulence = InletTurbulence.from_dataset(data, 0)
    assert turbulence.footprint.shape[-1] == 0
