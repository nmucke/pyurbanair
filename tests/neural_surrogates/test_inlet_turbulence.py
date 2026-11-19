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


def _solver_column(seed: int, start_index: int) -> np.ndarray:
    """Column 1 of every predicted frame as the solver's driver planes hold it
    (at U = 3 m/s its centre, 1.5 m from the inlet face, holds the inflow of
    0.5 s before the frame: exactly one driver record)."""
    from pyudales.utils.inlet_turbulence_utils import (
        build_driver_planes,
        driver_time_grid,
    )

    # As apply_inlet_turbulence builds them for this run (dtmax = 1 s).
    times = driver_time_grid(SPINUP + T_LEN * OUTPUT_FREQUENCY, 0.5, dtmax=1.0)
    kwargs = {
        "jtot": NY,
        "ktot": NZ,
        "zsize": float(NZ),
        "ylen": float(NY),
        "times": times,
        "start_index": start_index,
        "inflow_angle": np.zeros(times.size),
        "velocity_magnitude": np.full(times.size, 3.0),
        "profile_config": {"type": "uniform"},
        "length_scale_y": 2.0,
        "length_scale_z": 2.0,
        "length_scale_x": 4.0,
        "seed": seed,
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
    return expected


def test_replays_the_solver_driver_planes(tmp_path: Path) -> None:
    from pyurbanair.utils.digital_filter import derive_seed

    footprint = InletTurbulence.from_dataset(_corpus(tmp_path), 0).footprint
    expected = _solver_column(derive_seed("999"), 0)
    np.testing.assert_allclose(footprint[1:, ..., 1], expected, atol=1e-5)
    assert not footprint[0].any()


def _current_generator(root: Path, attrs: dict) -> TransitionDataset:
    """The corpus in generate_data.py's layout: `time.output_frequency` and the
    seed and start record in the state attrs."""
    data = _corpus(root)
    cfg = OmegaConf.load(root / "config.yaml")
    del cfg["training_data"]
    cfg.time = {"output_frequency": OUTPUT_FREQUENCY}
    OmegaConf.save(cfg, root / "config.yaml")
    path = data._state_files[0]
    state = xr.load_dataset(path)
    state.attrs.update(attrs)
    state.to_netcdf(path)
    return TransitionDataset(root, "test")


def test_replays_a_sample_of_the_current_generator(tmp_path: Path) -> None:
    """generate_data.py's later samples on one model start mid-history."""
    data = _current_generator(
        tmp_path,
        {"inlet_turbulence_seed": 200007, "inlet_turbulence_start_record": 10060},
    )
    footprint = InletTurbulence.from_dataset(data, 0).footprint
    expected = _solver_column(200007, 10060)
    np.testing.assert_allclose(footprint[1:, ..., 1], expected, atol=1e-5)


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


def test_rejects_a_current_generator_sample_without_its_seed(tmp_path: Path) -> None:
    data = _current_generator(tmp_path, {"inlet_turbulence_start_record": 0})
    with pytest.raises(ValueError, match="inlet_turbulence_seed"):
        InletTurbulence.from_dataset(data, 0)
    InletTurbulence.from_dataset(data, 0, seed=1)  # a fresh seed needs none


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


def test_current_generator_reads_its_knot_spacing(tmp_path: Path) -> None:
    footprints = []
    for layout, key in (
        ("archived", "training_data.params_sampler.seconds_per_knot"),
        ("current", "time.seconds_per_knot"),
        ("no_knots", None),
    ):
        root = tmp_path / layout
        root.mkdir()
        _corpus(root) if layout == "archived" else _current_generator(root, {})
        param = root / "param" / "test" / "sample_0000.nc"
        params = xr.load_dataset(param)
        params["velocity_magnitude"][:] = 3.0 + np.sin(np.arange(T_LEN) / 4.0)
        params.to_netcdf(param)
        if key is not None:
            _edit_config(root, key, 4.0)
        data = TransitionDataset(root, "test")
        footprints.append(InletTurbulence.from_dataset(data, 0, seed=1).footprint)
    torch.testing.assert_close(footprints[1], footprints[0])
    assert not torch.allclose(footprints[1], footprints[2])


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


def _full_reach(n_frames: int, n_cols: int) -> torch.Tensor:
    reach = torch.full((n_frames, NZ), n_cols)
    reach[:, 0] = 0  # a ground layer
    return reach


def test_add_is_the_old_behaviour() -> None:
    footprint = torch.randn(4, 3, NZ, NY, 5)
    pred = torch.randn(2, 3, NZ, NY, NX)
    expected = pred.clone()
    expected[..., :5] += footprint[2]
    for turbulence in (
        InletTurbulence(footprint),
        InletTurbulence(footprint, _full_reach(4, 5), mode="add", ramp_width=0),
    ):
        assert torch.equal(turbulence(pred, 2), expected)


def test_replace_keeps_the_mean_flow() -> None:
    """Without a ramp the reached strip is the prediction's y-mean plus the
    forcing; solid levels (reach 0) and the columns downstream are untouched."""
    footprint = torch.randn(4, 3, NZ, NY, 5)
    pred = torch.randn(2, 3, NZ, NY, NX)
    turbulence = InletTurbulence(footprint, _full_reach(4, 5), "replace", 0)
    out = turbulence(pred, 2)
    mean = pred[:, :, 1:, :, :5].mean(dim=-2, keepdim=True)
    torch.testing.assert_close(out[:, :, 1:, :, :5], mean + footprint[2, :, 1:])
    assert torch.equal(out[:, :, 0], pred[:, :, 0])
    assert torch.equal(out[..., 5:], pred[..., 5:])


@pytest.mark.parametrize("ramp_width", [0, 2, 4])
def test_replace_does_not_stack_the_fluctuations(ramp_width: int) -> None:
    """A prediction that is a mean flow plus the forcing comes back unchanged
    everywhere in the reach, ramp included; `add` doubles the forcing."""
    footprint = torch.randn(4, 3, NZ, NY, 5, dtype=torch.float64)
    mean_flow = torch.randn(2, 3, NZ, 1, NX, dtype=torch.float64)
    pred = mean_flow.expand(2, 3, NZ, NY, NX).clone()
    pred[..., :5] += footprint[2] - footprint[2].mean(dim=-2, keepdim=True)
    reach = _full_reach(4, 5)
    replaced = InletTurbulence(footprint, reach, "replace", ramp_width)(pred, 2)
    # The forcing's own y-mean sits on top of the kept mean flow.
    expected = pred.clone()
    expected[:, :, 1:, :, :5] += footprint[2, :, 1:].mean(dim=-2, keepdim=True)
    if ramp_width == 0:
        torch.testing.assert_close(replaced, expected)
    fluct = replaced[:, :, 1:, :, :5] - replaced[:, :, 1:, :, :5].mean(
        dim=-2, keepdim=True
    )
    own = pred[:, :, 1:, :, :5] - pred[:, :, 1:, :, :5].mean(dim=-2, keepdim=True)
    torch.testing.assert_close(fluct, own)  # the forcing's, once
    added = InletTurbulence(footprint)(pred, 2)[:, :, 1:, :, :5]
    doubled = added - added.mean(dim=-2, keepdim=True)
    torch.testing.assert_close(doubled, 2 * own)


@pytest.mark.parametrize("ramp_width", [0, 1, 2, 3, 9])
def test_replace_blends_over_the_ramp(ramp_width: int) -> None:
    """In the ramp the result is the convex blend (1 - w) pred + w (mean_y +
    forcing), w falling linearly by 1/(B+1) per column to 1/(B+1) in the last
    reached column, with B = min(ramp_width, reach // 2): monotone, no jump at
    the reach's edge, and at least half the reach fully replaced."""
    footprint = torch.randn(4, 3, NZ, NY, 8, dtype=torch.float64)
    reach = _full_reach(4, 8)
    reach[2, 1] = 3  # a short reach (near the ground)
    reach[2, 2] = 1
    turbulence = InletTurbulence(footprint, reach, "replace", ramp_width)
    pred = torch.randn(2, 3, NZ, NY, NX, dtype=torch.float64)
    out = turbulence(pred, 2)
    replaced = pred[..., :8].mean(dim=-2, keepdim=True) + footprint[2]
    x = torch.arange(8, dtype=torch.float64)
    for level in range(NZ):
        n = int(reach[2, level])
        ramp = min(ramp_width, n // 2)
        w = ((n - x) / (ramp + 1)).clamp(0, 1)
        expected = (1 - w) * pred[:, :, level, :, :8] + w * replaced[:, :, level]
        torch.testing.assert_close(out[:, :, level, :, :8], expected)
        if n:
            steps = -torch.diff(torch.cat([w[:n], w.new_zeros(1)]))
            assert (steps >= 0).all()  # monotone
            assert steps.max() <= 1 / (ramp + 1) + 1e-9  # no jump
            assert (w[: n - n // 2] == 1).all()  # at least half fully replaced
            assert torch.equal(out[:, :, level, :, n:], pred[:, :, level, :, n:])
    assert torch.equal(out[..., 8:], pred[..., 8:])


def test_replace_needs_a_reach_and_a_known_mode() -> None:
    footprint = torch.zeros(2, 3, NZ, NY, 3)
    with pytest.raises(ValueError, match="reach"):
        InletTurbulence(footprint, mode="replace")
    with pytest.raises(ValueError, match="mode"):
        InletTurbulence(footprint, _full_reach(2, 3), mode="blend")


def test_replace_on_a_corpus_spares_ground_and_unreached_cells(
    tmp_path: Path,
) -> None:
    """From a corpus: frame 0, the ground layer and the columns the frame's new
    inflow did not reach (U*dt/dx = 3) keep the prediction; in the reached
    cells the y-mean is kept and the forcing replaces the fluctuations."""
    data = _corpus(tmp_path)
    turbulence = InletTurbulence.from_dataset(data, 0, mode="replace", ramp_width=0)
    assert turbulence.footprint.shape[-1] == 3
    pred = torch.randn(2, 3, NZ, NY, NX)
    assert torch.equal(turbulence(pred, 0), pred)
    out = turbulence(pred, 5)
    assert torch.equal(out[:, :, 0], pred[:, :, 0])  # ground layer
    assert torch.equal(out[..., 3:], pred[..., 3:])
    mean = pred[:, :, 1:, :, :3].mean(dim=-2, keepdim=True)
    torch.testing.assert_close(
        out[:, :, 1:, :, :3], mean + turbulence.footprint[5, :, 1:]
    )
