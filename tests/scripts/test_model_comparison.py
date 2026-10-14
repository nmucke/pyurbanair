"""scripts/comparison/: run_models, compute_metrics and visualize_comparison.

The metrics and figures run on synthetic fields files (no solver); the
composition of the per-model configs is checked without running them; the
`integration` test runs the whole chain on tiny uDALES and PALM runs.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pytest
import xarray
from omegaconf import OmegaConf

from tests.conftest import compose, load_script

RUN = "scripts/comparison/run_models.py"
METRICS = "scripts/comparison/compute_metrics.py"
FIGURES = "scripts/comparison/visualize_comparison.py"


def comparison(root: pathlib.Path, *overrides: str) -> Any:
    return compose("model_comparison", "+test=tiny", *overrides, root=root)


# ---------------------------------------------------------------------------
# run_models: composition, without running a solver
# ---------------------------------------------------------------------------


def test_model_labels() -> None:
    labels = load_script(RUN).model_labels
    assert labels(["pyudales", "pypalm"]) == {
        "pyudales": "pyudales",
        "pypalm": "pypalm",
    }
    assert labels(["a:pyudales", "b:pyudales"]) == {"a": "pyudales", "b": "pyudales"}
    with pytest.raises(ValueError, match="listed twice"):
        labels(["pyudales", "pyudales"])
    with pytest.raises(ValueError, match="at least one"):
        labels([])


def test_model_config_applies_seed_and_overrides(tmp_path: pathlib.Path) -> None:
    script = load_script(RUN)
    cfg = comparison(
        tmp_path,
        "seed=5",
        "+overrides.pypalm.model.forward_model.boundary_condition=inflow_outflow",
        "+overrides.pypalm.time.spinup_time=7",
    )
    palm = script._model_config(cfg, "pypalm", "pypalm", 1, tmp_path)
    assert palm.model.name == "pypalm"
    assert palm.model.forward_model.random_seed == 6
    assert palm.model.forward_model.inlet_turbulence.initial_seed is True
    assert palm.model.forward_model.boundary_condition == "inflow_outflow"
    assert palm.time.spinup_time == 7 and palm.model.forward_model.spinup_time == 7
    assert palm.paths.results_dir == str(tmp_path / "runs" / "pypalm" / "replica_1")
    udales = script._model_config(cfg, "pyudales", "pyudales", 0, tmp_path)
    assert udales.model.forward_model.inlet_turbulence.seed == 5
    assert udales.model.forward_model.experiment_name == "005"
    assert udales.time.spinup_time == cfg.time.spinup_time  # overrides are per label
    lbm = script._model_config(cfg, "pylbm", "pylbm", 0, tmp_path)
    assert script._seed(cfg, lbm, 0) is None  # pylbm seeds itself
    # Every model resolves on the comparison's case grid.
    OmegaConf.to_container(palm, resolve=True)
    assert palm.model.forward_model.nx == cfg.domain.nx


def test_run_models_refuses_unknown_override_labels(tmp_path: pathlib.Path) -> None:
    cfg = comparison(tmp_path, "+overrides.pylbm.model.forward_model.ncpu=2")
    with pytest.raises(ValueError, match="not in models"):
        load_script(RUN).run(cfg)


# ---------------------------------------------------------------------------
# compute_metrics and visualize_comparison on synthetic fields
# ---------------------------------------------------------------------------


def test_pair_rows_combine_replicas() -> None:
    script = load_script(METRICS)
    items = {
        "a": [np.zeros(4), np.ones(4)],  # replica distance 1
        "b": [np.full(4, 3.0), np.full(4, 3.0)],  # no noise
    }
    rows = {
        (r["model_a"], r["model_b"]): r
        for r in script.pair_rows("field", "q", "all", items, script.squared_error)
    }
    assert rows[("a", "a")]["value"] == pytest.approx(1.0)
    assert rows[("b", "b")]["value"] == pytest.approx(0.0)
    pair = rows[("a", "b")]
    # a_i - b_j is -3 or -2: RMSE sqrt((9 + 4) / 2), bias -2.5.
    assert pair["value"] == pytest.approx(np.sqrt(6.5))
    assert pair["bias"] == pytest.approx(-2.5)
    assert pair["noise"] == pytest.approx(np.sqrt(0.5))
    assert pair["ratio"] == pytest.approx(np.sqrt(6.5) / np.sqrt(0.5))


def test_field_statistics() -> None:
    script = load_script(METRICS)
    t = np.arange(4.0)
    u = np.broadcast_to((2.0 + t)[:, None, None, None], (4, 1, 1, 1))
    ds = xarray.Dataset(
        {c: (("time", "z", "y", "x"), u * (1.0 if c == "u" else 0.0)) for c in "uvw"}
    )
    stats = script.field_statistics(ds)
    assert stats["mean_u"].item() == pytest.approx(3.5)
    assert stats["uu"].item() == pytest.approx(np.var(t, ddof=1))
    assert stats["tke"].item() == pytest.approx(0.5 * np.var(t, ddof=1))
    assert stats["speed"].item() == pytest.approx(3.5)
    assert stats["uw"].item() == 0.0


def _fields(
    rng: np.random.Generator, offset: float, frames: int = 16
) -> xarray.Dataset:
    """A fields file: u = offset + white noise, v and w noise, one building."""
    nz, ny, nx = 4, 6, 8
    solid = np.zeros((nz, ny, nx), bool)
    solid[:2, 2:4, 3:5] = True
    fluid = ~solid
    fluid[:3, 1:5, 2:6] = False
    noise = rng.normal(0.0, 0.5, (3, frames, nz, ny, nx))
    noise[0] += offset
    noise[:, :, solid] = np.nan
    sensors = rng.normal(0.0, 0.5, (3, frames, 2))
    sensors[0] += offset
    return xarray.Dataset(
        {
            **{c: (("time", "z", "y", "x"), noise[i]) for i, c in enumerate("uvw")},
            **{
                f"sensor_{c}": (("time", "sensor"), sensors[i])
                for i, c in enumerate("uvw")
            },
            "solid": (("z", "y", "x"), solid.astype(np.int8)),
            "fluid": (("z", "y", "x"), fluid.astype(np.int8)),
        },
        coords={
            "time": np.arange(1.0, frames + 1),
            "z": np.arange(nz) + 0.5,
            "y": np.arange(ny) + 0.5,
            "x": np.arange(nx) + 0.5,
            "sensor_set": ("sensor", ["assimilation", "validation"]),
            "sensor_x": ("sensor", [0.5, 7.5]),
            "sensor_y": ("sensor", [0.5, 5.5]),
            "sensor_z": ("sensor", [0.5, 0.5]),
        },
    )


@pytest.fixture  # type: ignore[misc]
def synthetic_run(tmp_path: pathlib.Path) -> pathlib.Path:
    """Three models, two replicas each: `same` as `base`, `fast` u + 2 m/s."""
    cfg = comparison(tmp_path, "models=[base,same,fast]")
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    rng = np.random.default_rng(0)
    models = {}
    for label, offset, solver in (
        ("base", 1.0, "udales"),
        ("same", 1.0, "palm"),
        ("fast", 3.0, "pylbm"),
    ):
        for r in range(2):
            path = tmp_path / "fields" / label / f"replica_{r}.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            _fields(rng, offset).to_netcdf(path)
        models[label] = {
            "model": label,
            "solver_name": solver,
            "overrides": {},
            "replicas": [
                {"seed": r, "wall_time_s": 1.0, "run_dir": ""} for r in (0, 1)
            ],
        }
    OmegaConf.save(OmegaConf.create(models), tmp_path / "models.yaml")
    return tmp_path


def test_compute_metrics_on_synthetic_fields(synthetic_run: pathlib.Path) -> None:
    load_script(METRICS).run(synthetic_run)
    meta = json.loads((synthetic_run / "metrics.json").read_text())
    rows = {
        (r["metric"], r["quantity"], r["region"], r["model_a"], r["model_b"]): r
        for r in meta["rows"]
    }
    # The building is two cells high: H = 2 m.
    assert meta["canopy_height"] == pytest.approx(2.0)
    assert meta["height_bands"]["canopy"] == pytest.approx([0.5, 2.0])
    assert meta["replicas"] == {"base": 2, "same": 2, "fast": 2}
    same = rows[("field", "mean_u", "all", "base", "same")]
    fast = rows[("field", "mean_u", "all", "base", "fast")]
    assert same["ratio"] == pytest.approx(1.0, abs=0.2)  # realisation noise only
    assert fast["bias"] == pytest.approx(-2.0, abs=0.1)
    assert fast["ratio"] > 10
    # Noise of the time mean of 16 frames of std 0.5: sqrt(2) * 0.5 / 4.
    noise = rows[("field", "mean_u", "all", "base", "base")]["value"]
    assert noise == pytest.approx(np.sqrt(2) * 0.5 / 4, rel=0.15)
    for key in (
        ("field", "tke", "above_canopy", "base", "fast"),
        ("profile", "uw", "all", "same", "fast"),
        ("sensor_mean", "speed", "validation", "base", "same"),
        ("sensor_w2", "u", "all", "base", "fast"),
        ("sensor_lsd", "w", "assimilation", "base", "fast"),
    ):
        assert np.isfinite(rows[key]["ratio"]), key
    assert rows[("sensor_w2", "u", "all", "base", "fast")]["ratio"] > 3
    csv = (synthetic_run / "metrics.csv").read_text().splitlines()
    assert csv[0].startswith("metric,quantity,region,model_a,model_b")
    assert len(csv) == len(meta["rows"]) + 1

    metrics = xarray.load_dataset(synthetic_run / "metrics.nc")
    assert metrics.stat.dims == ("label", "quantity", "z", "y", "x")
    assert list(metrics.label.values) == ["base", "same", "fast"]
    assert metrics.profile.sizes["replica"] == 2
    assert "spectrum" in metrics
    summary = (synthetic_run / "summary.md").read_text()
    assert "base vs fast" in summary and "noise same" in summary


def test_visualize_comparison_on_synthetic_fields(synthetic_run: pathlib.Path) -> None:
    load_script(METRICS).run(synthetic_run)
    load_script(FIGURES).run(synthetic_run)
    for name in (
        "profiles",
        "slices_speed",
        "slices_tke",
        "sensor_series",
        "sensor_spectra",
        "difference_noise",
    ):
        assert (synthetic_run / f"{name}.png").exists(), name


def test_single_replica_has_no_ratio(synthetic_run: pathlib.Path) -> None:
    models = OmegaConf.load(synthetic_run / "models.yaml")
    for entry in models.values():
        entry.replicas = entry.replicas[:1]
    OmegaConf.save(models, synthetic_run / "models.yaml")
    load_script(METRICS).run(synthetic_run)
    rows = json.loads((synthetic_run / "metrics.json").read_text())["rows"]
    assert all(r["ratio"] is None for r in rows)
    assert "No model has two replicas" in (synthetic_run / "summary.md").read_text()


# ---------------------------------------------------------------------------
# The whole chain on tiny uDALES and PALM runs
# ---------------------------------------------------------------------------


@pytest.mark.integration  # type: ignore[misc]
def test_compare_udales_and_palm(tmp_path: pathlib.Path) -> None:
    """Two replicas each of tiny uDALES and PALM, inflow_outflow, 10 s."""
    run_dir = tmp_path / "comparison"
    cfg = comparison(
        tmp_path,
        "models=[pyudales,pypalm]",
        "replicas=2",
        "time.simulation_time=10",
        f"paths.results_dir={run_dir}",
        # configs/model/*_tiny.yaml, as per-model overrides.
        "+overrides.pyudales.model.forward_model.nudging_config.nnudge_meters=4",
        "+overrides.pypalm.model.forward_model.ncpu=1",
        "+overrides.pypalm.model.forward_model.nz=16",
        "+overrides.pypalm.model.forward_model.nudging_config.nnudge_meters=4",
        "+overrides.pypalm.model.forward_model.boundary_condition=inflow_outflow",
    )
    load_script(RUN).run(cfg)
    models = OmegaConf.load(run_dir / "models.yaml")
    assert list(models) == ["pyudales", "pypalm"]
    assert [r.seed for r in models.pypalm.replicas] == [1, 2]
    assert models.pypalm.overrides["model.forward_model.nz"] == 16
    grid = None
    for label in models:
        for r in (0, 1):
            ds = xarray.load_dataset(run_dir / "fields" / label / f"replica_{r}.nc")
            assert ds.u.dims == ("time", "z", "y", "x")
            assert ds.sizes["z"] == cfg.domain.nz  # PALM's 16 levels interpolated
            assert np.isfinite(ds.u.values[:, ds.fluid.values == 1]).all()
            assert np.isnan(ds.u.values[:, ds.solid.values == 1]).all()
            assert ds.sensor_u.sizes["sensor"] == 5  # 4 obs + 1 validation
            coords = [ds[d].values for d in ("z", "y", "x")]
            grid = grid or coords
            for a, b in zip(grid, coords):
                np.testing.assert_array_equal(a, b)
        replicas = [
            xarray.load_dataset(run_dir / "fields" / label / f"replica_{r}.nc")
            for r in (0, 1)
        ]
        assert not np.array_equal(replicas[0].u.values, replicas[1].u.values), label

    load_script(METRICS).run(run_dir)
    load_script(FIGURES).run(run_dir)
    rows = json.loads((run_dir / "metrics.json").read_text())["rows"]
    pair = next(
        r
        for r in rows
        if (r["metric"], r["quantity"], r["region"], r["model_a"], r["model_b"])
        == ("field", "mean_u", "all", "pyudales", "pypalm")
    )
    assert pair["ratio"] is not None and pair["ratio"] > 0
    assert (run_dir / "difference_noise.png").exists()
