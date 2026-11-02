"""scripts/surrogate_comparison/: run_models, compute_metrics and visualize_comparison on
the tiny trained steppers.

`comparison_run` is one run dir shared by the module: run_models on the tiny
test split with the tiny Local-FNO (H=2) and DFT (H=1), on the CPU.
"""

from __future__ import annotations

import pathlib
import shutil
from typing import Any

import numpy as np
import pytest
import torch
import xarray
from omegaconf import OmegaConf

from tests.conftest import compose, inlet_turbulence_corpus, load_script

MODELS = ("local_fno", "dft")


def run_models(
    session_root: pathlib.Path, run_dir: pathlib.Path, *overrides: str
) -> Any:
    """Compose configs/comparison.yaml and run run_models.py into `run_dir`."""
    cfg = compose(
        "comparison",
        "device=cpu",
        f"paths.results_dir={run_dir}",
        *overrides,
        root=session_root,
    )
    load_script("scripts/surrogate_comparison/run_models.py").run(cfg)
    return cfg


@pytest.fixture(scope="module")  # type: ignore[misc]
def model_dirs(
    session_root: pathlib.Path, trained: dict[str, Any]
) -> list[pathlib.Path]:
    """The tiny DFT (from `trained`) and a tiny Local-FNO, both on the tiny corpus."""
    torch.set_num_threads(1)
    cfg = compose(
        "surrogate/baselines/local_fno/train", "+test=train", root=session_root
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    weights = session_root / "weights"
    return [weights / cfg.name, weights / trained["train_dft"].name]


@pytest.fixture(scope="module")  # type: ignore[misc]
def comparison_run(
    session_root: pathlib.Path, model_dirs: list[pathlib.Path]
) -> pathlib.Path:
    run_dir = session_root / "comparison"
    run_models(session_root, run_dir, f"models=[{model_dirs[0]},{model_dirs[1]}]")
    return run_dir


def test_run_models_writes_the_run_dir(comparison_run: pathlib.Path) -> None:
    cfg = OmegaConf.load(comparison_run / "config.yaml")
    assert cfg.split == "test" and len(cfg.models) == 2
    models = OmegaConf.load(comparison_run / "models.yaml")
    assert list(models) == list(MODELS)
    assert [models[m].num_history_steps for m in MODELS] == [2, 1]
    for label in MODELS:
        assert models[label].n_parameters > 0
        assert models[label].seconds_per_step > 0
        assert pathlib.Path(models[label].model_dir).is_dir()
        assert models[label].data.param_vars == [
            "inflow_angle",
            "velocity_magnitude",
            "pressure_gradient_magnitude",
        ]
    assert sorted(p.name for p in (comparison_run / "rollouts").iterdir()) == sorted(
        MODELS
    )  # no persistence


def test_rollouts_start_from_a_common_frame(
    comparison_run: pathlib.Path, training_data: pathlib.Path
) -> None:
    truth = xarray.load_dataset(training_data / "state" / "test" / "sample_0000.nc")
    start = 1  # max(H) - 1
    for label in MODELS:
        ds = xarray.load_dataset(comparison_run / "rollouts" / label / "sample_0000.nc")
        assert set(ds.data_vars) == {"u", "v", "w"}
        assert ds.u.dims == ("time", "z", "y", "x") and ds.u.dtype == np.float32
        assert (ds.attrs["sample"], ds.attrs["start"]) == (0, start)
        assert ds.attrs["split"] == "test"
        np.testing.assert_array_equal(ds.time, truth.time[start + 1 :])
        np.testing.assert_array_equal(ds.lead, np.arange(1, truth.sizes["time"] - 1))
        np.testing.assert_array_equal(ds.z, truth.zt)
        assert np.isfinite(ds.u.values).all()


def test_run_models_max_steps_and_sample_indices(
    tmp_path: pathlib.Path, session_root: pathlib.Path, model_dirs: list[pathlib.Path]
) -> None:
    run_models(
        session_root,
        tmp_path,
        f"models=[{model_dirs[0]}]",
        "sample_indices=[0]",
        "max_steps=3",
    )
    ds = xarray.load_dataset(tmp_path / "rollouts" / "local_fno" / "sample_0000.nc")
    assert list(ds.lead.values) == [1, 2, 3]
    with pytest.raises(ValueError, match="max_steps"):
        run_models(session_root, tmp_path, f"models=[{model_dirs[0]}]", "max_steps=0")


def test_run_models_refuses_other_data(
    tmp_path: pathlib.Path, session_root: pathlib.Path, model_dirs: list[pathlib.Path]
) -> None:
    # An explicit list equal to the corpus' names matches `param_vars: null`...
    explicit, other = tmp_path / "explicit", tmp_path / "other"
    for copy, param_vars in (
        (
            explicit,
            ["inflow_angle", "velocity_magnitude", "pressure_gradient_magnitude"],
        ),
        (other, ["inflow_angle"]),
    ):
        shutil.copytree(model_dirs[0], copy)
        saved = OmegaConf.load(copy / "config.yaml")
        saved.dataset.param_vars = param_vars
        OmegaConf.save(saved, copy / "config.yaml")
    run_models(
        session_root,
        tmp_path / "ok",
        f"models=[{model_dirs[0]},{explicit}]",
        "max_steps=1",
    )
    # ... another list does not.
    with pytest.raises(ValueError, match="different data"):
        run_models(session_root, tmp_path / "no", f"models=[{model_dirs[0]},{other}]")


def test_run_models_refuses_another_cadence(
    tmp_path: pathlib.Path,
    session_root: pathlib.Path,
    training_data: pathlib.Path,
    model_dirs: list[pathlib.Path],
) -> None:
    """Two corpora with the same folder name but another output frequency."""
    corpus = tmp_path / "elsewhere" / training_data.name
    shutil.copytree(training_data, corpus)
    data_cfg = OmegaConf.load(corpus / "config.yaml")
    data_cfg.time.output_frequency = 2 * float(data_cfg.time.output_frequency)
    OmegaConf.save(data_cfg, corpus / "config.yaml")
    other = tmp_path / "other"
    shutil.copytree(model_dirs[0], other)
    saved = OmegaConf.load(other / "config.yaml")
    saved.paths.data_dir = str(corpus)
    OmegaConf.save(saved, other / "config.yaml")
    with pytest.raises(ValueError, match="different data"):
        run_models(session_root, tmp_path / "out", f"models=[{model_dirs[0]},{other}]")


def test_run_models_labels(
    tmp_path: pathlib.Path, session_root: pathlib.Path, model_dirs: list[pathlib.Path]
) -> None:
    seeds = [tmp_path / f"seed{i}" / "local_fno" for i in range(2)]
    for seed in seeds:
        shutil.copytree(model_dirs[0], seed)
    run_models(
        session_root, tmp_path / "out", f"models=[{seeds[0]},{seeds[1]}]", "max_steps=1"
    )
    labels = ["seed0_local_fno", "seed1_local_fno"]
    assert list(OmegaConf.load(tmp_path / "out" / "models.yaml")) == labels
    assert sorted(p.name for p in (tmp_path / "out" / "rollouts").iterdir()) == labels
    reserved = tmp_path / "persistence"
    shutil.copytree(model_dirs[0], reserved)
    with pytest.raises(ValueError, match="reserved"):
        run_models(session_root, tmp_path / "r", f"models=[{reserved}]")
    same = model_dirs[0].parent / ".." / model_dirs[0].parent.name / model_dirs[0].name
    with pytest.raises(ValueError, match="listed twice"):
        run_models(session_root, tmp_path / "d", f"models=[{model_dirs[0]},{same}]")


def test_run_models_rerun_replaces_the_rollouts(
    tmp_path: pathlib.Path, comparison_run: pathlib.Path, session_root: pathlib.Path
) -> None:
    """Rollouts left by an earlier run into the same dir are not scored."""
    run_dir = tmp_path / "run"
    shutil.copytree(comparison_run, run_dir)
    stale = run_dir / "rollouts" / "dft" / "sample_0007.nc"
    shutil.copy(stale.with_name("sample_0000.nc"), stale)
    models = OmegaConf.load(run_dir / "config.yaml").models
    run_models(session_root, run_dir, f"models=[{models[0]}]", "max_steps=1")
    assert not (run_dir / "rollouts" / "dft").exists()
    assert list(OmegaConf.load(run_dir / "models.yaml")) == ["local_fno"]
    assert list(OmegaConf.load(run_dir / "config.yaml").sample_indices) == [0]


def test_run_models_inlet_turbulence(
    tmp_path: pathlib.Path,
    session_root: pathlib.Path,
    training_data: pathlib.Path,
    comparison_run: pathlib.Path,
) -> None:
    """The forcing changes the first predicted frame only upstream of the
    building (x < 7) and is recorded; the default run records nothing."""
    corpus = inlet_turbulence_corpus(training_data, tmp_path)
    run_dir = tmp_path / "run"
    models = OmegaConf.load(comparison_run / "config.yaml").models
    run_models(
        session_root,
        run_dir,
        f"models=[{models[0]},{models[1]}]",
        f"data_dir={corpus}",
        "max_steps=2",
        "inlet_turbulence.enabled=true",
    )
    recorded = OmegaConf.load(run_dir / "models.yaml")
    default = OmegaConf.load(comparison_run / "models.yaml")
    for label in MODELS:
        assert recorded[label].inlet_turbulence == {
            "enabled": True,
            "seed": None,
            "apply": "output",
        }
        assert "inlet_turbulence" not in default[label]
        name = pathlib.Path(label) / "sample_0000.nc"
        forced = xarray.load_dataset(run_dir / "rollouts" / name)
        plain = xarray.load_dataset(comparison_run / "rollouts" / name)
        assert forced.attrs["inlet_turbulence_seed"] == "truth"
        assert "inlet_turbulence_seed" not in plain.attrs
        assert "inlet_turbulence_apply" not in forced.attrs
        for var in ("u", "v", "w"):
            first, ref = forced[var].values[0], plain[var].values[0]
            assert np.isfinite(forced[var].values).all()
            assert not np.allclose(first[..., 0], ref[..., 0]), (label, var)
            np.testing.assert_array_equal(first[..., 7:], ref[..., 7:])


def test_run_models_inlet_turbulence_input(
    tmp_path: pathlib.Path,
    session_root: pathlib.Path,
    training_data: pathlib.Path,
    comparison_run: pathlib.Path,
) -> None:
    """`apply: input` stores the model's own output: the first predicted frame
    is the unforced one, and the forcing fed back changes the next."""
    corpus = inlet_turbulence_corpus(training_data, tmp_path)
    run_dir = tmp_path / "run"
    models = OmegaConf.load(comparison_run / "config.yaml").models
    run_models(
        session_root,
        run_dir,
        f"models=[{models[0]},{models[1]}]",
        f"data_dir={corpus}",
        "max_steps=2",
        "inlet_turbulence.enabled=true",
        "inlet_turbulence.apply=input",
    )
    recorded = OmegaConf.load(run_dir / "models.yaml")
    for label in MODELS:
        assert recorded[label].inlet_turbulence.apply == "input"
        name = pathlib.Path(label) / "sample_0000.nc"
        forced = xarray.load_dataset(run_dir / "rollouts" / name)
        plain = xarray.load_dataset(comparison_run / "rollouts" / name)
        assert forced.attrs["inlet_turbulence_apply"] == "input"
        for var in ("u", "v", "w"):
            assert np.isfinite(forced[var].values).all()
            np.testing.assert_allclose(
                forced[var].values[0], plain[var].values[0], rtol=1e-5, atol=1e-6
            )
            assert not np.allclose(forced[var].values[1], plain[var].values[1]), (
                label,
                var,
            )


def test_run_models_periodic_y(
    tmp_path: pathlib.Path,
    session_root: pathlib.Path,
    trained: dict[str, Any],
    model_dirs: list[pathlib.Path],
) -> None:
    """periodic_y changes the tiny UNet's predictions at the y edges and is
    recorded; the default run records nothing. The tiny DFT (global mode)
    refuses it: its ny (20) is not a multiple of 16."""
    stepper = session_root / "weights" / trained["train_stepper"].name
    runs = {}
    for enabled in (False, True):
        runs[enabled] = tmp_path / str(enabled)
        run_models(
            session_root,
            runs[enabled],
            f"models=[{stepper}]",
            "max_steps=2",
            f"periodic_y={str(enabled).lower()}",
        )
    label = stepper.name
    recorded = OmegaConf.load(runs[True] / "models.yaml")[label]
    assert recorded.periodic_y is True
    assert "periodic_y" not in OmegaConf.load(runs[False] / "models.yaml")[label]
    name = pathlib.Path("rollouts") / label / "sample_0000.nc"
    periodic = xarray.load_dataset(runs[True] / name)
    plain = xarray.load_dataset(runs[False] / name)
    assert periodic.attrs["periodic_y"] == 1
    assert "periodic_y" not in plain.attrs
    for var in ("u", "v", "w"):
        first, ref = periodic[var].values[0], plain[var].values[0]
        assert np.isfinite(periodic[var].values).all()
        for edge in (0, -1):
            assert not np.allclose(first[:, edge], ref[:, edge]), (var, edge)

    with pytest.raises(ValueError, match="periodic_y: ny"):
        run_models(
            session_root,
            tmp_path / "dft",
            f"models=[{model_dirs[1]}]",
            "max_steps=2",
            "periodic_y=true",
        )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    run_dir: pathlib.Path, *overrides: str
) -> tuple[dict, xarray.Dataset]:
    load_script("scripts/surrogate_comparison/compute_metrics.py").run(
        run_dir, list(overrides)
    )
    return (
        OmegaConf.to_container(OmegaConf.load(run_dir / "metrics.yaml")),
        xarray.load_dataset(run_dir / "metrics.nc"),
    )


def test_compute_metrics_schema(comparison_run: pathlib.Path) -> None:
    metrics, ds = compute_metrics(comparison_run)
    sources = [*MODELS, "persistence"]
    assert list(metrics) == [*sources, "truth", "truth_halves", "reference"]
    for name in sources:
        entry = metrics[name]
        assert list(entry["rmse"]["per_sample"]) == [0]
        assert entry["stable"]["mean"] is True
        for key in (
            "rmse_vs_persistence",
            "acc_horizon_seconds",
            "acc_horizon_censored",
            "vorticity_rmse",
            "rmse_pedestrian",
            "tke_rmse",
            "profile_rmse_uw",
            "spectra_db_u_mid",
            "integral_length",
            "wasserstein_w",
            "divergence_ratio",
        ):
            assert key in entry
    assert metrics["persistence"]["rmse_vs_persistence"]["mean"] == 1.0
    assert metrics["local_fno"]["n_parameters"] > 0
    assert "speedup" not in metrics["local_fno"]  # solver_seconds_per_step: null
    assert list(metrics["truth"]) == ["integral_length"]
    assert "mean_u_rmse" in metrics["truth_halves"]
    assert "rmse" not in metrics["truth_halves"]
    # The tiny building fills levels 0-1 of 6 x 6 columns.
    reference = {k: v["per_sample"][0] for k, v in metrics["reference"].items()}
    assert reference["k_c"] == 2
    assert reference["H"] == pytest.approx(2 * float(ds.z[1] - ds.z[0]))
    assert reference["z_c"] == pytest.approx(2.5 * float(ds.z[1] - ds.z[0]))
    assert reference["u_ref"] > 0 and reference["T"] > 0

    assert list(ds.source.values) == [*sources, "truth"]
    assert ds.sizes["lead"] == 8 and ds.lead_time.dims == ("lead",)
    np.testing.assert_allclose(ds.lead_over_T, ds.lead_time / ds.attrs["T_mean"])
    for var in (
        "rmse",
        "rmse_u",
        "acc",
        "acc_w",
        "rmse_pedestrian",
        "acc_pedestrian",
        "vorticity_rmse",
        "vorticity_rmse_z",
        "divergence_rms",
        "tke",
        "integral_length",
    ):
        assert ds[var].dims == ("source", "lead"), var
    assert np.isnan(ds.rmse.sel(source="truth")).all()
    for var in ("divergence_rms", "tke", "integral_length", "spectrum_u"):
        assert np.isfinite(ds[var].sel(source="truth")).all(), var
    assert (ds.tke.sel(source="truth") > 0).all()
    assert (ds.tke.sel(source="persistence") == 0).all()
    assert ds.spectrum_w.dims == ("source", "lead", "kh")
    assert (np.diff(ds.kh) > 0).all()
    assert ds.profile_tke.dims == ("source", "z") and ds.z_over_H.dims == ("z",)
    assert ds.pdf.dims == ("source", "pdf_lead", "var", "bin") and ds.sizes["bin"] == 60
    assert list(ds.pdf_lead.values) == [1, 8]  # snapshot_leads [1, 10, 50, -1]
    assert list(ds["var"].values) == ["u", "v", "w"]
    assert np.isfinite(ds.pdf.sel(source="truth")).all()
    assert ds.reference_H.dims == ("sample",) and int(ds.canopy_level[0]) == 2


def test_compute_metrics_exact_cases(
    tmp_path: pathlib.Path, comparison_run: pathlib.Path, training_data: pathlib.Path
) -> None:
    """A rollout equal to the truth scores perfectly; a non-finite one is reported."""
    run_dir = tmp_path / "run"
    shutil.copytree(comparison_run, run_dir)
    truth = xarray.load_dataset(training_data / "state" / "test" / "sample_0000.nc")
    perfect = run_dir / "rollouts" / "dft" / "sample_0000.nc"
    ds = xarray.load_dataset(perfect)
    for var in ("u", "v", "w"):
        ds[var].values = truth[var].values[ds.attrs["start"] + 1 :]
    ds.to_netcdf(perfect)
    broken = run_dir / "rollouts" / "local_fno" / "sample_0000.nc"
    ds = xarray.load_dataset(broken)
    ds["u"][-1] = np.nan
    ds.to_netcdf(broken)

    metrics, nc = compute_metrics(run_dir, "metrics.solver_seconds_per_step=10")
    dft = metrics["dft"]
    for key in ("rmse", "vorticity_rmse", "rmse_pedestrian", "mean_u_rmse"):
        assert dft[key]["mean"] == 0.0, key
    # ACC never drops: censored at the window's last lead time.
    assert dft["acc_horizon_censored"]["mean"] == 1.0
    assert dft["acc_horizon_seconds"]["mean"] == pytest.approx(float(nc.lead_time[-1]))
    for key in ("tke_rmse", "profile_rmse_tke", "wasserstein_u", "spectra_db_u_mid"):
        assert dft[key]["mean"] == 0.0, key
    assert dft["divergence_ratio"]["mean"] == 1.0
    assert dft["integral_length"]["mean"] == metrics["truth"]["integral_length"]["mean"]
    assert dft["speedup"] == pytest.approx(10 / dft["seconds_per_step"])
    for var in ("acc", "acc_u", "acc_pedestrian"):
        np.testing.assert_allclose(nc[var].sel(source="dft"), 1.0, rtol=1e-6)
    for var in ("divergence_rms", "tke", "integral_length", "spectrum_u", "pdf"):
        np.testing.assert_allclose(
            nc[var].sel(source="dft"), nc[var].sel(source="truth"), err_msg=var
        )
    np.testing.assert_allclose(
        nc.profile_uw.sel(source="dft"), nc.profile_uw.sel(source="truth")
    )

    fno = metrics["local_fno"]
    assert fno["stable"]["mean"] is False
    assert np.isnan(fno["rmse"]["mean"]) and np.isnan(fno["tke_rmse"]["mean"])
    for var in ("rmse", "tke", "spectrum_u"):
        assert np.isfinite(nc[var].sel(source="local_fno")[:-1]).all(), var
        assert np.isnan(nc[var].sel(source="local_fno")[-1]).all(), var
    assert np.isfinite(nc.pdf.sel(source="local_fno", pdf_lead=1)).all()
    assert np.isnan(nc.pdf.sel(source="local_fno", pdf_lead=8)).all()
    assert np.isnan(nc.profile_tke.sel(source="local_fno")).all()


def test_compute_metrics_visualize_and_tables_write_every_output(
    comparison_run: pathlib.Path,
) -> None:
    overrides = [
        "figures.sample=0",
        "figures.vertical_lead=4",
        "display_names={dft: Ours}",
    ]
    for script in ("compute_metrics", "visualize_comparison", "make_tables"):
        load_script(f"scripts/surrogate_comparison/{script}.py").run(
            comparison_run, overrides
        )
    figures = [
        "rmse",
        "acc",
        "vorticity",
        "physics",
        "pedestrian",
        "spectra",
        "spectral_ratio",
        "pdfs",
        "profiles",
        "station_profiles",
        "slices",
        "vertical_slices",
        "statistics_slices",
        "tke_heights",
    ]
    for name in figures:
        for suffix in ("pdf", "png"):
            path = comparison_run / f"{name}.{suffix}"
            assert path.stat().st_size > 0, path.name
    assert list(comparison_run.glob("animation_tke_z*H.*"))
    assert (comparison_run / "metrics_reference.pdf").stat().st_size > 0
    for name in ("accuracy", "physics", "cost"):
        tex = (comparison_run / f"table_{name}.tex").read_text()
        assert tex.startswith("\\begin{table}") and "\\label{tab:comparison_" in tex
        assert "Ours" in tex and "dft" not in tex
        rows = (comparison_run / f"table_{name}.csv").read_text().splitlines()
        assert [r.split(",")[0] for r in rows[1:3]] == list(MODELS)


def test_compute_metrics_refuses_rollouts_of_another_split(
    comparison_run: pathlib.Path,
) -> None:
    with pytest.raises(ValueError, match="rerun run_models"):
        compute_metrics(comparison_run, "split=val")
