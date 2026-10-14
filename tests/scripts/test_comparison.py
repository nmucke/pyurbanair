"""scripts/comparison/: run_models, compute_metrics and visualize_comparison on
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

from tests.conftest import compose, load_script

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
    load_script("scripts/comparison/run_models.py").run(cfg)
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


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    run_dir: pathlib.Path, *overrides: str
) -> tuple[dict, xarray.Dataset]:
    load_script("scripts/comparison/compute_metrics.py").run(run_dir, list(overrides))
    return (
        OmegaConf.to_container(OmegaConf.load(run_dir / "metrics.yaml")),
        xarray.load_dataset(run_dir / "metrics.nc"),
    )


def test_compute_metrics_schema(comparison_run: pathlib.Path) -> None:
    metrics, ds = compute_metrics(comparison_run)
    sources = [*MODELS, "persistence"]
    assert list(metrics) == [*sources, "truth_halves"]
    for name in sources:
        entry = metrics[name]
        assert list(entry["rmse"]["per_sample"]) == [0]
        assert entry["stable"]["mean"] is True
        assert "rmse_step1" in entry and "rmse_step50" not in entry  # 8 leads
        for key in (
            "acc_horizon_steps",
            "tke_rmse",
            "profile_rmse_uw",
            "spectra_db_u_mid",
            "wasserstein_w",
        ):
            assert key in entry
    assert metrics["persistence"]["rmse_vs_persistence"]["mean"] == 1.0
    assert metrics["local_fno"]["n_parameters"] > 0
    assert "mean_u_rmse" in metrics["truth_halves"]
    assert "rmse" not in metrics["truth_halves"]

    assert list(ds.source.values) == [*sources, "truth"]
    assert ds.sizes["lead"] == 8 and ds.lead_time.dims == ("lead",)
    for var in ("rmse", "rmse_u", "acc", "std_ratio_w", "divergence_rms"):
        assert ds[var].dims == ("source", "lead")
    assert np.isnan(ds.rmse.sel(source="truth")).all()
    assert np.isfinite(ds.divergence_rms.sel(source="truth")).all()
    assert ds.profile_tke.dims == ("source", "z")
    assert ds.pdf.dims == ("source", "var", "bin")
    assert list(ds["var"].values) == ["u", "v", "w"]
    assert np.isfinite(ds.pdf.sel(source="truth")).all()


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

    metrics, nc = compute_metrics(run_dir)
    dft = metrics["dft"]
    assert dft["rmse"]["mean"] == 0.0 and dft["final_rmse"]["mean"] == 0.0
    assert dft["acc_horizon_steps"]["mean"] is None
    for key in ("mean_u_rmse", "tke_rmse", "profile_rmse_tke", "wasserstein_u"):
        assert dft[key]["mean"] == 0.0
    np.testing.assert_allclose(nc.acc.sel(source="dft"), 1.0, rtol=1e-6)
    np.testing.assert_allclose(nc.std_ratio_u.sel(source="dft"), 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        nc.divergence_rms.sel(source="dft"), nc.divergence_rms.sel(source="truth")
    )
    np.testing.assert_allclose(
        nc.profile_uw.sel(source="dft"), nc.profile_uw.sel(source="truth")
    )

    fno = metrics["local_fno"]
    assert fno["stable"]["mean"] is False
    assert np.isnan(fno["final_rmse"]["mean"]) and np.isnan(fno["tke_rmse"]["mean"])
    assert np.isfinite(nc.rmse.sel(source="local_fno")[:-1]).all()
    assert np.isnan(nc.profile_tke.sel(source="local_fno")).all()


def test_compute_metrics_and_visualize_write_every_output(
    comparison_run: pathlib.Path,
) -> None:
    overrides = ["figures.level=1", "figures.spectra_levels=[1]", "figures.sample=0"]
    load_script("scripts/comparison/compute_metrics.py").run(comparison_run, overrides)
    load_script("scripts/comparison/visualize_comparison.py").run(
        comparison_run, overrides
    )
    figures = ["rmse", "acc", "energy", "divergence", "profiles", "spectra", "pdfs"]
    for name in [*figures, "slices", "statistics_slices"]:
        assert (comparison_run / f"{name}.png").stat().st_size > 0, name
    summary = (comparison_run / "summary.csv").read_text().splitlines()
    assert summary[0].startswith("source,rmse,rmse_vs_persistence")
    assert [row.split(",")[0] for row in summary[1:]] == [
        *MODELS,
        "persistence",
        "truth_halves",
    ]
    assert (comparison_run / "summary.tex").exists()
