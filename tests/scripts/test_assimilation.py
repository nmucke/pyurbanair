"""Assimilation runs (run_smoother/filtering/hybrid.py) and their metrics and
figures (compute_metrics.py, visualize_assimilation.py)."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
import xarray
import yaml

from tests.conftest import compose, load_script, surrogate

# Per method: the overrides that make a valid run on top of test/assimilation.
METHODS: dict[str, list[str]] = {
    "smoother": [],
    "filtering": ["params@prior_params=static"],
    "hybrid": [],
}
# Per method: the observation-space stages metrics.yaml scores. A hybrid's
# smoother stage has no analysis, so it has O_N only (no desroziers).
STAGES: dict[str, dict[str, set[str]]] = {
    "smoother": {"smoother": {"forecast_rmse", "data_mismatch"}},
    "filtering": {"filter": {"forecast_rmse", "innovation_chi2_diag"}},
    "hybrid": {"smoother": {"data_mismatch"}, "filter": {"forecast_rmse"}},
}
SURROGATE = [
    "model@truth_model=neural_surrogate_tiny",
    "model@assim_model=neural_surrogate_tiny",
]


def _run(method: str, cfg: Any) -> pathlib.Path:
    """Run the method, then compute its metrics and draw its figures."""
    load_script(f"scripts/run_{method}.py").run(cfg)
    run_dir = pathlib.Path(cfg.paths.results_dir) / method
    load_script("scripts/compute_metrics.py").run(run_dir)
    load_script("scripts/visualize_assimilation.py").run(run_dir)
    return run_dir


def _check_outputs(run_dir: pathlib.Path, num_windows: int = 2) -> None:
    for w in range(num_windows):
        posterior = xarray.load_dataset(
            run_dir / "windows" / f"window_{w}_posterior_params.nc"
        )
        assert posterior.sizes["ensemble"] == 2
        assert (run_dir / "windows" / f"window_{w}_posterior_state.nc").exists()
    metrics = yaml.safe_load((run_dir / "metrics.yaml").read_text())
    assert {"parameters", "state", "sensors"} <= set(metrics)
    for block in ("spread_skill", "climatology"):
        assert set(metrics[block]) == {"assimilation", "validation"}, block
    for name in ("parameter_evolution.png", "mean_slices.png", "tke_slices.png"):
        assert (run_dir / "figures" / name).exists(), name


@pytest.mark.parametrize("method", METHODS)  # type: ignore[misc]
def test_method(
    method: str, tmp_path: pathlib.Path, session_root: pathlib.Path, trained: Any
) -> None:
    cfg = compose(
        "assimilation",
        "+test=assimilation",
        *SURROGATE,
        *surrogate(session_root),
        *METHODS[method],
        root=tmp_path,
    )
    run_dir = _run(method, cfg)
    _check_outputs(run_dir)
    metrics = yaml.safe_load((run_dir / "metrics.yaml").read_text())
    for stage, keys in STAGES[method].items():
        assert keys <= set(metrics["observation"][stage]), stage
    assert set(metrics["desroziers"]) == {
        stage for stage, keys in STAGES[method].items() if "forecast_rmse" in keys
    }
    assert {"ratio", "spread"} <= set(metrics["spread_skill"]["validation"])
    assert "rmse_skill_vs_climatology" in metrics["climatology"]["validation"]


# Per method: the flag that saves a reference ensemble and the block it adds to
# sensor_statistics.
SAVED: dict[str, tuple[str, str]] = {
    "smoother": ("assimilation.save_prior_state=true", "prior"),
    "filtering": ("assimilation.save_forecast_history=true", "forecast"),
    "hybrid": ("assimilation.save_forecast_history=true", "forecast"),
}


@pytest.mark.parametrize("method", METHODS)  # type: ignore[misc]
def test_saved_reference_states(
    method: str, tmp_path: pathlib.Path, session_root: pathlib.Path, trained: Any
) -> None:
    """Saving the prior/forecast states adds their scores, not other changes."""
    flag, reference = SAVED[method]
    run_dirs = {}
    runs: dict[str, list[str]] = {"default": [], "saved": [flag]}
    for name, extra in runs.items():
        cfg = compose(
            "assimilation",
            "+test=assimilation",
            *SURROGATE,
            *surrogate(session_root),
            *METHODS[method],
            *extra,
            root=tmp_path / name,
        )
        load_script(f"scripts/run_{method}.py").run(cfg)
        run_dirs[name] = pathlib.Path(cfg.paths.results_dir) / method
        load_script("scripts/compute_metrics.py").run(run_dirs[name])

    default, saved = (sorted((d / "windows").iterdir()) for d in run_dirs.values())
    assert {f.name for f in default} < {f.name for f in saved}
    for path in default:
        xarray.testing.assert_identical(
            xarray.load_dataset(path),
            xarray.load_dataset(run_dirs["saved"] / "windows" / path.name),
        )
    metrics = yaml.safe_load((run_dirs["saved"] / "metrics.yaml").read_text())
    for name in ("assimilation", "validation"):
        stats = metrics["sensor_statistics"][name]
        assert reference in stats
        assert all(
            f"crps_reduction_vs_{reference}" in entry
            for entry in stats["posterior"].values()
        )


def test_truth_from_a_forward_run(
    tmp_path: pathlib.Path, session_root: pathlib.Path, trained: Any
) -> None:
    """A forward run's state.nc + params.nc serve as the smoother's truth."""
    forward = compose(
        "forward",
        "+test=forward",
        "model=neural_surrogate_tiny",
        *surrogate(session_root),
        "params=dynamic_sine",
        "forward.rollout_steps=1",  # covers both assimilation windows
        root=tmp_path,
    )
    load_script("scripts/run_forward.py").run(forward)
    cfg = compose(
        "assimilation",
        "+test=assimilation",
        *SURROGATE,
        *surrogate(session_root),
        f"assimilation.truth_dir={forward.paths.results_dir}",
        root=tmp_path,
    )
    _check_outputs(_run("smoother", cfg))


@pytest.mark.integration  # type: ignore[misc]
def test_smoother_on_udales(tmp_path: pathlib.Path) -> None:
    cfg = compose("assimilation", "+test=assimilation", root=tmp_path)
    _check_outputs(_run("smoother", cfg))
