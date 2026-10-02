"""Assimilation runs (run_smoother/filtering/hybrid.py) and their metrics and
figures (compute_metrics.py, visualize_assimilation.py)."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
import xarray
import yaml

from tests_new.conftest import compose, load_script, surrogate

# Per method: the overrides that make a valid run on top of test/assimilation.
METHODS: dict[str, list[str]] = {
    "smoother": [],
    "filtering": ["params@prior_params=static"],
    "hybrid": [],
}
SURROGATE = [
    "model@truth_model=neural_surrogate_tiny",
    "model@assim_model=neural_surrogate_tiny",
]


def _run(method: str, cfg: Any) -> pathlib.Path:
    """Run the method, then compute its metrics and draw its figures."""
    load_script(f"scripts_new/run_{method}.py").run(cfg)
    run_dir = pathlib.Path(cfg.paths.results_dir) / method
    load_script("scripts_new/compute_metrics.py").run(run_dir)
    load_script("scripts_new/visualize_assimilation.py").run(run_dir)
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
    _check_outputs(_run(method, cfg))


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
    load_script("scripts_new/run_forward.py").run(forward)
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
