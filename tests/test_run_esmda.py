"""End-to-end smoke tests for the unified scripts/run_esmda.py.

Covers the five modes the single script replaces (the old
run_{parameter,state_and_parameter,rollout,time_varying_parameter,
time_varying_parameters_rollout}_esmda.py family) plus a cross-model case and a
disk-loaded-truth case. Everything runs under the tiny `+size=test` overlay with
the global (unlocalized) update — the default correlation localization is
degenerate at this 2-member ensemble size and has its own test.
"""

import pathlib

import pytest


def _overrides(truth_model, assim_model, smoother, prior, num_windows):
    truth = "static_truth" if prior == "static" else "dynamic_truth"
    ov = [
        f"model@truth_model={truth_model}",
        f"model@assim_model={assim_model}",
        f"esmda/smoother={smoother}",
        f"params@prior_params={prior}",
        f"params@truth_params={truth}",
        "esmda.localization=null",
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=2",
        "esmda.num_steps=1",
        f"esmda.num_assimilation_windows={num_windows}",
        "run.skip_viz=true",
    ]
    if prior == "dynamic":
        ov += ["prior_params.time_coords.num=3", "truth_params.time_coords.num=3"]
    if truth_model == "pylbm":
        ov.append("truth_model.forward_model.cuda=false")
    if assim_model == "pylbm":
        ov.append("assim_model.forward_model.cuda=false")
    return ov


@pytest.mark.parametrize(  # type: ignore[misc]
    "truth_model,assim_model,smoother,prior,num_windows",
    [
        # The five modes the single script unifies (pylbm/pylbm).
        pytest.param("pylbm", "pylbm", "parameter", "static", 1, id="parameter"),
        pytest.param(
            "pylbm", "pylbm", "state_and_parameter", "static", 1, id="state_and_param"
        ),
        pytest.param(
            "pylbm", "pylbm", "state_and_parameter", "static", 2, id="rollout"
        ),
        pytest.param("pylbm", "pylbm", "time_varying", "dynamic", 1, id="tv_param"),
        pytest.param("pylbm", "pylbm", "time_varying", "dynamic", 2, id="tv_rollout"),
        # Extra backend coverage on the cheapest mode.
        pytest.param("pyudales", "pyudales", "parameter", "static", 1, id="udales"),
        pytest.param("pylbm", "pyudales", "parameter", "static", 1, id="cross"),
    ],
)
def test_run_esmda(
    truth_model: str,
    assim_model: str,
    smoother: str,
    prior: str,
    num_windows: int,
    compose_test_cfg,
) -> None:
    from scripts.run_esmda import run

    overrides = _overrides(truth_model, assim_model, smoother, prior, num_windows)
    run(compose_test_cfg(overrides, config_name="run_esmda"))


def test_run_esmda_loads_ground_truth_from_disk(
    tmp_path: pathlib.Path, compose_test_cfg
) -> None:
    """run_forward_model.py writes a time-varying ground-truth artifact; run_esmda
    consumes it via run.ground_truth_dir instead of simulating the truth."""
    from scripts.run_forward_model import run as run_forward

    gt_dir = tmp_path / "ground_truth"
    run_forward(
        compose_test_cfg(
            [
                "model=pylbm",
                "model.forward_model.cuda=false",
                "params=dynamic",
                "params.time_coords.num=3",
                "run.time_varying=true",
                "run.skip_viz=true",
                f"run.results_dir={gt_dir}",
                f"paths.base_results_dir={gt_dir}",
            ]
        )
    )
    # run_forward_model writes <out_dir>/<model>_time_varying/{state,params}.nc.
    truth_dir = next(gt_dir.rglob("state.nc")).parent

    from scripts.run_esmda import run

    overrides = _overrides("pylbm", "pylbm", "time_varying", "dynamic", 1)
    overrides.append(f"run.ground_truth_dir={truth_dir}")
    run(compose_test_cfg(overrides, config_name="run_esmda"))
