"""Smoke tests for scripts/run_forward_model.py.

Exercises the forward-model runner across the full knob matrix it exposes (minus
the esmda paths, refactored separately):

  * backend:    pylbm, pyudales
  * parameters: static  (`params=static`,  no `time` dim)
                dynamic (`params=dynamic`, time-varying inflow)
  * rollout:    off (`run.rollout_steps=0`, a single window)
                on  (`run.rollout_steps=2`, three stitched windows)
  * ensemble:   single member vs. an N-member ensemble (`run.ensemble=true`,
                sized by the `ensemble.*` fields of the `+size=test` overlay)

These are integration-style smoke tests: the goal is only to confirm the script
runs end to end on every combination, not to assert on the physics. Everything
is sized down hard by the `+size=test` overlay injected in conftest.

Two overrides exist purely to make the script composable outside `hydra.main`:
`paths.experiment_dir` (normally `${hydra:runtime.cwd}/.temp`, which needs a live
HydraConfig) and `paths.base_results_dir` (the fallback `resolve_output_dir`
reads when HydraConfig is absent — only hit on the dynamic write path). Both are
redirected into the test's tmp_path.
"""

import pathlib

import pytest


def _overrides(
    model: str,
    params: str,
    rollout_steps: int,
    ensemble: bool,
    tmp_path: pathlib.Path,
):
    overrides = [
        f"model={model}",
        f"params={params}",
        f"run.rollout_steps={rollout_steps}",
        f"run.ensemble={str(ensemble).lower()}",
        "run.skip_viz=true",
        # Concrete dirs so the script composes without a live HydraConfig.
        f"paths.experiment_dir={tmp_path / 'experiment'}",
        f"++paths.base_results_dir={tmp_path / 'results'}",
    ]
    if model == "pylbm":
        overrides.append("model.forward_model.cuda=false")
    return overrides


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("pylbm", id="pylbm"),
        pytest.param("pyudales", id="pyudales"),
    ],
)
@pytest.mark.parametrize(
    "params",
    [
        pytest.param("static", id="static"),
        pytest.param("dynamic", id="dynamic"),
    ],
)
@pytest.mark.parametrize(
    "rollout_steps",
    [
        pytest.param(0, id="no_rollout"),
        pytest.param(2, id="rollout"),
    ],
)
@pytest.mark.parametrize(
    "ensemble",
    [
        pytest.param(False, id="single"),
        pytest.param(True, id="ensemble"),
    ],
)
def test_run_forward_model(
    model: str,
    params: str,
    rollout_steps: int,
    ensemble: bool,
    tmp_path: pathlib.Path,
    compose_test_cfg,
) -> None:
    """run_forward_model.py runs end to end for every backend × params × rollout
    × single/ensemble combination."""
    from scripts.run_forward_model import run

    run(compose_test_cfg(_overrides(model, params, rollout_steps, ensemble, tmp_path)))
