"""Correct a finished smoother run's posterior frames, without model runs.

Config: configs/assimilation.yaml, with the overrides of the run; `field_analysis`
is taken from here and everything else from the run's config.yaml, e.g.

    python scripts/run_field_analysis.py 'field_analysis=${field_analysis_settings.local}'

The same correction as run_smoother.py with `field_analysis` set. It starts from
the frames as the model wrote them, so it can be repeated with other settings:
window_{w}_posterior_forecast_state.nc when the run kept it, else the
_posterior_state.nc of a run without `field_analysis`, which is hard-linked
there, so a failure leaves every window's posterior in place and a second call
finishes the job. A run that corrected its frames and did not keep them is
refused. The frames as the model wrote them are always kept here, whatever
`keep_forecast` says.

Outputs, in `<paths.results_dir>/smoother/`:
    windows/window_{w}_posterior_state.nc  the corrected frames
            _posterior_forecast_state.nc   the frames as the model wrote them
            _field_obs.nc                  what the correction used (kept if
                                           the run wrote it, else rebuilt from
                                           the truth and the run's seed)
    run_info.yaml                          `field_analysis` per window,
                                           `field_analysis_settings` as passed
                                           here (config.yaml keeps the run's)
"""

from __future__ import annotations

import pyurbanair.quiet_jax  # noqa: F401  (silences JAX CPU-fallback noise)

import pathlib
import sys

import hydra
import jax
import numpy as np
import xarray
import yaml
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "utils"))

from helper_functions import (  # noqa: E402
    correct_window,
    field_observations,
    flatten_obs,
    make_aggregation,
    make_observation_error,
    make_observation_operator,
    open_truth,
    save_yaml,
    window_observations,
)
from inconsistency_check import check_config  # noqa: E402


def run(cfg: DictConfig) -> None:
    run_dir = pathlib.Path(cfg.paths.results_dir) / "smoother"
    windows_dir = run_dir / "windows"
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    assert isinstance(run_cfg, DictConfig)
    if cfg.get("field_analysis") is None:
        raise ValueError("Set field_analysis: there is nothing to do without it.")
    run_cfg.field_analysis = OmegaConf.to_container(cfg.field_analysis, resolve=True)
    check_config(run_cfg, "smoother")
    num_windows = int(run_cfg.assimilation.num_windows)
    with open(run_dir / "run_info.yaml") as f:
        run_info = yaml.safe_load(f)

    def window_file(w: int, name: str) -> pathlib.Path:
        return windows_dir / f"window_{w}_{name}.nc"

    # Every window's frames as the model wrote them, before anything changes.
    for w in range(num_windows):
        written = window_file(w, "field_obs").exists()
        if written and not window_file(w, "posterior_forecast_state").exists():
            raise ValueError(
                f"Window {w} was corrected in the run without "
                "field_analysis.keep_forecast: the frames as the model wrote "
                "them are gone, and correcting the corrected ones would use "
                "the observations twice."
            )
    for w in range(num_windows):
        if not window_file(w, "posterior_forecast_state").exists():
            window_file(w, "posterior_forecast_state").hardlink_to(
                window_file(w, "posterior_state")
            )

    if not all(window_file(w, "field_obs").exists() for w in range(num_windows)):
        _write_field_observations(run_cfg, run_dir, run_info)
    run_info["field_analysis"] = [
        correct_window(run_cfg, windows_dir, w) for w in range(num_windows)
    ]
    run_info["field_analysis_settings"] = OmegaConf.to_container(run_cfg.field_analysis)
    save_yaml(run_info, run_dir / "run_info.yaml")
    print(f"Saved outputs in {run_dir}")


def _write_field_observations(
    cfg: DictConfig, run_dir: pathlib.Path, run_info: dict
) -> None:
    """The window_{w}_field_obs.nc of a run without `field_analysis`.

    The run's raw frames are not saved: they are drawn again from its seed,
    and must reproduce the observations it assimilated.
    """
    num_windows = int(cfg.assimilation.num_windows)
    truth = open_truth(cfg, run_dir)
    truth_operator = make_observation_operator(cfg, cfg.truth_model.solver_name)
    error = make_observation_error(cfg)
    aggregation = make_aggregation(cfg)
    observations, _ = window_observations(
        truth,
        truth.sizes["time"] // num_windows,
        num_windows,
        truth_operator,
        error,
        aggregation,
        jax.random.PRNGKey(cfg.assimilation.seed),
    )
    truth.close()
    operator = make_observation_operator(cfg, cfg.assim_model.solver_name)
    substitutions = run_info.get("posterior_substitutions", {})
    for w, (noisy, clean, _) in enumerate(observations):
        windows = run_dir / "windows"
        with xarray.open_dataset(windows / f"window_{w}_obs.nc") as saved:
            if not np.allclose(flatten_obs(noisy, aggregation), saved.obs.values):
                raise ValueError(
                    f"The observations drawn again for window {w} differ from "
                    "the run's: its truth or seed has changed."
                )
        field_observations(
            noisy,
            clean,
            truth_operator,
            error,
            operator,
            windows / f"window_{w}_posterior_forecast_state.nc",
            substitutions.get(w, {}),
        ).to_netcdf(windows / f"window_{w}_field_obs.nc")


@hydra.main(version_base=None, config_path="../configs", config_name="assimilation")  # type: ignore[misc, unused-ignore]
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
