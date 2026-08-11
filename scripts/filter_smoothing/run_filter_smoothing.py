"""Run windowed parameter-trajectory ESMDA with an inner EnKF state filter.

The hybrid of the repo's two existing assimilation entry points: the
high-dimensional STATE is estimated by the sequential EnKF of
``scripts/filtering/run_filtering.py`` (never smoothed), while the
low-dimensional parameter TRAJECTORY ``Theta = [theta_0 … theta_L]`` over one
window of ``filter_smoothing.num_cycles`` cycles is updated jointly by an outer
ESMDA loop, as in ``scripts/esmda/run_esmda.py``. Each ESMDA iteration runs one
inner EnKF pass through the whole window from the SAME initial state, records
the raw pre-analysis forecast observations ``d_k = H(x_f_k)``, stacks them into
``D in R^{L*N_d x N_e}`` and applies one tempered Kalman update to the flattened
trajectory; a final EnKF pass makes the returned state consistent with the final
trajectory (see ``docs/plans/filter_smoothing_windowed_esmda.md`` and
``libs/data-assimilation/src/data_assimilation/filter_smoothing/``).

This is the first stage of the usual three-stage shape, mirroring the filtering
pipeline:

  1. scripts/filter_smoothing/run_filter_smoothing.py (THIS) -- runs the method
                                       and saves every raw artifact (the
                                       trajectory posterior/prior params, the
                                       posterior state, the truth state/params,
                                       iteration_diagnostics.yaml,
                                       cycle_diagnostics.yaml of the final pass,
                                       truth_access.yaml and run_info.yaml).
  2. scripts/filtering/compute_filtering_metrics.py -- the per-cycle artifacts
                                       are laid out exactly as run_filtering.py's,
                                       so the filtering metric/figure stages read
                                       this run directory unchanged.

The machinery is declarative (see conf/run_filter_smoothing.yaml):

  * ``filter_smoothing/inner_analysis=stochastic|etkf|etkf_tsvd|letkf|letkf_tsvd``
        the INNER per-cycle state update math, reusing the filtering package's
        analysis schemes: the perturbed-observation stochastic EnKF, or a
        deterministic ensemble transform, globally (``etkf*``, which REQUIRES
        ``filter_smoothing/inner_localization=none``) or per block (``letkf*``,
        which REQUIRES a non-null inner localization).
  * ``filter_smoothing/inner_localization=none|correlation|distance``
        spatial localization of those inner state analyses.
  * ``filter_smoothing/inner_inflation=none|multiplicative|rtps|rtpp``
        inner ensemble spread maintenance.
  * ``filter_smoothing/temporal_localization=none|taper``
        localization of the OUTER trajectory update in TIME (|t_knot - t_obs|),
        a separate axis from the spatial inner localization.
  * ``filter_smoothing.num_steps=N_a`` outer ESMDA iterations,
    ``filter_smoothing.alpha=null`` -> equal weights (alpha = num_steps).

and the truth source mirrors run_filtering.py:

  * ``run.truth_dir=null``    simulate the truth inline (default).
  * ``run.truth_dir=<path>``  load a state.nc/params.nc artifact written by
                              run_forward_model.py.

One cycle consumes one forecast segment of ``time.simulation_time`` seconds:
the truth is generated over ``filter_smoothing.num_cycles`` such segments up
front, each segment's observations are extracted with the case's temporal
observation operator, and the estimator consumes the resulting
``(num_cycles, N_d)`` batch matrix in a single ``run()`` call.

The PRIOR must be a DYNAMIC (time-varying) sampler -- the exact inverse of
run_filtering.py's guard, and the point of the method: what is estimated is a
whole parameter trajectory, one knot per cycle. Pair with
``params@prior_params=dynamic``. The trajectory is sampled ONCE over the full
``num_cycles * time.simulation_time`` horizon, so its knots must be spaced one
cycle apart (``time.seconds_per_knot == time.simulation_time``, the default);
both the configured spacing and the SAMPLED knot times are validated loudly
below, and the trajectory length handed to the estimator is derived from the
sampled prior's ``time`` dim, never from a config literal.

Examples::

    python scripts/filter_smoothing/run_filter_smoothing.py \
        filter_smoothing.num_cycles=8 filter_smoothing.num_steps=4
    python scripts/filter_smoothing/run_filter_smoothing.py \
        filter_smoothing/temporal_localization=taper \
        filter_smoothing.temporal_localization.temporal_radius=2  # in CYCLES
    python scripts/filter_smoothing/run_filter_smoothing.py \
        filter_smoothing/inner_analysis=etkf filter_smoothing/inner_localization=none
    python scripts/filter_smoothing/run_filter_smoothing.py \
        filter_smoothing/inner_analysis=letkf filter_smoothing/inner_localization=distance
"""

import dataclasses
import pathlib
import sys
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    create_observation_operator,
    filter_parameter_config,
)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.esmda._esmda_common import open_truth, truth_x_min, write_yaml

# Relative tolerance for the knot-spacing / horizon checks. The knot axis is
# built in float32 by the samplers (jnp), so an exact comparison against the
# float64 config value would trip on representation alone.
_TIME_RTOL = 1e-4


def run(cfg: DictConfig) -> None:
    num_cycles = int(cfg.filter_smoothing.num_cycles)
    if num_cycles < 1:
        raise ValueError(f"filter_smoothing.num_cycles must be >= 1, got {num_cycles}.")
    num_steps = int(cfg.filter_smoothing.num_steps)
    if num_steps < 1:
        raise ValueError(f"filter_smoothing.num_steps must be >= 1, got {num_steps}.")
    sim_time = float(cfg.time.simulation_time)
    ensemble_size = int(cfg.ensemble.ensemble_size)
    final_time = sim_time * num_cycles
    domain_x_min = float(cfg.domain.bounds[0][0])
    rng_key = jax.random.PRNGKey(cfg.filter_smoothing.seed)

    # This method estimates a parameter TRAJECTORY: one knot per cycle, all
    # knots updated jointly by the outer ESMDA loop. That is the inverse of
    # run_filtering.py's guard (the plain filter refuses a dynamic prior because
    # it tracks a single scalar), so a static prior is the error case here.
    if "seconds_per_knot" not in list(cfg.prior_params.keys()):
        raise ValueError(
            "prior_params is a static (scalar) sampler, which filter smoothing "
            "does not support: it estimates a time-varying parameter trajectory "
            "with one knot per cycle. Use params@prior_params=dynamic (AR(2)), "
            "or scripts/filtering/run_filtering.py for a tracked scalar."
        )
    is_dynamic_truth = "seconds_per_knot" in list(cfg.truth_params.keys())

    # Knot spacing is load-bearing, not decorative: knot k backs cycle k's
    # forecast segment, so the spacing must BE the segment length. Check the
    # configured value up front (the actionable knob) — the sampled knot times
    # are re-checked after sampling, which is what the estimator actually sees.
    seconds_per_knot = float(cfg.time.seconds_per_knot)
    if abs(seconds_per_knot - sim_time) > _TIME_RTOL * max(sim_time, 1.0):
        raise ValueError(
            f"time.seconds_per_knot={seconds_per_knot:g} must equal "
            f"time.simulation_time={sim_time:g}: one parameter knot backs exactly "
            "one cycle's forecast segment. conf/run_filter_smoothing.yaml ties "
            "the two together by default — override time.simulation_time (which "
            "changes the cycle length) rather than the knot spacing."
        )

    # Select which parameters the method estimates (same contract as
    # run_filtering.py / run_esmda.py `params_to_estimate`): null -> every
    # parameter the sampler configs define; a list -> that subset, applied to
    # prior AND truth.
    selected = cfg.get("params_to_estimate", None)
    selected = list(selected) if selected is not None else None
    truth_params_cfg = filter_parameter_config(cfg.truth_params, selected)
    prior_params_cfg = filter_parameter_config(cfg.prior_params, selected)
    if selected is not None:
        print(f"Estimating parameters: {selected}")

    # --- Output dir -----------------------------------------------------------
    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=out_dir / "config.yaml")

    # --- Truth (simulated inline, or loaded from disk) ------------------------
    if cfg.run.truth_dir is None:
        true_forward_model = instantiate(
            cfg.truth_model.forward_model,
            results_dir=None,
            simulation_time=final_time,
        )
        # A dynamic (time-varying) truth must span the FULL horizon so its
        # parameter drifts across every cycle, not just the first: sample its
        # knots over final_time (same as run_filtering.py's is_dynamic branch).
        # A static truth ignores it.
        if is_dynamic_truth:
            truth_sampler = instantiate(truth_params_cfg, simulation_time=final_time)
        else:
            truth_sampler = instantiate(truth_params_cfg)
        true_params = truth_sampler.sample(1)

        instantiate(cfg.truth_model.prepare, forward_model=true_forward_model)
        clean_outputs(model_name=cfg.truth_model.name, forward_model=true_forward_model)
        true_state = true_forward_model(params=true_params.isel(ensemble=0))

        true_state_path = out_dir / "true_state.nc"
        true_state.to_netcdf(true_state_path)
        n_total = int(true_state.sizes["time"])
        x_offset = domain_x_min - truth_x_min(true_state)
        start_idx = 0
        t_offset = 0.0
        del true_state
    else:
        truth_dir = pathlib.Path(cfg.run.truth_dir)
        true_params = xarray.load_dataset(truth_dir / "params.nc")

        # Optionally begin the horizon partway into a pre-simulated truth
        # (skip a spin-up); frames are rebased so that time becomes t=0.
        start_time = float(cfg.run.truth_start_time or 0.0)
        true_state_path = truth_dir / "state.nc"
        with xarray.open_dataset(true_state_path) as _truth_meta:
            true_times = np.asarray(_truth_meta["time"].values, dtype=float)
            x_offset = domain_x_min - truth_x_min(_truth_meta)
        start_idx = int((true_times < start_time).sum())
        t_offset = start_time
        n_total = int(((true_times[start_idx:] - t_offset) < final_time).sum())

    if x_offset:
        print(
            f"Shifting truth x by {x_offset:+g} to align with domain x_min={domain_x_min:g}"
        )

    # Number of truth frames per cycle (contiguous, half-open blocks). Guard
    # the degenerate slicings explicitly: zero frames per cycle would feed the
    # observation operator empty segments, and a remainder would be dropped
    # silently.
    if n_total < num_cycles:
        raise ValueError(
            f"The truth provides {n_total} frame(s) within the {final_time:g}s "
            f"horizon, fewer than filter_smoothing.num_cycles={num_cycles} (each "
            "cycle needs at least one frame). Increase time.simulation_time, "
            "reduce filter_smoothing.num_cycles, or point run.truth_dir at a "
            "longer truth."
        )
    n_per_cycle = n_total // num_cycles
    n_dropped = n_total - n_per_cycle * num_cycles
    if n_dropped:
        print(
            f"Truth frames ({n_total}) do not divide evenly into {num_cycles} "
            f"cycles of {n_per_cycle}; the trailing {n_dropped} frame(s) are "
            "not assimilated."
        )
    true_params.to_netcdf(out_dir / "true_params.nc")

    # --- Assimilation ensemble model ------------------------------------------
    assim_results_dir = (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
    )
    assim_model = instantiate(
        cfg.assim_model.forward_model, results_dir=assim_results_dir
    )
    instantiate(cfg.assim_model.prepare, forward_model=assim_model)

    # Optional on-disk ensemble forecasts (one NetCDF per member per cycle,
    # under cycle_{k}/ dirs) — keeps a large ensemble field off host RAM. Every
    # outer iteration rewrites the same cycle dirs, so only the final pass's
    # files survive. The analyzed end-of-cycle state is carried in memory either
    # way.
    ensemble_states_dir = (
        out_dir / "_ensemble_states"
        if bool(cfg.run.get("ensemble_save_on_disk", False))
        else None
    )
    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=assim_model,
        results_dir=ensemble_states_dir,
    )

    # --- Prior parameter TRAJECTORY ensemble -----------------------------------
    # Sampled ONCE over the whole window (not per cycle): the knots ARE the
    # estimated quantity Theta, and the outer update touches all of them at once.
    prior_sampler = instantiate(prior_params_cfg, simulation_time=final_time)
    prior_params = prior_sampler.sample(ensemble_size)
    if "time" not in prior_params.dims:
        raise ValueError(
            "The prior sampler produced no `time` dimension, so there is no "
            "parameter trajectory to smooth. Use params@prior_params=dynamic."
        )
    prior_params.to_netcdf(out_dir / "prior_params.nc")

    # Validate the SAMPLED knot layout (what the estimator actually consumes),
    # not just the configured spacing: knot j must sit at the start of cycle j.
    # `build_knot_times` emits num_cycles+1 knots for a num_cycles*sim_time
    # horizon — the trailing knot at final_time is legitimate and rides along in
    # Theta, updated only through the prior's temporal correlations (it becomes
    # the leading edge of a future moving window; plan §5/§9).
    knot_times = np.asarray(prior_params["time"].values, dtype=float)
    n_knots = int(knot_times.size)
    knot_gaps = np.diff(knot_times)
    if knot_gaps.size and not np.allclose(
        knot_gaps, sim_time, rtol=_TIME_RTOL, atol=_TIME_RTOL * max(sim_time, 1.0)
    ):
        raise ValueError(
            f"The sampled prior's knots are not one cycle apart: spacings "
            f"{np.unique(np.round(knot_gaps, 6)).tolist()} vs the cycle length "
            f"time.simulation_time={sim_time:g}. Knot k must back cycle k's "
            "forecast segment; set time.seconds_per_knot = time.simulation_time "
            "and make sure the horizon is an exact multiple of it (an uneven "
            "final interval appends an extrapolated endpoint knot)."
        )
    if n_knots < num_cycles:
        raise ValueError(
            f"The sampled prior has {n_knots} knot(s), which does not cover "
            f"filter_smoothing.num_cycles={num_cycles}: cycles beyond the last "
            "knot would silently reuse it. Sample the prior over the full "
            f"{final_time:g}s window."
        )
    if n_knots > num_cycles + 1:
        raise ValueError(
            f"The sampled prior has {n_knots} knot(s) for {num_cycles} cycle(s) "
            f"(expected {num_cycles} or {num_cycles + 1}): the trajectory extends "
            "past the assimilation window, so its trailing knots are never "
            "constrained by any observation batch. Check time.simulation_time "
            "and filter_smoothing.num_cycles."
        )
    print(
        f"Prior trajectory: {n_knots} knot(s) over {final_time:g}s "
        f"({num_cycles} cycle(s) of {sim_time:g}s)"
        + (
            "; the trailing knot is updated only through prior temporal correlation."
            if n_knots == num_cycles + 1
            else "."
        )
    )

    # --- Observation operators and per-cycle observations ----------------------
    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)

    obs_rows = []
    for cycle in range(num_cycles):
        cycle_truth = open_truth(
            true_state_path, n_total, x_offset, start_idx, t_offset
        ).isel(time=slice(cycle * n_per_cycle, (cycle + 1) * n_per_cycle))
        obs_rows.append(jnp.asarray(truth_obs_op(cycle_truth)))
        cycle_truth.close()
    observations = jnp.stack(obs_rows, axis=0)  # (num_cycles, N_d)

    obs_error_std = float(cfg.filter_smoothing.obs_error_std)
    C_D_diag = (obs_error_std**2) * jnp.ones(observations.shape[1])
    rng_key, subkey = jax.random.split(rng_key)
    observations = observations + obs_error_std * jax.random.normal(
        subkey, observations.shape
    )

    # --- Estimator --------------------------------------------------------------
    # The `smoother:` node in conf/run_filter_smoothing.yaml already interpolates
    # the inner analysis / inner localization / inner inflation / temporal
    # localization group selections, so recursive instantiation builds those
    # objects bottom-up; only the run-time collaborators are injected here
    # (exactly as run_filtering.py injects them into `filtering.filter`).
    rng_key, smoother_key = jax.random.split(rng_key)
    smoother = instantiate(
        cfg.filter_smoothing.smoother,
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D_diag,
        rng_key=smoother_key,
    )

    save_history = bool(cfg.run.get("save_history", True))
    smoother_start = time.perf_counter()
    result = smoother.run(
        state=None,  # cold start; the first segment includes the model's spin-up
        params=prior_params,
        observations=observations,
        return_history=save_history,
    )
    smoother_seconds = time.perf_counter() - smoother_start

    # --- Outputs ----------------------------------------------------------------
    # The smoothed trajectory ensemble and the filtered end-of-window state (the
    # latter always from the final consistency pass), the per-iteration
    # trajectories when requested, and both diagnostics streams always.
    if result.params is not None:
        result.params.to_netcdf(out_dir / "posterior_params.nc")
    if result.state is not None:
        result.state.to_netcdf(out_dir / "posterior_state.nc")
    if result.params_history is not None:
        result.params_history.to_netcdf(out_dir / "params_iterations.nc")
    write_yaml(
        [dataclasses.asdict(d) for d in result.iteration_diagnostics],
        out_dir / "iteration_diagnostics.yaml",
    )

    # The final pass is a plain FilterResult, laid out exactly like
    # run_filtering.py's: its per-cycle diagnostics (and optional histories) are
    # what the filtering metric/figure stages read.
    final_pass = result.final_pass
    write_yaml(
        [dataclasses.asdict(d) for d in final_pass.diagnostics],
        out_dir / "cycle_diagnostics.yaml",
    )
    if final_pass.state_history is not None:
        final_pass.state_history.to_netcdf(out_dir / "state_history.nc")
    if final_pass.params_history is not None:
        final_pass.params_history.to_netcdf(out_dir / "params_history.nc")

    # Persist the truth-access parameters (slicing/offsets + per-cycle frame
    # count) so the downstream metric/figure scripts reconstruct the exact same
    # lazy view of the on-disk truth without re-deriving the horizon logic.
    # Byte-compatible with run_filtering.py's truth_access.yaml.
    write_yaml(
        {
            "true_state_path": str(true_state_path),
            "x_offset": float(x_offset),
            "start_idx": int(start_idx),
            "t_offset": float(t_offset),
            "n_total": int(n_total),
            "n_per_cycle": int(n_per_cycle),
            "num_cycles": int(num_cycles),
            "sim_time": float(sim_time),
            "truth_solver_name": str(cfg.truth_model.solver_name),
            "assim_solver_name": str(cfg.assim_model.solver_name),
        },
        out_dir / "truth_access.yaml",
    )

    write_yaml(
        {
            "configuration": {
                "estimator": type(smoother).__name__,
                "num_cycles": int(num_cycles),
                "num_steps": int(num_steps),
                # The configured alpha may be null (equal weights); record the
                # value the estimator resolved it to as well, so a benchmark
                # record never has to re-derive the tempering schedule.
                "alpha": (
                    float(cfg.filter_smoothing.alpha)
                    if cfg.filter_smoothing.alpha is not None
                    else None
                ),
                "alpha_effective": float(getattr(smoother, "alpha", num_steps)),
                "common_inner_noise": bool(cfg.filter_smoothing.common_inner_noise),
                "num_trajectory_knots": int(n_knots),
                "seconds_per_knot": float(seconds_per_knot),
                "ensemble_size": int(ensemble_size),
                "simulation_time_per_cycle": float(sim_time),
                "final_time": float(final_time),
                "observation_error_std": obs_error_std,
                "seed": int(cfg.filter_smoothing.seed),
                "truth_model": str(cfg.truth_model.name),
                "assimilation_model": str(cfg.assim_model.name),
                "truth_source": "disk" if cfg.run.truth_dir is not None else "inline",
                "truth_dir": (
                    str(cfg.run.truth_dir) if cfg.run.truth_dir is not None else None
                ),
                "num_truth_frames": int(n_total),
                # `estimator` above is the composed class, which stays
                # `FilterSmoothingESMDA` for every flavor (the schemes are
                # injected, not subclassed). The resolved subtrees below are
                # therefore the ONLY record of which update math ran. Never null
                # for the analysis: the group always sets a `_target_`.
                "inner_analysis": OmegaConf.to_container(
                    cfg.filter_smoothing.inner_analysis, resolve=True
                ),
                # The inner analysis and the inner localization are a coupled
                # pair (each analysis declares a localization_policy of optional
                # / forbidden / required), and the per-block LETKF diagnostics in
                # cycle_diagnostics.yaml are only interpretable against the
                # strategy and radius that produced them. `null` is a real value
                # (the global update).
                "inner_localization": (
                    OmegaConf.to_container(
                        cfg.filter_smoothing.inner_localization, resolve=True
                    )
                    if cfg.filter_smoothing.inner_localization is not None
                    else None
                ),
                "inner_inflation": (
                    OmegaConf.to_container(
                        cfg.filter_smoothing.inner_inflation, resolve=True
                    )
                    if cfg.filter_smoothing.inner_inflation is not None
                    else None
                ),
                # The outer update's localization axis — a different quantity
                # from `inner_localization` (time vs space), so recorded
                # separately; `null` is the global-in-time trajectory update.
                "temporal_localization": (
                    OmegaConf.to_container(
                        cfg.filter_smoothing.temporal_localization, resolve=True
                    )
                    if cfg.filter_smoothing.temporal_localization is not None
                    else None
                ),
            },
            "timing": {
                # num_steps inner passes plus the final consistency pass, each
                # of num_cycles forecast segments — that (num_steps + 1) factor
                # is the method's whole cost story, so break it out.
                "total_seconds": float(smoother_seconds),
                "num_inner_passes": int(num_steps + 1),
                "mean_pass_seconds": float(smoother_seconds / (num_steps + 1)),
                "mean_cycle_seconds": float(
                    smoother_seconds / max((num_steps + 1) * num_cycles, 1)
                ),
            },
        },
        out_dir / "run_info.yaml",
    )
    print(f"Saved outputs in {out_dir}")


@hydra.main(  # type: ignore[misc]
    version_base=None, config_path="../../conf", config_name="run_filter_smoothing"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
