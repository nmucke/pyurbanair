"""Config consistency checks, called first thing in every run_<workflow>.py.

    check_config(cfg, "smoother")   # forward | smoother | filtering | hybrid

Reads only the composed config (nothing is sampled or simulated), so a bad
combination fails in seconds instead of after the truth run. Every problem
found is reported at once, in a single ValueError.
"""

from __future__ import annotations

import pathlib
from typing import Any

from omegaconf import DictConfig, OmegaConf

from pyurbanair.config.discrepancy import (
    SGS_BIAS_PARAMETER_NAMES,
    validate_sgs_discrepancy_settings,
)

WORKFLOWS = ("forward", "smoother", "filtering", "hybrid")


def check_config(cfg: DictConfig, workflow: str) -> None:
    if workflow not in WORKFLOWS:
        raise ValueError(f"workflow must be one of {WORKFLOWS}, got {workflow!r}")
    problems = _models(cfg)
    if workflow != "forward":
        problems += _assimilation(cfg)
        problems += _discrepancy(cfg, workflow)
    if workflow in ("smoother", "hybrid"):
        problems += _smoothing(cfg, workflow)
    if workflow in ("filtering", "hybrid"):
        problems += _filtering(cfg, workflow)
    if problems:
        raise ValueError("Inconsistent config:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# Checks, one group per config block. Each returns a list of problems.
# ---------------------------------------------------------------------------


def _models(cfg: DictConfig) -> list[str]:
    problems = []
    for role in ("model", "truth_model", "assim_model"):
        model = cfg.get(role)
        if model is None:
            continue
        fm = model.forward_model
        if model.name == "pyudales" and fm.get("nx", 1) % fm.get("ncpu", 1):
            problems.append(
                f"{role}: forward_model.ncpu={fm.ncpu} must divide nx={fm.nx} "
                "(uDALES splits x across its MPI ranks)."
            )
        if (
            model.name == "pypalm"
            and fm.get("boundary_condition") == "periodic"
            and (fm.get("nx", 0) % 2 or fm.get("ny", 0) % 2)
        ):
            problems.append(
                f"{role}: periodic PALM needs an even nx and ny (its FFT "
                f"pressure solver), got {fm.nx} x {fm.ny}."
            )
        discrepancy = OmegaConf.select(cfg, f"{role}.forward_model.model_discrepancy")
        if discrepancy is None or not discrepancy.get("enabled", False):
            continue
        model = cfg[role]
        if model.name != "pyudales" or model.forward_model.get("closure") != "vreman":
            problems.append(f"{role}: model_discrepancy needs pyudales with vreman.")
        try:
            validate_sgs_discrepancy_settings(discrepancy)
        except ValueError as error:
            problems.append(f"{role}: {error}")
    return problems


def _assimilation(cfg: DictConfig) -> list[str]:
    problems = []
    da = cfg.assimilation
    if da.num_windows < 1:
        problems.append("assimilation.num_windows must be at least 1.")
    unknown = set(da.params_to_estimate or []) - _prior_names(cfg)
    if unknown:
        problems.append(
            f"assimilation.params_to_estimate names {sorted(unknown)}, which "
            "prior_params does not define."
        )
    if da.truth_dir is not None:
        for name in ("state.nc", "params.nc"):
            if not (pathlib.Path(da.truth_dir) / name).is_file():
                problems.append(f"assimilation.truth_dir has no {name}.")
    return problems


def _smoothing(cfg: DictConfig, workflow: str) -> list[str]:
    problems = []
    s = cfg.smoothing
    smoother = _class(s.smoother)
    state_bearing = "State" in smoother
    if ("TimeVarying" in smoother) != _dynamic(cfg.prior_params):
        problems.append(
            f"smoothing.smoother is {smoother} but prior_params is "
            f"{'time-varying' if _dynamic(cfg.prior_params) else 'static'}: pair "
            "a time-varying prior with ${smoother.dynamic} or "
            "${smoother.state_and_dynamic}, a static one with the others."
        )
    if workflow == "smoother" and _dynamic(cfg.truth_params) != _dynamic(
        cfg.prior_params
    ):
        problems.append(
            "truth_params and prior_params must both be time-varying or both "
            "static for the smoother (use filtering for a time-varying truth "
            "with a static prior)."
        )
    if workflow == "hybrid" and state_bearing:
        problems.append(
            "The hybrid's smoother must estimate parameters only: use "
            "${smoother.static} or ${smoother.dynamic}."
        )
    if s.localization is not None and s.state_reduction is not None:
        problems.append(
            "smoothing.localization and state_reduction cannot both be set."
        )
    if "Distance" in _class(s.localization) and not state_bearing:
        problems.append(
            "Distance localization needs a state-bearing smoother "
            "(state, state_and_parameter or state_and_dynamic)."
        )
    if s.final_time_smoothing and (s.state_reduction is None or not state_bearing):
        problems.append(
            "smoothing.final_time_smoothing needs a state-bearing smoother and "
            "a state_reduction."
        )
    return problems


def _filtering(cfg: DictConfig, workflow: str) -> list[str]:
    problems = []
    f = cfg.filtering
    analysis = _class(f.analysis)
    localized = f.localization is not None
    if workflow == "filtering" and _dynamic(cfg.prior_params):
        problems.append("Filtering needs a static prior_params.")
    modes = (
        ("state", "joint") if workflow == "hybrid" else ("state", "parameter", "joint")
    )
    if f.mode not in modes:
        problems.append(f"filtering.mode must be one of {modes} for the {workflow}.")
    if f.mode == "state" and f.parameter_evolution is not None:
        problems.append(
            "filtering.mode=state keeps the parameters fixed: set "
            "filtering.parameter_evolution=null."
        )
    if analysis.startswith("LETKF") and not localized:
        problems.append("The LETKF analysis needs a filtering.localization.")
    if analysis.startswith("ETKF") and localized:
        problems.append("The ETKF analysis forbids a filtering.localization.")
    if localized and f.state_reduction is not None:
        problems.append(
            "filtering.localization and state_reduction cannot both be set."
        )
    estimates_params = cfg.assimilation.params_to_estimate != []
    if (
        f.mode != "state"
        and estimates_params
        and f.inflation is None
        and f.parameter_evolution is None
    ):
        problems.append(
            f"filtering.mode={f.mode} updates parameters and needs an inflation "
            "or a parameter_evolution to keep their spread."
        )
    stride = int(cfg.assimilation.assimilate_every_n_step)
    frames = round(cfg.time.simulation_time / cfg.time.output_frequency)
    if stride < 1 or frames % stride:
        problems.append(
            f"assimilation.assimilate_every_n_step={stride} must divide the "
            f"{frames} output frames per window."
        )
    if workflow == "hybrid":
        allocation = cfg.hybrid.likelihood_allocation
        if allocation not in ("filter_only", "shared_budget"):
            problems.append(
                "hybrid.likelihood_allocation must be filter_only or shared_budget."
            )
        if allocation == "shared_budget" and (
            f.beta <= 1
            or stride != 1
            or (cfg.observation.aggregation or {}).get("interval_seconds") is not None
        ):
            problems.append(
                "hybrid.likelihood_allocation=shared_budget needs filtering.beta > 1, "
                "assimilate_every_n_step=1 and no observation aggregation."
            )
    return problems


def _discrepancy(cfg: DictConfig, workflow: str) -> list[str]:
    """Rules for estimating the SGS-discrepancy coefficients."""
    enabled = OmegaConf.select(
        cfg, "assim_model.forward_model.model_discrepancy.enabled", default=False
    )
    if not enabled:
        return []
    problems = []
    selected = cfg.assimilation.params_to_estimate
    estimates = selected is None or bool(set(selected) & set(SGS_BIAS_PARAMETER_NAMES))
    time_varying = set(cfg.prior_params.get("external_parameters") or {})
    if time_varying & set(SGS_BIAS_PARAMETER_NAMES):
        problems.append("The SGS-discrepancy coefficients must be static parameters.")
    if workflow in ("smoother", "hybrid"):
        expected = (
            "TimeVaryingParameterESMDA"
            if _dynamic(cfg.prior_params)
            else "ParameterESMDA"
        )
        if _class(cfg.smoothing.smoother) != expected:
            problems.append(
                f"With model_discrepancy the smoother must be {expected} "
                "(parameter-only, matching the prior)."
            )
    if workflow == "hybrid":
        if cfg.filtering.mode != "state":
            problems.append(
                "With model_discrepancy the hybrid needs filtering.mode=state, so "
                "the smoother's coefficients stay fixed in the filter."
            )
        if cfg.ensemble.failure.policy != "raise":
            problems.append(
                "With model_discrepancy the hybrid needs ensemble.failure.policy=raise."
            )
    if workflow == "filtering" and estimates and cfg.filtering.mode != "state":
        if cfg.filtering.parameter_evolution is not None:
            problems.append(
                "Estimating the SGS-discrepancy coefficients in a filter needs "
                "filtering.parameter_evolution=null."
            )
        if cfg.filtering.mode == "parameter" and "Distance" in _class(
            cfg.filtering.localization
        ):
            problems.append(
                "Parameter-only filtering of the SGS-discrepancy coefficients "
                "cannot use distance localization."
            )
    return problems


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------


def _class(node: Any) -> str:
    """Class name of an instantiable config node, '' for null."""
    return "" if node is None else str(node._target_).rsplit(".", 1)[-1]


def _dynamic(sampler: DictConfig) -> bool:
    """Whether a parameter sampler config is time-varying."""
    return "seconds_per_knot" in sampler


def _prior_names(cfg: DictConfig) -> set[str]:
    names: set[str] = set()
    for block in ("parameters", "external_parameters", "static_parameters"):
        names |= set(cfg.prior_params.get(block) or {})
    return names
