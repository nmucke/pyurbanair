"""Side-effect-free configuration checks and run records.

This module deliberately imports neither a solver nor an assimilation library.
It is also used by the lightweight preview before any runner is imported.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def _get(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def validate_run_config(cfg: DictConfig, workflow: str) -> None:
    """Reject incompatible choices before creating a run directory or solver."""
    for role in ("model", "truth_model", "assim_model"):
        if _get(cfg, f"{role}.forward_model.model_discrepancy.enabled", False):
            if _get(cfg, f"{role}.name") != "pyudales":
                raise ValueError(
                    "model_discrepancy currently requires pyudales/Vreman."
                )
            if role == "assim_model":
                raise ValueError(
                    "Discrepancy assimilation is not implemented yet; coefficient "
                    "prior/inference integration and recovery validation are still required."
                )
    if workflow == "render":
        if OmegaConf.is_missing(cfg, "input"):
            raise ValueError("input is required for render_les")
        if not pathlib.Path(str(cfg.input)).exists():
            raise ValueError(f"input does not exist: {cfg.input}")
    expected = _get(cfg, "experiment.workflow")
    if expected is not None and expected != workflow:
        raise ValueError(
            f"experiment.workflow={expected!r} cannot be used with {workflow!r}."
        )

    if workflow in {"esmda", "filter_smoothing"}:
        target = str(_get(cfg, "esmda.smoother._target_", ""))
        dynamic_smoother = "TimeVaryingParameter" in target
        dynamic_prior = _get(cfg, "prior_params.seconds_per_knot") is not None
        if dynamic_smoother != dynamic_prior:
            raise ValueError(
                "esmda.smoother and prior_params must both select dynamic "
                "parameters, or both select static parameters."
            )
        if workflow == "esmda":
            dynamic_truth = _get(cfg, "truth_params.seconds_per_knot") is not None
            if dynamic_truth != dynamic_prior:
                raise ValueError(
                    "truth_params and prior_params must both be dynamic or both "
                    "be static for ESMDA window extrapolation; use filtering "
                    "for a dynamic truth with a static prior."
                )
        if (
            _get(cfg, "esmda.localization") is not None
            and _get(cfg, "esmda.state_reduction") is not None
        ):
            raise ValueError(
                "esmda.localization and esmda.state_reduction cannot both be set."
            )
    if workflow in {"filtering", "filter_smoothing"}:
        analysis = str(_get(cfg, "filtering.analysis._target_", ""))
        localized = _get(cfg, "filtering.localization") is not None
        if "LETKFAnalysis" in analysis and not localized:
            raise ValueError(
                "filtering.analysis=LETKF requires filtering.localization."
            )
        if "ETKFAnalysis" in analysis and "LETKFAnalysis" not in analysis and localized:
            raise ValueError("filtering.analysis=ETKF forbids filtering.localization.")
        if localized and _get(cfg, "filtering.state_reduction") is not None:
            raise ValueError(
                "filtering.localization and filtering.state_reduction cannot both be set."
            )
        if workflow == "filter_smoothing":
            if _get(cfg, "filtering.mode") not in {"state", "joint"}:
                raise ValueError(
                    "filtering.mode must be state or joint for filter_smoothing."
                )
            if (
                "State"
                in str(_get(cfg, "esmda.smoother._target_", "")).rsplit(".", 1)[-1]
            ):
                raise ValueError(
                    "esmda.smoother must estimate parameters only for filter_smoothing."
                )
        if (
            _get(cfg, "prior_params.seconds_per_knot") is not None
            and workflow == "filtering"
        ):
            raise ValueError("prior_params must be static for filtering.")
        stride = int(_get(cfg, "filtering.assimilate_every_n_step", 1))
        period = float(_get(cfg, "time.output_frequency"))
        horizon = float(_get(cfg, "time.simulation_time"))
        if period <= 0 or stride < 1 or round(horizon / period) % stride:
            raise ValueError(
                "filtering.assimilate_every_n_step must divide the number of "
                "time.output_frequency frames in time.simulation_time."
            )
        if workflow == "filter_smoothing":
            cycle_seconds = stride * period
            cycles = round(horizon / cycle_seconds)
            if cycles < 1 or abs(horizon - cycles * cycle_seconds) > 1e-9 * max(
                horizon, cycle_seconds
            ):
                raise ValueError(
                    "time.simulation_time must be tiled exactly by "
                    "time.output_frequency times filtering.assimilate_every_n_step."
                )
    truth_dir = _get(cfg, "run.truth_dir")
    if truth_dir is not None:
        source = pathlib.Path(str(truth_dir))
        for filename in ("state.nc", "params.nc"):
            if not (source / filename).is_file():
                raise ValueError(f"run.truth_dir is missing {filename}: {source}")
    if workflow == "surrogate_data":
        mode = _get(cfg, "training_data.geometry.mode")
        legacy_source = _get(cfg, "training_data.geometry.source")
        if legacy_source in {"barcelona", "xie_and_castro"}:
            raise ValueError(
                f"training_data.geometry.source={legacy_source!r} is a legacy "
                "fixed-case selector; use case=... instead."
            )
        if mode != "fixed":
            raise ValueError(
                "training_data.geometry.mode must be fixed for "
                "generate_training_data.py; select "
                "training_data/geometry_mode=fixed and case=..."
            )
        output_name = _get(cfg, "training_data.geometry.output_name")
        case_name = _get(cfg, "case_name")
        if mode == "fixed" and output_name != case_name:
            raise ValueError(
                "training_data.geometry.mode=fixed requires "
                "training_data.geometry.output_name to follow case_name; "
                "select training_data/geometry_mode=fixed with case=..."
            )
    if workflow == "surrogate_random_data":
        if _get(cfg, "training_data.geometry.mode") != "random":
            raise ValueError(
                "training_data.geometry.mode must be random for "
                "generate_random_geometries_training_data.py."
            )
    if workflow == "probe":
        run_dir = pathlib.Path(
            str(_get(cfg, "probes.run_dir") or _get(cfg, "paths.results_dir"))
        )
        if not (run_dir / "truth_access.yaml").is_file():
            raise ValueError(f"probes.run_dir is missing truth_access.yaml: {run_dir}")


def _hydra_choices() -> tuple[dict[str, Any], list[str]]:
    if not HydraConfig.initialized():
        return {}, []
    hydra_cfg = HydraConfig.get()
    choices = OmegaConf.to_container(hydra_cfg.runtime.choices, resolve=True)
    overrides = list(hydra_cfg.overrides.task)
    if not isinstance(choices, dict):
        raise TypeError("Hydra runtime choices must be a mapping")
    return {str(key): value for key, value in choices.items()}, overrides


def write_run_record(
    cfg: DictConfig,
    output_dir: pathlib.Path,
    workflow: str,
    *,
    constructor_overrides: list[dict[str, Any]] | None = None,
    save_legacy_config: bool = True,
    artifact_dir: pathlib.Path | None = None,
    record_choices_as_requested: bool = False,
) -> None:
    """Write launch intent and concrete constructor arguments when available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_legacy_config:
        OmegaConf.save(cfg, output_dir / "config.yaml")
    OmegaConf.save(cfg, output_dir / "config.resolved.yaml", resolve=True)
    choices, overrides = _hydra_choices()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    manifest = {
        "workflow": workflow,
        "experiment": _get(cfg, "experiment.name", _get(cfg, "run.name")),
        "choices": {} if record_choices_as_requested else choices,
        "requested_choices": choices if record_choices_as_requested else {},
        "choice_provenance": (
            "requested" if record_choices_as_requested else "effective"
        ),
        "cli_overrides": overrides,
        "params_to_estimate": _get(cfg, "params_to_estimate"),
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "paths": {
            "output_dir": str((artifact_dir or output_dir).resolve()),
            "record_dir": str(output_dir.resolve()),
            "experiment_dir": _get(cfg, "paths.experiment_dir"),
            "output_root": _get(cfg, "paths.base_results_dir"),
            "truth_dir": _get(cfg, "run.truth_dir"),
        },
        "constructor_overrides": constructor_overrides or [],
    }
    OmegaConf.save(OmegaConf.create(manifest), output_dir / "run_manifest.yaml")


def append_constructor_override(
    output_dir: pathlib.Path,
    *,
    role: str,
    component: str,
    values: dict[str, Any],
    window: int | None = None,
) -> None:
    """Append data-dependent scalar constructor inputs without serializing objects."""
    path = output_dir / "run_manifest.yaml"
    manifest = OmegaConf.load(path)
    entry: dict[str, Any] = {"role": role, "component": component, "values": values}
    if window is not None:
        entry["window"] = window
    manifest.constructor_overrides.append(entry)
    OmegaConf.save(manifest, path)
