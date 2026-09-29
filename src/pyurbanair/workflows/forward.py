"""Shared forward execution and complete, indexed numerical artifacts.

The CLI and local workers call the same runner. Complete persistence writes each
finished window before starting the next; rollout history remains in memory.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pyurbanair.config.hydra_helpers import clean_outputs, resolve_output_dir
from pyurbanair.config.run_record import (
    append_constructor_override,
    validate_run_config,
    write_run_record,
)
from pyurbanair.utils.run_utils import add_velocity_magnitude


class _Stepper(Protocol):
    def __call__(
        self, *, params: xarray.Dataset, state: xarray.Dataset | None = None
    ) -> xarray.Dataset: ...


def get_stepper(model: Any, is_ensemble: bool) -> _Stepper:
    if is_ensemble:

        def step(
            params: xarray.Dataset, state: xarray.Dataset | None = None
        ) -> xarray.Dataset:
            out = model.run_ensemble(params=params, state=state, sim_name="state")
            return cast(xarray.Dataset, out if out is not None else model.get_states())

        return step
    else:

        def step(
            params: xarray.Dataset, state: xarray.Dataset | None = None
        ) -> xarray.Dataset:
            out = model(params=params, state=state)
            return cast(xarray.Dataset, out if out is not None else model.get_states())

        return step


# The sampler always emits an `ensemble` dim. A single-member run must hand
# the forward model params WITHOUT it (scalar for static, (time,) for
# dynamic) -- the solver's inflow application can't handle a size-1 ensemble
# axis. Keep the ensemble dim in params_list so extrapolate() still sees it;
# drop it only at the model call.
def _member_params(p: xarray.Dataset, is_ensemble: bool) -> xarray.Dataset:
    if not is_ensemble and "ensemble" in p.dims:
        return p.isel(ensemble=0, drop=True)
    return p


# Stitch window-local clocks without changing their first sample's physical
# offset. Forecast-only output begins after t=0; forcing may begin at t=0.
def _concat_windows(
    window_list: Sequence[xarray.Dataset], cfg: DictConfig
) -> xarray.Dataset:
    if len(window_list) == 1:
        return window_list[0]
    rebased = [
        _window_dataset(ds, window, cfg) for window, ds in enumerate(window_list)
    ]
    combined = xarray.concat(rebased, dim="time", join="override")
    combined.attrs.pop("window", None)
    combined.attrs.pop("window_start", None)
    return combined


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _fingerprint(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_initial_state(
    source: Any,
    cfg: DictConfig,
    *,
    model: Any = None,
) -> xarray.Dataset | None:
    """Select a member by label and a history ending at a positional time index.

    By default all provided members and all frames through the final frame are
    retained. A single run requires a member selection for multi-member files.
    Ensembles require one supplied state per member; states are never broadcast.
    """
    if source is None:
        if (
            OmegaConf.select(cfg, "model.forward_model.spinup_source")
            == "training_data"
        ):
            raise ValueError(
                "training_data has no cold start: supply initial_state with a NetCDF path, or explicitly select forward_model/generative spinup_source."
            )
        return None
    selection = (
        dict(source)
        if isinstance(source, (Mapping, DictConfig))
        and not isinstance(source, xarray.Dataset)
        else {}
    )
    source = selection.get("path", source)
    if isinstance(source, xarray.Dataset):
        state = source.copy(deep=True)
    else:
        with xarray.open_dataset(pathlib.Path(source)) as dataset:
            state = dataset.load()
    if "member" in selection:
        if "ensemble" not in state.dims:
            raise ValueError("initial_state.member requires an ensemble coordinate")
        state = state.sel(ensemble=[selection["member"]])
    is_ensemble = bool(cfg.run.ensemble)
    required_members = int(cfg.ensemble.ensemble_size) if is_ensemble else 1
    if state.sizes.get("ensemble", 1) != required_members:
        raise ValueError(
            f"initial_state requires {required_members} member(s); select a member explicitly for a single run"
        )
    if not is_ensemble and "ensemble" in state.dims:
        state = state.isel(ensemble=0, drop=True)
    elif is_ensemble and "ensemble" not in state.dims:
        state = state.expand_dims(ensemble=[0])
    if "time" not in state.dims or state.sizes["time"] == 0:
        raise ValueError("initial_state must have a nonempty time dimension")
    end = int(selection.get("time_index", -1))
    if end < 0:
        end += state.sizes["time"]
    if end < 0 or end >= state.sizes["time"]:
        raise ValueError("initial_state.time_index is outside the available history")
    state = state.isel(time=slice(0, end + 1))
    variables = tuple(getattr(model, "state_vars", ("u", "v", "w")))
    missing = set(variables) - set(state.data_vars)
    if missing:
        raise ValueError(f"initial_state is missing variables: {sorted(missing)}")
    for axis in ("time", "x", "y", "z", "xt", "yt", "zt", "xm", "ym", "zm", "xu", "yv"):
        if axis in state.dims:
            if axis not in state.coords:
                raise ValueError(f"initial_state is missing coordinate {axis}")
            values = np.asarray(state[axis].values, dtype=float)
            if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
                raise ValueError(
                    f"initial_state coordinate {axis} must be finite and strictly increasing"
                )
    history = int(getattr(model, "num_history_steps", 1))
    if state.sizes["time"] < history:
        raise ValueError(f"initial_state needs at least {history} history frames")
    if model is not None and hasattr(model, "trained_output_frequency"):
        dt = float(model.trained_output_frequency)
        if history > 1 and not np.allclose(
            np.diff(state.time.values[-history:]), dt, rtol=0.05
        ):
            raise ValueError(
                f"initial_state history must have trained cadence {dt} seconds"
            )
        for axis in ("x", "y", "z"):
            size = OmegaConf.select(cfg, f"domain.n{axis}")
            if size is not None and state.sizes.get(axis) != int(size):
                raise ValueError(f"initial_state {axis} grid must contain {size} cells")
        bounds = OmegaConf.select(cfg, "domain.bounds")
        if bounds is not None:
            for axis, extent in zip("xyz", bounds):
                coord = np.asarray(state[axis].values, dtype=float)
                spacing = (float(extent[1]) - float(extent[0])) / len(coord)
                if not np.allclose(np.diff(coord), spacing, rtol=1e-4) or (
                    coord[0] < float(extent[0]) or coord[-1] > float(extent[1])
                ):
                    raise ValueError(
                        f"initial_state {axis} coordinates do not match the configured domain"
                    )
        for variable in variables:
            spatial_dims = set(state[variable].dims) - {"ensemble", "time"}
            if spatial_dims != {"x", "y", "z"}:
                raise ValueError(
                    f"initial_state {variable} must use the canonical x/y/z grid"
                )
    return state


def _window_dataset(ds: xarray.Dataset, window: int, cfg: DictConfig) -> xarray.Dataset:
    reference = ds.attrs.get("time_reference", "window")
    if reference not in {"window", "global"}:
        raise ValueError("time_reference must be 'window' or 'global'")
    if "time" in ds.dims and reference == "window":
        t = np.asarray(ds.time.values, dtype=float)
        ds = ds.assign_coords(time=t + window * float(cfg.time.simulation_time))
    return ds.assign_attrs(
        window=window,
        window_start=window * float(cfg.time.simulation_time),
        time_reference="global",
    )


def _model_state(state: xarray.Dataset | None) -> xarray.Dataset | None:
    """Artifact clock metadata must not leak into a new backend forecast.

    Surrogates copy template attributes but emit a fresh window-local clock.
    Keep the actual supplied history coordinates, stripping only our clock tag.
    """
    if state is None:
        return None
    result = state.copy(deep=False)
    result.attrs.pop("time_reference", None)
    return result


def _carry_history(
    previous: xarray.Dataset | None,
    forecast: xarray.Dataset,
    window: int,
    cfg: DictConfig,
    history_steps: int,
    cadence: float,
) -> xarray.Dataset:
    """Keep only the real history needed for the next surrogate window."""
    current = _window_dataset(forecast, window, cfg).isel(
        time=slice(-history_steps, None)
    )
    pieces = []
    if previous is not None:
        prior = previous.isel(time=slice(-history_steps, None))
        if window == 0:
            # An external initial artifact ends at the new run's physical t=0.
            prior = prior.assign_coords(time=prior.time.values - prior.time.values[-1])
        if "ensemble" in current.dims:
            prior = prior.assign_coords(ensemble=current.ensemble.values)
        pieces.append(prior)
    pieces.append(current)
    history = xarray.concat(pieces, dim="time", join="exact")
    # Boundary-including output can repeat the carried endpoint.
    _, reverse_ids = np.unique(history.time.values[::-1], return_index=True)
    history = history.isel(time=np.sort(history.sizes["time"] - 1 - reverse_ids))
    history = history.isel(time=slice(-history_steps, None))
    if history.sizes["time"] < history_steps:
        raise ValueError(
            f"The next window requires {history_steps} real history frames; "
            "provide an initial state or increase time.simulation_time."
        )
    if not np.allclose(np.diff(history.time.values), cadence, rtol=0.05):
        raise ValueError(
            "Rollout history does not match the trained cadence; use "
            "time.output_frequency at the trained cadence and a window duration "
            "that is an integer multiple of it."
        )
    # Own only H frames, instead of keeping a view onto an entire prior window.
    result = _model_state(history.copy(deep=True))
    assert result is not None
    return result


def _consolidated_parameters(
    parameters: list[xarray.Dataset],
    cfg: DictConfig,
    dynamic: bool,
) -> tuple[xarray.Dataset, str]:
    if dynamic:
        return _concat_windows(parameters, cfg), "time"
    if all(parameters[0].equals(candidate) for candidate in parameters[1:]):
        return parameters[0], "static"
    combined = xarray.concat(parameters, dim="window", join="exact")
    windows = np.arange(len(parameters))
    return (
        combined.assign_coords(
            window=windows,
            window_start=("window", windows * float(cfg.time.simulation_time)),
        ),
        "window",
    )


def _persist_window(
    root: pathlib.Path,
    index: dict[str, Any],
    window: int,
    state: xarray.Dataset,
    params: xarray.Dataset,
    cfg: DictConfig,
) -> None:
    directory = root / "windows" / f"{window:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    for kind, dataset in (("state", state), ("params", params)):
        dataset = _window_dataset(dataset, window, cfg)
        for member in range(dataset.sizes.get("ensemble", 1)):
            selected = (
                dataset.isel(ensemble=[member])
                if "ensemble" in dataset.dims
                else dataset
            )
            path = directory / f"{kind}_{member:04d}.nc"
            temporary = path.with_suffix(".nc.tmp")
            selected.to_netcdf(temporary)
            temporary.replace(path)
            index["artifacts"].append(
                {
                    "kind": kind,
                    "path": str(path.relative_to(root)),
                    "window": window,
                    "member": (
                        selected.ensemble.values[0].item()
                        if "ensemble" in selected.dims
                        else None
                    ),
                    "time_start": (
                        float(selected.time.values[0])
                        if "time" in selected.dims
                        else None
                    ),
                    "time_end": (
                        float(selected.time.values[-1])
                        if "time" in selected.dims
                        else None
                    ),
                    "sha256": _fingerprint(path),
                }
            )
    _write_json(root / "artifact_index.json", index)


def _output_directory(cfg: DictConfig) -> pathlib.Path:
    suffix = str(cfg.model.name)
    if cfg.run.ensemble:
        suffix += "_ensemble"
    if int(cfg.run.rollout_steps) > 1:
        suffix += "_rollout"
    if OmegaConf.select(cfg, "params.seconds_per_knot") is not None:
        suffix += "_time_varying"
    return resolve_output_dir(cfg, "forward_model") / suffix


def _snapshot_solver_inputs(
    root: pathlib.Path,
    index: dict[str, Any],
    window: int,
    model: Any,
    params: xarray.Dataset,
    is_ensemble: bool,
) -> None:
    """Preserve generated native settings from this job's concrete model dirs.

    Only known input filenames are visited; shared source trees and arbitrary
    directory contents are never traversed. Large inputs remain at their source
    with a visible warning, bounded to 1 MiB/file and 16 MiB/window.
    """
    owned_root = root.resolve()
    models = (
        getattr(model, "ensemble_forward_models", [model]) if is_ensemble else [model]
    )
    remaining = 16 * 1024**2
    visited: set[pathlib.Path] = set()
    for position, member_model in enumerate(models):
        label = (
            np.asarray(params.ensemble.values[position]).item() if is_ensemble else None
        )
        for role, component in (
            ("forward", member_model),
            ("spinup", getattr(member_model, "spinup_forward_model", None)),
        ):
            dirs = getattr(component, "dirs", None)
            if dirs is None:
                continue
            candidates = []
            if getattr(dirs, "infile_path", None) is not None:
                candidates.append(pathlib.Path(dirs.infile_path))
            name = getattr(dirs, "experiment_name", None)
            if name is not None and getattr(dirs, "experiment_dir", None) is not None:
                candidates.append(
                    pathlib.Path(dirs.experiment_dir) / f"namoptions.{name}"
                )
            if name is not None and getattr(dirs, "input_dir", None) is not None:
                candidates.append(pathlib.Path(dirs.input_dir) / f"{name}_p3d")
            for candidate in candidates:
                source = candidate.resolve()
                if source in visited or not source.is_file():
                    continue
                visited.add(source)
                warning = None
                if not source.is_relative_to(owned_root):
                    warning = "outside owned run root"
                elif source.stat().st_size > min(1024**2, remaining):
                    warning = "exceeds bounded solver-input snapshot budget"
                if warning is not None:
                    index.setdefault("solver_input_warnings", []).append(
                        {
                            "window": window,
                            "member": label,
                            "source_path": str(source),
                            "reason": warning,
                        }
                    )
                    continue
                with source.open("rb") as handle:
                    content = handle.read(min(1024**2, remaining) + 1)
                if len(content) > min(1024**2, remaining):
                    raise ValueError("Solver input grew beyond snapshot size limit")
                remaining -= len(content)
                destination = (
                    root
                    / "windows"
                    / f"{window:04d}"
                    / "solver_inputs"
                    / f"member_{position:04d}"
                    / role
                    / source.name
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(content)
                temporary.replace(destination)
                index["artifacts"].append(
                    {
                        "kind": "solver_input",
                        "path": str(destination.relative_to(root)),
                        "window": window,
                        "member": label,
                        "role": role,
                        "source_path": str(source),
                        "source_relative_path": str(source.relative_to(owned_root)),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "bytes": len(content),
                    }
                )
    _write_json(root / "artifact_index.json", index)


def run(
    cfg: DictConfig,
    *,
    complete_artifacts: bool = False,
    initial_state: Any = None,
    provenance: dict[str, Any] | None = None,
    output_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Execute a forward run with CLI-compatible defaults.

    Complete mode writes all sampled parameters and member/window states,
    independent of plotting. It retains rollout history in RAM and applies the
    optional ``run.max_retained_bytes`` bound (default 2 GiB in complete mode).
    """
    validate_run_config(cfg, "forward")
    if cfg.run.get("ensemble_save_on_disk", False):
        raise ValueError(
            "run.ensemble_save_on_disk is not implemented by the forward workflow; use complete_artifacts for persistence (rollout history remains in memory)"
        )
    if int(cfg.run.rollout_steps) < 0:
        raise ValueError(
            "run.rollout_steps must be nonnegative (total windows = 1 + rollout_steps)"
        )
    root = (
        pathlib.Path(output_dir) if output_dir is not None else _output_directory(cfg)
    )
    root.mkdir(parents=True, exist_ok=True)
    write_run_record(cfg, root, "forward", provenance=provenance)
    status: dict[str, Any] = {"status": "preparing", "started_at": time.time()}
    index: dict[str, Any] = {
        "version": 1,
        "status": "running",
        "backend": str(cfg.model.name),
        "case": OmegaConf.select(cfg, "case_name"),
        "artifacts": [],
        "total_windows": 1 + int(cfg.run.rollout_steps),
        "ensemble_size": int(cfg.ensemble.ensemble_size) if cfg.run.ensemble else 1,
        "failure_substitutions": [],
    }
    _write_json(root / "forward_status.json", status)
    if complete_artifacts:
        _write_json(root / "artifact_index.json", index)
    try:
        source = (
            initial_state if initial_state is not None else cfg.run.get("initial_state")
        )
        state_input = load_initial_state(source, cfg)
        if complete_artifacts and state_input is not None:
            state_input.to_netcdf(root / "initial_state.nc")
            status["initial_state_snapshot"] = {
                "path": "initial_state.nc",
                "sha256": _fingerprint(root / "initial_state.nc"),
            }
            _write_json(root / "forward_status.json", status)
        if source is not None:
            source_path = (
                source.get("path")
                if isinstance(source, (Mapping, DictConfig))
                and not isinstance(source, xarray.Dataset)
                else source
            )
            if not isinstance(source_path, xarray.Dataset):
                path = pathlib.Path(str(source_path)).resolve()
                status["initial_state"] = {
                    "path": str(path),
                    "sha256": _fingerprint(path),
                }
                _write_json(root / "forward_status.json", status)
        import jax

        rng_key = jax.random.PRNGKey(int(cfg.params.get("seed", 0)))
        is_ensemble = bool(cfg.run.ensemble)
        sampler = instantiate(cfg.params)
        params = sampler.sample(index["ensemble_size"])
        dynamic = "time" in params.coords
        append_constructor_override(
            root,
            role="forward",
            component="parameter_sampler",
            values={
                "sampled_shape": dict(params.sizes),
                "ensemble_size": index["ensemble_size"],
                "windows": index["total_windows"],
            },
        )
        if complete_artifacts:
            params.to_netcdf(root / "sampled_params.nc")
        constructor_kwargs: dict[str, Any] = {}
        if (
            state_input is not None
            and OmegaConf.select(cfg, "model.forward_model.spinup_source")
            == "training_data"
        ):
            # Supplied warm states need no CFD backend, even when the config
            # retains a template for other initialization modes.
            constructor_kwargs["spinup_forward_model"] = None
            append_constructor_override(
                root,
                role="forward",
                component="forward_model",
                values={"spinup_forward_model": None},
            )
        model = instantiate(
            cfg.model.forward_model,
            results_dir=(
                pathlib.Path(cfg.run.results_dir)
                if cfg.run.results_dir is not None
                else None
            ),
            **constructor_kwargs,
        )
        if state_input is not None:
            state_input = load_initial_state(state_input, cfg, model=model)
        history_steps = int(getattr(model, "num_history_steps", 1))
        history_cadence = float(getattr(model, "trained_output_frequency", 1.0))
        if history_steps > 1 and state_input is not None:
            state_input = state_input.isel(time=slice(-history_steps, None)).copy(
                deep=True
            )
        state_input = _model_state(state_input)
        instantiate(cfg.model.prepare, forward_model=model)
        clean_outputs(model_name=cfg.model.name, forward_model=model)
        if is_ensemble:
            ensemble_paths = {
                key: pathlib.Path(str(cfg.model.ensemble_model[key]))
                for key in ("temp_dir", "results_dir")
                if cfg.model.ensemble_model.get(key) is not None
            }
            model = instantiate(
                cfg.model.ensemble_model, forward_model=model, **ensemble_paths
            )
        stepper = get_stepper(model, is_ensemble)
        states: list[xarray.Dataset] = []
        parameters: list[xarray.Dataset] = []
        status["status"] = "running"
        _write_json(root / "forward_status.json", status)
        retained_bytes = 0
        limit = int(
            cfg.run.get("max_retained_bytes", 2 * 1024**3 if complete_artifacts else 0)
        )
        if limit and state_input is not None and state_input.nbytes > limit:
            raise MemoryError("Initial state exceeds run.max_retained_bytes")
        for window in range(index["total_windows"]):
            if window and dynamic:
                rng_key, subkey = jax.random.split(rng_key)
                params = sampler.extrapolate(
                    params, np.asarray(sampler.time_coords), subkey
                )
            out = stepper(params=_member_params(params, is_ensemble), state=state_input)
            if is_ensemble:
                if out.sizes.get("ensemble") != params.sizes["ensemble"]:
                    raise ValueError(
                        "Forward output member count does not match sampled parameters"
                    )
                # Disk backends may retain each single-file ensemble=[0]. The
                # wrapper returns members in sampler order, so use its labels.
                out = out.assign_coords(ensemble=params.ensemble.values)
            substitutions = getattr(model, "_last_failure_substitutions", {})
            actual_params = params.copy(deep=True)
            if substitutions:
                # Persist donor parameters without jitter: these generated the saved field.
                for failed, donor in substitutions.items():
                    for variable in actual_params.data_vars:
                        if "ensemble" in actual_params[variable].dims:
                            actual_params[variable][{"ensemble": failed}] = params[
                                variable
                            ].isel(ensemble=donor)
                index["failure_substitutions"].append(
                    {"window": window, "donors": substitutions}
                )
            if complete_artifacts:
                _persist_window(root, index, window, out, actual_params, cfg)
                _snapshot_solver_inputs(
                    root, index, window, model, actual_params, is_ensemble
                )
            states.append(out)
            parameters.append(actual_params)
            retained_bytes += int(out.nbytes + actual_params.nbytes)
            next_state = _model_state(out)
            extra_history_bytes = 0
            if history_steps > 1 and window + 1 < index["total_windows"]:
                next_state = _carry_history(
                    state_input, out, window, cfg, history_steps, history_cadence
                )
                extra_history_bytes = int(next_state.nbytes)
            if limit and retained_bytes + extra_history_bytes > limit:
                raise MemoryError(
                    f"Forward history exceeded run.max_retained_bytes={limit}; completed window artifacts are preserved"
                )
            state_input = next_state
        state = _concat_windows(states, cfg)
        params, params_layout = _consolidated_parameters(parameters, cfg, dynamic)
        index["consolidated_params_layout"] = params_layout
        if complete_artifacts:
            # Window boundaries can share a sample; keep the newest at that instant.
            for filename, dataset in (("state.nc", state), ("params.nc", params)):
                if "time" in dataset.dims:
                    _, reverse_ids = np.unique(
                        dataset.time.values[::-1], return_index=True
                    )
                    dataset = dataset.isel(
                        time=np.sort(dataset.sizes["time"] - 1 - reverse_ids)
                    )
                dataset.to_netcdf(root / filename)
            index["status"] = "complete"
            _write_json(root / "artifact_index.json", index)
        elif dynamic:
            _member_params(state, False).to_netcdf(root / "state.nc")
            _member_params(params, False).to_netcdf(root / "params.nc")
        status.update(
            status="succeeded", finished_at=time.time(), dimensions=dict(state.sizes)
        )
        _write_json(root / "forward_status.json", status)
    except BaseException as exc:
        status.update(
            status="failed",
            finished_at=time.time(),
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_json(root / "forward_status.json", status)
        if complete_artifacts:
            index["status"] = "partial"
            _write_json(root / "artifact_index.json", index)
        raise
    print(
        f'Model: {cfg.model.name}; windows: {index["total_windows"]}; dims: {dict(state.sizes)}'
    )
    if not cfg.run.skip_viz:
        try:
            from scripts._common import (
                plot_derived_inflow_angle,
                plot_derived_velocity_magnitude,
                visualize_forward_state,
            )

            viz_state = _member_params(state, False) if dynamic else state
            viz_params = _member_params(params, False)
            if "ensemble" in viz_state.dims:
                viz_state = viz_state.mean("ensemble")
            visualize_forward_state(
                add_velocity_magnitude(viz_state),
                cfg.model.name,
                root,
                str(cfg.model.name),
            )
            if dynamic:
                plot_derived_inflow_angle(viz_state, viz_params, root)
                plot_derived_velocity_magnitude(viz_state, viz_params, root)
        except Exception as exc:
            status["visualization_error"] = f"{type(exc).__name__}: {exc}"
            _write_json(root / "forward_status.json", status)
            if not complete_artifacts:
                raise
    return root
