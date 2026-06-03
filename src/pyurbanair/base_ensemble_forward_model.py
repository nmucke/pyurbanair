import logging
import multiprocessing as mp
import os
import pathlib
import subprocess
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Literal, Optional

import numpy as np
import xarray
from tqdm import tqdm

from pyurbanair.base_forward_model import BaseForwardModel
from pyurbanair.utils.cpu_pinning import (
    build_cpu_queue,
    cpu_pinning_disabled,
    pin_worker_initializer,
)

logger = logging.getLogger(__name__)

FailurePolicy = Literal["raise", "resample_from_successes"]


def create_dir(
    dir_path: pathlib.Path,
) -> pathlib.Path:
    """Create a temporary directory in the given directory."""
    os.makedirs(pathlib.Path(dir_path), exist_ok=True)
    return pathlib.Path(dir_path)


class BaseEnsembleForwardModel:
    """
    Base class for ensemble forward models.

    Manages running multiple forward model instances as an ensemble,
    either sequentially or in parallel, with results saved in memory
    or on disk.

    Subclasses must:
    - Populate self.ensemble_forward_models in __init__
    - Implement _run_parallel for parallel execution
    - Optionally override sequential methods for custom behavior
    """

    def __init__(
        self,
        forward_model: BaseForwardModel,
        *args: Any,
        ensemble_size: int = 10,
        results_dir: Optional[pathlib.Path] = None,
        num_parallel_processes: int = 1,
        num_cpus_per_process: int = 1,
        temp_dir: Optional[pathlib.Path] = None,
        failure: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ensemble forward model.

        Args:
            forward_model: The forward model template.
            ensemble_size: Number of ensemble members.
            results_dir: Directory for saving results. If None, uses
                forward_model.results_dir.
            num_parallel_processes: Number of parallel processes.
            num_cpus_per_process: Number of CPUs per process.
            temp_dir: Temporary directory for ensemble experiments.
            failure: Failure-handling policy mapping
                ``{"policy", "jitter_scale", "seed"}``. Defaults to the
                historical ``"raise"`` behavior; can be reconfigured later
                via :meth:`configure_failure_policy`.
        """
        self.forward_model = forward_model
        self.ensemble_size = ensemble_size
        self.num_parallel_processes = num_parallel_processes
        self.num_cpus_per_process = num_cpus_per_process
        self.parallel_execution = num_parallel_processes > 1

        # Determine results directory: explicit > forward model's
        results_dir = (
            results_dir if results_dir is not None else forward_model.results_dir
        )
        self._set_save_mode(results_dir)

        if hasattr(forward_model, "rollout_step"):
            self.rollout = True
            self.rollout_step = 0
        else:
            self.rollout = False

        # Create ensemble experiment base directory
        if hasattr(forward_model, "dirs"):
            ensemble_temp_dir = forward_model.dirs.temp_dir
        else:
            ensemble_temp_dir = (
                temp_dir if temp_dir is not None else pathlib.Path(".temp")
            )
        self.ensemble_experiment_base_dir = create_dir(
            ensemble_temp_dir / "ensemble_experiments"
        )

        # Failure-handling policy, set at instantiation. Defaults preserve
        # historical behavior (any per-member exception aborts the ensemble
        # run); pass ``failure`` to opt into resample-from-successes, or call
        # ``configure_failure_policy`` later to reconfigure.
        self._last_failure_substitutions: dict[int, int] = {}
        failure = dict(failure) if failure else {}
        self.configure_failure_policy(
            policy=failure.get("policy", "raise"),
            jitter_scale=failure.get("jitter_scale", 0.05),
            seed=failure.get("seed", 0),
        )

        # Subclasses must populate this list with individual forward models
        self.ensemble_forward_models: list[BaseForwardModel] = []
        for ensemble_number in range(self.ensemble_size):
            self.ensemble_forward_models.append(
                self._create_new_forward_model(
                    forward_model=forward_model,
                    experiment_base_dir=self.ensemble_experiment_base_dir,
                    experiment_name=f"{ensemble_number:03d}",
                )
            )

    def configure_failure_policy(
        self,
        policy: FailurePolicy = "raise",
        jitter_scale: float = 0.05,
        seed: int = 0,
    ) -> None:
        """Configure how per-member forward-run failures are handled.

        - ``"raise"``: any ``CalledProcessError`` aborts the ensemble.
        - ``"resample_from_successes"``: failed members' states are cloned
          from a randomly chosen successful member; the failed slots in the
          parameter ensemble are also replaced (with jitter) when the caller
          invokes :meth:`apply_failure_substitutions_to_params`.
        """
        self._failure_policy = policy
        self._failure_jitter_scale = float(jitter_scale)
        self._failure_rng = np.random.default_rng(seed)
        self._last_failure_substitutions = {}

    @abstractmethod
    def _create_new_forward_model(
        self,
        forward_model: BaseForwardModel,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> BaseForwardModel:
        """Create a new forward model for the ensemble."""
        raise NotImplementedError

    def _set_save_mode(self, results_dir: Optional[pathlib.Path]) -> None:
        """Set save mode based on results_dir."""
        self.results_dir = results_dir
        self.save_on_disk = results_dir is not None
        self.save_in_memory = results_dir is None
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)

    def set_results_dir(self, results_dir: Optional[pathlib.Path]) -> None:
        """Change the results directory for the ensemble.

        This updates save mode flags and creates the directory if needed.
        Individual ensemble member forward models are updated when
        running sequentially on disk.
        """
        self._set_save_mode(results_dir)

    def get_member_state(
        self,
        state: Optional[xarray.Dataset | pathlib.Path],
        member_index: int,
        sim_name: str = "state",
    ) -> Optional[xarray.Dataset]:
        """
        Get state for a single ensemble member.

        Args:
            state: The state for the ensemble. Can be an xarray.Dataset with
                an ensemble dimension, a pathlib.Path to a directory of
                per-member files, or None.
            member_index: The index of the ensemble member.
            sim_name: The base simulation name. Each member will be saved
                as "{sim_name}_{i}.nc".

        Returns:
            The state for the ensemble member.
        """
        if isinstance(state, xarray.Dataset):
            return state.isel(ensemble=member_index)

        if isinstance(state, pathlib.Path):
            with xarray.open_dataset(
                state / f"{sim_name}_{member_index}.nc", engine="netcdf4"
            ) as dataset:
                return dataset.load()

        if self.save_on_disk:
            file_name = self.results_dir / f"{sim_name}_{member_index}.nc"  # type: ignore[operator]
            if file_name.exists():
                with xarray.open_dataset(file_name, engine="netcdf4") as dataset:
                    return dataset.load()

        return None

    def get_member_params(
        self,
        params: Optional[xarray.Dataset],
        member_index: int,
    ) -> Optional[xarray.Dataset]:
        """Get params for a single ensemble member."""
        if params is None:
            return None
        if "ensemble" in params.dims:
            return params.isel(ensemble=member_index)
        return params

    @abstractmethod
    def _pre_run_ensemble(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """
        Prepare the ensemble for a run.

        This can include applying inflow settings, handling warm starts, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def _post_run_ensemble(self, sim_name: str) -> xarray.Dataset | None:
        """Clean up after a run and collect results.

        This can include moving results to disk, cleaning output directories, etc.
        """
        raise NotImplementedError

    def _run_ensemble_sequentially_in_memory(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:
        """Run the ensemble sequentially, returning results in memory.

        Override in subclasses for custom behavior.
        """
        states: list[Optional[xarray.Dataset]] = []
        failed: list[int] = []
        self._last_failure_substitutions = {}
        pbar = tqdm(
            enumerate(self.ensemble_forward_models),
            total=self.ensemble_size,
            desc="Running ensemble",
        )
        for i, model in pbar:
            try:
                result = model(
                    state=self.get_member_state(state, i, sim_name),  # type: ignore[arg-type]
                    params=self.get_member_params(params, i),
                    sim_name=f"{sim_name}_{i}",
                )
            except subprocess.CalledProcessError as exc:
                if self._failure_policy == "raise":
                    raise
                logger.warning(
                    "Ensemble member %d failed (%s); will be resampled "
                    "from a successful member.",
                    i,
                    exc,
                )
                failed.append(i)
                states.append(None)
                continue
            states.append(result)

        resolved = self._resolve_failures(states, failed)
        return xarray.concat(resolved, dim="ensemble", join="override")

    def _resolve_failures(
        self,
        states: list[Optional[xarray.Dataset]],
        failed: list[int],
    ) -> list[xarray.Dataset]:
        """Replace ``None`` entries in ``states`` with clones of successful members.

        Records substitutions in ``self._last_failure_substitutions`` so that
        downstream callers (e.g. ESMDA) can apply matching substitutions to
        the parameter ensemble.
        """
        if not failed:
            # Cast away Optional once we know there are no Nones.
            return [s for s in states if s is not None]

        survivors = [i for i, s in enumerate(states) if s is not None]
        if not survivors:
            raise RuntimeError(
                "All ensemble members failed; cannot resample. "
                "Inspect per-member logs and forward-model inputs."
            )

        donors = self._failure_rng.choice(survivors, size=len(failed))
        substitutions = {int(j): int(d) for j, d in zip(failed, donors)}
        self._last_failure_substitutions = substitutions

        resolved: list[xarray.Dataset] = []
        for i, s in enumerate(states):
            if s is None:
                donor_idx = substitutions[i]
                resolved.append(states[donor_idx])  # type: ignore[arg-type]
            else:
                resolved.append(s)

        logger.warning(
            "Resampled %d failed ensemble members from successful donors: %s",
            len(failed),
            substitutions,
        )
        return resolved

    def apply_failure_substitutions_to_params(
        self,
        params: xarray.Dataset,
    ) -> xarray.Dataset:
        """Apply the most recent failure substitutions to a parameter ensemble.

        For each failed member ``j`` recorded in the previous ``run_ensemble``
        call, replace ``params.isel(ensemble=j)`` with ``params.isel(ensemble=donor)``
        plus Gaussian jitter scaled to ``failure_jitter_scale * std(ensemble)``
        per data variable. Returns a new dataset; ``params`` is not mutated.

        If there were no failures, returns ``params`` unchanged.
        """
        substitutions = self._last_failure_substitutions
        if not substitutions or "ensemble" not in params.dims:
            return params

        scale = self._failure_jitter_scale
        new_data_vars: dict = {}
        for name, da in params.data_vars.items():
            # Force a writable numpy copy: xarray's deep copy can preserve
            # read-only buffers when the original is JAX-backed.
            arr = np.array(da.values, copy=True)
            ensemble_axis = da.dims.index("ensemble")
            std = np.asarray(params[name].std(dim="ensemble").values)
            for j, donor in substitutions.items():
                donor_slice = np.take(arr, donor, axis=ensemble_axis)
                if scale > 0.0:
                    noise = self._failure_rng.standard_normal(donor_slice.shape)
                    donor_slice = donor_slice + scale * std * noise
                idx: list[slice | int] = [slice(None)] * arr.ndim
                idx[ensemble_axis] = j
                arr[tuple(idx)] = donor_slice
            new_data_vars[name] = (da.dims, arr, da.attrs)
        return xarray.Dataset(
            data_vars=new_data_vars, coords=params.coords, attrs=params.attrs
        )

    def _run_ensemble_sequentially_on_disk(
        self,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """Run the ensemble sequentially, saving results to disk.

        Each member's results_dir is set to the ensemble's results_dir
        before running. Override in subclasses for custom behavior.
        """
        pbar = tqdm(
            enumerate(self.ensemble_forward_models),
            total=self.ensemble_size,
            desc="Running ensemble",
        )
        for i, model in pbar:
            model.set_results_dir(self.results_dir)
            model(
                state=self.get_member_state(state, i, sim_name),  # type: ignore[arg-type]
                params=self.get_member_params(params, i),
                sim_name=f"{sim_name}_{i}",
            )

        return None

    def get_states(self) -> xarray.Dataset:
        """Get the state from disk."""
        states = []
        for i, model in enumerate(self.ensemble_forward_models):
            state = model.get_states(sim_name=f"state_{i}")
            states.append(state)
        return xarray.concat(states, dim="ensemble", join="override")

    def _clean_output(self) -> None:
        """Clean the output folders."""
        for model in self.ensemble_forward_models:
            model._clean_output()

    def _run_parallel(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the ensemble in parallel."""
        if self.save_on_disk:
            for model in self.ensemble_forward_models:
                model.set_results_dir(self.results_dir)

        self._last_failure_substitutions = {}
        failed: list[int] = []

        executor_kwargs: dict[str, Any] = {"max_workers": self.num_parallel_processes}
        # ``forkserver`` instead of ``fork``: the parent imports JAX at module
        # load time, which starts background threads. Bare ``fork()`` clones
        # only the calling thread and leaves the others' mutexes locked,
        # producing the deadlock the JAX/popen_fork RuntimeWarning calls out.
        # ``forkserver`` runs a small helper process that does the forking;
        # workers inherit no parent threads, so JAX/MPI cleanup at shutdown
        # stays clean (no Py_FinalizeEx exit code 120 from orphaned threads).
        ctx = mp.get_context("forkserver")
        executor_kwargs["mp_context"] = ctx
        if not cpu_pinning_disabled():
            cpu_queue = build_cpu_queue(
                ctx=ctx,
                num_workers=self.num_parallel_processes,
                cpus_per_worker=self.num_cpus_per_process,
            )
            executor_kwargs.update(
                initializer=pin_worker_initializer,
                initargs=(cpu_queue,),
            )

        with ProcessPoolExecutor(**executor_kwargs) as executor:
            futures = [
                executor.submit(
                    model.__call__,
                    state=self.get_member_state(state, i, sim_name),  # type: ignore[arg-type]
                    params=self.get_member_params(params, i),
                    sim_name=f"{sim_name}_{i}",
                )
                for i, model in enumerate(self.ensemble_forward_models)
            ]

            future_to_idx = {future: i for i, future in enumerate(futures)}
            states: dict[int, Optional[xarray.Dataset]] = {
                i: None for i in range(self.ensemble_size)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    states[idx] = future.result()
                except subprocess.CalledProcessError as exc:
                    if self._failure_policy == "raise":
                        raise
                    logger.warning(
                        "Ensemble member %d failed (%s); will be resampled "
                        "from a successful member.",
                        idx,
                        exc,
                    )
                    failed.append(idx)
                    states[idx] = None

        if self.rollout:
            self.rollout_step += 1
            for model in self.ensemble_forward_models:
                model.rollout_step = self.rollout_step  # type: ignore[attr-defined]

        if self.save_on_disk:
            if failed:
                # On-disk parallel failure handling is not implemented: failed
                # members produced no state file, so a downstream consumer
                # would read stale data. Fail loudly rather than silently.
                raise RuntimeError(
                    f"Parallel on-disk run had {len(failed)} member failure(s) "
                    f"({failed}); on-disk resample-from-successes is not yet "
                    "supported. Switch to save_in_memory or use sequential."
                )
            return None

        ordered = [states[i] for i in range(self.ensemble_size)]
        resolved = self._resolve_failures(ordered, sorted(failed))
        return xarray.concat(resolved, dim="ensemble", join="override")

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """
        Run the forward model ensemble.

        Dispatches to parallel, sequential in-memory, or sequential on-disk
        based on configuration.

        Args:
            state: The state for the ensemble. Can be an xarray.Dataset with
                an ensemble dimension, a pathlib.Path to a directory of
                per-member files, or None.
            params: The parameters with an ensemble dimension.
            sim_name: The base simulation name. Each member will be saved
                as "{sim_name}_{i}.nc".

        Returns:
            The ensemble state if save_in_memory, otherwise None.
        """
        if self.parallel_execution:
            return self._run_parallel(
                state=state,
                params=params,
                sim_name=sim_name,
            )
        elif self.save_in_memory:
            return self._run_ensemble_sequentially_in_memory(
                state=state,
                params=params,
                sim_name=sim_name,
            )
        else:
            return self._run_ensemble_sequentially_on_disk(  # type: ignore[func-returns-value]
                state=state,
                params=params,
                sim_name=sim_name,
            )
