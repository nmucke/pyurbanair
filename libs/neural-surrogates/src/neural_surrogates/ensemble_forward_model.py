"""Ensemble wrapper around :class:`NeuralSurrogateForwardModel`.

Each member shares the (stateless, read-only) trained network but owns an
isolated clone of the spin-up backend so per-member cold starts run in
their own experiment directories.

The expensive part of a surrogate ensemble forecast is the CFD *spin-up*
that bootstraps every member's cold start, not the network rollout. So this
ensemble splits the two:

* **Spin-up** is delegated to the backend's own ``EnsembleForwardModel``
  (pyudales / pylbm), which runs all members' spin-ups in parallel across
  processes — exactly the machinery the CFD backends already use for ESMDA.
* **Rollout** is then a single *batched* network pass over all members
  (batch dimension = ensemble member), run in the parent process so CUDA is
  never forked into the spin-up workers.
"""

from __future__ import annotations

import logging
import pathlib
from importlib import import_module
from typing import Optional

import xarray

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

from .forward_model import NeuralSurrogateForwardModel

logger = logging.getLogger(__name__)


class NeuralSurrogateEnsembleForwardModel(BaseEnsembleForwardModel):
    """Ensemble of neural-surrogate forward models."""

    def __init__(
        self,
        forward_model: NeuralSurrogateForwardModel,
        ensemble_size: int = 10,
        temp_dir: Optional[pathlib.Path] = None,
        results_dir: Optional[pathlib.Path] = None,
        num_parallel_processes: int = 1,
        num_cpus_per_process: int = 1,
    ) -> None:
        super().__init__(
            forward_model=forward_model,
            ensemble_size=ensemble_size,
            results_dir=results_dir,
            num_parallel_processes=num_parallel_processes,
            num_cpus_per_process=num_cpus_per_process,
            temp_dir=temp_dir,
        )
        # Built lazily on the first cold-start run and reused across forecast
        # steps (ESMDA calls run_ensemble once per iteration).
        self._spinup_ensemble: Optional[BaseEnsembleForwardModel] = None

    def _create_new_forward_model(  # type: ignore[override]
        self,
        forward_model: NeuralSurrogateForwardModel,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> NeuralSurrogateForwardModel:
        return forward_model.clone_for_member(experiment_base_dir, experiment_name)

    # -- parallel spin-up --------------------------------------------------

    def _get_spinup_ensemble(self) -> BaseEnsembleForwardModel:
        """Backend ensemble forward model that produces the per-member spin-up.

        Reuses the spin-up backends already cloned for each surrogate member
        (one isolated experiment directory per member) instead of cloning a
        second set, then wraps them in the backend's own
        ``EnsembleForwardModel`` so the spin-ups run in parallel.
        """
        if self._spinup_ensemble is not None:
            return self._spinup_ensemble

        spinup_template = self.forward_model.spinup_forward_model
        backend = type(spinup_template).__module__.split(".")[0]
        try:
            ensemble_cls = import_module(
                f"{backend}.ensemble_forward_model"
            ).EnsembleForwardModel
        except (ImportError, AttributeError) as exc:
            raise NotImplementedError(
                f"Spin-up backend '{backend}' does not expose "
                "ensemble_forward_model.EnsembleForwardModel; cannot run its "
                "spin-up as a parallel ensemble."
            ) from exc

        # ensemble_size=0 builds no clones of its own; we inject the spin-up
        # backends already cloned per surrogate member (see clone_for_member).
        spinup_ensemble = ensemble_cls(
            forward_model=spinup_template,
            ensemble_size=0,
            results_dir=None,  # in memory: states must seed the rollout
            num_parallel_processes=self.num_parallel_processes,
            num_cpus_per_process=self.num_cpus_per_process,
        )
        spinup_ensemble.ensemble_forward_models = [
            member.spinup_forward_model for member in self.ensemble_forward_models
        ]
        spinup_ensemble.ensemble_size = self.ensemble_size

        # The surrogate's cold start runs the spin-up for ``spinup_time``; make
        # every member's backend honour the surrogate's configured duration.
        for member in spinup_ensemble.ensemble_forward_models:
            member.spinup_time = self.forward_model.spinup_time

        # Mirror failure handling so a CFD member that crashes is resampled
        # from a successful one (and the substitution is reported upstream).
        spinup_ensemble._failure_policy = self._failure_policy
        spinup_ensemble._failure_jitter_scale = self._failure_jitter_scale
        spinup_ensemble._failure_rng = self._failure_rng

        self._spinup_ensemble = spinup_ensemble
        return spinup_ensemble

    def _spinup_templates(
        self,
        params: Optional[xarray.Dataset],
        sim_name: Optional[str],
    ) -> list[xarray.Dataset]:
        """Run all members' spin-ups in parallel and collocate the results.

        Returns one regular-grid initial-field template per member, ready to
        seed the batched network rollout.
        """
        spinup_ensemble = self._get_spinup_ensemble()
        # Spin-up uses the constant (t=0) inflow, matching the single-model
        # cold start; the time-varying schedule drives the rollout instead.
        initial_params = self.forward_model._initial_params(params)
        spinup_states = spinup_ensemble.run_ensemble(
            state=None,
            params=initial_params,
            sim_name=f"{sim_name}_spinup" if sim_name else "spinup",
        )
        if spinup_states is None:
            raise RuntimeError(
                "Spin-up ensemble must run in memory (results_dir=None) so its "
                "final fields can seed the surrogate rollout."
            )
        # Carry any CFD failure substitutions upstream (e.g. for ESMDA's
        # apply_failure_substitutions_to_params).
        self._last_failure_substitutions = dict(
            spinup_ensemble._last_failure_substitutions
        )
        return [
            self.forward_model._get_template_and_initial_state(
                state=spinup_states.isel(ensemble=i),
                params=self.get_member_params(params, i),
                sim_name=f"{sim_name}_{i}" if sim_name else None,
            )
            for i in range(self.ensemble_size)
        ]

    def _warm_start_templates(
        self,
        state: xarray.Dataset | pathlib.Path,
        params: Optional[xarray.Dataset],
        sim_name: Optional[str],
    ) -> list[xarray.Dataset]:
        """Collocate per-member provided states into rollout templates."""
        self._last_failure_substitutions = {}
        templates: list[xarray.Dataset] = []
        for i in range(self.ensemble_size):
            member_state = self.get_member_state(state, i, sim_name or "state")
            templates.append(
                self.forward_model._get_template_and_initial_state(
                    state=member_state,
                    params=self.get_member_params(params, i),
                    sim_name=f"{sim_name}_{i}" if sim_name else None,
                )
            )
        return templates

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the surrogate ensemble: parallel spin-up + batched rollout.

        Cold starts (``state is None``) bootstrap every member through the
        backend spin-up ensemble in parallel; warm starts roll the supplied
        per-member snapshots forward directly. Either way the network rollout
        is a single batched pass over all members.
        """
        if state is None:
            templates = self._spinup_templates(params, sim_name)
        else:
            templates = self._warm_start_templates(state, params, sim_name)

        member_params = [
            self.get_member_params(params, i) for i in range(self.ensemble_size)
        ]
        outputs = self.forward_model.rollout_batched(templates, member_params)

        if self.save_on_disk:
            self.forward_model.set_results_dir(self.results_dir)
            for i, member_output in enumerate(outputs):
                self.forward_model.save_results(member_output, f"{sim_name}_{i}")
            return None

        ensemble_output: xarray.Dataset = xarray.concat(
            outputs, dim="ensemble", join="override"
        )
        return ensemble_output
