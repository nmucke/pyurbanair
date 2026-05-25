"""Ensemble surrogate forward model with on-device batching (D2).

The existing parallel path (``_run_parallel``) is built for CPU-bound Fortran
subprocesses (forkserver + CPU pinning, DRAM-capped). For a GPU NN that is
wrong — N processes contend for one device. This ``EnsembleForwardModel``
**overrides** ``run_ensemble`` to stream members into batched arrays and run the
batched autoregressive rollout via ``jax.vmap``, in sub-batches sized to device
memory (``vmap_chunk_size``). It honors the **full** ``run_ensemble`` contract
(``docs/neural_surrogate_plan.md`` §4): all three save modes, ``state`` arriving
as a ``pathlib.Path`` to per-member files, and the ``rollout_step`` increment.
``num_parallel_processes > 1`` stays available as an optional CPU fallback.
"""

from __future__ import annotations

import copy
import logging
import pathlib
from typing import Optional

import numpy as np
import xarray

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

from .forward_model import ForwardModel

logger = logging.getLogger(__name__)


class EnsembleForwardModel(BaseEnsembleForwardModel):
    """Batched (``vmap``) ensemble surrogate (D2)."""

    def __init__(
        self,
        forward_model: ForwardModel,
        ensemble_size: int = 10,
        temp_dir: Optional[pathlib.Path] = None,
        results_dir: Optional[pathlib.Path] = None,
        num_parallel_processes: int = 1,
        num_cpus_per_process: int = 1,
        vmap_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            forward_model=forward_model,
            ensemble_size=ensemble_size,
            results_dir=results_dir,
            num_parallel_processes=num_parallel_processes,
            num_cpus_per_process=num_cpus_per_process,
            temp_dir=temp_dir,
        )
        self.vmap_chunk_size = vmap_chunk_size or getattr(
            forward_model, "vmap_chunk_size", None
        )

    def _create_new_forward_model(  # type: ignore[override]
        self,
        forward_model: ForwardModel,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> ForwardModel:
        # Share the (immutable) weights/checkpoint; only the per-member result
        # dir differs. A shallow copy keeps the loaded checkpoint reference.
        member = copy.copy(forward_model)
        return member

    # ----- batched run ----------------------------------------------------
    def run_ensemble(
        self,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        # Optional CPU fallback: fork N member processes (smoke only, D2).
        if self.parallel_execution:
            return super().run_ensemble(state=state, params=params, sim_name=sim_name)

        import jax
        import jax.numpy as jnp

        from .rollout import rollout_from_history
        from .utils import state_io

        fm = self.forward_model
        fm.ensure_loaded()
        ckpt = fm._ckpt
        grid = ckpt.grid
        var_names = ckpt.schema.state_var_names
        static = jnp.asarray(ckpt.static_channels)
        resolved_name = sim_name if sim_name is not None else "state"

        n_internal, sel = fm._rollout_plan()

        def run_chunk(hist_n, hist_p, hist_m, cond):
            return rollout_from_history(
                ckpt.arch, hist_n, hist_p, hist_m, cond, static, n_internal
            )

        batched_run = jax.jit(jax.vmap(run_chunk))

        chunk = self.vmap_chunk_size or self.ensemble_size
        member_datasets: list[xarray.Dataset] = []
        for start in range(0, self.ensemble_size, chunk):
            idxs = list(range(start, min(start + chunk, self.ensemble_size)))
            hist_n, hist_p, hist_m, cond = self._stack_member_inputs(
                state, params, idxs, resolved_name
            )
            preds_n = np.asarray(
                batched_run(
                    jnp.asarray(hist_n), jnp.asarray(hist_p),
                    jnp.asarray(hist_m), jnp.asarray(cond),
                )
            )  # [chunk, T, C, Z, Y, X]
            for local, member_idx in enumerate(idxs):
                preds = ckpt.normalization.invert(preds_n[local])
                preds = preds[sel]  # native frames -> requested output_frequency
                preds = fm._reapply_mask(preds)
                ds = state_io.trim_to_window(
                    state_io.tensor_to_state(preds, grid, var_names), fm.num_outputs
                )
                if self.save_on_disk:
                    out = self.results_dir / f"{resolved_name}_{member_idx}.nc"  # type: ignore[operator]
                    ds.to_netcdf(str(out))
                else:
                    member_datasets.append(ds.expand_dims(ensemble=[member_idx]))

        if self.rollout:
            self.rollout_step += 1
            for model in self.ensemble_forward_models:
                model.rollout_step = self.rollout_step  # type: ignore[attr-defined]

        if self.save_on_disk:
            return None
        return xarray.concat(member_datasets, dim="ensemble", join="override")

    def _stack_member_inputs(
        self,
        state: Optional[xarray.Dataset | pathlib.Path],
        params: Optional[xarray.Dataset],
        idxs: list[int],
        sim_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build batched per-member ``(hist_n, hist_p, hist_m, conditioning)``.

        Loads each member's state/params lazily (``state`` may be a ``Path`` to
        per-member files, §4) — never preloads the full ensemble.
        """
        from .utils.params_io import params_to_conditioning

        fm = self.forward_model
        ckpt = fm._ckpt
        schema = ckpt.schema.param_schema
        k = ckpt.history_len
        n_internal, _ = fm._rollout_plan()

        hist_list, histp_list, mask_list, cond_list = [], [], [], []
        for i in idxs:
            member_state = self.get_member_state(state, i, sim_name)
            member_params = self.get_member_params(params, i)
            if member_params is None:
                raise ValueError("Surrogate ensemble requires per-member params.")
            # Conditioning at the native Δt over the full internal-step count;
            # the rollout is decimated to output_frequency after stepping (§4).
            cond = params_to_conditioning(
                member_params, schema, n_internal, fm.native_dt
            )
            hist_fields, hist_mask = fm._resolve_initial_history(member_state, cond[0])
            hist_n = ckpt.normalization.apply(hist_fields)
            hist_params = np.broadcast_to(cond[0], (k, cond.shape[1])).copy()

            hist_list.append(hist_n)
            histp_list.append(hist_params)
            mask_list.append(hist_mask)
            cond_list.append(cond)

        return (
            np.stack(hist_list).astype(np.float32),
            np.stack(histp_list).astype(np.float32),
            np.stack(mask_list).astype(np.float32),
            np.stack(cond_list).astype(np.float32),
        )
