"""Architecture-agnostic surrogate forward model (``docs/neural_surrogate_plan.md`` §4).

``ForwardModel(BaseForwardModel)`` drops the trained surrogate into the same
machinery as ``pylbm`` / ``pyudales`` / ``pypalm`` (``docs/codebase_guide.md``
§3), so ESMDA, the observation operator, and plotting are unchanged. It is
**independent of architecture** — it instantiates whatever the checkpoint names
and only ever calls the ``SurrogateArchitecture`` interface via ``rollout``.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Optional

import numpy as np
import xarray

from pyurbanair.base_forward_model import BaseForwardModel

logger = logging.getLogger(__name__)


class ForwardModel(BaseForwardModel):
    """Neural-surrogate single-simulation forward model.

    Args:
        checkpoint_path: Path or ``run_id`` of the trained checkpoint (§7).
        nx, ny, nz, bounds: Composed domain — **validated** against the
            checkpoint grid, never used to size the network (§8.1).
        simulation_time, output_frequency: ``output_frequency`` is the
            **requested inference** output spacing; ``num_outputs =
            round(simulation_time / output_frequency)`` (the time-axis
            contract, §4). The network always steps at the checkpoint's
            **native** Δt (the corpus frequency it trained on); the rollout is
            resampled from native frames onto ``output_frequency``. When the
            requested frequency is not a multiple of the native Δt the
            nearest native frame is taken (accepted approximation).
        device: ``"cuda"`` or ``"cpu"`` (informational; JAX placement follows
            ``JAX_PLATFORMS``).
        cold_start: ``"canned"`` (default, IC bank), ``"raise"``, or ``"zeros"``.
        vmap_chunk_size: members per device pass for the ensemble override (D2).
    """

    def __init__(
        self,
        checkpoint_path: str | pathlib.Path,
        *,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        nz: Optional[int] = None,
        bounds: Any = None,
        simulation_time: float = 1.0,
        output_frequency: float = 1.0,
        device: str = "cpu",
        cold_start: str = "canned",
        vmap_chunk_size: Optional[int] = None,
        results_dir: Optional[pathlib.Path] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(results_dir=results_dir)
        self.checkpoint_path = checkpoint_path
        self._composed_grid = (nx, ny, nz, bounds)
        self.simulation_time = float(simulation_time)
        self.output_frequency = float(output_frequency)
        self.num_outputs = round(self.simulation_time / self.output_frequency)
        self.device = device
        self.cold_start = cold_start
        self.vmap_chunk_size = vmap_chunk_size
        # Lazy: defer loading the checkpoint + architecture until first use so
        # forkserver workers stay importable (§4).
        self._ckpt = None
        self._pending_conditioning: Optional[np.ndarray] = None

    # ----- lazy load ------------------------------------------------------
    def ensure_loaded(self) -> None:
        """Load the checkpoint (weights + arch + norm + grid + schema) once."""
        if self._ckpt is not None:
            return
        from .training.checkpoint import load_checkpoint
        from .utils.registry import resolve_checkpoint

        ckpt_dir = resolve_checkpoint(self.checkpoint_path)
        self._ckpt = load_checkpoint(ckpt_dir)
        self._validate_grid()

    def _validate_grid(self) -> None:
        from .data.grid import GridMeta

        nx, ny, nz, bounds = self._composed_grid
        if nx is None or bounds is None:
            return  # nothing composed to validate against
        composed = GridMeta(
            nx=nx, ny=ny, nz=nz,
            bounds=tuple(tuple(float(v) for v in b) for b in bounds),  # type: ignore[arg-type]
        )
        if not composed.matches(self._ckpt.grid):
            raise ValueError(
                "Composed domain does not match the checkpoint grid:\n"
                f"  composed:   {composed.to_dict()}\n"
                f"  checkpoint: {self._ckpt.grid.to_dict()}\n"
                "A neural-surrogate checkpoint is valid only for the grid it "
                "was trained on (D5)."
            )

    @property
    def native_dt(self) -> float:
        """Native autoregressive step size (physical time per ``arch.step``).

        Read from the checkpoint (the corpus frequency it trained on). Old
        checkpoints that predate this field fall back to the requested
        ``output_frequency`` — i.e. the legacy 1:1 step==output behavior.
        """
        self.ensure_loaded()
        native = self._ckpt.native_output_frequency
        return float(native) if native else self.output_frequency

    def _rollout_plan(self) -> tuple[int, np.ndarray]:
        """Native-resolution step count + native indices for each output frame.

        The model rolls out ``n_internal`` steps at ``native_dt``; output frame
        ``j`` (at physical time ``(j+1)*output_frequency``) is the nearest
        native frame (at ``(i+1)*native_dt``). When ``output_frequency`` is a
        multiple of ``native_dt`` this is exact decimation; otherwise the
        nearest frame is selected.
        """
        native_dt = self.native_dt
        n_internal = max(1, round(self.simulation_time / native_dt))
        t_native = (np.arange(n_internal) + 1) * native_dt
        t_out = (np.arange(self.num_outputs) + 1) * self.output_frequency
        sel = np.abs(t_native[None, :] - t_out[:, None]).argmin(axis=1)
        return n_internal, sel

    # ----- BaseForwardModel interface ------------------------------------
    def _apply_inflow_settings(self, params: xarray.Dataset) -> None:
        """Store the per-step conditioning sequence on ``self`` (no files, §4).

        Built at the **native** Δt over the full internal-step count, since the
        network steps at native resolution and the rollout is decimated to the
        requested ``output_frequency`` afterward.
        """
        from .utils.params_io import params_to_conditioning

        self.ensure_loaded()
        n_internal, _ = self._rollout_plan()
        self._pending_conditioning = params_to_conditioning(
            params,
            self._ckpt.schema.param_schema,
            n_internal,
            self.native_dt,
        )

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:
        import jax.numpy as jnp

        from .rollout import rollout_from_history
        from .utils import state_io

        self.ensure_loaded()
        ckpt = self._ckpt
        grid = ckpt.grid
        var_names = ckpt.schema.state_var_names
        k = ckpt.history_len

        # (1) params -> dense per-step conditioning sequence (§1.5).
        if params is not None:
            self._apply_inflow_settings(params)
        conditioning = self._pending_conditioning
        if conditioning is None:
            raise ValueError("No params/conditioning available for run_single.")

        # (2) resolve the initial K-frame history (normalized).
        hist_fields, hist_mask = self._resolve_initial_history(state, conditioning[0])
        hist_fields_n = ckpt.normalization.apply(hist_fields)
        hist_params = np.broadcast_to(conditioning[0], (k, conditioning.shape[1])).copy()

        static = jnp.asarray(ckpt.static_channels)

        # (3) autoregressive rollout at native Δt (§1.1).
        n_internal, sel = self._rollout_plan()
        preds_n = rollout_from_history(
            ckpt.arch,
            jnp.asarray(hist_fields_n),
            jnp.asarray(hist_params),
            jnp.asarray(hist_mask),
            jnp.asarray(conditioning),
            static,
            n_internal,
        )
        preds = ckpt.normalization.invert(np.asarray(preds_n))  # [n_internal, C,Z,Y,X]

        # (4) resample native frames onto the requested output_frequency (§4).
        preds = preds[sel]  # [num_outputs, C, Z, Y, X]

        # (5) re-apply the building mask (no flow in solid cells, D5).
        preds = self._reapply_mask(preds)

        ds = state_io.tensor_to_state(preds, grid, var_names)
        return state_io.trim_to_window(ds, self.num_outputs)

    def save_results(self, state: xarray.Dataset, sim_name: str = "state") -> None:
        """Concrete ``save_results`` delegating to the base NetCDF writer (§4)."""
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        """No external solver outputs to clean."""
        return None

    # ----- helpers --------------------------------------------------------
    def _resolve_initial_history(
        self, state: Optional[xarray.Dataset], cond0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from .utils import state_io

        ckpt = self._ckpt
        var_names = ckpt.schema.state_var_names
        k = ckpt.history_len

        if state is not None:
            # Warm start / rollout window: seed from up to the last K frames of
            # the incoming state (time-indexed or a single time-less frame, §4).
            return state_io.state_to_history(state, ckpt.grid, var_names, k)

        return self._cold_start_history(cond0)

    def _cold_start_history(self, cond0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from .utils import state_io

        ckpt = self._ckpt
        c = len(ckpt.schema.state_var_names)
        z, y, x = ckpt.grid.shape_zyx

        if self.cold_start == "raise":
            raise ValueError(
                "state=None but cold_start='raise': supply a spun-up initial "
                "condition, or use a checkpoint with a canned IC bank."
            )
        if self.cold_start == "zeros":
            frame = np.zeros((1, c, z, y, x), dtype=np.float32)
            return state_io.extract_history(frame, ckpt.history_len)

        # "canned": nearest IC by conditioning vector (§4 recommended default).
        bank = ckpt.ic_bank
        if not bank:
            raise ValueError(
                "cold_start='canned' but the checkpoint has no IC bank. "
                "Retrain with an IC bank or pass cold_start='raise'/'zeros'."
            )
        dists = np.linalg.norm(bank["params"] - cond0[None, :], axis=1)
        idx = int(np.argmin(dists))
        frame = bank["fields"][idx][None]  # [1, C, Z, Y, X]
        return state_io.extract_history(frame, ckpt.history_len)

    # Channels with a no-penetration / no-slip wall condition. Pressure does
    # NOT vanish in solid cells, so it is excluded from the mask (review §P2).
    _VELOCITY_VARS = ("u", "v", "w")

    def _reapply_mask(self, preds: np.ndarray) -> np.ndarray:
        """Zero **velocity** in solid cells after decode (no-penetration, D5).

        Pressure (and any non-velocity channel) is left unchanged — masking it
        would corrupt ``pres`` for checkpoints trained with ``include_pressure``.
        """
        fluid = (self._ckpt.geometry_mask <= 0.5).astype(np.float32)  # [Z, Y, X]
        out = preds.copy()
        for c, name in enumerate(self._ckpt.schema.state_var_names):
            if name in self._VELOCITY_VARS:
                out[:, c] = out[:, c] * fluid[None]
        return out
