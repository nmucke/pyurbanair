"""A learned one-step surrogate dressed up as a :class:`BaseForwardModel`.

``NeuralSurrogateForwardModel`` runs a trained one-step network (e.g.
:class:`~neural_surrogates.UNetConvNeXt`) autoregressively so it presents
the same ``run_single(state, params, sim_name)`` contract as the CFD
backends. That lets it slot into the ensemble / ESMDA machinery as a
drop-in fourth forward model alongside pylbm, pyudales and pypalm.

Design notes / requirements honoured here:

* **Trained step size.** The network advances one *trained* output step
  per evaluation. The requested ``output_frequency`` may be a multiple of
  the trained step size; the model then takes several internal steps per
  saved frame. A requested cadence that is not an integer multiple of the
  trained step is rejected.
* **Domain check.** The requested ``(nx, ny, nz, bounds)`` must match the
  domain the network was trained on, otherwise the geometry / channel
  layout would be meaningless. A mismatch raises.
* **Spin-up.** A cold start (``state is None``) is bootstrapped by the
  *spin-up forward model* — the very CFD backend that generated the
  training data — which produces a physically developed initial field.
  Warm starts (a ``state`` is supplied) skip spin-up and roll the network
  straight from the provided snapshot.
* **Geometry from STL.** When an ``stl_path`` is provided the binary
  geometry channel is voxelised from it onto the simulation grid;
  otherwise it falls back to the non-zero-state convention used by
  :class:`~neural_surrogates.data.TransitionDataset`.
"""

from __future__ import annotations

import copy
import logging
import pathlib
from importlib import import_module
from typing import Any, Optional, Sequence

import numpy as np
import torch
import xarray as xr

from pyurbanair.base_forward_model import BaseForwardModel

from .geometry import nonzero_fluid_mask, stl_to_fluid_mask

logger = logging.getLogger(__name__)

_BOUNDS_ATOL = 1e-6


def _clone_backend_forward_model(
    forward_model: BaseForwardModel,
    experiment_base_dir: pathlib.Path,
    experiment_name: str,
) -> BaseForwardModel:
    """Clone a backend forward model into its own per-member directories.

    Dispatches to the backend's own ``create_new_forward_model`` helper
    (both pylbm and pyudales expose one with the same signature), so each
    ensemble member's spin-up runs in an isolated experiment directory.
    """
    backend = type(forward_model).__module__.split(".")[0]
    try:
        helper = import_module(
            f"{backend}.utils.forward_model_utils"
        ).create_new_forward_model
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            f"Spin-up backend '{backend}' does not expose "
            "utils.forward_model_utils.create_new_forward_model; cannot "
            "clone it for an ensemble member."
        ) from exc
    return helper(forward_model, experiment_base_dir, experiment_name)


class NeuralSurrogateForwardModel(BaseForwardModel):
    """Autoregressive neural surrogate that behaves like a forward model."""

    def __init__(
        self,
        spinup_forward_model: BaseForwardModel,
        nx: int,
        ny: int,
        nz: int,
        bounds: Sequence[Sequence[float]],
        simulation_time: float,
        output_frequency: float,
        model_dir: Optional[str | pathlib.Path] = None,
        architecture: Any = None,
        trained_output_frequency: Optional[float] = None,
        trained_domain: Optional[dict[str, Any]] = None,
        state_vars: Optional[Sequence[str]] = None,
        param_vars: Optional[Sequence[str]] = None,
        weights_path: Optional[str | pathlib.Path] = None,
        spinup_time: float = 0.0,
        stl_path: Optional[str | pathlib.Path] = None,
        device: str = "cpu",
        dtype: str = "float32",
        default_params: Optional[dict[str, float]] = None,
        allow_uninitialized_weights: bool = False,
        results_dir: Optional[pathlib.Path] = None,
    ) -> None:
        """Initialise the surrogate.

        Everything describing *the trained network* — its architecture,
        ``state_vars``/``param_vars``, the grid and output cadence it was
        trained on, and its weights — is read from ``model_dir`` (the folder
        :mod:`scripts.train_neural_surrogate` writes: ``config.yaml`` +
        ``weights.pt``). Each of those can still be passed explicitly to
        override the folder (handy for tests), but the normal path is to
        only point at ``model_dir``.

        Args:
            spinup_forward_model: CFD backend (or its un-instantiated config
                node) used to bootstrap cold starts.
            nx, ny, nz, bounds: Requested simulation domain; checked against
                the trained domain.
            simulation_time, output_frequency: Requested run horizon and
                output cadence.
            model_dir: Trained-model folder containing ``config.yaml`` and
                ``weights.pt``. The training-data config it references
                (``dataset.root_dir/config.yaml``) supplies the trained
                domain and output frequency.
            architecture: Override for the network (built ``nn.Module`` or
                config node). Filled from ``model_dir`` when ``None``.
            trained_output_frequency: Override for the trained step size.
            trained_domain: Override for the trained grid.
            state_vars / param_vars: Override for the channel / parameter
                ordering the network expects.
            weights_path: Override for the checkpoint path.
            spinup_time: Spin-up duration for cold starts.
            stl_path: Optional building geometry for the geometry channel.
            device: Torch device string.
            dtype: Torch dtype name (e.g. ``"float32"``).
            default_params: Constant fallbacks for trained parameters that a
                caller's params dataset omits (e.g. uDALES
                ``pressure_gradient_magnitude`` when ESMDA only varies the
                inflow). A trained param absent from both raises.
            allow_uninitialized_weights: When ``True`` a missing checkpoint
                is tolerated (random init) — useful for smoke tests.
            results_dir: When set, results are written to disk like any
                other forward model; when ``None`` they are returned.
        """
        super().__init__(results_dir=results_dir)

        trained = self._load_trained_config(model_dir)
        architecture = architecture if architecture is not None else trained.get(
            "architecture"
        )
        state_vars = state_vars if state_vars is not None else trained.get("state_vars")
        param_vars = param_vars if param_vars is not None else trained.get("param_vars")
        weights_path = (
            weights_path if weights_path is not None else trained.get("weights_path")
        )
        trained_output_frequency = (
            trained_output_frequency
            if trained_output_frequency is not None
            else trained.get("output_frequency")
        )
        trained_domain = (
            trained_domain if trained_domain is not None else trained.get("domain")
        )
        self._require_resolved(
            architecture=architecture,
            state_vars=state_vars,
            param_vars=param_vars,
            trained_output_frequency=trained_output_frequency,
            trained_domain=trained_domain,
            model_dir=model_dir,
        )

        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.bounds = bounds
        self.simulation_time = float(simulation_time)
        self.output_frequency = float(output_frequency)
        self.trained_output_frequency = float(trained_output_frequency)
        self.spinup_time = float(spinup_time)
        self.state_vars = tuple(state_vars)
        self.param_vars = tuple(param_vars)
        self.default_params = dict(default_params) if default_params else {}
        self.stl_path = pathlib.Path(stl_path) if stl_path is not None else None
        self.device = torch.device(device)
        self.torch_dtype = getattr(torch, dtype)

        self._check_domain(trained_domain)
        self.substeps = self._resolve_substeps()

        # With ``_recursive_: false`` on the Hydra config the spin-up backend
        # arrives as an un-instantiated node; build it here (in memory, so its
        # final field can seed the rollout).
        if isinstance(spinup_forward_model, BaseForwardModel):
            self.spinup_forward_model = spinup_forward_model
        else:
            from hydra.utils import instantiate

            self.spinup_forward_model = instantiate(spinup_forward_model)

        # Build the network if a config node was passed, otherwise use it
        # directly. The channel/param counts are derived from the var lists.
        if isinstance(architecture, torch.nn.Module):
            self.model = architecture
        else:
            from hydra.utils import instantiate

            self.model = instantiate(
                architecture,
                n_state_channels=len(self.state_vars),
                n_params=len(self.param_vars),
            )
        self._load_weights(weights_path, allow_uninitialized_weights)
        self.model = self.model.to(device=self.device, dtype=self.torch_dtype)
        self.model.eval()

        # Geometry mask is grid-aligned; built lazily once we have a state
        # template (its dims/coords come from the spin-up backend).
        self._geometry: Optional[torch.Tensor] = None

    # -- trained-model resolution ------------------------------------------

    @staticmethod
    def _load_trained_config(
        model_dir: Optional[str | pathlib.Path],
    ) -> dict[str, Any]:
        """Read everything derivable about the trained network from disk.

        Returns a dict with any of ``architecture``, ``state_vars``,
        ``param_vars``, ``weights_path``, ``output_frequency`` and
        ``domain``. The trained grid and cadence come from the
        training-data config referenced by ``dataset.root_dir`` (written by
        ``scripts/generate_training_data.py``).
        """
        if model_dir is None:
            return {}
        from omegaconf import OmegaConf

        model_dir = pathlib.Path(model_dir)
        cfg_path = model_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"trained-model config not found at {cfg_path}; expected a "
                "folder written by scripts/train_neural_surrogate.py."
            )
        train_cfg = OmegaConf.load(cfg_path)

        resolved: dict[str, Any] = {
            "architecture": train_cfg.architecture,
            "state_vars": tuple(train_cfg.dataset.state_vars),
        }
        weights = model_dir / "weights.pt"
        if weights.exists():
            resolved["weights_path"] = weights

        root_dir = pathlib.Path(train_cfg.dataset.root_dir)
        param_vars = train_cfg.dataset.get("param_vars")
        if param_vars is not None:
            resolved["param_vars"] = tuple(param_vars)
        else:
            names = NeuralSurrogateForwardModel._read_param_names(root_dir)
            if names is not None:
                resolved["param_vars"] = names

        data_cfg_path = root_dir / "config.yaml"
        if data_cfg_path.exists():
            data_cfg = OmegaConf.load(data_cfg_path)
            # generate_training_data.py uses cfg.training_data.output_frequency
            # to drive the forward model, so that's the cadence the saved
            # state files actually sit on. cfg.time.output_frequency is the
            # /time group default and may have been overridden by the
            # training_data overlay (e.g. /time=small says 5.0 but
            # training_data=small sets 1.0).
            td = data_cfg.get("training_data")
            if td is not None and "output_frequency" in td:
                resolved["output_frequency"] = float(td.output_frequency)
            else:
                resolved["output_frequency"] = float(data_cfg.time.output_frequency)
            resolved["domain"] = data_cfg.domain
        return resolved

    @staticmethod
    def _read_param_names(root_dir: pathlib.Path) -> Optional[tuple[str, ...]]:
        """Parameter names (in file order) from the first training param file."""
        param_files = sorted((root_dir / "param" / "train").glob("sample_*.nc"))
        if not param_files:
            return None
        with xr.open_dataset(param_files[0]) as ds:
            return tuple(ds.data_vars)

    @staticmethod
    def _require_resolved(model_dir: Any, **resolved: Any) -> None:
        missing = [name for name, value in resolved.items() if value is None]
        if missing:
            raise ValueError(
                f"could not resolve {missing} for the neural surrogate. Pass a "
                f"model_dir with a complete config.yaml (got model_dir="
                f"{model_dir!r}) or supply these explicitly."
            )

    # -- construction-time validation --------------------------------------

    def _check_domain(self, trained_domain: dict[str, Any]) -> None:
        """Raise unless the requested domain matches the trained one."""
        from omegaconf import OmegaConf

        requested = {"nx": self.nx, "ny": self.ny, "nz": self.nz}
        for key, value in requested.items():
            trained_value = int(trained_domain[key])
            if value != trained_value:
                raise ValueError(
                    f"requested {key}={value} does not match the domain the "
                    f"surrogate was trained on ({key}={trained_value}). The "
                    "network only applies to its training grid."
                )
        trained_bounds = np.asarray(
            OmegaConf.to_container(trained_domain["bounds"])
            if OmegaConf.is_config(trained_domain.get("bounds"))
            else trained_domain["bounds"],
            dtype=float,
        )
        requested_bounds = np.asarray(
            OmegaConf.to_container(self.bounds)
            if OmegaConf.is_config(self.bounds)
            else self.bounds,
            dtype=float,
        )
        if not np.allclose(trained_bounds, requested_bounds, atol=_BOUNDS_ATOL):
            raise ValueError(
                f"requested bounds {requested_bounds.tolist()} do not match the "
                f"trained bounds {trained_bounds.tolist()}."
            )

    def _resolve_substeps(self) -> int:
        """Network steps per saved output frame, for the common case.

        Informational: equals ``round(output_frequency /
        trained_output_frequency)``. The actual emit schedule is computed by
        :meth:`_output_schedule`, which also handles non-integer ratios. The
        surrogate cannot emit *between* trained steps, so a requested cadence
        finer than the trained step size is rejected.
        """
        ratio = self.output_frequency / self.trained_output_frequency
        if ratio < 1.0 - 1e-6:
            raise ValueError(
                f"requested output_frequency={self.output_frequency} is finer "
                f"than the trained step size {self.trained_output_frequency} "
                f"(ratio={ratio:.6f}); the surrogate cannot emit between "
                "trained network steps."
            )
        return max(1, round(ratio))

    def _output_schedule(self) -> tuple[int, list[int]]:
        """Plan the rollout: total network steps + which steps to emit at.

        The network always advances at its trained cadence
        (``trained_output_frequency``). To honour a *requested*
        ``output_frequency`` that differs from it, we emit a frame at the
        internal step whose time is closest to each requested output time —
        so the returned trajectory lands on the requested grid regardless of
        whether the two cadences match or divide evenly.
        """
        n_outputs = round(self.simulation_time / self.output_frequency)
        if n_outputs < 1:
            raise ValueError(
                f"simulation_time={self.simulation_time} / output_frequency="
                f"{self.output_frequency} yields no output frames."
            )
        n_internal = max(
            round(self.simulation_time / self.trained_output_frequency), n_outputs
        )
        emit_steps: list[int] = []
        prev = 0
        for j in range(1, n_outputs + 1):
            target_time = j * self.output_frequency
            k = round(target_time / self.trained_output_frequency)
            # Keep emits strictly increasing and within range.
            k = min(max(k, prev + 1), n_internal)
            emit_steps.append(k)
            prev = k
        return n_internal, emit_steps

    def _load_weights(
        self,
        weights_path: Optional[str | pathlib.Path],
        allow_uninitialized: bool,
    ) -> None:
        path = pathlib.Path(weights_path) if weights_path is not None else None
        if path is not None and path.exists():
            self.model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            return
        if allow_uninitialized:
            logger.warning(
                "No weights found at %s; using randomly initialised surrogate "
                "weights (allow_uninitialized_weights=True).",
                weights_path,
            )
            return
        raise FileNotFoundError(
            f"surrogate weights not found at {weights_path}; pass "
            "allow_uninitialized_weights=True to run with random weights."
        )

    # -- BaseForwardModel hooks --------------------------------------------

    def _apply_inflow_settings(self, params: xr.Dataset) -> None:
        """No-op: parameters enter the network directly per step."""

    def save_results(self, state: xr.Dataset, sim_name: str = "state") -> None:
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        """Nothing on disk to clean for the surrogate itself."""

    # -- geometry ----------------------------------------------------------

    def _build_geometry(self, template: xr.Dataset) -> torch.Tensor:
        """Build the binary geometry channel aligned to ``template``."""
        template_var = template[self.state_vars[0]]
        if "time" in template_var.dims:
            template_var = template_var.isel(time=-1)
        if self.stl_path is not None:
            mask = stl_to_fluid_mask(self.stl_path, template_var)
        else:
            single = template.isel(time=-1) if "time" in template.dims else template
            mask = nonzero_fluid_mask(single, self.state_vars)
        return torch.from_numpy(mask).to(device=self.device, dtype=self.torch_dtype)

    # -- parameter handling ------------------------------------------------

    def _initial_params(self, params: Optional[xr.Dataset]) -> Optional[xr.Dataset]:
        """First-time-step slice of ``params`` to drive the spin-up run."""
        if params is None or "time" not in params.dims:
            return params
        return params.isel(time=0)

    def _param_schedule(
        self, params: Optional[xr.Dataset], n_internal: int
    ) -> torch.Tensor:
        """Per-internal-step parameter vectors of shape ``(n_internal, P)``.

        Time-varying params are linearly interpolated onto the network's
        internal step times ``(k+1) * trained_output_frequency``; scalar
        params are broadcast.
        """
        if params is None:
            raise ValueError("NeuralSurrogateForwardModel requires params.")

        target_times = (np.arange(n_internal) + 1) * self.trained_output_frequency
        columns: list[np.ndarray] = []
        for name in self.param_vars:
            if name not in params:
                if name in self.default_params:
                    columns.append(
                        np.full(n_internal, float(self.default_params[name]))
                    )
                    continue
                raise ValueError(
                    f"trained parameter '{name}' is missing from the provided "
                    f"params (have {tuple(params.data_vars)}) and has no entry "
                    f"in default_params."
                )
            da = params[name]
            if "time" in da.dims:
                src_t = np.asarray(da["time"].values, dtype=float)
                src_v = np.asarray(da.values, dtype=float)
                columns.append(np.interp(target_times, src_t, src_v))
            else:
                columns.append(np.full(n_internal, float(da.values)))
        schedule = np.stack(columns, axis=-1)
        return torch.from_numpy(schedule).to(
            device=self.device, dtype=self.torch_dtype
        )

    # -- the rollout -------------------------------------------------------

    @staticmethod
    def _to_regular_grid(state: xr.Dataset) -> xr.Dataset:
        """Collocate to the regular cell-centered grid the network trained on.

        ``scripts/generate_training_data.py`` interpolates pyudales'
        staggered C-grid output (``u@xm``, ``v@ym``, ``w@zm``) to cell
        centers before saving, so the network sees all channels on a common
        regular grid. The spin-up backend, however, returns the *raw*
        staggered field, so it must be collocated the same way before it is
        stacked and fed to the network. Coordinates are then renamed to
        ``(z, y, x)`` so the surrogate's output is a plain regular grid
        (``solver_name: pylbm``). pylbm output is already cell-centered and
        passes through unchanged; the operation is idempotent, so warm-start
        states (the surrogate's own previous output) are left as-is.
        """
        if {"xm", "ym", "zm"} & set(state.dims):
            from pyudales.utils.grid_utils import interpolate_grid

            state = interpolate_grid(state)
        rename = {
            src: dst
            for src, dst in (("xt", "x"), ("yt", "y"), ("zt", "z"))
            if src in state.dims
        }
        return state.rename(rename) if rename else state

    def _get_template_and_initial_state(
        self,
        state: Optional[xr.Dataset],
        params: Optional[xr.Dataset],
        sim_name: Optional[str],
    ) -> xr.Dataset:
        """Return a single-snapshot template carrying coords + initial field.

        On a warm start the supplied ``state`` is used; on a cold start the
        spin-up backend is run to produce a physically developed field. The
        field is collocated to the regular grid the network expects (see
        :meth:`_to_regular_grid`) before it is returned.
        """
        if state is not None:
            snap = state
        else:
            # Cold start: bootstrap with the CFD backend that produced the data.
            self.spinup_forward_model.spinup_time = self.spinup_time
            snap = self.spinup_forward_model(
                params=self._initial_params(params),
                state=None,
                sim_name=f"{sim_name}_spinup" if sim_name else "spinup",
            )
            if snap is None:
                raise RuntimeError(
                    "Spin-up forward model must run in memory (results_dir=None) "
                    "so its final field can seed the surrogate rollout."
                )

        snap = self._to_regular_grid(snap)
        return snap.isel(time=-1) if "time" in snap.dims else snap

    def run_single(
        self,
        state: Optional[xr.Dataset] = None,
        params: Optional[xr.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xr.Dataset:
        template = self._get_template_and_initial_state(state, params, sim_name)
        return self.rollout_batched([template], [params])[0]

    def rollout_batched(
        self,
        templates: Sequence[xr.Dataset],
        params: Sequence[Optional[xr.Dataset]],
    ) -> list[xr.Dataset]:
        """Roll the network forward for a batch of members at once.

        Each ``templates[b]`` is a single-snapshot initial field already on
        the regular grid the network expects (i.e. the output of
        :meth:`_get_template_and_initial_state`), and ``params[b]`` is that
        member's parameter dataset. The members share the trained network, so
        their rollouts run as a single batched forward pass per step (batch
        dimension = member) rather than one Python loop per member — this is
        where the ensemble gets its speed-up once the (parallel) spin-up has
        produced the per-member initial fields.

        Returns one assembled trajectory :class:`~xarray.Dataset` per member,
        in the same order as ``templates``.
        """
        if len(templates) != len(params):
            raise ValueError(
                f"templates ({len(templates)}) and params ({len(params)}) "
                "must have the same length."
            )

        n_internal, emit_steps = self._output_schedule()
        emit_at = {step: pos for pos, step in enumerate(emit_steps)}
        n_members = len(templates)

        # Stack per-member geometry, parameter schedule and initial state into
        # leading-batch tensors; the network forward is batched over dim 0.
        geom = torch.stack([self._build_geometry(t) for t in templates], dim=0)
        # (n_members, n_internal, P)
        schedule = torch.stack(
            [self._param_schedule(p, n_internal) for p in params], dim=0
        )
        # (n_members, C, *grid)
        initial = torch.stack([self._stack_state(t) for t in templates], dim=0)
        current = initial.to(self.device)

        # Per member, per emitted frame: (n_members, n_emit, C, *grid).
        member_frames: list[list[Optional[np.ndarray]]] = [
            [None] * len(emit_steps) for _ in range(n_members)
        ]
        with torch.no_grad():
            for k in range(n_internal):
                param_k = schedule[:, k, :]
                current = self.model(current, param_k, geom)
                pos = emit_at.get(k + 1)
                if pos is not None:
                    step_np = current.cpu().numpy()
                    for b in range(n_members):
                        member_frames[b][pos] = step_np[b]

        # Prepend each member's initial state at t=0 so the output trajectory
        # starts where the spin-up template ended (matching the training-data
        # convention, which includes t=0 in every saved trajectory).
        initial_np = initial.cpu().numpy()
        outputs: list[xr.Dataset] = []
        for b in range(n_members):
            frames = [initial_np[b], *member_frames[b]]
            outputs.append(self._assemble_output(templates[b], frames))
        return outputs

    # -- (de)serialisation between xarray and torch ------------------------

    def _stack_state(self, snapshot: xr.Dataset) -> torch.Tensor:
        """Stack ``state_vars`` of a single snapshot into ``(C, *grid)``."""
        channels = np.stack(
            [np.asarray(snapshot[v].values) for v in self.state_vars], axis=0
        )
        return torch.from_numpy(channels).to(self.torch_dtype)

    def _assemble_output(
        self, template: xr.Dataset, frames: list[np.ndarray]
    ) -> xr.Dataset:
        """Write predicted channels back onto copies of ``template``.

        Reusing the spin-up backend's template preserves the grid coords and
        per-variable dimension order, so the result is observable by the same
        ``ObservationOperator`` as the underlying backend.
        """
        base = template.isel(time=-1) if "time" in template.dims else template
        per_time: list[xr.Dataset] = []
        for frame in frames:
            snapshot = base.copy(deep=True)
            for c, var in enumerate(self.state_vars):
                snapshot[var] = (base[var].dims, frame[c])
            per_time.append(snapshot)
        out = xr.concat(per_time, dim="time", join="override")
        # Frame 0 is the initial state at t=0 (the spin-up template), and
        # frame j (j>=1) is the prediction after j requested-output steps,
        # i.e. at t = j * output_frequency.
        times = np.arange(len(per_time)) * self.output_frequency
        return out.assign_coords(time=times)

    def disable_spinup(self) -> None:
        """Disable spin-up so subsequent cold starts skip it."""
        self.spinup_time = 0.0
        if hasattr(self.spinup_forward_model, "disable_spinup"):
            self.spinup_forward_model.disable_spinup()

    # -- ensemble support --------------------------------------------------

    @property
    def dirs(self) -> Any:
        """Expose the spin-up backend's dirs so the ensemble base can find
        its temp directory."""
        return self.spinup_forward_model.dirs

    def clone_for_member(
        self, experiment_base_dir: pathlib.Path, experiment_name: str
    ) -> "NeuralSurrogateForwardModel":
        """Clone for an ensemble member, sharing the (read-only) network.

        The spin-up backend is cloned into its own experiment directories so
        per-member cold starts don't clobber each other; the torch model is
        shared because inference is stateless.
        """
        clone = copy.copy(self)
        clone.spinup_forward_model = _clone_backend_forward_model(
            self.spinup_forward_model, experiment_base_dir, experiment_name
        )
        clone._geometry = None
        return clone
