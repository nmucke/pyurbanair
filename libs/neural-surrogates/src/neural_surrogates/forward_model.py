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
  :class:`~neural_surrogates.datasets.transition.TransitionDataset`.
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

from .geometry import nonzero_fluid_mask, solid_c_fluid_mask, stl_to_fluid_mask

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
        spinup_source: str = "forward_model",
        geometry_var: str = "blanking",
        rollout_batch_size: Optional[int] = None,
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
            spinup_source: How a cold start (``state is None``) obtains its
                initial field. ``"forward_model"`` (default) runs the CFD
                spin-up backend that generated the training data — physically
                faithful but expensive. ``"training_data"`` means the run is
                warm-started from training snapshots that are loaded and passed
                in by the caller (``scripts/esmda/run_esmda.py`` via
                :mod:`neural_surrogates.training_spinup`), so the surrogate never
                runs a spin-up itself — it only rolls forward from the provided
                state. A cold start (``state is None``) is therefore rejected in
                this mode. The flag is kept so the ensemble can skip cloning the
                (unused) CFD backend per member and ``prepare`` can skip
                compiling it.
            geometry_var: State variable holding the per-cell obstacle
                indicator (``"blanking"``). When present on the initial-field
                template the geometry channel is built from it directly,
                matching :class:`~neural_surrogates.datasets.transition.TransitionDataset`.
            rollout_batch_size: Maximum number of ensemble members rolled
                forward through the network in a single batched pass. ``None``
                (default) rolls the whole batch at once; set a positive value to
                split a large ensemble into chunks of this size, trading a small
                amount of throughput for a lower peak GPU memory footprint (the
                fix for the rollout OOM on large ensembles / high resolution).
        """
        super().__init__(results_dir=results_dir)

        trained = self._load_trained_config(model_dir)
        architecture = (
            architecture if architecture is not None else trained.get("architecture")
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

        self.spinup_source = spinup_source
        if spinup_source not in ("forward_model", "training_data"):
            raise ValueError(
                f"spinup_source must be 'forward_model' or 'training_data', "
                f"got {spinup_source!r}."
            )
        self.geometry_var = geometry_var
        if rollout_batch_size is not None and rollout_batch_size < 1:
            raise ValueError(
                f"rollout_batch_size must be a positive int or None, "
                f"got {rollout_batch_size!r}."
            )
        self.rollout_batch_size = (
            int(rollout_batch_size) if rollout_batch_size is not None else None
        )

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

        # The domain check needs ``self.model`` to detect whether the network is
        # domain-flexible (decomposes the grid internally), so it runs after the
        # network is built.
        self._check_domain(trained_domain)

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
            # case default and may differ from the training_data generation
            # cadence, so prefer the training_data value when present.
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
        """Validate the requested domain against the trained one.

        A *domain-flexible* model (``getattr(self.model, "domain_flexible",
        False)`` -- e.g. :class:`~neural_surrogates.DomainDecomposed`) does the
        spatial decomposition internally and so applies to any global grid that
        shares the trained cell spacing ``(dx, dy, dz)``; for it ``nx/ny/nz``
        and the absolute bounds are free, and only the spacing must match.
        Every other model is pinned to its exact training grid, so the strict
        ``nx/ny/nz`` + bounds equality check is kept verbatim.
        """
        from omegaconf import OmegaConf

        trained_bounds = np.asarray(
            (
                OmegaConf.to_container(trained_domain["bounds"])
                if OmegaConf.is_config(trained_domain.get("bounds"))
                else trained_domain["bounds"]
            ),
            dtype=float,
        )
        requested_bounds = np.asarray(
            (
                OmegaConf.to_container(self.bounds)
                if OmegaConf.is_config(self.bounds)
                else self.bounds
            ),
            dtype=float,
        )

        if getattr(self.model, "domain_flexible", False):
            self._check_domain_flexible(
                trained_domain, trained_bounds, requested_bounds
            )
            return

        requested = {"nx": self.nx, "ny": self.ny, "nz": self.nz}
        for key, value in requested.items():
            trained_value = int(trained_domain[key])
            if value != trained_value:
                raise ValueError(
                    f"requested {key}={value} does not match the domain the "
                    f"surrogate was trained on ({key}={trained_value}). The "
                    "network only applies to its training grid."
                )
        if not np.allclose(trained_bounds, requested_bounds, atol=_BOUNDS_ATOL):
            raise ValueError(
                f"requested bounds {requested_bounds.tolist()} do not match the "
                f"trained bounds {trained_bounds.tolist()}."
            )

    def _check_domain_flexible(
        self,
        trained_domain: dict[str, Any],
        trained_bounds: np.ndarray,
        requested_bounds: np.ndarray,
    ) -> None:
        """Spacing-invariant domain check for a domain-flexible model.

        The trained and requested cell spacings ``(dx, dy, dz) = (hi - lo) /
        (nx, ny, nz)`` must agree per axis; ``nx/ny/nz`` and absolute bounds
        are otherwise free. ``bounds`` is ordered ``(x, y, z)`` to match
        ``(nx, ny, nz)``.
        """
        trained_dims = np.array(
            [
                int(trained_domain["nx"]),
                int(trained_domain["ny"]),
                int(trained_domain["nz"]),
            ],
            dtype=float,
        )
        requested_dims = np.array([self.nx, self.ny, self.nz], dtype=float)

        d_trained = (trained_bounds[:, 1] - trained_bounds[:, 0]) / trained_dims
        d_req = (requested_bounds[:, 1] - requested_bounds[:, 0]) / requested_dims

        if not np.allclose(d_trained, d_req, atol=_BOUNDS_ATOL):
            raise ValueError(
                f"requested cell spacing (dx, dy, dz)={d_req.tolist()} does not "
                f"match the trained cell spacing {d_trained.tolist()}. A "
                "domain-flexible surrogate may run on a different grid size or "
                "extent, but only at the spacing it was trained on."
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
            self.model.load_state_dict(torch.load(path, map_location=self.device))
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

    def _spinup_solid_c_path(self) -> Optional[pathlib.Path]:
        """Path to the spin-up backend's ``solid_c.txt``, if it wrote one.

        uDALES spin-up writes the obstacle indicator the training data was
        built from into its experiment dir; reading it back is the only way to
        reproduce the trained geometry exactly (see :func:`solid_c_fluid_mask`).
        Returns ``None`` for backends that don't (e.g. pylbm) or before the
        spin-up has run.
        """
        dirs = getattr(self.spinup_forward_model, "dirs", None)
        experiment_dir = getattr(dirs, "experiment_dir", None)
        if experiment_dir is None:
            return None
        path = pathlib.Path(experiment_dir) / "solid_c.txt"
        return path if path.exists() else None

    def _build_geometry(self, template: xr.Dataset) -> torch.Tensor:
        """Build the binary geometry channel aligned to ``template``.

        The mask must be IDENTICAL to the one the network trained on, so the
        sources are tried in order of fidelity to the training pipeline:

        0. The template's own ``geometry_var`` (``blanking``) when present —
           exactly the obstacle indicator the training data carries, used for
           ``training_data`` cold starts and any backend (pylbm) that writes it.
        1. The spin-up backend's ``solid_c.txt`` (uDALES) — exactly the file
           ``generate_training_data.py`` built the training ``blanking`` from.
        2. An explicit ``stl_path`` override, voxelised onto the grid.
        3. The non-zero-state fallback — only correct for backends that write
           exact zeros inside obstacles (pylbm), matching what
           :class:`~neural_surrogates.datasets.transition.TransitionDataset` falls back to.
        """
        if self.geometry_var and self.geometry_var in template.data_vars:
            # Same convention as TransitionDataset: blanking is the obstacle
            # indicator, inverted to the fluid mask (1 = fluid, 0 = obstacle).
            da = template[self.geometry_var]
            if "time" in da.dims:
                da = da.isel(time=-1)
            mask = 1.0 - np.asarray(da.values, dtype=np.float64)
            return torch.from_numpy(mask).to(device=self.device, dtype=self.torch_dtype)
        template_var = template[self.state_vars[0]]
        if "time" in template_var.dims:
            template_var = template_var.isel(time=-1)
        solid_c_path = self._spinup_solid_c_path()
        if solid_c_path is not None:
            mask = solid_c_fluid_mask(solid_c_path, template_var)
        elif self.stl_path is not None:
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
        return torch.from_numpy(schedule).to(device=self.device, dtype=self.torch_dtype)

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

        PALM training data (``solver_name: palm``) keeps the horizontal stagger
        — ``u`` on ``xu``, ``v`` on ``yv`` — at the *same* length as the
        cell-centered ``x``/``y`` axes. The training pipeline
        (:class:`~neural_surrogates.datasets.transition.TransitionDataset`)
        stacks those arrays by index, treating them as collocated, so the
        network never sees the half-cell offset. Reproduce that here: relabel
        ``xu``/``yv`` onto ``x``/``y`` *positionally* (no interpolation, which
        would shift a field the network never trained on) and share one set of
        cell-centered coords across ``u``/``v``/``w`` — yielding the regular
        ``(z, y, x)`` grid the surrogate's observation mapping expects.
        """
        if {"xm", "ym", "zm"} & set(state.dims):
            from pyudales.utils.grid_utils import interpolate_grid

            state = interpolate_grid(state)
        if {"xu", "yv"} & set(state.dims):
            state = NeuralSurrogateForwardModel._destagger_palm_horizontal(state)
        rename = {
            src: dst
            for src, dst in (("xt", "x"), ("yt", "y"), ("zt", "z"))
            if src in state.dims
        }
        return state.rename(rename) if rename else state

    @staticmethod
    def _destagger_palm_horizontal(state: xr.Dataset) -> xr.Dataset:
        """Relabel PALM's ``xu``/``yv`` axes onto the cell-centered ``x``/``y``.

        Each staggered axis has the same length as its cell-centered partner in
        the training data, so the collocation the trainer applies is a pure
        index relabel: rebuild every variable on the canonical ``(z, y, x)``
        dims and assign all of them the shared cell-centered coords. No values
        are interpolated. A direct ``Dataset.rename`` cannot do this because the
        target ``x``/``y`` dims already exist (carried by ``w``), so a per-variable
        rebuild is used instead.
        """
        dim_swap = {"xu": "x", "yv": "y"}
        canonical = {axis: state.coords[axis].values for axis in ("x", "y", "z")}
        if "time" in state.coords:
            canonical["time"] = state.coords["time"].values

        rebuilt: dict[str, tuple] = {}
        for name, da in state.data_vars.items():
            dims = tuple(dim_swap.get(d, d) for d in da.dims)
            rebuilt[name] = (dims, np.asarray(da.values), da.attrs)
        coords = {axis: values for axis, values in canonical.items()}
        return xr.Dataset(rebuilt, coords=coords, attrs=state.attrs)

    def _get_template_and_initial_state(
        self,
        state: Optional[xr.Dataset],
        params: Optional[xr.Dataset],
        sim_name: Optional[str],
        member_index: int = 0,
    ) -> xr.Dataset:
        """Return a single-snapshot template carrying coords + initial field.

        On a warm start the supplied ``state`` is used. On a cold start the
        field is produced by the CFD spin-up backend. The ``training_data``
        spin-up source has no cold start of its own — the caller
        (``scripts/esmda/run_esmda.py``) loads the training snapshots and passes them
        in as the initial ``state`` — so a cold start in that mode is an error.
        The field is collocated to the regular grid the network expects (see
        :meth:`_to_regular_grid`) before it is returned.
        """
        if state is not None:
            snap = state
        elif self.spinup_source == "training_data":
            raise RuntimeError(
                "spinup_source='training_data' has no cold start: the training "
                "snapshots are loaded and passed in as the initial state by the "
                "caller (scripts/esmda/run_esmda.py / neural_surrogates.training_spinup). "
                "Got state=None."
            )
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
        member_index: int = 0,
    ) -> xr.Dataset:
        template = self._get_template_and_initial_state(
            state, params, sim_name, member_index
        )
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

        When ``rollout_batch_size`` is set the members are processed in chunks
        of that size so the peak GPU footprint scales with the chunk, not the
        full ensemble; the per-chunk outputs are concatenated back in order.
        """
        if len(templates) != len(params):
            raise ValueError(
                f"templates ({len(templates)}) and params ({len(params)}) "
                "must have the same length."
            )

        chunk = self.rollout_batch_size
        if chunk is not None and len(templates) > chunk:
            outputs: list[xr.Dataset] = []
            for start in range(0, len(templates), chunk):
                stop = start + chunk
                outputs.extend(
                    self._rollout_chunk(templates[start:stop], params[start:stop])
                )
            return outputs
        return self._rollout_chunk(templates, params)

    def _rollout_chunk(
        self,
        templates: Sequence[xr.Dataset],
        params: Sequence[Optional[xr.Dataset]],
    ) -> list[xr.Dataset]:
        """Roll a single batch (one chunk) of members forward through the net.

        Carries the batched forward pass that :meth:`rollout_batched` splits the
        ensemble across; see that method for the batching rationale.
        """
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

        # The forecast trajectory is the sequence of PREDICTED frames only (one
        # per requested output step, at t = output_frequency, ..., simulation_time).
        # The initial spin-up state at t=0 is deliberately NOT prepended: an
        # assimilation window is half-open [0, simulation_time), so the trajectory
        # must span the same frames as the CFD backends and the truth window.
        # Prepending t=0 added a leading frame that stretched the span by one
        # observation interval, making the interval-binned prediction vector one
        # bin longer than the truth's and breaking the ESMDA Kalman update.
        outputs: list[xr.Dataset] = []
        for b in range(n_members):
            outputs.append(self._assemble_output(templates[b], member_frames[b]))
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
        # The trajectory holds only predicted frames (no t=0 initial state):
        # frame j is the prediction after (j + 1) requested-output steps, i.e. at
        # t = (j + 1) * output_frequency, spanning (0, simulation_time].
        times = (np.arange(len(per_time)) + 1) * self.output_frequency
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

        In ``training_data`` mode the surrogate never runs a spin-up (the caller
        warm-starts every window from provided states), so per-member experiment
        directories serve no purpose — the template's backend is shared instead
        of cloned. This skips building (and renaming namoptions for) one CFD
        ForwardModel per member.
        """
        clone = copy.copy(self)
        if self.spinup_source != "training_data":
            clone.spinup_forward_model = _clone_backend_forward_model(
                self.spinup_forward_model, experiment_base_dir, experiment_name
            )
        clone._geometry = None
        return clone
