"""Generative cold start for the neural surrogate (plan 07, phase 4).

With ``spinup_source: generative`` the surrogate's initial field is *sampled*
from a trained :class:`~neural_surrogates.TadpoleLatentGenerator` (conditional
latent flow matching) instead of being produced by a CFD spin-up or loaded from
a training snapshot. :class:`GenerativeSpinup` is the reusable loader/sampler
both :class:`~neural_surrogates.NeuralSurrogateForwardModel` (single member) and
:class:`~neural_surrogates.NeuralSurrogateEnsembleForwardModel` (batched) call.

Design points, in order of importance:

* **Regenerate on every cold forward call.** Nothing about a generated state is
  cached: every ``generate`` call samples afresh from the members' *current*
  parameter values, so an ESMDA parameter update is followed by a newly
  conditioned initial state at the next forecast. Only the immutable pieces --
  weights, the canonical template, the fluid mask and its SDF features -- are
  loaded once and reused.
* **Common random numbers.** Each member's base noise is seeded from
  ``(seed, stable_member_index)`` alone -- never from its batch position or the
  batch composition -- so member ``i`` sees the same latent noise across ESMDA
  iterations and across ``sample_batch_size`` settings. A changed parameter then
  changes the sample *only* through the conditioning, which keeps the
  assimilation map free of unrelated Monte Carlo noise.
* **Explicit template.** The deployment template supplies the canonical grid
  coordinates and the obstacle mask (``blanking``); its velocity values are
  never used. Obstacles are never inferred from generated zeros, and an
  unsupported geometry (one the generator was not trained on) is rejected
  rather than sampled blindly.
* **No CFD fallback.** A failed sample raises with the member index and its
  conditioning values.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
from typing import Any, Optional, Sequence, cast

import numpy as np
import torch
import xarray as xr

logger = logging.getLogger(__name__)

#: Relative tolerance on the template's cell spacing vs the schema's.
_SPACING_RTOL = 1e-4
#: Absolute slack (in cells) on the template's coordinate range vs the schema
#: bounds: cell-centred coordinates must lie inside the bounds, with the first
#: and last within one cell of the respective edge.
_BOUNDS_ATOL = 1e-6

#: The ONE mask polarity the whole plan-07 stack speaks. The training script
#: writes exactly this string into the artifact (it refuses a config that says
#: anything else) and the loader below compares for equality, so an artifact
#: can never carry a different polarity than the one the deployment applies:
#: a silently inverted mask would sample the flow inside the buildings.
MASK_CONVENTION = "blanking: 1 = obstacle; model fluid mask = 1 - blanking"


def geometry_fingerprint(mask: Any) -> str:
    """sha256 of a binary **fluid** mask, in ``(z, y, x)`` order.

    Shape and fluid-cell count do not identify a geometry: relocating the
    obstacles leaves both unchanged while producing a domain the generator has
    never seen. The training script hashes every train-split mask into
    ``supported_geometries`` and :meth:`GenerativeSpinup._check_supported_geometry`
    hashes the deployment template the same way, so only the very same obstacle
    layout is accepted.
    """
    arr = np.asarray(mask, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"geometry fingerprint needs a 3D (z, y, x) mask, got shape {arr.shape}."
        )
    if not np.all(np.isin(arr, (0.0, 1.0))):
        raise ValueError(
            "geometry fingerprint needs a binary fluid mask (values in {0, 1})."
        )
    return hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    ).hexdigest()


def _member_seed(seed: int, member_index: int) -> int:
    """Deterministic per-member RNG seed.

    ``seed * 1_000_003 + member_index``: a fixed affine combination so that
    (a) a member's noise depends on the configured seed and its stable index
    only, and (b) distinct ``(seed, member)`` pairs map to distinct generator
    seeds for any ensemble smaller than 1_000_003 members.
    """
    if seed < 0 or member_index < 0:
        raise ValueError(
            f"seed and member_index must be non-negative, got seed={seed}, "
            f"member_index={member_index}."
        )
    return int(seed) * 1_000_003 + int(member_index)


def constant_history(values: np.ndarray, hp: int) -> np.ndarray:
    """Repeat a ``(P,)`` parameter vector ``hp`` times -> ``(hp, P)``.

    The cold-start conditioning: the generator was trained on an oldest-first
    parameter history of ``hp`` steps, and a cold start has no history, so the
    current values are treated as having held for the whole history window
    (the ``constant_prehistory`` convention of the training dataset).
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if hp < 1:
        raise ValueError(f"hp must be >= 1, got {hp}.")
    return np.repeat(values[None, :], int(hp), axis=0)


def _plain(value: Any) -> Any:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


class GenerativeSpinup:
    """Load a trained latent generator and sample cold-start snapshots.

    Everything is loaded lazily on the first :meth:`generate` (or the first
    schema-dependent property access), so constructing one is free and an
    ensemble can share a single instance read-only across members.

    Args:
        model_dir: Generator artifact folder (``config.yaml`` + ``weights.pt``)
            written by ``scripts/neural_surrogate/train_latent_generator.py``.
        template_path: NetCDF carrying the canonical coordinates and the
            obstacle mask (``blanking``) of the deployment geometry. Its state
            values are ignored.
        seed: Base seed for the per-member latent noise (see
            :func:`_member_seed`).
        sample_batch_size: Maximum number of members sampled per generator
            call (memory bound; results are batch-independent).
        num_sampling_steps: Euler steps for the flow sampler; ``None`` uses the
            validated default recorded in the artifact
            (``generator.sampling.num_steps``).
        device: Torch device string.
        dtype: Torch dtype name for the weights / geometry tensors. Sampling
            itself always runs in fp32 with autocast disabled (the generator's
            own contract).
        default_params: Constant fallbacks for trained parameters a member's
            params dataset omits; a parameter absent from both raises.
        geometry_var: Name of the obstacle-indicator variable on the template
            (``1`` = obstacle; the model's fluid mask is ``1 - geometry_var``).
        expected_units: Optional ``{variable: unit}`` map the *deployment*
            asserts: every listed variable's unit must equal the artifact's.
            The artifact's own units are always required to cover every state
            and parameter variable; this adds the caller's expectation on top,
            so a generator trained in ``deg`` cannot be driven with ``rad``.
    """

    def __init__(
        self,
        model_dir: str | pathlib.Path,
        template_path: str | pathlib.Path,
        seed: int = 0,
        sample_batch_size: int = 8,
        num_sampling_steps: Optional[int] = None,
        device: str = "cpu",
        dtype: str = "float32",
        default_params: Optional[dict[str, float]] = None,
        geometry_var: str = "blanking",
        expected_units: Optional[dict[str, str]] = None,
    ) -> None:
        if model_dir is None or template_path is None:
            raise ValueError(
                "GenerativeSpinup needs both model_dir and template_path "
                f"(got model_dir={model_dir!r}, template_path={template_path!r})."
            )
        if int(sample_batch_size) < 1:
            raise ValueError(
                f"sample_batch_size must be a positive int, got {sample_batch_size!r}."
            )
        if num_sampling_steps is not None and int(num_sampling_steps) < 1:
            raise ValueError(
                "num_sampling_steps must be a positive int or None, got "
                f"{num_sampling_steps!r}."
            )
        self.model_dir = pathlib.Path(model_dir)
        self.template_path = pathlib.Path(template_path)
        self.seed = int(seed)
        self.sample_batch_size = int(sample_batch_size)
        self._requested_num_steps = (
            int(num_sampling_steps) if num_sampling_steps is not None else None
        )
        self.device = torch.device(device)
        self.torch_dtype = getattr(torch, dtype)
        self.default_params = dict(default_params) if default_params else {}
        self.geometry_var = geometry_var
        self.expected_units = (
            {str(k): str(v) for k, v in dict(expected_units).items()}
            if expected_units
            else None
        )

        # Optional diagnostics: when set, every generate() call writes its
        # snapshots under ``<diagnostics_dir>/call_<k>/member_<i>.nc``. These
        # files are write-only records -- nothing reads them back as an input.
        self._diagnostics_dir: Optional[pathlib.Path] = None
        self._diagnostics_calls = 0

        # Lazily populated, immutable once set.
        self._config: Any = None
        self._schema: Optional[dict[str, Any]] = None
        self._model: Any = None
        self._template: Optional[xr.Dataset] = None
        self._geometry: Optional[torch.Tensor] = None  # (1, *grid) fluid mask
        self._geom_features: Optional[torch.Tensor] = None  # (1, F, *grid)

    # -- diagnostics -------------------------------------------------------

    @property
    def diagnostics_dir(self) -> Optional[pathlib.Path]:
        return self._diagnostics_dir

    @diagnostics_dir.setter
    def diagnostics_dir(self, value: Optional[str | pathlib.Path]) -> None:
        self._diagnostics_dir = pathlib.Path(value) if value is not None else None
        # Calls are numbered per directory so a caller that points the
        # diagnostics at a new folder (e.g. per assimilation window) gets
        # ``call_0`` again.
        self._diagnostics_calls = 0

    # -- artifact loading --------------------------------------------------

    def _ensure_schema(self) -> dict[str, Any]:
        """Read ``config.yaml`` (cheap; no weights) and cache the physical schema."""
        if self._schema is not None:
            return self._schema
        from omegaconf import DictConfig, OmegaConf

        cfg_path = self.model_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"generator config not found at {cfg_path}; expected a folder "
                "written by scripts/neural_surrogate/train_latent_generator.py."
            )
        cfg = OmegaConf.load(cfg_path)
        if not isinstance(cfg, DictConfig):
            raise ValueError(f"{cfg_path} must hold a mapping, got a list.")
        generator = cfg.get("generator")
        if generator is None or generator.get("physical_schema") is None:
            raise ValueError(
                f"{cfg_path} carries no generator.physical_schema block; the "
                "generative spin-up needs the saved state/parameter ordering, "
                "grid and supported geometries."
            )
        schema: dict[str, Any] = dict(_plain(generator.physical_schema))
        for key in ("state_vars", "param_vars", "param_history_steps", "grid"):
            if key not in schema:
                raise ValueError(
                    f"generator.physical_schema in {cfg_path} lacks '{key}'."
                )
        schema["state_vars"] = tuple(schema["state_vars"])
        schema["param_vars"] = tuple(schema["param_vars"])
        schema["param_history_steps"] = int(schema["param_history_steps"])
        schema["coordinate_order"] = tuple(
            schema.get("coordinate_order", ("z", "y", "x"))
        )
        sampling = _plain(generator.get("sampling")) or {}
        schema["saved_num_steps"] = (
            int(sampling["num_steps"])
            if sampling.get("num_steps") is not None
            else None
        )
        convention = str(schema.get("geometry_mask_convention", ""))
        if convention != MASK_CONVENTION:
            raise ValueError(
                f"generator geometry_mask_convention {convention!r} is not the "
                f"canonical {MASK_CONVENTION!r}; the deployment applies exactly "
                "that polarity, so an artifact stating another one would invert "
                "the fluid mask (sampling the flow inside the obstacles)."
            )
        if self.geometry_var not in convention:
            raise ValueError(
                f"generator geometry_mask_convention {convention!r} does not "
                f"refer to the template mask variable {self.geometry_var!r}; "
                "the deployment mask would not match the training one."
            )
        schema["units"] = self._validated_units(schema, cfg_path)
        self._config = cfg
        self._schema = schema
        return schema

    def _validated_units(self, schema: dict[str, Any], cfg_path: Any) -> dict[str, str]:
        """The artifact's ``units`` map, complete and (optionally) as expected.

        The physical values the deployment feeds the generator are only
        meaningful in the units it was trained on, so the artifact must state a
        unit for EVERY state and parameter variable, and an explicit
        ``expected_units`` (the caller's own convention) must agree with it.
        """
        raw = schema.get("units") or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"generator.physical_schema.units in {cfg_path} must be a "
                f"{{variable: unit}} map, got {type(raw).__name__}."
            )
        units = {str(k): str(v) for k, v in raw.items() if v not in (None, "")}
        needed = (*schema["state_vars"], *schema["param_vars"])
        missing = [name for name in needed if name not in units]
        if missing:
            raise ValueError(
                f"generator.physical_schema.units in {cfg_path} lacks an entry "
                f"for {missing}; every state and parameter variable needs a unit "
                "before its values can be fed to the generator."
            )
        if self.expected_units:
            wrong = {
                name: (units.get(name), unit)
                for name, unit in self.expected_units.items()
                if units.get(name) != unit
            }
            if wrong:
                raise ValueError(
                    "generator units do not match the configured expected_units "
                    f"(variable: artifact vs expected): {wrong}."
                )
        return units

    def _load_model(self, cfg: Any, schema: dict[str, Any]) -> torch.nn.Module:
        """Rebuild the generator from the artifact and load its weights strictly.

        The seam tests replace to inject an instrumented stub: the returned
        module must expose ``sample``, ``set_conditioning_schema``,
        ``param_history_steps``, ``n_params``, ``n_state_channels``,
        ``state_latent_dim``, ``latent_grid_for`` and ``ae`` (for the optional
        SDF-feature hook).

        The artifact's conditioning schema is installed on the rebuilt model
        right after the weights: the ordered ``param_vars`` and the saved
        cadence are physical contracts no tensor shape encodes, so every
        :meth:`generate` call can have the model re-check them.
        """
        from hydra.utils import instantiate

        weights = self.model_dir / "weights.pt"
        if not weights.exists():
            raise FileNotFoundError(f"generator weights not found at {weights}.")
        model: torch.nn.Module = instantiate(
            cfg.architecture,
            n_state_channels=len(schema["state_vars"]),
            n_params=len(schema["param_vars"]),
            _convert_="all",
        )
        model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        dt = schema.get("history_dt_seconds")
        model.set_conditioning_schema(  # type: ignore[operator]
            schema["param_vars"], None if dt is None else float(dt)
        )
        return model

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        schema = self._ensure_schema()
        model = self._load_model(self._config, schema)
        model = model.to(device=self.device, dtype=self.torch_dtype)
        model.eval()
        # The artifact is the source of truth for the conditioning layout; a
        # model that disagrees with its own schema is a broken export.
        hp = int(getattr(model, "param_history_steps", schema["param_history_steps"]))
        if hp != schema["param_history_steps"]:
            raise ValueError(
                f"generator param_history_steps={hp} disagrees with the saved "
                f"physical_schema ({schema['param_history_steps']})."
            )
        n_params = int(getattr(model, "n_params", len(schema["param_vars"])))
        if n_params != len(schema["param_vars"]):
            raise ValueError(
                f"generator n_params={n_params} disagrees with the saved "
                f"param_vars {schema['param_vars']}."
            )
        self._model = model
        return model

    # -- template / geometry -----------------------------------------------

    def _ensure_template(self) -> xr.Dataset:
        """Load, canonicalise and validate the deployment template once."""
        if self._template is not None:
            return self._template
        # Local import: forward_model imports this module at load time.
        from .forward_model import NeuralSurrogateForwardModel

        schema = self._ensure_schema()
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"generative spin-up template not found at {self.template_path}."
            )
        with xr.open_dataset(self.template_path) as ds:
            raw = ds.load()
        snap = NeuralSurrogateForwardModel._to_regular_grid(raw)
        if "time" in snap.dims:
            snap = snap.isel(time=-1)
        if "time" in snap.coords:
            snap = snap.drop_vars("time")

        if self.geometry_var not in snap.data_vars:
            raise ValueError(
                f"template {self.template_path} carries no '{self.geometry_var}' "
                "variable. The generative spin-up needs an explicit obstacle "
                "mask; obstacles are never inferred from zero velocities."
            )
        dims = schema["coordinate_order"]
        missing = [v for v in schema["state_vars"] if v not in snap.data_vars]
        if missing:
            raise ValueError(
                f"template {self.template_path} lacks state variables {missing} "
                f"(generator state_vars={schema['state_vars']})."
            )
        for name in (*schema["state_vars"], self.geometry_var):
            if tuple(snap[name].dims) != dims:
                raise ValueError(
                    f"template variable '{name}' has dims {snap[name].dims}, "
                    f"expected the generator's coordinate order {dims}."
                )
        self._check_grid(snap, schema)

        blanking = np.asarray(snap[self.geometry_var].values, dtype=np.float64)
        if not np.all(np.isin(blanking, (0.0, 1.0))):
            raise ValueError(
                f"template '{self.geometry_var}' must be a binary obstacle "
                "indicator (1 = obstacle), got values outside {0, 1}."
            )
        fluid = 1.0 - blanking
        self._check_supported_geometry(fluid, schema)

        # Keep only what the generated snapshot must carry: the state variables
        # (overwritten per member) and the mask. Any other template variable
        # would otherwise leak stale values into every generated state.
        keep = [*schema["state_vars"], self.geometry_var]
        template = snap[keep].copy(deep=True)
        self._template = template

        model = self._ensure_model()
        geometry = torch.from_numpy(fluid).to(
            device=self.device, dtype=self.torch_dtype
        )
        self._geometry = geometry.unsqueeze(0)  # (1, *grid)
        ae = getattr(model, "ae", None)
        if (
            ae is not None
            and getattr(ae, "n_geom_feature_channels", 0) > 0
            and hasattr(ae, "_sdf_features")
        ):
            with torch.no_grad():
                self._geom_features = ae._sdf_features(self._geometry).to(
                    device=self.device, dtype=self.torch_dtype
                )
        return template

    def _check_grid(self, snap: xr.Dataset, schema: dict[str, Any]) -> None:
        """Grid shape, spacing and bounds vs the artifact's ``grid`` block."""
        grid = schema["grid"]
        dims = schema["coordinate_order"]
        expected_shape = tuple(int(grid[f"n{d}"]) for d in dims)
        got_shape = tuple(int(snap.sizes[d]) for d in dims)
        if got_shape != expected_shape:
            raise ValueError(
                f"template grid {dict(zip(dims, got_shape))} does not match the "
                f"generator's trained grid {dict(zip(dims, expected_shape))}."
            )
        bounds = grid.get("bounds")
        for d in dims:
            spacing = grid.get(f"d{d}")
            coord = np.asarray(snap[d].values, dtype=np.float64)
            if spacing is None or coord.size < 2:
                continue
            spacing = float(spacing)
            if not np.allclose(np.diff(coord), spacing, rtol=_SPACING_RTOL, atol=0.0):
                raise ValueError(
                    f"template '{d}' spacing {np.diff(coord).mean():.6g} does not "
                    f"match the generator's trained d{d}={spacing:.6g}."
                )
            if bounds is None:
                continue
            # bounds are ordered (x, y, z) like the domain config.
            lo, hi = (float(b) for b in bounds["xyz".index(d)])
            tol = _BOUNDS_ATOL * max(1.0, abs(hi - lo))
            if (
                coord[0] < lo - tol
                or coord[0] >= lo + spacing + tol
                or coord[-1] > hi + tol
                or coord[-1] <= hi - spacing - tol
            ):
                raise ValueError(
                    f"template '{d}' coordinates span [{coord[0]:.6g}, "
                    f"{coord[-1]:.6g}], which does not sit inside the generator's "
                    f"trained bounds [{lo:.6g}, {hi:.6g}] at spacing {spacing:.6g}."
                )

    def _check_supported_geometry(
        self, fluid: np.ndarray, schema: dict[str, Any]
    ) -> None:
        """The template geometry must be one the generator was trained on.

        Unseen geometries need a held-out evaluation before they can be
        trusted (plan 07 §4), so anything not in ``supported_geometries`` is
        rejected here rather than sampled blindly. Identity is the full
        :func:`geometry_fingerprint` -- shape and fluid-cell count alone are
        satisfied by any relocation of the same obstacles.
        """
        supported = schema.get("supported_geometries") or []
        shape = tuple(int(s) for s in fluid.shape)
        fluid_cells = int(round(float(fluid.sum())))
        fingerprint = geometry_fingerprint(fluid)
        for entry in supported:
            if entry.get("mask_sha256") is None:
                raise ValueError(
                    f"the generator artifact at {self.model_dir} has a "
                    f"supported_geometries entry without a 'mask_sha256' "
                    f"({entry}); re-export it with a current "
                    "train_latent_generator.py so the deployment geometry can "
                    "be identified by its mask, not only by its cell count."
                )
            if (
                tuple(int(s) for s in entry.get("shape", ())) == shape
                and int(entry["fluid_cells"]) == fluid_cells
                and str(entry["mask_sha256"]) == fingerprint
            ):
                return
        raise ValueError(
            f"template {self.template_path} carries a geometry (shape={shape}, "
            f"fluid_cells={fluid_cells}, mask_sha256={fingerprint}) the generator "
            f"was not trained on; supported_geometries = {supported}. Unseen "
            "geometries -- including a relocation of the same obstacles -- need "
            "a held-out evaluation before deployment."
        )

    # -- public properties -------------------------------------------------

    @property
    def hp(self) -> int:
        """Parameter-history length the generator conditions on."""
        return int(self._ensure_schema()["param_history_steps"])

    @property
    def param_vars(self) -> tuple[str, ...]:
        """Parameter names in the generator's saved conditioning order."""
        return tuple(self._ensure_schema()["param_vars"])

    @property
    def state_vars(self) -> tuple[str, ...]:
        """State variables (channel order) the generator produces."""
        return tuple(self._ensure_schema()["state_vars"])

    @property
    def history_dt_seconds(self) -> Optional[float]:
        dt = self._ensure_schema().get("history_dt_seconds")
        return float(dt) if dt is not None else None

    @property
    def num_sampling_steps(self) -> Optional[int]:
        """Euler steps used for sampling: the configured override, else the
        artifact's validated default (``None`` leaves it to the model)."""
        if self._requested_num_steps is not None:
            return self._requested_num_steps
        saved = self._ensure_schema()["saved_num_steps"]
        return int(saved) if saved is not None else None

    @property
    def grid_shape(self) -> tuple[int, ...]:
        schema = self._ensure_schema()
        return tuple(int(schema["grid"][f"n{d}"]) for d in schema["coordinate_order"])

    def describe(self) -> str:
        """One-line summary for run logs (reads the config, not the weights)."""
        schema = self._ensure_schema()
        return (
            f"GenerativeSpinup(model_dir={self.model_dir}, "
            f"template={self.template_path}, Hp={schema['param_history_steps']}, "
            f"param_vars={list(schema['param_vars'])}, "
            f"state_vars={list(schema['state_vars'])}, grid={self.grid_shape}, "
            f"seed={self.seed}, sample_batch_size={self.sample_batch_size}, "
            f"num_sampling_steps={self.num_sampling_steps}, device={self.device})"
        )

    # -- conditioning ------------------------------------------------------

    def current_param_vector(
        self, params: Optional[xr.Dataset], member_label: Any = "?"
    ) -> np.ndarray:
        """A member's CURRENT parameter values in the saved ``param_vars`` order.

        Time-varying variables contribute their first knot (``isel(time=0)``),
        static ones their scalar; a variable absent from ``params`` falls back
        to ``default_params`` or raises naming the member and the variable.
        """
        values = []
        have = tuple(params.data_vars) if params is not None else ()
        for name in self.param_vars:
            if params is not None and name in params.data_vars:
                da = params[name]
                if "time" in da.dims:
                    da = da.isel(time=0)
                if da.ndim != 0:
                    raise ValueError(
                        f"member {member_label}: parameter '{name}' must be a "
                        f"scalar per member, got dims {da.dims}."
                    )
                value = float(da.values)
            elif name in self.default_params:
                value = float(self.default_params[name])
            else:
                raise ValueError(
                    f"member {member_label}: trained parameter '{name}' is "
                    f"missing from the provided params (have {have}) and has "
                    "no entry in default_params."
                )
            if not np.isfinite(value):
                raise ValueError(
                    f"member {member_label}: parameter '{name}' is not finite "
                    f"({value!r})."
                )
            values.append(value)
        return np.asarray(values, dtype=np.float64)

    def _member_noise(self, member_index: int) -> torch.Tensor:
        """Base latent noise ``(D, Zl, Yl, Xl)`` for one member, drawn on CPU.

        Drawn per member from its own generator so the tensor is identical
        regardless of which members share a batch and of the batch size.
        """
        model = self._ensure_model()
        latent_grid = tuple(model.latent_grid_for(self.grid_shape))
        shape = (int(model.state_latent_dim), *latent_grid)
        gen = torch.Generator().manual_seed(_member_seed(self.seed, member_index))
        return torch.randn(shape, generator=gen, dtype=torch.float32)

    # -- sampling ----------------------------------------------------------

    def generate(
        self,
        member_params: Sequence[Optional[xr.Dataset]],
        member_indices: Sequence[int],
    ) -> list[xr.Dataset]:
        """Sample one canonical snapshot per member from its current parameters.

        Args:
            member_params: One params dataset per member (a member's slice of
                the ensemble params; ``None`` only works if every trained
                parameter has a default).
            member_indices: The members' stable ensemble indices, used to seed
                their base noise (see :func:`_member_seed`).

        Returns:
            One snapshot per member, in input order: the template's canonical
            coordinates, ``state_vars`` filled with the generated field and
            the obstacle mask carried through unchanged. Obstacle cells are
            zero in every state variable.
        """
        if len(member_params) != len(member_indices):
            raise ValueError(
                f"member_params ({len(member_params)}) and member_indices "
                f"({len(member_indices)}) must have the same length."
            )
        model = self._ensure_model()
        template = self._ensure_template()
        assert self._geometry is not None
        hp = self.hp

        histories = [
            constant_history(self.current_param_vector(p, idx), hp)
            for p, idx in zip(member_params, member_indices)
        ]
        noises = [self._member_noise(int(idx)) for idx in member_indices]

        snapshots: list[xr.Dataset] = []
        grid = tuple(self._geometry.shape[1:])
        batch = self.sample_batch_size
        for start in range(0, len(histories), batch):
            stop = start + batch
            idx_chunk = [int(i) for i in member_indices[start:stop]]
            params_hist = torch.from_numpy(np.stack(histories[start:stop], axis=0)).to(
                device=self.device, dtype=torch.float32
            )
            noise = torch.stack(noises[start:stop], dim=0).to(self.device)
            b = params_hist.shape[0]
            geometry = self._geometry.expand(b, *grid)
            geom_features = (
                self._geom_features.expand(b, *self._geom_features.shape[1:])
                if self._geom_features is not None
                else None
            )
            try:
                with torch.no_grad():
                    out = model.sample(
                        params_hist,
                        geometry,
                        geom_features,
                        initial_noise=noise,
                        num_steps=self.num_sampling_steps,
                        # The conditioning columns and their spacing are stated
                        # again here so the model re-checks them against the
                        # schema it was loaded with (see _load_model).
                        param_names=self.param_vars,
                        history_dt_seconds=self.history_dt_seconds,
                    )
            except Exception as exc:
                conditioning = {
                    idx: dict(zip(self.param_vars, histories[start + k][0].tolist()))
                    for k, idx in enumerate(idx_chunk)
                }
                raise RuntimeError(
                    f"generative spin-up failed for members {idx_chunk} with "
                    f"current parameters {conditioning}: {exc}"
                ) from exc
            out = out.detach().to(torch.float32)
            expected = (b, len(self.state_vars), *grid)
            if tuple(out.shape) != expected:
                raise RuntimeError(
                    f"generator returned shape {tuple(out.shape)}, expected "
                    f"{expected} for members {idx_chunk}."
                )
            # The generator masks obstacles itself; re-applying the fluid mask
            # makes "obstacle cells are zero" an invariant of this class.
            out = (out * geometry.unsqueeze(1).to(out.dtype)).cpu().numpy()
            if not np.all(np.isfinite(out)):
                raise RuntimeError(
                    f"generator produced non-finite values for members {idx_chunk}."
                )
            for k in range(b):
                snapshots.append(self._write_snapshot(template, out[k]))

        self._write_diagnostics(snapshots, member_indices)
        return snapshots

    def _write_snapshot(self, template: xr.Dataset, arr: np.ndarray) -> xr.Dataset:
        """Write a ``(C, *grid)`` field onto a deep copy of the template."""
        snapshot = template.copy(deep=True)
        for c, var in enumerate(self.state_vars):
            snapshot[var] = (template[var].dims, arr[c].astype(template[var].dtype))
        return snapshot

    def _write_diagnostics(
        self, snapshots: Sequence[xr.Dataset], member_indices: Sequence[int]
    ) -> None:
        if self._diagnostics_dir is None:
            return
        call_dir = self._diagnostics_dir / f"call_{self._diagnostics_calls}"
        self._diagnostics_calls += 1
        call_dir.mkdir(parents=True, exist_ok=True)
        for snapshot, idx in zip(snapshots, member_indices):
            snapshot.to_netcdf(call_dir / f"member_{int(idx)}.nc")
