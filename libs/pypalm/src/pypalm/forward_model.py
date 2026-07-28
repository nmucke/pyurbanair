"""pypalm.ForwardModel — wrapper around the PALM LES model."""

import logging
import os
import pathlib
import shutil
import subprocess
import time
from typing import Optional

import numpy as np
import xarray

from pyurbanair.base_forward_model import BaseForwardModel

from . import LOCAL_EXECUTE_SCRIPT, PALM_MODEL_SYSTEM_PATH, PALMRUN_BIN
from .stl_to_palm import stl_to_palm_topography
from .utils.clean_up_utils import clean_palm_output_dir
from .utils.compile_utils import compile_palm
from .utils.dir_utils import PALMDirectoryPaths, get_palm_directory_paths
from .utils.dynamic_driver_utils import (
    DEFAULT_PT_SURFACE,
    apply_time_varying_inflow,
    disable_turbulent_inflow,
    is_time_varying_params,
    remove_dynamic_driver_file,
)
from .utils.inflow_utils import angle_to_velocity
from .utils.inlet_turbulence_utils import (
    apply_inlet_turbulence,
    is_inlet_turbulence_enabled,
    validate_inlet_turbulence,
)
from .utils.ncpu_utils import derive_npex_npey
from .utils.nudging_utils import apply_nudging_driver, remove_nudging_files
from .utils.p3d_utils import P3DFile
from .utils.vertical_profile import build_profile_shape
from .utils.warm_start_utils import write_warmstart_driver

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# How many missing 3D outputs may be padded by repeating the last frame.
# PALM's adaptive timestep can land one dump short of the requested count;
# more than that means the run stopped early (PALM exits 0 on divergence),
# and padding would fabricate the window instead of reporting the failure.
MAX_PADDED_OUTPUTS = 1


DomainBounds = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


DEFAULT_PARAMS = xarray.Dataset(
    data_vars={
        "inflow_angle": 0,
        "velocity_magnitude": 3.0,
    },
)


# Nudging-driver defaults. Read via ``.get(key, default)`` so a partial
# ``nudging_config`` (e.g. one carrying only ``profile_config``) still resolves
# every knob — a user-supplied dict *replaces* DEFAULT_NUDGING_CONFIG rather
# than merging with it, so the per-key defaults must live here too.
DEFAULT_NUDGING_ENABLED = True
DEFAULT_TNUDGE = 15.0  # s; matches pyudales
DEFAULT_NNUDGE_METERS = 4.0  # no nudging below this height (via huge tnudge)

DEFAULT_NUDGING_CONFIG: dict = {
    "enabled": DEFAULT_NUDGING_ENABLED,
    "tnudge": DEFAULT_TNUDGE,
    "nnudge_meters": DEFAULT_NNUDGE_METERS,
    "profile_config": {"type": "power_law", "alpha": 0.25},
}

# The four &initialization_parameters switches PALM's nudging path requires
# (LSF0001/0003/0005 constraint chain); toggled together.
_NUDGING_SWITCHES = ("nudging", "large_scale_forcing", "lsf_exception", "humidity")


def _augment_runtime_library_paths(env: dict[str, str]) -> None:
    """Prepend active pixi/conda lib dirs to LD_LIBRARY_PATH."""
    lib_paths: list[pathlib.Path] = []
    conda_prefix = env.get("CONDA_PREFIX")
    pixi_env = env.get("PIXI_ENVIRONMENT")
    for prefix in (conda_prefix, pixi_env):
        if not prefix:
            continue
        p = pathlib.Path(prefix)
        if not p.exists():
            continue
        lib_dir = p / "lib"
        if lib_dir.exists():
            lib_paths.append(lib_dir)

    if not lib_paths:
        return

    existing = env.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(str(p) for p in lib_paths)
    env["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


def _is_time_varying_params(params: Optional[xarray.Dataset]) -> bool:
    # Delegate to the utils helper which matches pyudales/pylbm's
    # per-variable check (strictly more permissive than a Dataset-level
    # ``"time" in params.dims``).
    return is_time_varying_params(params)


def _merge_params(
    base: xarray.Dataset, overlay: Optional[xarray.Dataset]
) -> xarray.Dataset:
    if overlay is None:
        return base
    merged = base.copy()
    for name, var in overlay.data_vars.items():
        merged[name] = var
    return merged


class ForwardModel(BaseForwardModel):
    """PALM LES ForwardModel.

    Mirrors the interface of pylbm/pyudales:
    - Cold start, or warm start from a handed-in ``state`` (the resolved 3D
      velocity field) via PALM's dynamic-driver ``init_atmosphere_*`` LOD=2
      mechanism — see ``run_single`` and :mod:`.utils.warm_start_utils`.
    - Inflow driven by geostrophic wind under cyclic BCs, or turbulent_inflow
      under dirichlet/radiation BCs (via the ``boundary_condition`` kwarg).
    - Static or time-varying params (``inflow_angle``, ``velocity_magnitude``).
    """

    def __init__(
        self,
        case_dir: pathlib.Path,
        stl_path: str | pathlib.Path,
        experiment_name: str = "urban_run",
        ncpu: int = 4,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        bounds: DomainBounds | None = None,
        simulation_time: float | None = None,
        output_frequency: Optional[float] = None,
        spinup_time: float = 0.0,
        boundary_condition: str = "periodic",
        sgs_constant: Optional[float] = None,
        nudging_config: Optional[dict] = None,
        inlet_turbulence: Optional[dict] = None,
        save_only_last_timestep: bool = False,
        results_dir: Optional[pathlib.Path] = None,
        experiment_base_dir: Optional[pathlib.Path] = None,
        temp_dir: Optional[pathlib.Path] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(results_dir=results_dir)

        if boundary_condition not in ("periodic", "inflow_outflow"):
            raise ValueError(
                f"boundary_condition must be 'periodic' or 'inflow_outflow', "
                f"got '{boundary_condition}'"
            )
        self.boundary_condition = boundary_condition
        # Per-backend default for the SGS knob, overridden by a `sgs_constant` in
        # the params Dataset when one is supplied. None leaves PALM's prognostic
        # SGS-TKE closure (and its surface flux layer) untouched — which is the
        # right default here, since PALM's proxy is `km_constant` in m^2/s, not a
        # dimensionless Smagorinsky constant. See `_apply_sgs_setting`.
        self.sgs_constant = float(sgs_constant) if sgs_constant is not None else None

        self.verbose = verbose
        self.stdout = None if verbose else subprocess.DEVNULL
        self.stderr = None if verbose else subprocess.DEVNULL

        self.experiment_name = experiment_name
        self.ncpu = ncpu
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.bounds = bounds
        self.simulation_time = simulation_time
        self.output_frequency = output_frequency
        self.spinup_time = spinup_time
        self.save_only_last_timestep = save_only_last_timestep

        self._nudging_config = nudging_config or DEFAULT_NUDGING_CONFIG
        # Inlet turbulence: absent (None) is a strict no-op, identical to
        # ``enabled: false``. See utils/inlet_turbulence_utils.py for why this
        # maps onto PALM's random inflow-disturbance machinery rather than onto
        # ``turbulent_inflow`` (which is our *time-varying inflow reader*, not a
        # turbulence generator).
        self._inlet_turbulence = inlet_turbulence
        validate_inlet_turbulence(inlet_turbulence, boundary_condition)

        self.dirs = get_palm_directory_paths(
            case_dir=pathlib.Path(case_dir),
            experiment_name=experiment_name,
            temp_dir=temp_dir,
            experiment_base_dir=experiment_base_dir,
            results_dir=results_dir,
        )

        self._stage_input_dir()

        self.params = _merge_params(DEFAULT_PARAMS, None)
        self._apply_runtime_and_domain()
        self._apply_boundary_condition()

        if nx is not None and ny is not None and bounds is not None:
            dz = (bounds[2][1] - bounds[2][0]) / nz if nz else 1.0
            self.topography = stl_to_palm_topography(
                stl_path=stl_path,
                dirs=self.dirs,
                nx=nx,
                ny=ny,
                bounds=bounds,
                dz=dz,
            )
            self._p3d_set_string(
                "initialization_parameters", "topography", "read_from_file"
            )
        else:
            logger.info(
                "nx/ny/bounds not fully specified; skipping topography generation."
            )
            self.topography = None

        logger.info("PALM experiment staged at %s", self.dirs.experiment_dir)

    def _stage_input_dir(self) -> None:
        """Copy PALM namelist/topography files from the case-dir into ``INPUT/``.

        Only files that are part of PALM's job-directory convention are
        copied (suffix ``_p3d``, ``_topo``, ``_static``, ``_dynamic``). The
        STL file lives alongside these in the case dir but is not a PALM
        input — it is rasterized separately into ``<name>_topo`` by
        ``stl_to_palm_topography``.
        """
        palm_suffixes = ("_p3d", "_topo", "_static", "_dynamic")
        src = self.dirs.case_dir
        if not src.exists():
            raise FileNotFoundError(f"PALM case_dir not found: {src}")
        for item in src.iterdir():
            if not item.is_file():
                continue
            name = item.name
            matched_suffix = None
            for s in palm_suffixes:
                if name == s or name.endswith(s):
                    matched_suffix = s
                    break
            if matched_suffix is None:
                continue
            out_name = f"{self.experiment_name}{matched_suffix}"
            shutil.copy2(item, self.dirs.input_dir / out_name)

    @property
    def p3d_path(self) -> pathlib.Path:
        return self.dirs.input_dir / f"{self.experiment_name}_p3d"

    def _p3d_set_value(self, section: str, key: str, value: str | float | int) -> None:
        p3d = P3DFile(self.p3d_path)
        p3d.set_value(section, key, value)
        p3d.write()

    def _p3d_set_string(self, section: str, key: str, value: str) -> None:
        p3d = P3DFile(self.p3d_path)
        p3d.set_string(section, key, value)
        p3d.write()

    def _p3d_set_array(self, section: str, key: str, values) -> None:
        p3d = P3DFile(self.p3d_path)
        p3d.set_array(section, key, values)
        p3d.write()

    def _apply_runtime_and_domain(self) -> None:
        if not self.p3d_path.exists():
            raise FileNotFoundError(
                f"Expected _p3d file at {self.p3d_path}. Check the case_dir template."
            )
        p3d = P3DFile(self.p3d_path)

        if self.nx is not None:
            p3d.set_value("initialization_parameters", "nx", int(self.nx) - 1)
        if self.ny is not None:
            p3d.set_value("initialization_parameters", "ny", int(self.ny) - 1)
        if self.nz is not None:
            p3d.set_value("initialization_parameters", "nz", int(self.nz))

        if self.bounds is not None:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
            if self.nx:
                p3d.set_value(
                    "initialization_parameters", "dx", (xmax - xmin) / self.nx
                )
            if self.ny:
                p3d.set_value(
                    "initialization_parameters", "dy", (ymax - ymin) / self.ny
                )
            if self.nz:
                p3d.set_value(
                    "initialization_parameters", "dz", (zmax - zmin) / self.nz
                )

        effective_runtime = (
            (self.simulation_time + self.spinup_time)
            if self.simulation_time is not None
            else None
        )
        if effective_runtime is not None:
            p3d.set_value("runtime_parameters", "end_time", float(effective_runtime))
        if self.output_frequency is not None:
            p3d.set_value(
                "runtime_parameters", "dt_data_output", float(self.output_frequency)
            )
            p3d.set_value(
                "runtime_parameters",
                "dt_data_output_av",
                float(self.output_frequency),
            )
            # PALM requires averaging_interval <= dt_data_output_av.
            p3d.set_value(
                "runtime_parameters",
                "averaging_interval",
                float(self.output_frequency),
            )

        p3d.write()

    def _apply_boundary_condition(self) -> None:
        p3d = P3DFile(self.p3d_path)
        if self.boundary_condition == "periodic":
            # poisfft (the default solver) needs an even number of grid points
            # along every cyclic direction. Fail here rather than letting PALM
            # abort mid-run with PAC0071/PAC0072.
            for axis, n in (("x", self.nx), ("y", self.ny)):
                if n is not None and int(n) % 2 == 1:
                    raise ValueError(
                        f"periodic PALM runs need an even number of grid points "
                        f"along {axis}; got domain.n{axis} = {int(n)}. PALM's "
                        f"poisfft pressure solver rejects an odd count on a "
                        f"cyclic boundary (PAC0071/PAC0072)."
                    )
            p3d.set_string("initialization_parameters", "bc_lr", "cyclic")
            p3d.set_string("initialization_parameters", "bc_ns", "cyclic")
            p3d.write()
        else:
            # Standard PALM urban inflow: non-cyclic east-west, cyclic north-south.
            # PALM forbids both pairs being dirichlet/radiation at once, and the
            # default poisfft solver requires matching BCs — so switch to the
            # multigrid pressure solver, which supports mixed BCs.
            p3d.set_string("initialization_parameters", "bc_lr", "dirichlet/radiation")
            p3d.set_string("initialization_parameters", "bc_ns", "cyclic")
            p3d.set_string("initialization_parameters", "psolver", "multigrid_noopt")
            p3d.write()
            # multigrid needs uniform subdomains, and PALM's auto-decomposition
            # puts the larger factor on npey — which usually fails to divide the
            # grid. Pin a slab decomposition (npex=ncpu, npey=1) derived from
            # NCPU so any ncpu that divides nx+1 works (see ncpu_utils).
            self._apply_processor_topology()

    def _apply_processor_topology(self) -> None:
        """Pin npex/npey in runtime_parameters from NCPU (slab decomposition).

        Only meaningful once the grid is known; with nx unset PALM falls back to
        its automatic decomposition.
        """
        if self.nx is None:
            return
        npex, npey = derive_npex_npey(int(self.ncpu), int(self.nx))
        p3d = P3DFile(self.p3d_path)
        p3d.set_value("runtime_parameters", "npex", npex)
        p3d.set_value("runtime_parameters", "npey", npey)
        p3d.write()
        logger.info("Pinned PALM processor topology: npex=%d, npey=%d", npex, npey)

    def set_results_dir(self, results_dir: pathlib.Path | None) -> None:
        super().set_results_dir(results_dir)
        self.dirs.results_dir = results_dir

    def compile(self, compile: bool = True) -> None:
        """Build PALM via ``palmbuild`` when ``compile`` is True.

        Hydra dispatches to this method via the ``model.prepare._target_``
        block in ``conf/model/pypalm.yaml``, which instantiates
        ``pyurbanair.config.hydra_helpers.prepare_compile``; this method
        exists to honour that contract.
        """
        if not compile:
            return
        compile_palm(verbose=self.verbose)

    @property
    def dynamic_driver_path(self) -> pathlib.Path:
        return self.dirs.input_dir / f"{self.experiment_name}_dynamic"

    @property
    def nudge_driver_path(self) -> pathlib.Path:
        return self.dirs.input_dir / f"{self.experiment_name}_nudge"

    @property
    def lsf_driver_path(self) -> pathlib.Path:
        return self.dirs.input_dir / f"{self.experiment_name}_lsf"

    def _apply_inflow_settings(self, params: xarray.Dataset) -> None:
        self.params = _merge_params(self.params, params)

        # Model-error knobs apply on both inflow branches, so resolve them up
        # front (docs/esmda_model_error_parameters.md §6.2). ``profile_config``
        # carries an α override from ``vertical_inflow_exponent`` when estimated.
        profile_config = self._resolve_profile_config(self.params)

        # Driver selection (see docs/plans/palm_nudging_driver_plan.md §Design):
        #   periodic  -> nudging driver (static OR time-varying), unless the
        #                escape hatch nudging_config.enabled=false restores
        #                today's un-driven periodic staging;
        #   inflow_outflow static       -> current static path;
        #   inflow_outflow time-varying -> current dynamic-driver path.
        # The inflow_outflow branches also run symmetric hygiene so a template
        # (or a prior periodic run in the same experiment_dir) cannot leak the
        # LSF/nudging apparatus into an inflow run.
        nudging_enabled = self._nudging_config.get("enabled", DEFAULT_NUDGING_ENABLED)
        use_nudging = self.boundary_condition == "periodic" and nudging_enabled

        if use_nudging:
            angle, speed = self._stage_nudging_driver(profile_config)
        elif _is_time_varying_params(self.params):
            if self.bounds is None or not self.nz or not self.ny:
                raise ValueError(
                    "Time-varying inflow requires bounds, nz, and ny to be set "
                    "on ForwardModel (needed to construct the dynamic driver)."
                )
            # Writes <case>_dynamic NetCDF and flips on turbulent_inflow in the
            # namelist. Returns a scalar params Dataset holding the t=0 values
            # so we still populate ug_surface/u_profile for initialisation.
            self._disable_nudging_apparatus()
            init_params = apply_time_varying_inflow(
                params=self.params,
                p3d_path=self.p3d_path,
                driver_path=self.dynamic_driver_path,
                bounds=self.bounds,
                nz=self.nz,
                ny=self.ny,
                profile_config=profile_config,
                spinup_time=self.spinup_time,
            )
            angle = float(init_params["inflow_angle"].item())
            speed = float(init_params["velocity_magnitude"].item())
        else:
            # Static path — make sure a stale dynamic driver from a prior
            # time-varying run, and any nudging apparatus from a prior periodic
            # run, in the same experiment_dir don't leak in.
            disable_turbulent_inflow(self.p3d_path)
            remove_dynamic_driver_file(self.dynamic_driver_path)
            self._disable_nudging_apparatus()
            angle = float(self.params["inflow_angle"].item())
            speed = float(self.params["velocity_magnitude"].item())

        u0, v0 = angle_to_velocity(angle, speed)

        p3d = P3DFile(self.p3d_path)
        p3d.set_value("initialization_parameters", "ug_surface", float(u0))
        p3d.set_value("initialization_parameters", "vg_surface", float(v0))
        self._apply_sgs_setting(p3d, self.params)

        if self.bounds is not None and self.nz:
            zmin, zmax = self.bounds[2]
            dz = (zmax - zmin) / self.nz
            cell_heights = np.arange(self.nz) * dz + 0.5 * dz + zmin
            shape = build_profile_shape(
                profile_config,
                heights=cell_heights,
                zsize=zmax - zmin,
            )
            # PALM requires u_profile(1) = v_profile(1) = 0 at the surface
            # (no-slip). Prepend a z=0 anchor to the profile.
            heights = np.concatenate(([0.0], cell_heights))
            u_profile = np.concatenate(([0.0], shape * float(u0)))
            v_profile = np.concatenate(([0.0], shape * float(v0)))
            p3d.set_array("initialization_parameters", "u_profile", u_profile.tolist())
            p3d.set_array("initialization_parameters", "v_profile", v_profile.tolist())
            p3d.set_array("initialization_parameters", "uv_heights", heights.tolist())

        p3d.write()

    def _stage_nudging_driver(
        self, profile_config: Optional[dict]
    ) -> tuple[float, float]:
        """Stage the periodic nudging driver and return the t=0 (angle, speed).

        Writes ``<name>_nudge`` (NUDGING_DATA) + inert ``<name>_lsf`` (LSF_DATA),
        turns on the four ``&initialization_parameters`` switches PALM's nudging
        path requires, and clears the inflow_outflow apparatus (turbulent_inflow
        + dynamic driver) so a prior inflow run can't leak into a periodic one.
        """
        if self.bounds is None or not self.nz:
            raise ValueError(
                "Periodic nudging driver requires bounds and nz to be set on "
                "ForwardModel (needed for the NUDGING_DATA height column). "
                "Set them, or disable the driver with nudging_config.enabled=false."
            )
        self._validate_no_passive_scalar()
        init_params = apply_nudging_driver(
            params=self.params,
            nudge_path=self.nudge_driver_path,
            lsf_path=self.lsf_driver_path,
            bounds=self.bounds,
            nz=self.nz,
            profile_config=profile_config,
            tnudge=self._nudging_config.get("tnudge", DEFAULT_TNUDGE),
            nnudge_meters=self._nudging_config.get(
                "nnudge_meters", DEFAULT_NNUDGE_METERS
            ),
            spinup_time=self.spinup_time,
            simulation_time=self.simulation_time,
        )
        self._set_nudging_switches(True)
        # The nudging driver drives the flow instead of turbulent_inflow; make
        # sure neither the turbulent_inflow block nor a stale dynamic driver is
        # active. (Runs BEFORE _apply_warmstart, so the removed dynamic driver is
        # re-created for a warm start — see run_single's ordering note.)
        disable_turbulent_inflow(self.p3d_path)
        remove_dynamic_driver_file(self.dynamic_driver_path)
        return (
            float(init_params["inflow_angle"].item()),
            float(init_params["velocity_magnitude"].item()),
        )

    def _set_nudging_switches(self, on: bool) -> None:
        """Set the four nudging ``&initialization_parameters`` switches together."""
        p3d = P3DFile(self.p3d_path)
        for key in _NUDGING_SWITCHES:
            p3d.set_value("initialization_parameters", key, bool(on))
        p3d.write()

    def _disable_nudging_apparatus(self) -> None:
        """Remove staged nudging files and force the switches off — but only
        touch namelist keys that are actually present.

        Keeping this a no-op on a clean template (keys absent → left absent,
        PALM defaults them to ``.false.``) preserves the inflow_outflow staging
        byte-for-byte apart from the file cleanup. It only bites when a prior
        periodic run in the same experiment_dir set the switches ``.true.``.
        """
        remove_nudging_files(self.nudge_driver_path, self.lsf_driver_path)
        p3d = P3DFile(self.p3d_path)
        present = [
            key
            for key in _NUDGING_SWITCHES
            if key in p3d.sections.get("initialization_parameters", {})
        ]
        if not present:
            return
        for key in present:
            p3d.set_value("initialization_parameters", key, False)
        p3d.write()

    def _validate_no_passive_scalar(self) -> None:
        """Raise if the template enables ``passive_scalar`` under the nudging driver.

        PALM's ``large_scale_forcing`` is incompatible with ``passive_scalar``
        (LSF0004, fatal). Fail at staging time with an actionable message rather
        than letting PALM abort mid-run.
        """
        p3d = P3DFile(self.p3d_path)
        val = p3d.get_value("initialization_parameters", "passive_scalar")
        if val is not None and val.strip().lower() in (".t.", ".true.", "t", "true"):
            raise ValueError(
                "PALM's nudging driver (large_scale_forcing) is incompatible "
                "with passive_scalar = .T. (LSF0004). Run this case under "
                "boundary_condition='inflow_outflow', or disable the nudging "
                "driver with nudging_config.enabled=false."
            )

    @staticmethod
    def _param_value(params: Optional[xarray.Dataset], name: str) -> Optional[float]:
        """Scalar value of ``name`` in ``params``, or ``None`` when absent."""
        if params is None or name not in params:
            return None
        return float(params[name].item())

    def _resolve_profile_config(
        self, params: Optional[xarray.Dataset]
    ) -> Optional[dict]:
        """Return a ``profile_config`` with ``alpha`` overridden from ``params``.

        ``vertical_inflow_exponent`` overrides the power-law ``alpha`` so the
        inlet shear is per-member and ESMDA-estimable
        (docs/esmda_model_error_parameters.md §2.1). Falls back to the
        construction-time profile config when the parameter is absent.
        """
        base = self._nudging_config.get("profile_config")
        alpha = self._param_value(params, "vertical_inflow_exponent")
        if alpha is None:
            return base
        profile_config = dict(base or {})
        profile_config.setdefault("type", "power_law")
        profile_config["alpha"] = alpha
        return profile_config

    def _apply_sgs_setting(
        self, p3d: P3DFile, params: Optional[xarray.Dataset]
    ) -> None:
        """Write the per-member SGS knob via the ``km_constant`` proxy (Option A).

        PALM's LES TKE closure has no Smagorinsky-style namelist multiplier
        (``c_0`` is hardcoded), so ``sgs_constant`` is mapped to ``km_constant`` —
        a constant eddy diffusivity [m²/s]. This replaces the prognostic SGS-TKE
        closure with a constant-Km model: a different turbulence regime, accepted
        purely as a bias-absorbing knob. The PALM ``sgs_constant`` therefore is
        NOT the same quantity as the LBM/uDALES Smagorinsky constants
        (docs/esmda_model_error_parameters.md §2.3, §8). No-op when absent.

        PALM forbids a fixed ``km`` together with a Monin-Obukhov surface flux
        layer (check_parameters PAC0149), so a fixed ``km_constant`` also requires
        ``constant_flux_layer = .false.`` — part of the same constant-Km regime
        switch. No-op when ``sgs_constant`` is absent, leaving PALM's prognostic
        SGS-TKE closure and default surface flux layer untouched.
        """
        # Precedence: an estimated/sampled `sgs_constant` in ``params`` wins; the
        # model config's ``sgs_constant`` is the per-backend fallback; absent in
        # both leaves PALM's prognostic SGS-TKE closure untouched.
        sgs = self._param_value(params, "sgs_constant")
        source = "params"
        if sgs is None:
            sgs = self.sgs_constant
            source = "model config"
        if sgs is None:
            return
        p3d.set_value("initialization_parameters", "km_constant", float(sgs))
        # Required by PALM whenever km is fixed (PAC0149).
        p3d.set_value("initialization_parameters", "constant_flux_layer", False)
        logger.info(
            "Set PALM km_constant (sgs_constant proxy, Option A) to %.4f m^2/s "
            "(from %s)",
            float(sgs),
            source,
        )

    def save_results(self, state: xarray.Dataset, sim_name: str = "state") -> None:
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        clean_palm_output_dir(self.dirs)

    def _ensure_palm_config_in_cwd(self) -> None:
        """palmrun reads ``.palm.config.<id>`` from its working directory.

        We write a patched copy into ``experiment_dir`` (per-member) and point
        palmrun's data paths at the staged inputs/outputs. Each ensemble
        member gets its own tmp/ so parallel palmrun invocations do not
        collide on ``fast_io_catalog`` or CWD-local state.

        Layout: palmrun runs from ``experiment_dir`` with ``$base_data`` set
        to ``experiment_dir.parent`` (= ``experiment_base_dir``). palmrun
        then resolves ``$base_data/$run_identifier/INPUT`` to
        ``experiment_dir/INPUT``.

        ``fast_io_catalog`` is palmrun's per-run working directory: it copies
        the full build tree (~750 files: sources + prebuilt objects + the
        executable) there and runs PALM in it. PALM intends this to be a *fast
        local* filesystem; on networked scratch (beegfs) that per-member,
        per-window copy is slow for many small files. If
        ``PYPALM_FAST_IO_CATALOG`` is set (point it at node-local /tmp on a
        cluster), each member's working dir is isolated under it; otherwise we
        fall back to the per-member scratch ``tmp/`` (unchanged behaviour).
        """
        canonical = PALM_MODEL_SYSTEM_PATH / ".palm.config.default"
        if not canonical.exists():
            return
        base = str(self.dirs.experiment_dir.parent)
        fast_io_base = os.environ.get("PYPALM_FAST_IO_CATALOG", "").strip()
        if fast_io_base:
            tmp = str(pathlib.Path(fast_io_base) / self.experiment_name)
        else:
            tmp = str(self.dirs.experiment_dir / "tmp")
        os.makedirs(tmp, exist_ok=True)
        overrides = {
            "%base_data": base,
            "%user_source_path": f"{base}/$run_identifier/USER_CODE",
            "%fast_io_catalog": tmp,
            "%restart_data_path": tmp,
            "%output_data_path": base,
            "%local_jobcatalog": f"{base}/$run_identifier/LOG_FILES",
        }
        out_lines: list[str] = []
        for line in canonical.read_text().splitlines():
            replaced = False
            for key, new_val in overrides.items():
                if line.startswith(key):
                    out_lines.append(f"{key:21s}{new_val}")
                    replaced = True
                    break
            if not replaced:
                out_lines.append(line)
        (self.dirs.experiment_dir / ".palm.config.default").write_text(
            "\n".join(out_lines) + "\n"
        )

    def run(self) -> None:
        """Invoke PALM.

        Two paths:
          - ``direct_palm.run_direct`` (default) — bypasses palmrun + palmbuild;
            ~16x faster on tiny (see docs/palm_overhead_plan.md). It also runs
            ``combine_plot_fields.x`` itself with the ``rrtmg.so`` symlinks it
            needs, so the merged 3D netCDF is actually produced. The slurm
            scripts already default to this; M4 flips it for local runs too.
          - ``PYPALM_USE_DIRECT_RUN=0`` -> palmrun via the execute.sh wrapper
            (the historical fallback). NOTE: on macOS palmrun's combine step
            fails to load ``rrtmg.so`` and silently yields an all-zero field
            (caught by ``_assert_combine_succeeded``); prefer the default path.
        """
        if os.environ.get("PYPALM_USE_DIRECT_RUN", "1") != "0":
            self._run_direct()
            return

        if PALMRUN_BIN is None and not shutil.which("palmrun"):
            raise RuntimeError(
                "palmrun not found. Install palm_model_system and either:\n"
                "  - add palmrun to PATH, or\n"
                "  - set PALM_BIN to the palmrun executable, or\n"
                "  - set PALM_ROOT (palmrun is expected at $PALM_ROOT/bin/palmrun).\n"
                "See https://palm.muk.uni-hannover.de for installation."
            )
        self._ensure_palm_config_in_cwd()
        logger.info("Running PALM …")
        # Run palmrun from experiment_dir (per-member) so parallel ensemble
        # members don't share a CWD / .palm.config / tmp catalog.
        command = [
            "bash",
            str(LOCAL_EXECUTE_SCRIPT),
            str(self.dirs.experiment_dir),
            self.experiment_name,
            str(self.ncpu),
        ]
        env = os.environ.copy()
        _augment_runtime_library_paths(env)
        if PALMRUN_BIN is not None:
            bin_dir = str(PALMRUN_BIN.parent)
            env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

        # palmrun prompts interactively (">>> everything o.k. (y/n) ?") unless
        # it thinks it's in batch mode. With a blocking stdin this hangs forever
        # — which is exactly what happens for ensemble members run inside
        # forkserver pool workers (the serial truth run survives only because
        # the main process inherits sbatch's /dev/null stdin). Force stdin to
        # /dev/null so palmrun's `read` always hits EOF and proceeds.
        if self.verbose:
            subprocess.run(command, check=True, env=env, stdin=subprocess.DEVNULL)
            return

        # When not verbose, capture output so we can surface PALM's error
        # message on failure instead of leaving the user with just an exit code.
        _t0 = time.monotonic()
        result = subprocess.run(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logger.info(
            "palmrun(%s) wall=%.1fs rc=%s",
            self.experiment_name,
            time.monotonic() - _t0,
            result.returncode,
        )
        if result.returncode != 0:
            tail = "\n".join((result.stdout or "").splitlines()[-80:])
            logger.error(
                "palmrun failed (exit %s). Last lines of captured output:\n%s",
                result.returncode,
                tail,
            )
            raise subprocess.CalledProcessError(
                result.returncode, command, output=result.stdout
            )

    def _run_direct(self) -> None:
        """PYPALM_USE_DIRECT_RUN=1 branch — bypass palmrun + palmbuild.

        Stages an isolated tempdir from ``self.dirs.input_dir``, runs the
        prebuilt ``palm`` + ``combine_plot_fields.x`` (no mpirun on combine),
        and transfers ``DATA_3D_NETCDF`` to ``self.dirs.output_dir``. See
        ``pypalm.direct_palm`` for the staging contract and
        ``docs/palm_overhead_plan.md`` §M0/§M1 for the per-phase numbers.
        """
        # Import inside the method so non-direct runs don't pay the import
        # cost and so the existing palmrun path doesn't depend on the new module.
        from .direct_palm import run_direct

        env = os.environ.copy()
        _augment_runtime_library_paths(env)

        logger.info("Running PALM (direct, no palmrun) …")
        _t0 = time.monotonic()
        result = run_direct(
            dirs=self.dirs,
            experiment_name=self.experiment_name,
            ncpu=self.ncpu,
            host="default",
            env=env,
            keep_tempdir=False,
            verbose=self.verbose,
        )
        logger.info(
            "palm_direct(%s) wall=%.1fs (stage=%.2fs palm=%.2fs combine=%.2fs transfer=%.2fs) rc=%s",
            self.experiment_name,
            time.monotonic() - _t0,
            result.stage_s,
            result.palm_s,
            result.combine_s,
            result.transfer_s,
            result.palm_rc,
        )

    def _fit_output_window(self, state: xarray.Dataset) -> xarray.Dataset:
        """Trim/pad the 3D output onto the expected window.

        Drops the spin-up frames, trims a surplus from the front, and pads at
        most :data:`MAX_PADDED_OUTPUTS` missing frames. A larger shortfall
        raises ``CalledProcessError`` — see the comment inline.
        """
        if self.spinup_time > 0 and self.output_frequency:
            spinup_outputs = int(self.spinup_time / self.output_frequency)
            if state.sizes.get("time", 0) > spinup_outputs:
                state = state.isel(time=slice(spinup_outputs, None))

        if (
            self.simulation_time is not None
            and self.output_frequency
            and state.sizes.get("time", 0) > 0
        ):
            expected_outputs = int(self.simulation_time / self.output_frequency)
            actual = state.sizes["time"]
            if actual > expected_outputs:
                state = state.isel(time=slice(-expected_outputs, None))
            elif actual < expected_outputs:
                missing = expected_outputs - actual
                # PALM's timestep is adaptive, so the 3D file occasionally has
                # ONE fewer output than requested. Padding that single frame
                # keeps ensemble members concat-able along `time`.
                #
                # Anything more means PALM stopped early -- and PALM signals its
                # own numerical divergence by terminating with exit 0 (see
                # `_locate_3d_output`), so a short 3D file is the only evidence
                # left. Padding it would fabricate most of the window from a
                # repeated frame and report success, which reads downstream as a
                # flow that "runs a few steps and then stays constant". Raise
                # instead, matching the divergence path so the ensemble's
                # resample-from-successes policy replaces the member.
                if missing > MAX_PADDED_OUTPUTS:
                    msg = (
                        f"PALM produced only {actual} of {expected_outputs} "
                        f"expected 3D outputs ({missing} missing). PALM "
                        "terminates with exit 0 on its own numerical "
                        "divergence, so this most likely means the run "
                        "diverged and stopped early. Refusing to pad "
                        f"{missing} frames by repeating the last one."
                    )
                    logger.error(msg)
                    raise subprocess.CalledProcessError(
                        1, f"palm ({self.experiment_name})", output=msg
                    )
                logger.warning(
                    "PALM wrote %d of %d expected 3D outputs; padding %d frame(s) "
                    "by repeating the last (adaptive-timestep rounding).",
                    actual,
                    expected_outputs,
                    missing,
                )
                last = state.isel(time=-1)
                pads = [last.expand_dims(time=1) for _ in range(missing)]
                state = xarray.concat([state, *pads], dim="time")

        return state

    def _locate_3d_output(self) -> pathlib.Path:
        """Find the ``<name>_3d.nc`` output file palmrun wrote."""
        primary = self.dirs.output_dir / f"{self.experiment_name}_3d.nc"
        if primary.exists():
            return primary

        alternates = sorted(self.dirs.output_dir.glob(f"{self.experiment_name}_3d*.nc"))
        if alternates:
            return alternates[0]

        # PALM detects its own numerical divergence and terminates (exit 0)
        # before reaching the first ``dt_data_output`` dump, so it leaves the
        # timeseries but no 3D field. Report that as a ``CalledProcessError`` --
        # the same signal the ensemble layer's resample-from-successes policy
        # already catches for crashed members -- so a single diverged member is
        # replaced from a successful donor instead of aborting the whole run.
        msg = (
            f"No PALM 3D output found in {self.dirs.output_dir} (expected "
            f"{self.experiment_name}_3d.nc); the run most likely diverged and "
            "terminated before the first 3D output dump."
        )
        logger.error(msg)
        # Pass returncode/cmd positionally: CalledProcessError only records
        # *positional* constructor args in ``self.args``, and the ProcessPool
        # pickles the exception back to the parent via ``self.args``. With
        # keyword args ``self.args`` is empty and unpickling raises a TypeError
        # ("missing returncode and cmd"), breaking the pool instead of being
        # caught as a member failure.
        raise subprocess.CalledProcessError(
            1, f"palm ({self.experiment_name})", output=msg
        )

    @staticmethod
    def _assert_combine_succeeded(
        state: xarray.Dataset, output_file: pathlib.Path
    ) -> None:
        """Fail loudly when PALM wrote a zero-filled skeleton instead of data.

        For a ``__parallel`` PALM build with ``netcdf_data_format < 5`` the
        velocity fields are streamed to a Fortran binary file and only merged
        into the per-PE ``_3d.NNN.nc`` netCDF by ``combine_plot_fields.x``.
        When that post-processing step is skipped or crashes (e.g. the macOS
        dyld ``rrtmg.so`` load failure — see docs/pypalm_zero_field_debug.md),
        the netCDF still opens cleanly but every u/v/w cell is exactly 0 with
        **no** topography fill values. A correct PALM field always carries
        NaN/fill at solid cells, so "finite everywhere AND identically zero"
        is an unambiguous sentinel for a missing combine — surface it as an
        error rather than silently returning a dead field downstream.
        """
        for var in ("u", "v", "w"):
            if var not in state.data_vars:
                continue
            values = state[var].values
            if values.size == 0:
                continue
            if np.all(np.isfinite(values)) and not np.any(values):
                raise RuntimeError(
                    f"PALM 3D output {output_file} has {var} identically zero "
                    "with no topography fill values — combine_plot_fields almost "
                    "certainly did not run (the per-PE netCDF skeleton was read "
                    "instead of the merged field). See "
                    "docs/pypalm_zero_field_debug.md."
                )

    def _load_and_postprocess_state(self) -> xarray.Dataset:
        output_file = self._locate_3d_output()
        state = xarray.open_dataset(
            output_file, engine="netcdf4", decode_timedelta=False
        )

        self._assert_combine_succeeded(state, output_file)

        # PALM uses staggered vertical coords: u/v/scalars on zu_3d,
        # w on zw_3d. Preserve both; rename zu_3d -> z (the "canonical" z
        # used by u/v) and zw_3d -> zw so w keeps its own staggered axis.
        rename_map = {}
        if "zu_3d" in state.dims:
            rename_map["zu_3d"] = "z"
        if "zw_3d" in state.dims:
            rename_map["zw_3d"] = "zw"
        if "zs_3d" in state.dims and "zs_3d" != "z":
            rename_map["zs_3d"] = "zs"
        if rename_map:
            state = state.rename(rename_map)

        # PALM emits grid coords based at 0 (the topography is sampled at the
        # physical x_centers = xmin + ..., but PALM's NetCDF axes start at the
        # origin). Shift them onto the configured physical domain so x/y/z match
        # `bounds` — i.e. the same convention pyudales (coords + offset) and
        # pylbm (xmin + (i+0.5)*dx) already use, and therefore the shared obs
        # configs. Without this, sensors in the upstream inflow region (x < 0)
        # fall outside PALM's [0, xlen] grid and the observation operator raises
        # "Observation points for axis 'xu' are outside the grid bounds".
        if self.bounds is not None:
            (xmin, _), (ymin, _), (zmin, _) = self.bounds
            offsets = {
                "x": xmin,
                "xu": xmin,
                "y": ymin,
                "yv": ymin,
                "z": zmin,
                "zw": zmin,
                "zs": zmin,
            }
            coord_updates = {
                name: state.coords[name].values + off
                for name, off in offsets.items()
                if name in state.coords
            }
            if coord_updates:
                state = state.assign_coords(coord_updates)

        # PALM writes NaN for any cell occluded by topography (wall layer
        # zu_3d[0]/zw_3d[0] = 0, building interiors, etc.). The physical
        # BC is no-slip, so replace NaN with 0 across u/v/w. Without this,
        # observation operators that sample near-ground or inside-building
        # points produce NaN pred_obs and poison the Kalman update.
        for var in ("u", "v", "w"):
            if var in state.data_vars:
                state[var] = state[var].fillna(0.0)

        if "z" in state.dims and state.sizes["z"] > 1:
            state = state.isel(z=slice(1, None))

        # Unify onto a single z axis so u/v/w have matching shapes for
        # downstream viz and aggregation. Interpolate w from zw onto z
        # (linear, with extrapolation) and drop the zw dim.
        if "zw" in state.dims and "w" in state.data_vars and "z" in state.dims:
            w_on_z = state["w"].interp(
                zw=state["z"].values, kwargs={"fill_value": "extrapolate"}
            )
            w_on_z = w_on_z.rename({"zw": "z"}).assign_coords(z=state["z"].values)
            state = state.drop_vars("w").assign(w=w_on_z).drop_dims("zw")

        state = self._fit_output_window(state)

        # Store the time coordinate in seconds (0, dt, 2·dt, …) rather than bare
        # frame indices, matching pylbm/pyudales. Without this the rollout
        # window-concat in run_forward_model.py (which re-bases each window by
        # ``w * simulation_time``) and the temporal observation binning (which
        # bins by the ``time`` coordinate in seconds) see an axis spaced by 1
        # instead of ``output_frequency`` — a wrong, backend-specific clock.
        if state.sizes.get("time", 0) > 0:
            n = state.sizes["time"]
            step = float(self.output_frequency) if self.output_frequency else 1.0
            state = state.assign_coords(time=np.arange(n, dtype=float) * step)

        return state

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:
        """Run PALM, optionally warm-started from ``state``.

        ``state is None`` → cold start (with spinup): PALM initializes from
        analytic profiles (``initializing_actions='set_constant_profiles'``).

        ``state`` provided → warm start (no spinup): the resolved u/v/w field is
        injected as PALM's full 3D initial condition via the dynamic-driver
        ``init_atmosphere_*`` LOD=2 mechanism and ``initializing_actions`` is
        switched to ``'read_from_file'`` (see :mod:`.utils.warm_start_utils`).
        Warm-start carries no subgrid state — PALM re-derives SGS-TKE — so the
        run is stateless and needs no per-member persistence.
        """
        warm_start = state is not None
        if warm_start:
            # No spinup on warm windows; also stops the time-varying inflow path
            # from prepending a spinup plateau (it reads ``self.spinup_time``).
            self.disable_spinup()

        if params is not None:
            self._apply_inflow_settings(params)
        else:
            self._apply_inflow_settings(self.params)

        if warm_start:
            # Must run AFTER _apply_inflow_settings: the static path removes any
            # stale driver, and the time-varying path writes inflow_plane_* into
            # the driver we then augment with init_atmosphere_*.
            self._apply_warmstart(state)
        else:
            self._reset_cold_init()

        # Must run AFTER the warm/cold init block: both write
        # ``create_disturbances``, which also gates the in-run inflow
        # perturbations. The explicit knob takes precedence over that implicit
        # write (and suppresses only the *initial* kick on a warm start).
        self._apply_inlet_turbulence(warm_start=warm_start)

        self._clean_output()
        self.run()
        return self._load_and_postprocess_state()

    def _apply_warmstart(self, state: xarray.Dataset) -> None:
        """Inject ``state`` as PALM's 3D initial condition for a warm start."""
        if self.bounds is None or not self.nx or not self.ny or not self.nz:
            raise ValueError(
                "pypalm warm-start requires bounds, nx, ny, and nz to be set on "
                "ForwardModel (needed to build the init_atmosphere_* fields)."
            )
        write_warmstart_driver(
            driver_path=self.dynamic_driver_path,
            state=state,
            bounds=self.bounds,
            nx=int(self.nx),
            ny=int(self.ny),
            nz=int(self.nz),
            pt_surface=DEFAULT_PT_SURFACE,
        )
        self._p3d_set_string(
            "initialization_parameters", "initializing_actions", "read_from_file"
        )
        # Suppress the initial random velocity perturbation PALM otherwise adds
        # at init (init_3d_model.f90:1488, gated by create_disturbances +
        # disturbance_energy_limit). On a cold start that kick seeds turbulence,
        # but our injected field is already turbulent, so re-disturbing it just
        # shocks the flow at every window boundary. PALM's own restart path
        # (read_restart_data) skips this block entirely; mirror that. The
        # divergence-removing pressure solve still runs (it is also gated on
        # non-flat topography), so the injected field is still made solenoidal.
        self._p3d_set_value("runtime_parameters", "create_disturbances", False)

    def _apply_inlet_turbulence(self, warm_start: bool = False) -> bool:
        """Stage (or clear) the switchable inlet-turbulence knob.

        Delegates to :func:`.utils.inlet_turbulence_utils.apply_inlet_turbulence`.
        Kept as a thin method so staging tests can drive it without running PALM.

        Precedence, in one place:

        1. ``inlet_turbulence`` absent / ``enabled: false`` → nothing is written
           on a clean template. Today's behaviour is preserved exactly,
           *including* the implicit "time-varying params on the inflow_outflow
           path switch ``turbulent_inflow`` on" coupling — that coupling is the
           dynamic-driver reader, not a turbulence generator, and disabling it
           would sever the ESMDA-estimated inflow signal from the solver.
        2. ``enabled: true`` → PALM's random inflow disturbances are turned on
           on top of whatever inflow driver the params selected. The two are
           orthogonal in PALM, so this composes with both the static and the
           time-varying inflow paths.
        """
        return apply_inlet_turbulence(
            self.p3d_path, self._inlet_turbulence, warm_start=warm_start
        )

    @property
    def inlet_turbulence_enabled(self) -> bool:
        """Whether the inlet-turbulence knob is on for this model."""
        return is_inlet_turbulence_enabled(self._inlet_turbulence)

    def _reset_cold_init(self) -> None:
        """Restore the cold-start initialization mode.

        A prior warm-start run in this experiment_dir leaves
        ``initializing_actions = 'read_from_file'`` (and disturbances disabled)
        in the namelist; reset both so a subsequent cold start initializes from
        analytic profiles and re-enables the turbulence-seeding perturbation.
        """
        self._p3d_set_string(
            "initialization_parameters",
            "initializing_actions",
            "set_constant_profiles",
        )
        self._p3d_set_value("runtime_parameters", "create_disturbances", True)

    def disable_spinup(self) -> None:
        self.spinup_time = 0.0
        if self.simulation_time is not None:
            self._p3d_set_value(
                "runtime_parameters", "end_time", float(self.simulation_time)
            )
