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
    apply_time_varying_inflow,
    disable_turbulent_inflow,
    is_time_varying_params,
    remove_dynamic_driver_file,
)
from .utils.inflow_utils import angle_to_velocity
from .utils.p3d_utils import P3DFile
from .utils.vertical_profile import build_profile_shape

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


DEFAULT_NUDGING_CONFIG: dict = {
    "profile_config": {"type": "power_law", "alpha": 0.25},
}


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

    Mirrors the interface of pylbm/pyudales. Scope for v1:
    - Cold start only (``state`` argument is ignored with a warning).
    - Inflow driven by geostrophic wind under cyclic BCs, or turbulent_inflow
      under dirichlet/radiation BCs (via the ``boundary_condition`` kwarg).
    - Time-invariant params only (``inflow_angle``, ``velocity_magnitude``);
      time-varying params raise NotImplementedError.
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
        nudging_config: Optional[dict] = None,
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
            self._p3d_set_string("initialization_parameters", "topography", "read_from_file")
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
                p3d.set_value("initialization_parameters", "dx", (xmax - xmin) / self.nx)
            if self.ny:
                p3d.set_value("initialization_parameters", "dy", (ymax - ymin) / self.ny)
            if self.nz:
                p3d.set_value("initialization_parameters", "dz", (zmax - zmin) / self.nz)

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
            p3d.set_string("initialization_parameters", "bc_lr", "cyclic")
            p3d.set_string("initialization_parameters", "bc_ns", "cyclic")
        else:
            # Standard PALM urban inflow: non-cyclic east-west, cyclic north-south.
            # PALM forbids both pairs being dirichlet/radiation at once, and the
            # default poisfft solver requires matching BCs — so switch to the
            # multigrid pressure solver, which supports mixed BCs.
            p3d.set_string("initialization_parameters", "bc_lr", "dirichlet/radiation")
            p3d.set_string("initialization_parameters", "bc_ns", "cyclic")
            p3d.set_string("initialization_parameters", "psolver", "multigrid_noopt")
        p3d.write()

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

    def _apply_inflow_settings(self, params: xarray.Dataset) -> None:
        self.params = _merge_params(self.params, params)

        if _is_time_varying_params(self.params):
            if self.bounds is None or not self.nz or not self.ny:
                raise ValueError(
                    "Time-varying inflow requires bounds, nz, and ny to be set "
                    "on ForwardModel (needed to construct the dynamic driver)."
                )
            # Writes <case>_dynamic NetCDF and flips on turbulent_inflow in the
            # namelist. Returns a scalar params Dataset holding the t=0 values
            # so we still populate ug_surface/u_profile for initialisation.
            init_params = apply_time_varying_inflow(
                params=self.params,
                p3d_path=self.p3d_path,
                driver_path=self.dynamic_driver_path,
                bounds=self.bounds,
                nz=self.nz,
                ny=self.ny,
                profile_config=self._nudging_config.get("profile_config"),
                spinup_time=self.spinup_time,
            )
            angle = float(init_params["inflow_angle"].item())
            speed = float(init_params["velocity_magnitude"].item())
        else:
            # Static path — make sure a stale dynamic driver from a prior
            # time-varying run in the same experiment_dir doesn't leak in.
            disable_turbulent_inflow(self.p3d_path)
            remove_dynamic_driver_file(self.dynamic_driver_path)
            angle = float(self.params["inflow_angle"].item())
            speed = float(self.params["velocity_magnitude"].item())

        u0, v0 = angle_to_velocity(angle, speed)

        p3d = P3DFile(self.p3d_path)
        p3d.set_value("initialization_parameters", "ug_surface", float(u0))
        p3d.set_value("initialization_parameters", "vg_surface", float(v0))

        if self.bounds is not None and self.nz:
            zmin, zmax = self.bounds[2]
            dz = (zmax - zmin) / self.nz
            cell_heights = np.arange(self.nz) * dz + 0.5 * dz + zmin
            shape = build_profile_shape(
                self._nudging_config.get("profile_config"),
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
          - ``PYPALM_USE_DIRECT_RUN=1`` -> ``direct_palm.run_direct`` (bypasses
            palmrun + palmbuild; ~16x faster on tiny — see docs/palm_overhead_plan.md).
          - otherwise -> palmrun via the execute.sh wrapper (the historical path,
            kept as a fallback until M4 flips the default).
        """
        if os.environ.get("PYPALM_USE_DIRECT_RUN") == "1":
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

    def _locate_3d_output(self) -> pathlib.Path:
        """Find the ``<name>_3d.nc`` output file palmrun wrote."""
        primary = self.dirs.output_dir / f"{self.experiment_name}_3d.nc"
        if primary.exists():
            return primary

        alternates = sorted(self.dirs.output_dir.glob(f"{self.experiment_name}_3d*.nc"))
        if alternates:
            return alternates[0]

        raise FileNotFoundError(
            f"No PALM 3D output found in {self.dirs.output_dir} (expected "
            f"{self.experiment_name}_3d.nc)."
        )

    def _load_and_postprocess_state(self) -> xarray.Dataset:
        output_file = self._locate_3d_output()
        state = xarray.open_dataset(
            output_file, engine="netcdf4", decode_timedelta=False
        )

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
                "x": xmin, "xu": xmin,
                "y": ymin, "yv": ymin,
                "z": zmin, "zw": zmin, "zs": zmin,
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
                # PALM's timestep is adaptive, so the 3D file occasionally
                # has one fewer output than requested. Pad by repeating the
                # last frame so all ensemble members concat along `time`.
                last = state.isel(time=-1)
                pads = [last.expand_dims(time=1) for _ in range(expected_outputs - actual)]
                state = xarray.concat([state, *pads], dim="time")

        if state.sizes.get("time", 0) > 0:
            state = state.assign_coords(time=range(state.sizes["time"]))

        return state

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:
        if state is not None:
            logger.warning(
                "pypalm v1 does not support warm-start; ignoring provided state."
            )

        if params is not None:
            self._apply_inflow_settings(params)
        else:
            self._apply_inflow_settings(self.params)

        self._clean_output()
        self.run()
        return self._load_and_postprocess_state()

    def disable_spinup(self) -> None:
        self.spinup_time = 0.0
        if self.simulation_time is not None:
            self._p3d_set_value(
                "runtime_parameters", "end_time", float(self.simulation_time)
            )
