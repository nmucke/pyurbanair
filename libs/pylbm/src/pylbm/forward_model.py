import logging
import os
import pathlib
import re
import subprocess
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ``MAX_ITERATION`` and ``ITERATION_FIELD_WIDTH`` used to be defined here, as a
# 9-digit copy that disagreed with the Fortran and with nothing else in the
# process -- so nothing ever caught it. They now live beside the restart naming
# in ``.utils.warm_start_utils`` (imported below) and are re-exported from this
# module for callers that already read them off it.

import numpy as np
import xarray
from pylbm.utils import get_lbm_directory_paths

from pyurbanair.base_ensemble_forward_model import ForwardModelRunFailure
from pyurbanair.base_forward_model import BaseForwardModel

from .stl_to_lbm import stl_to_lbm_geometry
from .utils import (
    Infile,
    apply_inflow_settings,
    apply_inlet_turbulence,
    compile_lbm,
    create_infile,
    validate_inlet_turbulence,
)
from .utils.build_tree_utils import compute_build_signature, read_build_stamp
from .utils.compile_utils import validate_cuda_setting
from .utils.environment_utils import identify_environment
from .utils.infile_utils import _augment_runtime_library_paths
from .utils.mod_dimensions_utils import set_experiment
from .utils.params_utils import (
    apply_sgs_setting,
    extract_initial_params,
    is_time_varying_params,
    remove_uvel_shear_file,
    remove_uvel_time_file,
    resolve_profile_config,
    write_uvel_shear_file,
    write_uvel_time_file,
)
from .utils.state_utils import scale_velocity_to_physical
from .utils.warm_start_utils import (  # noqa: F401  (MAX_ITERATION re-exported)
    ITERATION_FIELD_WIDTH,
    MAX_ITERATION,
    identify_latest_restart_iteration,
    remove_old_restart_files,
    write_restart_file_from_xarray,
)


class ForwardModel(BaseForwardModel):
    def __init__(
        self,
        stl_path: str | pathlib.Path,
        rundir: pathlib.Path | None = None,
        nx: int = 120,
        ny: int = 120,
        nz: int = 8,
        simulation_time: float = 53.8,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (0, 160),
            (0, 160),
            (0, 40),
        ),
        output_frequency: float = 0.0538,
        results_dir: Optional[pathlib.Path] = None,
        temp_dir: Optional[pathlib.Path] = None,
        verbose: bool = True,
        experiment_name: str = "runcase",
        cuda: Union[bool, str] = "auto",
        enable_netcdf: Optional[bool] = None,
        boundary_condition: str = "periodic",
        sgs_constant: Optional[float] = None,
        spinup_time: float = 0.0,
        profile_config: Optional[dict] = None,
        inlet_turbulence: Optional[dict] = None,
    ) -> None:
        super().__init__(results_dir=results_dir)

        self.spinup_time = spinup_time
        self._spinup_outputs = 0

        if boundary_condition not in ("periodic", "inflow_outflow"):
            raise ValueError(
                f"boundary_condition must be 'periodic' or 'inflow_outflow', "
                f"got '{boundary_condition}'"
            )
        self.boundary_condition = boundary_condition
        # Per-backend default for the SGS constant, overridden by a `sgs_constant`
        # in the params Dataset when one is supplied. None leaves the infile
        # template's value alone. See `apply_sgs_setting`.
        self.sgs_constant = float(sgs_constant) if sgs_constant is not None else None

        # Inflow-turbulence forcing (m_inflow_turbulence_*.F90). Validated here
        # so a bad config fails before geometry voxelisation and compilation;
        # None keeps infile.in byte-identical to a run without the knob.
        self.inlet_turbulence = validate_inlet_turbulence(
            inlet_turbulence, boundary_condition
        )

        # Verbosity
        self.verbose = verbose
        # "auto" -> CUDA where NVHPC exists, gfortran otherwise (resolved in
        # compile_lbm). Validated here so a typo fails before voxelisation.
        self.cuda = validate_cuda_setting(cuda)
        # Keep NETCDF enabled by default for both CPU and CUDA paths.
        self.enable_netcdf = True if enable_netcdf is None else enable_netcdf
        self.stdout = None if self.verbose else subprocess.DEVNULL
        self.stderr = None if self.verbose else subprocess.DEVNULL

        self.dirs = get_lbm_directory_paths(
            temp_dir=(
                pathlib.Path(temp_dir)
                if temp_dir is not None
                else pathlib.Path(".temp")
            ),
            case_dir=pathlib.Path("examples/lbm"),
            experiment_name=experiment_name,
        )

        # Generate geometry file
        stl_to_lbm_geometry(
            stl_path=stl_path,
            dirs=self.dirs,
            nx=nx,
            ny=ny,
            nz=nz,
            bounds=bounds,
        )

        # Compute cell size from bounds
        dx = (bounds[0][1] - bounds[0][0]) / nx
        dy = (bounds[1][1] - bounds[1][0]) / ny
        dz = (bounds[2][1] - bounds[2][0]) / nz
        self.x_grid = (np.arange(nx) + 0.5) * dx + bounds[0][0]
        self.y_grid = (np.arange(ny) + 0.5) * dy + bounds[1][0]
        self.z_grid = (np.arange(nz) + 0.5) * dz + bounds[2][0]

        self.min_cell_size = min(dx, dy, dz)
        self.min_cell_size = np.round(self.min_cell_size, 1)

        # Set experiment dimensions in mod_dimensions.F90 (add or update experiment, set active)
        set_experiment(dirs=self.dirs, nx=nx, ny=ny, nz=nz)

        # Vertical inflow shear profile (e.g. power_law alpha=0.25), shared with
        # pyudales via the same convention.  The profile is static in z and
        # param-independent, so write it once here; ensemble members inherit the
        # file when the experiment dir is cloned.  Heights are cell centers
        # measured from the domain bottom and ``z_ref`` defaults to the domain
        # height, matching pyudales' ``build_profile_shape``.
        self.profile_config = profile_config
        zsize = bounds[2][1] - bounds[2][0]
        profile_heights = (np.arange(nz) + 0.5) * dz
        # Cached so _apply_inflow_settings can rewrite uvel_shear.dat per member
        # when an estimated vertical_inflow_exponent (α) overrides the
        # construction-time shear (docs/esmda_model_error_parameters.md §2.1).
        self._profile_heights = profile_heights
        self._zsize = zsize
        if profile_config is not None and profile_config.get("type") not in (
            None,
            "uniform",
        ):
            write_uvel_shear_file(
                dirs=self.dirs,
                heights=profile_heights,
                zsize=zsize,
                profile_config=profile_config,
            )
        else:
            remove_uvel_shear_file(self.dirs)

        self.simulation_time = simulation_time
        self.output_frequency = output_frequency
        self.seconds_per_timestep: float | None = None
        # Derived during compile() once infile.in exists (C_t = C_l / C_u).
        self.num_timesteps = 0
        self.output_frequency_timesteps = 0
        # Warm-start override: when set, _set_scaling_factors uses this as nt0
        # instead of defaulting to 0. Consumed (reset to None) after each use.
        self._nt0_override: int | None = None

    def _compute_seconds_per_timestep(self) -> float:
        """Compute seconds per timestep from infile constants C_l/C_u."""
        infile = Infile(self.dirs.infile_path)
        c_l = infile.get_value_as_float("C_l")
        c_u = infile.get_value_as_float("C_u")
        if c_l is None or c_u is None:
            raise ValueError(
                "Could not read C_l/C_u from infile.in to compute timestep duration."
            )
        if c_u <= 0:
            raise ValueError("C_u in infile.in must be > 0.")
        return c_l / c_u

    def _verify_prebuilt_binary(self) -> None:
        """
        Refuse to reuse a binary that was not built from the current sources.

        The solver bakes ``mod_dimensions.F90`` (the grid) and the geometry case in
        ``m_solid_objects_init.F90`` in at compile time, so a binary left over from
        a different experiment or grid does not fail loudly -- it produces
        wrong-shaped or all-NaN output. ``compile()`` writes a stamp of those
        sources after every successful build; with ``compile=false`` this compares
        the stamp against what would be compiled now.

        Raises:
            RuntimeError: If the binary is missing, unstamped, or stale.
        """
        build_root = self.dirs.lbm_src_path.parent
        remedy = "Rerun with model.compile=true to rebuild (a full rebuild)."

        if not self.dirs.executable_path.exists():
            raise RuntimeError(
                f"model.compile=false but no LBM binary exists at "
                f"{self.dirs.executable_path}. {remedy}"
            )

        expected = compute_build_signature(
            src_path=self.dirs.lbm_src_path,
            experiment_name=self.dirs.experiment_name,
            enable_cuda=self.cuda,
            enable_netcdf=self.enable_netcdf,
        )
        recorded = read_build_stamp(build_root)
        if recorded is None:
            raise RuntimeError(
                f"model.compile=false but the binary at {self.dirs.executable_path} "
                "carries no build stamp, so it cannot be checked against the "
                f"current grid and geometry. {remedy}"
            )

        # 'cuda' is excluded: it does not change the solver's numerics or array
        # shapes, and cuda=auto legitimately resolves differently per host.
        stale = [
            key
            for key in ("experiment", "netcdf", "sources")
            if recorded.get(key) != expected[key]
        ]
        if stale:
            raise RuntimeError(
                f"The LBM binary at {self.dirs.executable_path} is stale: "
                f"{', '.join(stale)} changed since it was built. Running it would "
                f"silently produce output for the wrong grid or geometry. {remedy}"
            )

        logger.info(
            "Reusing LBM binary at %s (build stamp matches current sources)",
            self.dirs.executable_path,
        )

    def compile(self, compile: bool = True) -> None:
        """Compile the LBM program."""
        # Compile program
        if compile:
            compile_lbm(
                dirs=self.dirs,
                verbose=self.verbose,
                enable_cuda=self.cuda,
                enable_netcdf=self.enable_netcdf,
            )
            # A rebuilt binary may use a different RANDOM_SEED size than the
            # stale seed_*.dat/.orig files written by the previous binary,
            # which causes a FIO read past end-of-file at startup.
            for pattern in ("seed_*.dat", "seed_*.orig"):
                for seed_file in self.dirs.experiment_dir.glob(pattern):
                    seed_file.unlink(missing_ok=True)
        else:
            self._verify_prebuilt_binary()

        # Create infile.in by running the executable (only if it doesn't exist)
        if not self.dirs.infile_path.exists():
            create_infile(dirs=self.dirs, verbose=self.verbose)
        elif self.verbose:
            logger.info(
                "infile.in already exists at %s, skipping creation.",
                self.dirs.infile_path,
            )

        # Set runtime controls in timestep units
        self._set_infile_value("experiment", self.dirs.experiment_name)
        self._set_infile_value("tecout", "3" if self.enable_netcdf else "0")

        # Apply x-direction boundary condition (y is always periodic: jbnd=0)
        ibnd = 0 if self.boundary_condition == "periodic" else 1
        self._set_infile_value("ibnd", ibnd)
        self._set_infile_value("jbnd", 0)

        # Inflow turbulence is static per model, so it is written once here and
        # inherited by ensemble members when experiment_dir is cloned. No-op
        # unless explicitly enabled.
        apply_inlet_turbulence(self.inlet_turbulence, self.dirs)

    def _set_scaling_factors(self, params: Optional[xarray.Dataset] = None) -> None:
        """Set the scaling factors for the LBM."""
        self._set_infile_value("C_l", self.min_cell_size)

        if params is not None:
            if is_time_varying_params(params):
                velocity_magnitude = float(params["velocity_magnitude"].max().item())
            else:
                velocity_magnitude = float(params["velocity_magnitude"].item())
            self.C_u = int(velocity_magnitude * 15)
            # uini is the physical inflow magnitude [m/s]; uvel_shear.dat only
            # carries the normalized vertical shape (=1 at the domain top), so the
            # actual inflow speed comes from uini. Track velocity_magnitude here so
            # the inflow matches the requested params instead of the LBM template
            # default (8 m/s). For time-varying params this is only the fallback
            # used when uvel_time.dat is absent (write_uvel_time_file overrides it).
            self._set_infile_value("uini", velocity_magnitude)
        else:
            self.C_u = 75
        self._set_infile_value("C_u", self.C_u)

        self.seconds_per_timestep = self._compute_seconds_per_timestep()

        # Compute a fixed number of output steps independent of C_u, then derive
        # iout and num_timesteps so every ensemble member produces the same count.
        num_outputs = round(self.simulation_time / self.output_frequency)
        self.output_frequency_timesteps = max(
            1, round(self.output_frequency / self.seconds_per_timestep)
        )
        self.num_timesteps = self.output_frequency_timesteps * num_outputs

        if self.num_timesteps <= 0:
            raise ValueError("Resolved num_timesteps must be > 0.")

        # Extend run by spinup period (outputs produced during spinup are
        # discarded after collection in run_single).
        if self.spinup_time > 0:
            self._spinup_outputs = round(self.spinup_time / self.output_frequency)
        else:
            self._spinup_outputs = 0
        spinup_timesteps = self._spinup_outputs * self.output_frequency_timesteps
        total_timesteps = self.num_timesteps + spinup_timesteps

        if self._nt0_override is not None:
            nt0 = self._nt0_override
            self._nt0_override = None
        else:
            nt0 = 0
        # LBM output/restart filenames encode the iteration in a fixed-width
        # field (i6.6 in the Fortran). Guard against silently overflowing it: an
        # overflowed name (out_0000_F******.nc) is dropped by the collector,
        # yielding mismatched time dims across members and a concat AlignmentError.
        # Note the iteration counter ACCUMULATES across warm starts, so a long
        # rollout reaches the ceiling even when no single window is near it.
        nt1 = nt0 + total_timesteps
        if nt1 > MAX_ITERATION:
            raise ValueError(
                f"LBM final timestep nt1={nt1} exceeds the maximum representable "
                f"iteration {MAX_ITERATION} (filenames use a "
                f"{ITERATION_FIELD_WIDTH}-digit field). The counter accumulates "
                "across warm starts, so start from a clean experiment directory, "
                "reduce the run length, or widen the i6.6 format in the LBM "
                "Fortran sources (m_diag/m_saverestart/m_save_uvw/m_readrestart) "
                "and ITERATION_FIELD_WIDTH in pylbm.utils.warm_start_utils "
                "together."
            )
        self._set_infile_value("nt0", nt0)
        self._set_infile_value("nt1", nt1)
        self._set_infile_value("iout", self.output_frequency_timesteps)
        # Disable the iprt2-based "every iteration output" trigger in m_diag.F90.
        # The default iprt2=60000 causes every iteration with it>=60000 to dump a
        # NetCDF file, which makes warm-start runs (where nt0 typically already
        # exceeds 60000) ~20x slower than cold starts.
        self._set_infile_value("iprt1", f"0 {nt1 + 1} 1")

    def set_results_dir(self, results_dir: pathlib.Path | None) -> None:
        """Change results directory, updating both base and dirs dataclass."""
        super().set_results_dir(results_dir)
        self.dirs.results_dir = results_dir

    def _set_infile_value(self, key: str, value: Union[str, int, float, bool]) -> None:
        """
        Set a value in infile.in by key.

        This is a reusable helper method for updating any value in the infile.in file.

        Args:
            key: The key name in infile.in (e.g. "nt1", "experiment", "tecout").
            value: The value to set (will be converted to string if needed).
        """
        infile = Infile(self.dirs.infile_path)
        infile.set_value(key, value)
        infile.write()

    def _get_infile_int_value(self, key: str, default: int) -> int:
        """Read an integer value from infile.in, with fallback to default."""
        infile = Infile(self.dirs.infile_path)
        value = infile.get_value_as_int(key)
        return value if value is not None else default

    def _get_output_files_for_current_run(self) -> list[pathlib.Path]:
        """
        Return output netCDF files corresponding to the configured timestep range.

        This supports both cold-start runs (nt0=0) and warm-start runs (nt0>0).
        """
        nt0 = self._get_infile_int_value("nt0", 0)
        nt1 = self._get_infile_int_value("nt1", self.num_timesteps)

        output_files: list[tuple[int, pathlib.Path]] = []
        for path in self.dirs.output_dir.glob("out_0000_F*.nc"):
            match = re.search(r"_F(\d+)$", path.stem)
            if match is None:
                continue
            timestep = int(match.group(1))
            if nt0 < timestep <= nt1:
                output_files.append((timestep, path))

        output_files = sorted(output_files, key=lambda x: x[0])
        if output_files:
            return [path for _, path in output_files]

        # Fallback to expected final file
        # Same width as the restarts: m_diag writes this name with
        # ``write(cit,'(a1,i6.6)') 'F', it``.
        expected_file = (
            self.dirs.output_dir / f"out_0000_F{nt1:0{ITERATION_FIELD_WIDTH}d}.nc"
        )
        if expected_file.exists():
            return [expected_file]

        raise FileNotFoundError(
            f"No LBM output files found in {self.dirs.output_dir} for timestep range "
            f"[{nt0}, {nt1}]"
        )

    def _expected_output_count(self) -> int:
        """How many snapshots a *completed* run of the configured window writes.

        Mirrors ``m_diag.F90``'s dump condition rather than re-deriving the count
        from ``simulation_time / output_frequency``, because the two only agree
        on a cold start. The solver iterates ``it = nt0+1, nt1`` and writes when

            mod(it,iout)==0 .or. it==nt1 .or. (it<=iprt1 .or. it>=iprt2) ...

        The third (every-iteration) clause is disabled by the ``iprt1 = "0 nt1+1
        1"`` line ``_set_scaling_factors`` writes, so what lands on disk is the
        ``iout`` grid inside ``(nt0, nt1]`` -- which is what
        ``_get_output_files_for_current_run`` collects -- plus, when ``nt1`` is
        not itself on that grid, one extra frame at ``nt1``.

        That last term is the warm-start case: ``nt1 - nt0`` is always a multiple
        of ``iout``, but ``nt0`` is the previous window's final iteration and
        ``iout = C_l / C_u / output_frequency`` moves with the member's inflow
        speed, so ``nt0 % iout != 0`` is normal from window 1 onwards and the run
        legitimately ends one off-grid frame long. (It is the trailing trim in
        ``run_single`` that drops it again.) On a cold start ``nt0 = 0`` and
        ``nt1`` is a multiple of ``iout``, so the term vanishes and the count is
        exactly ``spinup + simulation_time/output_frequency`` outputs.
        """
        nt0 = self._get_infile_int_value("nt0", 0)
        nt1 = self._get_infile_int_value("nt1", self.num_timesteps)
        iout = max(1, self.output_frequency_timesteps)

        on_grid = nt1 // iout - nt0 // iout
        final_frame_off_grid = 0 if nt1 % iout == 0 else 1
        return on_grid + final_frame_off_grid

    def _verify_run_produced_all_output(self, produced: int) -> None:
        """Fail the run when the solver wrote fewer snapshots than the window needs.

        The LBM's error paths call Fortran ``stop``, which exits **0**. That
        means ``run()``'s ``check=True`` sees a clean exit and the wrapper is left
        with a partial run that looks like a successful one: the collector still
        finds files, so it does not raise, and the trailing trim in ``run_single``
        only ever *removes* frames, so a short member passes straight through. The
        mismatch then surfaces windows later as a broadcast/``AlignmentError``
        when the ensemble is concatenated -- nowhere near the member that caused
        it. (This is not hypothetical: a member wrote 3 of 48 frames, exited 0,
        and killed a two-window ESMDA run at the concat.)

        Raising here, at the boundary where the expected frame count is known,
        turns it into an ordinary member failure: the ensemble runner's
        ``resample_from_successes`` policy clones the member from a survivor, and
        a single (non-ensemble) forward run -- which has no failure policy --
        still fails loudly, right where the truncation happened.

        Only a *shortfall* is an error. A surplus is trimmed by design (see
        ``_expected_output_count`` on the off-grid final frame), so this must not
        turn the ordinary warm-start run into a failure.

        Raises:
            ForwardModelRunFailure: If fewer than the expected snapshots exist.
        """
        expected = self._expected_output_count()
        if produced >= expected:
            return

        nt0 = self._get_infile_int_value("nt0", 0)
        nt1 = self._get_infile_int_value("nt1", self.num_timesteps)
        raise ForwardModelRunFailure(
            f"The LBM run in {self.dirs.experiment_dir} stopped early: it wrote "
            f"{produced} of the {expected} expected snapshots for timestep range "
            f"({nt0}, {nt1}] at iout={self.output_frequency_timesteps}. The binary "
            "still exited 0, which is the usual signature of a Fortran `stop` "
            "(missing restart file, dimension mismatch, ...) -- those exit 0, so "
            "no CalledProcessError is raised. Rerun with "
            "model.forward_model.verbose=true to see the solver's own message."
        )

    def _apply_inflow_settings(self, params: xarray.Dataset) -> None:
        """Apply the inflow settings to the forward model.

        For time-varying parameters (Dataset with a ``time`` dimension),
        writes ``uvel_time.dat`` for the Fortran code and sets initial
        static values in ``infile.in``.  For static parameters, removes
        any stale ``uvel_time.dat`` and applies the values directly.
        """
        # Model-error knobs (α shear exponent, SGS constant) apply identically to
        # the static and time-varying inflow paths, so consume them here, outside
        # the branch (docs/esmda_model_error_parameters.md §6.2). Each is a no-op
        # when its parameter is absent, keeping single-model/default runs
        # byte-identical.
        override_cfg = resolve_profile_config(params, self.profile_config)
        if override_cfg is not None and override_cfg.get("type") not in (
            None,
            "uniform",
        ):
            write_uvel_shear_file(
                dirs=self.dirs,
                heights=self._profile_heights,
                zsize=self._zsize,
                profile_config=override_cfg,
            )
        apply_sgs_setting(params, self.dirs, default=self.sgs_constant)

        if is_time_varying_params(params):
            # The LBM matches the schedule against its absolute clock
            # t = (iteration - 1) * dt, which continues across warm starts
            # (nt0 > 0). Shift the (window-relative) schedule onto that clock so
            # window w is read over [nt0*dt, nt0*dt + simulation_time].
            nt0 = self._get_infile_int_value("nt0", 0)
            dt = self.seconds_per_timestep or 0.0
            write_uvel_time_file(
                params=params,
                dirs=self.dirs,
                spinup_time=self.spinup_time,
                time_offset=nt0 * dt,
            )
            initial_params = extract_initial_params(params)
            apply_inflow_settings(params=initial_params, dirs=self.dirs)
        else:
            remove_uvel_time_file(self.dirs)
            apply_inflow_settings(params=params, dirs=self.dirs)

    def save_results(self, state: xarray.Dataset, sim_name: str = "state") -> None:
        """Save simulation results to disk."""
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        """Remove netCDF output files from the output directory.

        This prevents stale files from being picked up by subsequent runs
        that may use a different output frequency (iout).
        """
        for output_file in self.dirs.output_dir.glob("out_*.nc"):
            output_file.unlink(missing_ok=True)

    def run(self) -> None:
        """
        Run the LBM executable from the rundir.

        This executes the compiled boltzmann program from self.rundir,
        which will read infile.in and run the simulation.
        """

        if self.verbose:
            logger.info("Executable: %s", self.dirs.executable_path)

        original_cwd = pathlib.Path.cwd()

        os.chdir(self.dirs.experiment_dir)

        # Set up environment
        env = os.environ.copy()
        env["HOME"] = str(self.dirs.pixi_env_path)
        if "PIXI_ENVIRONMENT" not in env:
            env["PIXI_ENVIRONMENT"] = str(self.dirs.pixi_env_path)
        _augment_runtime_library_paths(env=env, pixi_env_path=self.dirs.pixi_env_path)

        # Raise the stack size limit before launching. The LBM binary uses
        # large automatic (stack) arrays sized by nx*ny*nz; on big grids (e.g.
        # the Barcelona case at 400x400x32) these blow past the default 8 MB
        # stack and the process dies with SIGSEGV. macOS /bin/sh refuses
        # "unlimited" for the stack, so fall back to the hard limit (~64 MB);
        # both attempts are best-effort so a failure never aborts the launch.
        shell_cmd = (
            f"ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null; "
            f"{self.dirs.executable_path}"
        )
        try:
            # check=True so a non-zero LBM exit raises CalledProcessError, which
            # the ensemble runner catches to resample the member from a survivor.
            # Without it, a crashed member silently produces partial/no output and
            # later breaks the cross-member concat with an AlignmentError.
            _ = subprocess.run(
                shell_cmd,
                shell=True,
                env=env,
                stderr=self.stderr,
                stdout=self.stdout,
                text=True,
                check=True,
            )
        finally:
            # Always return to original directory, even if the run failed.
            os.chdir(original_cwd)

    def _prepare_warmstart(self, state: xarray.Dataset) -> None:
        """Write a restart file from ``state`` and arm nt0 for the next run.

        Uses the latest existing ``restart/restart_0000_<iter>.uf`` as a
        template (for ghost cells and non-equilibrium content) when
        available; otherwise falls back to a pure-equilibrium restart.
        """
        latest_restart = identify_latest_restart_iteration(self.dirs)
        restart_iteration = write_restart_file_from_xarray(
            state=state,
            dirs=self.dirs,
            restart_iteration=latest_restart,
        )
        self._nt0_override = restart_iteration

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:
        """Run the LBM executable from the rundir."""
        if not self.enable_netcdf:
            raise RuntimeError(
                "run_single requires NETCDF output, but this model was compiled with "
                "enable_netcdf=False (default for cuda=True). "
                "Either set enable_netcdf=True with an NVFORTRAN-compatible netcdf.mod, "
                "or call run() and process non-NETCDF diagnostics."
            )

        saved_spinup_time = self.spinup_time
        try:
            if state is not None:
                self.spinup_time = 0.0
                # Set C_u for THIS window before reconstructing the restart, so
                # the carried state (m/s) is scaled to lattice units with the C_u
                # the run will actually use. Otherwise the warm start uses the
                # previous window's C_u and the velocity field jumps when the
                # inflow speed changes between windows.
                self._set_scaling_factors(params)
                self._prepare_warmstart(state)

            # Finalize scaling: applies nt0 from the warm-start restart (set by
            # _prepare_warmstart) and the current C_u/timestep duration.
            self._set_scaling_factors(params)

            # Written after scaling so the time-varying inflow schedule can be
            # shifted onto the LBM's absolute, warm-start-continuing clock
            # (t = iteration*dt); a window-local schedule would otherwise be read
            # at the wrong (large) clock time and clamp to its last value.
            if params is not None:
                self._apply_inflow_settings(params)

            # Remove stale output files before running to prevent files from a
            # previous run (which may have used a different iout) being collected.
            self._clean_output()

            self.run()
        finally:
            self.spinup_time = saved_spinup_time

        # Check the frame count BEFORE anything reads or trims it: the trims
        # below only shorten the series, so a truncated run that reaches them is
        # indistinguishable from a good one and escapes into the ensemble. A
        # solver that wrote nothing at all is the same failure with produced=0,
        # so re-raise the collector's FileNotFoundError as one too rather than
        # letting a second exception type describe the same event.
        try:
            output_files = self._get_output_files_for_current_run()
        except FileNotFoundError as exc:
            self._verify_run_produced_all_output(0)
            raise exc  # expected 0 outputs and got 0: a genuine config error
        self._verify_run_produced_all_output(len(output_files))

        state = [xarray.load_dataset(path, engine="netcdf4") for path in output_files]

        if len(state) > 1:
            state = xarray.concat(state, dim="time", join="override")
        else:
            state = state[0].expand_dims("time", axis=0)

        state = state.assign(x=self.x_grid, y=self.y_grid, z=self.z_grid)
        state = scale_velocity_to_physical(state, scale=self.C_u)

        if self._spinup_outputs > 0 and state.sizes["time"] > self._spinup_outputs:
            state = state.isel(time=slice(self._spinup_outputs, None))

        expected_outputs = round(self.simulation_time / self.output_frequency)
        if state.sizes["time"] > expected_outputs:
            state = state.isel(time=slice(-expected_outputs, None))

        # Store the time coordinate in seconds (0, dt, 2·dt, …) rather than
        # bare step indices, so downstream consumers (e.g. the temporal
        # observation operator's seconds-based interval binning) see a real
        # time axis consistent with the other backends.
        state = state.assign_coords(
            time=np.arange(state.sizes["time"], dtype=float) * self.output_frequency
        )

        remove_old_restart_files(self.dirs)

        return state

    def disable_spinup(self) -> None:
        """Disable spinup so subsequent runs use only simulation_time."""
        self.spinup_time = 0.0
