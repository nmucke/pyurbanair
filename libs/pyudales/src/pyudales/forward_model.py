import json
import logging
import os
import pathlib
import pdb
import shutil
import subprocess
import time
from typing import Optional

import xarray

from pyurbanair.base_forward_model import BaseForwardModel

from . import LOCAL_EXECUTE_SCRIPT, UDALES_PATH
from .utils.clean_up_utils import clean_output_dir, clean_temp_dir
from .utils.config_utils import create_config_sh
from .utils.dir_utils import get_project_root, get_udales_directory_paths
from .utils.file_utils import copy_files
from .utils.namoptions_utils import NamoptionsFile, rename_namoptions_file
from .utils.ncpu_utils import validate_and_sync_ncpu
from .utils.nudging_utils import apply_time_varying_inflow
from .utils.params_utils import (
    apply_inflow_settings,
    is_time_varying_params,
    merge_params,
)
from .utils.random_utils import apply_random_initial_condition
from .utils.save_frequency_utils import (
    apply_output_frequency,
    apply_save_only_last_timestep,
)
from .utils.warm_start_utils import (
    clean_output_except_warmstart_files,
    identify_generated_warmstart_file,
    remove_old_warmstart_files,
    set_trestart,
    set_warm_start,
    update_warmstart_file_from_xarray,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MATLAB_BIN = pathlib.Path("/Applications/MATLAB_R2025b.app/bin/matlab")

# Glob patterns for the grid-dependent IBM geometry files that the STL->IBM
# Fortran step produces and that a precomputed bundle reuses.
PRECOMPUTED_GEOM_PATTERNS = (
    "solid_*.txt",
    "fluid_boundary_*.txt",
    "facet_sections_*.txt",
)


def save_precomputed_geometry(
    experiment_dir: pathlib.Path,
    dest_dir: pathlib.Path,
    namoptions_path: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """Bundle IBM geometry files from a completed run for later reuse.

    Copies the ``solid_*``/``fluid_boundary_*``/``facet_sections_*`` files
    produced by a from-STL preprocessing run into ``dest_dir`` and writes a
    ``geom_meta.json`` recording the grid (itot/jtot/ktot) and nfcts. Point a
    later run's ``precomputed_geom_dir`` at ``dest_dir`` to skip the expensive
    STL->IBM Fortran step.

    Args:
        experiment_dir: Experiment dir of a from-STL run (``gen_geom=.true.``).
        dest_dir: Destination bundle directory (created if needed).
        namoptions_path: namoptions to read the grid from; defaults to the
            ``namoptions.*`` inside ``experiment_dir``.

    Returns:
        The destination directory.
    """
    experiment_dir = pathlib.Path(experiment_dir)
    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for pattern in PRECOMPUTED_GEOM_PATTERNS:
        for src in sorted(experiment_dir.glob(pattern)):
            shutil.copy2(src, dest_dir / src.name)
            copied.append(src.name)
    if not copied:
        raise FileNotFoundError(
            "No IBM geometry files (solid_*/fluid_boundary_*/facet_sections_*) "
            f"found in {experiment_dir}; run preprocessing from the STL first."
        )

    if namoptions_path is None:
        matches = sorted(experiment_dir.glob("namoptions.*"))
        if not matches:
            raise FileNotFoundError(f"No namoptions.* found in {experiment_dir}")
        namoptions_path = matches[0]
    namoptions = NamoptionsFile(pathlib.Path(namoptions_path))
    meta = {
        "grid": {
            k: namoptions.get_value_as_int("DOMAIN", k)
            for k in ("itot", "jtot", "ktot")
        },
        "nfcts": namoptions.get_value_as_int("WALLS", "nfcts"),
        "files": sorted(copied),
    }
    with open(dest_dir / "geom_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved precomputed geometry bundle (%d files) to %s", len(copied), dest_dir)
    return dest_dir

DEFAULT_TEMP_DIR = lambda cwd: pathlib.Path(f"{cwd}/.temp")

# Default parameter values as xarray.Dataset
DEFAULT_PARAMS = xarray.Dataset(
    data_vars={
        "inflow_angle": 0,
        "velocity_magnitude": 3,
        "pressure_gradient_magnitude": 0.0041912,
    },
)

DEFAULT_NUDGING_CONFIG = {
    "tnudge": 10.0,
    "nnudge": 0,
}

DomainBounds = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


def _augment_runtime_library_paths(env: dict[str, str]) -> None:
    """Ensure runtime loader can find shared libraries in active pixi/conda env."""
    lib_paths: list[pathlib.Path] = []

    conda_prefix = env.get("CONDA_PREFIX")
    pixi_environment = env.get("PIXI_ENVIRONMENT")

    prefix_candidates: list[pathlib.Path] = []
    if conda_prefix:
        prefix_candidates.append(pathlib.Path(conda_prefix))
    if pixi_environment:
        pixi_path = pathlib.Path(pixi_environment)
        if pixi_path.exists():
            prefix_candidates.append(pixi_path)
        else:
            prefix_candidates.append(
                get_project_root() / ".pixi" / "envs" / pixi_environment
            )

    for prefix_path in prefix_candidates:
        if not prefix_path.exists():
            continue
        lib_dir = prefix_path / "lib"
        if lib_dir.exists():
            lib_paths.append(lib_dir)

        # Also include optional NVHPC-local netcdf-fortran install used by CUDA/LBM.
        local_netcdf_lib = prefix_path / ".nvhpc" / "netcdf-fortran" / "lib"
        if local_netcdf_lib.exists():
            lib_paths.append(local_netcdf_lib)

    if not lib_paths:
        return

    existing = env.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(str(p) for p in lib_paths)
    env["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


class ForwardModel(BaseForwardModel):
    """
    Forward model class.

    The forward model is a wrapper around the uDALES code.
    """

    def __init__(
        self,
        case_dir: pathlib.Path,
        experiment_name: str = "300",
        ncpu: int = 4,
        simulation_time: float | None = None,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        bounds: DomainBounds | None = None,
        matlab_bin: pathlib.Path = DEFAULT_MATLAB_BIN,
        save_only_last_timestep: bool = False,
        output_frequency: Optional[float] = None,
        params: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
        verbose: bool = True,
        temp_dir: Optional[pathlib.Path] = None,
        experiment_base_dir: Optional[pathlib.Path] = None,
        output_dir: Optional[pathlib.Path] = None,
        random_initial_condition_args: Optional[dict] = None,
        boundary_condition: str = "periodic",
        spinup_time: float = 0.0,
        nudging_config: Optional[dict] = None,
        precomputed_geom_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the ForwardModel.

        Args:
            case_dir: The directory containing the original case files.
            experiment_name: The name of the experiment.
            ncpu: The number of CPUs to use.
            simulation_time: Total simulation runtime in seconds. If provided,
                writes &RUN runtime in namoptions.
            nx: Number of grid cells in x direction (maps to itot in namoptions).
            ny: Number of grid cells in y direction (maps to jtot in namoptions).
            nz: Number of grid cells in z direction (maps to ktot in namoptions).
            bounds: Domain bounds in the form
                ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
                Domain lengths are written to xlen/ylen/zsize in namoptions.
            matlab_bin: The path to the MATLAB binary.
            save_only_last_timestep: If True, only the last timestep will be saved. Overwrites save_frequency.
            output_frequency: The frequency at which the output will be saved.
            params: The parameters of the forward model.
                Currently, we only support the following parameters:
                - inflow_angle: The angle of the inflow wind speed in degrees (measured from positive x-axis).
                - velocity_magnitude: The magnitude of the inflow wind speed (m/s).
                - pressure_gradient_magnitude: The magnitude of the inflow pressure gradient (Pa/m).
            results_dir: The directory where the results will be saved.
            verbose: If True, print output from Fortran code execution. If False, suppress all output.
            temp_dir: The base temp directory (defaults to {cwd}/.temp).
            experiment_base_dir: The base directory for experiments (defaults to {temp_dir}/experiment).
            output_dir: The directory for intermediate uDALES outputs (defaults to {temp_dir}/outputs).
            nudging_config: Optional dict with nudging tunables for time-varying params.
                Supported keys: ``tnudge`` (relaxation timescale in seconds, default 10.0),
                ``nnudge`` (number of levels from bottom NOT nudged, default 0).
            precomputed_geom_dir: Optional path to a directory of precomputed IBM
                geometry files (``solid_*``/``fluid_boundary_*``/``facet_sections_*``,
                as written by a prior from-STL preprocessing run, e.g. via
                :func:`save_precomputed_geometry`). When set, preprocessing copies
                these files instead of re-running the expensive STL->IBM Fortran
                classifier. Relative paths resolve against the project root. Leave
                as ``None`` (default) to generate the geometry from the STL.
                The files are grid-specific; if the directory contains a
                ``geom_meta.json`` its recorded grid is checked against the active
                domain and a mismatch raises.
        """
        super().__init__(results_dir=results_dir)

        # Verbose flag for controlling output
        self.verbose = verbose
        self.stdout = None if self.verbose else subprocess.DEVNULL
        self.stderr = None if self.verbose else subprocess.DEVNULL

        self.clean_output = True

        # Create directory paths dataclass with defaults or provided paths
        self.dirs = get_udales_directory_paths(
            case_dir=case_dir,
            experiment_name=experiment_name,
            udales_root_path=UDALES_PATH,  # type: ignore[arg-type]
            temp_dir=temp_dir,
            experiment_base_dir=experiment_base_dir,
            output_dir=output_dir,
            results_dir=results_dir,
        )

        self._warmstart_template_dir = self.dirs.experiment_dir / "warmstart_template"
        self.warmstart_template_file: pathlib.Path | None = None

        # Save only the last timestep
        self.save_only_last_timestep = save_only_last_timestep

        # Save frequency
        if save_only_last_timestep:
            self.output_frequency = None
        else:
            self.output_frequency = output_frequency

        # MATLAB binary
        self.matlab_bin = matlab_bin

        # Initialize params by merging provided params with defaults
        self.params = merge_params(
            existing_params=DEFAULT_PARAMS,
            new_params=params,
        )
        if self.params is None:
            raise ValueError("ForwardModel requires at least one inflow parameter.")

        # Copy files from case_dir to experiment_dir
        copy_files(self.dirs.case_dir, self.dirs.experiment_dir)

        # Rename the namoptions file to have the experiment_name as its extension
        rename_namoptions_file(self.dirs.experiment_dir, self.dirs.experiment_name)

        self.bounds: DomainBounds | None = None

        self.spinup_time = spinup_time
        self._simulation_time = simulation_time

        self._apply_runtime_override(simulation_time=simulation_time)
        self._apply_domain_overrides(nx=nx, ny=ny, nz=nz, bounds=bounds)

        # Apply boundary conditions (x-direction configurable, y always periodic)
        if boundary_condition not in ("periodic", "inflow_outflow"):
            raise ValueError(
                f"boundary_condition must be 'periodic' or 'inflow_outflow', "
                f"got '{boundary_condition}'"
            )
        self.boundary_condition = boundary_condition
        self._apply_boundary_condition()

        # Optionally reuse precomputed IBM geometry instead of re-running the
        # (expensive) STL->IBM preprocessing. Runs after the domain overrides so
        # the grid-consistency check sees the final itot/jtot/ktot.
        self.precomputed_geom_dir = precomputed_geom_dir
        if precomputed_geom_dir is not None:
            self._configure_precomputed_geometry(precomputed_geom_dir)

        # Validate and sync NCPU with nprocx * nprocy from namoptions
        self.ncpu = validate_and_sync_ncpu(
            dirs=self.dirs,
            ncpu=ncpu,
        )

        # Create a config.sh file where the environment variables are set
        create_config_sh(
            dirs=self.dirs,
            matlab_bin=self.matlab_bin,
            ncpu=self.ncpu,
        )

        # Store nudging config for time-varying params.
        # When using inflow_outflow BCs, nudging is required for stability,
        # so provide sensible defaults if no nudging config was given.
        self._nudging_config = nudging_config or {}
        if not self._nudging_config and self.boundary_condition == "inflow_outflow":
            self._nudging_config = DEFAULT_NUDGING_CONFIG

        # NOTE: inflow settings (nudging files, prof.inp, lscale.inp updates)
        # are NOT applied here.  They are deferred to run_single() via
        # _apply_inflow_settings() so that they run AFTER preprocessing
        # (which deletes generated files via clean_temp_dir) and with the
        # final parameter values (which may be provided at call time).

        if self.save_only_last_timestep:
            apply_save_only_last_timestep(self.dirs)
        elif self.output_frequency is not None:
            apply_output_frequency(self.dirs, self.output_frequency)

        if random_initial_condition_args is not None:
            apply_random_initial_condition(self.dirs, random_initial_condition_args)

        logger.info(f"Experiment name: {self.dirs.experiment_name}")
        logger.info(f"Case dir: {self.dirs.case_dir}")
        logger.info(f"Temp dir: {self.dirs.temp_dir}")
        logger.info(f"Experiment base dir: {self.dirs.experiment_base_dir}")
        logger.info(f"Experiment dir: {self.dirs.experiment_dir}")
        logger.info(f"Output dir: {self.dirs.output_dir}")
        logger.info(f"NCPU: {self.ncpu}")
        logger.info(f"MATLAB bin: {self.matlab_bin}")

    def _apply_runtime_override(self, simulation_time: float | None) -> None:
        """Apply optional simulation runtime override to namoptions.

        When spinup_time > 0 the effective runtime written to namoptions is
        ``simulation_time + spinup_time``; the extra output produced during
        the spinup window is trimmed in ``run_single``.
        """
        if simulation_time is None:
            return
        if simulation_time <= 0:
            raise ValueError("simulation_time must be > 0.")

        effective_time = simulation_time + self.spinup_time

        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        namoptions = NamoptionsFile(namoptions_path)
        namoptions.set_value("RUN", "runtime", effective_time)
        namoptions.write()

    def _apply_domain_overrides(
        self,
        nx: int | None,
        ny: int | None,
        nz: int | None,
        bounds: DomainBounds | None,
    ) -> None:
        """Apply optional domain overrides to namoptions."""
        provided_any = any(v is not None for v in (nx, ny, nz, bounds))
        if not provided_any:
            return

        if nx is None or ny is None or nz is None or bounds is None:
            raise ValueError(
                "If one of nx/ny/nz/bounds is provided, all four must be provided."
            )

        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("nx, ny, and nz must all be positive integers.")

        for axis_name, axis_bounds in zip(("x", "y", "z"), bounds):
            if axis_bounds[1] <= axis_bounds[0]:
                raise ValueError(
                    f"Invalid {axis_name} bounds: upper bound must be greater than lower bound."
                )

        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        namoptions = NamoptionsFile(namoptions_path)
        namoptions.set_value("DOMAIN", "itot", nx)
        namoptions.set_value("DOMAIN", "jtot", ny)
        namoptions.set_value("DOMAIN", "ktot", nz)
        namoptions.set_value("DOMAIN", "xlen", bounds[0][1] - bounds[0][0])
        namoptions.set_value("DOMAIN", "ylen", bounds[1][1] - bounds[1][0])
        namoptions.set_value("INPS", "zsize", bounds[2][1] - bounds[2][0])
        namoptions.write()

        self.bounds = bounds

        # When the domain has non-zero lower bounds, the STL geometry must be
        # shifted so that it is correctly positioned within the [0, length]
        # computational domain.  uDALES always starts its grid at 0, so an
        # STL vertex originally at x=0 needs to move to x=-xmin (e.g. +100
        # when xmin=-100) inside the enlarged domain.
        x_offset = -bounds[0][0]
        y_offset = -bounds[1][0]
        z_offset = -bounds[2][0]
        if x_offset != 0.0 or y_offset != 0.0 or z_offset != 0.0:
            self._shift_stl_geometry(namoptions_path, (x_offset, y_offset, z_offset))

    def _shift_stl_geometry(
        self,
        namoptions_path: pathlib.Path,
        offsets: tuple[float, float, float],
    ) -> None:
        """Shift STL geometry vertices so buildings sit correctly in the enlarged domain.

        Args:
            namoptions_path: Path to the namoptions file (used to read the STL filename).
            offsets: (x_offset, y_offset, z_offset) to add to every vertex.
        """
        import trimesh

        namoptions = NamoptionsFile(namoptions_path)
        stl_filename = namoptions.get_value("INPS", "stl_file")
        if stl_filename is None:
            logger.warning("No stl_file found in namoptions; skipping STL shift.")
            return

        stl_path = self.dirs.experiment_dir / stl_filename.strip().strip("'\"")
        if not stl_path.exists():
            logger.warning("STL file %s not found; skipping STL shift.", stl_path)
            return

        mesh = trimesh.load(stl_path)
        mesh.vertices[:, 0] += offsets[0]
        mesh.vertices[:, 1] += offsets[1]
        mesh.vertices[:, 2] += offsets[2]
        mesh.export(stl_path)

        logger.info(
            "Shifted STL vertices by (%.2f, %.2f, %.2f) for negative-bound domain.",
            *offsets,
        )

    def _apply_boundary_condition(self) -> None:
        """Apply x-direction boundary condition to namoptions (y is always periodic)."""
        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        namoptions = NamoptionsFile(namoptions_path)
        bcxm = 1 if self.boundary_condition == "periodic" else 2
        namoptions.set_value("BC", "BCxm", bcxm)
        namoptions.set_value("BC", "BCym", 1)
        if self.boundary_condition == "inflow_outflow":
            namoptions.set_value("BC", "BCtopm", 3)
        namoptions.write()

    def set_results_dir(self, results_dir: pathlib.Path | None) -> None:
        """Change results directory, updating both base and dirs dataclass."""
        super().set_results_dir(results_dir)
        self.dirs.results_dir = results_dir

    def _apply_inflow_settings(self, params: xarray.Dataset) -> None:
        """Apply the inflow settings to the forward model."""
        if params is not None:
            self.params = merge_params(self.params, params)

        if self.params is None:
            raise ValueError("ForwardModel parameters are unexpectedly unset.")

        use_nudging = (
            is_time_varying_params(self.params)
            or self.boundary_condition == "inflow_outflow"
        )
        use_nudging = True

        if use_nudging:
            logger.info(
                "Applying inflow via nudging (time_varying=%s, BC=%s, nudging_config=%s)",
                is_time_varying_params(self.params),
                self.boundary_condition,
                self._nudging_config,
            )

            apply_time_varying_inflow(
                params=self.params,
                dirs=self.dirs,
                spinup_time=self.spinup_time,
                simulation_time=(
                    self._simulation_time if self._simulation_time is not None else 0.0
                ),
                boundary_condition=self.boundary_condition,
                **self._nudging_config,
            )
        else:
            logger.info("Applying inflow via static settings (periodic BC)")
            apply_inflow_settings(
                params=self.params,
                dirs=self.dirs,
                boundary_condition=self.boundary_condition,
            )

    def save_results(self, state: xarray.Dataset, sim_name: str = "state") -> None:
        """Save simulation results to disk."""
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        """Clean the output directory."""
        clean_output_dir(self.dirs)

    def _configure_precomputed_geometry(self, precomputed_geom_dir: str) -> None:
        """Point preprocessing at precomputed IBM geometry instead of the STL.

        The expensive part of uDALES preprocessing is the STL->IBM step
        (solid/fluid point classification + facet-to-cell matching), run by the
        Fortran ``IBM_preproc.exe``. For a fixed STL *and* grid it always
        produces the same ``solid_*``/``fluid_boundary_*``/``facet_sections_*``
        files, so they can be computed once and reused.

        This flips ``gen_geom`` to ``.false.`` and sets ``geom_path`` in
        namoptions; ``write_inputs`` then copies those files into the experiment
        dir (and derives the matching &WALLS counts) rather than re-running the
        Fortran. The STL is still used for the cheap facet geometry
        (facets.inp/facetarea.inp), so the from-STL path is fully intact when
        ``precomputed_geom_dir`` is left unset.
        """
        root = pathlib.Path(precomputed_geom_dir)
        if not root.is_absolute():
            root = get_project_root() / root
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"precomputed_geom_dir does not exist or is not a directory: {root}"
            )

        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        namoptions = NamoptionsFile(namoptions_path)

        meta_path = root / "geom_meta.json"
        if meta_path.exists():
            self._validate_precomputed_geometry_grid(meta_path, namoptions)
        else:
            logger.warning(
                "Using precomputed geometry from %s without geom_meta.json; grid "
                "consistency with the current itot/jtot/ktot cannot be verified. "
                "Ensure these files were generated for the active domain.",
                root,
            )

        # gen_geom/geom_path live in &INPS; geom_path must be whitespace-free
        # because the preprocessing namoptions reader strips all whitespace from
        # values (so the path cannot contain spaces).
        namoptions.set_value("INPS", "gen_geom", ".false.")
        namoptions.set_value("INPS", "geom_path", str(root))
        namoptions.write()
        logger.info(
            "Using precomputed uDALES geometry from %s (gen_geom=.false.)", root
        )

    def _validate_precomputed_geometry_grid(
        self, meta_path: pathlib.Path, namoptions: NamoptionsFile
    ) -> None:
        """Raise if the precomputed bundle's grid differs from the active grid."""
        with open(meta_path) as f:
            meta = json.load(f)
        grid = meta.get("grid", {})
        mismatches = []
        for key in ("itot", "jtot", "ktot"):
            expected = grid.get(key)
            actual = namoptions.get_value_as_int("DOMAIN", key)
            if expected is not None and actual is not None and int(expected) != int(actual):
                mismatches.append(f"{key}: bundle={expected} vs current={actual}")
        if mismatches:
            raise ValueError(
                "Precomputed geometry grid does not match the active domain ("
                + "; ".join(mismatches)
                + f"). Regenerate the bundle in {meta_path.parent} for this grid "
                "(see save_precomputed_geometry), or unset precomputed_geom_dir "
                "to run preprocessing from the STL."
            )

    def run_preprocessing(self, python_or_matlab: str = "python") -> None:
        """Run preprocessing."""

        logger.info("Running preprocessing...")

        clean_temp_dir(self.dirs)
        self._clean_output()

        if python_or_matlab == "python":
            # Use Python-based preprocessing script
            script_path = (
                pathlib.Path(__file__).parent.parent.parent
                / "shell_scripts"
                / "write_inputs.sh"
            )

            command = [
                "bash",
                str(script_path),
                str(self.dirs.experiment_dir),
            ]
            env = os.environ.copy()
            # Set environment variables needed by the script
            env["DA_EXPDIR"] = str(self.dirs.experiment_base_dir)
            env["DA_TOOLSDIR"] = str(
                pathlib.Path(self.dirs.udales_root_path).joinpath("tools")
            )
            _augment_runtime_library_paths(env)

        elif python_or_matlab == "matlab":
            # Use MATLAB-based preprocessing script
            command = [
                "bash",
                str(
                    pathlib.Path(self.dirs.udales_root_path).joinpath(
                        "tools", "write_inputs.sh"
                    )
                ),
                str(self.dirs.experiment_dir),
            ]
            # Add MATLAB bin directory to PATH so the script can find 'matlab'
            env = os.environ.copy()
            matlab_bin_dir = str(pathlib.Path(self.matlab_bin).parent)
            env["PATH"] = f"{matlab_bin_dir}:{env.get('PATH', '')}"
            _augment_runtime_library_paths(env)

        subprocess.run(
            command, check=True, env=env, stdout=self.stdout, stderr=self.stderr
        )

        # Wait for MATLAB preprocessing to complete if using MATLAB
        if python_or_matlab == "matlab":
            time.sleep(90)

        logger.info("Preprocessing completed.")

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:
        """Run the forward model.

        If ``state`` is None, run a cold start (with optional spinup). If a
        warmstart template has not been captured yet, trestart is enabled so
        the run produces a restart file that is harvested as the template.

        If ``state`` is provided, run a warm start from that state: the
        warmstart template is bootstrapped lazily if missing, filled with the
        provided state, and the simulation runs without spinup.
        """
        self._apply_inflow_settings(params=params)

        saved_spinup_time = self.spinup_time
        saved_namoptions = self._snapshot_namoptions()
        try:
            if state is None:
                self._run_executable()
                result = self._load_and_postprocess_state()
            else:
                self._ensure_warmstart_template()
                self.spinup_time = 0.0
                self._rewrite_runtime(self._simulation_time)
                set_trestart(self.dirs)
                self._prepare_warmstart(state)
                self._run_executable()
                result = self._load_and_postprocess_state()
                clean_output_except_warmstart_files(self.dirs)
                remove_old_warmstart_files(self.dirs)

            return result
        finally:
            self.spinup_time = saved_spinup_time
            self._restore_namoptions(saved_namoptions)

    def _run_executable(self) -> None:
        """Invoke the uDALES executable via LOCAL_EXECUTE_SCRIPT."""
        logger.info("Running forward model...")
        command = [
            "bash",
            str(LOCAL_EXECUTE_SCRIPT),
            str(self.dirs.experiment_dir),
        ]
        env = os.environ.copy()
        _augment_runtime_library_paths(env)
        subprocess.run(
            command,
            check=True,
            env=env,
            stdout=self.stdout,
            stderr=self.stderr,
        )

    def _load_and_postprocess_state(self) -> xarray.Dataset:
        """Load fielddump output, shift coordinates, trim spinup outputs."""
        output_file = self.dirs.output_dir.joinpath(
            self.dirs.experiment_name, f"fielddump.{self.dirs.experiment_name}.nc"
        )
        if not output_file.exists():
            single_proc_file = self.dirs.output_dir.joinpath(
                self.dirs.experiment_name,
                f"fielddump.000.000.{self.dirs.experiment_name}.nc",
            )
            if single_proc_file.exists():
                output_file = single_proc_file

        state = xarray.open_dataset(output_file, engine="netcdf4")

        if self.bounds is not None:
            x_offset = self.bounds[0][0]
            y_offset = self.bounds[1][0]
            z_offset = self.bounds[2][0]
            coord_updates = {}
            for coord_name, offset in [
                ("xt", x_offset),
                ("xm", x_offset),
                ("yt", y_offset),
                ("ym", y_offset),
                ("zt", z_offset),
                ("zm", z_offset),
            ]:
                if coord_name in state.coords:
                    coord_updates[coord_name] = state.coords[coord_name].values + offset
            if coord_updates:
                state = state.assign_coords(coord_updates)

        # uDALES emits fielddump at t ∈ {tf, 2·tf, …, k·tf} with
        # k = floor(runtime/tf); it does not emit a frame at t=0. So the
        # number of post-spinup frames to discard is floor(spinup/tf),
        # never round(): for spinup=15, tf=2, round(7.5)=8 would drop one
        # too many (frames at t=2..14 are only seven frames).
        if self.spinup_time > 0 and self.output_frequency is not None:
            spinup_outputs = int(self.spinup_time / self.output_frequency)
            if state.sizes.get("time", 0) > spinup_outputs:
                state = state.isel(time=slice(spinup_outputs, None))
                if "time" in state.coords and state.sizes["time"] > 0:
                    state = state.assign_coords(time=state.time - state.time.values[0])

        if (
            self._simulation_time is not None
            and self.output_frequency is not None
            and state.sizes.get("time", 0) > 0
        ):
            expected_outputs = int(self._simulation_time / self.output_frequency)
            actual = state.sizes["time"]
            if actual > expected_outputs:
                state = state.isel(time=slice(-expected_outputs, None))
            elif actual < expected_outputs:
                raise RuntimeError(
                    f"uDALES produced {actual} fielddump frames, expected "
                    f"{expected_outputs} "
                    f"(simulation_time={self._simulation_time}, "
                    f"output_frequency={self.output_frequency}, "
                    f"spinup_time={self.spinup_time}, "
                    f"runtime={self._simulation_time + self.spinup_time})."
                )

        return state

    def _snapshot_namoptions(self) -> str:
        """Return the full text of the namoptions file for later restore."""
        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        return namoptions_path.read_text()

    def _restore_namoptions(self, text: str) -> None:
        """Restore the namoptions file from a previously captured snapshot."""
        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        namoptions_path.write_text(text)

    def _rewrite_runtime(self, runtime: float | None) -> None:
        """Write ``RUN/runtime`` to namoptions; no-op when runtime is None."""
        if runtime is None:
            return
        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        namoptions = NamoptionsFile(namoptions_path)
        namoptions.set_value("RUN", "runtime", runtime)
        namoptions.write()

    def _prepare_warmstart(self, state: xarray.Dataset) -> None:
        """Fill the warmstart template with ``state`` and arm warm start flags."""
        if self.warmstart_template_file is None:
            raise RuntimeError(
                "warmstart_template_file is unset; call _ensure_warmstart_template first."
            )
        # Strip ``time`` so the warmstart's timee is written as 0. uDALES
        # treats RUN/runtime as an absolute end-time: leaving the previous
        # run's end-time in the state causes timee >= runtime and the
        # simulation dies with SIGILL before producing any output.
        state_for_warmstart = state
        if "time" in state_for_warmstart.dims:
            state_for_warmstart = state_for_warmstart.isel(time=-1)
        if "time" in state_for_warmstart.coords:
            state_for_warmstart = state_for_warmstart.drop_vars("time")
        update_warmstart_file_from_xarray(
            state_for_warmstart,
            self.dirs,
            warmstart_file=self.warmstart_template_file,
        )
        shutil.copy(
            self.warmstart_template_file,
            self.dirs.output_dir
            / self.dirs.experiment_name
            / self.warmstart_template_file.name,
        )
        set_warm_start(self.dirs)

    def _ensure_warmstart_template(self) -> None:
        """Create a warmstart template via a short cold-start if missing."""
        if (
            self.warmstart_template_file is not None
            and self.warmstart_template_file.exists()
        ):
            return

        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        original_namoptions_text = namoptions_path.read_text()
        namoptions = NamoptionsFile(namoptions_path)

        dtmax_str = namoptions.get_value("RUN", "dtmax")
        short_runtime = float(dtmax_str) if dtmax_str else 1.0

        try:
            namoptions.set_value("RUN", "runtime", short_runtime)
            namoptions.set_value("RUN", "trestart", short_runtime)
            namoptions.set_value("RUN", "lwarmstart", ".false.")
            namoptions.write()

            self._run_executable()
            self._capture_template_from_output()
        finally:
            namoptions_path.write_text(original_namoptions_text)
            self._clean_output()

    def _capture_template_from_output(self) -> None:
        """Harvest the restart file produced by the last run as the template."""
        self._warmstart_template_dir.mkdir(parents=True, exist_ok=True)
        filename = identify_generated_warmstart_file(self.dirs)
        template_path = self._warmstart_template_dir / filename
        template_path.unlink(missing_ok=True)
        src = self.dirs.output_dir / self.dirs.experiment_name / filename
        shutil.copy(src, template_path)
        self.warmstart_template_file = template_path

    def disable_spinup(self) -> None:
        """Disable spinup so subsequent runs use only simulation_time."""
        self.spinup_time = 0.0
        if self._simulation_time is not None:
            namoptions_path = (
                self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
            )
            namoptions = NamoptionsFile(namoptions_path)
            namoptions.set_value("RUN", "runtime", self._simulation_time)
            namoptions.write()
