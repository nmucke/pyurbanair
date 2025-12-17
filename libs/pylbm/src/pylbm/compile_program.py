"""
Compile the LBM program.
"""

import os
import pathlib
import shutil
import subprocess
import sys

from . import LBM_PATH


def identify_environment(repo_root: pathlib.Path, verbose: bool = True) -> pathlib.Path:
    """
    Identify the current pixi environment path.

    Checks in order:
    1. PIXI_ENVIRONMENT environment variable
    2. PIXI_PROJECT_ENVIRONMENT environment variable
    3. Checks for active environment in .pixi/envs/
    4. Defaults to .pixi/envs/default

    Args:
        repo_root: Root directory of the repository

    Returns:
        Path to the pixi environment directory
    """
    # Check PIXI_ENVIRONMENT environment variable first
    pixi_env = os.environ.get("PIXI_ENVIRONMENT")
    if pixi_env:
        pixi_env_path = pathlib.Path(pixi_env)
        if pixi_env_path.exists():
            if verbose:
                print(
                    f"Using pixi environment from PIXI_ENVIRONMENT: {pixi_env_path}",
                    file=sys.stderr,
                )
            return pixi_env_path

    # Check PIXI_PROJECT_ENVIRONMENT (set by pixi when activating an environment)
    pixi_proj_env = os.environ.get("PIXI_PROJECT_ENVIRONMENT")
    if pixi_proj_env:
        pixi_env_path = pathlib.Path(pixi_proj_env)
        if pixi_env_path.exists():
            if verbose:
                print(
                    f"Using pixi environment from PIXI_PROJECT_ENVIRONMENT: {pixi_env_path}",
                    file=sys.stderr,
                )
            return pixi_env_path

    # Check for .pixi/envs directory
    pixi_envs_dir = repo_root / ".pixi" / "envs"
    if pixi_envs_dir.exists():
        # Check for common environment names
        for env_name in ["dev", "default"]:
            env_path = pixi_envs_dir / env_name
            if env_path.exists():
                if verbose:
                    print(f"Using pixi environment: {env_name}", file=sys.stderr)
                return env_path

    # Default fallback
    default_env = repo_root / ".pixi" / "envs" / "default"
    if verbose:
        print(f"Using default pixi environment: {default_env}", file=sys.stderr)
    return default_env


def _check_netcdf_needed(rundir: pathlib.Path) -> bool:
    """
    Check if NETCDF compilation is needed by examining infile.in.

    NETCDF is needed when tecout is set to 3 (netcdf output format).

    Args:
        rundir: Directory containing infile.in

    Returns:
        True if NETCDF should be enabled, False otherwise
    """
    infile_path = rundir / "infile.in"
    if not infile_path.exists():
        # If infile.in doesn't exist yet, check the default template
        # The default template has tecout=3, so we enable NETCDF by default
        return True

    try:
        infile_content = infile_path.read_text()
        # Look for the tecout line - it's formatted as: "3                ! tecout"
        # We need to find the line with "tecout" and extract the number
        for line in infile_content.splitlines():
            if "tecout" in line.lower():
                # Extract the first number from the line
                parts = line.split()
                if parts:
                    try:
                        tecout_value = int(parts[0])
                        return tecout_value == 3
                    except ValueError:
                        continue
        # If tecout line not found, default to False (safer)
        return False
    except Exception as e:
        print(
            f"Warning: Could not read infile.in to check tecout: {e}",
            file=sys.stderr,
        )
        # Default to False if we can't read the file
        return False


def compile_lbm(
    rundir: pathlib.Path | None = None, case_name: str = "runcase", verbose: bool = True
) -> None:
    """
    Compile the LBM program.

    The program will create an infile.in in the compilation directory.
    This function ensures the infile.in ends up in the specified rundir.

    Args:
        rundir: Directory where infile.in should be placed. If None, uses .temp/lbm
        case_name: Name of the case module (default: "runcase")
        verbose: If True, print output from compilation
    """
    if rundir is None:
        rundir = pathlib.Path.cwd() / ".temp" / "lbm"

    stderr = sys.stderr if verbose else subprocess.DEVNULL
    stdout = sys.stdout if verbose else subprocess.DEVNULL

    # Convert to absolute path if relative
    if not rundir.is_absolute():
        rundir = pathlib.Path.cwd() / rundir

    # Ensure rundir exists
    rundir.mkdir(parents=True, exist_ok=True)

    # Check if NETCDF compilation is needed
    enable_netcdf = _check_netcdf_needed(rundir)
    if enable_netcdf:
        print("NETCDF support enabled (tecout=3 detected)", file=sys.stderr)

    if not LBM_PATH or not LBM_PATH.exists():
        raise FileNotFoundError(f"LBM_PATH not found: {LBM_PATH}")

    lbm_src_dir = LBM_PATH / "src"
    if not lbm_src_dir.exists():
        raise FileNotFoundError(f"LBM src directory not found: {lbm_src_dir}")

    # Find the makefile.macos in libs/pylbm
    _project_root = pathlib.Path(__file__).parent.parent.parent
    makefile_macos = _project_root / "makefile.macos"

    if not makefile_macos.exists():
        raise FileNotFoundError(f"makefile.macos not found at {makefile_macos}")

    # Get the pixi environment path
    # Find repo root
    _repo_root = _project_root.parent.parent
    while _repo_root != _repo_root.parent:
        if (_repo_root / ".git").exists() or (_repo_root / ".gitmodules").exists():
            break
        _repo_root = _repo_root.parent

    # Identify the current pixi environment
    pixi_env_path = identify_environment(_repo_root)

    # Read makefile.macos and update HOME path
    makefile_content = makefile_macos.read_text()

    # Replace HOME line with absolute path to pixi environment
    # Also remove -march=native to avoid "illegal hardware instruction" errors
    # This flag optimizes for the build CPU but can cause compatibility issues
    # Also update NCFDIR for NETCDF support when needed
    lines = makefile_content.splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("HOME = ") and not line.strip().startswith("#"):
            # Update HOME to absolute path
            new_lines.append(f"HOME = {pixi_env_path}")
        elif line.strip().startswith("NCFDIR = ") and not line.strip().startswith("#"):
            # Update NCFDIR to point to pixi environment root
            # In conda/pixi, netcdf-fortran is installed directly in include/ and lib/
            if enable_netcdf:
                new_lines.append(f"NCFDIR = {pixi_env_path}")
                print(
                    f"Updated NCFDIR to {pixi_env_path} for NETCDF support",
                    file=sys.stderr,
                )
            else:
                new_lines.append(line)
        elif "-march=native" in line:
            # Remove -march=native to ensure portability
            # This flag causes "illegal hardware instruction" errors when the executable
            # is run on a different CPU than it was compiled on
            new_line = line.replace("-march=native", "").replace("  ", " ")
            # Clean up multiple spaces but preserve line continuation
            if new_line.strip().endswith("\\"):
                new_lines.append(new_line.rstrip())
            else:
                new_lines.append(new_line.rstrip())
            print(f"Removed -march=native flag for portability", file=sys.stderr)
        else:
            new_lines.append(line)

    # Write modified makefile to LBM/src/makefile
    lbm_makefile = lbm_src_dir / "makefile"
    lbm_makefile.write_text("\n".join(new_lines) + "\n")
    print(f"Copied and updated makefile.macos to {lbm_makefile}", file=sys.stderr)
    print(f"Set HOME to {pixi_env_path}", file=sys.stderr)

    # Set up environment variables
    env = os.environ.copy()
    env["HOME"] = str(pixi_env_path)

    # Also set PIXI_ENVIRONMENT if not already set
    if "PIXI_ENVIRONMENT" not in env:
        env["PIXI_ENVIRONMENT"] = str(pixi_env_path)

    # Change to LBM/src directory and compile
    # According to LBM docs, compilation should be done from src directory
    original_cwd = pathlib.Path.cwd()

    try:
        os.chdir(lbm_src_dir)
        print(f"Changed to directory: {lbm_src_dir}", file=sys.stderr)

        # First, regenerate source.files and depends.file to include m_runcase.F90
        print("Regenerating source.files and depends.file...", file=sys.stderr)
        make_args = ["make", "source", "depend", "GFORTRAN=1"]
        if enable_netcdf:
            make_args.append("NETCDF=1")
        result_source: subprocess.CompletedProcess[str] = subprocess.run(  # type: ignore[call-overload]
            make_args,
            env=env,
            stderr=stderr,
            stdout=stdout,
            text=True,
        )

        if result_source.returncode != 0:
            print(
                f"Warning: Failed to regenerate source files (code {result_source.returncode})",
                file=sys.stderr,
            )
            if result_source.stderr:
                print(result_source.stderr, file=sys.stderr)
        else:
            print("Source files regenerated successfully", file=sys.stderr)

        # Manually add dependency: main.o depends on m_{case_name}.o
        # This ensures make compiles the case module before main
        depends_file = lbm_src_dir / "depends.file"
        if depends_file.exists():
            depends_content = depends_file.read_text()
            dependency_line = f"main.o:\tm_{case_name}.o\n"
            if dependency_line not in depends_content:
                # Append the dependency if it's not already there
                depends_file.write_text(depends_content + dependency_line)
                print(
                    f"Added dependency: main.o depends on m_{case_name}.o",
                    file=sys.stderr,
                )

        print(f"Compiling LBM program...", file=sys.stderr)

        # First, explicitly compile the case module to generate the .mod file
        # This ensures the module file exists before main.F90 tries to use it
        case_module_file = lbm_src_dir / f"m_{case_name}.F90"
        build_dir = LBM_PATH / "build"

        # Determine the actual module name from the file
        actual_module_name = f"m_{case_name}"  # Default
        if case_module_file.exists():
            # Read the first few lines to find the module name
            try:
                content = case_module_file.read_text()
                for line in content.splitlines()[:10]:
                    if line.strip().startswith("module "):
                        actual_module_name = line.strip().split()[1].strip()
                        break
            except Exception as e:
                print(
                    f"Warning: Could not read module name from {case_module_file}: {e}",
                    file=sys.stderr,
                )

        if case_module_file.exists():
            print(f"Compiling {case_module_file.name} first...", file=sys.stderr)
            make_args = ["make", "-B", f"m_{case_name}.o", "GFORTRAN=1"]
            if enable_netcdf:
                make_args.append("NETCDF=1")
            result_case: subprocess.CompletedProcess[str] = subprocess.run(  # type: ignore[call-overload]
                make_args,
                env=env,
                stderr=stderr,
                stdout=stdout,
                text=True,
            )

            # Always print output to see what happened
            if result_case.stdout:
                print("STDOUT:", file=sys.stderr)
                print(result_case.stdout, file=sys.stderr)
            if result_case.stderr:
                print("STDERR:", file=sys.stderr)
                print(result_case.stderr, file=sys.stderr)

            if result_case.returncode != 0:
                raise RuntimeError(
                    f"Failed to compile {case_module_file.name}: {result_case.stderr}"
                )

            # Verify that the .mod file was created
            # Use the actual module name, not necessarily m_{case_name}
            mod_file = build_dir / f"{actual_module_name}.mod"
            if not mod_file.exists():
                # Check if there are any .mod files in the build directory
                mod_files = list(build_dir.glob("*.mod")) if build_dir.exists() else []
                print(
                    f"Available .mod files in {build_dir}: {[f.name for f in mod_files]}",
                    file=sys.stderr,
                )

                # Try to see if compilation actually happened - check for .o file
                o_file = build_dir / f"m_{case_name}.o"
                if o_file.exists():
                    print(
                        f"Note: {o_file.name} exists but {mod_file.name} does not",
                        file=sys.stderr,
                    )
                    print(
                        f"Expected module name: {actual_module_name} (from file: {case_module_file.name})",
                        file=sys.stderr,
                    )
                    print(
                        "This suggests the module name in the file doesn't match the expected name",
                        file=sys.stderr,
                    )

                raise RuntimeError(
                    f"Module file {mod_file} was not created after compiling {case_module_file.name}. "
                    f"Expected module name: {actual_module_name}. "
                    f"Compilation may have failed silently. Check the output above for errors."
                )

            print(
                f"{case_module_file.name} compiled successfully, {mod_file.name} created",
                file=sys.stderr,
            )
        else:
            raise FileNotFoundError(f"Case module file not found: {case_module_file}")

        # Now compile everything else
        # Don't use -B flag here since we've already compiled m_runcase.F90
        # Make will handle dependencies correctly now
        print("Compiling remaining files...", file=sys.stderr)
        make_args = ["make", "GFORTRAN=1"]
        if enable_netcdf:
            make_args.append("NETCDF=1")
        result: subprocess.CompletedProcess[str] = subprocess.run(  # type: ignore[call-overload]
            make_args,
            env=env,
            stderr=stderr,
            stdout=stdout,
            text=True,
        )

        if result.returncode != 0:
            print(
                f"Compilation failed with return code {result.returncode}",
                file=sys.stderr,
            )
            if result.stdout:
                print("STDOUT:", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print("STDERR:", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"LBM compilation failed: {result.stderr}")

        print("Compilation successful", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)

        # Check if infile.in was created in the src directory
        infile_in_src = lbm_src_dir / "infile.in"
        infile_in_rundir = rundir / "infile.in"

        if infile_in_src.exists():
            # Move infile.in to rundir
            shutil.move(str(infile_in_src), str(infile_in_rundir))
            print(f"Moved infile.in to {infile_in_rundir}", file=sys.stderr)
        else:
            # If infile.in wasn't created during compilation, it will be created
            # when the program runs. But let's check if it exists in rundir already
            if not infile_in_rundir.exists():
                print(
                    f"Warning: infile.in was not created during compilation. "
                    f"It will be created when the program runs.",
                    file=sys.stderr,
                )

    finally:
        # Always return to original directory
        os.chdir(original_cwd)


def create_infile(rundir: pathlib.Path | None = None, verbose: bool = True) -> None:
    """
    Create infile.in template file in the rundir.

    Instead of running the executable (which may crash), we create the file
    directly based on the template from m_mkinfile.F90.

    Args:
        rundir: Directory where to create infile.in. If None, uses .temp/lbm
    """
    if rundir is None:
        rundir = pathlib.Path.cwd() / ".temp" / "lbm"

    # Convert to absolute path if relative
    if not rundir.is_absolute():
        rundir = pathlib.Path.cwd() / rundir

    # Ensure rundir exists
    rundir.mkdir(parents=True, exist_ok=True)

    infile_path = rundir / "infile.in"

    # If infile.in already exists, we don't need to create it
    if infile_path.exists():
        print(f"infile.in already exists at {infile_path}", file=sys.stderr)
        return

    print(f"Creating infile.in template at {infile_path}...", file=sys.stderr)

    # Create the infile.in template based on m_mkinfile.F90
    # This is the exact template that the program would generate
    infile_content = """# Development variables
 T                ! ltiming       : CPU timing
 F                ! ltesting      : testing final solution while developing code.
 F                ! lnodump       : No saving of diagnostics or restarts while optimizing GPU code
 256 1 1          ! ntx, nty, ntz : number of threads per block in x, y, and z direction
 F                ! lmeasurements : Saves predicted measurements for data assimilation experiments
# Experiment configuration
 runcase          ! experiment    : Cylinder, airfoil etc
 3                ! ibgk          : BGK order, ibgk=2 standard second order, ibgk=3 gives third order BGK
 1                ! ihrr          : Collision operator ihrr=1 includes regularization by 3rd ord Hermite
 1 0.15           ! ivreman smagor: Vreman subgridscale mixing (1) with Smagorinsky constant (0.15)
# Time stepping and outputs
 00000            ! nt0           : First timestep
 1000             ! nt1           : Final timestep
 5000             ! iout          : Number of steps between diag outputs
 10000            ! irestarts     : Number of steps between restart outputs
 00 60000 1       ! iprt1 iprt2 x : Output every x timestep for it <= iprt1 and it >= iprt2
 3                ! tecout        : full tecplot files (0), only solution (2), netcdf (3)
# Boundary conditions
 1                ! ibnd          : 0-periodic, 1 in/out flow,
 0                ! jbnd          : 0-periodic, 11,12,21,22 no-slip bb(1), free-slip bb(2) for j=1 and j=ny
 22               ! kbnd          : 0-periodic, 11,12,21,22 no-slip bb(1), free-slip bb(2) for k=1 and k=nz
# Inflow variables
 8.0 0.0          ! uini, udir    : Inflow wind velocity [m/s], direction in degrees (-45:45)
 F 0.00005  100   ! lturb amp nrtu: Add turbulence forcing on inflow, amplitude, number of prestored time ste
# Physical variables
 0.0000178        ! visckin       : Dimensional kinematic viscosity
 1.225            ! C_rho - Density of air at surface 15C and  101.325 kPa  [kg/m^3] Eq. (7.12)
 1.0              ! C_l   - Length of a lattice cell in meters [m]
 75.0             ! C_u   - Wind velocity conversion [m/s]   -> C_t=C_l/C_u
# Averaging variables
 F F              ! lave lavesec  : Switch on/off full averaging and turbine section averaging
 20000            ! avestart      : Iteration to start computing section averages
 40000            ! avestop       : Iteration to save section averages
# Turbine-definitions
 0                ! nturbines     : Number of turbines
 0.0              ! pitchangle    : Imposed pitchangle (0 until u=11.4, see table 7.1 in NREL doc).
 8.95             ! turbrpm       : Turbine RPM for actuator line model (max 12.1 9.22 8.95 12.06
 0.00             ! tipspeed ratio: Tipspeed ratio (7.55) (if given will override the given turbine RPM
 0                ! itiploss      : Tiploss(0-none, 1-Prandl, 2-Shen)
# T1
 96               ! ipos          : i-location turbine one
 61               ! jpos          : j-location turbine one
 61               ! kpos          : k-location turbine one
# T2
 64               ! ipos          : i-location turbine two
 75               ! jpos          : j-location turbine two
 48               ! kpos          : k-location turbine two


 kinematic viscosity of air is 1.78E-5
 Reynolds number becomes Re= u D /nu = 10^1 * 10^2 / 10^(-5)  = 10^7 - 10^8
"""

    # Write the file
    infile_path.write_text(infile_content)
    print(f"Successfully created infile.in at {infile_path}", file=sys.stderr)
