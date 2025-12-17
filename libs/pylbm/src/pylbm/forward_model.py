import os
import pathlib
import pdb
import subprocess
import sys

from . import LBM_PATH
from .compile_program import compile_lbm, create_infile, identify_environment
from .stl_to_lbm import stl_to_lbm_geometry
from .write_to_fortran import add_case_dimensions_to_mod_dimensions, add_module_to_main


class ForwardModel:
    def __init__(
        self,
        rundir: pathlib.Path | None = None,
        stl_path: str | pathlib.Path | None = None,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
    ):
        # Default rundir is .temp/lbm relative to current working directory
        if rundir is None:
            rundir = pathlib.Path(".temp/lbm")

        # Convert to absolute path if relative
        if not rundir.is_absolute():
            rundir = pathlib.Path.cwd() / rundir

        self.rundir = rundir
        # Ensure rundir exists
        self.rundir.mkdir(parents=True, exist_ok=True)

        # Path to infile.in in rundir
        self.infile_path = self.rundir / "infile.in"

        # Case name
        self.case_name = "runcase"

        # Generate geometry file
        stl_to_lbm_geometry(
            stl_path=stl_path,  # type: ignore[arg-type]
            output_path=LBM_PATH / "src" / f"m_{self.case_name}.F90",  # type: ignore[operator]
            module_name=f"m_{self.case_name}",
            subroutine_name=self.case_name,
            nx=nx,  # type: ignore[arg-type]
            ny=ny,  # type: ignore[arg-type]
            nz=nz,  # type: ignore[arg-type]
        )

        # Add module to main.F90
        add_module_to_main(self.case_name)

        # Set case dimensions
        add_case_dimensions_to_mod_dimensions(self.case_name, nx, ny, nz)  # type: ignore[arg-type]

        # Compile program
        compile_lbm(rundir=self.rundir, case_name=self.case_name)

        # Create infile.in by running the executable
        create_infile(rundir=self.rundir)

    def run(self) -> None:
        """
        Run the LBM executable from the rundir.

        This executes the compiled boltzmann program from self.rundir,
        which will read infile.in and run the simulation.
        """
        # Find the pixi environment path to locate the executable
        _project_root = pathlib.Path(__file__).parent.parent.parent
        _repo_root = _project_root.parent.parent
        while _repo_root != _repo_root.parent:
            if (_repo_root / ".git").exists() or (_repo_root / ".gitmodules").exists():
                break
            _repo_root = _repo_root.parent

        # Identify the current pixi environment
        pixi_env_path = identify_environment(_repo_root)
        executable_path = pixi_env_path / "bin" / "boltzmann"

        if not executable_path.exists():
            raise FileNotFoundError(
                f"Executable not found at {executable_path}. "
                f"Make sure the program has been compiled successfully."
            )

        # Ensure infile.in exists
        if not self.infile_path.exists():
            raise FileNotFoundError(
                f"infile.in not found at {self.infile_path}. "
                f"Make sure create_infile() has been called."
            )

        print(f"Running LBM simulation from {self.rundir}...", file=sys.stderr)
        print(f"Executable: {executable_path}", file=sys.stderr)

        original_cwd = pathlib.Path.cwd()

        try:
            os.chdir(self.rundir)

            # Set up environment
            env = os.environ.copy()
            env["HOME"] = str(pixi_env_path)
            if "PIXI_ENVIRONMENT" not in env:
                env["PIXI_ENVIRONMENT"] = str(pixi_env_path)

            # Set stack size limit to unlimited
            # shell_cmd = f"ulimit -s unlimited && {executable_path}"
            shell_cmd = f"{executable_path}"
            result = subprocess.run(
                shell_cmd,
                shell=True,
                env=env,
                stderr=subprocess.PIPE,  # Capture stderr to see segfault details
                stdout=subprocess.PIPE,  # Also capture stdout to see any error messages
                text=True,
            )

            if result.returncode != 0:
                error_msg = (
                    f"LBM simulation failed with return code {result.returncode}"
                )
                if result.returncode == -11:
                    error_msg += " (Segmentation fault - likely stack overflow)"
                    error_msg += "\nTry setting ulimit -s unlimited before running"
                if result.stderr:
                    error_msg += (
                        f"\nSTDERR:\n{result.stderr[-2000:]}"  # Last 2000 chars
                    )
                raise RuntimeError(error_msg)

            print(f"Simulation completed successfully", file=sys.stderr)

        finally:
            # Always return to original directory
            os.chdir(original_cwd)
