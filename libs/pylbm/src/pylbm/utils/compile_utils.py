"""
Compile the LBM program.
"""

import logging
import os
import pathlib
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

from .dir_utils import DirectoryPaths
from .makefile_utils import Makefile


def _resolve_build_environment(
    dirs: DirectoryPaths,
    enable_netcdf: bool,
    enable_cuda: bool = False,
) -> pathlib.Path:
    """
    Resolve the environment path used for compilation.

    If NETCDF is enabled, prefer an environment that actually contains
    include/netcdf.mod so the Fortran compiler can resolve `use netcdf`.
    """
    preferred_env = pathlib.Path(dirs.pixi_env_path)
    if not enable_netcdf:
        return preferred_env

    # CUDA + NETCDF requires compiler-compatible modules. Keep the active env.
    if enable_cuda:
        return preferred_env

    if (preferred_env / "include" / "netcdf.mod").exists():
        return preferred_env

    # Fallback to known pixi envs in this repository.
    repo_envs = dirs.cwd / ".pixi" / "envs"
    if repo_envs.exists():
        for env_name in ["delftblue", "dev", "default"]:
            candidate = repo_envs / env_name
            if (candidate / "include" / "netcdf.mod").exists():
                logger.warning(
                    "Selected NETCDF-capable environment '%s' because '%s' "
                    "does not contain include/netcdf.mod",
                    candidate,
                    preferred_env,
                )
                return candidate

    return preferred_env


def _probe_netcdf_module(
    include_dir: pathlib.Path,
    env: dict[str, str],
) -> bool:
    """Check whether nvfortran can consume netcdf.mod from include_dir."""
    probe_dir = include_dir.parent / ".netcdf_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_src = probe_dir / "probe_netcdf.F90"
    probe_obj = probe_dir / "probe_netcdf.o"
    probe_src.write_text(
        "program probe_netcdf\nuse netcdf\nimplicit none\nprint *, 'ok'\nend program\n"
    )
    result = subprocess.run(
        [
            "nvfortran",
            "-c",
            str(probe_src),
            "-I",
            str(include_dir),
            "-o",
            str(probe_obj),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.returncode == 0


def _ensure_cuda_netcdf_fortran(
    dirs: DirectoryPaths,
    build_env_path: pathlib.Path,
    env: dict[str, str],
) -> pathlib.Path:
    """
    Ensure an NVFORTRAN-compatible netcdf-fortran installation exists.

    Returns the prefix to use as NCFDIR.
    """
    override_root = os.environ.get("NETCDF_FORTRAN_ROOT")
    netcdf_root = (
        pathlib.Path(override_root)
        if override_root is not None
        else build_env_path / ".nvhpc" / "netcdf-fortran"
    )
    include_dir = netcdf_root / "include"
    lib_dir = netcdf_root / "lib"

    has_install = (include_dir / "netcdf.mod").exists() and (
        (lib_dir / "libnetcdff.so").exists() or (lib_dir / "libnetcdff.a").exists()
    )
    if has_install and _probe_netcdf_module(include_dir=include_dir, env=env):
        return netcdf_root

    install_script = dirs.cwd / "activation_scripts" / "install_nvhpc_netcdf.sh"
    if not install_script.exists():
        raise RuntimeError(
            "Missing activation script for NVHPC-compatible netcdf-fortran: "
            f"{install_script}"
        )

    install_env = env.copy()
    install_env["CONDA_PREFIX"] = str(build_env_path)
    install_env["NVHPC_NETCDF_PREFIX"] = str(netcdf_root)
    install_env["NETCDF_C_PREFIX"] = str(build_env_path)
    result = subprocess.run(
        ["bash", str(install_script)],
        env=install_env,
        stdout=None,
        stderr=None,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install NVHPC-compatible netcdf-fortran. "
            "See logs above for details."
        )

    if not (include_dir / "netcdf.mod").exists() or not (
        (lib_dir / "libnetcdff.so").exists() or (lib_dir / "libnetcdff.a").exists()
    ):
        raise RuntimeError(
            f"NVHPC netcdf-fortran install is incomplete at {netcdf_root}"
        )
    if not _probe_netcdf_module(include_dir=include_dir, env=env):
        raise RuntimeError(
            "Installed netcdf.mod is still incompatible with nvfortran. "
            f"Checked include path: {include_dir}"
        )
    return netcdf_root


def compile_lbm(
    dirs: DirectoryPaths,
    verbose: bool = True,
    enable_netcdf: bool = True,
    enable_cuda: bool = False,
) -> None:
    """
    Compile the LBM program.

    This function:
    1. Updates the HOME path in the makefile to the pixi environment path
    2. Changes to the LBM src directory
    3. Runs make to compile the program

    Args:
        dirs: DirectoryPaths object containing all relevant paths (including lbm_src_path,
              makefile_path, and pixi_env_path).
        verbose: If True, print compilation output. If False, suppress output.
        enable_netcdf: If True, enable NETCDF compilation flag.
        enable_cuda: If True, compile with CUDA=1 (NVFORTRAN).

    Raises:
        FileNotFoundError: If makefile or lbm_src_path doesn't exist.
        RuntimeError: If compilation fails.
    """
    if not dirs.makefile_path.exists():
        raise FileNotFoundError(f"Makefile not found at {dirs.makefile_path}")

    if not dirs.lbm_src_path.exists():
        raise FileNotFoundError(f"LBM src directory not found at {dirs.lbm_src_path}")

    build_env_path = _resolve_build_environment(
        dirs=dirs, enable_netcdf=enable_netcdf, enable_cuda=enable_cuda
    )

    # Set up environment variables
    env = os.environ.copy()
    env["HOME"] = str(build_env_path)
    if "PIXI_ENVIRONMENT" not in env:
        env["PIXI_ENVIRONMENT"] = str(build_env_path)
    env_lib_dir = build_env_path / "lib"

    if enable_cuda:
        nvhpc_install_base = build_env_path / ".nvhpc"
        nvfortran_candidates = sorted(
            nvhpc_install_base.glob("Linux_x86_64/*/compilers/bin/nvfortran")
        )
        if not nvfortran_candidates:
            raise RuntimeError(
                "CUDA build requested but NVFORTRAN was not found in "
                f"{nvhpc_install_base}. Activate the cuda Pixi environment first "
                "so NVHPC is installed."
            )

        nvfortran_bin_dir = nvfortran_candidates[-1].parent
        nvhpc_root = nvfortran_bin_dir.parents[1]
        path_parts = [str(nvfortran_bin_dir)]

        mpi_bin = nvhpc_root / "comm_libs" / "mpi" / "bin"
        if mpi_bin.exists():
            path_parts.append(str(mpi_bin))

        if env.get("PATH"):
            path_parts.append(env["PATH"])
        env["PATH"] = ":".join(path_parts)
        env["NVCOMPILERS"] = str(nvhpc_install_base)

        ld_parts: list[str] = []
        for lib_dir in [
            nvhpc_root / "compilers" / "lib",
            nvhpc_root / "math_libs" / "lib64",
            nvhpc_root / "comm_libs" / "mpi" / "lib",
        ]:
            if lib_dir.exists():
                ld_parts.append(str(lib_dir))
        if env.get("LD_LIBRARY_PATH"):
            ld_parts.append(env["LD_LIBRARY_PATH"])
        if ld_parts:
            env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    # Ensure linker/runtime can resolve conda-forge libs like fftw3f and netcdf.
    if env_lib_dir.exists():
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{env_lib_dir}:{existing_ld}" if existing_ld else str(env_lib_dir)
        )

    netcdf_root = build_env_path
    if enable_cuda and enable_netcdf:
        netcdf_root = _ensure_cuda_netcdf_fortran(
            dirs=dirs,
            build_env_path=build_env_path,
            env=env,
        )

    # Change to LBM src directory and run make
    original_cwd = pathlib.Path.cwd()
    stdout = sys.stdout if verbose else subprocess.DEVNULL
    stderr = sys.stderr if verbose else subprocess.DEVNULL

    try:
        # Update makefile HOME path to pixi environment
        makefile = Makefile(dirs.makefile_path)
        makefile.set_path("HOME", build_env_path)

        # Update NCFDIR to whichever NETCDF root we are actually using.
        if enable_netcdf:
            makefile.set_path("NCFDIR", netcdf_root)

        makefile.write()

        if verbose:
            logger.info("Updated makefile HOME to %s", build_env_path)
            if enable_netcdf:
                logger.info("Using NETCDF root: %s", netcdf_root)

        os.chdir(dirs.lbm_src_path)

        if verbose:
            logger.info("Changed to directory: %s", dirs.lbm_src_path)
            logger.info("Compiling LBM program...")

        # Build make command
        link_dirs: list[str] = []
        if enable_netcdf:
            netcdf_lib_dir = netcdf_root / "lib"
            if netcdf_lib_dir.exists():
                # Put NVHPC-built netcdf-fortran first so -lnetcdff does not
                # resolve to the incompatible conda-forge shared library.
                link_dirs.append(f"-L{netcdf_lib_dir}")
        link_dirs.append(f"-L{env_lib_dir}")

        # Install the binary into the LBM tree (LBM/bin) instead of the makefile's
        # default $(HOME)/bin (the shared pixi env). Keeps the executable beside
        # its -- possibly per-run isolated -- source tree so parallel builds don't
        # clobber a single shared boltzmann. Must match dir_utils.executable_path.
        bindir = dirs.lbm_src_path.parent / "bin"
        make_args = [
            "make",
            "-B",
            f"HOME={build_env_path}",
            f"BINDIR={bindir}",
            f"LIBDIR={' '.join(link_dirs)}",
        ]
        if enable_cuda:
            make_args.append("CUDA=1")
        else:
            make_args.append("GFORTRAN=1")
        if enable_netcdf:
            make_args.extend(["NETCDF=1", f"NCFDIR={netcdf_root}"])
            if enable_cuda:
                # netcdf-fortran is static in CUDA mode; it depends on libnetcdf.
                # Keep order so dependent C library appears after netcdff.
                make_args.append("LIBS=-lfftw3f -lnetcdff -lnetcdf")

        result = subprocess.run(  # type: ignore[call-overload]
            make_args,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"LBM compilation failed with exit code {result.returncode}. "
                f"See output above for details."
            )

        if verbose:
            logger.info("LBM compilation completed successfully")

    finally:
        # Always return to original directory
        os.chdir(original_cwd)
