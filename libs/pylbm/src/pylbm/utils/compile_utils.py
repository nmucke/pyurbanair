"""
Compile the LBM program.
"""

import logging
import os
import pathlib
import subprocess
import sys
from typing import Optional, Union

logger = logging.getLogger(__name__)

from .build_tree_utils import compute_build_signature, write_build_stamp
from .dir_utils import DirectoryPaths
from .makefile_utils import Makefile


def find_nvfortran(build_env_path: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Locate the NVHPC ``nvfortran`` this build would use, or None.

    Only the NVHPC installation the pixi cuda environment bootstraps under
    ``<env>/.nvhpc`` counts: that is the one the CUDA branch of ``compile_lbm``
    puts on PATH, so anything else on PATH would not actually be used.
    """
    candidates = sorted(
        (build_env_path / ".nvhpc").glob("Linux_x86_64/*/compilers/bin/nvfortran")
    )
    return candidates[-1] if candidates else None


_CUDA_AUTO_TOKENS = ("auto", "")
_CUDA_TRUE_TOKENS = ("true", "1", "yes", "on")
_CUDA_FALSE_TOKENS = ("false", "0", "no", "off")


def validate_cuda_setting(requested: Union[bool, str]) -> Union[bool, str]:
    """
    Reject an unusable ``cuda`` setting at construction instead of at compile time.

    Args:
        requested: The configured value.

    Returns:
        ``requested`` unchanged.

    Raises:
        ValueError: If it is a string other than auto/true/false.
    """
    if isinstance(requested, str):
        token = requested.strip().lower()
        if token not in (
            *_CUDA_AUTO_TOKENS,
            *_CUDA_TRUE_TOKENS,
            *_CUDA_FALSE_TOKENS,
        ):
            raise ValueError(
                f"cuda must be true, false, or 'auto'; got {requested!r}. "
                "'auto' uses CUDA when NVHPC is installed and gfortran otherwise."
            )
    return requested


def resolve_cuda(
    requested: Union[bool, str],
    build_env_path: pathlib.Path,
) -> bool:
    """
    Turn the configured ``cuda`` setting into a concrete build mode.

    ``"auto"`` (the shipped default) builds with CUDA where NVHPC is actually
    installed and falls back to gfortran everywhere else, so the same committed
    config runs on a GPU box and on a laptop. ``True`` still *demands* CUDA and
    lets ``compile_lbm`` raise when it is missing, which is what a GPU batch job
    wants -- silently dropping to a ~100x slower CPU build there would be worse
    than failing.

    Args:
        requested: ``True``, ``False``, or ``"auto"`` (case-insensitive).
        build_env_path: Environment prefix searched for NVHPC.

    Returns:
        True to build with CUDA/NVFORTRAN, False to build with gfortran.

    Raises:
        ValueError: If ``requested`` is a string other than auto/true/false.
    """
    validate_cuda_setting(requested)
    if isinstance(requested, str):
        token = requested.strip().lower()
        if token in _CUDA_AUTO_TOKENS:
            nvfortran = find_nvfortran(build_env_path)
            if nvfortran is not None:
                logger.info("cuda=auto: building with CUDA (found %s)", nvfortran)
                return True
            logger.info(
                "cuda=auto: no NVHPC toolchain under %s, building with gfortran "
                "(CPU). Set cuda=true to require a CUDA build instead.",
                build_env_path / ".nvhpc",
            )
            return False
        return token in _CUDA_TRUE_TOKENS
    return bool(requested)


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


def _detect_gpu_compute_capability() -> Optional[str]:
    """
    Return the host GPU compute capability without the dot (e.g. "86"), or None.

    Honors the ``PYLBM_GPU_ARCH`` override (e.g. ``PYLBM_GPU_ARCH=80``) for hosts
    where nvidia-smi is unavailable or a specific arch is wanted; otherwise queries
    nvidia-smi and uses the first GPU.
    """
    override = os.environ.get("PYLBM_GPU_ARCH")
    if override:
        cc = override.strip().replace(".", "")
        return cc if cc.isdigit() else None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as e:
        logger.warning("Could not query GPU compute capability via nvidia-smi: %s", e)
        return None
    caps = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not caps:
        return None
    cc = caps[0].replace(".", "")
    return cc if cc.isdigit() else None


def compile_lbm(
    dirs: DirectoryPaths,
    verbose: bool = True,
    enable_netcdf: bool = True,
    enable_cuda: Union[bool, str] = False,
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

    # Resolve "auto" before anything else: the rest of the build (environment
    # selection, netcdf-fortran flavour, make flags) all branch on the concrete mode.
    #
    # This probes the *active* env, while _resolve_build_environment below may pick
    # a different one. They agree today -- that function returns the active env
    # unchanged whenever CUDA is on, and only diverges on the non-CUDA netcdf.mod
    # fallback, where the NVHPC probe is irrelevant. Keep that invariant in mind if
    # its selection logic grows: the two must not disagree about where NVHPC lives.
    cuda = resolve_cuda(enable_cuda, pathlib.Path(dirs.pixi_env_path))

    build_env_path = _resolve_build_environment(
        dirs=dirs, enable_netcdf=enable_netcdf, enable_cuda=cuda
    )

    # Set up environment variables
    env = os.environ.copy()
    env["HOME"] = str(build_env_path)
    if "PIXI_ENVIRONMENT" not in env:
        env["PIXI_ENVIRONMENT"] = str(build_env_path)
    env_lib_dir = build_env_path / "lib"

    if cuda:
        nvhpc_install_base = build_env_path / ".nvhpc"
        nvfortran = find_nvfortran(build_env_path)
        if nvfortran is None:
            raise RuntimeError(
                "CUDA build requested but NVFORTRAN was not found in "
                f"{nvhpc_install_base}. Activate the cuda Pixi environment first "
                "so NVHPC is installed, or set cuda=auto to fall back to a "
                "gfortran (CPU) build on hosts without NVHPC."
            )

        nvfortran_bin_dir = nvfortran.parent
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
    if cuda and enable_netcdf:
        netcdf_root = _ensure_cuda_netcdf_fortran(
            dirs=dirs,
            build_env_path=build_env_path,
            env=env,
        )

    # Change to LBM src directory and run make
    original_cwd = pathlib.Path.cwd()
    # Capture rather than discard when quiet: a failed build otherwise raises
    # "See output above for details" with no output anywhere, which is exactly
    # what a CI log shows. Captured text is replayed into the exception below.
    stdout = sys.stdout if verbose else subprocess.PIPE
    stderr = sys.stderr if verbose else subprocess.STDOUT

    try:
        # Update makefile HOME path to pixi environment
        makefile = Makefile(dirs.makefile_path)
        makefile.set_path("HOME", build_env_path)

        # Update NCFDIR to whichever NETCDF root we are actually using.
        if enable_netcdf:
            makefile.set_path("NCFDIR", netcdf_root)

        # Retarget the GPU build to the host's compute capability. The upstream
        # makefile hardcodes a single arch (e.g. cc120) which produces a binary
        # whose device kernels are missing at run time on a different GPU.
        if cuda:
            cc = _detect_gpu_compute_capability()
            if cc is not None:
                if makefile.set_gpu_arch(cc) and verbose:
                    logger.info("Set GPU compute capability to cc%s", cc)
            else:
                logger.warning(
                    "Could not determine host GPU compute capability; building "
                    "with the makefile's default -gpu arch (set PYLBM_GPU_ARCH to "
                    "override, e.g. PYLBM_GPU_ARCH=86)."
                )

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
            f"HOME={build_env_path}",
            f"BINDIR={bindir}",
            f"LIBDIR={' '.join(link_dirs)}",
        ]
        if cuda:
            make_args.append("CUDA=1")
        else:
            make_args.append("GFORTRAN=1")
        if enable_netcdf:
            make_args.extend(["NETCDF=1", f"NCFDIR={netcdf_root}"])
            if cuda:
                # netcdf-fortran is static in CUDA mode; it depends on libnetcdf.
                # Keep order so dependent C library appears after netcdff.
                make_args.append("LIBS=-lfftw3f -lnetcdff -lnetcdf")

        # Regenerate source.files / depends.file first, on their own.
        #
        # The makefile derives both by scanning src/ (`ls *.F90` and a `use`-statement
        # crawl in bin/mkdepend.pl), and its depends.file rule deliberately *fails*
        # whenever the result changed, with ">>> Dependencies updated - please rerun
        # make". Since this wrapper rewrites m_solid_objects_init.F90's use-statements
        # per experiment, that fires on the first build in any fresh tree and aborts
        # the whole compile. Priming them in a separate throwaway invocation absorbs
        # the intended failure so the real build below starts with them up to date.
        prime = subprocess.run(
            [*make_args, "depends.file"],
            env=env,
            capture_output=True,
            text=True,
        )
        if prime.returncode != 0:
            prime_output = f"{prime.stdout or ''}{prime.stderr or ''}"
            # Only the rule's own "I regenerated them, run me again" failure is
            # expected. Anything else (no perl, unreadable mkdepend.pl, a broken
            # sed) would otherwise be swallowed here and resurface as a confusing
            # error from the real build.
            expected = any(
                marker in prime_output
                for marker in (
                    "Dependencies updated",
                    "First-time dependency generation",
                )
            )
            if expected:
                logger.debug(
                    "Dependency priming regenerated source.files/depends.file "
                    "(exit %s, as designed).",
                    prime.returncode,
                )
            else:
                logger.warning(
                    "Dependency priming pass failed unexpectedly (exit %s). The "
                    "build below may fail with a confusing error. Output:\n%s",
                    prime.returncode,
                    prime_output.strip()[-2000:],
                )

        # -B (unconditional full rebuild) is deliberate, not leftover. The
        # dependency graph is regenerated from a `use`-statement crawl that does
        # not model Fortran .mod staleness, and `mod_dimensions.F90` is a
        # compile-time input to essentially every object, so an incremental build
        # here risks linking objects from two different grids -- exactly the silent
        # wrong-output failure this module exists to prevent. Correctness over
        # build time; drop it only with real dependency tracking in the makefile.
        result = subprocess.run(  # type: ignore[call-overload]
            [*make_args, "-B"],
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

        if result.returncode != 0:
            captured = (result.stdout or "").strip()
            if captured:
                tail = "\n".join(captured.splitlines()[-40:])
                detail = f" Last 40 lines of build output:\n{tail}"
            else:
                detail = " See output above for details."
            raise RuntimeError(
                f"LBM compilation failed with exit code {result.returncode}."
                f"{detail}"
            )

        # Record what this binary was built from, so a later run with
        # compile=false can tell whether it still matches the current experiment.
        # mod_dimensions.F90 is compiled in, so a binary from another grid does
        # not fail -- it silently produces wrong-shaped or all-NaN output.
        write_build_stamp(
            build_root=dirs.lbm_src_path.parent,
            signature=compute_build_signature(
                src_path=dirs.lbm_src_path,
                experiment_name=dirs.experiment_name,
                enable_cuda=cuda,
                enable_netcdf=enable_netcdf,
            ),
        )

        if verbose:
            logger.info("LBM compilation completed successfully")

    finally:
        # Always return to original directory
        os.chdir(original_cwd)
