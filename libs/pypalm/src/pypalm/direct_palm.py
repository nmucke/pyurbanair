"""Direct PALM runner — bypass palmrun and palmbuild for per-member runs.

Replaces the ``palmrun → palmbuild → mpirun palm → mpirun combine_plot_fields``
chain (~134 s per invocation) with a single in-process Python sequence:

  1. Stage INPUT files into a per-run tempdir under ``fast_io_catalog``.
  2. Write ENVPAR (same Fortran namelist palmrun would have written).
  3. ``mpirun -n <ncpu> ./palm`` to run the solver.
  4. ``./combine_plot_fields.x`` (no mpirun) to fill in the 3D netCDF.
  5. Move ``DATA_3D_NETCDF`` + a small allow-listed set of outputs to OUTPUT/.

The prebuilt ``palm`` and ``combine_plot_fields.x`` are reused via symlink from
``palm_model_system/MAKE_DEPOSITORY_default`` (built once by ``prepare_compile``).

This module is exercised standalone by
``job_scripts/delftblue/pypalm/m1_direct_run.py``. Wiring into
``ForwardModel.run()`` lands in M2 behind ``PYPALM_USE_DIRECT_RUN``.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

from . import PALM_MODEL_SYSTEM_PATH
from .utils.dir_utils import PALMDirectoryPaths

logger = logging.getLogger(__name__)


PALM_BINARY = PALM_MODEL_SYSTEM_PATH / "MAKE_DEPOSITORY_default" / "palm"
COMBINE_BINARY = PALM_MODEL_SYSTEM_PATH / "MAKE_DEPOSITORY_default" / "combine_plot_fields.x"

# Mapping from pypalm's ``INPUT/<run>_<suffix>`` files to PALM's tempdir
# filenames (right-hand side, matching the ``in:tr`` rules in .palm.iofiles).
# Suffixes pypalm doesn't produce today (_static, _dynamic when not time-varying,
# etc.) are simply skipped when the source file is absent.
INPUT_FILE_MAP: dict[str, str] = {
    "_p3d": "PARIN",
    "_topo": "TOPOGRAPHY_DATA",
    "_dynamic": "PIDS_DYNAMIC",
    "_static": "PIDS_STATIC",
}

# Output files we care about and how to rename them when moving to OUTPUT/.
# Right-hand side becomes ``<run><suffix>.nc``. Matches the ``out:tr`` rules
# in .palm.iofiles for the formats we actually use today (3D + 1D-TS).
OUTPUT_FILE_MAP: dict[str, str] = {
    "DATA_3D_NETCDF": "_3d.nc",
    "DATA_1D_TS_NETCDF": "_ts.nc",
}


def _augment_runtime_library_paths(env: dict[str, str]) -> None:
    """Prepend the active pixi/conda lib dirs to LD_LIBRARY_PATH.

    Duplicated from ``ForwardModel._augment_runtime_library_paths`` so
    ``run_direct`` is self-contained when called standalone (M1 unit test,
    future direct-only drivers). When the caller already augmented its env
    this is a no-op (the same paths get re-prepended).
    """
    lib_paths: list[pathlib.Path] = []
    for key in ("CONDA_PREFIX", "PIXI_ENVIRONMENT"):
        prefix = env.get(key)
        if not prefix:
            continue
        lib_dir = pathlib.Path(prefix) / "lib"
        if lib_dir.exists():
            lib_paths.append(lib_dir)
    if not lib_paths:
        return
    existing = env.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(str(p) for p in lib_paths)
    env["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


def _stack_limited(argv: list[str]) -> list[str]:
    """Wrap ``argv`` in a shell that raises the stack limit before exec.

    PALM allocates large automatic (stack) arrays during model initialization;
    on big grids (e.g. the Barcelona case at 400x400x32) these overflow the
    default 8 MB soft stack and ``palm`` dies with SIGSEGV before time-stepping.
    palmrun raises this via ``ulimit -s unlimited``, but the direct path
    bypasses palmrun, so do it here. The raised limit is inherited by mpirun and
    its ``palm`` child. ``unlimited`` succeeds on Linux; macOS refuses it (and
    Python's ``resource.setrlimit`` can't raise ``RLIMIT_STACK`` at all there),
    so fall back to the hard cap (~64 MB) via ``ulimit -s hard``. Both attempts
    are best-effort — a failure to raise must never abort the launch.
    """
    inner = "exec " + shlex.join(argv)
    return [
        "bash",
        "-c",
        f"ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null; {inner}",
    ]


@dataclass
class DirectRunResult:
    """Timing / status for a single direct-run invocation. Returned for
    structured logging and so M3 can compare phase costs against the
    palmrun baseline."""

    tempdir: pathlib.Path
    stage_s: float
    palm_s: float
    combine_s: float
    transfer_s: float
    total_s: float
    palm_rc: int
    combine_rc: int
    output_files: list[str]


def _envpar_text(
    run_identifier: str, host: str, ncpu: int, progress_bar_disabled: bool = False
) -> str:
    """Render the ENVPAR Fortran namelist that PALM reads at startup.

    Mirrors palmrun's heredoc at
    ``palm_model_system/packages/palm/model/bin/palmrun`` (search for
    ``cat  >  ENVPAR``). The SVF/restart/spinup booleans are ``.false.`` because
    pypalm doesn't use those writes. ``progress_bar_disabled`` is flipped on for
    quiet (``verbose=False``) runs so PALM doesn't stream its progress bar.
    """
    pbar = ".true." if progress_bar_disabled else ".false."
    return (
        f" &envpar  run_identifier = '{run_identifier}', host = '{host}',\n"
        "          write_svf = .false., write_binary = .false., write_spinup_data = .false.,\n"
        f"          read_svf = .false., tasks_per_node = {ncpu},\n"
        "          maximum_parallel_io_streams = 1,\n"
        "          maximum_cpu_time_allowed = 10000000.,\n"
        "          version_string = 'PALM 25.10',\n"
        f"          progress_bar_disabled = {pbar}, /\n"
    )


def _stage_inputs(
    src_input: pathlib.Path,
    dst_tempdir: pathlib.Path,
    experiment_name: str,
) -> list[str]:
    staged: list[str] = []
    for suffix, target_name in INPUT_FILE_MAP.items():
        src = src_input / f"{experiment_name}{suffix}"
        if not src.exists():
            continue
        shutil.copy2(src, dst_tempdir / target_name)
        staged.append(target_name)
    return staged


def _link_binaries(dst_tempdir: pathlib.Path) -> None:
    for binary in (PALM_BINARY, COMBINE_BINARY):
        if not binary.exists():
            raise FileNotFoundError(
                f"Prebuilt PALM binary missing at {binary}. Run pypalm "
                f"compile_palm first (Hydra: model.compile=true)."
            )
        link = dst_tempdir / binary.name
        # symlink is enough — both binaries are dynamically linked and read
        # config/data from cwd. Avoids the per-member 750-file copy palmrun does.
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(binary)

    # PALM's `objdump -p palm` shows `NEEDED rrtmg/rrtmg.so` — a path-with-slash
    # entry that ld.so resolves *relative to the runtime CWD*, not RPATH or
    # LD_LIBRARY_PATH. Symlink the rrtmg/ subdir into the tempdir so
    # `./rrtmg/rrtmg.so` resolves when palm starts.
    rrtmg_subdir = PALM_BINARY.parent / "rrtmg"
    rrtmg_link = dst_tempdir / "rrtmg"
    if rrtmg_subdir.is_dir() and not rrtmg_link.exists():
        rrtmg_link.symlink_to(rrtmg_subdir, target_is_directory=True)

    # combine_plot_fields.x is linked against a *bare leaf* `rrtmg.so` (no
    # subdir): on macOS `otool -L` shows `rrtmg.so`, which dyld resolves only
    # against CWD and DYLD_*_PATH — and the `rrtmg/` subdir symlink above does
    # not satisfy it. Without this, combine exits on signal 9 (dyld "Library
    # not loaded: rrtmg.so"), the combined `_3d.nc` is never written, and
    # pypalm silently falls back to the zero-filled per-PE skeleton (see
    # docs/pypalm_zero_field_debug.md). Provide `./rrtmg.so` in the tempdir too.
    rrtmg_so = rrtmg_subdir / "rrtmg.so"
    rrtmg_so_link = dst_tempdir / "rrtmg.so"
    if rrtmg_so.is_file() and not rrtmg_so_link.exists():
        rrtmg_so_link.symlink_to(rrtmg_so)


def _transfer_outputs(
    tempdir: pathlib.Path,
    output_dir: pathlib.Path,
    experiment_name: str,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for src_name, suffix in OUTPUT_FILE_MAP.items():
        src = tempdir / src_name
        if not src.exists():
            continue
        # palmrun emits "<run>_3d.000.nc" even at single-PE; keep the same name
        # so pypalm's ``_locate_3d_output`` glob still hits.
        local_dot_pe = ".000" if suffix == "_3d.nc" else ""
        target = output_dir / f"{experiment_name}{suffix[:-3]}{local_dot_pe}.nc"
        shutil.copy2(src, target)
        saved.append(target.name)
    return saved


def run_direct(
    dirs: PALMDirectoryPaths,
    experiment_name: str,
    ncpu: int = 1,
    host: str = "default",
    env: Optional[dict[str, str]] = None,
    keep_tempdir: bool = False,
    extra_mpirun_args: Optional[list[str]] = None,
    verbose: bool = True,
) -> DirectRunResult:
    """Run one PALM invocation directly, bypassing palmrun/palmbuild.

    Mirrors what palmrun does for a single d3 run on a single PE:
    stage INPUT, generate ENVPAR, ``mpirun -n <ncpu> ./palm``,
    ``./combine_plot_fields.x``, transfer DATA_3D_NETCDF to OUTPUT/.

    When ``verbose`` is False the ``palm`` and ``combine_plot_fields.x`` stdout
    /stderr are captured rather than inherited, so nothing is streamed to the
    terminal; on a non-zero exit the captured tail is logged so failures are
    still diagnosable.
    """
    capture = not verbose
    _quiet = (
        {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
        if capture
        else {}
    )
    if env is None:
        env = os.environ.copy()
    _augment_runtime_library_paths(env)

    fast_io_root_str = env.get("PYPALM_FAST_IO_CATALOG", "")
    if fast_io_root_str:
        fast_io_root = pathlib.Path(fast_io_root_str) / experiment_name
    else:
        fast_io_root = dirs.experiment_dir / "tmp"
    fast_io_root.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    tempdir = pathlib.Path(tempfile.mkdtemp(prefix=f"{experiment_name}.", dir=fast_io_root))
    logger.info("direct_palm: tempdir = %s", tempdir)

    try:
        # 1. Stage INPUT + binaries + ENVPAR
        staged = _stage_inputs(dirs.input_dir, tempdir, experiment_name)
        _link_binaries(tempdir)
        (tempdir / "ENVPAR").write_text(
            _envpar_text(experiment_name, host, ncpu, progress_bar_disabled=capture)
        )
        stage_s = time.monotonic() - t_start
        logger.info("direct_palm: staged %s in %.2fs", staged, stage_s)

        if "PARIN" not in staged:
            raise FileNotFoundError(
                f"No {experiment_name}_p3d file in {dirs.input_dir} — cannot run PALM."
            )

        # 2. mpirun -n <ncpu> ./palm (under a raised stack limit; see _stack_limited)
        # Extra mpirun args may also be supplied via PYPALM_MPIRUN_EXTRA_ARGS
        # (space-separated), e.g. "--map-by :OVERSUBSCRIBE" so OpenMPI 5's PRRTE
        # launcher will start ncpu ranks under a --ntasks=1 SLURM allocation
        # (it otherwise miscounts slots as 1 and aborts). The OMPI_MCA_rmaps_*
        # env vars do NOT work for this in OpenMPI 5 — rmaps moved to PRRTE.
        env_mpirun_args = shlex.split(os.environ.get("PYPALM_MPIRUN_EXTRA_ARGS", ""))
        mpirun_cmd = [
            "mpirun",
            *(extra_mpirun_args or []),
            *env_mpirun_args,
            "-n",
            str(ncpu),
            "./palm",
        ]
        t_palm = time.monotonic()
        palm_result = subprocess.run(
            _stack_limited(mpirun_cmd),
            cwd=tempdir,
            env=env,
            stdin=subprocess.DEVNULL,
            **_quiet,
        )
        palm_s = time.monotonic() - t_palm
        logger.info("direct_palm: palm wall=%.2fs rc=%s", palm_s, palm_result.returncode)
        if palm_result.returncode != 0:
            if capture and palm_result.stdout:
                tail = "\n".join(palm_result.stdout.splitlines()[-80:])
                logger.error(
                    "direct_palm: palm failed (rc=%s). Last output:\n%s",
                    palm_result.returncode,
                    tail,
                )
            raise subprocess.CalledProcessError(
                palm_result.returncode,
                mpirun_cmd,
                output=palm_result.stdout if capture else None,
            )

        # 3. ./combine_plot_fields.x (no mpirun — M0 confirmed this is correct)
        t_combine = time.monotonic()
        combine_result = subprocess.run(
            ["./combine_plot_fields.x"],
            cwd=tempdir,
            env=env,
            stdin=subprocess.DEVNULL,
            **_quiet,
        )
        combine_s = time.monotonic() - t_combine
        logger.info(
            "direct_palm: combine_plot_fields.x wall=%.2fs rc=%s",
            combine_s,
            combine_result.returncode,
        )
        if capture and combine_result.returncode != 0 and combine_result.stdout:
            logger.error(
                "direct_palm: combine_plot_fields.x failed (rc=%s):\n%s",
                combine_result.returncode,
                "\n".join(combine_result.stdout.splitlines()[-40:]),
            )

        # 4. Transfer outputs
        t_transfer = time.monotonic()
        output_files = _transfer_outputs(tempdir, dirs.output_dir, experiment_name)
        transfer_s = time.monotonic() - t_transfer
        logger.info("direct_palm: transferred %s in %.2fs", output_files, transfer_s)

        return DirectRunResult(
            tempdir=tempdir,
            stage_s=stage_s,
            palm_s=palm_s,
            combine_s=combine_s,
            transfer_s=transfer_s,
            total_s=time.monotonic() - t_start,
            palm_rc=palm_result.returncode,
            combine_rc=combine_result.returncode,
            output_files=output_files,
        )
    finally:
        if not keep_tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)
