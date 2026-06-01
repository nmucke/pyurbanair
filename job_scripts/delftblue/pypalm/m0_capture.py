"""M0 capture driver: produces both `_3d.nc` (combine on) and the un-combined
`DATA_3D_NETCDF` (combine off via palmrun -Z), plus the per-PE binary
`PLOT3D_DATA_000000`, in a single slurm job.

Outputs three trees under ``$M0_STASH``:
  run1_combine_on/   - OUTPUT after a normal palmrun
                       + the leftover tempdir contents (kept via palmrun -B)
  run2_combine_off/  - OUTPUT after palmrun -Z -B (skip combine, keep tempdir)
  run3_combine_bare/ - tempdir layout, before/after running
                       ``./combine_plot_fields.x`` (no mpirun) on the binary
                       captured by run2; timing + rc in ``combine_bare.log``.

Driven directly via the pypalm Python API — no Hydra. Reads tiny-config
constants from ``conf/size/tiny.yaml`` (which inlines the domain/time) by
mirroring the values (keeps the script self-contained on a compute node).
"""

import os
import pathlib
import shutil
import subprocess
import sys
import time

from pypalm.forward_model import ForwardModel
from pypalm.utils.clean_up_utils import clean_palm_output_dir


REPO_ROOT = pathlib.Path("/home/ntmucke/pyurbanair").resolve()
CASE_DIR = REPO_ROOT / "examples/palm/experiments/xie_and_castro_palm"
STL_PATH = CASE_DIR / "xie_castro_2008_STL.stl"

PALM_MODEL_SYSTEM = REPO_ROOT / "libs/pypalm/palm_model_system"
PALMRUN_BIN = PALM_MODEL_SYSTEM / "bin/palmrun"
COMBINE_BIN = PALM_MODEL_SYSTEM / "MAKE_DEPOSITORY_default/combine_plot_fields.x"


def _stash_dir(name: str) -> pathlib.Path:
    root = pathlib.Path(os.environ["M0_STASH"]).resolve()
    p = root / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_globs(src: pathlib.Path, dst: pathlib.Path, patterns: list[str]) -> list[str]:
    saved: list[str] = []
    for pat in patterns:
        for path in src.rglob(pat):
            if not path.is_file():
                continue
            target = dst / path.name
            shutil.copy2(path, target)
            saved.append(str(path.relative_to(src)))
    return saved


def _find_palmrun_tempdir(fast_io_root: pathlib.Path, run_id: str) -> pathlib.Path | None:
    """Find the most recent tempdir palmrun used.

    palmrun layout: ``$fast_io_catalog/<run_id>.<RANDOM>``.
    """
    candidates = sorted(
        fast_io_root.glob(f"{run_id}.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_fm() -> ForwardModel:
    """Tiny-config ForwardModel, instantiated directly (no Hydra)."""
    temp_dir = pathlib.Path(os.environ["M0_TEMP"]).resolve()
    return ForwardModel(
        case_dir=CASE_DIR,
        stl_path=STL_PATH,
        experiment_name="urban_run",
        ncpu=1,
        nx=20, ny=20, nz=16,  # matches tiny.slurm's `domain.nz=16` override
        bounds=((0.0, 20.0), (0.0, 20.0), (0.0, 10.0)),
        simulation_time=5.0,
        output_frequency=1.0,
        spinup_time=5.0,
        boundary_condition="inflow_outflow",
        nudging_config={"profile_config": {"type": "uniform"}},
        verbose=True,
        temp_dir=temp_dir,
    )


def _direct_palmrun(fm: ForwardModel, extra_flags: list[str]) -> int:
    """Invoke palmrun directly with the given extra flags. Mirrors execute.sh."""
    fm._ensure_palm_config_in_cwd()
    cmd = [
        str(PALMRUN_BIN),
        "-r", fm.experiment_name,
        "-c", "default",
        "-a", "d3#",
        "-X", str(fm.ncpu),
        "-T", str(fm.ncpu),
        "-q", "none",
        "-v",
        *extra_flags,
    ]
    print(f"[m0] palmrun command: {' '.join(cmd)}", flush=True)
    print(f"[m0] cwd = {fm.dirs.experiment_dir}", flush=True)
    t0 = time.monotonic()
    result = subprocess.run(
        cmd,
        cwd=fm.dirs.experiment_dir,
        stdin=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    print(f"[m0] palmrun wall={time.monotonic() - t0:.1f}s rc={result.returncode}", flush=True)
    return result.returncode


def main() -> None:
    fm = build_fm()
    # Apply default inflow so the namelist is fully populated.
    fm._apply_inflow_settings(fm.params)

    fast_io_root = pathlib.Path(
        os.environ.get("PYPALM_FAST_IO_CATALOG", str(fm.dirs.experiment_dir / "tmp"))
    ) / fm.experiment_name

    output_dir = fm.dirs.output_dir

    # ---------------- Run 1: normal palmrun (-B to keep tempdir) ----------------
    print("\n[m0] === Run 1: palmrun -B (combine on, keep tempdir) ===", flush=True)
    clean_palm_output_dir(fm.dirs)
    rc = _direct_palmrun(fm, ["-B"])
    if rc != 0:
        print(f"[m0] Run 1 failed (rc={rc}); aborting.", flush=True)
        sys.exit(rc)

    stash1 = _stash_dir("run1_combine_on")
    saved = _copy_globs(output_dir, stash1, ["*_3d*.nc", "*_xy*.nc", "RUN_CONTROL"])
    print(f"[m0] Run 1 OUTPUT files stashed -> {stash1.name}: {saved}", flush=True)

    run1_tempdir = _find_palmrun_tempdir(fast_io_root, fm.experiment_name)
    print(f"[m0] Run 1 tempdir = {run1_tempdir}", flush=True)
    if run1_tempdir is not None:
        tempstash1 = _stash_dir("run1_tempdir")
        # Capture everything interesting: pre-combine netCDF (DATA_3D_NETCDF),
        # per-PE binary (PLOT3D_DATA_*), ENVPAR (M1 prep), iofiles, palmrun log.
        saved_tmp = _copy_globs(
            run1_tempdir,
            tempstash1,
            [
                "DATA_3D_NETCDF*",
                "PLOT3D_DATA_*",
                "DATA_2D_*_NETCDF*",
                "PLOT2D_*_*",
                "ENVPAR",
                ".palm.iofiles",
                "PARIN",
                "RUN_CONTROL",
                "palm",  # the executable, in case M1 needs it
            ],
        )
        print(f"[m0] Run 1 tempdir files stashed -> {tempstash1.name}: {saved_tmp}", flush=True)

    # ---------------- Run 2: palmrun -Z -B (skip combine, keep tempdir) ----------------
    print("\n[m0] === Run 2: palmrun -Z -B (combine off, keep tempdir) ===", flush=True)
    clean_palm_output_dir(fm.dirs)
    rc = _direct_palmrun(fm, ["-Z", "-B"])
    if rc != 0:
        print(f"[m0] Run 2 failed (rc={rc}); continuing to bare-combine test.", flush=True)

    stash2 = _stash_dir("run2_combine_off")
    saved = _copy_globs(output_dir, stash2, ["*_3d*.nc", "*_xy*.nc", "RUN_CONTROL"])
    print(f"[m0] Run 2 OUTPUT files stashed -> {stash2.name}: {saved}", flush=True)

    run2_tempdir = _find_palmrun_tempdir(fast_io_root, fm.experiment_name)
    # Avoid grabbing run1's tempdir again if run2 didn't make a new one.
    if run2_tempdir is not None and run2_tempdir != run1_tempdir:
        tempstash2 = _stash_dir("run2_tempdir")
        saved_tmp = _copy_globs(
            run2_tempdir,
            tempstash2,
            ["DATA_3D_NETCDF*", "PLOT3D_DATA_*", "ENVPAR", "PARIN", "RUN_CONTROL"],
        )
        print(f"[m0] Run 2 tempdir files stashed -> {tempstash2.name}: {saved_tmp}", flush=True)
    else:
        print(f"[m0] Run 2 tempdir = {run2_tempdir} (not stashed)", flush=True)

    # ---------------- Run 3: combine_plot_fields.x bare, in run2's tempdir ----------------
    print("\n[m0] === Run 3: ./combine_plot_fields.x (no mpirun) ===", flush=True)
    if run2_tempdir is None or not run2_tempdir.exists():
        print("[m0] Run 2 tempdir missing; skipping bare-combine test.", flush=True)
        return

    # Copy combine_plot_fields.x next to the data files (palmrun normally
    # symlinks it into tempdir; verify before timing).
    local_combine = run2_tempdir / "combine_plot_fields.x"
    if not local_combine.exists():
        shutil.copy2(COMBINE_BIN, local_combine)
        os.chmod(local_combine, 0o755)

    log_path = _stash_dir("run3_combine_bare") / "combine_bare.log"
    with log_path.open("w") as f:
        f.write(f"# combine_plot_fields.x bare run in {run2_tempdir}\n")
        # Snapshot the netCDF u/v/w stats before combine, for the diff.
        pre = run2_tempdir / "DATA_3D_NETCDF"
        if pre.exists():
            shutil.copy2(pre, _stash_dir("run3_combine_bare") / "DATA_3D_NETCDF.pre_bare_combine")
        t0 = time.monotonic()
        result = subprocess.run(
            ["./combine_plot_fields.x"],
            cwd=run2_tempdir,
            capture_output=True,
            text=True,
        )
        wall = time.monotonic() - t0
        f.write(f"wall={wall:.3f}s rc={result.returncode}\n\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n")
        print(f"[m0] combine_plot_fields.x bare wall={wall:.3f}s rc={result.returncode}", flush=True)

        # Snapshot the netCDF after bare combine.
        if pre.exists():
            shutil.copy2(pre, _stash_dir("run3_combine_bare") / "DATA_3D_NETCDF.post_bare_combine")


if __name__ == "__main__":
    main()
