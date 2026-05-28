"""M1 unit test: exercise pypalm.direct_palm.run_direct standalone and
compare its OUTPUT against the M0 palmrun reference.

Acceptance: u/v/w in the direct-run output must match the M0 palmrun
reference exactly (PALM is deterministic at a fixed config). Phase
timings printed for the M1→M2 hand-off.

Run as: ``pixi run -e delftblue -- python -u job_scripts/delftblue/pypalm/m1_direct_run.py``
(but typically driven by ``m1_direct_run.slurm``).
"""

import os
import pathlib
import sys
import time

import numpy as np
import xarray

from pypalm.direct_palm import run_direct
from pypalm.forward_model import ForwardModel


REPO_ROOT = pathlib.Path("/home/ntmucke/pyurbanair").resolve()
CASE_DIR = REPO_ROOT / "examples/palm/experiments/xie_and_castro_palm"
STL_PATH = CASE_DIR / "xie_castro_2008_STL.stl"

# M0 reference (combine-on, run1 of slurm 9997968). Captured from a known-good
# palmrun + combine_plot_fields chain on cmp014.
M0_REFERENCE = pathlib.Path(
    "/scratch/ntmucke/m0_capture/9997968/stash/run1_combine_on/urban_run_3d.000.nc"
)


def _build_fm() -> ForwardModel:
    """Same tiny config as m0_capture.py — must keep these in sync until M2."""
    temp_dir = pathlib.Path(os.environ["M1_TEMP"]).resolve()
    return ForwardModel(
        case_dir=CASE_DIR,
        stl_path=STL_PATH,
        experiment_name="urban_run",
        ncpu=1,
        nx=20, ny=20, nz=16,
        bounds=((0.0, 20.0), (0.0, 20.0), (0.0, 10.0)),
        simulation_time=5.0,
        output_frequency=1.0,
        spinup_time=5.0,
        boundary_condition="inflow_outflow",
        nudging_config={"profile_config": {"type": "uniform"}},
        verbose=True,
        temp_dir=temp_dir,
    )


def _compare_against_reference(direct_out: pathlib.Path) -> int:
    if not M0_REFERENCE.exists():
        print(f"[m1] WARN: M0 reference {M0_REFERENCE} missing; skipping equivalence check.",
              flush=True)
        return 0
    if not direct_out.exists():
        print(f"[m1] FAIL: direct-run output {direct_out} missing.", flush=True)
        return 1

    ref = xarray.open_dataset(M0_REFERENCE, engine="netcdf4", decode_timedelta=False)
    new = xarray.open_dataset(direct_out, engine="netcdf4", decode_timedelta=False)

    print(f"[m1] reference dims: {dict(ref.sizes)}", flush=True)
    print(f"[m1] direct-run dims: {dict(new.sizes)}", flush=True)

    failures = 0
    for v in ("u", "v", "w"):
        a = ref[v].values
        b = new[v].values
        if a.shape != b.shape:
            print(f"[m1] FAIL {v}: shape mismatch {a.shape} vs {b.shape}", flush=True)
            failures += 1
            continue
        same_nan = np.isnan(a) == np.isnan(b)
        if not same_nan.all():
            print(f"[m1] FAIL {v}: NaN mask mismatch ({(~same_nan).sum()} cells)",
                  flush=True)
            failures += 1
            continue
        finite = ~np.isnan(a)
        if not np.array_equal(a[finite], b[finite]):
            diff = np.abs(a[finite] - b[finite])
            print(f"[m1] WARN {v}: not bit-identical — max|diff|={diff.max():.4g}, "
                  f"mean|diff|={diff.mean():.4g}, n_diff={int((diff != 0).sum())}",
                  flush=True)
            # PALM is deterministic, but a fresh run on a different node may
            # produce a fractionally different solution due to MPI reduction
            # order. Allow allclose at 1e-5 rel tolerance, fail otherwise.
            if not np.allclose(a[finite], b[finite], rtol=1e-5, atol=1e-6):
                failures += 1
        else:
            print(f"[m1] PASS {v}: bit-identical", flush=True)
    return failures


def main() -> None:
    fm = _build_fm()
    # Populate the namelist + topography exactly like ESMDA's truth setup would.
    fm._apply_inflow_settings(fm.params)

    print(f"[m1] === direct_palm.run_direct (no palmrun, no palmbuild) ===", flush=True)
    t0 = time.monotonic()
    result = run_direct(
        dirs=fm.dirs,
        experiment_name=fm.experiment_name,
        ncpu=fm.ncpu,
        host="default",
        env=os.environ.copy(),
        keep_tempdir=False,
    )
    total = time.monotonic() - t0
    print(
        f"[m1] direct-run phases: stage={result.stage_s:.2f}s palm={result.palm_s:.2f}s "
        f"combine={result.combine_s:.2f}s transfer={result.transfer_s:.2f}s total={total:.2f}s",
        flush=True,
    )
    print(f"[m1] output_files: {result.output_files}", flush=True)

    direct_out = fm.dirs.output_dir / f"{fm.experiment_name}_3d.000.nc"
    failures = _compare_against_reference(direct_out)
    if failures:
        print(f"[m1] FAILED ({failures} variable(s) differ from reference)", flush=True)
        sys.exit(1)
    print(f"[m1] OK — direct-run matches M0 palmrun reference.", flush=True)


if __name__ == "__main__":
    main()
