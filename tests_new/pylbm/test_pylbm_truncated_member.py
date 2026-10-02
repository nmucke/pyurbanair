"""A pylbm member that stops early must fail, not return a short state.

The LBM's Fortran error paths call ``stop``, which exits **0**. ``run()`` uses
``subprocess.run(check=True)``, so a member that dies mid-run looks like a
success: it leaves some of its frames on disk, the collector finds them, and the
only length logic in ``run_single`` trims *surplus* frames — a short member walks
straight through and detonates windows later at the cross-member concat. These
tests pin the shortfall check that turns it into an ordinary member failure, and
(just as importantly) pin the correct-run shapes it must never fire on.

No solver is compiled or launched here: ``run()`` is replaced by a stub that
writes exactly the snapshots a real run would have written.
"""

import pathlib
from typing import Optional

import numpy as np
import pytest
import xarray
from pylbm.forward_model import ForwardModel
from pylbm.utils.dir_utils import DirectoryPaths

from pyurbanair.base_ensemble_forward_model import ForwardModelRunFailure

# Small enough to keep the fake snapshots tiny; the checks under test are pure
# frame bookkeeping and do not look at the fields.
NX, NY, NZ = 2, 2, 2
IOUT = 100
OUTPUT_FREQUENCY = 0.5


def _make_dirs(tmp_path: pathlib.Path) -> DirectoryPaths:
    """A DirectoryPaths pointing entirely inside ``tmp_path``.

    Built by hand rather than via ``get_lbm_directory_paths`` so no build tree is
    resolved or mirrored: only ``experiment_dir``, ``output_dir`` and
    ``infile_path`` are ever touched by the code under test.
    """
    experiment_dir = tmp_path / "experiment"
    output_dir = experiment_dir / "output"
    output_dir.mkdir(parents=True)
    return DirectoryPaths(
        lbm_src_path=tmp_path / "src",
        cwd=tmp_path,
        temp_dir=tmp_path,
        experiment_base_dir=tmp_path,
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        case_dir=tmp_path,
        experiment_name="runcase",
        infile_path=experiment_dir / "infile.in",
        main_f90_path=tmp_path / "main.F90",
        mod_dimensions_path=tmp_path / "mod_dimensions.F90",
        executable_path=tmp_path / "boltzmann",
        makefile_path=tmp_path / "makefile",
        pixi_env_path=tmp_path,
    )


def _write_infile(path: pathlib.Path, nt0: int, nt1: int, iout: int) -> None:
    """The three infile.in keys the collection path reads back."""
    path.write_text(
        f"{nt0}    ! nt0 : first timestep\n"
        f"{nt1}    ! nt1 : last timestep\n"
        f"{iout}   ! iout : output interval\n"
    )


def _make_model(
    tmp_path: pathlib.Path,
    *,
    nt0: int,
    num_outputs: int,
    spinup_outputs: int = 0,
) -> ForwardModel:
    """A ForwardModel with only the state ``run_single``'s collection path uses.

    ``__init__`` is bypassed (it voxelises an STL and rewrites Fortran sources);
    every attribute set here is one that ``_set_scaling_factors`` would have set
    on a real run for a window of ``spinup_outputs + num_outputs`` outputs
    starting at iteration ``nt0``.
    """
    model = ForwardModel.__new__(ForwardModel)
    model.dirs = _make_dirs(tmp_path)
    model.enable_netcdf = True
    model.verbose = False
    model.spinup_time = spinup_outputs * OUTPUT_FREQUENCY
    model._spinup_outputs = spinup_outputs
    model.simulation_time = num_outputs * OUTPUT_FREQUENCY
    model.output_frequency = OUTPUT_FREQUENCY
    model.output_frequency_timesteps = IOUT
    model.num_timesteps = num_outputs * IOUT
    model.seconds_per_timestep = OUTPUT_FREQUENCY / IOUT
    model.C_u = 75
    model._nt0_override = None
    model.x_grid = np.arange(NX, dtype=float)
    model.y_grid = np.arange(NY, dtype=float)
    model.z_grid = np.arange(NZ, dtype=float)

    nt1 = nt0 + (num_outputs + spinup_outputs) * IOUT
    _write_infile(model.dirs.infile_path, nt0=nt0, nt1=nt1, iout=IOUT)
    return model


def _snapshot_iterations(nt0: int, nt1: int, iout: int) -> list[int]:
    """The iterations a *complete* solver run writes into ``(nt0, nt1]``.

    Mirrors ``m_diag.F90``: it dumps when ``mod(it,iout)==0`` or ``it==nt1``
    (the every-iteration clause is disabled by the ``iprt1`` line the wrapper
    writes), over the loop ``it = nt0+1, nt1``.
    """
    return [it for it in range(nt0 + 1, nt1 + 1) if it % iout == 0 or it == nt1]


def _write_snapshot(output_dir: pathlib.Path, iteration: int) -> None:
    """One ``out_0000_F<iter>.nc`` shaped like the solver's own output."""
    field = np.full((NX, NY, NZ), float(iteration))
    xarray.Dataset(
        data_vars={
            "u": (("x", "y", "z"), field),
            "v": (("x", "y", "z"), field),
            "w": (("x", "y", "z"), field),
        }
    ).to_netcdf(output_dir / f"out_0000_F{iteration:06d}.nc")


def _stub_run(model: ForwardModel, iterations: list[int]) -> None:
    """Replace ``run()`` with a stub that writes exactly ``iterations``."""

    def fake_run() -> None:
        for iteration in iterations:
            _write_snapshot(model.dirs.output_dir, iteration)

    model.run = fake_run  # type: ignore[method-assign]
    # ``_set_scaling_factors`` would rewrite infile.in from params we do not
    # have; the window it would have written is already in place.
    model._set_scaling_factors = lambda params=None: None  # type: ignore[method-assign]


def _run(model: ForwardModel) -> xarray.Dataset:
    return model.run_single(state=None, params=None, sim_name="state")


def _complete_run(model: ForwardModel) -> list[int]:
    """The full iteration list for the window currently written into infile.in."""
    nt0 = model._get_infile_int_value("nt0", 0)
    nt1 = model._get_infile_int_value("nt1", 0)
    return _snapshot_iterations(nt0, nt1, IOUT)


# ---------------------------------------------------------------------------
# Correct runs: the check must never fire on any of these
# ---------------------------------------------------------------------------


def test_cold_start_complete_run_is_not_a_failure(tmp_path: pathlib.Path) -> None:
    model = _make_model(tmp_path, nt0=0, num_outputs=8)
    iterations = _complete_run(model)
    # nt0=0 and nt1=8*iout are both on the grid: no off-grid final frame.
    assert len(iterations) == 8
    _stub_run(model, iterations)

    state = _run(model)

    assert state.sizes["time"] == 8
    np.testing.assert_allclose(state["time"].values, np.arange(8) * OUTPUT_FREQUENCY)


def test_cold_start_with_spinup_is_not_a_failure(tmp_path: pathlib.Path) -> None:
    """Spin-up frames count towards the expectation, then get trimmed off.

    The run is ``spinup + window`` frames long, so a rule derived from
    ``simulation_time / output_frequency`` alone would call a healthy spin-up run
    short by ``spinup_outputs`` frames.
    """
    model = _make_model(tmp_path, nt0=0, num_outputs=6, spinup_outputs=3)
    iterations = _complete_run(model)
    assert len(iterations) == 9
    _stub_run(model, iterations)

    state = _run(model)

    # Spin-up trimmed from the front, leaving the window.
    assert state.sizes["time"] == 6


def test_warm_start_aligned_nt0_is_not_a_failure(tmp_path: pathlib.Path) -> None:
    """Warm start whose nt0 happens to sit on the output grid: no extra frame."""
    model = _make_model(tmp_path, nt0=5 * IOUT, num_outputs=8)
    iterations = _complete_run(model)
    assert len(iterations) == 8
    _stub_run(model, iterations)

    assert _run(model).sizes["time"] == 8


def test_warm_start_misaligned_nt0_keeps_its_extra_frame(
    tmp_path: pathlib.Path,
) -> None:
    """The normal warm start: ``nt0 % iout != 0``, so the solver also dumps at nt1.

    ``iout`` moves with the member's ``C_u``, so from window 1 onwards ``nt0``
    (the previous window's final iteration) is generally off this window's grid
    and the run legitimately ends one frame long. Counting only on-grid frames
    would make every warm start look one frame short of expectation, and
    demanding that extra frame on a *cold* start would fail every cold start.
    """
    model = _make_model(tmp_path, nt0=5 * IOUT + 13, num_outputs=8)
    iterations = _complete_run(model)
    assert len(iterations) == 9  # 8 on-grid + the off-grid final frame at nt1
    _stub_run(model, iterations)

    # Surplus is trimmed, not rejected.
    assert _run(model).sizes["time"] == 8


def test_expected_count_matches_the_solver_dump_rule(tmp_path: pathlib.Path) -> None:
    """``_expected_output_count`` against a direct enumeration of m_diag's rule."""
    for nt0, num_outputs, spinup in [
        (0, 8, 0),
        (0, 6, 3),
        (5 * IOUT, 8, 0),
        (5 * IOUT + 13, 8, 0),
        (999 * IOUT + 1, 48, 0),
    ]:
        model = _make_model(
            tmp_path / f"case_{nt0}_{num_outputs}_{spinup}",
            nt0=nt0,
            num_outputs=num_outputs,
            spinup_outputs=spinup,
        )
        assert model._expected_output_count() == len(_complete_run(model))


# ---------------------------------------------------------------------------
# Truncated runs: the failure the exit code never reported
# ---------------------------------------------------------------------------


def test_truncated_run_raises_forward_model_run_failure(
    tmp_path: pathlib.Path,
) -> None:
    """The real incident's shape: a member writes 3 of 48 frames and exits 0."""
    model = _make_model(tmp_path, nt0=0, num_outputs=48)
    _stub_run(model, _complete_run(model)[:3])

    with pytest.raises(ForwardModelRunFailure) as excinfo:
        _run(model)

    message = str(excinfo.value)
    # The shortfall itself, and the non-obvious reason the exit code was 0.
    assert "3 of the 48" in message
    assert "stop" in message
    assert "exited 0" in message


def test_truncated_warm_start_run_raises(tmp_path: pathlib.Path) -> None:
    """A warm start missing only its final frame is still short, not trimmable."""
    model = _make_model(tmp_path, nt0=5 * IOUT + 13, num_outputs=8)
    iterations = _complete_run(model)
    _stub_run(model, iterations[:-2])

    with pytest.raises(ForwardModelRunFailure, match="7 of the 9"):
        _run(model)


def test_truncated_spinup_run_raises(tmp_path: pathlib.Path) -> None:
    """A member that dies after the spin-up but inside the window.

    Left alone this is the nastiest variant: it still has enough frames for the
    spin-up trim to bite, so what escapes is a plausibly-shaped short state.
    """
    model = _make_model(tmp_path, nt0=0, num_outputs=6, spinup_outputs=3)
    _stub_run(model, _complete_run(model)[:5])

    with pytest.raises(ForwardModelRunFailure, match="5 of the 9"):
        _run(model)


def test_run_that_wrote_nothing_raises_the_same_failure(
    tmp_path: pathlib.Path,
) -> None:
    """Zero frames is the same failure, not a FileNotFoundError the runner ignores."""
    model = _make_model(tmp_path, nt0=0, num_outputs=8)
    _stub_run(model, [])

    with pytest.raises(ForwardModelRunFailure, match="0 of the 8"):
        _run(model)


def test_surplus_frames_are_trimmed_not_rejected(tmp_path: pathlib.Path) -> None:
    """A surplus keeps its historical behaviour: trim to the window, no failure."""
    model = _make_model(tmp_path, nt0=0, num_outputs=8)
    iterations = _complete_run(model)
    # An extra in-range dump (e.g. an iprt-triggered one) lands between frames.
    _stub_run(model, sorted(iterations + [iterations[0] + 1]))

    assert _run(model).sizes["time"] == 8


def test_failure_survives_pickling(tmp_path: pathlib.Path) -> None:
    """The parallel path carries the exception across a process boundary."""
    import pickle

    original = ForwardModelRunFailure("wrote 3 of the 48 expected snapshots")
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored, ForwardModelRunFailure)
    assert str(restored) == str(original)


def test_single_run_failure_is_not_swallowed(tmp_path: pathlib.Path) -> None:
    """A non-ensemble forward run has no failure policy: it must raise."""
    model = _make_model(tmp_path, nt0=0, num_outputs=8)
    _stub_run(model, _complete_run(model)[:2])

    state: Optional[xarray.Dataset] = None
    with pytest.raises(ForwardModelRunFailure):
        state = model(state=None, params=None, sim_name="state")
    assert state is None
