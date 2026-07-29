"""Tests for the pyudales ``inlet_turbulence`` knob.

uDALES v2.2.0 has two inlet-turbulence routes and the wrapper deliberately uses
the second one:

* The Lund (1998) recycling/rescaling generator (``iinletgen=1``) is **dead
  code** — not a member of any namelist (``u-dales/src/modstartup.f90:108-173``;
  ``&INLET`` at ``:141-144`` declares only ``Uinf``/``Vinf``/``di``/``dti``/
  ``inletav``/``linletRA``/``lstoreplane``/``lreadminl``/``lfixinlet``/
  ``lfixutauin``/``lwallfunc``), ``call initinlet`` is commented out
  (``program.f90:77``, ``modstartup.f90:627``), ``inletgen``
  (``modinlet.f90:204``) has no call site, and ``modboundary.f90`` never reads
  its output ``u0inletbc``. The tests at the bottom of this file pin all of that
  against the real source, so a solver bump surfaces any change.
* The **precursor/driver** route (``BCxm=3`` -> ``idriver=2``, ``moddriver.f90``)
  is wired end to end. ``pyudales.utils.inlet_turbulence_utils`` feeds it planes
  synthesised in Python instead of running a precursor.

So the contract under test is: the disabled/absent path is a strict no-op
(namoptions byte-identical, no driver files, and in particular no ``iinletgen``
key anywhere), and the enabled path writes a driver-plane set whose binary
layout, statistics and time coverage match what ``moddriver.f90`` reads.
"""

from __future__ import annotations

import inspect
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from pyudales.utils.dir_utils import DirectoryPaths

# The exact set of keys uDALES declares in the &INLET namelist
# (u-dales/src/modstartup.f90:141-144). `iinletgen` is deliberately absent.
INLET_NAMELIST_KEYS = {
    "uinf",
    "vinf",
    "di",
    "dti",
    "inletav",
    "linletra",
    "lstoreplane",
    "lreadminl",
    "lfixinlet",
    "lfixutauin",
    "lwallfunc",
}

# Smoke-shaped inlet plane: small enough to be fast, large enough that the
# digital filter has several cells of support in both directions.
JTOT, KTOT = 24, 16
YLEN, ZSIZE = 48.0, 32.0

NAMOPTIONS_TEMPLATE = (
    "&RUN\niexpnr = 999\nruntime = 100.\ndtmax = 1.\n/\n"
    f"&DOMAIN\nitot = 16\njtot = {JTOT}\nktot = {KTOT}\n"
    f"xlen = 32.\nylen = {YLEN}\n/\n"
    f"&INPS\nzsize = {ZSIZE}\nu0 = 3.\nv0 = 0.\ndpdx = 0.004\ndpdy = 0.\n/\n"
    "&BC\nBCxm = 2\nBCym = 1\nBCtopm = 3\n/\n"
    "&PHYSICS\nlnudge = .true.\nltimedepnudge = .true.\n/\n"
    "&INLET\nUinf = 0.\n/\n"
)


def _udales_src() -> pathlib.Path:
    """Path to the vendored uDALES Fortran sources (``UDALES_PATH`` may be None)."""
    from pyudales import UDALES_PATH

    if UDALES_PATH is None:  # pragma: no cover - submodule not checked out
        pytest.skip("u-dales source not available")
    assert UDALES_PATH is not None
    return pathlib.Path(UDALES_PATH) / "src"


def _make_dirs(
    tmp_path: pathlib.Path, experiment_name: str = "999"
) -> "DirectoryPaths":
    """A minimal DirectoryPaths pointing at a staged experiment directory."""
    from pyudales.utils.dir_utils import DirectoryPaths

    experiment_dir = tmp_path / "experiment" / experiment_name
    experiment_dir.mkdir(parents=True)
    (experiment_dir / f"namoptions.{experiment_name}").write_text(NAMOPTIONS_TEMPLATE)

    # prof.inp / lscale.inp are produced by preprocessing; the inflow writers
    # rewrite their velocity columns in place, so they must have ktot data rows.
    heights = (np.arange(KTOT) + 0.5) * (ZSIZE / KTOT)
    (experiment_dir / f"prof.inp.{experiment_name}").write_text(
        "# test\n# z thl qt u v tke\n"
        + "".join(
            f"  {z:20.15f}  288.000000     0.000000     0.000000"
            "     0.000000     0.000000\n"
            for z in heights
        )
    )
    (experiment_dir / f"lscale.inp.{experiment_name}").write_text(
        "# test\n# z uq vq pqx pqy wfls dqtdxls dqtdyls dqtdtls dthlrad\n"
        + "".join(
            f"  {z:20.15f}     0.000000     0.000000  0.000000000  0.000000000"
            "     0.000000000     0.000000     0.000000     0.000000"
            "    0.000000000000\n"
            for z in heights
        )
    )
    return DirectoryPaths(
        udales_root_path=tmp_path / "u-dales",
        cwd=tmp_path,
        temp_dir=tmp_path,
        experiment_base_dir=tmp_path / "experiment",
        experiment_dir=experiment_dir,
        output_dir=tmp_path / "outputs",
        case_dir=tmp_path / "case",
        experiment_name=experiment_name,
    )


def _params(angle: float = 0.0, speed: float = 3.0) -> xarray.Dataset:
    return xarray.Dataset(
        data_vars={"inflow_angle": angle, "velocity_magnitude": speed}
    )


ENABLED = {"enabled": True, "time_step": 0.5, "driverjobnr": 998}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_disabled_or_absent_is_accepted() -> None:
    """Absent / empty / ``enabled: false`` validates cleanly under either BC."""
    from pyudales.forward_model import validate_inlet_turbulence

    for block in (None, {}, {"enabled": False}):
        for bc in ("periodic", "inflow_outflow"):
            validate_inlet_turbulence(block, bc)


def test_enabled_under_inflow_outflow_is_accepted() -> None:
    """The driver route is reachable, so the enabled path no longer raises."""
    from pyudales.forward_model import validate_inlet_turbulence

    validate_inlet_turbulence({"enabled": True}, "inflow_outflow")


def test_enabled_under_periodic_is_rejected_for_having_no_inlet() -> None:
    """A turbulent *inlet* is meaningless when the x boundaries wrap."""
    from pyudales.forward_model import validate_inlet_turbulence

    with pytest.raises(ValueError, match="inflow_outflow"):
        validate_inlet_turbulence({"enabled": True}, "periodic")


@pytest.mark.parametrize(  # type: ignore[misc]
    "override, match",
    [
        ({"time_step": 0.0}, "time_step"),
        ({"time_step": -1.0}, "time_step"),
        ({"intensity": -0.1}, "intensity"),
        ({"length_scale_y": 0.0}, "length_scale_y"),
        ({"length_scale_z": -5.0}, "length_scale_z"),
        ({"length_scale_x": 0.0}, "length_scale_x"),
        ({"driverjobnr": 1000}, "driverjobnr"),
        ({"driverjobnr": -1}, "driverjobnr"),
        ({"chunkread_size": 0}, "chunkread_size"),
    ],
)
def test_bad_values_are_rejected(override: dict[str, float], match: str) -> None:
    from pyudales.forward_model import validate_inlet_turbulence

    with pytest.raises(ValueError, match=match):
        validate_inlet_turbulence({"enabled": True, **override}, "inflow_outflow")


def test_driverjobnr_may_not_collide_with_the_experiment_number() -> None:
    """Distinct suffixes keep synthesised planes apart from idriver=1 output."""
    from pyudales.forward_model import validate_inlet_turbulence

    with pytest.raises(ValueError, match="collides"):
        validate_inlet_turbulence(
            {"enabled": True, "driverjobnr": 999}, "inflow_outflow", "999"
        )
    # A non-numeric experiment name simply skips the check rather than blowing up.
    validate_inlet_turbulence(
        {"enabled": True, "driverjobnr": 999}, "inflow_outflow", "truth"
    )


def test_unknown_keys_warn_but_do_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown tunables are ignored with a warning (matches instability_check)."""
    from pyudales.forward_model import validate_inlet_turbulence

    with caplog.at_level("WARNING"):
        validate_inlet_turbulence({"enabled": False, "recycle_plane": 12}, "periodic")

    assert "recycle_plane" in caplog.text


def test_constructor_arg_defaults_to_none() -> None:
    """Default runs must not opt into anything (CLAUDE.md no-op rule)."""
    from pyudales.forward_model import ForwardModel

    assert (
        inspect.signature(ForwardModel).parameters["inlet_turbulence"].default is None
    )


# ---------------------------------------------------------------------------
# Driver file layout — the contract everything else is built on
# ---------------------------------------------------------------------------


def test_driver_file_round_trip(tmp_path: pathlib.Path) -> None:
    from pyudales.utils.driver_file_utils import read_driver_files, write_driver_files

    rng = np.random.default_rng(0)
    times = np.arange(5) * 0.25
    planes = [rng.standard_normal((5, KTOT + 2, JTOT + 2)) for _ in range(3)]

    write_driver_files(tmp_path, 998, times, *planes)
    read_times, *read_planes = read_driver_files(tmp_path, 998, JTOT, KTOT)

    assert np.array_equal(read_times, times)
    for written, read_back in zip(planes, read_planes):
        assert np.array_equal(read_back, written)


def test_driver_file_names_match_the_fortran_pattern(tmp_path: pathlib.Path) -> None:
    """``moddriver.f90`` builds ``[tuvw]driver_DDD.NNN`` with two i3.3 fields.

    ``DDD`` is ``driverid = mod(myidy, nprocy)``, which is 0 for every rank
    because the wrapper pins ``nprocy=1`` (utils.ncpu_utils).
    """
    from pyudales.utils.driver_file_utils import driver_file_names, write_driver_files

    assert driver_file_names(7) == {
        "t": "tdriver_000.007",
        "u": "udriver_000.007",
        "v": "vdriver_000.007",
        "w": "wdriver_000.007",
    }

    times = np.arange(3) * 1.0
    zeros = np.zeros((3, KTOT + 2, JTOT + 2))
    write_driver_files(tmp_path, 998, times, zeros, zeros, zeros)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "tdriver_000.998",
        "udriver_000.998",
        "vdriver_000.998",
        "wdriver_000.998",
    ]


def test_record_layout_is_j_fastest_at_a_fixed_byte_offset(
    tmp_path: pathlib.Path,
) -> None:
    """Record *n* starts at ``(n-1)*record_bytes``; within it, j varies fastest.

    ``readdriverfile`` uses ``access='direct'`` with ``recl`` from
    ``inquire(iolength=...)u0(ib,:,:)`` — bytes under gfortran — and the implied
    do-loop ``((store(j,k), j=jb-jh,je+jh), k=kb-kh,ke+kh)``.
    """
    from pyudales.utils.driver_file_utils import (
        DRIVER_DTYPE,
        plane_record_bytes,
        plane_shape,
        write_driver_files,
    )

    nk, nj = plane_shape(JTOT, KTOT)
    assert (nk, nj) == (KTOT + 2, JTOT + 2)
    record_bytes = plane_record_bytes(JTOT, KTOT)
    assert record_bytes == nk * nj * 8

    n_records = 4
    # Encode (record, k, j) so any transposition or offset error is visible.
    planes = np.arange(n_records * nk * nj, dtype=float).reshape(n_records, nk, nj)
    times = np.arange(n_records) * 0.5
    write_driver_files(tmp_path, 998, times, planes, planes, planes)

    path = tmp_path / "udriver_000.998"
    assert path.stat().st_size == n_records * record_bytes
    assert (tmp_path / "tdriver_000.998").stat().st_size == n_records * 8

    raw = np.fromfile(path, dtype=DRIVER_DTYPE)
    for n in range(n_records):
        start = n * record_bytes // DRIVER_DTYPE.itemsize
        record = raw[start : start + nk * nj]
        # j fastest -> consecutive values along the last axis of planes[n].
        assert np.array_equal(record, planes[n].reshape(-1))
        # And the first two values differ by one j step, not one k step.
        assert record[1] - record[0] == 1.0
        assert record[nj] - record[0] == float(nj)


def test_torn_writes_cannot_be_left_behind(tmp_path: pathlib.Path) -> None:
    """A pre-existing file is replaced atomically, never appended or truncated."""
    from pyudales.utils.driver_file_utils import write_driver_files

    stale = tmp_path / "udriver_000.998"
    stale.write_bytes(b"\x00" * 4096)

    times = np.arange(3) * 1.0
    zeros = np.zeros((3, KTOT + 2, JTOT + 2))
    write_driver_files(tmp_path, 998, times, zeros, zeros, zeros)

    assert stale.stat().st_size == 3 * (KTOT + 2) * (JTOT + 2) * 8
    assert not list(tmp_path.glob("*.tmp"))


def test_non_monotonic_times_are_rejected(tmp_path: pathlib.Path) -> None:
    """``drivergen`` interpolates between neighbours; unordered stamps break it."""
    from pyudales.utils.driver_file_utils import write_driver_files

    zeros = np.zeros((3, KTOT + 2, JTOT + 2))
    with pytest.raises(ValueError, match="ascending"):
        write_driver_files(
            tmp_path, 998, np.array([0.0, 2.0, 1.0]), zeros, zeros, zeros
        )


# ---------------------------------------------------------------------------
# Generator statistics
# ---------------------------------------------------------------------------


def _generate(
    n_records: int = 600, start_index: int = 0, **overrides: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pyudales.utils.inlet_turbulence_utils import build_driver_planes

    kwargs = dict(
        jtot=JTOT,
        ktot=KTOT,
        zsize=ZSIZE,
        ylen=YLEN,
        times=np.arange(n_records) * 0.5,
        start_index=start_index,
        inflow_angle=np.zeros(n_records),
        velocity_magnitude=np.full(n_records, 3.0),
        profile_config={"type": "power_law", "alpha": 0.25},
        intensity=0.1,
        length_scale_y=4.0,
        length_scale_z=4.0,
        length_scale_x=5.0,
        seed=20260729,
    )
    kwargs.update(overrides)
    return build_driver_planes(**kwargs)


def test_plane_means_follow_the_requested_inflow_profile() -> None:
    """The ESMDA parameters reach the solver through the plane means."""
    u, v, _ = _generate(n_records=800)

    zt = (np.arange(KTOT) + 0.5) * (ZSIZE / KTOT)
    target = 3.0 * (zt / ZSIZE) ** 0.25

    interior_u = u[:, 1 : KTOT + 1, 1 : JTOT + 1]
    assert np.allclose(interior_u.mean(axis=(0, 2)), target, rtol=0.05)
    # angle=0 -> the whole magnitude is in u.
    assert np.allclose(
        v[:, 1 : KTOT + 1, 1 : JTOT + 1].mean(axis=(0, 2)), 0.0, atol=0.05
    )


def test_inflow_angle_is_baked_into_the_planes() -> None:
    """The angle is applied in Python, so ``iangledeg`` stays at its default 0."""
    n = 400
    u, v, _ = _generate(
        n_records=n, inflow_angle=np.full(n, 30.0), velocity_magnitude=np.full(n, 3.0)
    )
    u_bar = u[:, 1 : KTOT + 1, 1 : JTOT + 1].mean()
    v_bar = v[:, 1 : KTOT + 1, 1 : JTOT + 1].mean()
    assert np.degrees(np.arctan2(v_bar, u_bar)) == pytest.approx(30.0, abs=1.0)


def test_fluctuation_rms_matches_the_requested_intensity() -> None:
    """``sigma(z) = intensity * |U(z)|`` for u, and 0.7x that for v and w."""
    from pyudales.utils.inlet_turbulence_utils import CROSS_COMPONENT_ANISOTROPY

    u, v, w = _generate(n_records=2000)

    zt = (np.arange(KTOT) + 0.5) * (ZSIZE / KTOT)
    sigma_u = 0.1 * 3.0 * (zt / ZSIZE) ** 0.25

    u_rms = u[:, 1 : KTOT + 1, 1 : JTOT + 1].std(axis=(0, 2))
    v_rms = v[:, 1 : KTOT + 1, 1 : JTOT + 1].std(axis=(0, 2))
    assert np.allclose(u_rms, sigma_u, rtol=0.15)
    assert np.allclose(v_rms, CROSS_COMPONENT_ANISOTROPY * sigma_u, rtol=0.15)

    # w lives on the faces; row 1 is the ground, where it must vanish exactly.
    assert np.all(w[:, 1, :] == 0.0)
    assert w[:, 2:, 1 : JTOT + 1].std() > 0.0


def test_plane_mean_of_the_streamwise_fluctuation_is_zero() -> None:
    """Instantaneous bulk inflow == mean-profile bulk, so the outlet BC agrees."""
    u, _, _ = _generate(n_records=200)

    zt = (np.arange(KTOT) + 0.5) * (ZSIZE / KTOT)
    profile_bulk = (3.0 * (zt / ZSIZE) ** 0.25).mean()
    per_record = u[:, 1 : KTOT + 1, 1 : JTOT + 1].mean(axis=(1, 2))
    assert np.allclose(per_record, profile_bulk, atol=1e-12)


def test_temporal_autocorrelation_follows_the_taylor_time_scale() -> None:
    """AR(1) with ``a = exp(-pi*dt / (2*length_scale_x/U_ref))``."""
    u, _, _ = _generate(n_records=6000)

    dt, length_scale_x, u_ref = 0.5, 5.0, 3.0
    a = np.exp(-np.pi * dt / (2.0 * length_scale_x / u_ref))

    signal = u[:, KTOT // 2, JTOT // 2]
    signal = signal - signal.mean()
    for lag in (1, 2, 4):
        empirical = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
        assert empirical == pytest.approx(a**lag, abs=0.06)


def test_ghost_columns_are_the_periodic_wrap() -> None:
    """y is periodic (``BCym=1``), so the j halo is the interior's image."""
    u, v, w = _generate(n_records=20)
    for plane in (u, v, w):
        assert np.array_equal(plane[:, :, 0], plane[:, :, JTOT])
        assert np.array_equal(plane[:, :, JTOT + 1], plane[:, :, 1])


def test_windows_continue_each_other(tmp_path: pathlib.Path) -> None:
    """Window *n*'s planes are the exact continuation of window *n-1*'s.

    Generating ``[0, t2]`` and slicing at ``t1`` must give the same records as
    generating a window that starts at ``t1`` with the same seed — that is what
    makes the AR(1) recursion a single continuous history despite the solver
    clock restarting at 0 every window.
    """
    long_run = _generate(n_records=60, start_index=0)
    second_window = _generate(n_records=20, start_index=40)

    for whole, window in zip(long_run, second_window):
        assert np.array_equal(whole[40:], window)


def test_seed_is_stable_and_per_member() -> None:
    """Seeds must survive a forkserver hop, so they cannot come from ``hash()``."""
    from pyudales.utils.inlet_turbulence_utils import derive_seed

    assert derive_seed("012") == derive_seed("012")
    assert derive_seed("012") != derive_seed("013")


def test_zero_intensity_gives_a_laminar_profile_inlet() -> None:
    """``intensity: 0`` is legal and produces the mean profile exactly."""
    u, v, w = _generate(n_records=10, intensity=0.0)

    zt = (np.arange(KTOT) + 0.5) * (ZSIZE / KTOT)
    target = 3.0 * (zt / ZSIZE) ** 0.25
    assert np.allclose(u[:, 1 : KTOT + 1, 1 : JTOT + 1], target[None, :, None])
    assert np.allclose(v, 0.0)
    assert np.allclose(w, 0.0)


# ---------------------------------------------------------------------------
# Time-axis coverage — a shortfall hard-stops the solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    "spinup_time, simulation_time",
    [(30.0, 300.0), (0.0, 300.0), (0.0, 7.0), (0.0, 0.5)],
)
def test_driver_records_cover_the_whole_window(
    spinup_time: float, simulation_time: float
) -> None:
    """``drivergen`` stops the run once ``timee`` exceeds the last record.

    ``btime`` is always 0 here because ``_prepare_warmstart`` writes ``timee=0``
    into every restart, so the requirement is simply
    ``max(storetdriver) >= spinup + runtime``.
    """
    from pyudales.utils.inlet_turbulence_utils import driver_time_grid

    times = driver_time_grid(spinup_time + simulation_time, 0.5)
    assert times[-1] >= spinup_time + simulation_time
    assert times[0] == 0.0
    assert np.allclose(np.diff(times), 0.5)


# ---------------------------------------------------------------------------
# Namoptions wiring
# ---------------------------------------------------------------------------


def test_enabled_path_writes_the_driver_namelist(tmp_path: pathlib.Path) -> None:
    """The full write set: BCxm=3, a &DRIVER block, nudging off, dpdx/dpdy zeroed."""
    from pyudales.utils.inlet_turbulence_utils import apply_inlet_turbulence
    from pyudales.utils.namoptions_utils import NamoptionsFile

    dirs = _make_dirs(tmp_path)
    apply_inlet_turbulence(
        params=_params(),
        dirs=dirs,
        config=ENABLED,
        profile_config={"type": "power_law", "alpha": 0.25},
        spinup_time=0.0,
        simulation_time=10.0,
    )

    namoptions = NamoptionsFile(dirs.experiment_dir / "namoptions.999")
    assert namoptions.get_value_as_int("BC", "BCxm") == 3

    # &DRIVER is absent from the case template; set_value must create it.
    assert namoptions.get_value_as_int("DRIVER", "idriver") == 2
    assert namoptions.get_value_as_int("DRIVER", "driverjobnr") == 998
    assert namoptions.get_value_as_float("DRIVER", "dtdriver") == pytest.approx(0.5)
    assert namoptions.get_value_as_float("DRIVER", "tdriverstart") == 0.0

    # driverstore must be the EXACT record count; a larger value makes
    # readdriverfile run off the end of the file.
    driverstore = namoptions.get_value_as_int("DRIVER", "driverstore")
    times = np.fromfile(dirs.experiment_dir / "tdriver_000.998", dtype="<f8")
    assert driverstore == times.size
    assert times[-1] >= 10.0

    assert namoptions.get_value_as_bool("PHYSICS", "lnudge") is False
    assert namoptions.get_value_as_bool("PHYSICS", "ltimedepnudge") is False

    assert namoptions.get_value_as_float("INPS", "dpdx") == 0.0
    assert namoptions.get_value_as_float("INPS", "dpdy") == 0.0
    assert namoptions.get_value_as_float("INPS", "u0") == pytest.approx(3.0)


def test_optional_chunkread_keys_are_only_written_when_configured(
    tmp_path: pathlib.Path,
) -> None:
    """Unset optionals leave uDALES' own defaults in place."""
    from pyudales.utils.inlet_turbulence_utils import apply_inlet_turbulence
    from pyudales.utils.namoptions_utils import NamoptionsFile

    dirs = _make_dirs(tmp_path)
    apply_inlet_turbulence(
        params=_params(), dirs=dirs, config=ENABLED, simulation_time=5.0
    )
    keys = NamoptionsFile(dirs.experiment_dir / "namoptions.999").get_section_keys(
        "DRIVER"
    )
    assert "lchunkread" not in keys
    assert "chunkread_size" not in keys

    dirs = _make_dirs(tmp_path / "chunked")
    apply_inlet_turbulence(
        params=_params(),
        dirs=dirs,
        config={**ENABLED, "lchunkread": True, "chunkread_size": 64},
        simulation_time=5.0,
    )
    namoptions = NamoptionsFile(dirs.experiment_dir / "namoptions.999")
    assert namoptions.get_value_as_bool("DRIVER", "lchunkread") is True
    assert namoptions.get_value_as_int("DRIVER", "chunkread_size") == 64


def test_regeneration_overwrites_rather_than_accumulates(
    tmp_path: pathlib.Path,
) -> None:
    """Every run rewrites the planes, so a previous window cannot leak through."""
    from pyudales.utils.inlet_turbulence_utils import apply_inlet_turbulence

    dirs = _make_dirs(tmp_path)
    apply_inlet_turbulence(
        params=_params(), dirs=dirs, config=ENABLED, simulation_time=20.0
    )
    first = (dirs.experiment_dir / "udriver_000.998").read_bytes()

    apply_inlet_turbulence(
        params=_params(),
        dirs=dirs,
        config=ENABLED,
        simulation_time=20.0,
        window_start_time=20.0,
    )
    second = (dirs.experiment_dir / "udriver_000.998").read_bytes()

    assert len(second) == len(first)
    assert second != first
    assert len(list(dirs.experiment_dir.glob("udriver_*"))) == 1


def test_time_varying_params_reach_the_plane_means(tmp_path: pathlib.Path) -> None:
    """A time-varying inflow is encoded in the plane sequence, not in nudging."""
    from pyudales.utils.driver_file_utils import read_driver_files
    from pyudales.utils.inlet_turbulence_utils import apply_inlet_turbulence

    dirs = _make_dirs(tmp_path)
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": ("time", [0.0, 0.0]),
            "velocity_magnitude": ("time", [2.0, 6.0]),
        },
        coords={"time": [0.0, 20.0]},
    )
    apply_inlet_turbulence(
        params=params,
        dirs=dirs,
        config={**ENABLED, "intensity": 0.0},
        profile_config={"type": "uniform"},
        simulation_time=20.0,
    )

    times, u, _, _ = read_driver_files(dirs.experiment_dir, 998, JTOT, KTOT)
    interior = u[:, 1 : KTOT + 1, 1 : JTOT + 1].mean(axis=(1, 2))
    assert interior[0] == pytest.approx(2.0)
    # np.interp clamps past the last knot, which is what the coverage margin needs.
    assert interior[np.argmin(np.abs(times - 20.0))] == pytest.approx(6.0)
    assert interior[-1] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# No-op guarantee
# ---------------------------------------------------------------------------


def test_disabled_path_writes_nothing(tmp_path: pathlib.Path) -> None:
    """A disabled run is byte-identical and leaves no driver files behind."""
    from pyudales.forward_model import validate_inlet_turbulence
    from pyudales.utils.inlet_turbulence_utils import is_inlet_turbulence_enabled
    from pyudales.utils.namoptions_utils import NamoptionsFile

    dirs = _make_dirs(tmp_path)
    namoptions_path = dirs.experiment_dir / "namoptions.999"
    before = namoptions_path.read_bytes()

    for block in (None, {}, {"enabled": False}, {"enabled": False, "intensity": 0.3}):
        assert not is_inlet_turbulence_enabled(block)
        validate_inlet_turbulence(block, "inflow_outflow", "999")
        assert namoptions_path.read_bytes() == before

    assert not list(dirs.experiment_dir.glob("*driver_*"))
    # And nothing anywhere in the wrapper writes the key that would crash uDALES.
    assert "iinletgen" not in NamoptionsFile(namoptions_path).get_section_keys("INLET")


def test_e2e_disabled_run_leaves_no_inlet_turbulence_artefacts(
    tmp_path: pathlib.Path, compose_test_cfg: Callable[..., "DictConfig"]
) -> None:
    """The no-op guarantee, asserted against a REAL run rather than the util.

    ``test_disabled_path_writes_nothing`` only exercises the validator, so it
    could not see ``run_single`` unconditionally persisting the clock — a file
    dropped into the experiment dir of every default pyudales run.
    """
    from hydra.utils import instantiate
    from pyudales.utils.inlet_turbulence_utils import elapsed_time_path
    from pyudales.utils.namoptions_utils import NamoptionsFile

    cfg = compose_test_cfg(
        [
            *_smoke_overrides(tmp_path),
            "model.forward_model.inlet_turbulence.enabled=false",
        ]
    )
    fm = instantiate(cfg.model.forward_model)
    instantiate(cfg.model.prepare, forward_model=fm)
    fm.run_single()

    assert not elapsed_time_path(fm.dirs).exists()
    assert not list(fm.dirs.experiment_dir.glob("*driver_*"))

    namoptions = NamoptionsFile(
        fm.dirs.experiment_dir / f"namoptions.{fm.dirs.experiment_name}"
    )
    # The nudging path owns the inlet, and no &DRIVER section was invented.
    assert namoptions.get_value_as_int("BC", "BCxm") == 2
    assert not namoptions.has_section("DRIVER")
    assert namoptions.get_value_as_bool("PHYSICS", "lnudge") is True


def test_wrapper_never_writes_iinletgen() -> None:
    """No code path may emit `iinletgen` — the solver aborts on the key."""
    forward_model = pathlib.Path(
        inspect.getfile(__import__("pyudales.forward_model", fromlist=["x"]))
    )
    utils_dir = forward_model.parent / "utils"
    sources = [forward_model, *utils_dir.glob("*.py")]

    for source in sources:
        text = source.read_text()
        for line in text.splitlines():
            # Comments/docstrings explain *why* the key is unusable; only actual
            # `set_value(..., "iinletgen", ...)` writes are forbidden.
            assert not (
                "set_value" in line and "iinletgen" in line
            ), f"{source}: writes iinletgen, which aborts the uDALES namelist read"


# ---------------------------------------------------------------------------
# Solver ground truth, asserted against the real source
# ---------------------------------------------------------------------------


def test_iinletgen_is_not_declared_in_any_udales_namelist() -> None:
    """Pin the reason the Lund generator cannot be used, so a bump surfaces it.

    If a future uDALES version adds ``iinletgen`` to a namelist this test fails,
    which is the signal to revisit whether the synthetic-plane route is still the
    right implementation.
    """
    declared = _declared_namelist_keys()

    # Sanity: the parser really did find the &INLET block.
    assert INLET_NAMELIST_KEYS <= declared

    assert "iinletgen" not in declared


def test_driver_namelist_declares_every_key_the_wrapper_writes() -> None:
    """An *undeclared* &DRIVER key aborts the namelist read with ``stop 1``."""
    declared = _declared_namelist_keys()

    for key in (
        "idriver",
        "driverjobnr",
        "driverstore",
        "dtdriver",
        "tdriverstart",
        "lchunkread",
        "chunkread_size",
    ):
        assert key in declared, f"&DRIVER no longer declares {key}"


def _declared_namelist_keys() -> set[str]:
    """Every key declared in any ``namelist/.../`` block in the uDALES source."""
    src = _udales_src()
    if not src.exists():  # pragma: no cover - submodule not checked out
        pytest.skip("u-dales source not available")

    declared: set[str] = set()
    for path in src.glob("*.f90"):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            if "namelist/" not in line.lower():
                continue
            body = line
            cursor = index
            while body.rstrip().endswith("&"):
                cursor += 1
                if cursor >= len(lines):
                    break
                body += lines[cursor]
            declared.update(
                token.strip().lower()
                for token in body.split("/", 2)[-1].replace("&", "").split(",")
            )
    return declared


def test_udales_never_calls_the_inlet_generator() -> None:
    """`inletgen`/`initinlet` have no live call sites in uDALES v2.2.0."""
    src = _udales_src()
    if not src.exists():  # pragma: no cover - submodule not checked out
        pytest.skip("u-dales source not available")

    calls = []
    for path in src.glob("*.f90"):
        for line in path.read_text().splitlines():
            code = line.split("!", 1)[0].lower()
            if "call initinlet" in code or "call inletgen" in code:
                calls.append(f"{path.name}: {line.strip()}")

    assert calls == [], f"inlet generator is now called: {calls}"


def test_bcxm_driver_is_wired_to_the_inlet_face() -> None:
    """``BCxm=3`` must still force ``idriver=2`` and reach ``xmi_driver``/``bcpup``."""
    src = _udales_src()
    if not src.exists():  # pragma: no cover - submodule not checked out
        pytest.skip("u-dales source not available")

    startup = (src / "modstartup.f90").read_text()
    boundary = (src / "modboundary.f90").read_text()

    assert "case(BCxm_driver)" in startup
    assert "idriver = 2" in startup
    # The inlet face and the pressure-step pin, i.e. the two places the planes
    # actually take effect.
    assert "subroutine xmi_driver" in boundary
    assert "u0driver(j, k) * rk3coefi" in boundary


# ---------------------------------------------------------------------------
# End-to-end: the solver must actually consume the planes
# ---------------------------------------------------------------------------
#
# These run the real uDALES binary on the conftest smoke shape. They are the
# only check that the *binary* layout in driver_file_utils is right — a wrong
# halo width or record order still round-trips through Python perfectly, it just
# makes the solver read garbage (or run off the end of the file).


def _smoke_overrides(tmp_path: pathlib.Path) -> list[str]:
    return [
        "model=pyudales",
        # Pin ncpu: the model config's value is tuned for production runs, and
        # decomposing the 20-cell smoke domain into that many x-strips is its
        # own source of instability. These tests are about the inlet, so keep
        # the decomposition out of the picture (conftest pins the domain for the
        # same reason).
        "model.forward_model.ncpu=1",
        "model.forward_model.inlet_turbulence.enabled=true",
        # The shipped length scales (~building height) exceed the 20x20x10 m
        # smoke domain, where the filter would wrap onto itself; scale them to
        # the test domain so the fluctuations are meaningful.
        "model.forward_model.inlet_turbulence.length_scale_y=4.0",
        "model.forward_model.inlet_turbulence.length_scale_z=4.0",
        "model.forward_model.inlet_turbulence.length_scale_x=6.0",
        "model.forward_model.inlet_turbulence.intensity=0.15",
        f"paths.experiment_dir={tmp_path / 'experiment'}",
        f"++paths.base_results_dir={tmp_path / 'results'}",
    ]


def test_e2e_solver_reads_and_interpolates_the_driver_planes(
    tmp_path: pathlib.Path, compose_test_cfg: Callable[..., "DictConfig"]
) -> None:
    """A real run consumes the planes and puts them on the inlet face.

    The face check is the sharp one: ``bcpup`` pins ``pup(ib,:,:) =
    u0driver*rk3coefi`` (``modboundary.f90:1239-1247``), so the fielddump's
    ``xm=0`` column *is* the driver plane, time-interpolated. Any transposition
    or halo-width error destroys that correlation while still producing a run
    that looks superficially healthy.
    """
    from hydra.utils import instantiate
    from pyudales.utils.driver_file_utils import read_driver_files

    cfg = compose_test_cfg(_smoke_overrides(tmp_path))
    fm = instantiate(cfg.model.forward_model)
    instantiate(cfg.model.prepare, forward_model=fm)

    # run_single, not __call__: __call__ cleans the output dir, taking the log
    # asserted on below with it.
    state = fm.run_single()

    log = (
        fm.dirs.output_dir
        / fm.dirs.experiment_name
        / f"run.{fm.dirs.experiment_name}.log"
    ).read_text()
    assert "Reading precursor driver simulation" in log
    assert "Inputs interpolated from driver tsteps" in log
    # The hard stop drivergen takes when timee outruns the records.
    assert "no more inlet data available" not in log

    assert state.sizes["time"] == int(
        cfg.time.simulation_time / cfg.time.output_frequency
    )

    jtot = cfg.domain.ny
    ktot = cfg.domain.nz
    times, u_driver, _, _ = read_driver_files(fm.dirs.experiment_dir, 998, jtot, ktot)

    # The returned state's time coordinate is rebased to 0 by the spinup trim,
    # so read the raw fielddump, which still carries absolute solver time.
    raw = fm._read_fielddump()
    face = raw["u"].isel(xm=0).values  # (time, zt, yt)
    assert face.std() > 0.0, "inlet face is uniform — no turbulence reached it"

    for index, t in enumerate(raw["time"].values):
        position = np.interp(float(t), times, np.arange(times.size))
        lo = int(np.floor(position))
        hi = min(lo + 1, times.size - 1)
        frac = position - lo
        plane = (1 - frac) * u_driver[lo, 1 : ktot + 1, 1 : jtot + 1] + frac * u_driver[
            hi, 1 : ktot + 1, 1 : jtot + 1
        ]
        correlation = np.corrcoef(face[index].ravel(), plane.ravel())[0, 1]
        assert correlation > 0.95, (
            f"inlet face does not match the driver plane at t={t} "
            f"(corr={correlation:.3f}) — check the record ordering/halo widths"
        )


def test_e2e_two_window_rollout_continues_the_turbulence(
    tmp_path: pathlib.Path, compose_test_cfg: Callable[..., "DictConfig"]
) -> None:
    """Window 2 warm-starts and picks the turbulence history up where 1 left off.

    ``_prepare_warmstart`` writes ``timee=0`` into the restart, so window 2's
    driver grid restarts at 0 too; only its *content* is offset. If the coverage
    margin were wrong, drivergen would stop the run outright rather than fail
    quietly.
    """
    from hydra.utils import instantiate
    from pyudales.utils.driver_file_utils import read_driver_files

    cfg = compose_test_cfg(_smoke_overrides(tmp_path))
    fm = instantiate(cfg.model.forward_model)
    instantiate(cfg.model.prepare, forward_model=fm)

    jtot, ktot = cfg.domain.ny, cfg.domain.nz
    time_step = cfg.model.forward_model.inlet_turbulence.time_step
    spinup, simulation = cfg.time.spinup_time, cfg.time.simulation_time

    # __call__ (not run_single) because it loads the result and cleans the
    # output dir, which is what the rollout layer does between windows.
    window1 = fm(state=None)
    times1, u1, _, _ = read_driver_files(fm.dirs.experiment_dir, 998, jtot, ktot)
    assert fm._elapsed_time == pytest.approx(spinup + simulation)
    assert times1[-1] >= spinup + simulation

    window2 = fm(state=window1)
    times2, u2, _, _ = read_driver_files(fm.dirs.experiment_dir, 998, jtot, ktot)
    assert fm._elapsed_time == pytest.approx(spinup + 2 * simulation)

    # Warm windows drop the spinup, so window 2 needs a shorter record grid that
    # still starts at 0 (the solver clock restarts every window).
    assert times2[0] == 0.0
    assert times2[-1] >= simulation
    assert times2.size < times1.size

    offset = int(round((spinup + simulation) / time_step))
    overlap = min(times1.size - offset, times2.size)
    assert overlap > 0
    assert np.allclose(u1[offset : offset + overlap], u2[:overlap])

    assert window2.sizes["time"] == int(simulation / cfg.time.output_frequency)


# ---------------------------------------------------------------------------
# The physical clock must survive the process boundary
# ---------------------------------------------------------------------------


def test_clock_round_trips_through_disk(tmp_path: pathlib.Path) -> None:
    from pyudales.utils.inlet_turbulence_utils import (
        read_elapsed_time,
        write_elapsed_time,
    )

    dirs = _make_dirs(tmp_path)
    assert read_elapsed_time(dirs) == 0.0
    write_elapsed_time(dirs, 42.5)
    assert read_elapsed_time(dirs) == 42.5


def test_a_clock_belonging_to_another_member_is_ignored(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``create_new_forward_model`` mirrors the template dir with an unfiltered
    ``copy_files``, so a stray clock could otherwise become every member's."""
    from pyudales.utils.inlet_turbulence_utils import (
        elapsed_time_path,
        read_elapsed_time,
        write_elapsed_time,
    )

    template = _make_dirs(tmp_path / "template", "999")
    member = _make_dirs(tmp_path / "member", "003")
    write_elapsed_time(template, 600.0)
    # Simulate the unfiltered copy.
    elapsed_time_path(member).write_text(elapsed_time_path(template).read_text())

    with caplog.at_level("WARNING"):
        assert read_elapsed_time(member) == 0.0
    assert "belongs to member '999'" in caplog.text


def test_reset_clears_a_persisted_clock(tmp_path: pathlib.Path) -> None:
    """A freshly built model starts at 0 however dirty the experiment dir is."""
    from pyudales.utils.inlet_turbulence_utils import (
        elapsed_time_path,
        read_elapsed_time,
        reset_elapsed_time,
        write_elapsed_time,
    )

    dirs = _make_dirs(tmp_path)
    write_elapsed_time(dirs, 900.0)
    reset_elapsed_time(dirs)
    assert not elapsed_time_path(dirs).exists()
    assert read_elapsed_time(dirs) == 0.0
    reset_elapsed_time(dirs)  # idempotent


def test_new_ensemble_members_do_not_inherit_the_template_clock(
    tmp_path: pathlib.Path,
) -> None:
    """The real path: deep-copied member + unfiltered dir copy."""
    from pyudales.utils.inlet_turbulence_utils import (
        elapsed_time_path,
        write_elapsed_time,
    )

    template = _make_dirs(tmp_path / "t", "999")
    write_elapsed_time(template, 600.0)
    assert elapsed_time_path(template).exists()

    # `create_new_forward_model` needs a real ForwardModel; assert the two
    # mechanisms it relies on instead, which is what the member inherits.
    member = _make_dirs(tmp_path / "m", "000")
    elapsed_time_path(member).write_text(elapsed_time_path(template).read_text())

    from pyudales.utils.inlet_turbulence_utils import (
        read_elapsed_time,
        reset_elapsed_time,
    )

    reset_elapsed_time(member)
    assert not elapsed_time_path(member).exists()
    assert read_elapsed_time(member) == 0.0


def test_clock_is_copied_on_failure_substitution(tmp_path: pathlib.Path) -> None:
    """A substituted member inherits the donor's clock, not just its carry.

    Without this the failed member's clock stays a window behind the state it
    was just handed, and the offset persists for the rest of the rollout.
    """
    from pyudales.utils.inlet_turbulence_utils import (
        copy_elapsed_time,
        read_elapsed_time,
        write_elapsed_time,
    )

    donor = _make_dirs(tmp_path / "donor", "001")
    failed = _make_dirs(tmp_path / "failed", "002")
    write_elapsed_time(donor, 600.0)
    write_elapsed_time(failed, 300.0)

    assert copy_elapsed_time(donor, failed) is True
    assert read_elapsed_time(failed) == 600.0

    # A donor that never ran leaves the destination alone rather than zeroing it.
    virgin = _make_dirs(tmp_path / "virgin", "003")
    assert copy_elapsed_time(virgin, failed) is False
    assert read_elapsed_time(failed) == 600.0


def test_e2e_parallel_ensemble_keeps_each_member_continuous(
    tmp_path: pathlib.Path, compose_test_cfg: Callable[..., "DictConfig"]
) -> None:
    """Continuity must hold when members run in forkserver worker processes.

    ``BaseEnsembleForwardModel._run_parallel`` submits ``model.__call__`` to a
    ProcessPoolExecutor, so the member is pickled into a worker and any
    attribute it mutates is discarded on exit. An in-memory clock therefore
    resets to 0 every window under ``num_parallel_processes > 1`` and the AR(1)
    history silently restarts — which is exactly what the clock exists to
    prevent. Single-model tests cannot see this; only a real parallel run can.
    """
    from hydra.utils import instantiate
    from pyudales.utils.driver_file_utils import read_driver_files

    cfg = compose_test_cfg(
        [
            *_smoke_overrides(tmp_path),
            "ensemble.ensemble_size=2",
            "ensemble.num_parallel_processes=2",
        ]
    )
    template = instantiate(cfg.model.forward_model)
    instantiate(cfg.model.prepare, forward_model=template)
    ensemble = instantiate(cfg.model.ensemble_model, forward_model=template)

    jtot, ktot = cfg.domain.ny, cfg.domain.nz
    time_step = cfg.model.forward_model.inlet_turbulence.time_step
    spinup, simulation = cfg.time.spinup_time, cfg.time.simulation_time

    states = ensemble.run_ensemble(state=None, sim_name="state")
    window1 = {
        i: read_driver_files(model.dirs.experiment_dir, 998, jtot, ktot)[1]
        for i, model in enumerate(ensemble.ensemble_forward_models)
    }
    for model in ensemble.ensemble_forward_models:
        assert model._elapsed_time == pytest.approx(
            spinup + simulation
        ), "clock did not survive the worker process"

    ensemble.run_ensemble(state=states, sim_name="state")

    offset = int(round((spinup + simulation) / time_step))
    for i, model in enumerate(ensemble.ensemble_forward_models):
        _, u2, _, _ = read_driver_files(model.dirs.experiment_dir, 998, jtot, ktot)
        u1 = window1[i]
        overlap = min(u1.shape[0] - offset, u2.shape[0])
        assert overlap > 0
        assert np.allclose(
            u1[offset : offset + overlap], u2[:overlap]
        ), f"member {i}'s window-2 planes do not continue its window-1 planes"

    # And the members must not all share one realisation.
    assert not np.allclose(window1[0], window1[1])


# ---------------------------------------------------------------------------
# Silent-failure guards
# ---------------------------------------------------------------------------


def test_time_step_must_divide_the_window(tmp_path: pathlib.Path) -> None:
    """A non-dividing time_step would slip the record grid a little every window."""
    from pyudales.utils.inlet_turbulence_utils import (
        apply_inlet_turbulence,
        validate_time_step_divides_window,
    )

    validate_time_step_divides_window(300.0, 0.5)
    validate_time_step_divides_window(0.0, 0.5)  # nothing to divide yet

    with pytest.raises(ValueError, match="does not divide the window"):
        validate_time_step_divides_window(100.0, 0.3)

    dirs = _make_dirs(tmp_path)
    with pytest.raises(ValueError, match="does not divide the window"):
        apply_inlet_turbulence(
            params=_params(),
            dirs=dirs,
            config={**ENABLED, "time_step": 0.3},
            simulation_time=100.0,
        )


def test_coverage_margin_absorbs_a_full_dtmax_overshoot() -> None:
    """uDALES' last step can overrun ``runtime`` by up to ``dtmax``."""
    from pyudales.utils.inlet_turbulence_utils import driver_time_grid

    # dtmax smaller than the default margin -> the default (2 records) stands.
    assert driver_time_grid(10.0, 0.5, dtmax=0.4)[-1] == pytest.approx(11.0)
    # dtmax larger than the margin -> the grid grows to cover it.
    assert driver_time_grid(10.0, 0.5, dtmax=3.0)[-1] >= 10.0 + 3.0


def test_large_length_scales_warn_about_the_discarded_energy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Calibration must not chase amplitude the plane-mean removal discards."""
    from pyudales.utils.inlet_turbulence_utils import (
        _WARNED_LENGTH_SCALES,
        _warn_if_length_scales_exceed_the_plane,
    )

    _WARNED_LENGTH_SCALES.clear()
    with caplog.at_level("WARNING"):
        _warn_if_length_scales_exceed_the_plane(
            length_scale_y=0.5 * YLEN, length_scale_z=1.0, ylen=YLEN, zsize=ZSIZE
        )
    assert "length_scale_y" in caplog.text

    # Deduplicated per process: a 32-member, 20-window rollout must not emit the
    # same line 640 times.
    caplog.clear()
    with caplog.at_level("WARNING"):
        _warn_if_length_scales_exceed_the_plane(
            length_scale_y=0.5 * YLEN, length_scale_z=1.0, ylen=YLEN, zsize=ZSIZE
        )
    assert caplog.text == ""

    caplog.clear()
    with caplog.at_level("WARNING"):
        _warn_if_length_scales_exceed_the_plane(
            length_scale_y=1.0, length_scale_z=1.0, ylen=YLEN, zsize=ZSIZE
        )
    assert caplog.text == ""
    _WARNED_LENGTH_SCALES.clear()


def test_shipped_default_length_scales_warn_on_this_domain(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Wired into the real entry point, not just callable in isolation.

    The shipped 25 m defaults exceed 30% of this plane, which is precisely the
    case the warning exists for.
    """
    from pyudales.utils.inlet_turbulence_utils import (
        _WARNED_LENGTH_SCALES,
        apply_inlet_turbulence,
    )

    _WARNED_LENGTH_SCALES.clear()
    dirs = _make_dirs(tmp_path)
    with caplog.at_level("WARNING"):
        apply_inlet_turbulence(
            params=_params(), dirs=dirs, config=ENABLED, simulation_time=5.0
        )
    assert "length_scale_y" in caplog.text
    _WARNED_LENGTH_SCALES.clear()


def test_large_driverstore_warns_about_solver_memory(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``initdriver`` allocates ~6 full-history plane arrays with lchunkread off."""
    from pyudales.utils.inlet_turbulence_utils import _warn_if_driver_arrays_are_large

    with caplog.at_level("WARNING"):
        _warn_if_driver_arrays_are_large(
            jtot=64, ktot=64, driverstore=3000, lchunkread=None
        )
    assert "lchunkread" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        _warn_if_driver_arrays_are_large(
            jtot=64, ktot=64, driverstore=3000, lchunkread=True
        )
        _warn_if_driver_arrays_are_large(
            jtot=24, ktot=16, driverstore=20, lchunkread=None
        )
    assert caplog.text == ""


def test_vertical_correlation_survives_the_edge_renormalisation() -> None:
    """The per-level renormalisation must not flatten near-ground z-correlation.

    ``_filter_bounded`` divides each level by the norm of the coefficients that
    actually reached it, which restores unit *variance* at the edges but leaves
    the *correlation* there to be checked separately — the rms test would pass
    even if it had collapsed entirely.
    """
    from pyudales.utils.inlet_turbulence_utils import (
        _filter_bounded,
        filter_coefficients,
    )

    dz = ZSIZE / KTOT
    length_scale_z = 4.0
    rng = np.random.default_rng(0)
    field = _filter_bounded(
        rng.standard_normal((4000, KTOT, JTOT)),
        filter_coefficients(length_scale_z / dz),
    )

    expected = np.exp(-np.pi * dz / (2.0 * length_scale_z))
    for level in (0, 1, KTOT // 2, KTOT - 2):
        pair = np.corrcoef(field[:, level, :].ravel(), field[:, level + 1, :].ravel())
        # Edge levels lose some correlation to truncation; require most of it.
        assert pair[0, 1] > 0.6 * expected, f"z-correlation collapsed at level {level}"


def test_burn_in_truncation_reproduces_the_full_replay() -> None:
    """Late windows replay a bounded prefix, not the whole history.

    ``test_windows_continue_each_other`` only covers offsets the burn-in still
    spans entirely (where the match is bit-exact). This covers the regime that
    actually keeps a long rollout linear: a start index far beyond the burn-in,
    where the prefix IS truncated and the guarantee is numerical rather than
    exact.
    """
    from pyudales.utils.inlet_turbulence_utils import (
        ar1_burn_in_records,
        generate_fluctuation_fields,
    )

    a = 0.6
    burn_in = ar1_burn_in_records(a)
    start = burn_in * 3
    shapes = {"u": (4, 4)}

    full = generate_fluctuation_fields(
        seed=99, ar1_coefficient=a, shapes=shapes, n_records=start + 5, start_index=0
    )["u"]
    truncated = generate_fluctuation_fields(
        seed=99, ar1_coefficient=a, shapes=shapes, n_records=5, start_index=start
    )["u"]

    assert np.allclose(full[start:], truncated, rtol=0, atol=1e-9)
    # The bound must be real work, not a no-op that replays everything anyway.
    assert burn_in < start


def test_burn_in_is_bounded_and_grows_with_the_correlation_time() -> None:
    from pyudales.utils.inlet_turbulence_utils import ar1_burn_in_records

    assert ar1_burn_in_records(0.0) == 0
    assert ar1_burn_in_records(0.5) < ar1_burn_in_records(0.99)
    # Production-ish: dtdriver=0.1, length_scale_x=50 m, U=3 m/s.
    a = float(np.exp(-np.pi * 0.1 / (2 * 50 / 3.0)))
    assert ar1_burn_in_records(a) < 5000
