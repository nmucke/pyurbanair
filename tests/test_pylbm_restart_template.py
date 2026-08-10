"""The pylbm warm start must actually read the restart template it writes.

A warm start is supposed to start from the previous restart's own distribution
and only swap the macroscopic state into it, so that the non-equilibrium part
(the stress the flow was carrying) and the ghost/solid cells survive the swap.
That never happened: ``_try_load_restart_distribution`` asked scipy for the
record with a list of *scalar* dtypes, scipy read that as one repeating 20-byte
compound, and since an LBM restart is four int32 followed by
``27*(nx+2)*(ny+2)*(nz+2)`` float32 -- never a multiple of 20 -- the call raised
on every invocation. The template was silently treated as absent and every warm
start in production was built from a pure equilibrium field.

So the load is not an implementation detail here, it is the physics. These tests
pin it from both ends: a restart written by this module reads back with its
values in the right cells (the round trip that would have caught the original
bug), and each way a restart can legitimately be refused still returns ``None``
*and says so in the log* -- the silence is what let the defect live.
"""

import logging
import pathlib
import re
import types
from typing import cast

import numpy as np
import pytest
import xarray
from pylbm.utils.dir_utils import DirectoryPaths
from pylbm.utils.warm_start_utils import (
    _build_equilibrium_restart_distribution,
    _compute_macrovars_from_distribution,
    _try_load_restart_distribution,
    restart_file_name,
    write_restart_file_from_xarray,
)
from scipy.io import FortranFile

REPO = pathlib.Path(__file__).resolve().parents[1]
LBM_SRC = REPO / "libs" / "pylbm" / "LBM" / "src"

LOGGER_NAME = "pylbm.utils.warm_start_utils"

# Deliberately not a cube and deliberately tiny: distinct extents are what make
# a transposed reshape fail loudly instead of merely relabelling axes.
NX, NY, NZ = 6, 5, 4
NL = 27

# The velocity scale in the infile. Anything but 1.0, so that a regression that
# forgets the m/s -> lattice-unit conversion cannot pass by coincidence.
C_U = 2.5


def _payload_size(nx: int = NX, ny: int = NY, nz: int = NZ) -> int:
    """Number of float32 in one restart record's payload."""
    return NL * (nx + 2) * (ny + 2) * (nz + 2)


def _write_infile(path: pathlib.Path) -> None:
    """A minimal ``infile.in`` holding only the keys the restart writer reads.

    ``Infile`` parses ``value ! key : description`` lines, so each line has to
    carry its key in the trailing comment. Real infiles are positional and much
    longer; nothing here depends on the order.
    """
    path.write_text(
        "\n".join(
            [
                f"{C_U}   ! C_u : velocity scale (m/s per lattice unit)",
                "1      ! ibnd : x boundary (non-periodic)",
                "0      ! jbnd : y boundary (periodic)",
                "22     ! kbnd : z boundary",
                "3      ! ibgk : equilibrium order",
                "F      ! lturb : inflow turbulence",
                "0      ! nturbines : number of turbines",
                "0      ! iablvisc : ABL viscosity model",
                "",
            ]
        )
    )


def _write_mod_dimensions(path: pathlib.Path) -> None:
    """The compiled grid, which the writer cross-checks the state against."""
    path.write_text(
        "module mod_dimensions\n"
        f"   integer, parameter :: nx = {NX}      ! grid dimension x-dir\n"
        f"   integer, parameter :: ny = {NY}      ! grid dimension y-dir\n"
        f"   integer, parameter :: nz = {NZ}      ! grid dimension z-dir\n"
        "   integer, parameter :: ntiles = 1      ! tiles in y\n"
        "   integer, parameter :: ntracer = 0     ! tracers\n"
        "end module\n"
    )


def _make_dirs(tmp_path: pathlib.Path) -> DirectoryPaths:
    """A stub standing in for the fourteen-path build tree.

    ``write_restart_file_from_xarray`` reads exactly three of ``DirectoryPaths``'
    fields, so a stub keeps these tests about the restart file rather than about
    constructing a build tree (the same trick as the filename tests).
    """
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    infile_path = experiment_dir / "infile.in"
    mod_dimensions_path = experiment_dir / "mod_dimensions.F90"
    _write_infile(infile_path)
    _write_mod_dimensions(mod_dimensions_path)
    return cast(
        DirectoryPaths,
        types.SimpleNamespace(
            experiment_dir=experiment_dir,
            infile_path=infile_path,
            mod_dimensions_path=mod_dimensions_path,
        ),
    )


def _make_state() -> xarray.Dataset:
    """A smooth, spatially varying, low-Mach state in physical units.

    Every component varies along a *different* axis. That is the point: a load
    that reshaped the record in C order instead of Fortran order, or transposed
    x and z, would still produce plausible-looking fields, but the macroscopic
    moments would land in the wrong cells and the round-trip assertions fail.
    """
    z = np.arange(NZ, dtype=np.float32)
    y = np.arange(NY, dtype=np.float32)
    x = np.arange(NX, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    rho = (1.0 + 0.01 * xx / max(NX - 1, 1)).astype(np.float32)
    # Physical m/s; the writer divides by C_u to reach lattice units, so keep
    # the resulting lattice velocities well inside the incompressible range.
    u = (C_U * 0.05 * (1.0 + zz / max(NZ - 1, 1))).astype(np.float32)
    v = (C_U * 0.02 * yy / max(NY - 1, 1)).astype(np.float32)
    w = (C_U * 0.01 * xx / max(NX - 1, 1)).astype(np.float32)

    dims = ("z", "y", "x")
    return xarray.Dataset(
        {
            "rho": (dims, rho),
            "u": (dims, u),
            "v": (dims, v),
            "w": (dims, w),
        },
        coords={"z": z, "y": y, "x": x},
    )


def _lattice_macros_from_state(
    state: xarray.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The state's interior fields as (x,y,z) arrays in lattice units."""
    to_xyz = (2, 1, 0)
    rho = np.transpose(state["rho"].values, to_xyz).astype(np.float32)
    u = np.transpose(state["u"].values, to_xyz).astype(np.float32) / C_U
    v = np.transpose(state["v"].values, to_xyz).astype(np.float32) / C_U
    w = np.transpose(state["w"].values, to_xyz).astype(np.float32) / C_U
    return rho, u, v, w


def _write_record(
    path: pathlib.Path, header: tuple[int, int, int, int], payload: np.ndarray
) -> None:
    """Write one restart-shaped Fortran record, header and payload as given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with FortranFile(str(path), "w") as handle:
        handle.write_record(
            *(np.int32(value) for value in header), payload.astype(np.float32)
        )


def test_a_restart_written_by_this_module_reads_back_as_a_template(
    tmp_path: pathlib.Path,
) -> None:
    """The round trip that would have caught the original bug.

    Write a restart from a state, read it back, and check the values landed
    where they were put. Before the fix this returned ``None`` for every file
    ever written, including its own.
    """
    dirs = _make_dirs(tmp_path)
    state = _make_state()

    iteration = write_restart_file_from_xarray(state=state, dirs=dirs)
    restart = dirs.experiment_dir / "restart" / restart_file_name(iteration)
    assert restart.is_file()

    loaded = _try_load_restart_distribution(restart_file=restart, nx=NX, ny=NY, nz=NZ)

    assert loaded is not None, "the template must load; None means it was skipped"
    assert loaded.shape == (NL, NX + 2, NY + 2, NZ + 2)
    assert loaded.dtype == np.float32
    assert loaded.flags.f_contiguous, "the caller ravels this back out in F order"

    # The moments of the distribution must reproduce the state that built it --
    # in the right cells. Equilibrium conserves rho and rho*u exactly on D3Q27,
    # so the only slack here is float32 summation over 27 directions.
    rho, u, v, w = _compute_macrovars_from_distribution(loaded)
    rho_ref, u_ref, v_ref, w_ref = _lattice_macros_from_state(state)
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(rho[interior], rho_ref, rtol=0, atol=1e-5)
    np.testing.assert_allclose(u[interior], u_ref, rtol=0, atol=1e-5)
    np.testing.assert_allclose(v[interior], v_ref, rtol=0, atol=1e-5)
    np.testing.assert_allclose(w[interior], w_ref, rtol=0, atol=1e-5)


def test_the_loaded_distribution_is_finite_and_positive(
    tmp_path: pathlib.Path,
) -> None:
    """A garbled read can still have the right shape; it cannot stay physical.

    For a low-Mach state every D3Q27 equilibrium population is strictly
    positive, so a load that mixed the header into the payload, picked up the
    record markers, or read at the wrong offset shows up here as a negative or
    non-finite population rather than as a subtly wrong warm start.
    """
    dirs = _make_dirs(tmp_path)
    iteration = write_restart_file_from_xarray(state=_make_state(), dirs=dirs)
    restart = dirs.experiment_dir / "restart" / restart_file_name(iteration)

    loaded = _try_load_restart_distribution(restart_file=restart, nx=NX, ny=NY, nz=NZ)
    assert loaded is not None

    assert np.isfinite(loaded).all(), "a restart population must never be NaN/inf"
    assert (loaded > 0.0).all(), (
        "every equilibrium population is positive at this Mach number; a "
        f"negative one (min {loaded.min()}) means the record was misread"
    )
    # And the density it carries is O(1) in lattice units, not O(1e38) garbage.
    assert 0.9 < float(loaded.sum(axis=0).min()) < 1.1


def test_the_template_branch_preserves_the_non_equilibrium_part(
    tmp_path: pathlib.Path,
) -> None:
    """The branch this fix switches on, exercised end to end.

    With a template present, the writer keeps ``f - feq(template) + feq(target)``
    rather than plain ``feq(target)``. Seed the template with a deliberate
    non-equilibrium perturbation, write the same macroscopic state back on top
    of it, and the perturbation must survive -- that is the entire reason the
    template is read at all. It also has to survive the file round trip, which
    is what ties this test to the load being correct.
    """
    dirs = _make_dirs(tmp_path)
    state = _make_state()

    # A first, template-free write: pure equilibrium, by definition.
    iteration = write_restart_file_from_xarray(state=state, dirs=dirs)
    restart = dirs.experiment_dir / "restart" / restart_file_name(iteration)

    # Perturb it into a genuinely non-equilibrium field, conserving rho and
    # rho*u by putting equal and opposite amounts on a direction and its
    # bounce-back partner (directions 1 and 2 are +x and -x).
    template = _try_load_restart_distribution(restart_file=restart, nx=NX, ny=NY, nz=NZ)
    assert template is not None
    perturbation = np.zeros_like(template)
    perturbation[1] = 1e-3
    perturbation[2] = 1e-3
    perturbation[0] = -2e-3
    perturbed = np.asfortranarray(template + perturbation)
    _write_record(restart, (NX, NY, NZ, NL), np.ravel(perturbed, order="F"))

    # Second write, same state -- now the template branch runs.
    second = write_restart_file_from_xarray(
        state=state, dirs=dirs, restart_iteration=iteration
    )
    reloaded = _try_load_restart_distribution(
        restart_file=dirs.experiment_dir / "restart" / restart_file_name(second),
        nx=NX,
        ny=NY,
        nz=NZ,
    )
    assert reloaded is not None

    # The macroscopic state is unchanged, so f_new = f_template exactly (up to
    # float32): the non-equilibrium content is carried through untouched.
    np.testing.assert_allclose(reloaded, perturbed, rtol=0, atol=1e-5)

    # And it is emphatically NOT the pure-equilibrium field the old, broken
    # path produced -- otherwise this test would pass with the bug still in.
    # Compared on the interior, where the ghost-fill convention cannot muddy it.
    rho_ref, u_ref, v_ref, w_ref = _lattice_macros_from_state(state)
    equilibrium = _build_equilibrium_restart_distribution(
        rho_xyz=rho_ref, u_xyz=u_ref, v_xyz=v_ref, w_xyz=w_ref, ibgk=3
    )
    assert np.abs(reloaded[:, 1:-1, 1:-1, 1:-1] - equilibrium).max() > 1e-4


def test_an_absent_restart_is_reported_not_silently_skipped(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        loaded = _try_load_restart_distribution(
            restart_file=tmp_path / "restart" / "restart_0000_000001.uf",
            nx=NX,
            ny=NY,
            nz=NZ,
        )
    assert loaded is None
    assert "No restart template" in caplog.text


def test_a_restart_for_another_grid_is_refused_by_its_own_header(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A well-formed restart for a different grid is a mismatch, not corruption.

    It has to be named as one: reading it into this grid would reinterpret every
    population, so the guard fires on the header the file carries -- which is
    also why the payload below is sized consistently with that header, leaving
    the header as the only thing wrong with the file.
    """
    restart = tmp_path / "restart" / restart_file_name(1)
    other_nx = NX + 1
    _write_record(
        restart,
        (other_nx, NY, NZ, NL),
        np.zeros(_payload_size(nx=other_nx), dtype=np.float32),
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        loaded = _try_load_restart_distribution(
            restart_file=restart, nx=NX, ny=NY, nz=NZ
        )

    assert loaded is None
    assert "describes a" in caplog.text
    assert f"{other_nx}x{NY}x{NZ}" in caplog.text


def test_a_restart_with_the_wrong_payload_length_is_refused(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Right header, wrong amount of data behind it -- e.g. a D3Q19 build."""
    restart = tmp_path / "restart" / restart_file_name(1)
    short = _payload_size() - NL
    _write_record(restart, (NX, NY, NZ, NL), np.zeros(short, dtype=np.float32))

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        loaded = _try_load_restart_distribution(
            restart_file=restart, nx=NX, ny=NY, nz=NZ
        )

    assert loaded is None
    assert f"holds {short} values, expected {_payload_size()}" in caplog.text


@pytest.mark.parametrize(  # type: ignore[misc]
    "corruption",
    ["truncated", "stub", "ragged"],
    ids=["truncated-mid-record", "shorter-than-the-header", "non-integral-payload"],
)
def test_a_corrupt_restart_is_refused_and_logged(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture, corruption: str
) -> None:
    """The catch-all stays: a member killed mid-write leaves exactly this.

    What must not happen again is this being the path *every* call takes, which
    is why the round-trip test above exists beside it.
    """
    good = tmp_path / "restart" / restart_file_name(1)
    _write_record(good, (NX, NY, NZ, NL), np.ones(_payload_size(), dtype=np.float32))
    if corruption == "truncated":
        # Full record marker, real header, half the populations: the file a
        # process killed during ``write_record`` leaves behind.
        good.write_bytes(good.read_bytes()[: 20 + 4 * (_payload_size() // 2)])
    elif corruption == "stub":
        good.write_bytes(b"\x00\x01\x02")
    else:
        # A record length that is not a header plus whole float32 values. The
        # grid header still matches, so this can only be caught by the framing.
        raw = bytearray(good.read_bytes())
        raw[:4] = np.uint32(4 * 4 + 4 * _payload_size() + 2).tobytes()
        good.write_bytes(bytes(raw))

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        loaded = _try_load_restart_distribution(restart_file=good, nx=NX, ny=NY, nz=NZ)

    assert loaded is None
    assert "Cannot read the restart template" in caplog.text


def test_the_write_read_pair_agree_on_the_record_layout(
    tmp_path: pathlib.Path,
) -> None:
    """Header, payload and the two record markers, counted in bytes.

    The reader derives the payload length from the record's own leading marker,
    so this pins the arithmetic it depends on: a Fortran record here is the
    4-byte marker, four int32, the float32 populations, and the closing marker.
    """
    dirs = _make_dirs(tmp_path)
    iteration = write_restart_file_from_xarray(state=_make_state(), dirs=dirs)
    restart = dirs.experiment_dir / "restart" / restart_file_name(iteration)

    header_and_payload = 4 * 4 + 4 * _payload_size()
    assert restart.stat().st_size == 4 + header_and_payload + 4
    marker = int(np.frombuffer(restart.read_bytes()[:4], dtype=np.uint32)[0])
    assert marker == header_and_payload


requires_lbm_sources = pytest.mark.skipif(
    not LBM_SRC.is_dir(),
    reason="the LBM submodule is not checked out, so its lattice cannot be read",
)


@requires_lbm_sources  # type: ignore[misc]
def test_the_python_lattice_matches_the_fortran_d3q27_set() -> None:
    """Reading the template makes the direction ORDER load-bearing.

    While the warm start was pure equilibrium, a permuted lattice would only
    have relabelled directions consistently. Now the writer forms
    ``f - feq(template) + feq(target)`` direction by direction against
    populations the *solver* wrote, so the Python ``cxs/cys/czs/weights`` must
    be the Fortran's ``mod_D3Q27setup`` set in the Fortran's order or the
    subtraction cancels the wrong things.
    """
    text = (LBM_SRC / "mod_D3Q27setup.F90").read_text()

    def _fortran_ints(name: str) -> list[int]:
        """The 27-entry definition of ``name``; D3Q19/D2Q9 share the file."""
        matches = re.findall(
            rf"integer,\s*parameter\s*::\s*{name}\(1:nl\)\s*=\s*\[([^\]]*)\]", text
        )
        values = [[int(v) for v in m.replace("\n", "").split(",")] for m in matches]
        of_27 = [v for v in values if len(v) == NL]
        assert of_27, f"no 27-entry {name} found in mod_D3Q27setup.F90"
        return of_27[0]

    # The Python copies live inside the two builder functions rather than at
    # module scope, so probe them the way the physics does. At rest with unit
    # density the equilibrium is exactly the weights ...
    ones = np.ones((1, 1, 1), dtype=np.float32)
    zeros = np.zeros((1, 1, 1), dtype=np.float32)
    feq_rest = _build_equilibrium_restart_distribution(
        rho_xyz=ones, u_xyz=zeros, v_xyz=zeros, w_xyz=zeros, ibgk=3
    )
    np.testing.assert_allclose(
        feq_rest[:, 0, 0, 0],
        np.array(
            [8.0 / 27.0] + [2.0 / 27.0] * 6 + [1.0 / 54.0] * 12 + [1.0 / 216.0] * 8,
            dtype=np.float32,
        ),
        rtol=1e-6,
    )

    # ... and a distribution holding a single unit population has exactly that
    # direction's own velocity vector, which isolates one lattice direction at
    # a time and compares it against the Fortran's own list.
    for name, axis in (("cxs_h", 0), ("cys_h", 1), ("czs_h", 2)):
        expected = np.array(_fortran_ints(name), dtype=np.float32)
        recovered = []
        for direction in range(NL):
            f = np.zeros((NL, 1, 1, 1), dtype=np.float32)
            f[direction] = 1.0
            macros = _compute_macrovars_from_distribution(f)
            recovered.append(float(macros[axis + 1][0, 0, 0]))
        np.testing.assert_allclose(np.array(recovered), expected, rtol=0, atol=0)


def test_the_blanking_mask_gives_the_new_state_to_the_fluid_cells(
    tmp_path: pathlib.Path,
) -> None:
    """The mask's orientation, on its first-ever live execution.

    ``fluid_mask_zyx`` is computed unconditionally by the writer but consulted
    only inside ``if template_f is not None:`` -- the branch that was dead until
    the template read was fixed. So the convention "blanking non-zero == solid"
    has never actually run. Inverted, it would hand the assimilated state to the
    cells inside buildings and leave the flow field frozen at the template, over
    the whole domain, with nothing raising.

    A warm-start smoke run cannot catch this: there the template is the solver's
    own restart at the same iteration as the state handed back, so the two fields
    are near-identical and either orientation looks right. This makes them differ
    on purpose -- a quiescent template against a moving state -- and then asks
    where the motion landed.
    """
    dirs = _make_dirs(tmp_path)

    # Template: at rest everywhere.
    quiescent = _make_state()
    for name in ("u", "v", "w"):
        quiescent[name] = (quiescent[name].dims, np.zeros_like(quiescent[name].values))
    iteration = write_restart_file_from_xarray(state=quiescent, dirs=dirs)
    restart = dirs.experiment_dir / "restart" / restart_file_name(iteration)

    # State: moving, everywhere -- including inside the solid, which is what an
    # interpolated or assimilated field actually looks like. A solid slab over
    # the lower half of x makes a swap unmistakable.
    moving = _make_state()
    blanking = np.zeros((NZ, NY, NX), dtype=np.float32)
    blanking[:, :, : NX // 2] = 1.0  # non-zero == solid
    moving["blanking"] = (("z", "y", "x"), blanking)
    solid_zyx = blanking > 0.5

    second = write_restart_file_from_xarray(
        state=moving, dirs=dirs, restart_iteration=iteration
    )
    reloaded = _try_load_restart_distribution(
        restart_file=dirs.experiment_dir / "restart" / restart_file_name(second),
        nx=NX,
        ny=NY,
        nz=NZ,
    )
    assert reloaded is not None
    assert restart.exists()

    # Macros come back as (x, y, z) including the ghost shell; compare the
    # interior against the (z, y, x) state.
    _, u_xyz, _, _ = _compute_macrovars_from_distribution(reloaded)
    u_zyx = np.transpose(np.asarray(u_xyz)[1:-1, 1:-1, 1:-1], (2, 1, 0))
    wanted = np.transpose(_lattice_macros_from_state(moving)[1], (2, 1, 0))

    # Fluid cells take the state...
    np.testing.assert_allclose(u_zyx[~solid_zyx], wanted[~solid_zyx], rtol=0, atol=1e-5)
    # ...and solid cells keep the template's rest state. Asserted as a distinct
    # value, not merely "different": if the mask were inverted these two
    # assertions would swap, and only checking one of them would still pass.
    assert np.abs(u_zyx[solid_zyx]).max() < 1e-5
    assert np.abs(wanted[solid_zyx]).max() > 1e-3, "the fixture must actually differ"
