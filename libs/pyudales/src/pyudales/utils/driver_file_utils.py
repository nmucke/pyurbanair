"""Binary I/O for uDALES precursor/driver inlet planes (``&DRIVER idriver=2``).

No physics lives here — this module owns exactly one thing: the on-disk layout
that ``u-dales/src/moddriver.f90::readdriverfile`` expects.

Layout contract (all line references to ``u-dales/src/moddriver.f90``)
---------------------------------------------------------------------
* **File names** (``:786-872``). ``tdriver_DDD.NNN`` plus ``udriver_``/
  ``vdriver_``/``wdriver_`` with the same suffixes, where ``DDD`` is the
  y-decomposition driver id and ``NNN`` is ``driverjobnr`` (both ``i3.3``).
  ``tdriver`` hardcodes ``DDD='000'``; the velocity files use
  ``cdriverid``, and ``driverid = mod(myidy, nprocy)`` (``initdriver:62``).
  The wrapper pins ``nprocy=1`` (``utils.ncpu_utils``), so ``driverid`` is 0 on
  every rank and a single file set serves the whole run — only the x-rank that
  owns the inlet (``ibrank``) reads it.
* **Encoding**. ``form='unformatted', access='direct'`` — a raw stream with no
  record markers. The build is gfortran with ``-fdefault-real-8``
  (``u-dales/CMakeLists.txt:42``), so every value is a little-endian float64 and
  record *n* starts at exactly ``(n-1) * record_bytes``.
* **Record shape**. Each velocity record is one inlet plane read as
  ``((store(j,k), j=jb-jh,je+jh), k=kb-kh,ke+kh)`` (``:833``) — *j fastest*.
  With the cd2 momentum advection this build is limited to
  (``docs/pyudales.md`` §1), ``jh = kh = 1`` (``modglobal.f90:575-582``), so a
  plane is ``(ktot+2) x (jtot+2)`` values. ``tdriver`` records are a single
  float64 each, in ascending time order.

The C-order array shape ``(ktot+2, jtot+2)`` is what makes ``ndarray.tofile()``
emit j-fastest, matching the Fortran read loop with no transpose.
"""

import logging
import os
import pathlib
import sys

import numpy as np

logger = logging.getLogger(__name__)

# Ghost-cell widths of the inlet plane, i.e. jh/kh in modglobal.f90:575-582.
# The uDALES build only implements cd2 momentum advection (docs/pyudales.md §1),
# which is the branch that sets both to 1. A kappa-scheme build would use jh=2
# and these files would need regenerating -- hence the named constants.
PLANE_HALO_J = 1
PLANE_HALO_K = 1

# float64 everywhere: the solver is built with -fdefault-real-8.
DRIVER_DTYPE = np.dtype("<f8")

_DRIVER_ID = "000"  # nprocy=1 -> driverid = mod(myidy, nprocy) = 0 on every rank


def plane_shape(jtot: int, ktot: int) -> tuple[int, int]:
    """Return the C-order shape ``(ktot+2*kh, jtot+2*jh)`` of one inlet plane."""
    return (ktot + 2 * PLANE_HALO_K, jtot + 2 * PLANE_HALO_J)


def plane_record_bytes(jtot: int, ktot: int) -> int:
    """Byte size of one velocity record (one full inlet plane, halos included)."""
    nk, nj = plane_shape(jtot, ktot)
    return int(nk * nj * DRIVER_DTYPE.itemsize)


def driver_file_names(driverjobnr: int) -> dict[str, str]:
    """Return the ``{variable: filename}`` map for one driver job number."""
    if not 0 <= driverjobnr <= 999:
        raise ValueError(
            f"driverjobnr must be a 3-digit integer in [0, 999], got {driverjobnr}: "
            "moddriver.f90 formats it with i3.3, so a wider value would build a "
            "file name the solver cannot open."
        )
    suffix = f"{driverjobnr:03d}"
    return {
        "t": f"tdriver_{_DRIVER_ID}.{suffix}",
        "u": f"udriver_{_DRIVER_ID}.{suffix}",
        "v": f"vdriver_{_DRIVER_ID}.{suffix}",
        "w": f"wdriver_{_DRIVER_ID}.{suffix}",
    }


def _write_atomic(path: pathlib.Path, array: np.ndarray) -> None:
    """Write ``array`` to ``path`` via a temp file + rename.

    A member killed mid-write (the dt watchdog does exactly that) must not leave
    a half-written plane file behind: the next run would read it as valid data
    and inject garbage at the inlet, or trip readdriverfile's silent short read.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as handle:
        array.astype(DRIVER_DTYPE, copy=False).tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def write_driver_files(
    experiment_dir: pathlib.Path,
    driverjobnr: int,
    times: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> dict[str, pathlib.Path]:
    """Write the four ``*driver_000.NNN`` files uDALES reads under ``idriver=2``.

    Args:
        experiment_dir: Directory holding ``namoptions.<expnr>``. ``local_execute.sh``
            copies this whole directory into the run's output dir before invoking
            ``mpiexec`` there, so the solver finds the files under their bare
            relative names.
        driverjobnr: ``&DRIVER driverjobnr`` — the 3-digit file-name suffix.
        times: ``(n,)`` ascending record times in seconds (solver clock).
        u, v, w: ``(n, ktot+2, jtot+2)`` inlet planes, C order (j fastest on disk).

    Returns:
        ``{variable: path}`` for the four files written.
    """
    if sys.byteorder != "little":
        raise RuntimeError(
            "Driver files are written as little-endian float64 to match the "
            f"gfortran build, but this host is {sys.byteorder}-endian."
        )

    experiment_dir = pathlib.Path(experiment_dir)
    times = np.asarray(times, dtype=DRIVER_DTYPE)
    if times.ndim != 1:
        raise ValueError(f"times must be 1-D, got shape {times.shape}")
    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "driver record times must be strictly ascending: drivergen locates "
            "the bracketing records with minloc(abs(storetdriver-timee)) and "
            "interpolates between neighbours (moddriver.f90:219-241)."
        )

    planes = {name: np.asarray(a) for name, a in (("u", u), ("v", v), ("w", w))}
    for name, array in planes.items():
        if array.ndim != 3 or array.shape[0] != times.size:
            raise ValueError(
                f"{name} must have shape (n_records, ktot+2, jtot+2) with "
                f"n_records={times.size}, got {array.shape}"
            )
    expected = planes["u"].shape
    for name, array in planes.items():
        if array.shape != expected:
            raise ValueError(
                f"u/v/w must share a shape; got {expected} and {array.shape}"
            )

    names = driver_file_names(driverjobnr)
    written = {"t": experiment_dir / names["t"]}
    _write_atomic(written["t"], times)
    for name, array in planes.items():
        path = experiment_dir / names[name]
        # np.ascontiguousarray guarantees the C-order (k, j) traversal that puts
        # j fastest on disk, matching the Fortran implied-do read loop.
        _write_atomic(path, np.ascontiguousarray(array, dtype=DRIVER_DTYPE))
        written[name] = path

    logger.info(
        "Wrote uDALES driver files %s in %s (%d records, plane %dx%d, %.1f MB total)",
        ", ".join(sorted(names.values())),
        experiment_dir,
        times.size,
        expected[1],
        expected[2],
        3 * times.size * expected[1] * expected[2] * 8 / 1e6,
    )
    return written


def read_driver_files(
    experiment_dir: pathlib.Path,
    driverjobnr: int,
    jtot: int,
    ktot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read back the driver files written by :func:`write_driver_files`.

    The inverse of the write path, used by the tests and for debugging a run
    whose inlet looks wrong. Returns ``(times, u, v, w)`` with the planes shaped
    ``(n, ktot+2, jtot+2)``.
    """
    experiment_dir = pathlib.Path(experiment_dir)
    names = driver_file_names(driverjobnr)
    nk, nj = plane_shape(jtot, ktot)
    record_bytes = plane_record_bytes(jtot, ktot)

    times = np.fromfile(experiment_dir / names["t"], dtype=DRIVER_DTYPE)
    n_records = times.size

    planes = []
    for name in ("u", "v", "w"):
        path = experiment_dir / names[name]
        size = path.stat().st_size
        if size != n_records * record_bytes:
            raise ValueError(
                f"{path.name} is {size} B but {n_records} records of "
                f"{record_bytes} B were expected "
                f"(jtot={jtot}, ktot={ktot}, halos jh=kh=1)."
            )
        planes.append(np.fromfile(path, dtype=DRIVER_DTYPE).reshape(n_records, nk, nj))

    return times, planes[0], planes[1], planes[2]
