"""Digital-filter inflow turbulence shared by the uDALES and PALM backends.

Both ``pyudales`` (driver planes, ``BCxm=3``) and ``pypalm`` (``inflow_plane_*``
in the dynamic driver, ``turbulent_inflow_method='read_from_file'``) impose a
synthetic turbulent inflow built from the same Xie & Castro (2008) digital
filter: white noise correlated in y (periodic), in z (bounded) and in time by an
AR(1) recursion whose time scale follows from Taylor's hypothesis. The backends
only differ in how the planes are laid out on disk, so the generator, the
per-member seed and the per-member clock that makes window *n* continue window
*n-1* live here, once.

Each record's white noise comes from a counter-based Philox stream keyed on
``(seed, record index)``, so two backends asked for the same seed, plane shape
and record grid get the *same* inflow realisation.
"""

import hashlib
import json
import logging
import math
import pathlib
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class MemberDirs(Protocol):
    """The two directory attributes the per-member clock needs."""

    experiment_dir: pathlib.Path
    experiment_name: str


# Rough neutral-ABL anisotropy: the cross-stream and vertical fluctuations carry
# less energy than the streamwise one. A full Lund/Cholesky Reynolds-stress
# decomposition is the phase-2 refinement; this single ratio is enough for the
# fluctuations to survive the adaptation fetch, which is what phase 1 is for.
CROSS_COMPONENT_ANISOTROPY = 0.7


# The member's physical clock lives on disk, NOT just on the ForwardModel.
# `BaseEnsembleForwardModel._run_parallel` submits `model.__call__` to a
# ProcessPoolExecutor (forkserver): the member is pickled into a worker, so any
# attribute the worker mutates is discarded when it exits. An in-memory counter
# would therefore silently reset to 0 every window under
# `num_parallel_processes > 1` -- restarting the AR(1) history each window,
# which is exactly what the counter exists to prevent, with no error to show
# for it. The file sits in experiment_dir (shared disk, per member) rather than
# inside warmstart_carry/, which `store_carry` wipes with rmtree on every run.
ELAPSED_TIME_FILENAME = "inlet_turbulence_clock.json"


def elapsed_time_path(dirs: MemberDirs) -> pathlib.Path:
    """Path of the member's persisted physical clock."""
    return dirs.experiment_dir / ELAPSED_TIME_FILENAME


def read_elapsed_time(dirs: MemberDirs, default: float = 0.0) -> float:
    """Seconds of turbulence history this member has already consumed.

    Every successful ``run_single`` advances this clock. With the opt-in
    discrepancy replay lifecycle, the assimilation layer restores the
    window-start clock before each forecast and accepts only the final
    endpoint. Repeated forecasts then see the same synthetic inlet history.
    Outside that lifecycle the legacy behavior remains: every forecast advances
    the offset, including repeated ESMDA iterations over one physical interval.
    In that case the clock indexes stationary inlet history rather than physical
    elapsed time.

    The stored ``experiment_name`` is checked, not decorative: ensemble members
    are built by ``create_new_forward_model``, which mirrors the template's
    experiment dir with an unfiltered ``copy_files``. A clock left in the
    template would otherwise be inherited by every member — one member's history
    silently becoming everyone's. ``reset_elapsed_time`` is the primary defence;
    this is the backstop for any other copy route.
    """
    path = elapsed_time_path(dirs)
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text())
        elapsed_time = float(payload["elapsed_time"])
        owner = payload.get("experiment_name")
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        logger.warning(
            "Could not read %s; restarting the inlet-turbulence clock at %.1f s. "
            "The injected turbulence will be discontinuous across this window.",
            path,
            default,
        )
        return default

    if owner is not None and owner != dirs.experiment_name:
        logger.warning(
            "Ignoring inlet-turbulence clock in %s: it belongs to member '%s', "
            "not '%s' (a stale file copied from another experiment dir). "
            "Restarting this member's clock at %.1f s.",
            path,
            owner,
            dirs.experiment_name,
            default,
        )
        return default
    return elapsed_time


def reset_elapsed_time(dirs: MemberDirs) -> None:
    """Discard any persisted clock so a freshly built model starts at t=0.

    The clock is scoped to a ``ForwardModel``'s lifetime — it is persisted only
    so forkserver workers can see it *within* a run, never to resume a later
    one. Without this reset, a run reusing ``paths.experiment_dir`` (or an
    ensemble whose template dir happens to hold a clock) would start mid-history
    and produce a different inlet realisation from an identical config and seed.
    The turbulence is stationary, so that is statistically harmless — and
    exactly the kind of irreproducibility that is expensive to track down.
    """
    elapsed_time_path(dirs).unlink(missing_ok=True)


def write_elapsed_time(dirs: MemberDirs, elapsed_time: float) -> None:
    """Persist the member's physical clock (see :data:`ELAPSED_TIME_FILENAME`)."""
    payload = {
        "elapsed_time": float(elapsed_time),
        "experiment_name": dirs.experiment_name,
    }
    elapsed_time_path(dirs).write_text(json.dumps(payload, indent=2))


def copy_elapsed_time(src_dirs: MemberDirs, dst_dirs: MemberDirs) -> bool:
    """Copy the donor's clock to a member that was substituted after failing.

    The failed member inherits the donor's *state* (and its warmstart carry), so
    it must inherit the donor's clock too — otherwise it would sit permanently
    offset from the flow it is now carrying, and its inlet turbulence would jump
    at the substitution and stay out of step for the rest of the rollout.
    Returns ``True`` if a clock was copied.
    """
    if not elapsed_time_path(src_dirs).exists():
        return False
    write_elapsed_time(dst_dirs, read_elapsed_time(src_dirs))
    return True


def derive_seed(experiment_name: str) -> int:
    """Stable per-member RNG seed derived from the experiment name.

    ``hash()`` is salted per interpreter, so a member's fluctuation sequence
    would differ between the parent process and a ``forkserver`` worker. A digest
    keeps every window of every member reproducible from its name alone, which is
    what makes the cross-window continuation property (see module docstring)
    hold.
    """
    digest = hashlib.blake2b(experiment_name.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


# --- Digital filter ----------------------------------------------------------


def filter_coefficients(n_cells: float) -> np.ndarray:
    """Xie & Castro (2008) exponential filter coefficients, ``sum(b**2) == 1``.

    ``b_k ~ exp(-pi*|k| / (2*n))`` over the support ``|k| <= 2n``, where
    ``n = L/dx`` is the integral length scale in cells. Normalising to unit
    2-norm makes the filtered field have unit pointwise variance when the input
    is standard white noise, so the caller can scale straight to a target rms.
    """
    n = max(1, int(round(n_cells)))
    k = np.arange(-2 * n, 2 * n + 1, dtype=float)
    b = np.exp(-np.pi * np.abs(k) / (2.0 * n))
    return b / np.sqrt(np.sum(b**2))


def _filter_periodic(field: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution of ``field`` with ``b`` along the last (y) axis.

    y is periodic in every uDALES run this wrapper produces (``BCym=1`` is
    hardcoded in ``_apply_boundary_condition``), so wrapping is not an
    approximation — it is what makes the j ghost columns the natural periodic
    image of the interior. Periodicity is also what lets this be an FFT product
    rather than ``ny`` full-array ``np.roll`` passes, which matters once the
    calibration runs use production-sized planes.

    When the filter support is wider than the domain the wrapped coefficients
    add, and their 2-norm is no longer 1; renormalising by the *wrapped* norm
    keeps the output variance at 1 for any length scale.
    """
    ny = field.shape[-1]
    half = (b.size - 1) // 2
    wrapped = np.zeros(ny)
    for index, coeff in enumerate(b):
        wrapped[(index - half) % ny] += coeff
    wrapped /= np.linalg.norm(wrapped)

    # out[j] = sum_s wrapped[s] * field[j-s] -- a circular convolution, so a
    # plain product in Fourier space.
    spectrum = np.fft.rfft(field, axis=-1) * np.fft.rfft(wrapped)
    return np.fft.irfft(spectrum, n=ny, axis=-1)


def _filter_bounded(field: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convolve ``field`` with ``b`` along the second-to-last (z) axis.

    z is *not* periodic, so the kernel is truncated at the ground and at the
    domain top. Truncation would otherwise shrink the variance near both edges;
    dividing each level by the 2-norm of the coefficients that actually reached
    it restores unit variance exactly, at the cost of a slightly shorter
    effective correlation length in the edge cells.
    """
    nz = field.shape[-2]
    half = (b.size - 1) // 2

    out = np.zeros_like(field)
    norm = np.zeros(nz)
    for index, coeff in enumerate(b):
        shift = index - half
        dst_lo, dst_hi = max(0, -shift), min(nz, nz - shift)
        if dst_lo >= dst_hi:
            continue
        out[..., dst_lo:dst_hi, :] += (
            coeff * field[..., dst_lo + shift : dst_hi + shift, :]
        )
        norm[dst_lo:dst_hi] += coeff**2
    return out / np.sqrt(norm)[:, None]


# Relative precision the bounded burn-in reproduces the exact AR(1) state to.
# The recursion forgets its history geometrically (a**n), so replaying only the
# last `log(eps)/log(a)` records before the window is indistinguishable from
# replaying all of them -- and turns an O(windows**2) rollout into O(windows).
#
# 1e-12 costs ~2900 records at the production shape (L_x=50 m, U=3 m/s,
# dtdriver=0.1 s), which is bounded but not free. Relaxing to 1e-6 halves it at
# no physical cost -- the AR(1) state is a random draw, not a measurement, so
# reproducing it to 6 digits is already far beyond what matters. Worth doing if
# the calibration runs feel slow.
AR1_BURN_IN_EPSILON = 1e-12


def ar1_burn_in_records(ar1_coefficient: float) -> int:
    """How many pre-window records must be replayed to forget the initial state."""
    if ar1_coefficient <= 0.0:
        return 0
    if ar1_coefficient >= 1.0:  # pragma: no cover - guarded by validation
        raise ValueError(f"ar1_coefficient must be < 1, got {ar1_coefficient}")
    return int(math.ceil(math.log(AR1_BURN_IN_EPSILON) / math.log(ar1_coefficient)))


def _record_noise(
    seed: int, index: int, shapes: dict[str, tuple[int, int]]
) -> dict[str, np.ndarray]:
    """Draw record ``index``'s white noise from a counter-based stream.

    Philox is indexed by ``counter``, so record *n*'s noise depends only on
    ``(seed, n)`` — never on how many records were drawn before it. That is what
    lets the burn-in above start at an arbitrary offset and still reproduce the
    same history bit-for-bit; with a sequential ``default_rng`` the draw order
    *is* the state, so any truncation would change every subsequent record.

    The ``<< 64`` stride is not cosmetic. A Philox stream advances its *own*
    counter as it draws, so seeding consecutive records at ``counter = n`` makes
    record *n* walk straight into record *n+1*'s block after a few hundred
    values — the two share noise, and the resulting spurious correlation shows
    up as a lag-2 autocorrelation well above the AR(1) prediction. Giving each
    record 2**64 blocks of its own makes the streams genuinely independent.
    """
    rng = np.random.Generator(np.random.Philox(key=seed, counter=index << 64))
    return {name: rng.standard_normal(shape) for name, shape in shapes.items()}


def generate_fluctuation_fields(
    *,
    seed: int,
    ar1_coefficient: float,
    n_records: int,
    start_index: int,
    shapes: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Return unit-variance, time-correlated *white* fields per component.

    The AR(1) recursion ``b_n = a*b_{n-1} + sqrt(1-a**2)*eta_n`` is seeded at a
    bounded burn-in offset before ``start_index`` (``b = eta``, already the
    stationary distribution) and only the last ``n_records`` states are kept.
    Replaying that prefix is what makes window *n* continue window *n-1*; the
    burn-in bound keeps the cost per window constant instead of growing with the
    rollout. The spatial filter is linear and time-invariant, so it commutes
    with the recursion and is applied afterwards, to the kept records only.

    Args:
        seed: Per-member RNG seed (see :func:`derive_seed`).
        ar1_coefficient: ``a`` in the recursion above, in ``[0, 1)``.
        n_records: Number of records to return.
        start_index: Global record index of the first returned record.
        shapes: ``{component: (nz, ny)}`` interior plane shapes.

    Returns:
        ``{component: array of shape (n_records, nz, ny)}``, unit variance.
    """
    root = math.sqrt(max(0.0, 1.0 - ar1_coefficient**2))
    burn_in = min(start_index, ar1_burn_in_records(ar1_coefficient))
    first = start_index - burn_in

    state: dict[str, np.ndarray] = {}
    kept: dict[str, list[np.ndarray]] = {name: [] for name in shapes}

    for index in range(first, start_index + n_records):
        noise = _record_noise(seed, index, shapes)
        for name in shapes:
            if index == first:
                state[name] = noise[name]
            else:
                state[name] = ar1_coefficient * state[name] + root * noise[name]
            if index >= start_index:
                kept[name].append(state[name])

    return {name: np.stack(records) for name, records in kept.items()}


def filtered_fluctuations(
    *,
    seed: int,
    start_index: int,
    times: np.ndarray,
    speed: np.ndarray,
    n_z: int,
    n_y: int,
    dz: float,
    dy: float,
    length_scale_x: float,
    length_scale_y: float,
    length_scale_z: float,
) -> dict[str, np.ndarray]:
    """Unit-variance, space- and time-correlated u/v/w inflow fluctuations.

    u and v live on the ``n_z`` cell centres, w on the ``n_z + 1`` faces from
    the ground to the top (``(k + 0.5) * dz`` and ``k * dz``). The caller scales
    them to its target rms and lays them out for its solver. Taylor's
    hypothesis turns ``length_scale_x`` into the AR(1) time scale
    ``T = length_scale_x / mean(|speed|)``, ``a = exp(-pi * dt / (2T))``.

    Args:
        seed: Per-member RNG seed (see :func:`derive_seed`).
        start_index: Global record index of ``times[0]``.
        times: Record times; their spacing is the AR(1) step.
        speed: Inflow speed per record (anything ``np.mean`` reduces).
        n_z, n_y: Cell counts of the inflow plane.
        dz, dy: Cell sizes in m.
        length_scale_x, length_scale_y, length_scale_z: Integral length scales
            in m.

    Returns:
        ``{"u": (n, n_z, n_y), "v": (n, n_z, n_y), "w": (n, n_z + 1, n_y)}``.
    """
    n_records = times.size
    u_ref = float(np.mean(np.abs(speed)))
    if u_ref <= 0.0:
        ar1 = 0.0
    else:
        t_scale = length_scale_x / u_ref
        dt = float(times[1] - times[0]) if n_records > 1 else length_scale_x
        ar1 = math.exp(-math.pi * dt / (2.0 * t_scale))

    white = generate_fluctuation_fields(
        seed=seed,
        ar1_coefficient=ar1,
        n_records=n_records,
        start_index=start_index,
        shapes={"u": (n_z, n_y), "v": (n_z, n_y), "w": (n_z + 1, n_y)},
    )

    b_y = filter_coefficients(length_scale_y / dy)
    b_z = filter_coefficients(min(length_scale_z / dz, float(n_z + 1)))

    return {
        name: _filter_periodic(_filter_bounded(field, b_z), b_y)
        for name, field in white.items()
    }


def validate_time_step_divides_window(runtime_total: float, time_step: float) -> None:
    """Require the window to be a whole number of driver records.

    ``start_index = round(window_start_time / time_step)`` is how a window picks
    its slice of the turbulence history. If the window length is not a multiple
    of ``time_step`` the record grid slips against physical time by up to half a
    record per window and the error accumulates over a rollout — silently. Fail
    loudly instead, which is cheaper than a drifting inlet nobody notices.

    Deliberately a *runtime* check rather than part of
    the backends' ``validate_inlet_turbulence``: the window length is
    ``spinup_time + simulation_time`` for a cold window and ``simulation_time``
    for a warm one, so it is not known until the call. That means a bad
    ``time_step`` surfaces on the first ``run_single`` rather than at
    construction — after the ensemble has been built, but still before any
    solver time is spent.
    """
    if time_step <= 0.0 or runtime_total <= 0.0:
        return
    records = runtime_total / time_step
    if abs(records - round(records)) > 1e-6:
        suggestions = [
            step
            for step in (0.1, 0.2, 0.25, 0.5, 1.0)
            if abs(runtime_total / step - round(runtime_total / step)) <= 1e-6
        ]
        hint = (
            f" Divisors of {runtime_total} s include: "
            + ", ".join(f"{s}" for s in suggestions)
            + "."
            if suggestions
            else ""
        )
        raise ValueError(
            f"inlet_turbulence.time_step={time_step} s does not divide the window "
            f"length {runtime_total} s ({records:.4f} records). The driver record "
            "grid would slip against physical time by up to half a record per "
            f"window and accumulate over a rollout.{hint}"
        )
