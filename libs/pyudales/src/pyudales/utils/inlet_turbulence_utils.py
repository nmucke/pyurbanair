"""Turbulent inlet for uDALES via **synthetic driver planes** (``BCxm=3``).

uDALES v2.2.0 ships two inlet-turbulence routes. The Lund (1998) recycling
generator (``iinletgen``) is dead code — not in any namelist, never called, its
output never read (see ``docs/pyudales.md`` §6.1, still true and still tested).
The *precursor/driver* route is fully wired end to end: ``BCxm=3`` sets
``idriver=2`` (``modstartup.f90:831-835``), ``readdriverfile`` loads a stack of
inlet planes at startup, ``drivergen`` interpolates them in time, ``xmi_driver``
writes them onto the west face, and ``bcpup`` pins the u-face exactly
(``pup(ib,:,:) = u0driver*rk3coefi``, ``modboundary.f90:1239-1247``).

This module drives that second route **without a precursor run**: the planes are
synthesised in Python from the same ESMDA parameters the nudging path uses
(``velocity_magnitude``, ``inflow_angle``, ``vertical_inflow_exponent``), plus
digital-filter turbulent fluctuations in the style of Xie & Castro (2008) — the
same authors as the benchmark case. Design record:
``docs/plans/udales_inlet_turbulence.md``.

Why synthetic rather than a real precursor
------------------------------------------
A per-member precursor would double ensemble cost *and* disconnect the estimated
inflow: under ``BCxm=3`` the west face no longer reads ``uprof``, so whatever the
precursor produced would replace the ESMDA signal. Generating the planes here
keeps ``velocity_magnitude``/``inflow_angle``/α in the loop — they set the plane
means — which is precisely the objection that made the ``iinletgen`` route a
``ValueError`` in the first place. A "precursor library" (record once, replay and
rescale) remains a future option behind the same file interface.

Why a digital filter and not white noise
----------------------------------------
The pressure projection plus SGS dissipation annihilate spatially uncorrelated
noise within a couple of cells. Fluctuations only survive and roll up into real
turbulence if they carry coherent structures, so the noise is correlated in y, z
(exponential kernel ``exp(-pi*r/(2L))``) and in time (an AR(1) recursion whose
timescale follows from Taylor's hypothesis, ``T = length_scale_x / U_ref``).

Interaction with nudging
------------------------
When this knob is on, volume nudging is switched **off** (``lnudge=.false.``,
``ltimedepnudge=.false.``). Under ``BCxm=3`` nudging no longer feeds the inlet,
and relaxing the interior toward a smooth mean profile on the ``tnudge=15 s``
scale would damp exactly the fluctuations being injected. Time variation of the
inflow is carried by the plane sequence itself, which is time-resolved.

Window continuity
-----------------
``ForwardModel._prepare_warmstart`` deliberately writes ``timee = 0`` into the
restart file (uDALES otherwise dies with ``timee >= runtime``), so every window
starts the solver clock at 0 and ``btime`` is always 0. The *physical* elapsed
time is therefore tracked by the wrapper and passed in as ``window_start_time``:
the AR(1) recursion is replayed from a fixed per-member seed up to that point and
only the current window's records are written, so window *n*'s planes are the
exact continuation of window *n-1*'s without persisting any filter state.
"""

import hashlib
import logging
import math
import pathlib
from typing import Any, Optional

import numpy as np
import xarray

from .dir_utils import DirectoryPaths
from .driver_file_utils import write_driver_files
from .file_update_utils import update_lscale_file_profile, update_prof_file_profile
from .inflow_utils import angle_to_velocity
from .namoptions_utils import NamoptionsFile
from .vertical_profile import build_profile_shape

logger = logging.getLogger(__name__)


# --- Config schema -----------------------------------------------------------
#
# `intensity` plays the role pylbm calls `amplitude` and pypalm calls
# `disturbance_amplitude`: unit-alike knobs with per-backend names, as with
# `sgs_constant`. The length scales default to roughly the Xie & Castro cube
# height -- a starting point to calibrate against, not a tuned value.
DEFAULT_INLET_TURBULENCE_CONFIG: dict[str, Any] = {
    "enabled": False,
    "intensity": 0.1,  # u'_rms / |U_mean(z)|
    "length_scale_y": 25.0,  # m, spanwise integral length scale
    "length_scale_z": 25.0,  # m, vertical integral length scale
    "length_scale_x": 50.0,  # m, streamwise -> AR(1) timescale via Taylor
    "time_step": 0.5,  # s, &DRIVER dtdriver
    "driverjobnr": 998,  # 3-digit driver file suffix
    "seed": None,  # None -> derived from the member's experiment name
    "lchunkread": None,  # None -> leave uDALES' default (.false.)
    "chunkread_size": None,
}
INLET_TURBULENCE_KEYS = frozenset(DEFAULT_INLET_TURBULENCE_CONFIG)

# Rough neutral-ABL anisotropy: the cross-stream and vertical fluctuations carry
# less energy than the streamwise one. A full Lund/Cholesky Reynolds-stress
# decomposition is the phase-2 refinement; this single ratio is enough for the
# fluctuations to survive the adaptation fetch, which is what phase 1 is for.
CROSS_COMPONENT_ANISOTROPY = 0.7

# Extra records written past the end of the run. `drivergen` HARD-STOPS the
# simulation when `timee` exceeds the last record time (moddriver.f90:236-241),
# so the coverage margin is not cosmetic.
DRIVER_TIME_MARGIN_RECORDS = 2


def is_inlet_turbulence_enabled(config: Optional[dict]) -> bool:
    """True only when an ``inlet_turbulence`` dict explicitly enables the knob.

    ``None``/``{}``/``enabled: false`` are all a strict no-op — nothing is
    written to namoptions and no driver files appear (CLAUDE.md no-op rule).
    """
    if not config:
        return False
    return bool(config.get("enabled", False))


def resolve_inlet_turbulence_config(config: Optional[dict]) -> dict[str, Any]:
    """Merge ``config`` over the defaults, warning about unknown keys."""
    resolved = dict(DEFAULT_INLET_TURBULENCE_CONFIG)
    if not config:
        return resolved
    unknown = set(config) - INLET_TURBULENCE_KEYS
    if unknown:
        logger.warning(
            "Ignoring unknown inlet_turbulence keys: %s (supported: %s)",
            ", ".join(sorted(unknown)),
            ", ".join(sorted(INLET_TURBULENCE_KEYS)),
        )
    resolved.update({k: v for k, v in config.items() if k in INLET_TURBULENCE_KEYS})
    return resolved


def validate_inlet_turbulence(
    config: Optional[dict],
    boundary_condition: str,
    experiment_name: Optional[str] = None,
) -> None:
    """Reject ``inlet_turbulence`` blocks uDALES cannot honour.

    The disabled path only warns about unknown keys; it never raises and never
    touches a file, so a default run stays byte-identical.
    """
    resolved = resolve_inlet_turbulence_config(config)
    if not is_inlet_turbulence_enabled(config):
        return

    if boundary_condition != "inflow_outflow":
        raise ValueError(
            "inlet_turbulence.enabled=true requires "
            "boundary_condition='inflow_outflow': under 'periodic' the x "
            "boundaries wrap (BCxm=1) so there is no inlet face for a turbulent "
            "inlet to be imposed on."
        )

    time_step = float(resolved["time_step"])
    if time_step <= 0.0:
        raise ValueError(
            f"inlet_turbulence.time_step must be > 0 s, got {time_step}; it is "
            "written to &DRIVER dtdriver and sets the driver record spacing."
        )

    intensity = float(resolved["intensity"])
    if intensity < 0.0:
        raise ValueError(
            f"inlet_turbulence.intensity must be >= 0, got {intensity} "
            "(0 means a laminar mean-profile inlet, which is legal)."
        )

    for key in ("length_scale_x", "length_scale_y", "length_scale_z"):
        value = float(resolved[key])
        if value <= 0.0:
            raise ValueError(
                f"inlet_turbulence.{key} must be > 0 m, got {value}; it is a "
                "digital-filter integral length scale."
            )

    driverjobnr = int(resolved["driverjobnr"])
    if not 0 <= driverjobnr <= 999:
        raise ValueError(
            f"inlet_turbulence.driverjobnr must be in [0, 999], got {driverjobnr}: "
            "moddriver.f90 formats it with i3.3."
        )
    if experiment_name is not None and _looks_like_int(experiment_name):
        if driverjobnr == int(experiment_name):
            raise ValueError(
                f"inlet_turbulence.driverjobnr={driverjobnr} collides with the "
                f"experiment number '{experiment_name}'. Keep them distinct so "
                "the synthesised planes can never be confused with the output of "
                "an idriver=1 recording run (which this wrapper never produces)."
            )

    chunk = resolved["chunkread_size"]
    if chunk is not None and int(chunk) <= 0:
        raise ValueError(f"inlet_turbulence.chunkread_size must be > 0, got {chunk}.")


def _looks_like_int(value: str) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
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
    image of the interior.

    When the filter support is wider than the domain the wrapped coefficients
    add, and their 2-norm is no longer 1; renormalising by the *wrapped* norm
    keeps the output variance at 1 for any length scale.
    """
    ny = field.shape[-1]
    half = (b.size - 1) // 2
    wrapped = np.zeros(ny)
    for index, coeff in enumerate(b):
        wrapped[(index - half) % ny] += coeff

    out = np.zeros_like(field)
    for shift, coeff in enumerate(wrapped):
        if coeff != 0.0:
            out += coeff * np.roll(field, shift, axis=-1)
    return out / np.linalg.norm(wrapped)


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


def generate_fluctuation_fields(
    *,
    seed: int,
    ar1_coefficient: float,
    n_records: int,
    start_index: int,
    shapes: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Return unit-variance, time-correlated *white* fields per component.

    The AR(1) recursion ``b_n = a*b_{n-1} + sqrt(1-a**2)*eta_n`` is run from
    global record 0 (``b_0 = eta_0``, which is already the stationary
    distribution) and only the last ``n_records`` states are kept. Replaying the
    skipped prefix is what makes window *n* continue window *n-1* — and it is
    cheap because the spatial filter is linear and time-invariant, so it commutes
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
    rng = np.random.default_rng(seed)
    root = math.sqrt(max(0.0, 1.0 - ar1_coefficient**2))
    total = start_index + n_records

    state = {name: np.zeros(shape) for name, shape in shapes.items()}
    kept: dict[str, list[np.ndarray]] = {name: [] for name in shapes}

    for index in range(total):
        for name, shape in shapes.items():
            eta = rng.standard_normal(shape)
            if index == 0:
                state[name] = eta
            else:
                state[name] = ar1_coefficient * state[name] + root * eta
            if index >= start_index:
                kept[name].append(state[name])

    return {name: np.stack(records) for name, records in kept.items()}


# --- Plane assembly ----------------------------------------------------------


def driver_time_grid(runtime_total: float, time_step: float) -> np.ndarray:
    """Record times covering ``[0, runtime_total]`` plus a safety margin.

    ``drivergen`` stops the whole simulation the moment ``timee`` exceeds the
    last record (``moddriver.f90:236-241``), so the grid deliberately overshoots
    by :data:`DRIVER_TIME_MARGIN_RECORDS` records.
    """
    if time_step <= 0.0:
        raise ValueError(f"time_step must be > 0, got {time_step}")
    n_records = (
        int(math.ceil(runtime_total / time_step - 1e-9))
        + 1
        + DRIVER_TIME_MARGIN_RECORDS
    )
    return np.arange(max(n_records, 2), dtype=float) * time_step


def build_driver_planes(
    *,
    jtot: int,
    ktot: int,
    zsize: float,
    ylen: float,
    times: np.ndarray,
    start_index: int,
    inflow_angle: np.ndarray,
    velocity_magnitude: np.ndarray,
    profile_config: Optional[dict],
    intensity: float,
    length_scale_y: float,
    length_scale_z: float,
    length_scale_x: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise the ``(n, ktot+2, jtot+2)`` u/v/w inlet planes.

    Pure function — no I/O, no namoptions. ``inflow_angle`` and
    ``velocity_magnitude`` are already sampled onto ``times``.

    Staggering follows uDALES' C-grid: u and v fluctuations are generated at cell
    centres ``zt``, w at the faces ``zm`` (``xmi_driver`` reads ``k = kb..ke`` for
    u/v and ``kb..ke+1`` for w). The half-cell y offset of v is ignored — at
    synthetic-noise fidelity it is meaningless.

    The k halo rows are filled by edge copy with zero fluctuation (the mean is
    extrapolated flat below ground); the j halo columns are the periodic wrap,
    matching ``BCym=1``.
    """
    n_records = times.size
    dz = zsize / ktot
    dy = ylen / jtot
    zt = (np.arange(ktot) + 0.5) * dz  # cell centres: u, v
    zm = np.arange(ktot + 1) * dz  # faces: w (kb .. ke+1)

    shape_t = build_profile_shape(profile_config, zt, zsize)
    shape_m = build_profile_shape(profile_config, zm, zsize)

    u_ref, v_ref = angle_to_velocity(inflow_angle, velocity_magnitude)
    u_mean = np.asarray(u_ref)[:, None] * shape_t[None, :]  # (n, ktot)
    v_mean = np.asarray(v_ref)[:, None] * shape_t[None, :]

    # Target rms. |U_mean(z)| = velocity_magnitude * s(z) because the angle
    # decomposition preserves the magnitude (inflow_utils.angle_to_velocity).
    speed = np.asarray(velocity_magnitude, dtype=float)[:, None]
    sigma_u = intensity * speed * shape_t[None, :]
    sigma_v = CROSS_COMPONENT_ANISOTROPY * sigma_u
    sigma_w = CROSS_COMPONENT_ANISOTROPY * intensity * speed * shape_m[None, :]
    # The ground face carries no vertical velocity, whatever the profile shape.
    sigma_w[:, 0] = 0.0

    # Taylor's hypothesis turns the streamwise length scale into a time scale.
    u_ref_scalar = float(np.mean(np.abs(speed)))
    if u_ref_scalar <= 0.0:
        ar1 = 0.0
    else:
        t_scale = length_scale_x / u_ref_scalar
        dt = float(times[1] - times[0]) if n_records > 1 else length_scale_x
        ar1 = math.exp(-math.pi * dt / (2.0 * t_scale))

    white = generate_fluctuation_fields(
        seed=seed,
        ar1_coefficient=ar1,
        n_records=n_records,
        start_index=start_index,
        shapes={"u": (ktot, jtot), "v": (ktot, jtot), "w": (ktot + 1, jtot)},
    )

    b_y = filter_coefficients(length_scale_y / dy)
    b_z = filter_coefficients(min(length_scale_z / dz, float(ktot + 1)))

    fluct = {
        name: _filter_periodic(_filter_bounded(field, b_z), b_y)
        for name, field in white.items()
    }

    u_fluct = fluct["u"] * sigma_u[:, :, None]
    # Zero the plane mean of u' so the instantaneous bulk inflow equals the
    # mean-profile bulk; otherwise the outflow convective velocity (recomputed
    # from u0av every step, modboundary.f90:143-162) fights the inlet.
    #
    # This costs some variance, because the removed bulk mode is itself part of
    # the filtered field: ~2% at length scales small against the inlet plane,
    # but 40-50% once `length_scale_y/z` approach the plane's own dimensions
    # (there the field is nearly uniform, so the plane mean IS the fluctuation).
    # Worth remembering when calibrating `intensity` against long length scales.
    u_fluct -= u_fluct.mean(axis=(1, 2), keepdims=True)
    v_fluct = fluct["v"] * sigma_v[:, :, None]
    w_fluct = fluct["w"] * sigma_w[:, :, None]

    u_plane = _assemble_centred_plane(u_mean, u_fluct, jtot, ktot)
    v_plane = _assemble_centred_plane(v_mean, v_fluct, jtot, ktot)
    w_plane = _assemble_face_plane(w_fluct, jtot, ktot)
    return u_plane, v_plane, w_plane


def _wrap_j_halo(plane: np.ndarray, jtot: int) -> np.ndarray:
    """Fill the j ghost columns with the periodic image of the interior."""
    plane[:, :, 0] = plane[:, :, jtot]
    plane[:, :, jtot + 1] = plane[:, :, 1]
    return plane


def _assemble_centred_plane(
    mean: np.ndarray, fluct: np.ndarray, jtot: int, ktot: int
) -> np.ndarray:
    """Build a ``(n, ktot+2, jtot+2)`` plane for a cell-centred component."""
    plane = np.zeros((mean.shape[0], ktot + 2, jtot + 2))
    plane[:, 1 : ktot + 1, 1 : jtot + 1] = mean[:, :, None] + fluct
    # Halo rows: `xmi_driver` never reads them for u/v, but leaving them at zero
    # would make a debugging plot of the raw file misleading. Flat mean, no
    # fluctuation.
    plane[:, 0, 1 : jtot + 1] = mean[:, 0][:, None]
    plane[:, ktot + 1, 1 : jtot + 1] = mean[:, -1][:, None]
    return _wrap_j_halo(plane, jtot)


def _assemble_face_plane(fluct: np.ndarray, jtot: int, ktot: int) -> np.ndarray:
    """Build a ``(n, ktot+2, jtot+2)`` plane for w (zero mean, face levels).

    ``fluct`` rows 0..ktot are the faces ``k = kb .. ke+1``, i.e. array rows
    1..ktot+1. Row 0 (``kb-1``) is below ground and never read.
    """
    plane = np.zeros((fluct.shape[0], ktot + 2, jtot + 2))
    plane[:, 1 : ktot + 2, 1 : jtot + 1] = fluct
    return _wrap_j_halo(plane, jtot)


# --- Inflow schedule ---------------------------------------------------------


def resolve_inflow_schedule(
    params: xarray.Dataset,
    spinup_time: float,
    simulation_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(time_seconds, inflow_angle, velocity_magnitude)`` knots.

    Mirrors :func:`.nudging_utils.apply_time_varying_inflow` exactly, so the
    driver path and the nudging path see the same inflow signal: scalar params
    become a constant 2-knot schedule spanning the window, and a ``spinup_time``
    plateau at the initial values is prepended when there is one.
    """
    if "time" not in params.dims:
        time_seconds = np.linspace(0.0, max(simulation_time, 0.0), 2)
        inflow_angle = np.full(2, float(params["inflow_angle"].values))
        velocity_magnitude = np.full(2, float(params["velocity_magnitude"].values))
    else:
        time_seconds = params["time"].values.astype(float)
        inflow_angle = np.asarray(params["inflow_angle"].values, dtype=float)
        velocity_magnitude = np.asarray(
            params["velocity_magnitude"].values, dtype=float
        )
        if inflow_angle.ndim == 0:
            inflow_angle = np.full_like(time_seconds, float(inflow_angle))
        if velocity_magnitude.ndim == 0:
            velocity_magnitude = np.full_like(time_seconds, float(velocity_magnitude))

    if spinup_time > 0:
        time_seconds = np.concatenate([[0.0], time_seconds + spinup_time])
        inflow_angle = np.concatenate([[inflow_angle[0]], inflow_angle])
        velocity_magnitude = np.concatenate(
            [[velocity_magnitude[0]], velocity_magnitude]
        )

    return time_seconds, inflow_angle, velocity_magnitude


def _sample_schedule(
    times: np.ndarray,
    knot_times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a schedule onto the driver record times.

    ``np.interp`` clamps outside the knot range, which is what the coverage
    margin past the end of the run needs.
    """
    if knot_times.size == 1:
        return np.full(times.size, float(values[0]))
    return np.interp(times, knot_times, values)


# --- Orchestration -----------------------------------------------------------


def apply_inlet_turbulence(
    params: xarray.Dataset,
    dirs: DirectoryPaths,
    config: Optional[dict],
    profile_config: Optional[dict] = None,
    spinup_time: float = 0.0,
    simulation_time: float = 0.0,
    window_start_time: float = 0.0,
) -> None:
    """Generate driver planes and point namoptions at them.

    Regenerates the whole record sequence for this window and overwrites the
    driver files in place, so stale planes from a previous window — or from the
    donor a failed member was substituted from — can never leak into a run.

    Args:
        params: Inflow parameters (``inflow_angle``, ``velocity_magnitude``),
            scalar or with a ``time`` dimension.
        dirs: Experiment directory paths.
        config: The ``inlet_turbulence`` block. Must be enabled — callers check
            with :func:`is_inlet_turbulence_enabled`.
        profile_config: Resolved vertical-profile config (already carrying the
            per-member ``vertical_inflow_exponent`` override, if estimated).
        spinup_time: Spinup seconds prepended to this call's run (0 on a warm
            window).
        simulation_time: Window length in seconds.
        window_start_time: Physical seconds already simulated by this member.
            Selects which slice of the member's synthetic turbulence history is
            written, giving continuity across windows even though the solver
            clock restarts at 0 (see module docstring).
    """
    resolved = resolve_inlet_turbulence_config(config)
    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"
    namoptions = NamoptionsFile(namoptions_path)

    jtot = namoptions.get_value_as_int("DOMAIN", "jtot")
    ktot = namoptions.get_value_as_int("DOMAIN", "ktot")
    ylen = namoptions.get_value_as_float("DOMAIN", "ylen")
    zsize = namoptions.get_value_as_float("INPS", "zsize")
    if jtot is None or ktot is None or ylen is None or zsize is None:
        raise ValueError(
            "Cannot read jtot/ktot/ylen (&DOMAIN) or zsize (&INPS) from "
            f"{namoptions_path}; all four are required to size the driver planes."
        )

    time_step = float(resolved["time_step"])
    runtime_total = float(spinup_time) + float(simulation_time)
    times = driver_time_grid(runtime_total, time_step)

    knot_times, knot_angle, knot_speed = resolve_inflow_schedule(
        params, spinup_time, simulation_time
    )
    inflow_angle = _sample_schedule(times, knot_times, knot_angle)
    velocity_magnitude = _sample_schedule(times, knot_times, knot_speed)

    seed = resolved["seed"]
    seed = derive_seed(dirs.experiment_name) if seed is None else int(seed)
    start_index = int(round(max(0.0, window_start_time) / time_step))

    u_plane, v_plane, w_plane = build_driver_planes(
        jtot=jtot,
        ktot=ktot,
        zsize=zsize,
        ylen=ylen,
        times=times,
        start_index=start_index,
        inflow_angle=inflow_angle,
        velocity_magnitude=velocity_magnitude,
        profile_config=profile_config,
        intensity=float(resolved["intensity"]),
        length_scale_y=float(resolved["length_scale_y"]),
        length_scale_z=float(resolved["length_scale_z"]),
        length_scale_x=float(resolved["length_scale_x"]),
        seed=seed,
    )

    driverjobnr = int(resolved["driverjobnr"])
    write_driver_files(
        dirs.experiment_dir, driverjobnr, times, u_plane, v_plane, w_plane
    )

    _write_driver_namoptions(
        namoptions_path,
        driverjobnr=driverjobnr,
        driverstore=times.size,
        dtdriver=time_step,
        lchunkread=resolved["lchunkread"],
        chunkread_size=resolved["chunkread_size"],
    )
    _write_mean_inflow_settings(
        dirs,
        inflow_angle=float(inflow_angle[0]),
        velocity_magnitude=float(velocity_magnitude[0]),
        ktot=ktot,
    )

    logger.info(
        "uDALES inlet turbulence ON: %d driver records at dt=%.3f s covering "
        "[0, %.1f] s (window starts at t=%.1f s -> global record %d), "
        "plane %dx%d, intensity=%.3f, L=(%.1f, %.1f, %.1f) m, seed=%d",
        times.size,
        time_step,
        times[-1],
        window_start_time,
        start_index,
        ktot + 2,
        jtot + 2,
        float(resolved["intensity"]),
        float(resolved["length_scale_x"]),
        float(resolved["length_scale_y"]),
        float(resolved["length_scale_z"]),
        seed,
    )


def _write_driver_namoptions(
    namoptions_path: pathlib.Path,
    *,
    driverjobnr: int,
    driverstore: int,
    dtdriver: float,
    lchunkread: Optional[bool],
    chunkread_size: Optional[int],
) -> None:
    """Switch the inlet to the driver route and disable nudging.

    ``BCxm=3`` is what actually turns the driver on — it forces ``idriver=2``
    and ``linoutflow`` in ``modstartup.f90:831-835``. ``idriver`` is written too
    purely so the mode is greppable in the namoptions of a finished run; it is a
    declared ``&DRIVER`` key, so writing it is safe (an *undeclared* key aborts
    the namelist read).

    ``NamoptionsFile.set_value`` creates a missing section, so case templates
    without a ``&DRIVER`` block need no changes.
    """
    namoptions = NamoptionsFile(namoptions_path)

    namoptions.set_value("BC", "BCxm", 3)

    namoptions.set_value("DRIVER", "idriver", 2)
    namoptions.set_value("DRIVER", "driverjobnr", driverjobnr)
    namoptions.set_value("DRIVER", "driverstore", driverstore)
    namoptions.set_value("DRIVER", "dtdriver", f"{dtdriver:.6f}")
    namoptions.set_value("DRIVER", "tdriverstart", "0.0")
    if lchunkread is not None:
        namoptions.set_value(
            "DRIVER", "lchunkread", ".true." if lchunkread else ".false."
        )
    if chunkread_size is not None:
        namoptions.set_value("DRIVER", "chunkread_size", int(chunkread_size))

    # Under BCxm=3 the inlet no longer comes from uprof, so time-dependent
    # nudging has nothing left to drive; interior nudging would actively damp the
    # injected fluctuations on the tnudge timescale. Both go off.
    namoptions.set_value("PHYSICS", "lnudge", ".false.")
    namoptions.set_value("PHYSICS", "ltimedepnudge", ".false.")

    namoptions.write()


def _write_mean_inflow_settings(
    dirs: DirectoryPaths,
    *,
    inflow_angle: float,
    velocity_magnitude: float,
    ktot: int,
) -> None:
    """Write the reference scalars and the initial condition.

    Mirrors the inflow-outflow handling of the nudging path: ``dpdx``/``dpdy``
    are zeroed so the inlet face is the sole streamwise driver, and
    ``prof.inp``/``lscale.inp`` start the flow from rest — writing a full-speed
    profile stagnates it against the building walls before the pressure solver
    has settled. ``u0``/``v0`` are uDALES reference scalars, independent of the
    initial condition.
    """
    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"
    u0, v0 = angle_to_velocity(inflow_angle, velocity_magnitude)
    namoptions = NamoptionsFile(namoptions_path)
    namoptions.set_value("INPS", "u0", f"{float(u0):.7f}")
    namoptions.set_value("INPS", "v0", f"{float(v0):.7f}")
    namoptions.set_value("INPS", "dpdx", "0.0")
    namoptions.set_value("INPS", "dpdy", "0.0")
    namoptions.write()

    zeros = np.zeros(ktot)
    update_prof_file_profile(
        dirs.experiment_dir / f"prof.inp.{dirs.experiment_name}",
        u_profile=zeros,
        v_profile=zeros,
    )
    update_lscale_file_profile(
        dirs.experiment_dir / f"lscale.inp.{dirs.experiment_name}",
        u_profile=zeros,
        v_profile=zeros,
        dpdx_profile=zeros,
        dpdy_profile=zeros,
    )
