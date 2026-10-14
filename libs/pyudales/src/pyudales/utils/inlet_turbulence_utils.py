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
``docs/plans/implemented/udales_inlet_turbulence.md``.

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
only the current window's records are written, so window *n*'s planes continue
window *n-1*'s without any filter state having to be serialised. The *clock*
itself is persisted (see
:data:`pyurbanair.utils.digital_filter.ELAPSED_TIME_FILENAME`) because parallel
ensembles run each member in a worker process that discards attribute
mutations, and the replay is bounded by :func:`ar1_burn_in_records` so a long
rollout stays linear rather than quadratic.
"""

import logging
import math
from typing import Any, Optional

import numpy as np
import xarray

from pyurbanair.utils.digital_filter import (
    CROSS_COMPONENT_ANISOTROPY,
    derive_seed,
    filtered_fluctuations,
    validate_time_step_divides_window,
)

from .dir_utils import DirectoryPaths
from .driver_file_utils import write_driver_files
from .file_update_utils import update_lscale_file_profile, update_prof_file_profile
from .inflow_utils import angle_to_velocity
from .namoptions_utils import NamoptionsFile
from .nudging_utils import build_inflow_schedule
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


# Extra records written past the end of the run. `drivergen` HARD-STOPS the
# simulation when `timee` exceeds the last record time (moddriver.f90:236-241),
# so the coverage margin is not cosmetic.
DRIVER_TIME_MARGIN_RECORDS = 2


def is_inlet_turbulence_enabled(config: Optional[dict]) -> bool:
    """True only when an ``inlet_turbulence`` dict explicitly enables the knob.

    ``None``/``{}``/``enabled: false`` are all a strict no-op — nothing is
    written to namoptions and no driver files appear (AGENTS.md no-op rule).
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


# --- Plane assembly ----------------------------------------------------------


def driver_time_grid(
    runtime_total: float, time_step: float, dtmax: Optional[float] = None
) -> np.ndarray:
    """Record times covering ``[0, runtime_total]`` plus a safety margin.

    ``drivergen`` stops the whole simulation the moment ``timee`` exceeds the
    last record (``moddriver.f90:236-241``), and uDALES' final step can overshoot
    ``runtime`` by up to ``dtmax``, so the grid overshoots by whichever is larger:
    :data:`DRIVER_TIME_MARGIN_RECORDS`, or enough records to cover ``dtmax``.
    """
    if time_step <= 0.0:
        raise ValueError(f"time_step must be > 0, got {time_step}")
    margin = DRIVER_TIME_MARGIN_RECORDS
    if dtmax is not None and dtmax > 0.0:
        margin = max(margin, int(math.ceil(dtmax / time_step)) + 1)
    n_records = int(math.ceil(runtime_total / time_step - 1e-9)) + 1 + margin
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
    # The TOP face (k = ke+1) is deliberately NOT zeroed: BCxm=3 forces
    # BCtopm=3 (`BCtopm_pressure`, modstartup.f90:866-869), which exists
    # precisely to "allow vertical velocity at top", so a no-through-flow
    # condition there would be wrong rather than merely conservative.
    sigma_w[:, 0] = 0.0

    fluct = filtered_fluctuations(
        seed=seed,
        start_index=start_index,
        times=times,
        speed=speed,
        n_z=ktot,
        n_y=jtot,
        dz=dz,
        dy=dy,
        length_scale_x=length_scale_x,
        length_scale_y=length_scale_y,
        length_scale_z=length_scale_z,
    )

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


# Above this length-scale-to-plane-dimension ratio the filtered field is nearly
# uniform across the inlet, so the plane-mean removal below eats most of the
# fluctuation energy and the realised rms falls far short of `intensity`.
LENGTH_SCALE_PLANE_FRACTION_WARN = 0.3


# Warnings below fire from a per-member, per-window code path: a 32-member
# 20-window rollout would otherwise emit 640 identical lines. Deduplicated per
# process, which is per worker under a parallel ensemble -- still a handful
# rather than hundreds, and the message is about config, not about a run.
_WARNED_LENGTH_SCALES: set[tuple[str, float, float]] = set()


def _warn_if_length_scales_exceed_the_plane(
    *,
    length_scale_y: float,
    length_scale_z: float,
    ylen: float,
    zsize: float,
) -> None:
    """Warn when the requested eddies barely fit on the inlet plane.

    Zeroing the plane mean of ``u'`` is required (it keeps the instantaneous
    bulk inflow equal to the mean-profile bulk), but the removed bulk mode is
    itself part of the filtered field. That costs ~2% of the target rms at
    length scales small against the plane and 40-50% once they approach its
    dimensions — so calibration can otherwise spend a long time raising
    `intensity` to chase amplitude the code has already discarded.
    """
    for axis, scale, extent in (
        ("length_scale_y", length_scale_y, ylen),
        ("length_scale_z", length_scale_z, zsize),
    ):
        if extent > 0 and scale / extent > LENGTH_SCALE_PLANE_FRACTION_WARN:
            key = (axis, scale, extent)
            if key in _WARNED_LENGTH_SCALES:
                continue
            _WARNED_LENGTH_SCALES.add(key)
            logger.warning(
                "inlet_turbulence.%s=%.1f m is %.0f%% of the inlet plane's %.1f m "
                "extent. Zeroing the plane mean of u' will discard a large part "
                "of the fluctuation energy (up to ~50%%), so the realised rms "
                "will fall well short of intensity*|U(z)|. Reduce the length "
                "scale, or calibrate against the realised rms rather than the "
                "nominal one.",
                axis,
                scale,
                100.0 * scale / extent,
                extent,
            )


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

    Delegates to :func:`.nudging_utils.build_inflow_schedule`, the single
    implementation shared with the nudging path, so the driver planes and
    ``timedepnudge.inp`` can never encode different inflow signals.
    """
    return build_inflow_schedule(
        params,
        spinup_time=spinup_time,
        simulation_time=simulation_time,
    )


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
    # One read / modify / write for the whole call: the driver keys, the nudging
    # switches and the &INPS reference scalars all land in this one object.
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
    validate_time_step_divides_window(runtime_total, time_step)
    times = driver_time_grid(
        runtime_total, time_step, dtmax=namoptions.get_value_as_float("RUN", "dtmax")
    )
    _warn_if_length_scales_exceed_the_plane(
        length_scale_y=float(resolved["length_scale_y"]),
        length_scale_z=float(resolved["length_scale_z"]),
        ylen=ylen,
        zsize=zsize,
    )
    _warn_if_driver_arrays_are_large(
        jtot=jtot,
        ktot=ktot,
        driverstore=times.size,
        lchunkread=resolved["lchunkread"],
    )

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

    _apply_driver_namoptions(
        namoptions,
        driverjobnr=driverjobnr,
        driverstore=times.size,
        dtdriver=time_step,
        lchunkread=resolved["lchunkread"],
        chunkread_size=resolved["chunkread_size"],
    )
    _apply_mean_inflow_settings(
        namoptions,
        inflow_angle=float(inflow_angle[0]),
        velocity_magnitude=float(velocity_magnitude[0]),
    )
    namoptions.write()
    _write_initial_condition_files(dirs, ktot=ktot)

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


def _apply_driver_namoptions(
    namoptions: NamoptionsFile,
    *,
    driverjobnr: int,
    driverstore: int,
    dtdriver: float,
    lchunkread: Optional[bool],
    chunkread_size: Optional[int],
) -> None:
    """Switch the inlet to the driver route and disable nudging.

    Mutates ``namoptions`` in place; the caller writes once.

    ``BCxm=3`` is what actually turns the driver on — it forces ``idriver=2``
    and ``linoutflow`` in ``modstartup.f90:831-835``. ``idriver`` is written too
    purely so the mode is greppable in the namoptions of a finished run; it is a
    declared ``&DRIVER`` key, so writing it is safe (an *undeclared* key aborts
    the namelist read).

    ``NamoptionsFile.set_value`` creates a missing section, so case templates
    without a ``&DRIVER`` block need no changes.
    """
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


def _apply_mean_inflow_settings(
    namoptions: NamoptionsFile,
    *,
    inflow_angle: float,
    velocity_magnitude: float,
) -> None:
    """Set the &INPS reference scalars; mutates ``namoptions`` in place.

    ``dpdx``/``dpdy`` are zeroed so the inlet face is the sole streamwise
    driver, matching the inflow-outflow handling of the nudging path.
    ``u0``/``v0`` are uDALES reference scalars, independent of the initial
    condition written by :func:`_write_initial_condition_files`.
    """
    u0, v0 = angle_to_velocity(inflow_angle, velocity_magnitude)
    namoptions.set_value("INPS", "u0", f"{float(u0):.7f}")
    namoptions.set_value("INPS", "v0", f"{float(v0):.7f}")
    namoptions.set_value("INPS", "dpdx", "0.0")
    namoptions.set_value("INPS", "dpdy", "0.0")


def _write_initial_condition_files(dirs: DirectoryPaths, *, ktot: int) -> None:
    """Start the interior from rest, as the inflow-outflow nudging path does.

    Writing a full-speed profile here stagnates the flow against the building
    walls before the pressure solver has settled. Note the consequence, which
    differs from the nudging path: under the driver route there is no interior
    relaxation either (`lnudge=.false.`), so the domain fills from the inlet
    face alone and the effective spin-up is LONGER than nudging-path experience
    would suggest. Size `spinup_time` accordingly (docs/pyudales.md 6.1).
    """
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


# initdriver allocates ~6 full-history plane arrays when lchunkread is off
# (moddriver.f90:89-130), resident for the whole run on the inlet rank.
DRIVER_ARRAY_COUNT = 6
DRIVER_MEMORY_WARN_BYTES = 512 * 1024 * 1024


def _warn_if_driver_arrays_are_large(
    *,
    jtot: int,
    ktot: int,
    driverstore: int,
    lchunkread: Optional[bool],
) -> None:
    """Warn before uDALES silently allocates hundreds of MB per member."""
    if lchunkread:
        return
    plane_values = (jtot + 2) * (ktot + 2)
    total = DRIVER_ARRAY_COUNT * plane_values * driverstore * 8
    if total < DRIVER_MEMORY_WARN_BYTES:
        return
    logger.warning(
        "Driver planes will make uDALES allocate ~%.0f MB per member "
        "(%d arrays x %d records x %dx%d x 8 B, moddriver.f90:89-130), resident "
        "for the whole run and multiplied by ensemble.num_parallel_processes. "
        "Set inlet_turbulence.lchunkread=true (with chunkread_size) to bound it, "
        "or increase inlet_turbulence.time_step.",
        total / 1e6,
        DRIVER_ARRAY_COUNT,
        driverstore,
        ktot + 2,
        jtot + 2,
    )
