"""Inflow (inlet) turbulence for the LBM solver.

``inlet_turbulence.method`` picks one of two mechanisms. Both go through the
Fortran's inflow-turbulence subsystem (``m_inflow_turbulence_*.F90``), driven by
three values that ``m_readinfile.F90`` reads with a single list-directed
statement::

    read(10,*,err=100) inflowturbulence, turbulence_ampl, nrturb

which appears in ``infile.in`` as one positional line::

     F 0.00005  100   ! lturb amp nrtu  : Add turbulence forcing on inflow, ...

Because the whole line is consumed by one ``read``, the three values must be
written together — see ``Infile.set_value_tokens``.

``random_forcing`` (default) — the LBM's own perturbations
    Every ``nrturb`` steps ``inflow_turbulence_compute`` draws ``nrturb + 1``
    fresh pseudo-random planes per component (``m_pseudo2D``: Gaussian
    covariance, decorrelation length ``10/sqrt(3)`` cells, periodic in y *and*
    z), chains them with an AR(1) of 0.98 per step and normalises each to unit
    variance. ``inflow_turbulence_forcing`` then adds ``amplitude`` times the
    plane of the current step to the velocity of the slab ``i = iturb_pos = 10``
    as a Kupershtokh exact-difference forcing (``f += feq(rho, u + du) -
    feq(rho, u)``, ``main.F90:320,327``). Written once at ``compile()``.

``digital_filter`` — the shared Xie & Castro (2008) inflow
    The fluctuations uDALES' driver planes and PALM's inflow planes carry, from
    :mod:`pyurbanair.utils.digital_filter` with the same keys and statistics,
    imposed through the *same* forcing slab. The LBM has no time-varying inflow
    plane: the Krüger inflow (``m_boundary_i_inflow_kernel.F90``) takes one
    ``uvel(nz)`` profile and one direction, uniform in y and without w. The
    only per-(y, z, step) input it reads is the forcing buffer
    ``uu/vv/ww(ny, nz, 0:nrturb)``, which ``m_readrestart.F90`` loads from
    ``restart/turbulence_<tile>_<nt0>.uf`` on every run with ``nt0 > 0``. So
    each window is run as a restart run whose turbulence file pylbm writes:

    - ``nrturb`` is the window length ``L`` in steps and ``nt0`` a multiple of
      it (cold starts write an equilibrium restart at ``nt0 = L``, the state
      ``inipert`` would give). The forcing reads ``uu(:, :, mod(it, L))`` with
      0 mapped to ``L``, so step ``nt0 + l`` reads slot ``l`` and the buffer is
      used once, front to back; the refresh at ``mod(it, nrturb) == 0`` falls on
      ``nt1`` only, after the last forcing.
    - The forcing adds ``du`` to the slab's velocity every step. Fluid crosses
      the one-cell slab in ``1/U_x`` steps (lattice units), so in the mean flow
      a slab increment ``du`` leaves a jump ``du/U_x`` downstream. Writing
      ``du = U_x(z) * u'/C_u`` (and the same for v, w) with ``amplitude = 1``
      therefore puts the digital-filter fluctuation ``u'`` (m/s) on the flow
      just downstream of the slab. ``U_x`` is the inflow's own x-velocity
      profile at that step.
    - The exact-difference forcing leaves rho unchanged, so it conserves mass
      exactly; with the plane mean of ``u'`` removed per record (as on the
      other backends) the downstream mass flux carries no mean perturbation.
    - Planes come every ``time_step`` s and are interpolated linearly onto the
      steps, as uDALES and PALM interpolate theirs. Step ``l`` sits at window
      time ``(l - 1) * dt``, the clock ``m_inflow.F90`` uses for the mean.
    - The planes live on the LBM's cell centres, which are the case grid's: u,
      v as on the other backends, w generated on the faces and averaged to the
      centres.

    Differences from uDALES/PALM: the fluctuations enter 10 cells downstream of
    the inlet face rather than on it, and the response of the flow to the
    forcing is the advective jump only in the mean.
"""

import logging
import math
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from scipy.io import FortranFile

from pyurbanair.utils.digital_filter import (
    CROSS_COMPONENT_ANISOTROPY,
    filtered_fluctuations,
)

if TYPE_CHECKING:
    from .dir_utils import DirectoryPaths

from .infile_utils import Infile
from .vertical_profile import build_profile_shape
from .warm_start_utils import restart_file_name

logger = logging.getLogger(__name__)

# Key of the three-value inflow-turbulence line in infile.in (the first word of
# its "! lturb amp nrtu : ..." comment, which is how Infile indexes lines).
INLET_TURBULENCE_KEY = "lturb"

RANDOM_FORCING = "random_forcing"
DIGITAL_FILTER = "digital_filter"
INLET_TURBULENCE_METHODS = (RANDOM_FORCING, DIGITAL_FILTER)

# digital_filter: same keys and defaults as pyudales' and pypalm's blocks, so
# one set of values gives all three backends the same inflow statistics.
DIGITAL_FILTER_DEFAULTS: dict[str, Any] = {
    "intensity": 0.15,  # u'_rms / |U_mean(z)|; v', w' get 0.7x
    "length_scale_x": 24.0,  # m; AR(1) time scale via Taylor
    "length_scale_y": 12.0,  # m
    "length_scale_z": 12.0,  # m
    "time_step": 0.5,  # s between planes
    "seed": None,  # None -> derived from the member's experiment name
}

# Accepted keys of the ``inlet_turbulence`` config block.
INLET_TURBULENCE_FIELDS = (
    "enabled",
    "method",
    "amplitude",
    "update_interval",
    *DIGITAL_FILTER_DEFAULTS,
)

# Planes generated past the end of the window, so the interpolation onto the
# last steps never extrapolates.
PLANE_MARGIN_RECORDS = 2

# uu/vv/ww/rr(ny, nz, 0:nrturb) live on the device for the whole run, and
# m_readrestart / m_saverestart hold a host copy of all four while copying.
TURBULENCE_BUFFER_WARN_BYTES = 512 * 1024 * 1024


def validate_inlet_turbulence(
    inlet_turbulence: Optional[dict],
    boundary_condition: str,
) -> Optional[dict[str, Any]]:
    """Validate the ``inlet_turbulence`` config block at construction time.

    Args:
        inlet_turbulence: ``{"enabled": bool, "method": str, ...}``; ``None``
            disables the feature entirely.
        boundary_condition: The model's x-boundary condition.

    Returns:
        A normalized copy of the block, or ``None`` when it is absent.

    Raises:
        ValueError: On unknown keys, an unknown ``method``, a non-positive
            ``update_interval``, a negative ``amplitude``, bad digital-filter
            values, or turbulence requested with a periodic boundary condition.
    """
    if inlet_turbulence is None:
        return None

    cfg = dict(inlet_turbulence)
    unknown = sorted(set(cfg) - set(INLET_TURBULENCE_FIELDS))
    if unknown:
        raise ValueError(
            f"Unknown inlet_turbulence key(s) {unknown}; "
            f"expected any of {list(INLET_TURBULENCE_FIELDS)}."
        )

    enabled = bool(cfg.get("enabled", False))
    cfg["enabled"] = enabled

    method = cfg.get("method") or RANDOM_FORCING
    if method not in INLET_TURBULENCE_METHODS:
        raise ValueError(
            f"inlet_turbulence.method must be one of {list(INLET_TURBULENCE_METHODS)}, "
            f"got {method!r}."
        )

    if enabled and boundary_condition != "inflow_outflow":
        raise ValueError(
            "inlet_turbulence.enabled=true requires "
            "boundary_condition='inflow_outflow', got "
            f"'{boundary_condition}'. The LBM only refreshes the inflow "
            "turbulence when ibnd==1 (main.F90), so it is a no-op with "
            "periodic boundaries."
        )

    if cfg.get("amplitude") is not None:
        amplitude = float(cfg["amplitude"])
        if amplitude < 0:
            raise ValueError(
                f"inlet_turbulence.amplitude must be >= 0, got {amplitude}."
            )
        cfg["amplitude"] = amplitude

    if cfg.get("update_interval") is not None:
        update_interval = int(cfg["update_interval"])
        if update_interval < 1:
            raise ValueError(
                "inlet_turbulence.update_interval must be >= 1 (the Fortran "
                "aborts on nrturb <= 0 and uses it as a modulus), got "
                f"{update_interval}."
            )
        cfg["update_interval"] = update_interval

    if enabled and method == DIGITAL_FILTER:
        resolved = digital_filter_settings(cfg)
        if resolved["intensity"] < 0:
            raise ValueError(
                f"inlet_turbulence.intensity must be >= 0, got {resolved['intensity']}."
            )
        for key in ("length_scale_x", "length_scale_y", "length_scale_z", "time_step"):
            if resolved[key] <= 0:
                raise ValueError(
                    f"inlet_turbulence.{key} must be > 0, got {resolved[key]}."
                )

    return cfg


def uses_digital_filter(inlet_turbulence: Optional[dict]) -> bool:
    """True when the block is enabled with ``method: digital_filter``."""
    return bool(
        inlet_turbulence
        and inlet_turbulence.get("enabled", False)
        and inlet_turbulence.get("method") == DIGITAL_FILTER
    )


def digital_filter_settings(inlet_turbulence: dict) -> dict[str, Any]:
    """The digital-filter keys of the block, defaults filled in, as floats."""
    resolved = dict(DIGITAL_FILTER_DEFAULTS)
    resolved.update(
        {
            k: inlet_turbulence[k]
            for k in DIGITAL_FILTER_DEFAULTS
            if inlet_turbulence.get(k) is not None
        }
    )
    for key in ("intensity", "length_scale_x", "length_scale_y", "length_scale_z"):
        resolved[key] = float(resolved[key])
    resolved["time_step"] = float(resolved["time_step"])
    return resolved


def apply_inlet_turbulence(
    inlet_turbulence: Optional[dict],
    dirs: "DirectoryPaths",
) -> None:
    """Write the ``lturb`` line of ``infile.in`` for ``random_forcing``.

    Strict no-op when ``inlet_turbulence`` is ``None`` or disabled, so default
    runs keep the solver-generated line byte-identical. Values not supplied fall
    back to the ones already in the file rather than to invented defaults.
    ``digital_filter`` writes its line per window instead
    (:func:`write_digital_filter_forcing`).

    Args:
        inlet_turbulence: Validated config block (see
            :func:`validate_inlet_turbulence`).
        dirs: Directory paths holding ``infile_path``.

    Raises:
        ValueError: If the ``lturb`` line is missing or malformed, which would
            mean the infile does not match the expected solver template.
    """
    if inlet_turbulence is None or not inlet_turbulence.get("enabled", False):
        return
    if uses_digital_filter(inlet_turbulence):
        return

    tokens = _lturb_tokens(dirs)
    amplitude = inlet_turbulence.get("amplitude")
    update_interval = inlet_turbulence.get("update_interval")
    amplitude_token = tokens[1] if amplitude is None else f"{float(amplitude):.10g}"
    interval_token = tokens[2] if update_interval is None else str(int(update_interval))

    set_lturb(dirs, True, amplitude_token, interval_token)
    logger.info(
        "Enabled LBM inflow turbulence (amplitude=%s, update_interval=%s)",
        amplitude_token,
        interval_token,
    )


def _lturb_tokens(dirs: "DirectoryPaths") -> list[str]:
    tokens = Infile(dirs.infile_path).get_value_tokens(INLET_TURBULENCE_KEY)
    if tokens is None or len(tokens) != 3:
        raise ValueError(
            f"Expected a 3-value '{INLET_TURBULENCE_KEY}' line "
            f"(inflowturbulence, turbulence_ampl, nrturb) in {dirs.infile_path}, "
            f"found {tokens!r}. Refusing to edit a positional infile whose "
            "layout does not match the solver template."
        )
    return tokens


def set_lturb(
    dirs: "DirectoryPaths", enabled: bool, amplitude: Any, nrturb: Any
) -> None:
    """Rewrite the whole three-value ``lturb`` line."""
    _lturb_tokens(dirs)
    infile = Infile(dirs.infile_path)
    infile.set_value_tokens(INLET_TURBULENCE_KEY, [enabled, amplitude, nrturb])
    infile.write()


def disable_lturb(dirs: "DirectoryPaths") -> None:
    """Switch the forcing off, keeping the line's other two values."""
    tokens = _lturb_tokens(dirs)
    set_lturb(dirs, False, tokens[1], tokens[2])


# --- digital_filter ----------------------------------------------------------


def aligned_start_iteration(
    previous_iteration: Optional[int], window_steps: int
) -> int:
    """The first multiple of ``window_steps`` at or after ``previous_iteration``.

    With ``nrturb = window_steps`` and ``nt0`` on that grid, step ``nt0 + l``
    reads buffer slot ``l`` and the solver's own refresh (at ``mod(it, nrturb)
    == 0``) lands on ``nt1``, after the last forcing. Never 0: ``nt0 = 0`` is
    the solver's cold start, which draws its own random buffer instead of
    reading ours.
    """
    if window_steps < 1:
        raise ValueError(f"window_steps must be >= 1, got {window_steps}")
    previous = max(1, int(previous_iteration or 0))
    return int(math.ceil(previous / window_steps)) * window_steps


def build_digital_filter_fluctuations(
    *,
    times: np.ndarray,
    start_index: int,
    velocity_magnitude: np.ndarray,
    ny: int,
    nz: int,
    dy: float,
    dz: float,
    zsize: float,
    profile_config: Optional[dict],
    intensity: float,
    length_scale_x: float,
    length_scale_y: float,
    length_scale_z: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Digital-filter fluctuations in m/s, as uDALES' and PALM's planes carry.

    Pure function. Statistics are those of pyudales' ``build_driver_planes``:
    ``sigma_u = intensity * |U_mean(z)|``, v' and w' at
    :data:`CROSS_COMPONENT_ANISOTROPY` of that, no w' at the ground face, and
    the plane mean of u' removed per record. For the same seed, grid and
    records the values equal the fluctuation part of uDALES' planes.

    Returns:
        ``{"u": (n, nz, ny), "v": (n, nz, ny), "w": (n, nz + 1, ny)}``; u and v
        on the cell centres, w on the faces from the ground to the top.
    """
    zt = (np.arange(nz) + 0.5) * dz
    zm = np.arange(nz + 1) * dz
    shape_t = build_profile_shape(profile_config, zt, zsize)
    shape_m = build_profile_shape(profile_config, zm, zsize)

    speed = np.asarray(velocity_magnitude, dtype=float)[:, None]
    sigma_u = intensity * speed * shape_t[None, :]
    sigma_v = CROSS_COMPONENT_ANISOTROPY * sigma_u
    sigma_w = CROSS_COMPONENT_ANISOTROPY * intensity * speed * shape_m[None, :]
    sigma_w[:, 0] = 0.0

    fluct = filtered_fluctuations(
        seed=seed,
        start_index=start_index,
        times=times,
        speed=speed,
        n_z=nz,
        n_y=ny,
        dz=dz,
        dy=dy,
        length_scale_x=length_scale_x,
        length_scale_y=length_scale_y,
        length_scale_z=length_scale_z,
    )
    u = fluct["u"] * sigma_u[:, :, None]
    u -= u.mean(axis=(1, 2), keepdims=True)
    return {
        "u": u,
        "v": fluct["v"] * sigma_v[:, :, None],
        "w": fluct["w"] * sigma_w[:, :, None],
    }


def _interpolate_records(
    records: np.ndarray, record_times: np.ndarray, step_times: np.ndarray
) -> np.ndarray:
    """Linear interpolation along axis 0, clamped at both ends."""
    position = np.interp(step_times, record_times, np.arange(record_times.size))
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, record_times.size - 1)
    weight = (position - lower)[:, None, None]
    return (1.0 - weight) * records[lower] + weight * records[upper]


def build_forcing_buffers(
    *,
    fluctuations: dict[str, np.ndarray],
    record_times: np.ndarray,
    step_times: np.ndarray,
    advection_speed: np.ndarray,
    c_u: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn m/s fluctuation planes into the solver's per-step forcing buffers.

    Args:
        fluctuations: Output of :func:`build_digital_filter_fluctuations`.
        record_times: Window time of each plane (s).
        step_times: Window time of each step ``l = 1..L`` (s), shape ``(L,)``.
        advection_speed: The inflow's x-velocity in lattice units per step and
            level, shape ``(L, nz)``; the forcing increment is this times the
            fluctuation (see the module docstring).
        c_u: Lattice velocity scale (m/s per lattice unit).

    Returns:
        ``uu, vv, ww`` of shape ``(ny, nz, L + 1)`` (the Fortran's
        ``(ny, nz, 0:nrturb)``), float32. Slot 0 is never read and stays zero.
    """
    w_centres = 0.5 * (fluctuations["w"][:, :-1, :] + fluctuations["w"][:, 1:, :])
    buffers = []
    for plane in (fluctuations["u"], fluctuations["v"], w_centres):
        per_step = _interpolate_records(plane, record_times, step_times)
        increment = per_step / c_u * advection_speed[:, :, None]  # (L, nz, ny)
        buffer = np.zeros((plane.shape[2], plane.shape[1], step_times.size + 1))
        buffer[:, :, 1:] = np.transpose(increment, (2, 1, 0))
        buffers.append(buffer.astype(np.float32))
    return buffers[0], buffers[1], buffers[2]


def write_turbulence_restart(
    dirs: "DirectoryPaths",
    iteration: int,
    uu: np.ndarray,
    vv: np.ndarray,
    ww: np.ndarray,
) -> None:
    """Write ``restart/turbulence_0000_<iteration>.uf`` as m_saverestart does.

    One unformatted record ``ny, nz, nrturb, uu, vv, ww, rr`` with each buffer
    ``(ny, nz, 0:nrturb)`` in Fortran order. ``rr`` is never read by the
    forcing (``m_inflow_turbulence_forcing.F90`` keeps rho), so it is zero.
    """
    ny, nz, slots = uu.shape
    restart_dir = dirs.experiment_dir / "restart"
    restart_dir.mkdir(parents=True, exist_ok=True)
    path = restart_dir / restart_file_name(iteration, prefix="turbulence")
    with FortranFile(str(path), "w") as f:
        f.write_record(
            np.int32(ny),
            np.int32(nz),
            np.int32(slots - 1),
            *(
                np.ravel(a, order="F").astype(np.float32)
                for a in (uu, vv, ww, np.zeros_like(uu))
            ),
        )


def write_digital_filter_forcing(
    *,
    dirs: "DirectoryPaths",
    config: dict,
    seed: int,
    nt0: int,
    window_steps: int,
    seconds_per_timestep: float,
    c_u: float,
    window_start_time: float,
    knot_times: np.ndarray,
    knot_speed: np.ndarray,
    knot_angle: np.ndarray,
    speed_scale: float,
    profile_config: Optional[dict],
    ny: int,
    nz: int,
    dy: float,
    dz: float,
    zsize: float,
) -> None:
    """Write this window's forcing buffers and switch the forcing on.

    Args:
        dirs: The member's directories.
        config: The validated ``inlet_turbulence`` block.
        seed: Digital-filter seed.
        nt0: First iteration of the window, a multiple of ``window_steps``.
        window_steps: ``nt1 - nt0``, written as ``nrturb``.
        seconds_per_timestep: ``C_l / C_u``.
        c_u: Lattice velocity scale.
        window_start_time: The member's elapsed physical time (s); picks the
            slice of its turbulence history, so windows continue each other.
        knot_times, knot_speed, knot_angle: The window's inflow schedule (s,
            m/s, degrees), window-relative and including any spinup plateau.
        speed_scale: ``reference_speed_scale`` of the active profile.
        profile_config: The active vertical-profile config.
        ny, nz, dy, dz, zsize: The LBM grid.
    """
    settings = digital_filter_settings(config)
    time_step = settings["time_step"]

    step_times = np.arange(window_steps) * seconds_per_timestep
    n_records = int(math.ceil(step_times[-1] / time_step - 1e-9)) + 1
    record_times = np.arange(n_records + PLANE_MARGIN_RECORDS) * time_step

    def schedule(times: np.ndarray, values: np.ndarray) -> np.ndarray:
        if knot_times.size == 1:
            return np.full(times.size, float(values[0]))
        return np.interp(times, knot_times, values)

    start_index = int(round(max(0.0, window_start_time) / time_step))
    fluctuations = build_digital_filter_fluctuations(
        times=record_times,
        start_index=start_index,
        velocity_magnitude=schedule(record_times, knot_speed),
        ny=ny,
        nz=nz,
        dy=dy,
        dz=dz,
        zsize=zsize,
        profile_config=profile_config,
        intensity=settings["intensity"],
        length_scale_x=settings["length_scale_x"],
        length_scale_y=settings["length_scale_y"],
        length_scale_z=settings["length_scale_z"],
        seed=seed,
    )

    # The x-velocity the inflow kernel imposes at each step: uini * uvel_shear
    # (normalised by its top cell, m_inflow.F90) * cos(udir), in lattice units.
    heights = (np.arange(nz) + 0.5) * dz
    shape = build_profile_shape(profile_config, heights, zsize)
    shape = shape / shape[-1]
    speed = schedule(step_times, knot_speed) * speed_scale
    angle = np.deg2rad(schedule(step_times, knot_angle))
    advection = (speed * np.cos(angle))[:, None] * shape[None, :] / c_u

    uu, vv, ww = build_forcing_buffers(
        fluctuations=fluctuations,
        record_times=record_times,
        step_times=step_times,
        advection_speed=advection,
        c_u=c_u,
    )
    buffer_bytes = 4 * uu.nbytes
    if buffer_bytes > TURBULENCE_BUFFER_WARN_BYTES:
        logger.warning(
            "The digital-filter forcing buffers take %.0f MB (4 x %d x %d x %d "
            "float32): resident on the device for the whole run and copied "
            "through the host when read. Shorten the window or coarsen the "
            "grid if memory is tight.",
            buffer_bytes / 1e6,
            ny,
            nz,
            window_steps + 1,
        )
    write_turbulence_restart(dirs, nt0, uu, vv, ww)
    set_lturb(dirs, True, "1.0", window_steps)
    logger.info(
        "LBM digital-filter inflow: %d planes at dt=%.3f s onto %d steps from "
        "nt0=%d (window starts at t=%.1f s -> record %d), intensity=%.3f, "
        "L=(%.1f, %.1f, %.1f) m, seed=%d",
        record_times.size,
        time_step,
        window_steps,
        nt0,
        window_start_time,
        start_index,
        settings["intensity"],
        settings["length_scale_x"],
        settings["length_scale_y"],
        settings["length_scale_z"],
        seed,
    )
