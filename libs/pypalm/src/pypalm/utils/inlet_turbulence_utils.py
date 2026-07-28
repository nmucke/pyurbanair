"""Switchable inlet turbulence for PALM's ``inflow_outflow`` runs.

This module maps the cross-backend ``inlet_turbulence`` knob onto PALM's
built-in **random inflow-disturbance** machinery (``create_disturbances`` +
``dt_disturb`` + ``disturbance_amplitude`` + ``inflow_disturbance_begin/end``).

Why *this* machinery and not ``turbulent_inflow``
-------------------------------------------------
PALM ships three inlet-turbulence mechanisms. Only one of them is
independently toggleable on pypalm's staging (all line numbers refer to
``libs/pypalm/palm_model_system/MAKE_DEPOSITORY_default/``, PALM 25.10):

1. ``turbulent_inflow_method = 'recycle_*'`` — the classic Kataoka & Mizuno
   turbulence recycling (``turbulent_inflow_mod.f90:283-286`` lists the four
   legal method strings). It is the true analogue of uDALES ``iinletgen=1``,
   **but** ``turbulent_inflow_mod.f90:337-343`` (``TUI0006``) rejects it unless
   ``initializing_actions`` is ``'cyclic_fill'`` or ``'read_restart_data'``.
   pypalm cold-starts from ``'set_constant_profiles'`` and warm-starts from
   ``'read_from_file'``, so recycling would require a separate precursor run
   plus restart-data plumbing. Not reachable from a config toggle.

2. The synthetic turbulence generator (``synthetic_turbulence_generator_mod.f90``)
   — requires an ``STG_PROFILES`` input file for idealized setups
   (``:412-417``, ``STG0007``), ``random_generator='random-parallel'``
   (``:451-455``), and is **mutually exclusive** with ``turbulent_inflow``
   (``:444-448``, ``STG0009``) — i.e. it cannot coexist with pypalm's
   time-varying inflow driver, which is what carries the ESMDA-estimated
   ``inflow_angle``/``velocity_magnitude``.

3. Random inflow disturbances — ``time_integration.f90:966-989``. For a
   non-cyclic lateral boundary PALM re-injects random u/v perturbations in the
   strip ``[inflow_disturbance_begin, inflow_disturbance_end]``
   ("runs with a non-cyclic lateral wall need perturbations near the inflow
   throughout the whole simulation", ``time_integration.f90:980-985``). It
   needs no extra input file, no precursor run, and is orthogonal to
   ``turbulent_inflow`` — so it composes with the dynamic driver.

Mechanism 3 is what this module drives.

``turbulent_inflow`` is NOT an inlet-turbulence switch for us
-------------------------------------------------------------
pypalm already flips ``&turbulent_inflow_parameters switch_off_module=.false.``
with ``turbulent_inflow_method='read_from_file'`` on the time-varying inflow
path (see :mod:`.dynamic_driver_utils`). In that mode the module is not a
turbulence generator at all — it is the *reader* for the ``_dynamic`` NetCDF
that carries the time-varying ``inflow_plane_u/v`` planes
(``turbulent_inflow_mod.f90:456-459``). Switching it off would silently
disconnect the ESMDA-estimated inflow signal from the solver, so
``inlet_turbulence.enabled: false`` deliberately does **not** touch it.

Namelist sections
-----------------
``inflow_disturbance_begin`` / ``inflow_disturbance_end`` live in
``&initialization_parameters`` (``parin.f90:222-223``); ``create_disturbances``,
``disturbance_amplitude``, ``disturbance_energy_limit``,
``disturbance_level_b/t`` and ``dt_disturb`` live in ``&runtime_parameters``
(``parin.f90:348-369``).
"""

import logging
import pathlib
from typing import Callable, Optional

from .p3d_utils import P3DFile

logger = logging.getLogger(__name__)


# --- pypalm-side defaults for the enabled path -------------------------------
DEFAULT_INLET_TURBULENCE_ENABLED = False

# The one-off random perturbation PALM imposes at cold-start init
# (init_3d_model.f90:1488). Defaults True because it is what breaks symmetry so
# turbulence can develop from PALM's smooth analytic initial profile -- turning
# it off is legitimate but leaves transition to be tripped by the topography
# alone. Independent of `enabled`, which governs the in-run perturbations.
DEFAULT_INLET_TURBULENCE_INITIAL_SEED = True
# PALM's own dt_disturb default is effectively infinite (see below), so the
# disturbance loop never fires today. A finite value is what actually turns the
# mechanism on; 5 s is short relative to any of our windows.
DEFAULT_DT_DISTURB = 5.0
# modules.f90:928 — PALM's own default, kept so `enabled: true` with no tunables
# reproduces upstream's perturbation strength.
DEFAULT_DISTURBANCE_AMPLITUDE = 0.25

# --- PALM's namelist defaults, used to restore the "off" state ---------------
PALM_DEFAULT_DT_DISTURB = 9999999.9  # modules.f90:939 (effectively never)
PALM_DEFAULT_DISTURBANCE_AMPLITUDE = 0.25  # modules.f90:928
PALM_DEFAULT_ENERGY_LIMIT = 0.01  # modules.f90:929
PALM_AUTO_INFLOW_DISTURBANCE = -1  # modules.f90:650-651 (-1 => derive from nx)

DEFAULT_INLET_TURBULENCE_CONFIG: dict = {
    "enabled": DEFAULT_INLET_TURBULENCE_ENABLED,
    "dt_disturb": DEFAULT_DT_DISTURB,
    "amplitude": DEFAULT_DISTURBANCE_AMPLITUDE,
    "begin": None,
    "end": None,
    "level_b": None,
    "level_t": None,
}

_INIT = "initialization_parameters"
_RUNTIME = "runtime_parameters"

# Keys this module may write. Used by the teardown path so a run with the knob
# off cannot inherit a previous enabled run's namelist in the same experiment
# dir. ``create_disturbances`` is deliberately absent: it is owned by the
# warm-start / cold-start init path in ForwardModel.
_RESTORE_ON_DISABLE: tuple[tuple[str, str, float], ...] = (
    (_RUNTIME, "dt_disturb", PALM_DEFAULT_DT_DISTURB),
    (_RUNTIME, "disturbance_amplitude", PALM_DEFAULT_DISTURBANCE_AMPLITUDE),
    (_RUNTIME, "disturbance_energy_limit", PALM_DEFAULT_ENERGY_LIMIT),
    (_INIT, "inflow_disturbance_begin", PALM_AUTO_INFLOW_DISTURBANCE),
    (_INIT, "inflow_disturbance_end", PALM_AUTO_INFLOW_DISTURBANCE),
)


def is_initial_seed_enabled(config: Optional[dict]) -> bool:
    """True when the cold-start seeding perturbation should be applied.

    Absent config keeps PALM's (and this wrapper's) historical behaviour: the
    seed is ON. It is deliberately independent of ``enabled`` -- the seed is a
    single kick at t=0, whereas ``enabled`` governs perturbations re-injected
    near the inflow for the whole run.
    """
    if not config:
        return DEFAULT_INLET_TURBULENCE_INITIAL_SEED
    return bool(config.get("initial_seed", DEFAULT_INLET_TURBULENCE_INITIAL_SEED))


def is_inlet_turbulence_enabled(config: Optional[dict]) -> bool:
    """True when an ``inlet_turbulence`` dict explicitly enables the knob.

    ``None`` (absent) is a strict no-op, exactly like ``enabled: false``.
    """
    if not config:
        return False
    return bool(config.get("enabled", DEFAULT_INLET_TURBULENCE_ENABLED))


def validate_inlet_turbulence(config: Optional[dict], boundary_condition: str) -> None:
    """Reject ``inlet_turbulence`` combinations PALM cannot honour.

    Under fully cyclic boundaries PALM's disturbance machinery has no inflow
    strip at all: ``check_parameters.f90:2754`` only derives
    ``inflow_disturbance_begin/end`` when ``bc_lr /= 'cyclic'``, and the
    "keep perturbing near the inflow" branch in ``time_integration.f90:976-985``
    is gated on ``.NOT. bc_lr_cyc .OR. .NOT. bc_ns_cyc``. Enabling the knob for
    a periodic run would therefore silently mean "random kicks everywhere while
    TKE is below ``disturbance_energy_limit``", which is not inlet turbulence.
    Fail loudly instead.
    """
    if not is_inlet_turbulence_enabled(config):
        return
    if boundary_condition != "inflow_outflow":
        raise ValueError(
            "inlet_turbulence.enabled=true requires "
            "boundary_condition='inflow_outflow'. PALM only maintains inflow "
            "perturbations at a non-cyclic lateral boundary "
            "(time_integration.f90:976-985); under 'periodic' the knob would be "
            "a no-op dressed up as a feature. Use inflow_outflow, or turn the "
            "knob off."
        )


def apply_inlet_turbulence(
    p3d_path: pathlib.Path,
    config: Optional[dict],
    warm_start: bool = False,
) -> bool:
    """Write (or clear) PALM's random inflow-disturbance settings in ``_p3d``.

    Returns ``True`` when the knob was applied, ``False`` when it was off.

    Precedence notes:

    - ``config is None`` or ``enabled: false`` → teardown only, and even that
      touches solely the keys that are already present in the namelist. On a
      clean template nothing is written, so staging is byte-identical to the
      pre-knob behaviour. In particular the implicit
      ``turbulent_inflow`` + dynamic-driver coupling on the time-varying path is
      left untouched.
    - ``enabled: true`` runs *after* the warm-start / cold-start init block, so
      it takes precedence over the ``create_disturbances`` value written there.
      On a warm start ``disturbance_energy_limit`` is forced to ``0.0``, which
      suppresses the one-off initial perturbation of the injected field
      (``init_3d_model.f90:1488`` is gated on
      ``create_disturbances .AND. disturbance_energy_limit /= 0.0``) while
      leaving the in-run inflow perturbations active via the ``ELSEIF`` branch
      at ``time_integration.f90:976``. That preserves today's "don't shock the
      warm-started field" intent.
    """
    p3d = P3DFile(p3d_path)
    seed = is_initial_seed_enabled(config)

    if not is_inlet_turbulence_enabled(config):
        _restore_defaults(p3d)
        # `create_disturbances` is written unconditionally by the cold-start init
        # path, so leaving it alone here means "no inlet turbulence" still fires
        # the t=0 kick. Honour `initial_seed` — but only WRITE when the value
        # must change, so a clean template stays byte-identical (PALM's own
        # default is .TRUE., matching initial_seed's default):
        #   * seed false -> always write .false., adding the key if absent.
        #   * seed true  -> only rewrite a key that is already present, so a
        #     previous `initial_seed: false` run in the same experiment dir
        #     cannot leak its .false. into this one.
        # dt_disturb is left at PALM's huge default by _restore_defaults, so the
        # in-run loop never fires either way.
        if not seed:
            p3d.set_value(_RUNTIME, "create_disturbances", False)
        elif "create_disturbances" in p3d.sections.get(_RUNTIME, {}):
            p3d.set_value(_RUNTIME, "create_disturbances", True)
        p3d.write()
        logger.info(
            "PALM inlet turbulence OFF (initial_seed=%s -> create_disturbances=%s)",
            seed,
            ".true." if seed else ".false.",
        )
        return False

    cfg = config or {}
    dt_disturb = float(cfg.get("dt_disturb", DEFAULT_DT_DISTURB))
    if dt_disturb <= 0.0:
        raise ValueError(
            f"inlet_turbulence.dt_disturb must be > 0 s, got {dt_disturb}. "
            "PALM compares an accumulated time against it "
            "(time_integration.f90:971)."
        )
    amplitude = float(cfg.get("amplitude", DEFAULT_DISTURBANCE_AMPLITUDE))

    # create_disturbances gates BOTH the initial kick and the in-run loop
    # (time_integration.f90:966). Force it on; the energy limit below decides
    # whether the initial kick fires.
    p3d.set_value(_RUNTIME, "create_disturbances", True)
    p3d.set_value(_RUNTIME, "dt_disturb", dt_disturb)
    p3d.set_value(_RUNTIME, "disturbance_amplitude", amplitude)
    # `disturbance_energy_limit = 0.0` is how the initial kick is suppressed
    # while the in-run perturbations stay alive: init_3d_model.f90:1488 requires
    # `energy_limit /= 0`, whereas time_integration.f90:976 falls through to the
    # non-cyclic "keep perturbing near the inflow" ELSEIF, which does not. So
    # `enabled: true` + `initial_seed: false` is representable, and a warm start
    # forces the same suppression so the injected field is not shocked.
    suppress_initial_kick = warm_start or not seed
    p3d.set_value(
        _RUNTIME,
        "disturbance_energy_limit",
        0.0 if suppress_initial_kick else PALM_DEFAULT_ENERGY_LIMIT,
    )

    _set_optional(p3d, _INIT, "inflow_disturbance_begin", cfg.get("begin"), int)
    _set_optional(p3d, _INIT, "inflow_disturbance_end", cfg.get("end"), int)
    _set_optional(p3d, _RUNTIME, "disturbance_level_b", cfg.get("level_b"), float)
    _set_optional(p3d, _RUNTIME, "disturbance_level_t", cfg.get("level_t"), float)

    p3d.write()
    if warm_start:
        why = " (warm start: initial kick suppressed)"
    elif not seed:
        why = " (initial_seed=false: initial kick suppressed)"
    else:
        why = ""
    logger.info(
        "PALM inlet turbulence ON: dt_disturb=%.3f s, amplitude=%.3f m/s%s",
        dt_disturb,
        amplitude,
        why,
    )
    return True


def _set_optional(
    p3d: P3DFile,
    section: str,
    key: str,
    value: Optional[float],
    cast: Callable[[float], float | int],
) -> None:
    """Write ``key`` only when the config supplies it (``None`` → PALM default)."""
    if value is None:
        return
    p3d.set_value(section, key, cast(value))


def _restore_defaults(p3d: P3DFile) -> None:
    """Reset only the disturbance keys that are actually present.

    Keeping this present-keys-only mirrors ``_disable_nudging_apparatus``: a
    clean template is left untouched (PALM defaults them itself), while a
    previous ``enabled: true`` run in the same experiment dir cannot leak its
    settings into a run with the knob off.
    """
    for section, key, default in _RESTORE_ON_DISABLE:
        if key in p3d.sections.get(section, {}):
            p3d.set_value(section, key, default)
