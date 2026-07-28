"""Inflow (inlet) turbulence forcing for the LBM solver.

The Fortran ships a complete inflow-turbulence subsystem
(``m_inflow_turbulence_{init,compute,apply,update,forcing}.F90``) that
superimposes smooth pseudo-random perturbations on the inlet plane. It is
driven by three values that ``m_readinfile.F90`` reads with a single
list-directed statement::

    read(10,*,err=100) inflowturbulence, turbulence_ampl, nrturb

which appears in ``infile.in`` as one positional line::

     F 0.00005  100   ! lturb amp nrtu  : Add turbulence forcing on inflow, ...

Because the whole line is consumed by one ``read``, the three values must be
written together — see ``Infile.set_value_tokens``.

``main.F90`` gates every turbulence call on ``inflowturbulence`` and gates the
periodic refresh on ``ibnd == 1``, so the feature is only meaningful with the
inflow/outflow boundary condition.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .dir_utils import DirectoryPaths

from .infile_utils import Infile

logger = logging.getLogger(__name__)

# Key of the three-value inflow-turbulence line in infile.in (the first word of
# its "! lturb amp nrtu : ..." comment, which is how Infile indexes lines).
INLET_TURBULENCE_KEY = "lturb"

# Accepted keys of the ``inlet_turbulence`` config block. The same three names
# are used by pyudales and pypalm so the knob reads identically across backends.
INLET_TURBULENCE_FIELDS = ("enabled", "amplitude", "update_interval")


def validate_inlet_turbulence(
    inlet_turbulence: Optional[dict],
    boundary_condition: str,
) -> Optional[dict[str, Any]]:
    """Validate the ``inlet_turbulence`` config block at construction time.

    Args:
        inlet_turbulence: ``{"enabled": bool, "amplitude": float,
            "update_interval": int}``; ``None`` disables the feature entirely.
        boundary_condition: The model's x-boundary condition.

    Returns:
        A normalized copy of the block, or ``None`` when it is absent.

    Raises:
        ValueError: On unknown keys, a non-positive ``update_interval``, a
            negative ``amplitude``, or turbulence requested with a periodic
            boundary condition.
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

    return cfg


def apply_inlet_turbulence(
    inlet_turbulence: Optional[dict],
    dirs: "DirectoryPaths",
) -> None:
    """Write the ``lturb`` line of ``infile.in``.

    Strict no-op when ``inlet_turbulence`` is ``None`` or disabled, so default
    runs keep the solver-generated line byte-identical. Values not supplied fall
    back to the ones already in the file rather than to invented defaults.

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

    infile = Infile(dirs.infile_path)
    tokens = infile.get_value_tokens(INLET_TURBULENCE_KEY)
    if tokens is None or len(tokens) != 3:
        raise ValueError(
            f"Expected a 3-value '{INLET_TURBULENCE_KEY}' line "
            f"(inflowturbulence, turbulence_ampl, nrturb) in {dirs.infile_path}, "
            f"found {tokens!r}. Refusing to edit a positional infile whose "
            "layout does not match the solver template."
        )

    amplitude = inlet_turbulence.get("amplitude")
    update_interval = inlet_turbulence.get("update_interval")
    amplitude_token = tokens[1] if amplitude is None else f"{float(amplitude):.10g}"
    interval_token = tokens[2] if update_interval is None else str(int(update_interval))

    infile.set_value_tokens(
        INLET_TURBULENCE_KEY, [True, amplitude_token, interval_token]
    )
    infile.write()
    logger.info(
        "Enabled LBM inflow turbulence (amplitude=%s, update_interval=%s)",
        amplitude_token,
        interval_token,
    )
