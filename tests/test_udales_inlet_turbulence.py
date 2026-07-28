"""Tests for the pyudales ``inlet_turbulence`` knob.

The knob exists for schema parity with the other backends, but uDALES v2.2.0
cannot honour it: the Lund (1998) recycling/rescaling inlet generator
(``iinletgen=1``) is dead code in the vendored solver —

* ``iinletgen`` is not a member of any namelist
  (``u-dales/src/modstartup.f90:108-173``; ``&INLET`` at ``:141-144`` declares
  only ``Uinf``/``Vinf``/``di``/``dti``/``inletav``/``linletRA``/
  ``lstoreplane``/``lreadminl``/``lfixinlet``/``lfixutauin``/``lwallfunc``), so
  writing it makes the namelist read abort with
  ``ERROR: Problem in namoptions INLET`` / ``iostat error: 5010``;
* ``call initinlet`` is commented out (``program.f90:77``,
  ``modstartup.f90:627``) and ``inletgen`` (``modinlet.f90:204``) has no call
  site anywhere;
* ``modboundary.f90`` never reads the generator's output ``u0inletbc`` — the
  west inlet face is set purely by ``select case(BCxm)`` (``:1204-1262``).

So the contract under test is: the disabled/absent path is a strict no-op
(namoptions byte-identical, and in particular no ``iinletgen`` key anywhere),
and the enabled path fails loudly at construction instead of shipping a run that
crashes at solver startup.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

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


def _udales_src() -> pathlib.Path:
    """Path to the vendored uDALES Fortran sources (``UDALES_PATH`` may be None)."""
    from pyudales import UDALES_PATH

    if UDALES_PATH is None:  # pragma: no cover - submodule not checked out
        pytest.skip("u-dales source not available")
    assert UDALES_PATH is not None
    return pathlib.Path(UDALES_PATH) / "src"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_disabled_or_absent_is_accepted() -> None:
    """Absent / empty / ``enabled: false`` validates cleanly under either BC."""
    from pyudales.forward_model import validate_inlet_turbulence

    for block in (None, {}, {"enabled": False}):
        for bc in ("periodic", "inflow_outflow"):
            validate_inlet_turbulence(block, bc)


def test_enabled_under_inflow_outflow_is_rejected_with_a_reason() -> None:
    """The blocking finding is surfaced, not silently swallowed."""
    from pyudales.forward_model import validate_inlet_turbulence

    with pytest.raises(ValueError) as excinfo:
        validate_inlet_turbulence({"enabled": True}, "inflow_outflow")

    message = str(excinfo.value)
    # The message must name the culprit and point at the evidence, so a future
    # reader does not re-derive it.
    assert "iinletgen" in message
    assert "modstartup.f90" in message
    assert "BCxm=3" in message


def test_enabled_under_periodic_is_rejected_for_having_no_inlet() -> None:
    """A turbulent *inlet* is meaningless when the x boundaries wrap."""
    from pyudales.forward_model import validate_inlet_turbulence

    with pytest.raises(ValueError, match="inflow_outflow"):
        validate_inlet_turbulence({"enabled": True}, "periodic")


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
# No-op guarantee
# ---------------------------------------------------------------------------


def test_disabled_path_writes_nothing_to_namoptions(tmp_path: pathlib.Path) -> None:
    """Validation touches no file, so a disabled run is byte-identical.

    ``validate_inlet_turbulence`` is the *entire* implementation — there is no
    namoptions write site — so the strongest available statement of the no-op
    rule is that the file is unchanged across every accepted input.
    """
    from pyudales.forward_model import validate_inlet_turbulence
    from pyudales.utils.namoptions_utils import NamoptionsFile

    namoptions_path = tmp_path / "namoptions.999"
    namoptions_path.write_text(
        "&RUN\niexpnr = 999\nruntime = 100.\n/\n"
        "&DOMAIN\nitot = 4\njtot = 4\nktot = 4\n/\n"
        "&BC\nBCxm = 2\nBCym = 1\nBCtopm = 3\n/\n"
        "&INLET\nUinf = 0.\n/\n"
    )
    before = namoptions_path.read_bytes()

    for block in (None, {}, {"enabled": False}):
        validate_inlet_turbulence(block, "inflow_outflow")
        assert namoptions_path.read_bytes() == before

    # And nothing anywhere in the wrapper writes the key that would crash uDALES.
    assert "iinletgen" not in NamoptionsFile(namoptions_path).get_section_keys("INLET")


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
# The ground truth this feature is blocked on, asserted against the real source
# ---------------------------------------------------------------------------


def test_iinletgen_is_not_declared_in_any_udales_namelist() -> None:
    """Pin the reason the knob cannot be honoured, so a solver bump surfaces it.

    If a future uDALES version adds ``iinletgen`` to a namelist this test fails,
    which is the signal to revisit ``validate_inlet_turbulence``.
    """
    src = _udales_src()
    if not src.exists():  # pragma: no cover - submodule not checked out
        pytest.skip("u-dales source not available")

    # Collect every namelist declaration body in the source tree.
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

    # Sanity: the parser really did find the &INLET block.
    assert INLET_NAMELIST_KEYS <= declared

    assert "iinletgen" not in declared


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
