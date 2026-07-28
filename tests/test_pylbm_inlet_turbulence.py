"""Tests for the pylbm inlet (inflow) turbulence knob.

The LBM ships a full inflow-turbulence subsystem
(``m_inflow_turbulence_*.F90``) driven by three values that
``m_readinfile.F90`` reads with one list-directed statement::

    read(10,*,err=100) inflowturbulence, turbulence_ampl, nrturb

``infile.in`` is positional: the Fortran reads value lines strictly in order, so
a rewrite that drops, splits, or reorders a line silently shifts every
subsequent read. These tests therefore assert not only the new values but also
that the surrounding lines and the overall line count are untouched.
"""

import pathlib
import types
from typing import Any, Optional

import pytest
from pylbm.utils.inlet_turbulence_utils import (
    INLET_TURBULENCE_KEY,
    apply_inlet_turbulence,
    validate_inlet_turbulence,
)

# Verbatim excerpt of the infile.in the solver generates for itself
# (m_mkinfile.F90), including the neighbours of the turbulence line.
TEMPLATE_INFILE = """# Boundary conditions
 1                ! ibnd            : 0-periodic, 1 in/out flow,
 0                ! jbnd            : 0-periodic, 11,12,21,22 no-slip bb(1), free-slip bb(2) for j=1 and j=ny
 22               ! kbnd            : 0-periodic, 11,12,21,22 no-slip bb(1), free-slip bb(2) for k=1 and k=nz
# Inflow variables
 8.0 0.0          ! uini, udir      : Inflow wind velocity [m/s], direction in degrees (-45:45)
 F 0.00005  100   ! lturb amp nrtu  : Add turbulence forcing on inflow, amplitude, number of prestored time ste
# Physical variables
 0.0000178        ! visckin         : Dimensional kinematic viscosity
 1.225            ! C_rho - Density of air at surface 15C and  101.325 kPa  [kg/m^3] Eq. (7.12)
"""

# Index of the ' F 0.00005  100 ...' line within TEMPLATE_INFILE.
TURBULENCE_LINE_INDEX = 6


def _dirs(tmp_path: pathlib.Path) -> Any:
    """Minimal stand-in exposing the only attribute the writer uses."""
    infile_path = tmp_path / "infile.in"
    infile_path.write_text(TEMPLATE_INFILE)
    return types.SimpleNamespace(infile_path=infile_path)


def _turbulence_tokens(infile_path: pathlib.Path) -> list[str]:
    line = infile_path.read_text().splitlines()[TURBULENCE_LINE_INDEX]
    assert INLET_TURBULENCE_KEY in line
    return line.split("!")[0].split()


class TestApplyInletTurbulenceEnabled:
    def test_writes_all_three_values(self, tmp_path: pathlib.Path) -> None:
        dirs = _dirs(tmp_path)
        apply_inlet_turbulence(
            {"enabled": True, "amplitude": 0.0002, "update_interval": 250}, dirs
        )

        flag, amplitude, interval = _turbulence_tokens(dirs.infile_path)
        assert flag == "T"
        assert float(amplitude) == pytest.approx(0.0002)
        assert int(interval) == 250

    def test_preserves_positional_layout(self, tmp_path: pathlib.Path) -> None:
        """Only the lturb line may change; the infile is read positionally."""
        dirs = _dirs(tmp_path)
        before = TEMPLATE_INFILE.splitlines()

        apply_inlet_turbulence(
            {"enabled": True, "amplitude": 0.0002, "update_interval": 250}, dirs
        )
        after = dirs.infile_path.read_text().splitlines()

        assert len(after) == len(before)
        for i, (old, new) in enumerate(zip(before, after)):
            if i == TURBULENCE_LINE_INDEX:
                assert old != new
            else:
                assert old == new, f"line {i} was modified"

    def test_comment_is_preserved(self, tmp_path: pathlib.Path) -> None:
        dirs = _dirs(tmp_path)
        apply_inlet_turbulence({"enabled": True, "amplitude": 1e-4}, dirs)

        line = dirs.infile_path.read_text().splitlines()[TURBULENCE_LINE_INDEX]
        assert (
            line.split("!", 1)[1]
            == TEMPLATE_INFILE.splitlines()[TURBULENCE_LINE_INDEX].split("!", 1)[1]
        )

    def test_unspecified_values_fall_back_to_the_infile(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Missing knobs keep the solver template's values, not invented ones."""
        dirs = _dirs(tmp_path)
        apply_inlet_turbulence({"enabled": True}, dirs)

        flag, amplitude, interval = _turbulence_tokens(dirs.infile_path)
        assert flag == "T"
        assert float(amplitude) == pytest.approx(5e-05)
        assert int(interval) == 100

    def test_amplitude_is_fortran_readable(self, tmp_path: pathlib.Path) -> None:
        """A list-directed real read needs a parseable numeric literal."""
        dirs = _dirs(tmp_path)
        apply_inlet_turbulence({"enabled": True, "amplitude": 1.25e-06}, dirs)

        _, amplitude, _ = _turbulence_tokens(dirs.infile_path)
        assert float(amplitude) == pytest.approx(1.25e-06)
        assert "," not in amplitude

    def test_rejects_a_malformed_turbulence_line(self, tmp_path: pathlib.Path) -> None:
        infile_path = tmp_path / "infile.in"
        infile_path.write_text(" F                ! lturb amp nrtu  : truncated\n")
        dirs: Any = types.SimpleNamespace(infile_path=infile_path)

        with pytest.raises(ValueError, match="3-value"):
            apply_inlet_turbulence({"enabled": True}, dirs)


class TestApplyInletTurbulenceNoOp:
    @pytest.mark.parametrize(
        "cfg",
        [
            None,
            {"enabled": False},
            {"enabled": False, "amplitude": 0.5, "update_interval": 7},
        ],
        ids=["absent", "disabled", "disabled_with_values"],
    )
    def test_infile_is_byte_identical(
        self, tmp_path: pathlib.Path, cfg: Optional[dict]
    ) -> None:
        dirs = _dirs(tmp_path)
        original = dirs.infile_path.read_bytes()

        apply_inlet_turbulence(cfg, dirs)

        assert dirs.infile_path.read_bytes() == original


class TestValidateInletTurbulence:
    def test_absent_is_none(self) -> None:
        assert validate_inlet_turbulence(None, "periodic") is None

    def test_enabled_with_periodic_raises(self) -> None:
        with pytest.raises(ValueError, match="inflow_outflow"):
            validate_inlet_turbulence({"enabled": True}, "periodic")

    def test_disabled_with_periodic_is_allowed(self) -> None:
        """The shipped default (enabled: false) must not break periodic runs."""
        cfg = validate_inlet_turbulence(
            {"enabled": False, "amplitude": 5e-05, "update_interval": 100}, "periodic"
        )
        assert cfg is not None and cfg["enabled"] is False

    def test_enabled_with_inflow_outflow_is_allowed(self) -> None:
        cfg = validate_inlet_turbulence({"enabled": True}, "inflow_outflow")
        assert cfg is not None and cfg["enabled"] is True

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown inlet_turbulence key"):
            validate_inlet_turbulence(
                {"enabled": True, "amplitud": 1e-5}, "inflow_outflow"
            )

    @pytest.mark.parametrize("interval", [0, -5])
    def test_non_positive_update_interval_raises(self, interval: int) -> None:
        with pytest.raises(ValueError, match="update_interval"):
            validate_inlet_turbulence(
                {"enabled": True, "update_interval": interval}, "inflow_outflow"
            )

    def test_negative_amplitude_raises(self) -> None:
        with pytest.raises(ValueError, match="amplitude"):
            validate_inlet_turbulence(
                {"enabled": True, "amplitude": -1.0}, "inflow_outflow"
            )


class TestInfileMultiValueEditor:
    """The generic 3-value support added to the Infile editor."""

    def test_get_and_set_tokens_roundtrip(self, tmp_path: pathlib.Path) -> None:
        from pylbm.utils.infile_utils import Infile

        dirs = _dirs(tmp_path)
        infile = Infile(dirs.infile_path)
        assert infile.get_value_tokens(INLET_TURBULENCE_KEY) == ["F", "0.00005", "100"]

        infile.set_value_tokens(INLET_TURBULENCE_KEY, [True, "1e-4", 50])
        infile.write()

        assert Infile(dirs.infile_path).get_value_tokens(INLET_TURBULENCE_KEY) == [
            "T",
            "1e-4",
            "50",
        ]

    def test_missing_key_returns_none(self, tmp_path: pathlib.Path) -> None:
        from pylbm.utils.infile_utils import Infile

        assert Infile(_dirs(tmp_path).infile_path).get_value_tokens("nope") is None

    def test_empty_tokens_raise(self, tmp_path: pathlib.Path) -> None:
        from pylbm.utils.infile_utils import Infile

        infile = Infile(_dirs(tmp_path).infile_path)
        with pytest.raises(ValueError):
            infile.set_value_tokens(INLET_TURBULENCE_KEY, [])


class TestForwardModelWiring:
    def test_constructor_accepts_inlet_turbulence(self) -> None:
        """The knob must be a real constructor arg (Hydra instantiates by name)."""
        import inspect

        from pylbm.forward_model import ForwardModel

        signature = inspect.signature(ForwardModel.__init__)
        assert "inlet_turbulence" in signature.parameters
        assert signature.parameters["inlet_turbulence"].default is None

    def test_config_defaults_match_the_solver_template(self) -> None:
        """conf/model/pylbm.yaml must not silently change the solver defaults."""
        import yaml  # type: ignore[import-untyped]

        conf_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "conf"
            / "model"
            / "pylbm.yaml"
        )
        cfg = yaml.safe_load(conf_path.read_text())["forward_model"]["inlet_turbulence"]

        template_tokens = TEMPLATE_INFILE.splitlines()[TURBULENCE_LINE_INDEX]
        _, amplitude, interval = template_tokens.split("!")[0].split()
        assert cfg["enabled"] is False
        assert float(cfg["amplitude"]) == pytest.approx(float(amplitude))
        assert int(cfg["update_interval"]) == int(interval)
