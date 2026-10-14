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

import json
import pathlib
import re
import types
from typing import Any, Optional

import numpy as np
import pytest
import xarray
from pylbm.utils.infile_utils import Infile
from pylbm.utils.inlet_turbulence_utils import (
    INLET_TURBULENCE_KEY,
    aligned_start_iteration,
    apply_inlet_turbulence,
    build_digital_filter_fluctuations,
    build_forcing_buffers,
    uses_digital_filter,
    validate_inlet_turbulence,
    write_turbulence_restart,
)
from pylbm.utils.warm_start_utils import restart_file_name
from scipy.io import FortranFile

REPO = pathlib.Path(__file__).resolve().parents[2]
LBM_SRC = REPO / "libs" / "pylbm" / "LBM" / "src"
STL_PATH = REPO / "geometries" / "xie_and_castro" / "xie_castro_2008_STL.stl"
PARAMS = xarray.Dataset(data_vars={"inflow_angle": 10.0, "velocity_magnitude": 6.0})

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
    @pytest.mark.parametrize(  # type: ignore[misc]
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

    @pytest.mark.parametrize("interval", [0, -5])  # type: ignore[misc]
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


class TestMethodSelection:
    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            validate_inlet_turbulence(
                {"enabled": True, "method": "recycling"}, "inflow_outflow"
            )

    def test_digital_filter_keys_are_accepted(self) -> None:
        cfg = validate_inlet_turbulence(
            {
                "enabled": True,
                "method": "digital_filter",
                "intensity": 0.1,
                "length_scale_x": 24.0,
                "length_scale_y": 12.0,
                "length_scale_z": 12.0,
                "time_step": 0.5,
                "seed": 3,
            },
            "inflow_outflow",
        )
        assert uses_digital_filter(cfg)

    @pytest.mark.parametrize(  # type: ignore[misc]
        "key, value",
        [("time_step", 0.0), ("length_scale_y", -1.0), ("intensity", -0.1)],
    )
    def test_bad_digital_filter_values_raise(self, key: str, value: float) -> None:
        with pytest.raises(ValueError, match=key):
            validate_inlet_turbulence(
                {"enabled": True, "method": "digital_filter", key: value},
                "inflow_outflow",
            )

    def test_method_absent_is_random_forcing(self) -> None:
        cfg = validate_inlet_turbulence({"enabled": True}, "inflow_outflow")
        assert not uses_digital_filter(cfg)

    @pytest.mark.parametrize("enabled", [True, False])  # type: ignore[misc]
    def test_compile_time_write_skips_digital_filter(
        self, tmp_path: pathlib.Path, enabled: bool
    ) -> None:
        """The digital filter writes its lturb line per window, not at compile."""
        dirs = _dirs(tmp_path)
        original = dirs.infile_path.read_bytes()
        apply_inlet_turbulence({"enabled": enabled, "method": "digital_filter"}, dirs)
        assert dirs.infile_path.read_bytes() == original


class TestDigitalFilterPlanes:
    """The fluctuations are the shared generator's, laid out for the forcing."""

    NY, NZ, DY, DZ, ZSIZE = 12, 6, 2.0, 2.0, 12.0
    PROFILE = {"type": "power_law", "alpha": 0.25, "z_ref": 64.0}
    SCALES = dict(
        intensity=0.15, length_scale_x=24.0, length_scale_y=6.0, length_scale_z=4.0
    )

    def _fluctuations(self, times: np.ndarray, start_index: int = 0) -> dict:
        return build_digital_filter_fluctuations(
            times=times,
            start_index=start_index,
            velocity_magnitude=np.full(times.size, 6.0),
            ny=self.NY,
            nz=self.NZ,
            dy=self.DY,
            dz=self.DZ,
            zsize=self.ZSIZE,
            profile_config=self.PROFILE,
            seed=11,
            **self.SCALES,
        )

    def test_matches_the_udales_inlet_planes(self) -> None:
        """Same seed, grid and records -> the uDALES inlet's fluctuations."""
        from pyudales.utils.inflow_utils import angle_to_velocity
        from pyudales.utils.inlet_turbulence_utils import build_driver_planes
        from pyudales.utils.vertical_profile import build_profile_shape as udales_shape

        times = np.arange(8) * 0.5
        speed, angle = np.full(8, 6.0), np.full(8, 10.0)
        u, v, w = build_driver_planes(
            jtot=self.NY,
            ktot=self.NZ,
            zsize=self.ZSIZE,
            ylen=self.NY * self.DY,
            times=times,
            start_index=5,
            inflow_angle=angle,
            velocity_magnitude=speed,
            profile_config=self.PROFILE,
            seed=11,
            **self.SCALES,
        )
        shape = udales_shape(self.PROFILE, (np.arange(self.NZ) + 0.5) * self.DZ, 12.0)
        u_ref, v_ref = angle_to_velocity(angle, speed)
        interior = (slice(None), slice(1, self.NZ + 1), slice(1, self.NY + 1))

        ours = self._fluctuations(times, start_index=5)
        u_fluct = u[interior] - (u_ref[:, None] * shape)[:, :, None]
        v_fluct = v[interior] - (v_ref[:, None] * shape)[:, :, None]
        np.testing.assert_allclose(ours["u"], u_fluct, atol=1e-12)
        np.testing.assert_allclose(ours["v"], v_fluct, atol=1e-12)
        np.testing.assert_allclose(
            ours["w"], w[:, 1 : self.NZ + 2, 1 : self.NY + 1], atol=1e-12
        )

    def test_seeds_give_independent_realisations(self) -> None:
        times = np.arange(40) * 0.5
        first = self._fluctuations(times)["u"].ravel()
        other = build_digital_filter_fluctuations(
            times=times,
            start_index=0,
            velocity_magnitude=np.full(times.size, 6.0),
            ny=self.NY,
            nz=self.NZ,
            dy=self.DY,
            dz=self.DZ,
            zsize=self.ZSIZE,
            profile_config=self.PROFILE,
            seed=12,
            **self.SCALES,
        )["u"].ravel()
        assert abs(np.corrcoef(first, other)[0, 1]) < 0.1

    def test_forcing_buffers_layout_and_scaling(self) -> None:
        """Slot l is step l at window time (l-1)*dt, scaled by U_x / C_u**2."""
        record_times = np.arange(5) * 0.5
        fluct = self._fluctuations(record_times)
        dt, c_u = 0.125, 90.0
        steps = 12
        step_times = np.arange(steps) * dt
        advection = np.linspace(0.02, 0.06, self.NZ)[None, :].repeat(steps, axis=0)
        uu, vv, ww = build_forcing_buffers(
            fluctuations=fluct,
            record_times=record_times,
            step_times=step_times,
            advection_speed=advection,
            c_u=c_u,
        )
        assert uu.shape == vv.shape == ww.shape == (self.NY, self.NZ, steps + 1)
        assert uu.dtype == np.float32
        assert not uu[:, :, 0].any()
        # Step 1 sits on record 0; step 5 (t = 0.5 s) on record 1; step 3
        # (t = 0.25 s) halfway between them.
        scale = advection[0][None, :] / c_u
        np.testing.assert_allclose(uu[:, :, 1], fluct["u"][0].T * scale, rtol=1e-5)
        np.testing.assert_allclose(vv[:, :, 5], fluct["v"][1].T * scale, rtol=1e-5)
        halfway = 0.5 * (fluct["u"][0] + fluct["u"][1]).T * scale
        np.testing.assert_allclose(uu[:, :, 3], halfway, rtol=1e-5, atol=1e-12)
        # w is averaged from the faces onto the cell centres.
        w_centres = 0.5 * (fluct["w"][0, :-1] + fluct["w"][0, 1:]).T * scale
        np.testing.assert_allclose(ww[:, :, 1], w_centres, rtol=1e-5, atol=1e-12)

    def test_turbulence_restart_matches_the_fortran_read(
        self, tmp_path: pathlib.Path
    ) -> None:
        """One record: ny, nz, nrturb, then uu, vv, ww, rr in Fortran order."""
        if (LBM_SRC / "m_readrestart.F90").exists():
            source = (LBM_SRC / "m_readrestart.F90").read_text()
            assert "read(iunit,err=998)j,k,l,uu,vv,ww,rr" in source.replace(" ", "")
            assert "prefix='turbulence'" in source
        rng = np.random.default_rng(0)
        uu, vv, ww = (
            rng.standard_normal((self.NY, self.NZ, 8)).astype(np.float32)
            for _ in range(3)
        )
        dirs: Any = types.SimpleNamespace(experiment_dir=tmp_path)
        write_turbulence_restart(dirs, 70, uu, vv, ww)

        path = tmp_path / "restart" / restart_file_name(70, prefix="turbulence")
        n = uu.size
        with FortranFile(str(path), "r") as f:
            header, *arrays = f.read_record(
                np.dtype((np.int32, (3,))),
                *(np.dtype((np.float32, (n,))) for _ in range(4)),
            )
        assert header.tolist() == [self.NY, self.NZ, 7]
        for written, read in zip((uu, vv, ww), arrays):
            np.testing.assert_array_equal(
                read.reshape(written.shape, order="F"), written
            )
        assert not arrays[3].any()

    @pytest.mark.parametrize(  # type: ignore[misc]
        "previous, steps, expected",
        [
            (None, 100, 100),
            (0, 100, 100),
            (100, 100, 100),
            (101, 100, 200),
            (450, 90, 450),
        ],
    )
    def test_aligned_start_iteration(
        self, previous: Optional[int], steps: int, expected: int
    ) -> None:
        assert aligned_start_iteration(previous, steps) == expected


def _solver_template_infile() -> str:
    """The infile.in the solver writes for itself (m_mkinfile.F90)."""
    source = (LBM_SRC / "m_mkinfile.F90").read_text()
    lines = re.findall(r"write\(10,'\(a\)'\)'(.*)'", source)
    return "\n".join(lines) + "\n"


class _Launched(Exception):
    """Raised instead of running the solver, once everything is staged."""


@pytest.mark.skipif(  # type: ignore[misc]
    not (LBM_SRC / "m_mkinfile.F90").exists(), reason="LBM sources not checked out"
)
class TestDigitalFilterStaging:
    """What a digital-filter window hands the solver, without running it."""

    def _model(self, tmp_path: pathlib.Path) -> Any:
        from pylbm.forward_model import ForwardModel

        model = ForwardModel(
            stl_path=STL_PATH,
            temp_dir=tmp_path,
            nx=12,
            ny=10,
            nz=4,
            bounds=((0.0, 12.0), (0.0, 10.0), (0.0, 4.0)),
            cuda=False,
            verbose=False,
            boundary_condition="inflow_outflow",
            profile_config={"type": "power_law", "alpha": 0.25, "z_ref": 8.0},
            inlet_turbulence={
                "enabled": True,
                "method": "digital_filter",
                "length_scale_y": 2.0,
                "length_scale_z": 2.0,
                "seed": 4,
            },
            simulation_time=2.0,
            output_frequency=1.0,
            spinup_time=1.0,
        )
        model.dirs.infile_path.write_text(_solver_template_infile())
        return model

    def _stage(self, model: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
        def launched() -> None:
            raise _Launched

        monkeypatch.setattr(model, "run", launched)
        with pytest.raises(_Launched):
            model.run_single(params=PARAMS, **kwargs)
        return Infile(model.dirs.infile_path)

    def test_cold_window_is_a_restart_run(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = self._model(tmp_path)
        infile = self._stage(model, monkeypatch)

        steps = model._window_timesteps
        # C_u = 90, C_l = 1 m: 3 s of spinup + window is 270 steps.
        assert steps == 270
        assert infile.get_value_as_int("nt0") == steps
        assert infile.get_value_as_int("nt1") == 2 * steps
        assert infile.get_value_tokens(INLET_TURBULENCE_KEY) == ["T", "1.0", str(steps)]
        restart = model.dirs.experiment_dir / "restart"
        assert (restart / restart_file_name(steps)).exists()
        turbulence = restart / restart_file_name(steps, prefix="turbulence")
        with FortranFile(str(turbulence), "r") as f:
            header = f.read_ints(np.int32)[:3]
        assert header.tolist() == [10, 4, steps]
        # The run never finished, so the clock did not advance.
        assert not (model.dirs.experiment_dir / "inlet_turbulence_clock.json").exists()
        # z_ref = 8 m: uini is the speed at the top cell centre, 6*(3.5/8)**0.25.
        uini = float(infile.get_value_tokens("uini,")[0])  # type: ignore[index]
        assert uini == pytest.approx(6.0 * (3.5 / 8.0) ** 0.25, abs=1e-4)

    def test_warm_window_starts_on_the_nrturb_grid(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = self._model(tmp_path)
        restart = model.dirs.experiment_dir / "restart"
        restart.mkdir()
        # The previous window ended off the grid of this one's 180 steps.
        (restart / restart_file_name(500)).write_bytes(b"not a restart")
        state = xarray.Dataset(
            {
                name: (("z", "y", "x"), np.full((4, 10, 12), value))
                for name, value in (("rho", 1.0), ("u", 5.0), ("v", 1.0), ("w", 0.0))
            }
        )
        infile = self._stage(model, monkeypatch, state=state)

        assert model._window_timesteps == 180
        assert infile.get_value_as_int("nt0") == 540
        assert infile.get_value_as_int("nt1") == 720
        assert (restart / restart_file_name(540, prefix="turbulence")).exists()


def _tiny_digital_filter_run(
    root: pathlib.Path, enabled: bool = True
) -> tuple[Any, xarray.Dataset]:
    """A cold window of the tiny LBM case with the digital-filter inflow."""
    from hydra.utils import instantiate

    from tests.conftest import compose

    cfg = compose(
        "forward",
        "+test=forward",
        "model=pylbm_tiny",
        "model.forward_model.boundary_condition=inflow_outflow",
        "model.forward_model.sgs_constant=0.15",
        "++model.forward_model.profile_config.z_ref=6.0",
        f"model.forward_model.inlet_turbulence.enabled={str(enabled).lower()}",
        "model.forward_model.inlet_turbulence.method=digital_filter",
        "model.forward_model.inlet_turbulence.intensity=0.15",
        "model.forward_model.inlet_turbulence.length_scale_x=4.0",
        "model.forward_model.inlet_turbulence.length_scale_y=2.0",
        "model.forward_model.inlet_turbulence.length_scale_z=2.0",
        "model.forward_model.inlet_turbulence.time_step=0.5",
        "model.forward_model.inlet_turbulence.seed=7",
        "time.simulation_time=3.0",
        "time.output_frequency=1.0",
        "time.spinup_time=1.0",
        root=root,
    )
    model = instantiate(cfg.model.forward_model)
    instantiate(cfg.model.prepare, forward_model=model)
    return model, model.run_single(params=PARAMS).load()


@pytest.mark.integration  # type: ignore[misc]
def test_digital_filter_runs_reproducibly_across_windows(
    tmp_path: pathlib.Path,
) -> None:
    """Cold + warm window: aligned restarts, a clock, and no clock-seeded RNG."""
    model, cold = _tiny_digital_filter_run(tmp_path / "a")
    _, again = _tiny_digital_filter_run(tmp_path / "b")
    _, laminar = _tiny_digital_filter_run(tmp_path / "off", enabled=False)

    for name in ("u", "v", "w"):
        assert np.isfinite(cold[name]).all()
        # The forcing is ours, not the solver's clock-seeded RNG: bit for bit.
        np.testing.assert_array_equal(cold[name], again[name], err_msg=name)
    # The same run without the forcing is laminar (the tiny case wraps the
    # Xie & Castro block, so the flow varies in y either way): the
    # fluctuations show up as the difference, at the 0.15 * 6 m/s scale.
    difference = float(np.abs(cold.u - laminar.u).max())
    assert 0.05 < difference < 10.0

    clock = model.dirs.experiment_dir / "inlet_turbulence_clock.json"
    assert json.loads(clock.read_text())["elapsed_time"] == pytest.approx(4.0)
    infile = Infile(model.dirs.infile_path)
    first_nt1 = infile.get_value_as_int("nt1")
    assert first_nt1 is not None

    warm = model.run_single(state=cold, params=PARAMS).load()
    assert np.isfinite(warm.u).all()
    infile = Infile(model.dirs.infile_path)
    nt0, nt1 = infile.get_value_as_int("nt0"), infile.get_value_as_int("nt1")
    assert nt0 is not None and nt1 is not None
    assert nt0 >= first_nt1 and nt0 % (nt1 - nt0) == 0
    assert json.loads(clock.read_text())["elapsed_time"] == pytest.approx(7.0)
