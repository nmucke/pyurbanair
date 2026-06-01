"""Regression tests for the inflow-body-force alignment between uDALES and LBM.

Under ``inflow_outflow`` boundary conditions the uDALES flow is driven by the
west inlet face (plus nudging), so the constant body-force pressure gradient
(``INPS`` ``dpdx``/``dpdy``) inherited from the case ``namoptions`` must be
zeroed.  Left non-zero it adds a direction-frozen momentum source that LBM (an
inlet-driven solver with no body force) has no equivalent of, which biases
cross-model ESMDA (uDALES truth -> LBM assim).

Under ``periodic`` boundary conditions there is no inlet, so the body force is
the sole driver and must be preserved.
"""

import pathlib

import numpy as np
import xarray
from pyudales.utils.dir_utils import DirectoryPaths
from pyudales.utils.namoptions_utils import NamoptionsFile
from pyudales.utils.nudging_utils import apply_time_varying_inflow
from pyudales.utils.params_utils import apply_inflow_settings

# Non-zero body force inherited from the case namoptions (matches the
# xie_and_castro example values).
CASE_DPDX = 0.0029343
CASE_DPDY = -0.0029343


def _make_experiment(tmp_path: pathlib.Path) -> DirectoryPaths:
    """Create a minimal experiment dir (namoptions + prof.inp + lscale.inp)."""
    namoptions_path = tmp_path / "namoptions.999"
    namoptions_path.write_text(
        "&RUN\niexpnr = 999\nruntime = 100.\n/\n"
        "&DOMAIN\nitot = 4\njtot = 4\nktot = 4\n/\n"
        "&INPS\nzsize = 10.\nu0 = 2.0\nv0 = -2.0\n"
        f"dpdx = {CASE_DPDX}\ndpdy = {CASE_DPDY}\n/\n"
    )

    dz = 10.0 / 4
    heights = [0.5 * dz + k * dz for k in range(4)]
    with open(tmp_path / "prof.inp.999", "w") as f:
        f.write("# SDBL flow \n")
        f.write("# z thl qt u v tke\n")
        for z in heights:
            f.write(
                f"{z:<20.15f} 288.000000   0.000000     2.000000     "
                "-2.000000    0.000000\n"
            )
    with open(tmp_path / "lscale.inp.999", "w") as f:
        f.write("# SDBL flow \n")
        f.write("# z uq vq pqx pqy wfls dqtdxls dqtdyls dqtdtls dthlrad\n")
        for z in heights:
            f.write(
                f"{z:<20.15f} 0.000000     0.000000     {CASE_DPDX:.9f}  "
                f"{CASE_DPDY:.9f} 0.000000000     0.000000     0.000000     "
                "0.000000     0.000000000000\n"
            )

    return DirectoryPaths(
        case_dir=tmp_path,
        experiment_dir=tmp_path,
        experiment_name="999",
        experiment_base_dir=tmp_path,
        temp_dir=tmp_path,
        output_dir=tmp_path,
        udales_root_path=tmp_path,
        cwd=tmp_path,
        results_dir=None,
    )


def _scalar_params() -> xarray.Dataset:
    return xarray.Dataset(
        data_vars={
            "inflow_angle": 0.0,
            "velocity_magnitude": 5.0,
            "pressure_gradient_magnitude": 0.0041912,
        }
    )


class TestNudgingPath:
    """``apply_time_varying_inflow`` is the path that actually runs at runtime."""

    def test_inflow_outflow_zeroes_body_force(self, tmp_path: pathlib.Path) -> None:
        dirs = _make_experiment(tmp_path)
        apply_time_varying_inflow(
            _scalar_params(),
            dirs,
            boundary_condition="inflow_outflow",
            tnudge=10.0,
            nnudge=0,
        )

        nf = NamoptionsFile(tmp_path / "namoptions.999")
        assert nf.get_value_as_float("INPS", "dpdx") == 0.0
        assert nf.get_value_as_float("INPS", "dpdy") == 0.0

    def test_periodic_preserves_body_force(self, tmp_path: pathlib.Path) -> None:
        dirs = _make_experiment(tmp_path)
        apply_time_varying_inflow(
            _scalar_params(),
            dirs,
            boundary_condition="periodic",
            tnudge=10.0,
            nnudge=0,
        )

        nf = NamoptionsFile(tmp_path / "namoptions.999")
        np.testing.assert_allclose(nf.get_value_as_float("INPS", "dpdx"), CASE_DPDX)
        np.testing.assert_allclose(nf.get_value_as_float("INPS", "dpdy"), CASE_DPDY)


class TestStaticPath:
    """``apply_inflow_settings`` honours the same contract for consistency."""

    def test_inflow_outflow_zeroes_body_force(self, tmp_path: pathlib.Path) -> None:
        dirs = _make_experiment(tmp_path)
        apply_inflow_settings(
            _scalar_params(), dirs, boundary_condition="inflow_outflow"
        )

        nf = NamoptionsFile(tmp_path / "namoptions.999")
        assert nf.get_value_as_float("INPS", "dpdx") == 0.0
        assert nf.get_value_as_float("INPS", "dpdy") == 0.0

    def test_periodic_writes_pressure_gradient(self, tmp_path: pathlib.Path) -> None:
        dirs = _make_experiment(tmp_path)
        # angle=0 with magnitude 0.0041912 -> dpdx = magnitude, dpdy = 0.
        apply_inflow_settings(_scalar_params(), dirs, boundary_condition="periodic")

        nf = NamoptionsFile(tmp_path / "namoptions.999")
        np.testing.assert_allclose(
            nf.get_value_as_float("INPS", "dpdx"), 0.0041912, atol=1e-7
        )
        np.testing.assert_allclose(
            nf.get_value_as_float("INPS", "dpdy"), 0.0, atol=1e-7
        )
