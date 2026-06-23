"""Unit tests for the model-error compensation parameters.

Covers the per-member consumption of ``vertical_inflow_exponent`` (the power-law
shear exponent α) and ``sgs_constant`` (the sub-grid-scale mixing constant) added
in docs/esmda_model_error_parameters.md, plus the mixed static+dynamic sampler
and the uDALES param whitelist that previously dropped unknown variables.

These are deliberately solver-free: they exercise the file-writer / sampler
helpers directly so the suite stays fast (the end-to-end ESMDA smoke runs live in
test_run_esmda.py).
"""

from __future__ import annotations

import pathlib
import types

import jax
import numpy as np
import pytest
import xarray


# ---------------------------------------------------------------------------
# pylbm: α override + SGS (ivreman) write
# ---------------------------------------------------------------------------


def test_pylbm_resolve_profile_config_overrides_alpha() -> None:
    from pylbm.utils.params_utils import resolve_profile_config

    default = {"type": "power_law", "alpha": 0.25, "z_ref": 40.0}

    # Absent -> None (keep construction-time uvel_shear.dat).
    assert resolve_profile_config(xarray.Dataset(), default) is None

    # Present -> alpha overridden, other keys preserved.
    params = xarray.Dataset({"vertical_inflow_exponent": 0.4})
    out = resolve_profile_config(params, default)
    assert out["alpha"] == pytest.approx(0.4)
    assert out["type"] == "power_law"
    assert out["z_ref"] == 40.0


def test_pylbm_apply_sgs_setting_writes_ivreman(tmp_path: pathlib.Path) -> None:
    from pylbm.utils.infile_utils import Infile
    from pylbm.utils.params_utils import apply_sgs_setting

    infile_path = tmp_path / "infile.in"
    infile_path.write_text(
        " 1 0.15           ! ivreman smagor  : Vreman subgridscale mixing\n"
    )
    dirs = types.SimpleNamespace(infile_path=infile_path)

    # Absent -> untouched.
    apply_sgs_setting(xarray.Dataset(), dirs)
    assert Infile(infile_path).get_value("ivreman") == "1 0.15"

    # Present -> the Smagorinsky token is rewritten, ivreman flag kept at 1.
    apply_sgs_setting(xarray.Dataset({"sgs_constant": 0.3}), dirs)
    assert Infile(infile_path).get_value("ivreman") == "1 0.3000"


# ---------------------------------------------------------------------------
# uDALES: the param whitelist must carry the new knobs through (§6.3)
# ---------------------------------------------------------------------------


def test_udales_whitelist_carries_model_error_params() -> None:
    from pyudales.utils.params_utils import extract_inflow_params, merge_params

    params = xarray.Dataset(
        {
            "inflow_angle": 0.0,
            "velocity_magnitude": 5.0,
            "vertical_inflow_exponent": 0.3,
            "sgs_constant": 0.2,
        }
    )

    extracted = extract_inflow_params(params)
    assert "vertical_inflow_exponent" in extracted
    assert "sgs_constant" in extracted

    merged = merge_params(xarray.Dataset({"inflow_angle": 1.0}), params)
    assert merged["sgs_constant"].item() == pytest.approx(0.2)
    assert merged["vertical_inflow_exponent"].item() == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Mixed static + dynamic sampler (AR(2) relaxation)
# ---------------------------------------------------------------------------


def _mixed_model(seed: int = 1):
    from pyurbanair.dynamic_parameters.ar2_relaxation import AR2RelaxationModel
    from pyurbanair.static_parameters import Constant, Normal

    return AR2RelaxationModel(
        external_parameters={"inflow_angle": Normal(25.0, 6.0)},
        simulation_time=10.0,
        seconds_per_knot=5.0,
        correlation_length=100.0,
        seed=seed,
        static_parameters={
            "sgs_constant": Normal(0.15, 0.04, min=0.01),
            "vertical_inflow_exponent": Constant(0.25),
        },
    )


def test_ar2_static_parameters_have_no_time_dim() -> None:
    model = _mixed_model()
    ds = model.sample(16)

    # Dynamic var keeps the time dim; statics are ensemble-only.
    assert "time" in ds["inflow_angle"].dims
    assert ds["sgs_constant"].dims == ("ensemble",)
    assert ds["vertical_inflow_exponent"].dims == ("ensemble",)

    # Constant static is identical across the ensemble; Normal static varies.
    assert np.allclose(ds["vertical_inflow_exponent"].values, 0.25)
    assert float(ds["sgs_constant"].std()) > 0.0


def test_ar2_statics_carried_forward_not_rerandomized() -> None:
    """extrapolate() inherits the ESMDA-updated statics from the posterior."""
    model = _mixed_model()
    posterior = model.sample(16)  # stand-in for an updated posterior

    prediction_times = jax.numpy.asarray(model.time_coords) + 10.0
    nxt = model.extrapolate(posterior, prediction_times, jax.random.PRNGKey(3))

    # Statics flow through unchanged (no fresh draw between windows).
    np.testing.assert_allclose(
        nxt["sgs_constant"].values, posterior["sgs_constant"].values
    )
    np.testing.assert_allclose(
        nxt["vertical_inflow_exponent"].values,
        posterior["vertical_inflow_exponent"].values,
    )


def test_parameter_metrics_and_plotting_cover_static_knobs(
    tmp_path: pathlib.Path,
) -> None:
    """The parameter figures/metrics include the static model-error knobs.

    Covers the generalized `_plotted_param_names` path: a posterior carrying a
    time-varying inflow var plus two constant-in-time knobs (one with an
    ``(ensemble,)`` truth) yields metrics for all three, and the rollout figure
    renders without error (constant truth drawn as a horizontal line).
    """
    from pyurbanair.plotting import (
        compute_parameter_metrics,
        plot_rollout_time_evolution,
    )

    ens = 5
    rng = np.random.RandomState(0)
    posterior = xarray.Dataset(
        {
            "inflow_angle": (("time", "ensemble"), rng.randn(3, ens) + 10.0),
            "sgs_constant": (("ensemble",), np.full(ens, 0.22)),
            "vertical_inflow_exponent": (("ensemble",), np.linspace(0.2, 0.3, ens)),
        },
        coords={"ensemble": np.arange(ens), "time": [0.0, 1.5, 3.0]},
    )
    truth = xarray.Dataset(
        {
            "inflow_angle": (("time", "ensemble"), np.full((3, 1), 10.0)),
            "sgs_constant": (("ensemble",), [0.24]),
            "vertical_inflow_exponent": (("ensemble",), [0.25]),
        },
        coords={"ensemble": [0], "time": [0.0, 1.5, 3.0]},
    )

    metrics = compute_parameter_metrics(posterior, truth, prior_params=posterior)
    assert {"inflow_angle", "sgs_constant", "vertical_inflow_exponent"} <= set(metrics)
    for name, m in metrics.items():
        assert m["rmse"].shape == m["x"].shape

    out = tmp_path / "rollout.png"
    plot_rollout_time_evolution(
        esmda_params=posterior,
        true_params=truth,
        esmda_state=None,
        true_state=None,
        output_path=out,
        prior_params=posterior,
        rmse=np.array([1.0, 0.5, 0.25]),
    )
    assert out.exists()


def test_pypalm_sgs_setting_disables_constant_flux_layer(
    tmp_path: pathlib.Path,
) -> None:
    """PALM forbids a fixed km with a surface flux layer (PAC0149): writing
    km_constant must also set constant_flux_layer=.false.; absent -> untouched."""
    import shutil

    from pypalm.forward_model import ForwardModel
    from pypalm.utils.p3d_utils import P3DFile

    case_p3d = pathlib.Path("examples/palm/xie_and_castro/_p3d")
    staged = tmp_path / "urban_run_p3d"
    shutil.copy2(case_p3d, staged)

    class _Stub:
        _param_value = staticmethod(ForwardModel._param_value)

    # Absent -> neither key is written (default prognostic SGS-TKE regime).
    p3d = P3DFile(staged)
    ForwardModel._apply_sgs_setting(_Stub(), p3d, xarray.Dataset())
    p3d.write()
    reread = P3DFile(staged)
    assert reread.get_value("initialization_parameters", "km_constant") is None
    assert reread.get_value("initialization_parameters", "constant_flux_layer") is None

    # Present -> km_constant set AND constant_flux_layer forced false (PAC0149).
    p3d = P3DFile(staged)
    ForwardModel._apply_sgs_setting(_Stub(), p3d, xarray.Dataset({"sgs_constant": 0.3}))
    p3d.write()
    reread = P3DFile(staged)
    assert float(
        reread.get_value("initialization_parameters", "km_constant")
    ) == pytest.approx(0.3)
    assert (
        reread.get_value("initialization_parameters", "constant_flux_layer")
        == ".false."
    )


def test_ar2_without_static_parameters_is_unchanged() -> None:
    """Omitting static_parameters reproduces the pure time-varying Dataset."""
    from pyurbanair.dynamic_parameters.ar2_relaxation import AR2RelaxationModel
    from pyurbanair.static_parameters import Normal

    model = AR2RelaxationModel(
        external_parameters={"inflow_angle": Normal(25.0, 6.0)},
        simulation_time=10.0,
        seconds_per_knot=5.0,
        correlation_length=100.0,
        seed=1,
    )
    ds = model.sample(8)
    assert set(ds.data_vars) == {"inflow_angle"}
