"""Plan 07 phase 3: ``neural_surrogates.generator_evaluation``.

Unit tests hit the metric functions on tiny analytic fields (a divergence-free
field, a constant field, a pure translation, an obstacle next to a stencil
cell).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from neural_surrogates import generator_evaluation as ge


def _grid_fields(n: int = 1, shape: tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    return np.zeros((n, 3, *shape))


# --------------------------------------------------------------------------- #
# Masks / divergence
# --------------------------------------------------------------------------- #


def test_stencil_mask_excludes_boundary_and_obstacle_neighbours() -> None:
    fluid = np.ones((6, 6, 6), dtype=bool)
    fluid[3, 3, 3] = False  # one obstacle cell
    stencil = ge.stencil_fluid_mask(fluid)
    # Domain boundary is never stencil-valid.
    assert not stencil[0].any() and not stencil[-1].any()
    assert not stencil[:, 0].any() and not stencil[:, :, -1].any()
    # The obstacle and its six face neighbours are excluded; diagonals are not.
    assert not stencil[3, 3, 3]
    for dz, dy, dx in (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ):
        assert not stencil[3 + dz, 3 + dy, 3 + dx]
    assert stencil[2, 2, 2] and stencil[4, 4, 4] and stencil[1, 1, 1]


def test_divergence_free_field_is_zero_and_linear_field_is_exact() -> None:
    nz, ny, nx = 10, 10, 10
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    fluid = np.ones((nz, ny, nx), dtype=bool)
    stencil = ge.stencil_fluid_mask(fluid)
    # u = y, v = -x is divergence-free (and exactly so for central differences).
    fields = np.stack([y, -x, np.zeros_like(x)], axis=0)[None].astype(float)
    res = ge.divergence(fields, stencil, (1.0, 1.0, 1.0))
    assert res["count"] == stencil.sum() == 8**3
    assert res["rms"] == pytest.approx(0.0, abs=1e-12)
    assert res["per_sample_rms"].shape == (1,)
    # u = 2x, v = 3y, w = 4z with spacing (dz, dy, dx) = (0.5, 1, 2):
    # du/dx = 2/2, dv/dy = 3, dw/dz = 4/0.5 -> 12 everywhere.
    fields = np.stack([2.0 * x, 3.0 * y, 4.0 * z], axis=0)[None].astype(float)
    res = ge.divergence(fields, stencil, (0.5, 1.0, 2.0))
    assert res["rms"] == pytest.approx(12.0)
    assert res["mean_abs"] == pytest.approx(12.0)
    # An empty stencil is nan, not zero.
    empty = ge.divergence(fields, np.zeros_like(stencil), (1.0, 1.0, 1.0))
    assert np.isnan(empty["rms"]) and empty["count"] == 0


# --------------------------------------------------------------------------- #
# Profiles / stresses / diversity
# --------------------------------------------------------------------------- #


def test_profiles_average_fluid_cells_per_level_only() -> None:
    fields = _grid_fields(2, (4, 3, 3))
    fields[0, 0] = 1.0
    fields[1, 0] = 3.0
    fluid = np.ones((4, 3, 3), dtype=bool)
    fluid[1] = False  # a level with no fluid
    fluid[2, 0, 0] = False
    fields[:, 0, 2, 0, 0] = 100.0  # an obstacle value that must not leak in
    prof = ge.profiles(fields, fluid)
    assert prof["mean"][0, 0] == pytest.approx(2.0)
    assert prof["mean"][0, 2] == pytest.approx(2.0)
    assert np.isnan(prof["mean"][0, 1]) and np.isnan(prof["rms"][0, 1])
    assert prof["rms"][0, 0] == pytest.approx(1.0)  # samples at 1 and 3 about 2
    assert prof["count"].tolist() == [18, 0, 16, 18]
    assert prof["sample_mean"].shape == (2, 3, 4)
    assert ge.profile_rmse(prof, prof) == 0.0
    boot = ge.bootstrap_profile_rmse(prof["sample_mean"], n_resamples=20)
    assert np.isfinite(boot) and boot > 0


def test_constant_field_has_zero_reynolds_stress_and_shear_field_known() -> None:
    fluid = np.ones((4, 4, 4), dtype=bool)
    const = (
        np.ones((3, 3, 4, 4, 4)) * np.array([1.0, 2.0, 3.0])[None, :, None, None, None]
    )
    res = ge.reynolds_stresses(const, fluid)
    assert np.allclose(res["stress"], 0.0)
    assert np.allclose(np.nan_to_num(res["profile"]), 0.0)
    # Samples alternating u = +a, -a and v = +b, -b at every cell: <u'v'> = a*b
    # (ddof=1 over 4 samples), <u'u'> = a^2, w constant -> zero row / column.
    a, b = 0.5, 2.0
    signs = np.array([1.0, -1.0, 1.0, -1.0])
    fields = np.zeros((4, 3, 4, 4, 4))
    fields[:, 0] = a * signs[:, None, None, None]
    fields[:, 1] = b * signs[:, None, None, None]
    fields[:, 2] = 7.0
    res = ge.reynolds_stresses(fields, fluid)
    expected = np.array([[a * a, a * b, 0.0], [a * b, b * b, 0.0], [0.0, 0.0, 0.0]])
    expected *= 4 / 3  # ddof = 1
    assert np.allclose(res["stress"], expected)
    # One sample cannot give a covariance.
    single = ge.reynolds_stresses(fields[:1], fluid)
    assert np.isnan(single["stress"]).all()


def test_diversity_and_pairwise_spread() -> None:
    fluid = np.ones((4, 4, 4), dtype=bool)
    base = np.zeros((3, 4, 4, 4))
    # Two seeds per conditioning: identical for condition 0, offset by 1 for 1.
    samples = np.stack(
        [np.stack([base, base + 0.0]), np.stack([base, base + 1.0])], axis=1
    )  # (S=2, B=2, C, ...)
    div = ge.diversity(samples, fluid)
    assert div["per_condition"].tolist() == pytest.approx([0.0, 1.0])
    assert div["mean"] == pytest.approx(0.5)
    real = np.stack([base, base + 2.0, base + 4.0])
    assert ge.pairwise_rms_distance(real, fluid) == pytest.approx((2 + 4 + 2) / 3)
    assert np.isnan(ge.pairwise_rms_distance(real[:1], fluid))
    ke = ge.kinetic_energy(real, fluid)
    assert ke.tolist() == pytest.approx([0.0, 0.5 * 3 * 4.0, 0.5 * 3 * 16.0])


# --------------------------------------------------------------------------- #
# Distributions / spectra
# --------------------------------------------------------------------------- #


def test_wasserstein_of_a_translation_is_the_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5000)
    assert ge.wasserstein_1(a, a) == 0.0
    assert ge.wasserstein_1(a, a + 0.75) == pytest.approx(0.75, abs=1e-9)
    assert ge.wasserstein_1(a, a - 2.0) == pytest.approx(2.0, abs=1e-9)
    # Histograms on shared bins + W1 per component.
    ref = np.stack([a, a, a])
    vals = np.stack([a + 0.75, a, a - 2.0])
    edges = ge.histogram_edges(ref, n_bins=16)
    assert edges.shape == (3, 17)
    h = ge.histograms_and_w1(vals, ref, edges)
    assert h["w1"].tolist() == pytest.approx([0.75, 0.0, 2.0], abs=1e-9)
    assert h["counts"].shape == (3, 16) and h["counts"][1].sum() == a.size
    boot = ge.bootstrap_w1(ref, n_resamples=5)
    assert boot.shape == (3,) and np.all(boot > 0)
    # Fluid-value pooling / subsampling.
    fields = np.stack([np.stack([np.full((2, 2, 2), float(c)) for c in range(3)])])
    fluid = np.ones((2, 2, 2), dtype=bool)
    fluid[0, 0, 0] = False
    v = ge.fluid_values(fields, fluid)
    assert v.shape == (3, 7) and v[2].tolist() == [2.0] * 7
    assert ge.fluid_values(fields, fluid, max_values=3).shape == (3, 3)
    assert ge.pool_values([v, v], max_values=10).shape == (3, 10)


def test_spectra_pick_out_a_single_wavenumber_over_fluid_rows_only() -> None:
    nz, ny, nx = 3, 4, 32
    dx = 0.5
    x = np.arange(nx) * dx
    k0 = 2 * np.pi * 2 / (nx * dx)  # two wavelengths across the row
    u = np.broadcast_to(np.cos(k0 * x), (nz, ny, nx))
    fields = np.stack([u, np.zeros_like(u), np.zeros_like(u)])[None]
    fluid = np.ones((nz, ny, nx), dtype=bool)
    fluid[0, 0, 5] = False  # one obstacle removes one row
    spec = ge.spectra(fields, fluid, dx=dx)
    assert spec["rows"] == nz * ny - 1
    assert spec["k"][2] == pytest.approx(k0)
    peak = int(np.argmax(spec["energy"][0]))
    assert peak == 2
    assert spec["energy"][0, 2] == pytest.approx(0.25)  # |cos| amplitude 1 -> 1/4
    assert np.allclose(np.delete(spec["energy"][0], 2), 0.0, atol=1e-20)
    assert np.allclose(spec["energy"][1:], 0.0)
    # Distances: identical spectra are 0 dB; a factor 10 is 10 dB; empty -> nan.
    assert ge.log_spectral_distance(spec["energy"], spec["energy"])[0] == 0.0
    tenfold = ge.log_spectral_distance(spec["energy"], spec["energy"] / 10.0)
    assert tenfold[0] == pytest.approx(10.0) and np.isnan(tenfold[1])
    none = ge.spectra(fields, np.zeros_like(fluid), dx=dx)
    assert none["rows"] == 0 and np.isnan(none["energy"]).all()


def test_padding_sensitivity_is_none_without_padded_axes() -> None:
    fields = np.random.default_rng(1).standard_normal((2, 3, 20, 6, 6))
    fluid = np.ones((20, 6, 6), dtype=bool)
    assert (
        ge.padding_sensitivity(fields, fluid, (1, 1, 1), (False, False, False)) is None
    )
    res = ge.padding_sensitivity(fields, fluid, (1, 1, 1), (True, False, False), edge=4)
    assert res is not None and set(res) == {"z"}
    assert set(res["z"]) == {"edge", "interior"}
    assert np.isfinite(res["z"]["edge"]["rms_velocity"])
    assert np.isfinite(res["z"]["interior"]["divergence_rms"])


# --------------------------------------------------------------------------- #
# Merge / verdict
# --------------------------------------------------------------------------- #


def _group(fields: np.ndarray, fluid: np.ndarray) -> dict[str, Any]:
    stencil = ge.stencil_fluid_mask(fluid)
    return {
        "profiles": ge.profiles(fields, fluid),
        "spectra": ge.spectra(fields, fluid),
        "reynolds": ge.reynolds_stresses(fields, fluid),
        "divergence": ge.divergence(fields, stencil, (1.0, 1.0, 1.0)),
        "n": fields.shape[0],
    }


def test_merge_group_metrics_matches_pooled_computation() -> None:
    rng = np.random.default_rng(2)
    fluid = np.ones((6, 6, 8), dtype=bool)
    a = rng.standard_normal((3, 3, 6, 6, 8))
    b = rng.standard_normal((5, 3, 6, 6, 8))
    merged = ge.merge_group_metrics([_group(a, fluid), _group(b, fluid)])
    pooled = _group(np.concatenate([a, b]), fluid)
    assert merged["n"] == 8
    assert np.allclose(merged["profiles"]["mean"], pooled["profiles"]["mean"])
    assert np.allclose(merged["profiles"]["count"], pooled["profiles"]["count"])
    # The RMS merges in the square, so it matches the pooled RMS up to the
    # (tiny, here) between-group spread of the level means.
    assert np.allclose(merged["profiles"]["rms"], pooled["profiles"]["rms"], rtol=0.01)
    assert merged["profiles"]["sample_mean"].shape == (8, 3, 6)
    assert np.allclose(merged["spectra"]["energy"], pooled["spectra"]["energy"])
    assert merged["divergence"]["rms"] == pytest.approx(pooled["divergence"]["rms"])
    assert merged["divergence"]["per_sample_rms"].shape == (8,)
    cmp = ge.compare_to_real(merged, pooled)
    assert cmp["profile_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert cmp["spectra_lsd_db"] == pytest.approx(0.0, abs=1e-9)


def test_aggregate_report_applies_declared_tolerances() -> None:
    tol = {
        "profile_rmse_factor": 2.0,
        "w1_factor": 2.0,
        "divergence_factor": 3.0,
        "diversity_min_ratio": 0.25,
    }
    reference = {
        "bootstrap_profile_rmse": 0.1,
        "bootstrap_w1": 0.05,
        "real_divergence_rms": 1.0,
        "real_pairwise_spread": 2.0,
    }
    ok = {
        "ae_recon": {"profile_rmse": 0.2, "w1": 0.1, "divergence_rms": 1.5},
        "generated": {
            "profile_rmse": 0.3,
            "w1": 0.15,
            "divergence_rms": 4.0,
            "diversity": 1.0,
        },
    }
    rep = ge.aggregate_report(ok, reference, tol)
    assert rep["acceptance"] == {"passed": True, "failures": []}
    assert rep["tolerances"] == tol
    assert rep["checks"]["profile_rmse"]["bound"] == pytest.approx(0.4)
    assert rep["checks"]["divergence_rms"]["bound"] == pytest.approx(4.5)
    assert rep["checks"]["diversity_ratio"]["value"] == pytest.approx(0.5)
    bad = {
        "ae_recon": ok["ae_recon"],
        "generated": {
            "profile_rmse": 0.5,
            "w1": 0.15,
            "divergence_rms": 4.0,
            "diversity": 0.1,
        },
    }
    rep = ge.aggregate_report(bad, reference, tol)
    assert not rep["acceptance"]["passed"]
    failed = [f.split(":")[0] for f in rep["acceptance"]["failures"]]
    assert failed == ["profile_rmse", "diversity_ratio"]
    # A nan on the generated side never passes.
    rep = ge.aggregate_report(
        {"ae_recon": ok["ae_recon"], "generated": {}}, reference, tol
    )
    assert len(rep["acceptance"]["failures"]) == 4
