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


# --------------------------------------------------------------------------- #
# Latent spectrum
# --------------------------------------------------------------------------- #


def test_latent_spectrum_measures_generated_variance_in_the_real_basis() -> None:
    rng = np.random.default_rng(3)
    d = 40
    basis, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eig = np.logspace(1, -6, d)  # a tail down to near-empty directions

    def draw(n: int, variances: np.ndarray) -> np.ndarray:
        coords = rng.standard_normal((n, d)) * np.sqrt(variances)
        tokens = 0.3 + coords @ basis.T  # (n, D), mean off zero
        return tokens.T.reshape(1, d, n, 1, 1)  # (B, D, Zl, Yl, Xl)

    real = ge.merge_latent_moments([ge.latent_moments(draw(20000, eig))])
    noisy = eig.copy()
    noisy[20:] = 1.0  # unit noise in the bottom half, as a confined generator
    spec = ge.latent_spectrum(real, ge.latent_moments(draw(20000, noisy)), (20,))
    assert np.allclose(spec["real"], eig, rtol=0.05)
    assert list(spec["bands"]) == ["1_20", "21_40"]
    assert spec["bands"]["1_20"] == pytest.approx(1.0, rel=0.05)
    expected_tail = np.exp(np.mean(np.log(1.0 / eig[20:])))
    assert spec["bands"]["21_40"] == pytest.approx(expected_tail, rel=0.05)
    # Batches merge to the pooled moments; bands past D are dropped.
    a, b = draw(300, eig), draw(500, eig)
    merged = ge.merge_latent_moments([ge.latent_moments(a), ge.latent_moments(b)])
    pooled = ge.latent_moments(np.concatenate([a, b], axis=2))
    assert merged["n"] == 800 and np.allclose(merged["outer"], pooled["outer"])
    assert list(ge.latent_spectrum(merged, merged)["bands"]) == ["1_32", "33_40"]


# --------------------------------------------------------------------------- #
# Seed-ensemble metrics (synthetic Gaussians with a known answer)
# --------------------------------------------------------------------------- #

_SHAPE = (4, 6, 6)  # (nz, ny, nx)


def _gaussian_ensemble(
    rng: np.random.Generator,
    n: int,
    scale: float = 1.0,
    shift: float = 0.0,
) -> np.ndarray:
    """``n`` iid draws of a field whose mean and spread vary over the cells and
    grow with height; ``scale`` / ``shift`` distort the spread / mean."""
    z = np.arange(_SHAPE[0])[:, None, None]
    mean = 2.0 + np.sin(np.arange(np.prod(_SHAPE))).reshape(_SHAPE) + shift
    sigma = (0.5 + 0.25 * z) * np.ones(_SHAPE) * scale
    noise = rng.standard_normal((n, 3, *_SHAPE))
    return mean + sigma * noise


def _fluid() -> np.ndarray:
    fluid = np.ones(_SHAPE, dtype=bool)
    fluid[0, :2, :2] = False  # a building
    return fluid


def test_ensemble_moments_are_unbiased_per_fluid_cell() -> None:
    rng = np.random.default_rng(0)
    fields = _gaussian_ensemble(rng, 5)
    fluid = _fluid()
    mom = ge.ensemble_moments(fields.astype(np.float32), fluid)
    assert mom["n"] == 5 and mom["mean"].shape == (3, fluid.sum())
    cells = fields[:, :, fluid]
    assert np.allclose(mom["mean"], cells.mean(axis=0), atol=1e-6)
    assert np.allclose(mom["var"], cells.var(axis=0, ddof=1), atol=1e-5)
    assert list(np.bincount(mom["z"])) == list(fluid.sum(axis=(1, 2)))
    assert np.isnan(ge.ensemble_moments(fields[:1], fluid)["var"]).all()


def test_same_distribution_scores_as_the_solver() -> None:
    """Two independent draws of one distribution: debiased bias ~ 0, spread
    ratio ~ 1, fair CRPS = the leave-one-out oracle, flat ranks, ED ~ 0."""
    rng = np.random.default_rng(1)
    fluid = _fluid()
    real, gen = _gaussian_ensemble(rng, 64), _gaussian_ensemble(rng, 64)
    rm, gm = ge.ensemble_moments(real, fluid), ge.ensemble_moments(gen, fluid)
    bias = ge.mean_bias(gm, rm)
    assert np.all(np.abs(bias["standardised"]) < 0.02)
    assert np.all(bias["raw"] > bias["debiased"])  # the noise term is removed
    sp = ge.spread(gm, rm)
    assert np.allclose(sp["ratio"], 1.0, atol=0.05)
    assert np.allclose(sp["ratio_profile"], 1.0, atol=0.1)
    points = ge.select_points(fluid, 64, seed=0)
    rp, gp = ge.point_values(real, points), ge.point_values(gen, points)
    ratio = ge.fair_crps(gp, rp) / ge.fair_crps(rp, rp, exclude_self=True)
    assert np.allclose(ratio, 1.0, atol=0.05)
    ranks = ge.rank_histogram(gp, rp, bins=8)
    assert ranks["counts"].sum() == 3 * 64 * 64
    assert np.allclose(ranks["outer_mass"], 2 / 8, atol=0.03)
    assert np.allclose(ranks["density"], 1.0, atol=0.2)
    d = ge.rms_distances([real, gen], fluid)
    r_idx, g_idx = np.arange(64), np.arange(64, 128)
    ed = ge.energy_distance(d, g_idx, r_idx)
    assert abs(ed["relative"]) < 0.01


def test_under_and_over_dispersed_ensembles_show_their_signatures() -> None:
    rng = np.random.default_rng(2)
    fluid = _fluid()
    real = _gaussian_ensemble(rng, 64)
    narrow = _gaussian_ensemble(rng, 64, scale=0.5)
    wide = _gaussian_ensemble(rng, 64, scale=2.0)
    rm = ge.ensemble_moments(real, fluid)
    sp = ge.spread(ge.ensemble_moments(narrow, fluid), rm)
    assert np.allclose(sp["ratio"], 0.5, atol=0.05)
    assert sp["ratio_total"] == pytest.approx(0.5, abs=0.05)
    assert np.all(sp["log_ratio_rms"] > 0.5)  # log(0.5) = -0.69
    assert np.all(sp["map_corr"] > 0.5)  # the pattern is still right
    assert np.all(sp["excluded"] == 0)
    # A cell where the solver ensemble is constant (an enclosed courtyard) is
    # left out of the map correlation and the log ratio, and counted.
    walled = real.copy()
    walled[:, :, 3, 2, 2] = 1.0
    held = ge.spread(
        ge.ensemble_moments(narrow, fluid), ge.ensemble_moments(walled, fluid)
    )
    assert list(held["excluded"]) == [1, 1, 1]
    assert np.all(np.isfinite(held["log_ratio_rms"]) & np.isfinite(held["map_corr"]))
    points = ge.select_points(fluid, 64)
    rp = ge.point_values(real, points)
    oracle = ge.fair_crps(rp, rp, exclude_self=True)
    # Gaussian CRPS ratio for a spread s x the truth's: (sqrt(2 (1 + s^2)) - s)
    # / (2 - 1), so 1.081 at s = 0.5 and 1.162 at s = 2.
    for distorted, expected in ((narrow, 1.081), (wide, 1.162)):
        ratio = ge.fair_crps(ge.point_values(distorted, points), rp) / oracle
        assert np.allclose(ratio, expected, atol=0.04)
    u_shape = ge.rank_histogram(ge.point_values(narrow, points), rp, bins=8)
    hump = ge.rank_histogram(ge.point_values(wide, points), rp, bins=8)
    flat = ge.rank_histogram(rp, rp, bins=8, exclude_self=True)
    assert np.all(u_shape["outer_mass"] > 0.4) and np.all(hump["outer_mass"] < 0.1)
    assert np.all(u_shape["chi2_per_bin"] > 10 * flat["chi2_per_bin"])
    d = ge.rms_distances([real, narrow], fluid)
    floor = ge.split_half_floor(
        lambda src, x, b: ge.energy_distance(d, x + (64 if src else 0), b),
        64,
        num_splits=8,
        n_source=64,
    )
    assert floor["matched"]["relative"] > 3 * abs(floor["floor"]["relative"])


def test_mean_bias_recovers_a_known_shift() -> None:
    rng = np.random.default_rng(3)
    fluid = _fluid()
    rm = ge.ensemble_moments(_gaussian_ensemble(rng, 512), fluid)
    shifted = ge.ensemble_moments(_gaussian_ensemble(rng, 512, shift=0.3), fluid)
    bias = ge.mean_bias(shifted, rm)
    assert np.allclose(bias["debiased"], 0.09, atol=0.01)
    # Standardised by the mean real variance; per level by that level's.
    var_r = rm["var"].mean(axis=1)
    assert np.allclose(bias["standardised"], bias["debiased"] / var_r)
    assert bias["profile"].shape == (3, _SHAPE[0])
    # A paired source: S_i = R_i + 0.3 exactly, so the difference has no noise.
    diff_var = np.zeros_like(rm["var"])
    same = dict(rm, mean=rm["mean"] + 0.3)
    paired = ge.mean_bias(same, rm, diff_var=diff_var)
    assert np.allclose(paired["debiased"], 0.09)


def test_fair_crps_matches_its_definition() -> None:
    # Two members {0, 1} and truth 0: mean |x - y| = 0.5, pair term 2 / 4.
    ens = np.array([0.0, 1.0]).reshape(2, 1, 1)
    assert ge.fair_crps(ens, np.zeros((1, 1, 1)))[0] == pytest.approx(0.0)
    assert ge.fair_crps(ens, np.full((1, 1, 1), 2.0))[0] == pytest.approx(1.0)
    # Leave-one-out equals scoring each truth against the others explicitly.
    rng = np.random.default_rng(4)
    x = rng.standard_normal((7, 2, 5))
    loo = ge.fair_crps(x, x, exclude_self=True)
    explicit = np.mean(
        [ge.fair_crps(np.delete(x, r, axis=0), x[r : r + 1]) for r in range(7)],
        axis=0,
    )
    assert np.allclose(loo, explicit)
    with pytest.raises(ValueError, match="exclude_self"):
        ge.fair_crps(x, x[:3], exclude_self=True)


def test_rms_distances_and_paired_energy_distance() -> None:
    rng = np.random.default_rng(5)
    fluid = _fluid()
    a = rng.standard_normal((3, 3, *_SHAPE)) + 10.0  # a large mean
    b = rng.standard_normal((2, 3, *_SHAPE)) + 10.0
    d = ge.rms_distances([a, b], fluid, chunk_cells=7)
    stack = np.concatenate([a, b])[:, :, fluid].reshape(5, -1)
    brute = np.sqrt(((stack[:, None] - stack[None]) ** 2).mean(axis=2))
    assert np.allclose(d, brute) and np.all(np.diag(d) == 0)
    # Paired: the i = j cross pairs (a source derived from its own truth)
    # are dropped; identical sets then have ED exactly 0.
    same = ge.rms_distances([a, a], fluid)
    x, y = np.arange(3), np.arange(3, 6)
    assert ge.energy_distance(same, x, y, paired=True)["energy_distance"] == (
        pytest.approx(0.0, abs=1e-9)
    )
    assert ge.energy_distance(same, x, y)["energy_distance"] < 0  # i = j pairs at 0
    # Distinct paired sets: the cross mean is over the i != j pairs only.
    near = ge.rms_distances([a, a + 0.1 * rng.standard_normal(a.shape)], fluid)
    cross = near[np.ix_(x, y)]
    off = ~np.eye(3, dtype=bool)
    brute_ed = (
        2 * cross[off].mean()
        - near[np.ix_(x, x)][off].mean()
        - near[np.ix_(y, y)][off].mean()
    )
    assert ge.energy_distance(near, x, y, paired=True)["energy_distance"] == (
        pytest.approx(brute_ed)
    )


def test_split_half_floor_bootstrap_points_and_w1() -> None:
    calls: list[tuple[bool, list[int], list[int]]] = []

    def metric(src: bool, x: np.ndarray, b: np.ndarray) -> dict[str, Any]:
        calls.append((src, list(x), list(b)))
        return {"size": float(len(x)), "pair": np.array([1.0, 2.0])}

    out = ge.split_half_floor(metric, 9, num_splits=3, seed=0, n_source=6)
    assert out["floor"]["size"] == 4 and out["matched"]["size"] == 4
    assert np.allclose(out["floor"]["pair"], [1, 2]) and out["floor_std"]["size"] == 0
    for src, x, b in calls:
        assert len(b) == 4 and (src or not set(x) & set(b))
    calls.clear()
    ge.split_half_floor(metric, 8, num_splits=2, n_source=8, paired=True)
    assert calls[0][1] == calls[1][1]  # the paired source subset is half a
    # The splits do not depend on the source: paired and unpaired runs share them.
    calls.clear()
    ge.split_half_floor(metric, 8, num_splits=3, seed=1, n_source=8, paired=True)
    paired_splits = [(x, b) for src, x, b in calls if not src]
    calls.clear()
    ge.split_half_floor(metric, 8, num_splits=3, seed=1, n_source=8)
    assert [(x, b) for src, x, b in calls if not src] == paired_splits
    ci = ge.bootstrap_ci([1.0, 2.0, 3.0, np.nan], seed=0)
    assert (
        ci["mean"] == 2.0 and ci["n"] == 3 and 1.0 <= ci["lo"] < 2.0 < ci["hi"] <= 3.0
    )
    assert np.isnan(ge.bootstrap_ci([5.0])["lo"])
    fluid = _fluid()
    points = ge.select_points(fluid, 20, seed=7)
    assert np.array_equal(points, ge.select_points(fluid, 20, seed=7))
    assert len(points) == 20 and fluid.ravel()[points].all()
    assert len(ge.select_points(fluid, 10**6)) == fluid.sum()
    rng = np.random.default_rng(6)
    a, b = rng.standard_normal((5, 2, 3)), rng.standard_normal((5, 2, 3)) + 1.0
    expected = [
        np.mean([ge.wasserstein_1(a[:, c, p], b[:, c, p]) for p in range(3)])
        for c in range(2)
    ]
    assert np.allclose(ge.pointwise_w1(a, b), expected)
    assert np.allclose(ge.pointwise_w1(a, b[:4]).shape, (2,))
