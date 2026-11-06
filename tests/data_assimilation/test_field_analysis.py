"""Stage 2 field analysis: Eqs. (11) and (12) of the multi-scale note."""

import pathlib

import numpy as np
import pytest
import xarray
from data_assimilation.io import concat_member_files
from data_assimilation.localization.base import gaspari_cohn
from data_assimilation.smoothing.field_analysis import (
    field_analysis,
    field_analysis_netcdf,
    global_field_weights,
    within_parameter_basis,
)

J = 24


def _problem(seed=0, num_params=3, frames=6, sensors=4, rows=40):
    """Random window: rows of ``frames`` frames, one observation per sensor."""
    rng = np.random.default_rng(seed)
    params = rng.normal(size=(num_params, J))
    sensor_xyz = rng.uniform(0.0, 100.0, (sensors, 3))
    times = np.arange(frames, dtype=float)
    obs_time = np.repeat(times, sensors)
    obs_coords = np.tile(sensor_xyz, (frames, 1))
    n_o = obs_time.size
    pred = rng.normal(size=(n_o, J)) + rng.normal(size=(n_o, num_params)) @ params
    return {
        "states": rng.normal(size=(frames, rows, J)) + 3.0,
        "times": times,
        "row_coords": rng.uniform(0.0, 100.0, (rows, 3)),
        "kwargs": {
            "pred_obs": pred,
            "obs": rng.normal(size=n_o),
            "obs_var": rng.uniform(0.5, 2.0, n_o),
            "obs_time": obs_time,
            "obs_coords": obs_coords,
            "params": params,
        },
    }


def _run(problem, states=None, **options):
    """Run in memory; returns the updated frames and the info."""
    states = np.array(problem["states"] if states is None else states)

    def write(index, values):
        states[index] = values

    info = field_analysis(
        ((t, f.copy()) for t, f in zip(problem["times"], states)),
        write,
        **problem["kwargs"],
        **options,
    )
    return states, info


def _dense_reference(pred, obs, obs_var, params):
    """Direct transcription in observation space, ``(N_o, N_o)`` solve."""
    anomalies = params - params.mean(axis=1, keepdims=True)
    spanned = np.c_[np.ones(J), anomalies.T]
    projector = np.eye(J) - spanned @ np.linalg.pinv(spanned)
    nu = J - 1 - np.linalg.matrix_rank(anomalies)
    centred = pred - pred.mean(axis=1, keepdims=True)
    within = centred @ projector
    system = within @ within.T / nu + np.diag(obs_var)
    gain = projector @ centred.T @ np.linalg.inv(system) / nu
    return projector, nu, centred, within, gain


def _plain_weights(problem):
    k = problem["kwargs"]
    return global_field_weights(k["pred_obs"], k["obs"], k["obs_var"], k["params"])


def test_mean_and_covariance_match_the_perturbed_form_of_eq_11():
    k = _problem()["kwargs"]
    pred, obs, var = k["pred_obs"], k["obs"], k["obs_var"]
    projector, nu, _, within, gain = _dense_reference(pred, obs, var, k["params"])
    weights = _plain_weights(_problem())

    # With E = 0 the note's corrected W has the same member means as ours.
    perturbed = np.eye(J) + gain @ (obs[:, None] - pred)
    np.testing.assert_allclose(weights.mean(axis=1), perturbed.mean(axis=1), atol=1e-9)

    # Member-wise conditional mean: the between-member part is conditioned too.
    between = np.eye(J) - projector - np.ones((J, J)) / J
    np.testing.assert_allclose(weights @ between, perturbed @ between, atol=1e-9)

    # Kalman covariance of the within-parameter anomalies, in ensemble space.
    kalman = (projector - gain @ within) / nu
    centred_weights = weights @ projector
    np.testing.assert_allclose(
        centred_weights @ centred_weights.T / nu, kalman, atol=1e-9
    )


def test_perturbed_form_of_eq_11_converges_to_the_deterministic_spread():
    problem = _problem(frames=2, sensors=3)
    k = problem["kwargs"]
    pred, obs, var = k["pred_obs"], k["obs"], k["obs_var"]
    projector, nu, _, _, gain = _dense_reference(pred, obs, var, k["params"])
    rng = np.random.default_rng(1)
    total = np.zeros((J, J))
    draws = 20000
    for _ in range(draws):
        noise = rng.normal(size=pred.shape) * np.sqrt(var)[:, None]
        noise = noise @ projector  # E lives in the within-parameter space too
        drawn = (np.eye(J) + gain @ (obs[:, None] + noise - pred)) @ projector
        total += drawn @ drawn.T / nu
    ours = _plain_weights(problem) @ projector
    np.testing.assert_allclose(total / draws, ours @ ours.T / nu, atol=2e-2)


def test_eq_11_without_parameters_is_the_note_equation():
    problem = _problem()
    k = problem["kwargs"]
    constant = np.ones((2, J))
    pred, obs, var = k["pred_obs"], k["obs"], k["obs_var"]
    centred = pred - pred.mean(axis=1, keepdims=True)
    system = centred @ centred.T / (J - 1) + np.diag(var)
    gain = centred.T @ np.linalg.inv(system) / (J - 1)
    weights = global_field_weights(pred, obs, var, constant)
    np.testing.assert_allclose(
        weights.mean(axis=1),
        (np.eye(J) + gain @ (obs[:, None] - pred)).mean(axis=1),
        atol=1e-9,
    )


def test_columns_sum_to_one():
    weights = _plain_weights(_problem())
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-10)


def test_null_radius_is_eq_11_exactly():
    problem = _problem()
    updated, info = _run(problem, L_t=1.0)
    expected = problem["states"] @ _plain_weights(problem)
    np.testing.assert_allclose(updated, expected, atol=1e-10)
    assert info.num_updated_frames == info.num_frames == 6
    assert info.updated_row_fraction == 1.0


def test_huge_radii_reach_eq_11_through_the_local_path():
    problem = _problem()
    far = 1e9
    local, _ = _run(problem, L_r=far, L_t=far, row_coords=problem["row_coords"])
    np.testing.assert_allclose(local, _run(problem)[0], atol=1e-6)


def test_global_w_keeps_affine_constraints():
    problem = _problem()
    rng = np.random.default_rng(3)
    constraint = rng.normal(size=40)
    # Project every member onto constraint . x = 7 (a discrete divergence).
    states = problem["states"]
    excess = (np.einsum("r,trj->tj", constraint, states) - 7.0) / (
        constraint @ constraint
    )
    states = states - excess[:, None, :] * constraint[None, :, None]
    updated, _ = _run(problem, states)
    np.testing.assert_allclose(
        np.einsum("r,trj->tj", constraint, updated), 7.0, atol=1e-9
    )
    assert np.abs(updated - states).max() > 1e-3


def test_local_w_keeps_zero_in_buildings_and_far_cells_bit_identical():
    problem = _problem()
    states = problem["states"].astype(np.float32)
    states[:, :5] = 0.0
    sensors = problem["kwargs"]["obs_coords"][:4]
    coords = problem["row_coords"].copy()
    coords[:5] = sensors[0] + 0.5  # a building next to a sensor
    coords[5:10] = 1e4  # far from every sensor
    updated, info = _run(problem, states, L_r=40.0, L_t=2.0, row_coords=coords)
    assert updated.dtype == np.float32
    assert np.array_equal(updated[:, :5], states[:, :5])
    assert np.array_equal(updated[:, 5:10], states[:, 5:10])
    assert np.abs(updated - states).max() > 1e-3
    assert 0.0 < info.updated_row_fraction < 1.0


def test_frames_without_a_local_observation_are_not_written():
    problem = _problem()
    k = dict(problem["kwargs"], obs_time=np.zeros(24))
    written = []
    info = field_analysis(
        ((t, f.copy()) for t, f in zip(problem["times"], problem["states"])),
        lambda index, values: written.append(index),
        **{**k, "row_coords": problem["row_coords"], "L_r": 1e9, "L_t": 2.5},
    )
    assert written == [0, 1, 2]
    assert info.num_frames == 6 and info.num_updated_frames == 3


def test_a_sensor_silent_at_a_frame_leaves_its_cells_bit_identical():
    problem = _problem(sensors=2, frames=2)
    k = problem["kwargs"]
    coords = problem["row_coords"].copy()
    coords[:] = k["obs_coords"][0]  # every row at sensor 0
    k["obs_coords"][:, 0] = np.where(np.arange(4) % 2, 1e5, k["obs_coords"][:, 0])
    k["obs_time"] = np.array([5.0, 0.0, 5.0, 1.0])  # sensor 0 only at t = 5
    updated, info = _run(problem, L_r=10.0, L_t=0.5, row_coords=coords)
    assert np.array_equal(updated, problem["states"])
    assert info.num_updated_frames == 0


@pytest.mark.parametrize("localized", [False, True])
def test_chunk_size_invariance(localized):
    problem = _problem()
    options = (
        {"L_r": 60.0, "L_t": 2.0, "row_coords": problem["row_coords"]}
        if localized
        else {}
    )
    reference, _ = _run(problem, chunk_size=4096, **options)
    for chunk_size in (1, 7, 16):
        updated, _ = _run(problem, chunk_size=chunk_size, **options)
        np.testing.assert_allclose(updated, reference, rtol=0, atol=1e-11)


@pytest.mark.parametrize("localized", [False, True])
def test_masked_members_leave_the_statistics_and_stay_unchanged(localized):
    problem = _problem()
    options = (
        {"L_r": 60.0, "L_t": 2.0, "row_coords": problem["row_coords"]}
        if localized
        else {}
    )
    mask = np.ones(J, dtype=bool)
    mask[[2, 9]] = False
    states = problem["states"].copy()
    states[:, :, 2] = np.nan
    problem["kwargs"]["pred_obs"][:, 9] = np.nan
    updated, info = _run(problem, states, member_mask=mask, **options)
    assert info.num_members == J - 2
    assert np.array_equal(updated[..., ~mask], states[..., ~mask], equal_nan=True)

    # The same as an analysis of the other members alone.
    alone = {
        **problem,
        "states": problem["states"][..., mask],
        "kwargs": {
            **problem["kwargs"],
            "pred_obs": problem["kwargs"]["pred_obs"][:, mask],
            "params": problem["kwargs"]["params"][:, mask],
        },
    }
    np.testing.assert_allclose(
        updated[..., mask], _run(alone, **options)[0], atol=1e-10
    )


def test_linear_gaussian_toy_gives_the_exact_conditional_moments():
    """x = a theta + eps, y = H x + noise: the moments of x | theta, y.

    The sample is built with the exact moments (eps orthogonal to [1, theta],
    sample covariance C), so the comparison has no sampling error.
    """
    rng = np.random.default_rng(7)
    members, n_x, n_o = 64, 3, 2
    a = np.array([1.0, -0.5, 2.0])
    root = np.array([[1.0, 0, 0], [0.6, 0.8, 0], [0.2, -0.3, 0.9]])
    covariance = root @ root.T
    H = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R = np.array([0.5, 0.2])
    y = np.array([1.0, -2.0])

    theta = rng.normal(size=members)
    theta -= theta.mean()
    nu = members - 2
    basis = within_parameter_basis(theta[None, :])
    assert basis.shape[1] == nu
    white, _ = np.linalg.qr(basis @ rng.normal(size=(nu, n_x)))
    x = a[:, None] * theta + np.sqrt(nu) * root @ white.T
    gain = covariance @ H.T @ np.linalg.inv(H @ covariance @ H.T + np.diag(R))
    updated = x.copy()

    def write(index, values):
        updated[:] = values

    field_analysis(
        [(0.0, x.copy())],
        write,
        pred_obs=H @ x,
        obs=y,
        obs_var=R,
        obs_time=np.zeros(n_o),
        obs_coords=np.zeros((n_o, 3)),
        params=theta[None, :],
    )

    # Regression on theta: intercept K y, slope (I - K H) a, residual covariance
    # (I - K H) C, exactly what conditioning each member on y gives.
    design = np.c_[np.ones(members), theta]
    coefficients, *_ = np.linalg.lstsq(design, updated.T, rcond=None)
    residual = updated.T - design @ coefficients
    np.testing.assert_allclose(coefficients[0], gain @ y, atol=1e-10)
    np.testing.assert_allclose(coefficients[1], a - gain @ (H @ a), atol=1e-10)
    posterior = (np.eye(n_x) - gain @ H) @ covariance
    np.testing.assert_allclose(residual.T @ residual / nu, posterior, atol=1e-10)


def test_basis_drops_constant_rows_and_uses_the_numerical_rank():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(2, J))
    params = np.vstack([theta, 3.0 * theta[0] - theta[1], np.full((1, J), 2.5)])
    basis = within_parameter_basis(params)
    assert basis.shape == (J, J - 3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(J - 3), atol=1e-12)
    np.testing.assert_allclose(basis.T @ np.ones(J), 0.0, atol=1e-12)
    np.testing.assert_allclose(basis.T @ theta.T, 0.0, atol=1e-12)


def test_too_few_degrees_of_freedom_are_refused():
    problem = _problem(num_params=J - 3)
    with pytest.raises(ValueError, match="degrees of freedom"):
        _run(problem)


@pytest.mark.parametrize(
    "options",
    [
        {"L_r": 5.0},
        {"L_r": 5.0, "L_t": 1.0},
        {"L_r": -1.0, "L_t": 1.0},
        {"chunk_size": 0},
    ],
)
def test_incomplete_localization_is_refused(options):
    problem = _problem()
    if options.get("L_r", 0) < 0:
        options = {**options, "row_coords": problem["row_coords"]}
    with pytest.raises(ValueError):
        _run(problem, **options)


def test_gaspari_cohn_shape():
    distance = np.array([0.0, 0.5, 1.0, 1.999, 2.0, 3.0])
    taper = np.asarray(gaspari_cohn(distance, 2.0))
    assert taper[0] == 1.0 and taper[4] == 0.0 and taper[5] == 0.0
    assert np.isclose(taper[2], 0.2083333, atol=1e-5)  # 5/24 at the half-width
    assert np.all(np.diff(taper) <= 0.0) and taper[3] >= 0.0


def _staggered_members(frames=3, members=J):
    """Per-member uDALES-like Datasets: u and v on different grids."""
    rng = np.random.default_rng(5)
    coords = {
        "time": np.arange(frames, dtype=float),
        "zt": np.array([1.0, 3.0]),
        "yt": np.array([1.0, 3.0, 5.0]),
        "xm": np.array([0.0, 2.0, 4.0, 6.0]),
        "xt": np.array([1.0, 3.0, 5.0, 7.0]),
    }
    dims_u, dims_v = ("time", "zt", "yt", "xm"), ("time", "zt", "yt", "xt")
    return [
        xarray.Dataset(
            {
                "u": (dims_u, rng.normal(size=(frames, 2, 3, 4)).astype("f4")),
                "v": (dims_v, rng.normal(size=(frames, 2, 3, 4)).astype("f4")),
            },
            coords=coords,
        )
        for _ in range(members)
    ]


@pytest.mark.parametrize("mode", ["disk", "memory"])
@pytest.mark.parametrize("localized", [False, True])
def test_netcdf_file_in_both_save_modes(tmp_path: pathlib.Path, mode, localized):
    members = _staggered_members()
    path = tmp_path / "window_0_posterior_state.nc"
    if mode == "disk":
        files = []
        for i, member in enumerate(members):
            files.append(tmp_path / f"state_{i}.nc")
            member.to_netcdf(files[-1])
        concat_member_files(files, path)
    else:
        ensemble = xarray.concat(members, dim="ensemble", join="override")
        ensemble.to_netcdf(path)  # (ensemble, time, ...), as save_state does

    with xarray.open_dataset(path) as stacked:
        before = stacked.load()
    rng = np.random.default_rng(6)
    sensor = np.array([[3.0, 3.0, 1.0], [1.0, 1.0, 3.0]])
    times = before["time"].values
    n_o = 2 * times.size
    options = {"L_r": 3.0, "L_t": 0.9} if localized else {}
    kwargs = dict(
        pred_obs=rng.normal(size=(n_o, J)),
        obs=rng.normal(size=n_o),
        obs_var=np.full(n_o, 0.5),
        obs_time=np.repeat(times, 2),
        obs_coords=np.tile(sensor, (times.size, 1)),
        params=rng.normal(size=(2, J)),
    )
    info = field_analysis_netcdf(path, variables=["u", "v"], **kwargs, **options)
    assert info.num_frames == times.size
    with xarray.open_dataset(path) as stacked:
        after = stacked.load()

    # The same result as the in-memory stream, with u then v rows.
    u = before["u"].transpose("time", "zt", "yt", "xm", "ensemble").values
    v = before["v"].transpose("time", "zt", "yt", "xt", "ensemble").values
    states = np.concatenate([u.reshape(3, -1, J), v.reshape(3, -1, J)], axis=1)
    grid = [
        np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)[:, ::-1]
        for axes in (
            (before.zt.values, before.yt.values, before.xm.values),
            (before.zt.values, before.yt.values, before.xt.values),
        )
    ]
    problem = {"states": states, "times": times, "kwargs": kwargs}
    row_coords = {"row_coords": np.concatenate(grid)} if localized else {}
    expected, _ = _run(problem, **options, **row_coords)
    u_after = after["u"].transpose("time", "zt", "yt", "xm", "ensemble").values
    v_after = after["v"].transpose("time", "zt", "yt", "xt", "ensemble").values
    np.testing.assert_allclose(u_after.reshape(3, -1, J), expected[:, :24], atol=1e-5)
    np.testing.assert_allclose(v_after.reshape(3, -1, J), expected[:, 24:], atol=1e-5)
    assert not np.array_equal(after["u"].values, before["u"].values)


def test_netcdf_refuses_an_unknown_field(tmp_path: pathlib.Path):
    path = tmp_path / "state.nc"
    xarray.concat(_staggered_members(), dim="ensemble", join="override").to_netcdf(path)
    with pytest.raises(ValueError, match="absent"):
        field_analysis_netcdf(path, variables=["pres"])
