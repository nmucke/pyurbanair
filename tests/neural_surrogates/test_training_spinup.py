"""Tests for :mod:`neural_surrogates.training_spinup`.

The training-data warm start (used by ``scripts/esmda/run_esmda.py`` when the neural
surrogate is the assimilation model with ``spinup_source: training_data``) loads
the LAST frame of each training trajectory as the window-0 initial state and
anchors each member's sampled prior to that sample's final inflow value.
"""

import numpy as np
import pytest
import xarray as xr
from neural_surrogates.training_spinup import (
    anchor_prior_params,
    list_split_samples,
    member_sample_index,
    write_initial_state_files,
)

NZ, NY, NX = 3, 4, 4


def _write_training_data(root, n_samples, t_len=5):
    """Minimal ``training_data/`` split; sample s has distinct last-frame values.

    ``inflow_angle`` ramps to ``100 + s`` and ``velocity_magnitude`` to ``5 + s``
    at the final time step, so a test can tell which sample seeded a member from
    the anchored value. ``sgs_constant`` is a static (time-less) param.
    """
    state_dir = root / "state" / "train"
    param_dir = root / "param" / "train"
    state_dir.mkdir(parents=True)
    param_dir.mkdir(parents=True)
    for s in range(n_samples):
        time = np.arange(t_len, dtype=float)
        # The frame value equals s + time, so the last frame is s + (t_len - 1).
        state = xr.Dataset(
            {
                "u": (
                    ("time", "zt", "yt", "xt"),
                    np.broadcast_to(
                        (s + time)[:, None, None, None], (t_len, NZ, NY, NX)
                    ).copy(),
                )
            },
            coords={"time": time},
        )
        state.to_netcdf(state_dir / f"sample_{s:04d}.nc")

        param = xr.Dataset(
            {
                "inflow_angle": ("time", np.linspace(0.0, 100.0 + s, t_len)),
                "velocity_magnitude": ("time", np.linspace(0.0, 5.0 + s, t_len)),
                "sgs_constant": 0.22,  # static: must be left untouched
            },
            coords={"time": time},
        )
        param.to_netcdf(param_dir / f"sample_{s:04d}.nc")
    return root


def _prior(ensemble_size, t_len=4):
    """A small AR(2)-style prior ensemble: (time, ensemble) inflow params."""
    time = np.arange(t_len, dtype=float)
    # Distinct per-member shape so the constant shift is observable.
    angle = np.stack([10.0 + i + 0.5 * time for i in range(ensemble_size)], axis=1)
    vel = np.stack([3.0 + 0.1 * i + 0.2 * time for i in range(ensemble_size)], axis=1)
    return xr.Dataset(
        {
            "inflow_angle": (("time", "ensemble"), angle),
            "velocity_magnitude": (("time", "ensemble"), vel),
            "sgs_constant": ("ensemble", np.full(ensemble_size, 0.22)),
        },
        coords={"time": time, "ensemble": np.arange(ensemble_size)},
    )


def test_member_sample_index_cycles():
    assert member_sample_index(0, 2) == (0, 0)
    assert member_sample_index(1, 2) == (1, 0)
    assert member_sample_index(2, 2) == (0, 1)
    assert member_sample_index(3, 2) == (1, 1)


def test_list_split_samples_mismatch_raises(tmp_path):
    root = _write_training_data(tmp_path / "td", n_samples=2)
    (root / "param" / "train" / "sample_0001.nc").unlink()
    with pytest.raises(ValueError, match="sample count mismatch"):
        list_split_samples(root, "train")


def test_write_initial_state_files_uses_last_frame_and_cycles(tmp_path):
    root = _write_training_data(tmp_path / "td", n_samples=2, t_len=5)
    state_files, _ = list_split_samples(root, "train")
    out = write_initial_state_files(state_files, n_members=3, out_dir=tmp_path / "ic")

    # One file per member, named for the per-member path warm start.
    for i in range(3):
        assert (out / f"state_{i}.nc").exists()
    # Member i is seeded from sample (i % 2)'s LAST frame: value = sample + 4.
    for i, expected_sample in enumerate([0, 1, 0]):
        with xr.open_dataset(out / f"state_{i}.nc") as ds:
            assert "time" not in ds.dims  # only the single last frame was kept
            assert float(ds["u"].max()) == pytest.approx(expected_sample + 4.0)


def test_anchor_prior_params_pins_last_value_preserving_shape(tmp_path):
    root = _write_training_data(tmp_path / "td", n_samples=2)
    _, param_files = list_split_samples(root, "train")
    prior = _prior(ensemble_size=2)

    anchored = anchor_prior_params(prior, param_files, n_members=2, frame=-1)

    # Member 0 -> sample 0 last value (100); member 1 -> sample 1 last value (101).
    assert float(anchored["inflow_angle"].isel(ensemble=0, time=0)) == pytest.approx(
        100.0
    )
    assert float(anchored["inflow_angle"].isel(ensemble=1, time=0)) == pytest.approx(
        101.0
    )
    assert float(
        anchored["velocity_magnitude"].isel(ensemble=0, time=0)
    ) == pytest.approx(5.0)
    # The per-member AR draw shape (deviation from t=0) is preserved by the shift.
    for v in ("inflow_angle", "velocity_magnitude"):
        a = anchored[v].transpose("time", "ensemble").values
        p = prior[v].transpose("time", "ensemble").values
        np.testing.assert_allclose(a - a[0:1], p - p[0:1])
    # Static params are untouched.
    np.testing.assert_allclose(
        anchored["sgs_constant"].values, prior["sgs_constant"].values
    )


def test_anchor_prior_params_jitters_reused_samples(tmp_path):
    root = _write_training_data(tmp_path / "td", n_samples=2)
    _, param_files = list_split_samples(root, "train")
    prior = _prior(ensemble_size=3)

    anchored = anchor_prior_params(
        prior, param_files, n_members=3, initial_param_jitter_scale=0.05
    )
    # Member 0 (first pass) gets the exact last value; member 2 reuses sample 0
    # with jitter, so it differs but stays close.
    assert float(anchored["inflow_angle"].isel(ensemble=0, time=0)) == pytest.approx(
        100.0
    )
    pinned2 = float(anchored["inflow_angle"].isel(ensemble=2, time=0))
    assert pinned2 != pytest.approx(100.0)
    assert abs(pinned2 - 100.0) < 100.0 * 0.3
