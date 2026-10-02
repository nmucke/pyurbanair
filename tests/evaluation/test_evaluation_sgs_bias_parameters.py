"""Evaluation support for the three persistent SGS discrepancy coefficients."""

from __future__ import annotations

import numpy as np
import xarray as xr
from evaluation.figures import _marginal_members, _param_axis_label
from evaluation.scores import _PLOTTED_PARAMS, compute_parameter_metrics
from evaluation.style import PARAM_LABELS, PARAM_UNITS

SGS_BIAS_NAMES = ("sgs_bias_b0", "sgs_bias_b1", "sgs_bias_b2")


def _members(**values: list[float]) -> xr.Dataset:
    return xr.Dataset(
        {
            name: ("ensemble", np.asarray(samples, dtype=float))
            for name, samples in values.items()
        }
    )


def test_sgs_bias_coefficients_are_scored_and_plotted_in_declared_order() -> None:
    posterior = _members(
        sgs_bias_b0=[0.0, 0.2],
        sgs_bias_b1=[-0.3, -0.1],
        sgs_bias_b2=[0.1, 0.3],
    )
    truth = xr.Dataset(
        {
            name: xr.DataArray(value)
            for name, value in zip(SGS_BIAS_NAMES, (0.1, -0.2, 0.2))
        }
    )

    metrics = compute_parameter_metrics(posterior, truth)
    entries = list(_marginal_members(posterior, truth, None))

    assert _PLOTTED_PARAMS[-3:] == SGS_BIAS_NAMES
    assert tuple(metrics) == SGS_BIAS_NAMES
    assert tuple(entry[0] for entry in entries) == SGS_BIAS_NAMES
    assert all(np.isfinite(metrics[name]["rmse"]).all() for name in SGS_BIAS_NAMES)


def test_prior_only_sgs_bias_names_do_not_create_missing_truth() -> None:
    posterior = _members(
        sgs_bias_b0=[0.0, 0.2],
        sgs_bias_b1=[-0.3, -0.1],
        sgs_bias_b2=[0.1, 0.3],
    )
    prior = _members(
        sgs_bias_b0=[-0.1, 0.3],
        sgs_bias_b1=[-0.4, 0.0],
        sgs_bias_b2=[0.0, 0.4],
    )
    truth = xr.Dataset({"sgs_bias_b0": xr.DataArray(0.1)})

    metrics = compute_parameter_metrics(posterior, truth, prior)
    entries = list(_marginal_members(posterior, truth, prior))

    assert tuple(metrics) == ("sgs_bias_b0",)
    assert tuple(entry[0] for entry in entries) == ("sgs_bias_b0",)


def test_sgs_bias_coefficient_plot_labels_and_units_are_defined() -> None:
    for index, name in enumerate(SGS_BIAS_NAMES):
        assert PARAM_LABELS[name] == rf"SGS discrepancy coefficient $b_{index}$ [1]"
        assert PARAM_UNITS[name] == "1"
        assert _param_axis_label(name) == PARAM_LABELS[name]
