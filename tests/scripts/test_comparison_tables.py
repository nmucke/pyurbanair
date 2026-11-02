"""scripts/surrogate_comparison/make_tables.py: the bolding and formatting rules,
and the three tables from a small hand-made metrics.nc + metrics.yaml."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import xarray
import yaml

from tests.conftest import load_script

tables = load_script("scripts/surrogate_comparison/make_tables.py")
MODELS = ["a", "b", "c"]


def column(values: dict, best: str | float | None) -> object:
    return tables.Column("", "x", "x", values, best)


def test_best_rows() -> None:
    nan = math.nan
    # min, ties at the printed precision all bold, never persistence
    errors = column({"a": 0.1201, "b": 0.1199, "c": 0.2, "persistence": 0.01}, "min")
    assert tables.best_rows(errors, MODELS, 3) == {"a", "b"}
    assert tables.best_rows(errors, MODELS, 4) == {"b"}
    # max; NaN never bold
    horizon = column({"a": 3.0, "b": 5.0, "c": nan}, "max")
    assert tables.best_rows(horizon, MODELS, 2) == {"b"}
    # closest to a target, by absolute distance (dB: both signs tie)
    db = column({"a": -4.1, "b": 4.1, "c": 6.0}, 0.0)
    assert tables.best_rows(db, MODELS, 2) == {"a", "b"}
    assert tables.best_rows(column({"a": 0.8, "b": 1.1}, 0.9), MODELS, 3) == {"a"}
    # nothing finite, or no rule: no bold
    assert tables.best_rows(column({"a": nan, "b": None}, "min"), MODELS, 2) == set()
    assert tables.best_rows(column({"a": 1.0}, None), MODELS, 2) == set()


def test_decimals_and_escape() -> None:
    assert tables.decimals([0.081, 0.12, 0.2]) == 3
    assert tables.decimals([2.5, 3.1, math.nan]) == 2
    assert tables.decimals([14062.0, 37500.0]) == 0
    assert tables.decimals([math.nan]) == 2
    assert tables.tex_escape(r"dft_s & 50% #1 \x") == (
        r"dft\_s \& 50\% \#1 \textbackslash{}x"
    )


def test_tables(tmp_path: pathlib.Path) -> None:
    lead = np.arange(1, 11)
    sources = ["dft_s", "unstable", "persistence", "truth"]
    rmse = np.stack([0.1 * lead, 0.2 * lead, 0.3 * lead, np.full(10, np.nan)])
    rmse[1, 5:] = np.nan  # blows up after lead 5
    xarray.Dataset(
        {"rmse": (("source", "lead"), rmse), "acc": (("source", "lead"), 1 - rmse)},
        coords={
            "source": sources,
            "lead": lead,
            "lead_over_T": ("lead", lead / 2.0),  # T-bar = 2 steps of 1 s
        },
        attrs={"T_mean": 2.0, "H_mean": 10.0},
    ).to_netcdf(tmp_path / "metrics.nc")

    def entry(value: object) -> dict:
        return {"mean": value, "per_sample": {0: value}}

    scalars = {
        "dft_s": {
            "acc_horizon_seconds": entry(10.0),  # censored samples: the window end
            "acc_horizon_censored": entry(0.5),
            "stable": entry(True),
            "integral_length": entry(0.8),
            "spectra_db_u_mid": entry(-1.5),
            "seconds_per_step": 0.012,
            "n_parameters": 41_000_000,
        },
        "unstable": {
            "acc_horizon_seconds": entry(math.nan),
            "acc_horizon_censored": entry(0.0),
            "stable": entry(False),
            "integral_length": entry(math.nan),
            "spectra_db_u_mid": entry(math.nan),
            "seconds_per_step": 0.002,
            "n_parameters": 1_000_000,
        },
        "persistence": {
            "acc_horizon_seconds": entry(4.0),
            "acc_horizon_censored": entry(0.0),
            "stable": entry(True),
            "integral_length": entry(0.9),
        },
        "truth": {"integral_length": entry(1.0)},
        "truth_halves": {"spectra_db_u_mid": entry(0.2)},
    }
    (tmp_path / "metrics.yaml").write_text(yaml.safe_dump(scalars, sort_keys=False))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"metrics": {"acc_threshold": 0.5, "horizons": [1, 2, 50]}})
    )
    tables.run(tmp_path, ["display_names={dft_s: Ours}"])

    accuracy = (tmp_path / "table_accuracy.tex").read_text()
    assert r"\label{tab:comparison_accuracy}" in accuracy
    # horizons 1 and 2 T-bar (50 is beyond the 5 T-bar window), then the window
    assert accuracy.count(r"\bar{T}$ &") >= 2 and "window" in accuracy
    ours = next(line for line in accuracy.splitlines() if line.startswith("Ours"))
    # RMS of 0.1 * lead over leads 1, 2 (t <= 1 T-bar): sqrt(0.025); a censored
    # horizon (10 s / T-bar 2 s, one sample never drops): a bold lower bound
    assert r"\textbf{\boldmath 0.158}" in ours
    assert r"\textbf{\boldmath $\geq$5.00}" in ours
    unstable = next(
        line for line in accuracy.splitlines() if line.startswith("unstable")
    )
    assert unstable.count("--") == 3  # the window means and the horizon
    assert "persistence" not in accuracy and "Persistence" in accuracy

    physics = (tmp_path / "table_physics.tex").read_text()
    rows = [line for line in physics.splitlines() if line.endswith(r"\\")]
    assert any(r.startswith("Truth (LES)") and "1.000" in r for r in rows)
    assert not any(r.startswith("truth") or "halves" in r for r in rows)
    # Lambda/H closest to the truth among the models; negative numbers in math
    assert r"\textbf{\boldmath $-$1.50}" in physics

    cost = (tmp_path / "table_cost.tex").read_text()
    assert r"\textbf{\boldmath 2.00}" in cost and "Speedup" not in cost
    with open(tmp_path / "table_cost.csv") as f:
        assert f.readline().strip() == "source,n_parameters,ms_per_step"
