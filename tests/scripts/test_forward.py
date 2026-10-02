"""Forward runs (run_forward.py) and their figures (visualize_forward.py)."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest
import xarray

from tests.conftest import compose, load_script, surrogate

FIGURES = ("field_snapshot.png", "parameters.png")


def _run(cfg: Any) -> pathlib.Path:
    load_script("scripts/run_forward.py").run(cfg)
    run_dir = pathlib.Path(cfg.paths.results_dir)
    load_script("scripts/visualize_forward.py").run(run_dir)
    return run_dir


def test_single_member(
    tmp_path: pathlib.Path, session_root: pathlib.Path, trained: Any
) -> None:
    cfg = compose(
        "forward",
        "+test=forward",
        "model=neural_surrogate_tiny",
        *surrogate(session_root),
        root=tmp_path,
    )
    run_dir = _run(cfg)
    state = xarray.load_dataset(run_dir / "state.nc")
    assert "ensemble" not in state.dims
    assert {"u", "v", "w"} <= set(state.data_vars)
    assert np.isfinite(state.u.values).all()
    assert not (run_dir / "windows").exists()  # forward.save_windows is off
    for name in FIGURES:
        assert (run_dir / "figures" / name).exists(), name
    # MP4 with ffmpeg, else the animation helper's GIF fallback.
    assert any(
        (run_dir / "figures" / f"animation.{ext}").exists() for ext in ("mp4", "gif")
    )


def test_ensemble_over_several_windows(
    tmp_path: pathlib.Path, session_root: pathlib.Path, trained: Any
) -> None:
    cfg = compose(
        "forward",
        "+test=forward",
        "model=neural_surrogate_tiny",
        *surrogate(session_root),
        "forward.ensemble=true",
        "forward.rollout_steps=1",
        "forward.save_windows=true",
        root=tmp_path,
    )
    run_dir = _run(cfg)
    state = xarray.load_dataset(run_dir / "state.nc")
    params = xarray.load_dataset(run_dir / "params.nc")
    assert state.sizes["ensemble"] == 2
    # Two 3 s windows joined on one strictly increasing time axis.
    times = state.time.values
    assert np.all(np.diff(times) > 0) and times[-1] == pytest.approx(6.0)
    assert params.time.values[-1] == pytest.approx(6.0)
    # Each window on its own, on the same global clock.
    last = xarray.load_dataset(run_dir / "windows" / "state_0001.nc")
    assert (run_dir / "windows" / "params_0000.nc").exists()
    xarray.testing.assert_equal(last, state.sel(time=last.time))


def test_initial_state_selects_member_and_frame(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "state.nc"
    xarray.Dataset(
        {"u": (("ensemble", "time"), np.arange(6.0).reshape(2, 3))},
        coords={"ensemble": [3, 7], "time": [0.0, 1.0, 2.0]},
    ).to_netcdf(source)
    initial_state = load_script("scripts/run_forward.py")._initial_state
    # A path alone: the last frame of the first member.
    assert float(initial_state(str(source), False).u) == 2.0
    picked = initial_state({"path": str(source), "member": 7, "time_index": 0}, False)
    assert float(picked.u) == 3.0
    ensemble = initial_state({"path": str(source), "time_index": 1}, True)
    assert ensemble.u.values.tolist() == [1.0, 4.0]


@pytest.mark.integration  # type: ignore[misc]
@pytest.mark.parametrize("model", ["pyudales_tiny", "pylbm_tiny"])  # type: ignore[misc]
def test_solver(tmp_path: pathlib.Path, model: str) -> None:
    cfg = compose(
        "forward",
        "+test=forward",
        f"model={model}",
        "forward.ensemble=true",
        root=tmp_path,
    )
    run_dir = _run(cfg)
    state = xarray.load_dataset(run_dir / "state.nc")
    assert state.sizes["ensemble"] == 2
    assert np.isfinite(state.u.values).all()
    assert (run_dir / "figures" / "parameters.png").exists()
