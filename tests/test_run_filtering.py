"""End-to-end smoke tests for scripts/filtering/run_filtering.py (sequential EnKF).

Mirrors tests/test_run_esmda.py: everything runs under the tiny smoke config
(conftest ``_SMOKE_OVERRIDES``) with the global (unlocalized) update — the
default correlation localization is degenerate at this 2-member ensemble size.
One test per filter mode (state / parameter / joint), plus a multi-cycle run
and distance localization (purely geometric, so meaningful at 2 members).
"""

import pathlib
from typing import Any, Optional

import pytest


def _overrides(
    mode: str, num_cycles: int, extra: Optional[list[str]] = None
) -> list[str]:
    return [
        # The barcelona default's STL has no solid cells on the tiny smoke
        # domain (the same environment limitation that blocks test_run_esmda);
        # the xie_and_castro cube array voxelizes fine there.
        "case=xie_and_castro",
        "model@truth_model=pylbm",
        "model@assim_model=pylbm",
        f"filtering.mode={mode}",
        f"filtering.num_cycles={num_cycles}",
        # The global (unlocalized) update; correlation localization is
        # degenerate at 2 members.
        "filtering/localization=none",
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=2",
        "truth_model.forward_model.cuda=false",
        "assim_model.forward_model.cuda=false",
        # The conftest smoke overrides pin a tiny [0,20]^2 domain but do not
        # supply matching sensor coordinates; place the assimilation sensors in
        # the open N-S lanes of that domain (same points as test_run_esmda).
        "obs.x_points=[2.5,2.5,18.0,18.0]",
        "obs.y_points=[5.0,15.0,5.0,15.0]",
        "obs.z_points=[3.0,3.0,3.0,3.0]",
        "obs.interval_seconds=3.0",
        *(extra or []),
    ]


@pytest.mark.parametrize(  # type: ignore[misc]
    "mode,num_cycles,extra",
    [
        pytest.param("joint", 1, None, id="joint"),
        pytest.param("joint", 2, None, id="joint_two_cycles"),
        pytest.param("state", 1, None, id="state"),
        # Parameter mode needs spread maintenance; the random-walk evolution
        # exercises the parameter forecast model too.
        pytest.param(
            "parameter", 2, ["filtering/evolution=random_walk"], id="parameter"
        ),
    ],
)
def test_run_filtering(
    mode: str, num_cycles: int, extra: Optional[list[str]], compose_test_cfg: Any
) -> None:
    import xarray

    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(
        _overrides(mode, num_cycles, extra), config_name="run_filtering"
    )
    run(cfg)

    out_dir = pathlib.Path(cfg.paths.results_dir)
    assert (out_dir / "posterior_state.nc").exists()
    assert (out_dir / "cycle_diagnostics.yaml").exists()
    assert (out_dir / "run_info.yaml").exists()
    posterior_state = xarray.open_dataset(out_dir / "posterior_state.nc")
    assert posterior_state.sizes["ensemble"] == 2
    if mode != "state":
        posterior = xarray.open_dataset(out_dir / "posterior_params.nc")
        assert "inflow_angle" in posterior.data_vars
        # History: prior + one entry per cycle.
        history = xarray.open_dataset(out_dir / "params_history.nc")
        assert history.sizes["cycle"] == num_cycles + 1


def test_run_filtering_distance_localization(compose_test_cfg: Any) -> None:
    """State rows localized by physical distance (params stay global)."""
    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(
        _overrides(
            "joint",
            1,
            [
                "filtering/localization=distance",
                "filtering.localization.localization_radius=10.0",
            ],
        ),
        config_name="run_filtering",
    )
    run(cfg)


def test_run_filtering_rejects_zero_cycles(compose_test_cfg: Any) -> None:
    """num_cycles < 1 fails before any simulation is started."""
    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(_overrides("joint", 0), config_name="run_filtering")
    with pytest.raises(ValueError, match="num_cycles"):
        run(cfg)


def test_run_filtering_rejects_dynamic_prior(compose_test_cfg: Any) -> None:
    """The filter is static-parameters-only; a dynamic prior fails loudly."""
    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(
        _overrides(
            "parameter",
            1,
            [
                "params@prior_params=dynamic",
                "params@truth_params=dynamic_truth",
                "filtering/evolution=random_walk",
            ],
        ),
        config_name="run_filtering",
    )
    with pytest.raises(ValueError, match="time-varying"):
        run(cfg)
