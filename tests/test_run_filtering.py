"""End-to-end smoke tests for scripts/filtering/run_filtering.py (sequential EnKF).

Mirrors tests/test_run_esmda.py: everything runs under the tiny smoke config
(conftest ``_SMOKE_OVERRIDES``) with the global (unlocalized) update — the
default correlation localization is degenerate at this 2-member ensemble size.
One test per filter mode (state / parameter / joint), plus a multi-cycle run,
distance localization (purely geometric, so meaningful at 2 members), and the
mixed drift-tracking path (a dynamic truth tracked by a static prior).
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
        # Pin a static truth so these stay deterministic static-parameter smoke
        # tests independent of the config's default truth sampler (which is a
        # dynamic drift-tracking truth); the mixed dynamic-truth path has its own
        # test below.
        "params@truth_params=static_truth",
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
    "option,target",
    [
        ("none", None),
        ("svd_current", "data_assimilation.reduction.OnlineStateReduction"),
        ("svd_streaming", "data_assimilation.reduction.StreamingStateReduction"),
    ],
)
def test_filtering_state_reduction_config_composes(
    option: str, target: Optional[str], compose_test_cfg: Any
) -> None:
    """Every reduction-group option mounts into the filter target."""
    cfg = compose_test_cfg(
        _overrides("state", 1, [f"filtering/state_reduction={option}"]),
        config_name="run_filtering",
    )
    reduction = cfg.filtering.filter.state_reduction
    if target is None:
        assert reduction is None
    else:
        assert reduction._target_ == target
    if option == "svd_streaming":
        # The exponentially weighted accumulator keeps absorbing directions, so
        # an energy criterion alone would let the basis grow every cycle.
        assert reduction.max_rank is not None


@pytest.mark.parametrize(  # type: ignore[misc]
    "mode,num_cycles,extra",
    [
        pytest.param("joint", 1, None, id="joint"),
        pytest.param("joint", 2, None, id="joint_two_cycles"),
        pytest.param("state", 1, None, id="state"),
        pytest.param(
            "state",
            1,
            ["filtering/state_reduction=svd_current"],
            id="state_svd_current",
        ),
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
    from scripts.esmda._esmda_common import read_yaml

    configuration = read_yaml(out_dir / "run_info.yaml")["configuration"]
    diagnostics = read_yaml(out_dir / "cycle_diagnostics.yaml")
    reduction_enabled = bool(extra) and any(
        override.startswith("filtering/state_reduction=")
        and not override.endswith("=none")
        for override in (extra or [])
    )
    if not reduction_enabled:
        # The unreduced default must still write a well-formed run_info and a
        # complete (all-None) reduction block in the diagnostics schema.
        assert configuration["state_reduction"] is None
        assert configuration["state_reduction_resolved_variable_scales"] is None
        assert diagnostics[0]["reduction_rank"] is None
        assert diagnostics[0]["reduction_basis_time"] is None
        assert diagnostics[0]["obs_posterior_rmse_kind"] == "exact"
        assert diagnostics[0]["analysis_time"] is not None
    else:
        reduction = configuration["state_reduction"]
        assert reduction["_target_"] == (
            "data_assimilation.reduction.OnlineStateReduction"
        )
        assert reduction["whiten"] is False
        resolved_scales = configuration["state_reduction_resolved_variable_scales"]
        assert resolved_scales
        assert set(resolved_scales.values()) == {1.0}

        # Two members leave exactly one statistical mode.
        assert diagnostics[0]["reduction_rank"] == 1
        assert diagnostics[0]["reduction_available_rank"] == 1
        assert diagnostics[0]["analysis_time"] is not None
        assert diagnostics[0]["reduction_basis_time"] is not None
        assert diagnostics[0]["reduction_basis_updated"] is True
        assert diagnostics[0]["reduction_discarded_increment_fraction"] is not None
        # The observation-space posterior rides along on the full-space update,
        # so it must be labelled rather than compared with an unreduced run.
        assert diagnostics[0]["obs_posterior_rmse_kind"] == "unreduced_ride_along"
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


def test_run_filtering_tracks_dynamic_truth(compose_test_cfg: Any) -> None:
    """Mixed mode: a time-varying (dynamic) TRUTH tracked by a static prior.

    The filter's scalar estimate tracks a drifting truth. The dynamic truth is
    sampled over the full num_cycles horizon (so it varies across cycles), and
    the static prior is analysed/evolved each cycle. Exercises the truth-sampling
    horizon override and the parameter-block update against a time-varying truth.
    """
    import xarray

    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(
        _overrides(
            "joint",
            2,
            [
                # Override the pinned static truth back to the dynamic one; the
                # prior stays static (the default), which is the mixed setup.
                "params@truth_params=dynamic_truth",
                "filtering/evolution=random_walk",
            ],
        ),
        config_name="run_filtering",
    )
    run(cfg)

    out_dir = pathlib.Path(cfg.paths.results_dir)
    # The truth params are time-varying and span the full horizon (num_cycles *
    # simulation_time), so the truth drifts across every cycle.
    true_params = xarray.open_dataset(out_dir / "true_params.nc")
    assert "time" in true_params["inflow_angle"].dims
    assert (
        float(true_params["time"].max()) >= 2 * float(cfg.time.simulation_time) - 1e-6
    )
    # The prior/posterior stay static scalars the filter tracks.
    posterior = xarray.open_dataset(out_dir / "posterior_params.nc")
    assert "time" not in posterior["inflow_angle"].dims


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
