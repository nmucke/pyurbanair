"""End-to-end smoke tests for scripts/filter_smoothing/run_filter_smoothing.py.

Mirrors tests/test_run_filtering.py: everything runs under the tiny smoke config
(conftest ``_SMOKE_OVERRIDES``) with the global (unlocalized) inner update — the
correlation localization is degenerate at this 2-member ensemble size. One
solver-running test covering the whole windowed loop (2 cycles x 2 ESMDA
iterations plus the final consistency pass), plus compose-only and
guard tests that need no solver.

Like the other e2e suites, these share the pylbm build tree and ``.temp``, so
pytest sessions must not run concurrently (auto-memory: serial e2e sessions).
"""

import pathlib
from typing import Any, Optional

import pytest


def _overrides(
    num_cycles: int, num_steps: int, extra: Optional[list[str]] = None
) -> list[str]:
    return [
        # The barcelona default's STL has no solid cells on the tiny smoke
        # domain; the xie_and_castro cube array voxelizes fine there.
        "case=xie_and_castro",
        "model@truth_model=pylbm",
        "model@assim_model=pylbm",
        # This entry point is the inverse of run_filtering's guard: the PRIOR
        # must be a time-varying (dynamic) sampler, since the estimated object
        # is a parameter *trajectory* over the window. Both are the config's
        # defaults; pinned here so a retuned default cannot silently change what
        # this test exercises.
        "params@truth_params=dynamic_sine",
        "params@prior_params=dynamic",
        f"filter_smoothing.num_cycles={num_cycles}",
        f"filter_smoothing.num_steps={num_steps}",
        # Global inner update and no temporal taper: the smoke run pins
        # plumbing, not skill.
        "filter_smoothing/inner_localization=none",
        "filter_smoothing/temporal_localization=none",
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=2",
        "truth_model.forward_model.cuda=false",
        "assim_model.forward_model.cuda=false",
        # The conftest smoke overrides pin a tiny [0,20]^2 domain but no matching
        # sensor coordinates; place the sensors in the open N-S lanes of that
        # domain (same points as test_run_filtering / test_run_esmda).
        "obs.x_points=[2.5,2.5,18.0,18.0]",
        "obs.y_points=[5.0,15.0,5.0,15.0]",
        "obs.z_points=[3.0,3.0,3.0,3.0]",
        "obs.interval_seconds=3.0",
        # One knot backs exactly one cycle, and the script enforces it. The
        # entry point ties seconds_per_knot to simulation_time by interpolation,
        # but conftest's _SMOKE_OVERRIDES pins it to 1.5 (half a cycle), so it
        # has to be restored here.
        "time.seconds_per_knot=3.0",
        *(extra or []),
    ]


@pytest.mark.parametrize(  # type: ignore[misc]
    "option,localized",
    [("none", False), ("taper", True)],
)
def test_temporal_localization_config_composes(
    option: str, localized: bool, compose_test_cfg: Any
) -> None:
    """Both temporal-localization options mount into the composed target."""
    cfg = compose_test_cfg(
        _overrides(2, 2, [f"filter_smoothing/temporal_localization={option}"]),
        config_name="run_filter_smoothing",
    )
    temporal = cfg.filter_smoothing.smoother.temporal_localization
    if not localized:
        assert temporal is None
    else:
        assert temporal._target_ == (
            "data_assimilation.filter_smoothing.TemporalLocalization"
        )
        # The taper's three knobs plus the block-grouping policy are the whole
        # of Eq. (17)'s configuration surface.
        #
        # The radius is pinned to its literal value on purpose: it is measured
        # in CYCLES (FilterSmoothingESMDA builds the knot/observation time
        # coordinates as cycle indices), so a seconds-scaled number here — 900
        # for a 300 s cycle, say — would exceed every in-window lag and make
        # `temporal_localization=taper` a silently GLOBAL update, which no
        # other assertion in this suite would notice.
        assert temporal.temporal_radius == pytest.approx(3.0)
        assert 0.0 < temporal.tapering_beta < 1.0
        assert temporal.max_inflation >= 1.0
        # A group_ids block is all knots of ONE parameter, and a block shares a
        # single observation selection — so grouping would erase the taper
        # along the very axis this strategy localizes.
        assert temporal.block_grouping is False


@pytest.mark.parametrize(  # type: ignore[misc]
    "option,target,localization",
    [
        (
            "stochastic",
            "data_assimilation.filtering.analysis.StochasticEnKFAnalysis",
            "none",
        ),
        ("etkf", "data_assimilation.filtering.etkf.ETKFAnalysis", "none"),
        ("letkf", "data_assimilation.filtering.etkf.LETKFAnalysis", "distance"),
    ],
)
def test_inner_analysis_config_composes(
    option: str, target: str, localization: str, compose_test_cfg: Any
) -> None:
    """The inner analysis group reuses the filtering schemes unchanged.

    Each option is composed with the inner localization its
    ``localization_policy`` allows (``etkf*`` forbids one, ``letkf*`` requires
    one), so the composed config is one the inner filter would accept.
    """
    cfg = compose_test_cfg(
        _overrides(
            2,
            2,
            [
                f"filter_smoothing/inner_analysis={option}",
                f"filter_smoothing/inner_localization={localization}",
            ],
        ),
        config_name="run_filter_smoothing",
    )
    smoother = cfg.filter_smoothing.smoother
    assert smoother.inner_analysis._target_ == target
    if localization == "none":
        assert smoother.inner_localization is None
    else:
        assert smoother.inner_localization is not None


def test_filter_smoothing_target_and_alpha_default_compose(
    compose_test_cfg: Any,
) -> None:
    """The entry point instantiates the windowed smoother with alpha unset."""
    cfg = compose_test_cfg(_overrides(2, 2), config_name="run_filter_smoothing")

    assert cfg.filter_smoothing.smoother._target_ == (
        "data_assimilation.filter_smoothing.FilterSmoothingESMDA"
    )
    # ``alpha: null`` -> the equal-weight schedule (alpha = num_steps) is chosen
    # by the class, never pinned as a config literal that could drift out of
    # step with num_steps.
    assert cfg.filter_smoothing.smoother.alpha is None
    assert int(cfg.filter_smoothing.smoother.num_steps) == 2
    assert cfg.filter_smoothing.smoother.common_inner_noise is True


class _StubEnsembleModel:
    """The whole ensemble-model surface the estimator touches at construction.

    ``BaseFilter.__init__`` reads ``save_on_disk`` (and ``results_dir`` when it
    is True) and nothing else, so this is enough to build the inner filter
    without a solver.
    """

    save_on_disk = False
    results_dir = None


def _stub_observation_operator(state: Any) -> Any:  # pragma: no cover - never called
    raise AssertionError("construction must not call the observation operator")


@pytest.mark.parametrize(  # type: ignore[misc]
    "extra,expected_alpha",
    [
        pytest.param([], 2.0, id="default"),
        pytest.param(["filter_smoothing/temporal_localization=taper"], 2.0, id="taper"),
        pytest.param(
            ["filter_smoothing.num_steps=4", "filter_smoothing.alpha=4.0"],
            4.0,
            id="alpha_override",
        ),
        pytest.param(
            [
                "filter_smoothing/inner_analysis=etkf",
                "filter_smoothing/inner_localization=none",
            ],
            2.0,
            id="etkf",
        ),
        pytest.param(
            [
                "filter_smoothing/inner_analysis=letkf",
                "filter_smoothing/inner_localization=distance",
            ],
            2.0,
            id="letkf_distance",
        ),
        pytest.param(["filter_smoothing/inner_inflation=rtpp"], 2.0, id="rtpp"),
    ],
)
def test_composed_smoother_actually_constructs(
    extra: list[str], expected_alpha: float, compose_test_cfg: Any
) -> None:
    """Instantiate the composed node, not just compose it.

    ``compose()`` alone proves the YAML resolves; it cannot see a constructor
    that rejects the combination (the inner analysis/localization policy pairs
    are validated in ``BaseFilter.__init__``) nor a knob whose value is
    out of contract. This runs the option grid through the real class with a
    stub forward model — still solver-free, so it stays in the fast set.
    """
    import jax.numpy as jnp
    from hydra.utils import instantiate

    cfg = compose_test_cfg(_overrides(2, 2, extra), config_name="run_filter_smoothing")
    smoother = instantiate(
        cfg.filter_smoothing.smoother,
        observation_operator=_stub_observation_operator,
        forward_model=_StubEnsembleModel(),
        C_D=jnp.array([0.1]),
    )

    assert type(smoother).__name__ == "FilterSmoothingESMDA"
    # ``alpha: null`` resolves to num_steps (=2 here) inside the class; an
    # explicit scalar wins. Either way the resolution happens at construction,
    # not at run time, so it is observable without a forecast.
    assert float(smoother.alpha) == pytest.approx(expected_alpha)


def test_shipped_defaults_construct(compose_test_cfg: Any) -> None:
    """The entry point's OWN ``num_steps``/``alpha`` pair must be admissible.

    Deliberately overrides nothing (beyond conftest's smoke shape, which
    touches neither key): every other test in this file sets both
    ``num_steps`` and ``alpha``, so none of them ever reads the shipped
    values. Editing ``conf/run_filter_smoothing.yaml`` to ``alpha: 3`` while
    leaving ``num_steps: 4`` would make every real run of this entry point die
    at construction on the ES-MDA schedule check while the rest of the suite
    stayed green.

    Asserted against the config's own ``num_steps`` rather than a literal, so
    retuning the shipped default is free and only an *inconsistent* pair fails.
    """
    import jax.numpy as jnp
    from hydra.utils import instantiate

    cfg = compose_test_cfg([], config_name="run_filter_smoothing")
    smoother = instantiate(
        cfg.filter_smoothing.smoother,
        observation_operator=_stub_observation_operator,
        forward_model=_StubEnsembleModel(),
        C_D=jnp.array([0.1]),
    )

    assert float(smoother.alpha) == pytest.approx(float(cfg.filter_smoothing.num_steps))


def test_composed_smoother_rejects_an_inconsistent_alpha(
    compose_test_cfg: Any,
) -> None:
    """``sum_a 1/alpha_a = 1`` is enforced at construction, not silently tempered.

    A scalar ``alpha`` other than ``num_steps`` tempers the likelihood by
    ``num_steps / alpha`` — the same guard ``_BaseESMDA`` applies. Only an
    instantiation can see it, which is why the compose-only tests above are not
    sufficient coverage of the ``alpha`` knob.
    """
    import jax.numpy as jnp
    from hydra.errors import InstantiationException
    from hydra.utils import instantiate

    cfg = compose_test_cfg(
        _overrides(2, 2, ["filter_smoothing.alpha=4.0"]),
        config_name="run_filter_smoothing",
    )
    with pytest.raises(InstantiationException, match="Inconsistent ES-MDA schedule"):
        instantiate(
            cfg.filter_smoothing.smoother,
            observation_operator=_stub_observation_operator,
            forward_model=_StubEnsembleModel(),
            C_D=jnp.array([0.1]),
        )


def test_run_filter_smoothing(compose_test_cfg: Any) -> None:
    """One full window: 2 cycles x 2 ESMDA iterations + the final pass.

    Asserts the artifact set, the shape of the trajectory posterior and the
    per-iteration / per-cycle diagnostics counts. At the smoke ensemble size (2
    members) nothing about assimilation skill can be read off the result; the
    unit tests in tests/test_filter_smoothing.py pin the algorithm's semantics.
    """
    import xarray

    from scripts.esmda._esmda_common import read_yaml
    from scripts.filter_smoothing.run_filter_smoothing import run

    num_cycles, num_steps = 2, 2
    cfg = compose_test_cfg(
        _overrides(num_cycles, num_steps), config_name="run_filter_smoothing"
    )
    run(cfg)

    out_dir = pathlib.Path(cfg.paths.results_dir)
    for artifact in (
        "posterior_params.nc",
        "posterior_state.nc",
        "prior_params.nc",
        "true_params.nc",
        "iteration_diagnostics.yaml",
        "cycle_diagnostics.yaml",
        "truth_access.yaml",
        "run_info.yaml",
    ):
        assert (out_dir / artifact).exists(), f"missing artifact: {artifact}"

    # The posterior is a parameter TRAJECTORY: one knot per cycle plus the
    # trailing knot (the leading edge of a future moving window), which the
    # sampler emits for a num_cycles * simulation_time horizon.
    posterior_params = xarray.open_dataset(out_dir / "posterior_params.nc")
    assert "time" in posterior_params["inflow_angle"].dims
    assert posterior_params.sizes["ensemble"] == 2
    assert posterior_params.sizes["time"] == num_cycles + 1
    prior_params = xarray.open_dataset(out_dir / "prior_params.nc")
    assert posterior_params.sizes["time"] == prior_params.sizes["time"]
    posterior_state = xarray.open_dataset(out_dir / "posterior_state.nc")
    assert posterior_state.sizes["ensemble"] == 2

    # One diagnostics record per ESMDA iteration; the cycle diagnostics come
    # from the final consistency pass, so there is one per cycle.
    assert len(read_yaml(out_dir / "iteration_diagnostics.yaml")) == num_steps
    assert len(read_yaml(out_dir / "cycle_diagnostics.yaml")) == num_cycles

    configuration = read_yaml(out_dir / "run_info.yaml")["configuration"]
    assert configuration["num_cycles"] == num_cycles
    assert configuration["num_steps"] == num_steps
    # alpha unset -> the equal-weight schedule the class resolved.
    assert configuration["alpha_effective"] == pytest.approx(float(num_steps))
    assert configuration["num_trajectory_knots"] == num_cycles + 1
    assert configuration["seconds_per_knot"] == pytest.approx(
        float(cfg.time.simulation_time)
    )


def test_run_filter_smoothing_rejects_a_static_prior(compose_test_cfg: Any) -> None:
    """The inverse of run_filtering's guard: a static prior has no trajectory."""
    from scripts.filter_smoothing.run_filter_smoothing import run

    cfg = compose_test_cfg(
        _overrides(2, 2, ["params@prior_params=static"]),
        config_name="run_filter_smoothing",
    )
    with pytest.raises(ValueError, match="static"):
        run(cfg)


def test_run_filter_smoothing_rejects_a_knot_spacing_that_is_not_the_cycle(
    compose_test_cfg: Any,
) -> None:
    """Knot k must back cycle k's segment, so the spacing IS the cycle length.

    Fails before any simulation starts: a half-cycle knot spacing would
    silently re-interpret which parameter drove which forecast.
    """
    from scripts.filter_smoothing.run_filter_smoothing import run

    cfg = compose_test_cfg(
        _overrides(2, 2, ["time.seconds_per_knot=1.5"]),
        config_name="run_filter_smoothing",
    )
    with pytest.raises(ValueError, match="seconds_per_knot"):
        run(cfg)


@pytest.mark.parametrize("scalar", ["num_cycles", "num_steps"])  # type: ignore[misc]
def test_run_filter_smoothing_rejects_degenerate_counts(
    scalar: str, compose_test_cfg: Any
) -> None:
    """A zero window length or zero ESMDA iterations fails up front."""
    from scripts.filter_smoothing.run_filter_smoothing import run

    cfg = compose_test_cfg(
        _overrides(2, 2, [f"filter_smoothing.{scalar}=0"]),
        config_name="run_filter_smoothing",
    )
    with pytest.raises(ValueError, match=scalar):
        run(cfg)
