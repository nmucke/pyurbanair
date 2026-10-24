"""configs/ composes, resolves and passes the consistency check, and the
test overlays make it tiny."""

from __future__ import annotations

import pathlib
import re
import warnings
from typing import Any

import numpy as np
import pytest
import xarray
from omegaconf import OmegaConf

from tests.conftest import CONFIGS, REPO, TEST_CONFIGS, compose, load_script

ENTRY_POINTS = [
    "forward",
    "assimilation",
    "comparison",
    "model_comparison",
    *(f"surrogate/{p.stem}" for p in sorted((CONFIGS / "surrogate").glob("*.yaml"))),
]
ENTRY_POINTS.remove("surrogate/architectures")  # entries, not a run config
ENTRY_POINTS.remove("surrogate/training")  # shared defaults, not a run config
# Every option of the groups forward.yaml picks from, incl. the tiny test backends.
OPTIONS = {
    group: sorted(
        p.stem for d in (CONFIGS, TEST_CONFIGS) for p in (d / group).glob("*.yaml")
    )
    for group in ("model", "params", "case")
}


@pytest.mark.parametrize("name", ENTRY_POINTS)  # type: ignore[misc]
def test_entry_point_resolves(name: str, tmp_path: pathlib.Path) -> None:
    OmegaConf.to_container(compose(name, root=tmp_path), resolve=True)


@pytest.mark.parametrize("group", OPTIONS)  # type: ignore[misc]
def test_every_option_resolves(group: str, tmp_path: pathlib.Path) -> None:
    for option in OPTIONS[group]:
        cfg = compose("forward", f"{group}={option}", root=tmp_path)
        OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize(  # type: ignore[misc]
    "name", ["forward", "assimilation", "surrogate/generate_data"]
)
def test_test_overlay_is_tiny(name: str, tmp_path: pathlib.Path) -> None:
    overlay = "+test=" + name.split("/")[-1]
    cfg = compose(name, overlay, root=tmp_path)
    assert (cfg.domain.nx, cfg.domain.ny, cfg.domain.nz) == (20, 20, 6)
    assert cfg.ensemble.ensemble_size == 2


@pytest.mark.parametrize("case", OPTIONS["case"])  # type: ignore[misc]
def test_case_geometry_paths_exist(case: str, tmp_path: pathlib.Path) -> None:
    geometry = compose("forward", f"case={case}", root=tmp_path).geometry
    case_dir = REPO / geometry.case_dir
    assert (case_dir / "namoptions.300").is_file()
    assert (case_dir / "_p3d").is_file()
    assert (REPO / geometry.stl_path).is_file()
    # uDALES reads the STL that namoptions names from its case folder.
    name = re.escape(pathlib.Path(geometry.stl_path).name)
    namoptions = (case_dir / "namoptions.300").read_text()
    assert re.search(rf"^\s*stl_file\s*=\s*{name}\s*$", namoptions, re.M)
    if geometry.get("udales_precomputed_geom_dir"):
        assert (
            REPO / geometry.udales_precomputed_geom_dir / "geom_meta.json"
        ).is_file()


@pytest.mark.parametrize(  # type: ignore[misc]
    "case", ["xie_and_castro", "xie_and_castro_periodic"]
)
def test_case_sensors_are_clear_of_the_buildings(
    case: str, tmp_path: pathlib.Path
) -> None:
    """Every sensor, assimilated or held out, reads only fluid cells that do not
    touch a wall or roof (the comparison's `fluid_mask` of the case STL), at the
    case grid and at 1 m; the held-out ones stay 2 case cells (4 m) from the
    assimilated ones."""
    from evaluation.turbulence import fluid_mask

    cfg = compose("forward", f"case={case}", root=tmp_path)
    obs = cfg.obs
    sensors = {
        "assimilation": np.array([obs.x_points, obs.y_points, obs.z_points]).T,
        "validation": np.array(
            [obs.validation_x_points, obs.validation_y_points, obs.validation_z_points]
        ).T,
    }
    bounds = np.array(cfg.domain.bounds, dtype=float)
    case_cells = [cfg.domain.nx, cfg.domain.ny, cfg.domain.nz]
    for cells in (case_cells, (bounds[:, 1] - bounds[:, 0]).astype(int)):
        x, y, z = (
            lo + (np.arange(n) + 0.5) * (hi - lo) / n
            for n, (lo, hi) in zip(cells, bounds)
        )
        fluid = fluid_mask(REPO / cfg.geometry.stl_path, z, y, x)
        for name, points in sensors.items():
            for point in points:
                # The cell centres a trilinear interpolation at the point reads:
                # the one it sits on, or the two that bracket it, per axis.
                stencil = [
                    np.flatnonzero(np.abs(c - p) < c[1] - c[0] - 1e-9)
                    for c, p in zip((z, y, x), point[::-1])
                ]
                assert all(len(s) for s in stencil), (name, point, "outside")
                assert fluid[np.ix_(*stencil)].all(), (name, point, cells)
    distance = np.linalg.norm(
        sensors["validation"][:, None] - sensors["assimilation"][None], axis=-1
    )
    assert distance.min() >= 4.0


@pytest.mark.parametrize("source", ["idealized", "realistic"])  # type: ignore[misc]
def test_random_geometry_templates_exist(source: str, tmp_path: pathlib.Path) -> None:
    cfg = compose(
        "surrogate/generate_data", f"data.geometry.source={source}", root=tmp_path
    )
    case_dir = REPO / cfg.data.geometry.case_dir
    assert (case_dir / "namoptions.300").is_file()
    assert (case_dir / "_p3d").is_file()


# Workflow -> overrides that make the production assimilation config valid for it.
# filtering.mode=state needs filtering.parameter_evolution=null; set it here so the
# test doesn't depend on the value in configs/assimilation.yaml.
VALID: dict[str, list[str]] = {
    "smoother": [],
    "filtering": ["params@prior_params=static", "filtering.parameter_evolution=null"],
    "hybrid": ["filtering.parameter_evolution=null"],
}


@pytest.mark.parametrize("workflow", ["forward", *VALID])  # type: ignore[misc]
def test_check_config_accepts_production(workflow: str, tmp_path: pathlib.Path) -> None:
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    name = "forward" if workflow == "forward" else "assimilation"
    check(compose(name, *VALID.get(workflow, []), root=tmp_path), workflow)


@pytest.mark.parametrize(  # type: ignore[misc]
    "workflow, overrides, message",
    [
        ("filtering", [], "static prior_params"),
        ("smoother", ["params@prior_params=static"], "smoothing.smoother is"),
        ("smoother", ["assimilation.params_to_estimate=[nope]"], "does not define"),
        ("smoother", ["assimilation.save_states=every"], "save_states must be"),
        (
            "smoother",
            [
                "domain.nx=30",
                "flow.outflow_buffer=null",
                "assim_model.forward_model.ncpu=7",
            ],
            "must divide nx",
        ),
        (
            # 50 points over 100 m plus a 20 m buffer: PALM runs on 60.
            "smoother",
            [
                "model@assim_model=pypalm",
                "flow.boundary_condition=inflow_outflow",
                "domain.nx=50",
                "domain.bounds=[[-20.0,80.0],[0.0,80.0],[0.0,32.0]]",
                "flow.outflow_buffer=20.0",
                "assim_model.forward_model.ncpu=25",
            ],
            "must divide nx=60",
        ),
        (
            "smoother",
            [
                "model@assim_model=pypalm",
                "assim_model.forward_model.boundary_condition=periodic",
                "domain.nx=31",
            ],
            "even nx and ny",
        ),
    ],
)
def test_check_config_rejects(
    workflow: str, overrides: list[str], message: str, tmp_path: pathlib.Path
) -> None:
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    with pytest.raises(ValueError, match=message):
        check(compose("assimilation", *overrides, root=tmp_path), workflow)


# A valid smoother run estimating the SGS-discrepancy coefficients; set
# explicitly so the test doesn't depend on configs/assimilation.yaml's tuning.
DISCREPANCY = [
    "assim_model.forward_model.model_discrepancy.enabled=true",
    "params@truth_params=dynamic_sine",
    "params@prior_params=dynamic",
    "smoothing.smoother=${smoother.dynamic}",
    "smoothing.localization=null",
    "smoothing.state_reduction=null",
    "smoothing.final_time_smoothing=false",
    "assimilation.params_to_estimate=null",
    "filtering.mode=state",
    "filtering.parameter_evolution=null",
    "ensemble.failure.policy=raise",
]
STATIC_COEFFICIENT_FILTER = [
    "params@prior_params=static",
    "filtering.mode=parameter",
    "assimilation.params_to_estimate=[sgs_bias_b0]",
]


def test_check_config_accepts_discrepancy(tmp_path: pathlib.Path) -> None:
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    check(compose("assimilation", *DISCREPANCY, root=tmp_path), "smoother")


@pytest.mark.parametrize(  # type: ignore[misc]
    "workflow, overrides, message",
    [
        ("smoother", ["assim_model.name=pylbm"], "needs pyudales with vreman"),
        (
            "smoother",
            ["assim_model.forward_model.closure=smagorinsky"],
            "needs pyudales with vreman",
        ),
        (
            "smoother",
            [
                "+prior_params.external_parameters.sgs_bias_b0="
                "{_target_:pyurbanair.static_parameters.Normal,mean:0.0,std:0.1}"
            ],
            "must be static parameters",
        ),
        (
            "smoother",
            ["smoothing.smoother=${smoother.state_and_dynamic}"],
            "smoother must be TimeVaryingParameterESMDA",
        ),
        ("hybrid", ["filtering.mode=joint"], "needs filtering.mode=state"),
        (
            "hybrid",
            ["ensemble.failure.policy=resample_from_successes"],
            "needs ensemble.failure.policy=raise",
        ),
        (
            "filtering",
            [
                *STATIC_COEFFICIENT_FILTER,
                "filtering.parameter_evolution={_target_:data_assimilation."
                "filtering.parameter_evolution.RandomWalkEvolution,"
                "std:{sgs_bias_b0:0.1}}",
            ],
            "parameter_evolution=null",
        ),
        (
            "filtering",
            [
                *STATIC_COEFFICIENT_FILTER,
                "filtering.localization=${localization.distance}",
            ],
            "cannot use distance localization",
        ),
    ],
)
def test_check_config_rejects_discrepancy(
    workflow: str, overrides: list[str], message: str, tmp_path: pathlib.Path
) -> None:
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    cfg = compose("assimilation", *DISCREPANCY, *overrides, root=tmp_path)
    with pytest.raises(ValueError, match=message):
        check(cfg, workflow)


# A localized state smoother whose update holds N_aug x N_d^2 float32 values:
# N_aug = 3 * 20 * 20 * 6 = 7200 rows, N_d = 4 sensors * 2 states * 10 bins = 80,
# so 7200 * 80**2 * 4 bytes = 184.32 MB.
LOCALIZED_STATE_SMOOTHER = [
    "+test=assimilation",
    "assim_model.forward_model.model_discrepancy.enabled=false",
    "assimilation.params_to_estimate=null",
    "params@truth_params=dynamic_sine",
    "params@prior_params=dynamic",
    "smoothing.smoother=${smoother.state_and_dynamic}",
    "smoothing.localization=${localization.correlation}",
    "smoothing.state_reduction=null",
    "domain.nx=20",
    "domain.ny=20",
    "domain.nz=6",
    "obs.x_points=[1.0,2.0,3.0,4.0]",
    "obs.states=[u,v]",
    "time.simulation_time=10.0",
    "observation.aggregation.interval_seconds=1.0",
]


@pytest.mark.parametrize(  # type: ignore[misc]
    "memory, outcome", [(100e6, "fails"), (300e6, "warns"), (1e12, "passes")]
)
def test_check_config_guards_localized_update_memory(
    memory: float, outcome: str, tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    module = load_script("scripts/utils/inconsistency_check.py")
    monkeypatch.setattr(module, "_physical_memory", lambda: int(memory))
    cfg = compose("assimilation", *LOCALIZED_STATE_SMOOTHER, root=tmp_path)
    if outcome == "fails":
        with pytest.raises(ValueError, match="0.2 GB.*state_reduction"):
            module.check_config(cfg, "smoother")
    elif outcome == "warns":
        with pytest.warns(UserWarning, match="interval_seconds"):
            module.check_config(cfg, "smoother")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            module.check_config(cfg, "smoother")


def test_check_config_skips_memory_guard_for_parameter_smoother(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    module = load_script("scripts/utils/inconsistency_check.py")
    monkeypatch.setattr(module, "_physical_memory", lambda: 1)
    overrides = [*LOCALIZED_STATE_SMOOTHER, "smoothing.smoother=${smoother.dynamic}"]
    module.check_config(compose("assimilation", *overrides, root=tmp_path), "smoother")


def test_check_config_checks_the_replica_horizon(tmp_path: pathlib.Path) -> None:
    """assimilation.replica_dir needs a state.nc that covers the horizon."""
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    replica = tmp_path / "replica"
    replica.mkdir()
    overrides = [
        "+test=assimilation",
        "time.simulation_time=3.0",
        "assimilation.num_windows=2",
        "assimilation.truth_start_time=1.0",
        f"assimilation.replica_dir={replica}",
    ]
    cfg = compose("assimilation", *overrides, root=tmp_path)
    with pytest.raises(ValueError, match="replica_dir has no state.nc"):
        check(cfg, "smoother")
    # The horizon ends at truth_start_time + 2 * 3 s = 7 s; the output times
    # may jitter by up to half an output interval (1 s here).
    for end, ok in ((6.0, False), (6.98, True), (7.0, True)):
        times = [*np.arange(1.0, 6.0), end]
        xarray.Dataset(coords={"time": times}).to_netcdf(replica / "state.nc")
        if ok:
            check(cfg, "smoother")
        else:
            with pytest.raises(ValueError, match="before the horizon's end"):
                check(cfg, "smoother")


# The log-space z0 prior, set here so the test doesn't depend on static.yaml.
LOG_Z0 = [
    "params@truth_params=static_truth",
    "params@prior_params=static",
    "smoothing.smoother=${smoother.static}",
    "assim_model.forward_model.model_discrepancy.enabled=false",
    "+prior_params.parameters.roughness_length={_target_:pyurbanair."
    "static_parameters.LogNormal,median:0.05,log_std:0.7}",
    "assimilation.params_to_estimate=[inflow_angle,roughness_length]",
]


@pytest.mark.parametrize("workflow", ["smoother", "filtering"])  # type: ignore[misc]
def test_log_space_prior_composes(workflow: str, tmp_path: pathlib.Path) -> None:
    from hydra.utils import instantiate

    cfg = compose(
        "assimilation",
        "+test=assimilation",
        *LOG_Z0,
        "filtering.mode=joint",
        "filtering.parameter_evolution={_target_:data_assimilation.filtering."
        "parameter_evolution.RandomWalkEvolution,std:{roughness_length:0.05}}",
        root=tmp_path,
    )
    load_script("scripts/utils/inconsistency_check.py").check_config(cfg, workflow)
    sampler = instantiate(cfg.prior_params)
    params = sampler.sample(4)
    assert bool((params.roughness_length > 0).all())
    helpers = load_script("scripts/utils/helper_functions.py")
    selected, _, log_names = helpers.parameter_names(cfg, params, sampler)
    assert selected == ["inflow_angle", "roughness_length"]
    assert log_names == {"roughness_length": None}


def test_static_yaml_log_z0_block_is_ready_to_use() -> None:
    """The commented-out roughness_length priors (single and wall/ground)
    instantiate once uncommented."""
    from hydra.utils import instantiate

    from pyurbanair.static_parameters import LogNormal

    lines = (CONFIGS / "params" / "static.yaml").read_text().splitlines()
    for name in (
        "roughness_length",
        "roughness_length_wall",
        "roughness_length_ground",
    ):
        start = lines.index(f"  # {name}:")
        block = [lines[start]]
        for line in lines[start + 1 :]:
            if not line.startswith("  #   "):
                break
            block.append(line)
        block = [line.replace("# ", "", 1) for line in block]
        prior = instantiate(OmegaConf.create("\n".join(block))[name])
        assert isinstance(prior, LogNormal)
        # Bounded: the DA keeps z0 inside (min, max) through every update.
        assert prior.bounds == (0.005, 0.3)


def _z0_param(block: str, name: str, value: float) -> str:
    """Add a Constant z0 parameter to a params sampler (the test sets it)."""
    return (
        f"+{block}.parameters.{name}={{_target_:pyurbanair.static_parameters."
        f"Constant,value:{value}}}"
    )


# Static truth, prior and smoother, set here so the roughness tests don't depend
# on assimilation.yaml's parameter groups; uDALES' discrepancy off (PALM has none).
STATIC_DA = [
    "params@truth_params=static_truth",
    "params@prior_params=static",
    "smoothing.smoother=${smoother.static}",
    "assimilation.params_to_estimate=[inflow_angle]",
]
NO_UDALES_DISCREPANCY = "assim_model.forward_model.model_discrepancy.enabled=false"
WALL, GROUND = "roughness_length_wall", "roughness_length_ground"
ONE_ALONE = "set either roughness_length or both"
PALM_ONLY = "supports only roughness_length"


@pytest.mark.parametrize(  # type: ignore[misc]
    "overrides, message",
    [
        ([_z0_param("params", WALL, 0.1)], ONE_ALONE),
        ([_z0_param("params", GROUND, 0.1)], ONE_ALONE),
        (
            [
                _z0_param("params", "roughness_length", 0.1),
                _z0_param("params", WALL, 0.1),
                _z0_param("params", GROUND, 0.1),
            ],
            ONE_ALONE,
        ),
        (
            [
                "model=pypalm",
                _z0_param("params", WALL, 0.1),
                _z0_param("params", GROUND, 0.1),
            ],
            PALM_ONLY,
        ),
    ],
)
def test_check_config_rejects_forward_roughness(
    overrides: list[str], message: str, tmp_path: pathlib.Path
) -> None:
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    cfg = compose(
        "forward", "+test=forward", "params=static", *overrides, root=tmp_path
    )
    with pytest.raises(ValueError, match=f"params: .*{message}"):
        check(cfg, "forward")


@pytest.mark.parametrize(  # type: ignore[misc]
    "overrides, message",
    [
        ([_z0_param("prior_params", WALL, 0.1)], f"prior_params: {ONE_ALONE}"),
        ([_z0_param("prior_params", GROUND, 0.1)], f"prior_params: {ONE_ALONE}"),
        ([_z0_param("truth_params", WALL, 0.1)], f"truth_params: {ONE_ALONE}"),
        ([_z0_param("truth_params", GROUND, 0.1)], f"truth_params: {ONE_ALONE}"),
        (
            [
                _z0_param("prior_params", "roughness_length", 0.1),
                _z0_param("prior_params", WALL, 0.1),
                _z0_param("prior_params", GROUND, 0.1),
            ],
            f"prior_params: {ONE_ALONE}",
        ),
        (
            [
                _z0_param("truth_params", "roughness_length", 0.1),
                _z0_param("truth_params", GROUND, 0.1),
            ],
            f"truth_params: {ONE_ALONE}",
        ),
        (
            [
                "model@truth_model=pypalm",
                _z0_param("truth_params", WALL, 0.1),
                _z0_param("truth_params", GROUND, 0.1),
            ],
            f"truth_params: .*{PALM_ONLY}",
        ),
        (
            [
                "model@assim_model=pypalm",
                _z0_param("prior_params", WALL, 0.1),
                _z0_param("prior_params", GROUND, 0.1),
            ],
            f"prior_params: .*{PALM_ONLY}",
        ),
    ],
)
def test_check_config_rejects_assimilation_roughness(
    overrides: list[str], message: str, tmp_path: pathlib.Path
) -> None:
    check = load_script("scripts/utils/inconsistency_check.py").check_config
    if "model@assim_model=pypalm" not in overrides:
        overrides = [*overrides, NO_UDALES_DISCREPANCY]
    cfg = compose(
        "assimilation", "+test=assimilation", *STATIC_DA, *overrides, root=tmp_path
    )
    with pytest.raises(ValueError, match=message):
        check(cfg, "smoother")


def test_check_config_accepts_split_prior_with_a_palm_truth(
    tmp_path: pathlib.Path,
) -> None:
    """The intended cross-model run: PALM truth with one z0, uDALES prior with
    the wall/ground pair; the DA treats both LogNormal priors in bounded log
    space, like roughness_length."""
    from hydra.utils import instantiate

    log_z0 = (
        "{_target_:pyurbanair.static_parameters.LogNormal,median:0.05,"
        "log_std:0.7,min:0.005,max:0.3}"
    )
    cfg = compose(
        "assimilation",
        "+test=assimilation",
        *STATIC_DA,
        NO_UDALES_DISCREPANCY,
        "model@truth_model=pypalm",
        "truth_model.forward_model.sgs_constant=null",
        _z0_param("truth_params", "roughness_length", 0.1),
        f"+prior_params.parameters.{WALL}={log_z0}",
        f"+prior_params.parameters.{GROUND}={log_z0}",
        f"assimilation.params_to_estimate=[inflow_angle,{WALL},{GROUND}]",
        root=tmp_path,
    )
    load_script("scripts/utils/inconsistency_check.py").check_config(cfg, "smoother")
    sampler = instantiate(cfg.prior_params)
    params = sampler.sample(4)
    for name in (WALL, GROUND):
        values = params[name].values
        assert bool(((values > 0.005) & (values < 0.3)).all())
    helpers = load_script("scripts/utils/helper_functions.py")
    selected, _, log_names = helpers.parameter_names(cfg, params, sampler)
    assert selected == ["inflow_angle", WALL, GROUND]
    assert log_names == {WALL: (0.005, 0.3), GROUND: (0.005, 0.3)}
