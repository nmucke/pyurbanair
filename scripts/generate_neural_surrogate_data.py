"""Generate a neural-surrogate training corpus from a Fortran solver (§5, P1).

Principle: **the existing Fortran backends are the data generator** — reuse the
ensemble machinery rather than inventing a runner. Because the corpus is
architecture-agnostic, the *same* corpus trains the UNet now and UPT later.

    pixi run -e dev python scripts/generate_neural_surrogate_data.py \
        model=pylbm +generate.n_trajectories=200 \
        +generate.corpus_path=.temp/neural_surrogate/corpus

**Do the §5 sizing table first** (storage = N_traj · N_frames · nx·ny·nz ·
n_vars · bytes; compute = N_traj · wall_time / workers). Record before running.
"""

import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    configure_failure_policy,
    create_parameter_ensemble,
)


def _generate_cfg(cfg: DictConfig) -> dict:
    """Read corpus-generation knobs (conf/neural_surrogate/data.yaml shape)."""
    defaults = {
        "corpus_path": ".temp/neural_surrogate/corpus",
        "n_trajectories": 64,
        "time_varying": False,
        "include_pressure": False,
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "seed": 0,
        "geometry_static": {"include_sdf": True, "include_mask": True},
    }
    gen = OmegaConf.select(cfg, "generate")
    if gen is not None:
        defaults.update(OmegaConf.to_container(gen, resolve=True))  # type: ignore[arg-type]
    return defaults


def _augment_constant_params(tv, schema_names, cfg, ensemble_size):
    """Add schema params the time-series model doesn't sample as constants.

    The external priors typically cover ``inflow_angle`` + ``velocity_magnitude``;
    a uDALES corpus also needs ``pressure_gradient_magnitude``, so fill it from
    ``params.true`` as a per-member constant the conditioning builder can read.
    """
    import numpy as _np

    for name in schema_names:
        if name in tv:
            continue
        value = float(cfg.params.true[name])
        tv = tv.assign(
            {name: ("ensemble", _np.full(ensemble_size, value, dtype=float))}
        )
    return tv


def _assign_split(idx: int, n: int, val_frac: float, test_frac: float) -> str:
    # Split BY TRAJECTORY before indexing (§5): never leak frames across splits.
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    if idx < n_test:
        return "test"
    if idx < n_test + n_val:
        return "val"
    return "train"


def run(cfg: DictConfig) -> pathlib.Path:
    # Heavy NN imports stay function-local (lazy invariant).
    from neural_surrogates.data.generate import CorpusWriter
    from neural_surrogates.data.grid import GridMeta, build_occupancy_mask, build_static_channels
    from neural_surrogates.data.normalization import fit_normalization
    from neural_surrogates.utils import state_io
    from neural_surrogates.utils.params_io import conditioning_for_frames
    from neural_surrogates.utils.schema import (
        ContractSchema,
        default_state_var_names,
        param_schema_for_solver,
    )

    gen = _generate_cfg(cfg)
    model_name = cfg.model.name
    solver_name = cfg.model.solver_name
    rng = np.random.default_rng(gen["seed"])

    grid = GridMeta(
        nx=cfg.domain.nx, ny=cfg.domain.ny, nz=cfg.domain.nz,
        bounds=tuple(tuple(float(v) for v in b) for b in cfg.domain.bounds),  # type: ignore[arg-type]
    )
    var_names = default_state_var_names(include_pressure=gen["include_pressure"])
    param_schema = param_schema_for_solver(solver_name)
    contract = ContractSchema(solver_name, param_schema, var_names)

    # Geometry: voxelize the STL ONCE, on the collocated grid (D5). Not every
    # backend exposes an STL on its forward_model (uDALES uses case_dir), so
    # allow an explicit `generate.stl_path` override and fall back to the
    # model's own field when present.
    stl_path = gen.get("stl_path") or OmegaConf.select(
        cfg, "model.forward_model.stl_path"
    )
    if stl_path is None:
        raise ValueError(
            "No STL geometry available: set generate.stl_path explicitly for "
            f"backends without a forward_model.stl_path (model={model_name})."
        )
    mask = build_occupancy_mask(stl_path, grid)
    static = build_static_channels(
        mask,
        include_sdf=gen["geometry_static"]["include_sdf"],
        include_mask=gen["geometry_static"]["include_mask"],
        grid=grid,
    )

    # Build the solver + ensemble on disk (full simulation_time + spin-up, §5).
    forward_model = instantiate(cfg.model.forward_model, results_dir=None)
    instantiate(cfg.model.prepare, forward_model=forward_model)

    writer = CorpusWriter(gen["corpus_path"], grid, contract, var_names, static, mask)
    output_frequency = float(cfg.time.output_frequency)

    n_traj = int(gen["n_trajectories"])
    ensemble_size = int(cfg.ensemble.ensemble_size)

    # Optional time-varying parameters (§5): sample transient inflow series from
    # the parameter_time_series machinery so the corpus teaches transient BCs.
    # The external priors (params.external) may carry per-window mean/std
    # *profiles* (lists), letting x_ext(t) / Σ_ext(t) vary over the window.
    time_varying = bool(gen.get("time_varying", False))
    ts_model = None
    tv_time_coords = None
    if time_varying:
        import jax

        from pyurbanair.config.hydra_helpers import make_time_coords
        from pyurbanair.parameter_time_series import build_parameter_time_series

        ts_model = build_parameter_time_series(
            method=cfg.time_varying.method,
            external_priors=OmegaConf.to_container(cfg.params.external, resolve=True),
            ensemble_size=ensemble_size,
            method_kwargs=OmegaConf.to_container(
                cfg.time_varying.method_kwargs, resolve=True
            ),
        )
        tv_time_coords = make_time_coords(
            float(cfg.time.simulation_time), int(cfg.time_varying.num_time_points)
        )

    traj_idx = 0
    round_seed = int(gen["seed"])
    while traj_idx < n_traj:
        if time_varying:
            import jax

            tv = ts_model.sample_prior(tv_time_coords, jax.random.PRNGKey(round_seed))
            params_ensemble = _augment_constant_params(
                tv, param_schema.names, cfg, ensemble_size
            )
        else:
            params_ensemble = create_parameter_ensemble(
                model_name=model_name,
                prior_cfg=cfg.params.prior,
                ensemble_size=ensemble_size,
                seed=round_seed,
                param_names=param_schema.names,
            )
        round_seed += 1

        ensemble_model = instantiate(cfg.model.ensemble_model, forward_model=forward_model)
        configure_failure_policy(ensemble_model, cfg.ensemble.failure)
        for member in ensemble_model.ensemble_forward_models:
            clean_outputs(model_name=model_name, forward_model=member)

        states = ensemble_model.run_ensemble(params=params_ensemble, sim_name="state")
        if states is None:
            states = ensemble_model.get_states()

        # uDALES staggered -> collocated grid before tensors (D3).
        if model_name == "pyudales":
            from pyudales.utils.grid_utils import interpolate_grid

            states = interpolate_grid(states)

        for member in range(ensemble_size):
            if traj_idx >= n_traj:
                break
            ds = states.isel(ensemble=member)
            fields = state_io.state_to_tensor(ds, grid, var_names)  # [T, C, Z, Y, X]
            n_frames = fields.shape[0]
            frame_times = np.arange(n_frames, dtype=float) * output_frequency
            member_params = params_ensemble.isel(ensemble=member)
            cond = conditioning_for_frames(member_params, param_schema, frame_times)
            split = _assign_split(
                traj_idx, n_traj, gen["val_fraction"], gen["test_fraction"]
            )
            writer.add_trajectory(
                f"traj_{traj_idx:05d}", fields, cond, frame_times, split
            )
            traj_idx += 1

    corpus_path = writer.finalize(
        extra={
            "source_solver_name": solver_name,
            "output_frequency": output_frequency,
            "git_sha": _git_sha(),
        }
    )

    # Fit normalization on the train split only (§6.3); store at the corpus root.
    from neural_surrogates.data.generate import open_corpus

    corpus = open_corpus(corpus_path)
    import json

    norm = fit_normalization(
        (corpus.load_fields(i) for i in corpus.split_ids("train")), var_names, mask=mask
    )
    with open(pathlib.Path(corpus_path) / "normalization.json", "w") as f:
        json.dump(norm.to_dict(), f, indent=2)

    print(f"Wrote corpus with {traj_idx} trajectories to {corpus_path}")
    return pathlib.Path(corpus_path)


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
