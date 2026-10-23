"""Roll trained steppers out on the same trajectories, from the same start frame.

Config: configs/comparison.yaml. The first step of
workflows/compare_surrogates.sh; compute_metrics.py and visualize_comparison.py
(next to this script) score and draw the run dir it writes.

    python scripts/surrogate_comparison/run_models.py \
        'models=[model_weights/local_fno,model_weights/dft_s]'

Every model starts from the same frame, the latest any model needs as history
(``max(H) - 1``), each from its own last ``H`` true frames up to it, and rolls
out `max_steps` steps (to the trajectory's end if null) with the trajectory's
parameters. The models must have been trained on the same data: the same
training-data folder name, state and parameter variables (``param_vars: null``
read as the corpus' parameter names) and, where the corpus' config.yaml is
readable, the same output frequency; the script refuses to compare them
otherwise. Models are labelled by their folder name (if two names clash, by the
shortest path suffix that tells them apart, e.g. `seed0_local_fno`).

Outputs, in `paths.results_dir` (the run dir):
  config.yaml                        the resolved config, `sample_indices` set
                                     to the trajectories compared
  models.yaml                        per label: model_dir, num_history_steps,
                                     n_parameters, seconds_per_step (inference
                                     on `device`, first step of each rollout
                                     excluded) and the data signature
  rollouts/<label>/sample_<NNNN>.nc  the predicted frames (lead steps 1..N) of
                                     trajectory NNNN of the split
Persistence is not written: compute_metrics.py derives it from the truth.
A rerun into the same run dir first removes the earlier rollouts and models.yaml.

`inlet_turbulence.enabled` adds uDALES' inlet turbulence to every predicted
frame (neural_surrogates.inlet_turbulence.InletTurbulence; the corpus must
have been generated with it), outside the timed inference. The setting is
then recorded in models.yaml and the rollouts' `inlet_turbulence_seed` attr
(`truth`: the truth run's own realization, else the seed).
`inlet_turbulence.apply: input` adds it only to the frame fed back to the
model, u_t = F(u_{t-1} + forcing), and stores the model's own output; the
rollouts then also carry `inlet_turbulence_apply: input`.

`periodic_y` makes every model y-periodic at inference (circular y padding in
its convolutions, y-periodic SDF features:
neural_surrogates.periodic.make_periodic_y), for uDALES' y-periodic domains.
It is then recorded in models.yaml and the rollouts' `periodic_y` attr (1).
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import time
from typing import Any

import hydra
import numpy as np
import torch
import xarray
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from eval_common import (  # noqa: E402
    device,
    inlet_turbulence,
    load_model,
    load_states,
    open_dataset,
)

RESERVED = ("truth", "persistence", "truth_halves")


def run(cfg: DictConfig) -> None:
    if not cfg.models:
        raise ValueError("list at least one trained model in `models`")
    if cfg.max_steps is not None and int(cfg.max_steps) < 1:
        raise ValueError(f"max_steps must be at least 1, got {cfg.max_steps}")
    dev = device(cfg.device)
    out = pathlib.Path(cfg.paths.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "rollouts", ignore_errors=True)
    (out / "models.yaml").unlink(missing_ok=True)

    labels = _labels(list(cfg.models))
    signatures = _check_same_data(
        {
            label: OmegaConf.load(pathlib.Path(d) / "config.yaml")
            for label, d in labels.items()
        }
    )
    models = {
        label: load_model(d, dev, periodic_y=cfg.periodic_y)
        for label, d in labels.items()
    }
    first_cfg = next(iter(models.values()))[1]
    data = open_dataset(first_cfg, cfg.data_dir, cfg.split, sdf_features="none")
    histories = {
        label: int(c.dataset.get("num_history_steps") or 1)
        for label, (_, c) in models.items()
    }
    start = max(histories.values()) - 1
    samples = (
        list(range(len(data._state_files)))
        if cfg.sample_indices is None
        else [int(s) for s in cfg.sample_indices]
    )
    cfg.sample_indices = samples
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)

    timings = {label: [0.0, 0] for label in models}  # seconds, steps
    for sample in samples:
        length = data._traj_lengths[sample]
        steps = length - 1 - start
        if cfg.max_steps is not None:
            steps = min(steps, int(cfg.max_steps))
        if steps < 1:
            raise ValueError(
                f"sample {sample} has {length} frames: nothing to predict after "
                f"the start frame {start}"
            )
        window = load_states(data, sample, range(start + 1))
        params = data._params[sample]
        geometry = data.geometry_for(sample)
        frames = _frame_coords(data, sample, start, steps)
        inlet = inlet_turbulence(cfg.inlet_turbulence, data, sample)
        for label, (model, _) in models.items():
            pred, seconds = _rollout(
                model,
                window,
                params,
                geometry,
                histories[label],
                steps,
                dev,
                inlet,
                cfg.inlet_turbulence.apply == "input",
            )
            timings[label][0] += seconds
            timings[label][1] += max(steps - 1, 1)
            ds = xarray.Dataset(
                {
                    var: (("time", "z", "y", "x"), pred[:, c])
                    for c, var in enumerate(data.state_vars)
                },
                coords=frames,
                attrs={
                    "sample": sample,
                    "start": start,
                    "model_dir": str(labels[label]),
                    "split": str(cfg.split),
                    "data_dir": str(pathlib.Path(data.root).resolve()),
                },
            )
            if inlet is not None:  # "truth": the truth run's own realization
                seed = cfg.inlet_turbulence.seed
                ds.attrs["inlet_turbulence_seed"] = (
                    "truth" if seed is None else str(seed)
                )
                if cfg.inlet_turbulence.apply == "input":
                    ds.attrs["inlet_turbulence_apply"] = "input"
            if cfg.periodic_y:
                ds.attrs["periodic_y"] = 1
            path = out / "rollouts" / label / f"sample_{sample:04d}.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(path)
        print(f"sample {sample}: {steps} steps from frame {start}")

    OmegaConf.save(
        OmegaConf.create(
            {
                label: {
                    "model_dir": str(pathlib.Path(labels[label]).resolve()),
                    "num_history_steps": histories[label],
                    "n_parameters": sum(p.numel() for p in model.parameters()),
                    "seconds_per_step": timings[label][0] / timings[label][1],
                    "device": str(dev),
                    "data": signatures[label],
                }
                | (
                    {"inlet_turbulence": cfg.inlet_turbulence}
                    if cfg.inlet_turbulence.enabled
                    else {}
                )
                | ({"periodic_y": True} if cfg.periodic_y else {})
                for label, (model, _) in models.items()
            }
        ),
        out / "models.yaml",
    )
    print(f"Saved rollouts in {out}")


def _labels(model_dirs: list[str]) -> dict[str, str]:
    """A unique label per model: its folder name or, if names clash, the
    shortest path suffix that tells them apart, joined by `_` (a folder name)."""
    parts = [pathlib.Path(d).resolve().parts for d in model_dirs]
    if len(set(parts)) != len(parts):
        raise ValueError(f"a model is listed twice: {model_dirs}")
    for n in range(1, max(map(len, parts)) + 1):
        labels = ["_".join(p[-n:]).lstrip("_/") for p in parts]
        if len(set(labels)) == len(labels):
            break
    reserved = set(RESERVED) & set(labels)
    if reserved:
        raise ValueError(
            f"model labels {sorted(reserved)} are reserved; rename the folder"
        )
    return dict(zip(labels, model_dirs))


def _check_same_data(train_cfgs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Refuse models trained on different data; return each one's signature.

    Compares the training-data folder name (paths may differ between machines),
    the state and parameter variables (`param_vars: null` read as the corpus'
    parameter names, unknown if its param files can't be read), and the
    corpus' output frequency where its config.yaml can be read. A root whose
    interpolation cannot be resolved is left out of the folder comparison.
    """

    def signature(cfg: Any) -> dict[str, Any]:
        dataset = cfg.dataset
        try:
            root = pathlib.Path(str(OmegaConf.select(cfg, "dataset.root_dir")))
        except Exception:
            root = None
        frequency = None
        if root is not None and (root / "config.yaml").exists():
            frequency = OmegaConf.select(
                OmegaConf.load(root / "config.yaml"), "time.output_frequency"
            )
        param_vars = dataset.get("param_vars")
        if param_vars is not None:
            param_vars = list(param_vars)
        elif root is not None:
            files = sorted((root / "param" / "train").glob("sample_*.nc"))
            if files:
                with xarray.open_dataset(files[0]) as ds:
                    param_vars = [str(v) for v in ds.data_vars]
        return {
            "data": root.name if root is not None else None,
            "output_frequency": frequency,
            "state_vars": list(dataset.state_vars),
            "param_vars": param_vars,
        }

    signatures = {name: signature(cfg) for name, cfg in train_cfgs.items()}
    different = False
    for key in ("data", "output_frequency", "state_vars", "param_vars"):
        known = {str(s[key]) for s in signatures.values() if s[key] is not None}
        different |= len(known) > 1
    if different:
        raise ValueError(
            "models trained on different data:\n  "
            + "\n  ".join(f"{n}: {s}" for n, s in signatures.items())
        )
    return signatures


def _frame_coords(data: Any, sample: int, start: int, steps: int) -> dict:
    """Coordinates of the predicted frames: corpus times, lead and cell centres."""
    frames = slice(start + 1, start + 1 + steps)
    with xarray.open_dataset(data._state_files[sample]) as ds:
        dims = [d for d in ds[data.state_vars[0]].dims if d != "time"]
        coords = {"time": ds["time"].values[frames]}
        coords |= {new: ds[d].values for new, d in zip(("z", "y", "x"), dims)}
    return coords | {"lead": ("time", np.arange(1, steps + 1))}


@torch.no_grad()  # type: ignore[misc, unused-ignore]
def _rollout(
    model: Any,
    window: np.ndarray,
    params: torch.Tensor,
    geometry: torch.Tensor,
    history: int,
    steps: int,
    dev: torch.device,
    inlet: Any = None,
    inlet_input: bool = False,
) -> tuple[np.ndarray, float]:
    """`steps` predicted frames after the last of `window` (its last `history`
    frames as input), and the inference time of all steps but the first.

    `inlet` (if given) adds the inlet turbulence to every predicted frame
    before it is stored and fed back; with `inlet_input` only to the frame fed
    back, so the stored frame is the model's own output. Its time is not
    counted.
    """
    start = len(window) - 1
    channels, grid = window.shape[1], window.shape[2:]
    state = torch.from_numpy(window[start - history + 1 :])
    state = state.reshape(1, history * channels, *grid).to(dev)
    geom = geometry.unsqueeze(0).to(dev)
    pred = np.empty((steps, *window.shape[1:]), dtype=np.float32)
    tic = time.perf_counter()
    excluded = 0.0
    for k in range(steps):
        if k == 1:  # the first step pays the warm-up
            _sync(dev)
            tic = time.perf_counter()
        step = model(state, params[start + k].unsqueeze(0).to(dev), geom)
        forced = step
        if inlet is not None:
            _sync(dev)
            injected = time.perf_counter()
            forced = inlet(step, start + k + 1)
            _sync(dev)
            if k > 0:
                excluded += time.perf_counter() - injected
        pred[k] = (step if inlet_input else forced)[0].float().cpu().numpy()
        state = (
            forced if history == 1 else torch.cat([state[:, channels:], forced], dim=1)
        )
    _sync(dev)
    return pred, time.perf_counter() - tic - excluded


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None, config_path="../../configs", config_name="comparison"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
