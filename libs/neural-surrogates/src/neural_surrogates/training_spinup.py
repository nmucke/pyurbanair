"""Seed an ESMDA assimilation from pre-computed training trajectories.

When the neural surrogate is the assimilation model and the run opts into the
``training_data`` spin-up source, the first assimilation window is warm-started
from training snapshots instead of a CFD spin-up: each ensemble member starts
from the **last** frame of a training trajectory, and its prior inflow
parameters are anchored to that sample's final inflow value. This module owns
the on-disk ``training_data/`` layout knowledge and the (streaming,
memory-bounded) loading; the surrogate forward model itself stays a pure
roll-from-the-provided-state model.

The two pieces:

* :func:`write_initial_state_files` streams the last frame of each member's
  training trajectory to its own ``state_{i}.nc`` under a directory ESMDA reads
  per member (so the full ensemble never sits in RAM — peak is one frame).
* :func:`anchor_prior_params` shifts each member's sampled prior so its first
  time step matches that member's training sample's final inflow, preserving the
  prior draw's shape (the AR(2) anomaly) while pinning its level — exactly what
  keeps the params ESMDA estimates consistent with the warm-start state.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import xarray as xr


def resolve_training_root(
    model_dir: Optional[str | pathlib.Path],
    override: Optional[str | pathlib.Path] = None,
) -> pathlib.Path:
    """Training-data root: ``override`` if given, else the trained dataset root.

    The trained-model folder's ``config.yaml`` records the ``dataset.root_dir``
    the surrogate was trained on; that is the training-data tree the cold-start
    snapshots are drawn from when no explicit ``override`` is configured.
    """
    if override is not None:
        return pathlib.Path(override)
    if model_dir is None:
        raise ValueError(
            "training-data spin-up needs a root: pass training_data_spinup.root "
            "or a model_dir whose config.yaml records dataset.root_dir."
        )
    from omegaconf import OmegaConf

    cfg_path = pathlib.Path(model_dir) / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"trained-model config not found at {cfg_path}; cannot resolve the "
            "training-data root for the spin-up."
        )
    train_cfg = OmegaConf.load(cfg_path)
    return pathlib.Path(train_cfg.dataset.root_dir)


def list_split_samples(
    root: str | pathlib.Path, split: str
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Sorted ``(state, param)`` sample files for one split of ``training_data/``.

    Globs are sorted so member ``i`` maps to the same ``sample_*`` pair the
    trainer's ``TransitionDataset`` would index.
    """
    root = pathlib.Path(root)
    state_dir = root / "state" / split
    param_dir = root / "param" / split
    state_files = sorted(state_dir.glob("sample_*.nc"))
    param_files = sorted(param_dir.glob("sample_*.nc"))
    if not state_files:
        raise FileNotFoundError(
            f"no training samples found under {state_dir}; cannot seed a "
            "training_data warm start."
        )
    if len(state_files) != len(param_files):
        raise ValueError(
            f"training-data sample count mismatch in split '{split}': "
            f"{len(state_files)} state vs {len(param_files)} param under {root}."
        )
    return state_files, param_files


def member_sample_index(member_index: int, n_samples: int) -> tuple[int, int]:
    """Map a member to ``(sample_index, repeat)``, cycling over the split.

    ``repeat`` is how many full passes over the split precede this member (``0``
    for the first ``n_samples`` members); it gates the per-member parameter
    jitter that keeps members reusing a sample from coinciding.
    """
    return member_index % n_samples, member_index // n_samples


def write_initial_state_files(
    state_files: list[pathlib.Path],
    n_members: int,
    out_dir: str | pathlib.Path,
    frame: int = -1,
) -> pathlib.Path:
    """Stream each member's window-0 initial state to ``out_dir/state_{i}.nc``.

    Member ``i`` is seeded from the ``frame``-th snapshot (default the last) of
    training sample ``i % n_samples``. Only that one frame is read and written
    per member, so peak memory is a single snapshot regardless of ensemble size
    or trajectory length. The directory layout (``state_{i}.nc``) is exactly what
    :meth:`BaseEnsembleForwardModel.get_member_state` reads for a path warm
    start, so the directory can be handed straight to ESMDA as the window-0
    initial state. The raw (un-collocated) frame is written; the forward model
    collocates it to the regular grid on warm start, as before.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(state_files)
    for i in range(n_members):
        sample_index, _ = member_sample_index(i, n_samples)
        with xr.open_dataset(state_files[sample_index]) as ds:
            snapshot = ds.isel(time=frame).load()
        snapshot.to_netcdf(out_dir / f"state_{i}.nc")
    return out_dir


def anchor_prior_params(
    prior_params: xr.Dataset,
    param_files: list[pathlib.Path],
    n_members: int,
    initial_param_jitter_scale: float = 0.0,
    frame: int = -1,
) -> xr.Dataset:
    """Anchor each member's prior so its first time step matches its sample.

    For every time-varying parameter present in both the prior and the training
    samples, member ``i``'s trajectory is shifted by a constant so its first
    value equals training sample ``i % n_samples``'s value at ``frame`` (the same
    frame :func:`write_initial_state_files` used as the initial state, so the
    inflow and the state it produced agree). The prior draw's shape — the AR(2)
    anomaly relative to its first step — is preserved by the constant shift.

    When the ensemble is larger than the split, members beyond the first pass
    reuse a sample and would share an identical anchor; a small multiplicative
    jitter (seeded per member) breaks the tie. Static (time-less) params and
    params absent from the training samples are left untouched.
    """
    n_samples = len(param_files)

    # Per-sample anchor value for each time-varying parameter (one file read per
    # sample, not per member).
    sample_anchors: list[dict[str, float]] = []
    for f in param_files:
        with xr.open_dataset(f) as ds:
            loaded = ds.load()
        sample_anchors.append(
            {
                str(name): float(loaded[name].isel(time=frame).values)
                for name in loaded.data_vars
                if "time" in loaded[name].dims
            }
        )

    out = prior_params.copy()
    anchored_names = {
        str(name)
        for name in prior_params.data_vars
        if "time" in prior_params[name].dims
        and any(str(name) in anchors for anchors in sample_anchors)
    }
    for name in anchored_names:
        arr = out[name].transpose("time", "ensemble").values.copy()  # (T, E)
        anchors = np.empty(arr.shape[1])
        for i in range(n_members):
            sample_index, repeat = member_sample_index(i, n_samples)
            value = sample_anchors[sample_index][name]
            if repeat > 0 and initial_param_jitter_scale > 0.0:
                rng = np.random.default_rng(i)
                value *= 1.0 + initial_param_jitter_scale * float(rng.standard_normal())
            anchors[i] = value
        arr = arr - arr[0:1, :] + anchors[None, :]
        out[name] = (("time", "ensemble"), arr)
    return out
