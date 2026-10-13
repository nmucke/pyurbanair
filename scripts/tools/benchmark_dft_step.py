"""Time DFT training steps on one fixed batch through the real Trainer code path.

Usage (from the repo root):
    pixi run -e cuda python scripts/tools/benchmark_dft_step.py \
        --pretrained model_weights/tadpole_ae_b_realistic --grid 32 144 192 \
        --batch 1 --steps 4 --warmup 8 --iters 30 --out bench.jsonl -- \
        lora.fused_forward=true [any other train_dft override]

Builds the model like `scripts/surrogate/train.py` with `task: dft` (frozen AE,
LoRA, trainable modules) and the configured `trainer`, then runs
`trainer._train_epoch()` on `--warmup` and then `--iters` identical batches: a
train sample of `paths.data_dir` tiled (mirrored) or cropped to `--grid`, with
SDF features recomputed for the tiled mask, and `--steps` pushforward steps.
A fixed shape keeps cuDNN autotuning and compilation out of the timed part.
Prints `RESULT {...}` (s/it, samples/s, peak CUDA memory) and appends it as a
JSON line to `--out`. The loss is meaningless (tiled fields); only time and
memory are.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch


class FakeLoader(list):
    """A list of batches with the DataLoader attributes the trainer reads."""

    dataset = None
    batch_sampler = None


def tile_to(t: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    """Tile the last three dims of t (reflecting, so edges stay continuous) and crop."""
    out = t
    for axis, n in zip((-3, -2, -1), grid):
        reps = []
        size = 0
        flip = False
        while size < n:
            reps.append(out.flip(axis) if flip else out)
            size += out.shape[axis]
            flip = not flip
        out = torch.cat(reps, dim=axis).narrow(axis, 0, n)
    return out.contiguous()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--grid", type=int, nargs=3, default=None, help="z y x (default: sample's own)"
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=4, help="pushforward horizon K")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--sample", type=int, default=0, help="train sample index to tile")
    p.add_argument("--label", default="")
    p.add_argument("--out", default=None, help="append a JSON result line here")
    p.add_argument("--pretrained", required=True, help="AE export dir (pretrained_dir)")
    args, overrides = p.parse_known_args()
    overrides = [o for o in overrides if o != "--"]

    root = pathlib.Path.cwd()
    sys.path.insert(0, str(root / "scripts" / "utils"))
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from neural_surrogates.sdf import sdf_features
    from omegaconf import OmegaConf
    from tasks import TASKS

    with initialize_config_dir(str(root / "configs"), version_base=None):
        cfg = compose(
            "surrogate/train_dft",
            [
                f"pretrained_dir={args.pretrained}",
                "name=bench",
                "paths.weights_dir=.temp/bench_runs",
                "dataloader.num_workers=0",
                f"dataset.pushforward_steps={args.steps}",
                f"trainer.pushforward_start_steps={args.steps}",
                "trainer.resume=false",
                *overrides,
            ],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.dataloader.persistent_workers = False
    out_dir = root / ".temp" / "bench_runs" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup = TASKS[cfg.task](cfg, out_dir)
    collate = instantiate(cfg.dataloader.collate_fn)

    item = setup.train[args.sample]
    batch = collate([item] * args.batch)
    batch = {k: v.clone() for k, v in batch.items()}  # drop shared identity
    K = batch["params_n"].shape[1]
    if K != args.steps:
        raise SystemExit(f"dataset gave K={K}, wanted {args.steps}")
    if args.grid is not None:
        grid = tuple(args.grid)
        for key in ("state_n", "state_next", "geometry"):
            batch[key] = tile_to(batch[key], grid)
        if "geom_features" in batch:
            batch["geom_features"] = sdf_features(
                batch["geometry"][0],
                clamp_cells=cfg.dataset.sdf_clamp_cells,
                mode=cfg.dataset.sdf_features,
            )[None].to(batch["geom_features"].dtype)
    for k, v in batch.items():
        batch[k] = v.pin_memory()
    grid = tuple(batch["state_n"].shape[-3:])
    cells = grid[0] * grid[1] * grid[2]

    FakeLoader.dataset = setup.train
    trainable = [q for q in setup.model.parameters() if q.requires_grad]
    trainer = instantiate(cfg.trainer, _partial_=True)(
        model=setup.model,
        train_loader=FakeLoader([batch]),
        val_loader=FakeLoader([batch]),
        optimizer=instantiate(cfg.optimizer, params=trainable),
        loss_fn=instantiate(cfg.loss),
        weights_path=None,
        num_epochs=1,
        **{k: v for k, v in setup.trainer_kwargs.items() if k != "weights_transform"},
    )
    trainer._profile_window = None

    def epoch(n: int) -> tuple[float, float]:
        trainer.train_loader = FakeLoader([batch] * n)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = trainer._train_epoch()
        torch.cuda.synchronize()
        return time.perf_counter() - t0, loss

    t_warm, _ = epoch(args.warmup)
    torch.cuda.reset_peak_memory_stats()
    dt, loss = epoch(args.iters)
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    res = dict(
        label=args.label,
        grid=list(grid),
        mcells=round(cells / 1e6, 3),
        batch=args.batch,
        steps=K,
        overrides=overrides,
        warmup_s=round(t_warm, 2),
        s_per_it=round(dt / args.iters, 4),
        samples_per_s=round(args.batch * args.iters / dt, 3),
        mcell_samples_per_s=round(args.batch * args.iters * cells / 1e6 / dt, 3),
        peak_alloc_gb=round(peak, 2),
        peak_reserved_gb=round(reserved, 2),
        loss=loss,
        gpu=torch.cuda.get_device_name(),
    )
    print("RESULT " + json.dumps(res))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
