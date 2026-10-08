# Pre-chunked data for stepper and baseline training

Note (2026-10-08). Not implemented. Whether the `stepper` and
`finetune_stepper` tasks, and with them our steppers and the baselines in
[neural_surrogate_baselines.md](../neural_surrogate_baselines.md), should train
on a re-chunked copy of the corpus, as the autoencoder and the DFT already can
([neural_surrogates.md](../neural_surrogates.md) §29, `prechunk`). The final
check is the DelftBlue measurement in §4.

## 1. Why it matters

Every training sample is a `TransitionDataset` read of a few whole frames:

| Model | Frames per sample |
|---|---|
| DFT | t and t+K (K up to 5) |
| Local-FNO | t−1, t, t+1 |
| SSRollingUrbanNet | t−1 … t+1 (Roll-1), t−1 … t+3 (Roll-3) |

The corpora are stored in large time chunks: 60 frames × (11, 64, 64) cells
in the local realistic sample, 40-frame chunks in the cluster corpus. Any read
decompresses whole chunks, about 20× more data than a 3-frame window needs.
The netCDF chunk cache (64 MB) is smaller than one frame's chunks, so nothing
carries over between samples. The whole-frame copy (`spatial_chunks: null`:
time chunk 1, one chunk per frame and variable, zlib 1) only reads what the
sample uses.

## 2. Measured (laptop SSD, file in the OS cache: optimistic)

Seconds per sample on the local realistic sample (120 frames, 32×128×128):

| Read pattern | Source | Whole-frame copy |
|---|---|---|
| DFT (H=1, K=5) | 1.95 | 0.03 |
| Local-FNO (H=2, K=1) | 1.17 | 0.05 |
| SSRollingUrbanNet Roll-3 (H=2, K=3) | 1.18 | 0.08 |

On DelftBlue's BeeGFS the DFT notes measured about 1.4 s per frame from the
source and about 0.55 s from the copy. The gain is smaller there but still
large, and either way the source read is likely the training bottleneck: the
default 4 DataLoader workers deliver only a few samples per second, below the
GPU's rate for both baselines. The copy is about the source's size, and making
it took about 1.5 min per 700 MB trajectory.

## 3. What implementing it takes

- `scripts/utils/tasks.py`: in `_stepper` and `_finetune_stepper`, build the
  datasets from `prechunked_root(cfg)` with `param_root=cfg.dataset.root_dir`,
  as `_dft` does. Without a `prechunk` block nothing changes.
  `RolloutTransitionDataset` inherits `param_root`.
- A `prechunk` block (`output_root: null`, `spatial_chunks: null`) in
  `train_stepper.yaml`, `finetune_stepper.yaml` and the baseline configs.
- A CPU job makes the copy first: `surrogate_prechunk_data.slurm` with the
  stepper's config. A whole-frame copy made for the DFT on the same corpus can
  be reused.
- A test as in `tests/scripts/test_surrogate.py` (prechunked DFT), and the
  docs.

Independent of this, raise `dataloader.num_workers` (14 on a 16-CPU job):
decompression parallelises across workers.

## 4. DelftBlue check

On a compute node, with the corpus on `/projects` and the copy on `/scratch`
(it takes the time of a full copy; restrict it to a few trajectories for a
quick test):

```bash
pixi run -e delftblue python - <<'EOF'
import time, numpy as np
from neural_surrogates import TransitionDataset
from neural_surrogates.datasets.rechunk import prepare_rechunked_dataset

src = "/projects/urbanair/training_data/pyudales_realistic"
copy = prepare_rechunked_dataset(src, "/scratch/<user>/pyudales_realistic_frames", spatial_chunks=None)
for root in (src, copy):
    for H, K in ((1, 5), (2, 1), (2, 3)):
        ds = TransitionDataset(root_dir=root, split="train", state_vars=["u", "v", "w"],
                               param_vars=None, param_root=src,
                               num_history_steps=H, pushforward_steps=K)
        idx = np.random.default_rng(0).choice(len(ds), 9, replace=False)
        ds[int(idx[0])]
        t0 = time.perf_counter()
        for i in idx[1:]:
            ds[int(i)]
        print(root, H, K, f"{(time.perf_counter() - t0) / 8:.2f} s/sample")
EOF
```

Implement §3 if the copy is clearly faster per sample and a short GPU run
shows data loading limiting the step rate (time per batch well above the
model's forward/backward time).
