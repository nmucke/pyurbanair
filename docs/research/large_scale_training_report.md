# Engineering large-scale training of the 3D urban-flow surrogates

Report for the methods section of the article. It covers the techniques that make
training the neural surrogates on a large corpus of 3D large-eddy simulations
feasible and efficient. The model architecture itself (Tadpole autoencoder, DFT
stepper, latent flow-matching generator) is out of scope; it is referenced only where
a technique depends on it. Status: 13 October 2026, branch `feat/dft-grad-accum`.
Implementation references and the knob names behind each technique are collected in
[Appendix A](#appendix-a-implementation-map).

Every number below is a measurement unless it is marked as an estimate (est.).
Hardware: NVIDIA H100 (93 GB usable, Snellius `gpu_h100`), A100 (40 GB on Snellius,
80 GB on DelftBlue), and an RTX 3090 (24 GB) for development. All training uses bf16
autocast.

---

## 1. Setting and the scaling problem

**Training data.** The realistic corpus consists of 520 uDALES large-eddy simulations
of flow through real urban layouts (500 train / 10 validation / 10 test
trajectories). Each trajectory is a separate geometry on its own grid:

- Resolution and size: 4 m resolution, 32 vertical cells, and horizontal extents from
  96×128 to 272×320 cells. That gives 0.29M–2.8M cells per frame, in 69 distinct grid
  shapes.
- Time: 120 stored frames at a 5 s interval of the three velocity components, plus a
  static solid/fluid mask.
- Size: 324 GB of NetCDF.
- Distribution of the 500 train trajectories by grid size:

| Cells per frame | ≤ 0.6M | ≤ 1.0M | ≤ 1.5M | ≈ 2.5M | 2.8M |
|---|---|---|---|---|---|
| Trajectories | 98 (cumulative) | 262 (cumulative) | 394 (cumulative) | 26 | 16 |

**Training task (DFT stepper).** The stepper is fine-tuned from a frozen,
pretrained size-B autoencoder (38.1M parameters), using LoRA adapters plus fully
trained latent modules: 8.45M of 50.2M parameters are trainable. It is trained with
a pushforward rollout of up to 4 steps:

- The first K − 2 steps run without gradients.
- Gradients are back-propagated through the last 2 model calls.
- One epoch at horizon 4 has 500 × (120 − 4) = 58,000 samples. Every sample reads two
  whole 3D frames.
- The target effective batch size is 8.

**Why this is hard at scale.**

1. **Activation memory.** With gradients through two full-resolution encoder–decoder
   passes, the original implementation needed ≈ 19–20 GB of saved activations per
   million grid cells per sample (H100 and 3090 measurements agree). A single sample
   of the largest grid peaked at 54 GB.
   - On a 40 GB A100, training was impossible even at micro-batch 1: 11 % of the
     trajectories exceed the card.
   - On a 93 GB H100, micro-batches above 1 ran out of memory on large grids. Measured
     at horizon 1: micro-batch 8 fails on the first step, and micro-batch 4 fails as
     soon as a large grid appears.
2. **Heterogeneous shapes.** The 69 grid shapes rule out a fixed batch shape,
   whole-model graph compilation and CUDA graphs. They also make every new shape pay
   cuDNN autotuning.
3. **I/O.** The source files are compressed in 40-frame chunks, so every sample
   decompresses far more data than it uses.
4. **Throughput and time-limited jobs.** Before this work, one epoch at horizon 4 took
   7.9 h on an H100. The planned run (originally 50 epochs at horizon 5) projected to
   more than 250 GPU-hours, executed as chained 48 h Slurm jobs.

The techniques below address these problems. Taken together, they turned an
infeasible configuration (out-of-memory or > 250 GPU-h) into a 2–3 job run (est.
49–52 GPU-h) at effective batch 8. On the H100 they doubled sustained training
throughput, from 2.05 to 4.4 samples/s, at a lower peak memory per sample.

---

## 2. Generating the training corpus at scale

**Per-geometry domains.** Each trajectory gets its own building layout from the
UrbanTALES pool and a domain fitted to it:

- The building footprint is padded by 64 m upstream, downstream and laterally.
- The horizontal extents are rounded up to multiples of 16 cells, the total
  down-sampling stride of the autoencoder, so no grid needs runtime padding to fit
  the architecture.
- Layouts reaching the domain top (128 m) are skipped.
- Validation and test use held-out geometries.

**Adaptive spin-up.** A simulation is only recorded once the turbulence has
developed. The spin-up time is

T_spin = clamp(6 · L_x / u_slow, 300 s, 1500 s), with u_slow = 0.8 · U · max(cos θ, 0.1),

where L_x is the domain length, U the inflow speed and θ the inflow angle. The flow
has then crossed the domain six times. T_spin is rounded up to whole output
intervals. After spin-up, 600 s are simulated and stored every 5 s.

**Time-varying forcing.** Inflow angle and speed follow AR(2) time series:

- Angle: mean ~ U(−45°, 45°), standard deviation 15°.
- Speed: mean ~ U(5, 10) m s⁻¹, standard deviation 0.5 m s⁻¹.
- Correlation length 300 s.

The parameters are interpolated to the stored frame times and stored next to the
states.

**Embarrassingly parallel generation.**

- Samples are grouped by geometry. Each group prepares one solver case and runs its
  members in sequence.
- Groups are distributed round-robin over independent Slurm jobs (shard k runs groups
  g with g mod n_shards = k). Planning (geometry split, parameter draws, inlet seeds)
  is a deterministic function of the configuration, so every shard plans the
  identical corpus without communication.
- The solver itself is MPI-parallel. DelftBlue generation costs 1.6–2.1 µs per grid
  cell per simulated second (the source does not say whether this is wall or core
  time). One 1M-cell trajectory of 900–2,100 s simulated time (spin-up plus 600 s)
  therefore takes about 0.4–1.2 h of solver time (est.).

**Crash-safe, resumable output.**

- Every file is written under a temporary name and atomically renamed. The parameter
  file is written last, so a sample counts as complete only when both files exist.
- A restarted shard skips completed samples. A failing geometry is logged and skipped
  without aborting the shard.

**Storage format.**

- The staggered solver output is interpolated to cell centres.
- The solid mask is rebuilt from the solver's own solid-cell list, because uDALES
  leaves non-zero velocities inside buildings.
- Fields are stored with zlib level 1 and a lossy quantisation to three significant
  decimal digits.

---

## 3. Storage and I/O

### 3.1 Lossless re-chunked training copies

The source NetCDF files store each variable in 40-frame zlib chunks. A training
sample that reads two frames therefore decompresses about 20× more data than it
uses.

**What the copy is.** A one-off, lossless copy of the state files with a layout
matched to the access pattern:

- One time step per chunk.
- Spatial chunks of either a whole frame (for the stepper, which reads whole
  domains) or 16×64×64 tiles (for the autoencoder, which trains on random crops).

**How the copy guarantees integrity.**

- Every written block is re-read and compared bit for bit with the source.
- Each output file is written under a temporary name and atomically renamed.
- A JSON manifest records each source file's size and modification time and the
  chunking options.
- Training validates the manifest and every output file in seconds, without locks or
  writes. The copy can therefore be shared read-only by concurrent jobs.
- Any change in the sources or options fails closed.
- Parameters are still read from the source.
- For the stepper and baselines, normalisation statistics are streamed from the
  source, whose large chunks read a split about 3× faster. The autoencoder computes
  them from its tile copy and caches them there.

**Results.**

| Measurement (DelftBlue BeeGFS unless noted) | Source | AE-tile copy | Whole-frame copy |
|---|---|---|---|
| Time per whole-frame read | ≈ 1.4 s (decompression-bound) | ≈ 2.5 s (latency-bound) | ≈ 0.55 s (≈ 0.1 s CPU) |
| CPU per 3-frame baseline sample | ≈ 1.7 s | – | ≈ 0.3 s (≈ 0.1 s per frame) |
| Per-sample read time, local SSD: DFT | 1.95 s | – | 0.03 s |
| Per-sample read time, local SSD: Local-FNO | 1.17 s | – | 0.05 s |
| Per-sample read time, local SSD: SSRolling | 1.18 s | – | 0.08 s |

Further figures:
- **Size:** the copies are about the size of the source: 323 GB for whole frames,
  354 GB for tiles.
- **Copy time:** about 13 h for the 520 files on one core, about half that for the
  whole-frame copy, at ≤ 1 GB RAM.
- **Speed-up from writing whole output chunks at once:** 4× faster copying for tiles,
  10× for frames.

### 3.2 Latent cache for the generator

The latent flow-matching generator is trained in the latent space of the frozen
autoencoder. Encoding every frame on the fly costs about 8× the generator network's
own FLOPs per step.

- A single GPU pass encodes every train and validation frame once and stores
  memory-mappable float32 latents per trajectory, together with the geometry
  conditioning and float64 running sums for the latent statistics.
- Training then reads latents and skips the encoder entirely.
- Size: about 112 GB for the 500 train trajectories.
- The cache is keyed on the SHA-256 of the autoencoder weights and its settings, so a
  retrained autoencoder cannot silently use stale latents.
- It is resumable and validated like the re-chunked copies.

---

## 4. Data loading and batching

**Lazy windows.**

- A dataset item is a (state at t, parameters over t … t+K, state at t+K) window.
- At start-up only the number of time steps per file is read.
- Each item issues one read for the frames it needs, so intermediate frames of the
  rollout are never read.
- Each loader worker keeps one open file at a time, because every open NetCDF file
  holds a 0.1–0.2 GB chunk cache.

**Per-trajectory batching with a cell budget.**

- Grids of different shapes cannot be stacked, so every micro-batch is drawn from a
  single trajectory.
- The micro-batch size adapts to the grid:

  b = min(b_max, max(1, ⌊C / N_cells⌋)),

  where C is a cell budget derived from the measured memory model (Section 7).
- In the launched configuration (b_max = 4, C = 7.5M cells), a micro-batch has up to
  4 samples on grids up to 1.87M cells, 3 at ≈ 2.5M and 2 at 2.8M. On average there
  are 3.76 samples per micro-batch.
- Shuffling is seeded per epoch, so a resumed job does not replay earlier epochs'
  orders.

**One shared copy of the geometry.** Every trajectory has its own solid mask and
signed-distance features. These are precomputed once per unique geometry at start-up
(exact Euclidean distance transform, plus gradient, clamped to ±32 cells) and
deduplicated by content.

- *The problem:* PyTorch's DataLoader moves every tensor a worker returns into shared
  memory. A per-geometry tensor kept by the dataset is therefore duplicated once per
  worker for every geometry that worker serves. On the realistic corpus this was
  estimated at 11–12 GB per worker, or 170–180 GB of host memory for 15 workers.
  That is a likely host out-of-memory error during the first epoch.
- *The fix:* all masks and features are copied into one shared-memory buffer before
  the workers start. Workers then exchange handles to that single copy.
- *Idealized corpus:* `/dev/shm` dropped from 775 MB to about 350 MB.
- *90-minute H100 run (realistic corpus):* `/dev/shm` stayed at 3.85 GB. The
  proportional set size of the main process and its 16 loader processes stayed at
  34.6 GB, growing by 54 MB over 10 minutes. The main process's anonymous memory was
  constant at 3.98 GB.

**Batches carry one geometry.** Because a micro-batch has a single geometry, the
mask and features are sent once per micro-batch, not once per sample. A batch that
mixes geometries fails loudly.

**Loader sizing.**

- With a whole-frame read of about 0.55 s, a sample costs about 1.1 s of I/O wait.
- Workers needed ≈ 1.1 s × (samples/s) × 1.5 safety (est.), so 15 workers are enough
  for about 8 samples/s. That is above the sustained 4.4 samples/s.
- Locally the loader delivered 48–66 batches/s.

---

## 5. Normalisation statistics

- State statistics (per-channel mean and standard deviation) are streamed file by
  file as float64 sums and sums of squares.
- They are restricted to fluid cells, using each trajectory's own mask.
- They are cached on disk, keyed by the file list with sizes and modification times,
  variables and split.
- The fine-tuned stepper reuses the frozen autoencoder's state statistics, so no pass
  over the 324 GB corpus is needed when fine-tuning. Only the parameter statistics
  are computed, from the in-memory parameter tables.

---

## 6. Training procedure

**Mixed precision.**

- bf16 autocast for all convolutions and matrix products.
- fp32 for normalisation, softmax and the loss, following PyTorch's autocast policy.
- Weights, gradients and AdamW state stay in fp32.
- No loss scaling, because bf16 has fp32's exponent range.
- TF32 is enabled for the remaining fp32 matrix products.

**Pushforward with two-step gradients.**

- The rollout runs its first K − 2 steps under `no_grad`. Their activations are
  freed at once.
- The last two calls keep their activations.
- Peak memory is therefore independent of the horizon for K ≥ 2. Measured: horizons
  3, 4 and 5 peak like horizon 2.
- The no-grad and gradient-carrying parts use separate autocast contexts. Autocast's
  cache of casted weights would otherwise be shared between them, with the wrong
  `requires_grad`.

**Horizon curriculum.**

- The training horizon grows by one step every 2 epochs (1, 1, 2, 2, 3, 3, 4, …), so
  9 of 15 epochs run at horizon 4.
- Validation always runs at the target horizon, so best-model selection compares like
  with like.
- The patience counter restarts at every horizon change. Early stopping can therefore
  only trigger in the final stage.

**Gradient accumulation with a fixed number of samples.** The adaptive micro-batches
of Section 4 have 1–4 samples. Accumulating a fixed number of micro-batches would
make the effective batch vary with grid size, and would weight samples in small
micro-batches more. Instead, gradients are accumulated until at least N = 8 samples
have been seen:

- Each micro-batch's mean loss is multiplied by its sample count.
- The accumulated gradient is divided by the group's sample total before clipping and
  the optimizer step.
- Every optimizer step therefore uses the per-sample mean gradient over 8 samples.

Validated:
- against a single large batch, matching to a relative 2.4 × 10⁻⁷ in fp32;
- against a hand-computed reference with variable micro-batch sizes and a short final
  group.

**Learning-rate schedule.**

- Linear warmup from 10⁻⁶ to the peak (7 × 10⁻⁴), then cosine decay to 5 × 10⁻⁶.
- Gradient clipping at norm 1.
- The epoch-wise warmup used originally spent the whole first epoch (58,000 samples,
  about 3.6 h) at 10⁻⁶. The warmup now advances per optimizer step across the first
  epoch, so the peak is used from epoch 2.
- The number of optimizer steps in that epoch is computed exactly by replaying the
  sampler's planned micro-batch sizes at that epoch's horizon and seeded shuffle.

**Sync-free masked loss.**

- The loss is restricted to fluid cells, because the solver targets contain arbitrary
  values inside buildings. Boolean-mask indexing forced two GPU-to-host
  synchronisations per step.
- It is replaced by an exact equivalent, ℓ = mean(f(m ⊙ ŷ, m ⊙ y)) · N / N_fluid, where
  m is the fluid mask. The factor N / N_fluid is kept on the device and recomputed
  only when the geometry changes.
- This is valid for element-wise losses with f(a, a) = 0 (MSE, L1, Huber).
- It also tolerates non-finite values inside obstacles.
- Running losses are accumulated on the device and synchronised once per epoch.
- Geometry uploads are asynchronous from pinned memory.

---

## 7. Compute and memory efficiency of the fine-tuning step

### 7.1 Fused LoRA

**Why LoRA was the largest cost.** A LoRA-adapted convolution with frozen weight W
normally computes

y = conv(x; W) + s · conv(conv(x; A); B),

where A is an r-channel 3×3×3 convolution and B a 1×1×1 convolution. At rank
r = 32 the low-rank branch is as wide as many of the autoencoder's layers. It also
stores its own r-channel full-resolution intermediate for the backward pass.

- On the size-S model, the adapters cost about 1.5× the frozen convolutions.
- The extreme case is the full-resolution output layer `decompress` (32 → 1 channels).
  There the adapter costs 32× the frozen layer.

**The fused computation.** The same function is computed with the merged weight:

y = conv(x; W + s · B A).

- B A is formed in fp32 outside autocast on every call (W stays frozen).
- Gradients reach A and B through the merge. The low-rank branch is thereby replaced
  by one weight-gradient convolution per layer.
- It falls back to the standard path for dropout, LoRA variants, bias adapters and
  grouped convolutions.

**Equivalence.**

- fp64: loss relative difference 7.5 × 10⁻¹⁶; LoRA gradients 2 × 10⁻¹¹.
- fp32: gradients 3 × 10⁻⁶.
- In bf16 the fused form is slightly closer to the fp32 reference, because it rounds
  W + δW once instead of rounding the base and the adapter outputs separately.
- State-dict keys, the adapter files and the merged export are bit-identical.

**Per-layer cost (fwd + bwd, 3090, at 32×144×192).**

| Layer | Standard LoRA | Fused |
|---|---|---|
| B `feature_embed` | 28.3 ms | 15.7 ms |
| B `down0` | 32.5 ms | 18.5 ms |
| B `decompress` | 56.0 ms | 29.2 ms |
| B 128-channel block | 24.7 ms | 22.1 ms |

**Whole training step.**

| Hardware and grid | Step time | Peak memory |
|---|---|---|
| 3090, size S, 0.885M cells | −36 % | −31 % |
| H100, size B, 0.885M cells | −26 % | −16 % |
| H100, size B, 2.8M cells (vs the original code, so including the always-on changes) | −34 % | −30 % |

A related always-on fix: the PEFT library casts each adapter's input to fp32, and
under bf16 autocast it is cast straight back. Removing this cast lowers peak memory
by 13 %, bit-identically.

### 7.2 Regional compilation

Whole-model `torch.compile` is impractical here:
- 69 shapes;
- gradient-free and gradient-carrying calls of the same modules;
- adapter wrappers;
- tensor rearrangements in the transformer stages, where compiling exceeded 14 minutes
  and was abandoned.

Instead, only the two convolutional stems of the encoder and decoder are compiled,
in place, with dynamic shapes:

- **What it fuses:** GroupNorm, GELU and the element-wise operations around the
  convolutions.
- **Memory:** fewer fp32 intermediates are saved for backward.
- **Compilation behaviour:**
  - The first compilation takes 45–115 s per process.
  - Over a 90-minute run covering all 69 shapes and micro-batch sizes 1–4, exactly
    three recompilations occurred. All three were caused by gradient mode (no-grad
    rollout step, first gradient step, and an input that requires gradients in the
    second gradient step). None was caused by a new shape.
  - Compiling in place keeps state-dict keys, adapter merging and checkpoints
    unchanged.
- **Equivalence with the uncompiled model:** loss relative difference 2 × 10⁻⁴ and
  gradients 4 × 10⁻³ in bf16; 4 × 10⁻⁵ and 2 × 10⁻⁴ in fp32/TF32.

On the H100 (B model), compiling on top of fused LoRA gives:
- −11 % step time and −33 % peak memory at 0.885M cells;
- −11 % and −34 % at 2.8M cells.

### 7.3 Launch overhead and redundant work removed (always on, bit-identical)

At micro-batch 1 the fused step launches about 8,000 GPU kernels. On small grids the
step time was set by the CPU issuing kernels, not by the GPU.

| Change | Effect |
|---|---|
| The latent transformer's positional embedding was rebuilt on the host and copied to the GPU every call (the last per-step host→device synchronisation); it is now cached on the device | −11 % step time |
| The frozen window-attention relative-position bias (a small MLP evaluated per window size) was recomputed every call; it is now computed once and reused | ≈ 800 fewer kernels per step, −4 % step time |
| The frozen geometry encoder is cached on the identity of the geometry tensor, but the trainer created a new broadcast view on every call, so the cache never hit; it now reuses the view while the geometry and batch size are unchanged | geometry-encoder evaluations 6 → 1 for 2 micro-batches × 3 rollout steps (CPU check) |
| Global spatial mode returned its single whole-grid region through a zero-filled buffer and a copy; it now returns it directly | small memory reduction |
| An index list was built with a tensor → list conversion; replacing it with a Python range removed a graph break for compilation | enables regional compilation |

Combined effect of all always-on changes, at micro-batch 1 and 0.885M cells:

| Hardware | Step time | Peak memory |
|---|---|---|
| H100, size B | −15 % | −16 % |
| 3090, size S | −12 % | −18 % |

### 7.4 cuDNN autotuning with many shapes

`cudnn.benchmark` autotunes each new combination of convolution shape and batch size.
The plan search runs on the CPU while the GPU idles:

- About 12 s per new grid on the 3090, covering about 43 forward and 32 backward
  convolution configurations.
- About 25–30 min of slow steps at the start of every H100 job for the 69 grids. GPU
  utilisation was 12 % during this warm-up, against 86 % afterwards with the original
  code.

Disabling autotuning saves the warm-up but costs 4–7 % of steady-state speed. For
48 h jobs autotuning stays on. Mixing micro-batch sizes multiplies the number of
cached plans beyond cuDNN's default limit of 10,000, so the cache limit is lifted
(`TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=0`).

### 7.5 Memory model and choice of micro-batch

With fused LoRA and regional compilation, the H100 peak memory follows

peak ≈ 0.5 GB + 8.9 GB × (million cells) × (micro-batch),

with the same slope at 0.885M and 2.8M cells. The original implementation needed
19.3–19.8 GB per million cells per sample.

| Hardware, model size | Original | Fused | Fused + compile |
|---|---|---|---|
| H100, B | 19.3–19.8 | ≈ 14 | 8.9 |
| 3090, B | 19.7 | 14.3 | ≈ 11 (est.) |
| 3090, S | 8.7 | 5.9 | 4.3 |

Values are GB of peak memory per million cells per sample, with gradients through
2 steps. The H100 fused-only value is derived from the measured 12.4 GB at 0.885M
cells and 38.1 GB at 2.8M cells.

The cell budget C = 7.5M cells caps the largest micro-batch at about 67 GB. The
90-minute run peaked at 71.3 GB including allocator reserve and cuDNN workspace,
which leaves about 20 GB of headroom on the 93 GB card.

**Larger micro-batches.** They help less on the H100 than on the 3090, because the
size-B step is mostly bound by GPU work.

| Grid | Micro-batch 1 → 2 | Micro-batch 1 → 4 |
|---|---|---|
| 0.885M cells | +6 % throughput | +10 % throughput |
| 2.8M cells | +2 % throughput | – |
| 3090, size S (for comparison) | +15 % | +25 % |

**Rejected or optional memory techniques:**

- **Activation checkpointing of the convolutional blocks** (non-reentrant).
  - 3090: −27 to −43 % memory for +6 to 11 % time.
  - H100: −18 % memory for −10 % throughput.
  - Not needed after the changes above.
  - Pitfall: the reentrant checkpointing built into the vendored model drops the
    gradients of encoder adapters silently, because the encoder input does not
    require gradients. Only non-reentrant checkpointing is safe with LoRA on frozen
    layers.
- **Overlapping-tile ("halo") spatial decomposition instead of whole-domain (global)
  processing.**
  - Overlap and padding cost 1.8–7.8× the cells.
  - Without per-tile recomputation it needed more training memory than global: 12.4
    vs 3.75 GB at 0.4M cells.
  - At 2.8M cells (size S): global 1.23 s / 21.8 GB; global with recomputation
    1.37 s / 10.1 GB; best halo configuration 4.06 s / 11.6 GB.
  - After a 400-step fine-tune all modes reached the same one-step error, so global
    was kept.
- **Running all but the last gradient step's encoder without gradients.**
  - Rejected because it changes the gradient: encoder-adapter gradient cosine 0.994,
    norm 0.61×.

---

## 8. Autoencoder pretraining at scale

The size-B autoencoder is trained from scratch on the realistic corpus before
any stepper or generator. Its training problem differs from the stepper's in two
ways: it reconstructs single snapshots rather than rolling out in time, and it
trains on crops rather than whole domains. The techniques below apply to
that setting. Shared infrastructure already described (bf16, TF32, cuDNN
autotuning, the shared geometry copy, checkpoint/resume) is not repeated.

**Setting of the production run.**

- **Model:** 38.4M parameters in the encoder and decoder, plus a 3.5M-parameter
  geometry branch (41.9M in total).
- **Hardware:** a full 80 GB A100 on DelftBlue, in 48 h jobs.
- **Run length:** 200 epochs.
- **Epoch size:** one sample per stored snapshot, so 500 × 120 = 60,000 crops.
- **Batches:** 48 crops. Batches are drawn per trajectory and the last partial batch
  is kept (120 frames give 48 + 48 + 24), so an epoch is 1,500 optimizer steps.
- **Schedule:** learning rate 10⁻⁴ with a 5-epoch warmup and a cosine decay to 10⁻⁶.
- **Result:** best validation loss 0.0052, at epoch 195.

### 8.1 Crop-based training with tiled encoding

- **Lazy random crops.** Each item is a 64³ crop at a random origin. Only the crop is
  read from disk; the rest of the frame never leaves it. An axis shorter than the
  crop is taken whole, so on the realistic corpus (32 vertical cells) every crop is
  32×64×64 cells.
- **Geometry on the full grid.** The geometry mask and signed-distance features are
  computed once per unique geometry on the full grid, then cropped per item. The
  distance field therefore still sees buildings outside the crop.
- **Tiled encoding.** The autoencoder encodes in local mode. Each crop is cut into
  16×32×32 tiles, and the three velocity components and all tiles are folded into the
  batch dimension. A 32×64×64 crop becomes 2×2×2 tiles per component, i.e. 24
  single-channel tiles per sample and 1,152 per batch of 48.
- **What tiling buys.**
  - Activation memory is set by the tile and crop size, not by the domain. The batch
    size can therefore be chosen once for all 69 grid shapes.
  - Every batch has the same shape, so cuDNN autotunes it only once.
  - It follows the intermediate pre-cropping of the original Tadpole training.
- **Anisotropic tiles.** Tile shapes may differ per axis (each a multiple of 16). The
  16-cell vertical extent avoids padding a 16-cell idealized domain to 32. The
  32-cell realistic domain divides it exactly, so no compute is spent on padding.
- **Tile size shapes the learned function.** Reconstruction error (NRMSE, idealized
  corpus, size S) on the tile it was trained on is 0.139. On 16×64×64 tiles it is
  0.248, and on whole domains 0.288. This is why the downstream stepper re-adapts the
  frozen autoencoder through LoRA rather than using it as is (Section 7).

### 8.2 Batch size from a memory probe

- **Why the cell budget is off.** The cell-budget sampler of Section 4 counts
  full-grid cells, not crop cells, so it would cap every realistic trajectory at
  batch 1. It is therefore disabled for the autoencoder, and the batch size is set
  directly.
- **The probe:**

| Batch size | Reserved GPU memory (A100 80 GB) |
|---|---|
| 32 | ≈ 50 GiB |
| 55 | ≈ 77 GiB (10 steps completed) |
| 48 (chosen) | – |

- **Derived memory rate:** about 10.7 GiB reserved per million crop cells.

### 8.3 I/O for crops and host memory

- **The source layout.** A crop from the source files decompresses whole 40-frame
  chunks to use one frame.
- **The tile-chunked copy.** The copy of Section 3.1 with 16×64×64 spatial chunks and
  one frame per chunk. A random 32×64×64 crop then touches 2–3 × 1–2 × 1–2 chunks of
  one frame. No per-crop read time was measured.
- **Bounded file handles.** netCDF4 keeps a 64 MiB chunk cache per variable per open
  file. The first full-corpus run, with 4 loader workers, hit the 32 GB host-memory
  limit after 123 batches. Each worker now keeps at most one file open, which bounds
  the cache memory at one file per worker.
- **Normalisation statistics** are streamed once from the copy and cached with it.

### 8.4 Wall time of the production run

The run's own log (`metrics.csv`) records three clearly separated regimes of epoch
time:

| Epochs | Mean epoch time | Crops/s (derived, incl. validation) |
|---|---|---|
| 1–8 | 20,111 s (5.6 h) | ≈ 3.0 |
| 9–56 | 4,820 s (1.3 h) | ≈ 12 |
| 57–200 | 1,740 s (0.48 h) | ≈ 34 |

- **Totals:** 178.6 GPU-hours, which needs at least four chained 48 h jobs.
- **Likely causes of the two speed-ups.** The surviving launch record shows a resumed
  job on 5 October that read from the tile-chunked copy with 12 loader workers. The
  configuration written by the final submission has 48 workers. This is consistent
  with:
  - **Epochs 1–8:** reading the source files, before the copy existed.
  - **Epochs 9–56:** reading the copy, ≈ 4.2× faster.
  - **Epochs 57–200:** 48 instead of 12 workers, a further ≈ 2.8×.

  The attribution is inferred from these records, not measured. The DelftBlue job
  logs would confirm it.
- **Cost of the per-epoch warmup.** The warmup advanced per epoch, so the first five
  epochs (about 28 h in the slow I/O regime) trained below the peak learning rate,
  and the first at 10⁻⁶. This motivated the per-step warmup used for the stepper
  (Section 6).

### 8.5 Consistent hand-off to downstream models

- **What is exported.** The encoder, decoder and geometry-branch weights that
  downstream models load are re-cut from the best full-model weights. This happens at
  the start of every job, at every new best validation loss, and at the end of
  training.
- **Why.** Before this, the separately written exports could lag the best weights by
  an epoch (an epoch-40 export next to epoch-39 best weights). Downstream models
  could then silently load an inconsistent encoder–decoder pair.
- **Hardware-independent parity.** The decoder's latent is made contiguous, so the
  autoencoder and the stepper built from it give identical outputs across CPU
  instruction sets. Before this, AVX-512 introduced ≈ 10⁻⁶ drift.

### 8.6 Evaluation on whole domains

- **What is evaluated.** The trained autoencoder is evaluated on uncropped domains, on
  fluid cells only, with per-grid vertical profiles.
- **Bounding memory.** On the largest grid a single snapshot folds into 540
  single-channel tiles. A micro-batching limit on the encoder (`max_internal_batchsize`)
  processes them in chunks.
- **Hardware needed.** Evaluation then fits on a 10 GB MIG slice of an A100, at
  batch 1 with chunks of 2 tiles. In training this limit does not reduce memory,
  because autograd keeps every chunk's activations.

### 8.7 Optional adversarial loss (not used in production)

- **Mechanism.** A geometry-conditioned hinge-loss critic can be added to sharpen
  reconstructions. Its weight is scaled by the adaptive factor of Esser et al.
  (clamped to [0, 1]) and ramped in after 1,000 optimizer steps. The critic has its
  own optimizer and reuses the detached reconstruction, so it adds no second
  autoencoder forward pass.
- **Cost.** It adds three critic forward passes and two gradient-norm probes per
  step.
- **Constraints.** It cannot be combined with gradient accumulation, because the
  critic steps once per optimizer step.
- **Status.** Production used no critic.

---

## 9. Fault tolerance on time-limited jobs

- **Periodic full checkpoint.** Training runs as chained 48 h Slurm jobs. A full
  checkpoint (model, optimizer, scheduler, autocast scaler, curriculum horizon and
  best-model state) is written every epoch. Resubmitting the identical command
  resumes where the last job stopped.
  - The epoch count is the total, not the epochs to add.
  - The cosine schedule is extended if a run is lengthened.
  - The sampler's per-epoch seed and the per-step warmup state are restored.
- **Best-model weights.** They are written on every validation improvement, with a
  sidecar that records the score, so a resumed run never overwrites a better model
  with a worse one.
- **Fine-tuned exports.** The merged weights are written as a plain state dict and
  the deployment configuration is stamped to skip loading the pretrained
  autoencoder, so deployment does not need the autoencoder directory.
- **Atomic intermediate artifacts.** Data generation, the re-chunked copies, the
  latent cache and the statistics cache all write atomically, so any job can be
  killed and resubmitted.

---

## 10. Verification

- **Gradient and value parity.** Every change that should not alter the mathematics
  has a CPU test against the reference path. Tested: fused LoRA, regional
  compilation, non-reentrant checkpointing, the masked loss with non-finite values in
  solids, gradient accumulation by steps and by samples, the warmup schedule under
  the curriculum, and resume. 625 tests pass.
- **Bit-identity under deterministic kernels.** With cuDNN determinism and
  deterministic algorithms enabled, 40 training steps of the new code with default
  settings reproduce the original code bit for bit, including optimizer updates. Two
  runs of the original code are also bit-identical, which validates the check.
- **Training equivalence on real data.** 400 identical training steps on real
  transitions (3090, size S, horizon 4, batch 1):
  - original code: final train loss 0.595, held-out loss 0.723, peak 4.13 GB;
  - recommended settings (fused LoRA + compile): 0.595 / 0.723 / 2.51 GB;
  - repeat of the original code: 0.605 / 0.752.

  The spread between repeats comes from cuDNN's non-deterministic backward and is
  larger than the difference between methods.
- **Benchmark noise.** Repeated fixed-shape H100 benchmarks agreed within 0.3 %.

---

## 11. Results on the H100

**Fixed-shape benchmark.** One fixed batch, 8 warm-up and 30 timed steps, horizon 4
with 2 gradient steps, size-B autoencoder, one process per configuration, run back to
back on one GPU.

| 0.885M cells (32×144×192) | Micro-batch | s/step | samples/s | Peak |
|---|---|---|---|---|
| original | 1 | 0.345 | 2.90 | 17.5 GB |
| always-on changes | 1 | 0.295 | 3.39 | 14.7 GB |
| + fused LoRA | 1 | 0.219 | 4.58 | 12.4 GB |
| + regional compile | 1 | 0.195 | 5.14 | 8.3 GB |
| + regional compile | 2 | 0.366 | 5.46 | 16.2 GB |
| + regional compile | 4 | 0.706 | 5.67 | 31.9 GB |

| 2.8M cells (32×272×320) | Micro-batch | s/step | samples/s | Peak |
|---|---|---|---|---|
| original | 1 | 0.973 | 1.03 | 54.1 GB |
| fused LoRA | 1 | 0.644 | 1.55 | 38.1 GB |
| fused + compile | 1 | 0.570 | 1.75 | 25.3 GB |
| fused + compile | 2 | 1.115 | 1.79 | 50.1 GB |
| + conv-block checkpointing | 2 | 1.238 | 1.62 | 40.9 GB |

Per sample, this is 1.7–1.8× the throughput at 47 % of the peak memory.

**Full pipeline on the real corpus.** All 69 shapes, horizon 4, effective batch 8, a
90-minute run per configuration.

| | Original pipeline | Final pipeline |
|---|---|---|
| Micro-batching | 1 sample × 8 accumulation steps | 1–4 samples by cell budget, 8 samples per step |
| Sustained throughput (after warm-up) | 2.05 samples/s | 4.4 samples/s (2.16×) |
| GPU busy (`nvidia-smi` SM activity, steady state) | 86 %, one CPU core saturated | 100 % (42 % averaged over the warm-up) |
| Peak GPU memory (incl. reserve) | 57 GB at micro-batch 1 (horizon-5 run) | 71 GB at micro-batches up to 4 |
| Epoch at horizon 4 | 7.9 h | 3.6 h |

**Training budget.** The final schedule is 15 epochs with the horizon curriculum.
Est. 49–52 GPU-hours, i.e. two to three 48 h jobs, against about 100 GPU-hours for
the same schedule before this work (est.) and more than 250 GPU-hours for the
original 50-epoch, horizon-5 plan (est.).

---

## 12. Caveats and open points for the article

- **Shape-dependent throughput.** Throughput depends on the grid mix. The 2.16× is
  the sustained average over the realistic corpus, while the fixed-shape gains are
  1.7–1.8× per sample.
- **Mid-epoch progress is lost.** Checkpoints are written only at epoch ends, so a
  job killed at its time limit loses its partial epoch.
- **Non-atomic checkpoint write.** `checkpoint.pt` is not written atomically. Copy it
  aside between jobs.
- **Learning rate in the last epoch.** The cosine reaches its minimum only after the
  final epoch, so the last epoch trains at 1.4 × 10⁻⁵ rather than 5 × 10⁻⁶.
- **Unseeded weight initialisation.** The trainable modules' initial weights are not
  seeded, so runs are reproducible in configuration, data order and evaluation but
  not in initial weights.
- **Parameter timing lag.** The current corpora label their frames from t = 0, so the
  trained models carry a one-interval (5 s) lag between parameters and states. This
  affects any claim about the parameter–time convention (see
  `docs/plans/surrogate_retraining.md`).
- **Autoencoder settings.** The repository's autoencoder config defaults are the
  size-S idealized ones. The size-B production run used command-line overrides
  (size B, batch 48, no cell budget, last partial batch kept), recorded in the
  run's own config.
- **Noisy autoencoder validation loss.** Validation crops are drawn at random each
  epoch, so the autoencoder's validation loss carries crop-sampling noise.
- **Stale autoencoder evaluation.** The stored evaluation metrics of the autoencoder
  (relative L2 0.032 on 48 snapshots) predate its final best weights and should be
  recomputed before quoting.
- **Limited validation of the speed work.** The accuracy equivalence of the speed
  changes was established on the idealized corpus with the size-S model. For the
  size-B production run it rests on the parity tests and on bit-identity.

---

## Appendix A: implementation map

| Technique | Where | Setting in the production run |
|---|---|---|
| Sharded, resumable corpus generation, adaptive spin-up | `scripts/surrogate/generate_data.py`, `configs/surrogate/generate_data.yaml` | `data.sharding.*`, `spinup.fill_times: 6` |
| Re-chunked copies | `neural_surrogates/datasets/rechunk.py` | `prechunk.output_root=…/pyudales_realistic_frames` |
| Latent cache | `neural_surrogates/datasets/latent_cache.py` | `latent_cache.output_root` (generator) |
| Autoencoder crops and tiles | `datasets/snapshot.py`, `architectures/tadpole_ae.py`, `_tadpole_crop.py` | `dataset.random_crop_size=64`, `architecture.encoder_crop_size=[16,32,32]`, `spatial_mode: local` |
| Autoencoder batch and tile copy | `configs/surrogate/train_autoencoder.yaml` (overrides) | `batch_sampler.batch_size=48 batch_sampler.cell_budget=null batch_sampler.drop_last=false`, `prechunk.spatial_chunks=[16,64,64]`, 48 loader workers |
| Autoencoder hand-off exports | `scripts/utils/tasks.py` (`_autoencoder`) | always on |
| Whole-domain evaluation, encoder micro-batching | `scripts/surrogate/evaluate_autoencoder.py` | `autoencoder.max_internal_batchsize=2` |
| Adversarial critic (optional) | `training/autoencoder.py` | `discriminator: null` (off) |
| Lazy windows, shared geometry | `datasets/transition.py` (`share_once`) | always on |
| Per-trajectory cell-budget batching | `datasets/sampler.py` | `batch_sampler.batch_size=4 batch_sampler.cell_budget=7500000` |
| Streaming fluid-only statistics | `training/data_utils.py` | always on |
| bf16 autocast, TF32, cuDNN autotune | `training/base.py` | `trainer.amp`, `tf32`, `cudnn_benchmark` on |
| Pushforward, 2-step gradients, curriculum | `training/base.py` | `dataset.pushforward_steps=4 trainer.grad_unroll_steps=2 trainer.pushforward_epochs_per_step=2` |
| Sample-count accumulation | `training/base.py` | `trainer.grad_accum_samples=8` |
| Per-step warmup, cosine schedule | `training/base.py` | `trainer.lr_warmup_epochs=1 trainer.lr_warmup_per_step=true optimizer.lr=7e-4 trainer.lr_min=5e-6` |
| Sync-free masked loss | `training/standard.py` | always on |
| Fused LoRA, no input cast, merged export | `finetuning/inject.py` | `lora.fused_forward=true` |
| Regional compilation | `training/base.py` | `trainer.compile_modules=[dft.encoder.conv_encoder,dft.decoder.conv_decoder]` |
| Activation checkpointing (optional) | `training/base.py` | off |
| Checkpoint/resume | `training/base.py` | `trainer.checkpoint_every=1`, `resume: true` |
| Benchmark tool | `scripts/tools/benchmark_dft_step.py` | – |

Further detail: [neural_surrogates.md](../neural_surrogates.md) (§6, §10, §26–38),
[job_scripts.md](../job_scripts.md), and the handover
[dft_training_speed_handover.md](../plans/dft_training_speed_handover.md).
