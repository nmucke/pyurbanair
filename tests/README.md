# tests

One folder per package under test, so a change to one package maps to one
folder (and to one CI workflow, `.github/workflows/tests-<folder>.yml`):

| Folder | Tests |
|---|---|
| `pylbm/`, `pyudales/`, `pypalm/` | the CFD backends (`libs/py*`) |
| `data_assimilation/` | ESMDA, filters, localization, observation operator/error, state reduction |
| `neural_surrogates/` | architectures, datasets, training, generative spin-up |
| `evaluation/` | the `evaluation` scoring library |
| `pyurbanair/` | `src/pyurbanair`: base classes, parameter samplers |
| `mcp/` | the MCP server (`libs/mcp-server`); runs in the `mcp` env, skipped elsewhere |
| `visualization/` | the forward-run viewer and renderer (`libs/visualization`) |
| `scripts/` | `scripts/`, `configs/` and `workflows/`; relative links in the Markdown docs |

```bash
pixi run -e dev py.test             # no compiled CFD solver
pixi run -e dev test-integration    # tiny real uDALES / LBM / PALM runs
pixi run -e dev test-all            # both
pixi run -e dev python -m pytest tests/scripts
pixi run --locked -e mcp python -m pytest tests/mcp   # the MCP server's own env
```

Tests that run a compiled CFD solver are marked `integration` and are skipped
by default.

## Configs with test overlays

The tests compose the real `configs/` entry points and make them tiny with an
overlay from `tests/configs/`:

```python
cfg = compose("forward", "+test=forward", root=tmp_path)
cfg = compose("surrogate/train_stepper", "+test=train_stepper", root=tmp_path)
```

| Overlay | Makes tiny |
|---|---|
| `test/tiny.yaml` | grid 20×20×4, 3 s windows, sensors, 2 members on 1 worker (shared) |
| `test/forward.yaml`, `test/assimilation.yaml` | forward / assimilation on `model/pyudales_tiny` |
| `surrogate/test/generate_data.yaml` | 2 + 1 + 1 trajectories on the case geometry |
| `surrogate/test/train_*.yaml`, `finetune_stepper.yaml` | one CPU epoch on the synthetic data (`training.yaml`) |
| `surrogate/test/eval.yaml` | the evaluations of those models |
| `model/*_tiny.yaml` | each backend on the tiny grid; `neural_surrogate_tiny` runs the trained test surrogates |
| `model/pyudales_stock.yaml` | `pyudales_tiny` on one rank with Vreman pinned and inlet turbulence and SGS discrepancy off, for the uDALES library tests |

`compose(..., root=...)` puts every output and scratch dir under `root`. From
the command line, `--config-dir tests/configs` adds the overlays:

```bash
python scripts/run_forward.py --config-dir tests/configs +test=forward
```

Most script tests run on the neural-surrogate backend, so they need no
compiled solver: once per session, `training_data` writes a small synthetic
dataset and `trained` trains every surrogate on it (stepper, fine-tune,
autoencoder, latent generator, DFT). `neural_surrogate_tiny` then runs that
stepper, cold-started from the latent generator.

| File | Covers |
|---|---|
| `test_configs.py` | every entry point and option resolves; the test overlays are tiny; `inconsistency_check` accepts and rejects |
| `test_forward.py` | `run_forward` (single, ensemble over two saved windows, initial-state selection) + `visualize_forward` |
| `test_assimilation.py` | smoother, filtering and hybrid + `compute_metrics` + `visualize_assimilation`; a forward run as the truth |
| `test_surrogate.py` | every training task, the autoencoder's and DFT's `prechunk` data copies, the latent generator's `latent_cache`, the three evaluations, `generate_data` |
| `test_workflows.py` | `workflows/*.sh` |
