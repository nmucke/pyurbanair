# tests_new

The successor of `tests/`, which is retired once the refactoring to
`scripts_new/` + `configs_new/` is done. One folder per package under test, so
a change to one package maps to one folder:

| Folder | Tests |
|---|---|
| `pylbm/`, `pyudales/`, `pypalm/` | the CFD backends (`libs/py*`) |
| `data_assimilation/` | ESMDA, filters, localization, observation operator/error, state reduction |
| `neural_surrogates/` | architectures, datasets, training, generative spin-up |
| `evaluation/` | the `evaluation` scoring library |
| `pyurbanair/` | `src/pyurbanair`: base classes, parameter samplers, forward preparation and visualization |
| `mcp/` | the MCP server |
| `scripts/` | `scripts_new/`, `configs_new/` and `workflows/` |

```bash
pixi run -e dev test-new               # no compiled CFD solver
pixi run -e dev test-new-integration   # tiny real uDALES / LBM / PALM runs
pixi run -e dev python -m pytest tests_new/scripts
```

Tests that run a compiled CFD solver are marked `integration` and are skipped
by default.

## scripts/: configs_new with test overlays

The script tests compose the real `configs_new/` entry points and make them
tiny with an overlay from `configs/`:

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

`compose(..., root=...)` puts every output and scratch dir under `root`. From
the command line, `--config-dir tests_new/configs` adds the overlays:

```bash
python scripts_new/run_forward.py --config-dir tests_new/configs +test=forward
```

Most script tests run on the neural-surrogate backend, so they need no
compiled solver: once per session, `training_data` writes a small synthetic
dataset and `trained` trains every surrogate on it (stepper, fine-tune,
autoencoder, latent generator, DFT). `neural_surrogate_tiny` then runs that
stepper, cold-started from the latent generator.

| File | Covers |
|---|---|
| `test_configs.py` | every entry point and option resolves; the test overlays are tiny; `inconsistency_check` accepts and rejects |
| `test_forward.py` | `run_forward` (single, ensemble over two windows) + `visualize_forward` |
| `test_assimilation.py` | smoother, filtering and hybrid + `compute_metrics` + `visualize_assimilation`; a forward run as the truth |
| `test_surrogate.py` | every training task, the three evaluations, `generate_data` |
| `test_workflows.py` | `workflows/*.sh` |

## legacy/: carried over from tests/

The library tests came over from `tests/` unchanged apart from import paths.
Some compose the frozen test configs of the old `conf/` schema through the
`compose_test_cfg` / `compose_module_cfg` / `surrogate_model_dir_factory`
fixtures; those, `legacy/conf/`, `legacy/fixtures/` (golden files) and
`legacy/config_loader.py` live in `legacy/`. A few library tests also call an
old `scripts/` runner. Move them to `configs/` overlays and `scripts_new/` as
they are rewritten; `legacy/` goes when nothing uses it.
