# Test suite

Use the smallest layer that proves the behavior:

```bash
pixi run -e dev py.test             # default: no real CFD solves
pixi run -e dev test-integration    # tiny compiled-solver runs
pixi run -e dev test-all            # both layers
pixi run -e dev python -m pytest tests/test_hydra_config.py
```

`integration` means a real compiled CFD solver is executed. It does not mean
"calls run(cfg)": runner tests with deterministic fake models, file staging,
numerical kernels and small neural-network training tests remain in the default
suite. Select `-m integration` or `-m ''` explicitly when calling pytest directly.
The full suite is still necessary before changes to solver orchestration ship.
Tests run serially by default; shared build caches are not safe for concurrent
pytest processes that compile the same backend.

## Independent configuration

Every test that composes a numerical run uses `tests/conf/`. These configs do not
inherit `conf/run_*.yaml`, production cases, backend defaults, or experiment
recipes. Changing a production run must not retune a test. The duplication here
is intentional: the test inputs are fixtures, not another set of user defaults.
When a constructor contract changes, update the affected fixture and test
explicitly; do not synchronize the entire tree from production.

Use `compose_test_cfg` (function scope) or `compose_module_cfg` (module scope).
They inject a fresh directory for all output/scratch roots before applying the
caller's overrides. Direct config-only tests can use
`tests.config_loader.compose_test_config` or `TEST_CONF_DIR`. A direct composer
does not allocate isolated paths; callers that execute a run must supply their
own `tmp_path` outputs. Config contract/preview tests may inspect production
composition, but must never execute those production settings.

Core CFD fixtures use a 20×20×4 grid, three-second windows, two ensemble members,
one worker and one MPI rank. Sensor positions and nudging height fit that grid;
inlet turbulence is disabled unless the test enables it. Smaller grids remove
buildings and do not appreciably reduce fixed solver startup overhead. Surrogate
fixtures use CPU, small batches and short training; export fixtures remain
compatible with the library's model artifact format. Tiny PyTorch tests use one
CPU thread to avoid thread-pool overhead; the previous setting is restored.
The Pixi test commands report the 20 slowest cases. CI also caps OpenMP and BLAS
threads at one for these small workloads and selects Open MPI's `ob1` transport
to avoid UCX interface probing on hosted runners.

Unit tests of pure functions can continue to use inline arrays/dictionaries.
They do not need Hydra simply because a test-config folder exists.

## Keep coverage useful

- Test numerical invariants, public behavior and artifacts rather than pinning
  mutable production defaults or checking implementation text unnecessarily.
- Prefer representative real-solver cases over Cartesian products of unrelated
  flags. Retain distinct backend, state-carry, disk I/O and parallel regressions.
- Share expensive setup within a test/module when state can be isolated. Do not
  make tests depend on the execution order of previous tests.
- Inspect plot artists without encoding PNGs for every assertion. Keep explicit
  real export tests for each plot and the complete figure stage.
- Keep a regression if it protects distinct behavior, even if it is short or
  resembles another regression. A lower test count is not an end in itself.

This cleanup reduced the forward runner matrix from 16 to four representative
runs and the velocity timing checks from 15 solver calls to three. The velocity
check now asserts the expected frame count as well as cross-run consistency.
The spinup comparison explicitly disables spinup in its baseline, correcting an
old test that compared two spinup-enabled runs. The actual tiny test sensors are
checked for distinctness, bounds and fluid
occupancy. A separate frozen reference layout retains the multi-resolution
held-out sensor and above-canopy regressions without reading mutable production
cases. Figure-content checks retain assertions while avoiding repeated PNG
encoding; independent real export tests remain.
