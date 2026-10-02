# Handover: migrate the last tests off `tests/legacy/`

**For:** the agent doing this work. **Branch:** create one from
`feat/simplified-configs-and-scripts` and open the PR back into that branch
(not into `main`). Read `AGENTS.md` first; its rules apply.

## Why

`tests/` composes the real `configs/` made tiny by overlays in `tests/configs/`
(`compose("forward", "+test=forward", root=tmp_path)` from `tests/conftest.py`).
13 library tests still use the frozen configs of the *old* config schema in
`tests/legacy/conf/`, through fixtures carried over from the retired test suite.
That is two config systems for one test suite, and the legacy one no longer
matches what the code runs. Move those tests to the overlays and delete
`tests/legacy/`.

## The 13 files

| File | Uses |
|---|---|
| `tests/pyudales/test_udales_discrepancy_physics.py` | `compose_test_cfg`, `compose_module_cfg` |
| `tests/pyudales/test_udales_inlet_turbulence.py` | `compose_test_cfg` |
| `tests/pyudales/test_warmstart_carry_integration.py` | `compose_test_cfg` |
| `tests/pyudales/test_udales_discrepancy_wiring.py` | `compose_test_cfg` |
| `tests/pyurbanair/test_spinup.py` | `compose_test_cfg` |
| `tests/pylbm/test_varying_inflow_velocity.py` | `compose_module_cfg` |
| `tests/neural_surrogates/test_neural_surrogate_forward_model.py` | `surrogate_model_dir_factory` |
| `tests/neural_surrogates/test_forward_model_history.py` | `surrogate_model_dir_factory` |
| `tests/neural_surrogates/test_upt_architecture.py` | `TEST_CONF_DIR` (legacy architecture configs) |
| `tests/neural_surrogates/test_p3d_architecture.py` | `TEST_CONF_DIR`, `legacy/fixtures/p3d_sdf_off_golden.npz` |
| `tests/data_assimilation/test_state_reduction.py` | `TEST_CONF_DIR` |
| `tests/pypalm/test_pypalm_nudging_driver.py` | `TEST_CONF_DIR` (`config_name="run_forward_model"`) |
| `tests/pypalm/test_palm_inlet_turbulence.py` | `TEST_CONF_DIR` (`config_name="run_forward_model"`) |

Verify the list first. This grep should find exactly these files; anything new
it finds is in scope too:

```bash
grep -rlE "compose_test_cfg|compose_module_cfg|surrogate_model_dir_factory|TEST_CONF_DIR|compose_test_config|tests\.legacy|legacy/fixtures" tests --include='*.py' | grep -v "^tests/legacy/"
```

`tests/legacy/` also holds:
- `fixtures.py`: the legacy fixtures, plus three autouse fixtures
  (`_isolate_run_outputs`, `_limit_torch_test_threads`,
  `_restore_hydra_config_singleton`), re-exported by `tests/conftest.py`;
- `config_loader.py`;
- data fixtures in `fixtures/`: `p3d_sdf_off_golden.npz` and
  `stl_to_lbm/xie_castro_2008_STL.stl`. The STL is byte-identical to
  `geometries/xie_and_castro/xie_castro_2008_STL.stl`. `stl_to_lbm/` also has
  an `m_city3.F90`; check who uses it.

## What to do

- **Per test, use the overlays.**
  - Compose `configs/forward.yaml` (or `assimilation.yaml`) with `+test=forward`
    / `+test=assimilation` and the tiny backend models
    (`model=pyudales_tiny`, `pylbm_tiny`, ...) through `compose(...)` in
    `tests/conftest.py`.
  - Set every value a test depends on in the overlay or as an override. Never
    rely on the current values in `configs/`; that's an `AGENTS.md` rule.
  - If several tests need the same extra settings, add one small overlay under
    `tests/configs/` rather than repeating overrides.
- **Architecture tests (`TEST_CONF_DIR`):** build the architectures from
  `configs/surrogate/architectures.yaml` (the `architectures.<family>_<size>`
  entries) or construct them directly in the test, whichever is shorter. Keep
  the golden-file test's numerics unchanged: same weights seed, same input,
  same tolerance.
- **The surrogate forward-model tests (`surrogate_model_dir_factory`)** need a
  tiny model folder. `tests/conftest.py` already trains tiny surrogates once
  per session (`training_data`, `trained`); reuse them where the test only
  needs *a* valid model folder. Keep a tiny factory only where a test needs a
  specific architecture or history length, and put it in `tests/conftest.py`
  or the test file.
- **PALM tests on `run_forward_model`:** compose `forward` with
  `model=pypalm` and the tiny overlay instead. PALM needs at least 14 vertical
  cells (see `docs/pypalm.md`); give it its own small overlay if `tiny` is too
  small.
- **Autouse fixtures:** move whichever of the three are still needed into
  `tests/conftest.py`, and drop the rest.
- **Data fixtures:**
  - Put `p3d_sdf_off_golden.npz` next to its test (e.g.
    `tests/neural_surrogates/data/`).
  - Point the LBM geometry test at `geometries/xie_and_castro/` instead of a
    copy.
  - Move or drop `m_city3.F90` depending on its use.
- **Delete `tests/legacy/`.** Also remove every mention of it:
  - `tests/README.md`;
  - the `tests/legacy/**` path filter in each `.github/workflows/tests-*.yml`;
  - the mypy `exclude` in `.pre-commit-config.yaml`;
  - `AGENTS.md`, if mentioned;
  - `docs/`.

  The folders you rewrite should pass mypy: take them out of the mypy exclude
  where they do.
- **Keep coverage.** Each migrated test must still check the same behaviour.
  Delete a test only if it tests something that no longer exists, and list it
  in the PR.

## Constraints

- **Lean:** short tests, few new overlays, no new helper layers.
- **`integration`:** tests that run a compiled solver are marked `integration`;
  keep the marks.
- **Run solver tests one at a time:** concurrent runs collide on the shared
  build caches. Check `pgrep -fl "pytest|u-dales|boltzmann"` before starting
  integration tests; other agents may be testing on this machine.
- **Known local (macOS) failures:**
  - `tests/pyudales/test_udales_discrepancy_native.py` (gfortran);
  - the LBM build;
  - some multi-rank uDALES integration tests are flaky.

  Compare against the base branch before blaming your change. A separate PR
  (`docs/plans/platform_stability_handover.md`) fixes these.
- **Untouchable files:** never edit `archive/`. Never commit, stash or reset
  `configs/*.yaml` edits you didn't make (the user tunes them between runs).

## Done when

- [ ] The grep above finds nothing, and `tests/legacy/` is gone.
- [ ] Each of the 13 files composes from `configs/` + `tests/configs/` overlays
  or builds its objects directly.
- [ ] `pixi run -e dev py.test` passes. The migrated `integration` tests pass
  locally where the backend runs here (uDALES); otherwise say so. All CI
  workflows pass, and pre-commit passes.
- [ ] The CI path filters, the mypy exclude and the docs no longer mention
  `tests/legacy/`.
- [ ] The PR lists each file's migration in one line and any test removed, with
  the reason.
