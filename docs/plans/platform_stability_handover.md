# Handover: make everything run reliably on Linux and macOS

**For:** the agent doing this work. **Branch:** create one from
`feat/simplified-configs-and-scripts` and open the PR back into that branch
(not into `main`). Read `AGENTS.md` first; its rules apply. Resolves issue #148
(`gh issue view 148`); read it.

## Where you work

You work on a **Linux (linux-64) machine**: that is your local platform. The
macOS (osx-arm64) side is verified in two ways:
- the macOS CI runner this PR adds (`macos-14`), which is your main macOS
  feedback loop;
- a final check by the user/reviewer on their Mac once the PR is ready.

Another agent is meanwhile working on the user's Mac (see "Coordination"), so
you can't run anything there yourself.

Start from a fresh clone. That is part of the test, since a fresh clone is what
must work:

```bash
git clone --recurse-submodules https://github.com/nmucke/pyurbanair.git
cd pyurbanair && git checkout feat/simplified-configs-and-scripts
git submodule update --init --recursive
```

## The principle

From the user: **anyone must be able to clone the repo, install the pixi
environments and run all the code.** Nothing may be tied to a single machine.
Snellius and DelftBlue are the only exception, and their specifics belong in
`activation_scripts/`, `job_scripts/` and the per-machine `paths.scratch.*`
entries of `configs/common.yaml`. This is now in `AGENTS.md` ("Portable") and
is the yardstick for every change in this PR.

## Goal

We claim to support the platforms in `pyproject.toml`: **linux-64** and
**osx-arm64** (Windows and other architectures are out of scope). On a fresh
clone of either, this must work with no manual environment setup:

```bash
pixi run setup-dev
pixi run -e dev py.test            # every test, nothing deselected for the platform
pixi run -e dev test-integration   # real uDALES, LBM and PALM runs
pixi install -e mcp && pixi run --locked -e mcp python -m pytest tests/mcp -m ''
```

It must also be stable: the same result on every run, and clear errors when
something is genuinely missing. Today neither platform fully meets this, and no
CI job builds or runs a solver.

## Known problems

Reproduce each one before fixing it; some may already be fixed.

**From issue #148 (local Linux):**
- **NetCDF/FFTW paths are empty.**
  `libs/pyudales/shell_scripts/build_udales_macos.sh` passes
  `NETCDF_DIR`, `NETCDF_FORTRAN_DIR`, `FFTW_DOUBLE_LIB` and `FFTW_FLOAT_LIB` to
  CMake, defaulting to empty. Only the HPC activation scripts
  (`activation_scripts/*`) set them, so on a workstation CMake fails to find
  NetCDF-Fortran (`NETCDF_HAS_INTERFACES`) or picks up the system FFTW. The
  libraries are in the pixi env (`$CONDA_PREFIX`).
- **Build scripts named `*_macos.sh`** (`build_udales_macos.sh`,
  `build_preprocessing_macos.sh`) are the only path used on Linux too.
- **Failed builds aren't retried.** That was the old `CMakeCache.txt` check.
  uDALES now builds into a managed cache (`libs/pyudales/src/pyudales/utils/solver_build.py`,
  `.cache/pyudales/<hash>/`). Its `_valid_build` checks the binary, the hashes
  and `capability.json`, so this may be fixed: verify that a failed configure
  or compile surfaces a clear error and is retried on the next run.
- **Workaround branch:** `origin/feat/local-forward-mcp` has a local workaround
  (paths defaulting to `$CONDA_PREFIX`, checking for the binary). Use it as a
  reference, not as the design.

**Seen on macOS (osx-arm64) during the refactor.** You can't reproduce these
locally. Reproduce them on the macOS CI runner: add the workflow early in the
PR, possibly as a temporary on-demand job that runs just the failing tests, so
you get macOS feedback while you work.
- **LBM build:** `prepare_compile` failed (SIGABRT from the compiled binary,
  2026-08), and the LBM build has never been confirmed working on this Mac.
  Note: calling `.pixi/envs/dev/bin/python` directly instead of `pixi run`
  skips activation and fails to link (`ld: library 'System' not found`); that
  one is expected.
- **The discrepancy kernel test**
  (`tests/pyudales/test_udales_discrepancy_native.py::test_native_discrepancy_kernel`)
  fails to compile its kernel with the conda gfortran (`-isysroot` +
  `-ffpe-trap` flags). It fails on a clean checkout too.
- **`import torch`** aborts on a duplicate OpenMP runtime unless
  `KMP_DUPLICATE_LIB_OK=TRUE` (torch's bundled libomp vs the conda env's). That
  setting is a workaround, not a fix.
- **Flaky uDALES integration tests:**
  - `test_smoother_on_udales`, the zero-coefficient MPI-rank agreement test,
    and a different 2-rank test on each run;
  - occasionally `test_solver[pyudales_tiny]` (one member exits non-zero).
  - They fail on the base branch too, and sometimes pass alone.
  - Suspects: members near the `c_vreman` stability floor (`sgs_constant`
    0.24), `prterun --oversubscribe` under load, and tolerances too tight for
    multi-rank runs.
  - Two identical tiny uDALES runs also don't give byte-identical outputs, and
    the STL→IBM preprocessing writes different `facet_sections_*`/`nfctsecs_*`
    each time. Find out whether that's expected (MPI/float order) or a bug.
- **Orphaned solver processes:** a cancelled or killed run once left a
  `prterun … u-dales` process running for hours. Check that cancellation,
  timeouts and test teardown kill the whole process tree.

**Tied to one machine (audit and remove):**
- `configs/model/pyudales.yaml` and `configs/model/neural_surrogate.yaml` set
  `matlab_bin: /opt/sw/matlab-2023b/bin/matlab`, a path from one specific
  machine. MATLAB is only needed for MATLAB preprocessing (the default is
  Python). Make it unset by default (e.g. `null`) and required only when
  MATLAB preprocessing is chosen. Check `libs/pyudales` (`DEFAULT_MATLAB_BIN`)
  and the MCP's `matlab_bin` trust check in
  `libs/mcp-server/src/mcp_server/jobs/composition.py`.
- **Audit everything else** for absolute paths, user names, assumed system
  tools (e.g. a system MPI, compiler or ffmpeg outside pixi) and steps missing
  from `pixi run setup-dev`. Start from this search, and also look for
  undocumented manual steps in the docs:
  `grep -rnE "/Users/|/home/|/opt/|/export/|/projects/|/scratch|/usr/local" configs scripts src workflows tests libs/*/src libs/*/shell_scripts activation_scripts pyproject.toml`.
  Everything found must either come from the pixi env, be derived from the
  repo, or be an explicit Snellius/DelftBlue setting in the places above.

**Not covered anywhere:**
- **CI is Linux-only** (`ubuntu-latest`) and runs only the default suite. No
  solver is built or run in CI on either platform.
- **No per-platform setup docs,** and `AGENTS.md`'s "Environment notes" only
  list workarounds.

## Scope

1. **One place sets the local build environment.**
   - Point NetCDF/FFTW (and anything else the builds need) at the pixi env on
     both platforms. Do it either in a pixi activation script for the local
     features, or as defaults inside the build scripts/`solver_build.py`.
     Choose one; don't do both.
   - The HPC (`snellius`, `delftblue`) and `cuda` activation must keep working
     unchanged: only fill in values that are unset.
2. **Platform-neutral build scripts.** Rename `*_macos.sh` (e.g.
   `build_udales.sh`, `build_preprocessing.sh`) and update their callers
   (`solver_build.py`, tests, docs). Branch on `uname` only where behaviour
   truly differs (`sed -i ''`, Apple `ld`, the SDK).
3. **Builds are correct and loud.**
   - "Built" means the artifacts exist and verify.
   - A failed configure or compile raises with the log tail, and the next run
     retries.
   - Apply the same to pylbm (`libs/pylbm/src/pylbm/utils/compile_utils.py`) and
     pypalm (`libs/pypalm/shell_scripts/install_palm.sh` and its Python caller).
4. **Fix the macOS failures above** at their cause:
   - **LBM build:** make it work on osx-arm64.
   - **Discrepancy kernel test:** make it compile with the conda toolchain; the
     fix is in the test's compile command, not the vendored source.
   - **OpenMP clash:** resolve it in the environment (e.g. a single OpenMP
     runtime, via a pin or activation), so `KMP_DUPLICATE_LIB_OK` isn't needed.
     If no clean fix exists, set the variable once in pixi activation for
     osx-arm64 and explain why.
5. **Make the uDALES integration tests deterministic.** Find the cause of the
   flakiness:
   - **Tiny runs:** give the test runs settings that are safely stable (e.g.
     `sgs_constant` well above the `c_vreman` floor in the tiny overlays).
   - **Multi-rank tolerances:** justify them from the physics/precision, not
     by retrying.
   - **Process cleanup:** make sure no stray solver processes survive a test.
   - **Retries:** never add a retry or `flaky` marker to hide a failure.
6. **CI on both platforms.**
   - Add a `macos-14` (osx-arm64) runner alongside `ubuntu-latest` for the
     default suite. A matrix in the shared setup is fine if it stays simple.
   - Add one integration workflow that builds and runs the tiny uDALES, LBM and
     PALM tests on both platforms. It's slow, so trigger it on PRs touching
     `libs/py*`, the build scripts or `activation_scripts/`, plus nightly and
     on demand.
   - Use the existing setup action and pixi cache.
7. **Docs:**
   - per-platform setup notes in `README.md` (install) and the backend docs
     (`docs/pyudales.md`, `docs/pylbm.md`, `docs/pypalm.md`);
   - update `AGENTS.md` "Environment notes": remove each workaround you
     eliminate, and keep the section short.

## Notes for the macOS follow-up session

After this PR, a separate session on a Mac (osx-arm64, a current macOS/Xcode)
checks that everything fully works there. Prepare it: write
`docs/plans/platform_stability_macos_followup.md` and keep it up to date as
you work. It lists everything you could not run or verify yourself on macOS.
- Anything only the CI runner covered, which has an older SDK than a current
  Mac.
- Fixes that only a current macOS SDK can confirm. For example, the shared
  Apple-linker fix for the uDALES and LBM builds and the discrepancy kernel
  test.
- Anything else you couldn't verify.

For each item give the exact commands to run (from a fresh clone), the expected
result, and what to look at if it fails. Keep it short and actionable. That
session works through it and then moves it to `docs/plans/implemented/`.

## Constraints

- **Lean and simple:** the smallest change that makes each problem go away,
  in one obvious place. No new config knobs, wrapper layers or per-platform
  copies of scripts.
- **Vendored code is upstream:** `libs/pyudales/u-dales/`,
  `libs/pypalm/palm_model_system/` and `libs/pylbm/LBM/` aren't edited. Fix
  things in the wrappers, build scripts, activation or pixi. uDALES has a
  sanctioned patch mechanism (`solver_extensions/`); if an upstream change is
  truly unavoidable, ask the user first.
- **No-op elsewhere:** HPC and GPU environments behave exactly as before, and
  default solver numerics stay identical (check `solver_build.py`'s
  environment identity: changing build flags invalidates cached builds, which
  is fine but should be intentional).
- **Testing:**
  - **Linux locally:** run solver tests one at a time; concurrent runs collide
    on the shared build caches. Run the integration suite at least twice to
    show it's stable.
  - **macOS:** use the CI runner. Note in the PR which macOS results come from
    CI only.
- **Coordination:** the `tests/legacy/` migration PR
  (`docs/plans/legacy_tests_migration_handover.md`) is in progress in parallel,
  on the user's Mac. It touches some of the same test files
  (`tests/pyudales/*`, `tests/pylbm/*`, `tests/pypalm/*`, `tests/conftest.py`).
  - Keep your test edits minimal.
  - Fetch the base branch regularly.
  - Expect to rebase if that PR merges first.
- **Untouchable files:** never edit `archive/`, or commit, stash or reset
  `configs/*.yaml` edits you didn't make.

## Done when

- [ ] On linux-64 (locally, from a fresh clone, and in CI) and osx-arm64 (CI
  runner; the reviewer re-checks on a Mac), a fresh clone passes
  `pixi run setup-dev`, `py.test`, `test-integration` and the MCP suite, twice
  in a row, with no platform-specific deselects or workaround variables.
- [ ] Issue #148's checklist is done (link the PR to it).
- [ ] Nothing outside the Snellius/DelftBlue places is tied to one machine: no
  machine-specific defaults (incl. `matlab_bin`), and no setup step beyond
  cloning and `pixi run setup-dev` / `pixi install -e <env>`.
- [ ] No `*_macos.sh` names remain. Build failures raise clearly and are
  retried. Local NetCDF/FFTW come from the pixi env.
- [ ] CI runs the default suite on both platforms, plus an integration workflow
  (both platforms) that builds all three solvers.
- [ ] `AGENTS.md`, `README.md` and the backend docs describe per-platform
  setup, and obsolete workarounds are removed.
- [ ] The PR lists each problem above with its root cause and fix, or why it
  was out of reach.
- [ ] `docs/plans/platform_stability_macos_followup.md` lists every macOS item
  you couldn't run or verify yourself, with exact commands and expected
  results for the follow-up session on a Mac.
