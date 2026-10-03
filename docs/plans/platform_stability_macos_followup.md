# macOS follow-up: verify the platform-stability PR on a current Mac

**For:** the session on a Mac (osx-arm64, current macOS and Xcode) that
verifies the platform-stability PR (#159, issue #148). Work through the items in
order, from a fresh clone, then move this file to `docs/plans/implemented/`.

The PR was written on Linux, and GitHub's macOS runners (SDK 26.5 at most)
cannot reproduce the conda linker failure of a current SDK, so this needs a
real Mac.

**Status after the first Mac run** (macOS 26.7, SDK 27.0, commit `02b88e2`):
items 0, 1, 2, 5 and 6 passed and item 7 did not recur (about 10
preprocessing runs). Item 3 failed: PALM did not link on SDK 27, and it linked
Homebrew's FFTW. Both are fixed since; rerun everything, and item 3 in
particular.

## 0. Fresh clone

```bash
git clone --recurse-submodules https://github.com/nmucke/pyurbanair.git
# the branch under verification: fix/platform-stability until #159 merges,
# feat/simplified-configs-and-scripts after
cd pyurbanair && git checkout fix/platform-stability
xcode-select -p || xcode-select --install   # SDK + Apple's ld
pixi run setup-dev
```

Expected: `setup-dev` finishes without errors, and `git status` stays clean
(`pyproject.toml` requires pixi >= 0.63, the version that wrote `pixi.lock`;
`pixi self-update` if it refuses). Set no environment variables (in particular
not `KMP_DUPLICATE_LIB_OK`) anywhere in this follow-up.

## 1. One OpenMP runtime (torch from conda-forge)

```bash
pixi run -e dev python -c "import numpy, scipy, torch, torchvision; print(torch.__version__, torchvision.__version__, torch.ones(3) @ torch.ones(3))"
pixi list -e dev | grep -E "^(pytorch|torch|torchvision|llvm-openmp) "
```

Expected: prints `2.12.1 0.27.1 tensor(3.)` with no `OMP: Error #15`; the list
shows `pytorch`, `torchvision` and `llvm-openmp` as conda packages and no PyPI
`torch`. If a PyPI torch shows up, check `[tool.pixi.feature.neural-surrogates.target.osx-arm64.dependencies]`
in `pyproject.toml` and the `torchvision` entry in `conda-pypi-mapping.json`.

## 2. Apple's linker for every native build (the SDK 27 failure)

Conda's ld64-956.6 cannot read a current SDK's `libSystem.tbd` ("unknown
architecture arm64e.x1", then missing `expf`, `memcpy`).
`pyurbanair.utils.toolchain.apple_linker_flags` adds `-B/usr/bin/` for the
uDALES build, the IBM geometry compiler, the LBM build and the discrepancy
kernel test. No GitHub runner has an SDK new enough to show the failure, so this
is the first real check.

```bash
pixi run -e dev python -c "from pyurbanair.utils.toolchain import apple_linker_flags as f; print(f('gfortran'))"
pixi run -e dev python -m pytest tests/pyudales/test_udales_discrepancy_native.py -q
pixi run -e dev python -m pytest "tests/scripts/test_forward.py::test_solver[pylbm_tiny]" "tests/scripts/test_forward.py::test_solver[pyudales_tiny]" -m integration -q
```

Expected: `['-B/usr/bin/']`, then both pytest commands pass. If the kernel test
or the LBM build fails to link, look for `-B/usr/bin/` on the failing link line
(LBM: the `LIBDIR` make variable in `pylbm/utils/compile_utils.py`).

## 3. PALM builds and runs, with the pixi env's libraries

The first Mac run found two problems, both fixed in `pypalm.install_palm()`,
which now builds PALM in a prepared environment:
- **Linker:** PALM's own CMake checks and its `mpif90` links used conda's ld
  (`libSystem.tbd ... unknown architecture arm64e.x1`). It now gets
  `apple_linker_flags` through `LDFLAGS` and `OMPI_LDFLAGS`. `OMPI_LDFLAGS`
  replaces the wrapper's own `-L`/rpath flags, so those are carried over. It
  also gets `-Wl,-headerpad_max_install_names`, so `install_palm.sh`'s
  `install_name_tool` fix-up fits. That fix-up now fails the build, instead of
  printing "PALM installed" after an error.
- **FFTW:** PALM's CMake found `/opt/homebrew/lib/libfftw3.dylib`.
  `CMAKE_PREFIX_PATH` now starts with the pixi env.

It also sets `HOME` to `libs/pypalm/palm_model_system`, so the installer's
`palmtest*.yml` no longer lands in `~/.palm/`.

```bash
ls ~/.palm > /tmp/palm_home_before 2>/dev/null
pixi run -e dev python -m pytest tests/pypalm/test_pypalm_install.py -q
pixi run -e dev python -m pytest "tests/scripts/test_forward.py::test_solver[pypalm_tiny]" -m integration -q
grep -i "found fftw" libs/pypalm/palm_install.log
otool -L libs/pypalm/palm_model_system/MAKE_DEPOSITORY_default/palm
ls ~/.palm 2>/dev/null | diff /tmp/palm_home_before - && echo "~/.palm untouched"
```

Expected:
- the tests pass, and the first run builds PALM (about 1.5 min);
- `Found FFTW` names `.pixi/envs/dev/lib`;
- `otool -L` lists only `.pixi/envs/dev/lib`, `MAKE_DEPOSITORY_default/rrtmg/rrtmg.so` and `/usr/lib` paths, with no `/opt/homebrew`;
- `~/.palm` is unchanged.

If the build fails, the error ends with the tail of `libs/pypalm/palm_install.log`. On a link error, look for `-B/usr/bin/` on the failing link line; the flags come from `_install_environment` in `libs/pypalm/src/pypalm/__init__.py`.

Known and harmless: the installer's optional kpp4palm (`malloc.h`) and INIFOR
(`greadlink`) builds fail on macOS. Neither is part of the `palm` binary.

## 4. The whole suites, twice

```bash
for i in 1 2; do
  pixi run -e dev py.test && pixi run -e dev test-integration || break
done
pixi install -e mcp && pixi run --locked -e mcp python -m pytest tests/mcp -m ''
```

Expected: both rounds pass with nothing deselected for the platform (the only
skips are the optional pyvista/playwright viewer tests), and the MCP suite
passes, including its `[pypalm]` forward run. A uDALES failure now carries the tail of `run.<expnr>.log`; a
`problem in wallfunmom` there means IBM geometry reached the domain's top cell
(see item 5).

## 5. Identical runs, identical IBM files

The intermittent macOS uDALES failures came from upstream's IBM preprocessing
reading past its arrays when a roof ends in the top cell; pyudales now refuses
such geometry and the tiny test grid has a 15 m lid. Check that two identical
runs agree:

```bash
for k in a b; do
  pixi run -e dev python scripts/run_forward.py --config-dir tests/configs +test=forward \
    model=pyudales_tiny paths.results_root=$PWD/.temp/det_$k paths.experiment_dir=$PWD/.temp/det_exp_$k
done
diff -rq .temp/det_exp_a .temp/det_exp_b
```

Expected: only `config.sh` (the paths) and `write_inputs.<expnr>.log` (the
IBM routine's elapsed time) differ; in particular no `facet_sections_*` or
`fluid_boundary_*` file differs.

## 6. No solver outlives its owner

```bash
pixi run -e dev python -m pytest tests/pyurbanair/test_solver_process.py -q
```

Then start a long run and kill its Python with SIGKILL:

```bash
pixi run -e dev python scripts/run_forward.py --config-dir tests/configs +test=forward \
  model=pyudales_tiny time.simulation_time=20000 paths.results_root=$PWD/.temp/orphan &
sleep 60; pkill -KILL -f "bin/python scripts/run_forward.py"; sleep 10
pgrep -l -x 'u-dales|prterun'
```

Expected: the tests pass and `pgrep` prints nothing.

## 7. Watch: one IBM preprocessor crash on macos-15 (not seen on a Mac yet)

Once, on the `macos-15` runner, `IBM_preproc.exe` (upstream's STL-to-IBM
Fortran tool, compiled at run time by `pyudales/python_udgeom/ibm.py`) died
with SIGSEGV during `test_spinup_trims_output[pyudales_stock]`; every other
preprocessing of the same grid in that job, and on the other runners, worked.
On Linux the same inputs run clean under `-fcheck=bounds`, AddressSanitizer
and valgrind, with 1 and 8 threads and with 64 KB OpenMP stacks. The one
remaining suspect is its parallel file I/O (`IBM_preproc_io.f90` reads and
writes four files in concurrent OpenMP sections). Watch for it in item 4: the
error names `IBM_preproc.exe ... SIGSEGV` in `write_inputs.<expnr>.log`. If it
recurs, rerun that log's preprocessing with `OMP_NUM_THREADS=1` and with the
sources built `-O0 -g -fcheck=all` to get a symbolized backtrace.

## Background: identical runs agree to roundoff, not bit for bit

uDALES plans its Poisson FFTs with `FFTW_MEASURE`, which picks codelets by
timing and array alignment: two identical runs on one Mac can differ by one ulp
(2.8e-17 m/s seen on `macos-26`). That is expected; the replay test compares to
1e-12 m/s for that reason. Larger differences are a bug.
