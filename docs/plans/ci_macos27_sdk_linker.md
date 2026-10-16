# macOS 27 SDK breaks the solver links on the `macos-26` runner

Open issue (2026-10-16). Workaround in place: `.github/workflows/tests-integration.yml`
runs on `macos-15` instead of `macos-26`. Undo that once the link works again.

## What happens

Since the GitHub `macos-26-arm64` runner image `20261013.0424` (rolled out on
2026-10-15), the integration job fails at the LBM build. The compiles succeed;
the final link of `boltzmann` fails:

```text
gfortran -o boltzmann *.o timer.o -L<env>/lib -B/usr/bin/ -lfftw3f -lnetcdff
ld: tapi error: malformed file
/Library/Developer/CommandLineTools/SDKs/MacOSX27.0.sdk/usr/lib/libSystem.B.tbd:4:20: error: unknown architecture
                   arm64e.x1-macos, arm64e.x1-maccatalyst ]
 in '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/libSystem.tbd'
collect2: error: ld returned 1 exit status
```

The first failing test is
`tests/pylbm/test_pylbm_inlet_turbulence.py::test_digital_filter_runs_reproducibly_across_windows`.
`test-integration` runs with `--exitfirst`, so the uDALES and PALM builds were
never reached. They link through the same `apple_linker_flags`, so expect the
same failure there.

The failure doesn't depend on the code. `main` at the same commit (58b0828)
passed on the old image and fails on the new one:

| Run | Runner image | macOS integration |
|---|---|---|
| main, 2026-10-15 07:55 | `macos-26-arm64/20260907.0351` | 26 passed |
| PR #205, 2026-10-15 20:41 | `macos-26-arm64/20261013.0424` | link error |
| main, 2026-10-16 07:48 | `macos-26-arm64/20261013.0424` | link error |

All the other workflows still pass on `macos-26`, since they compile no solver.

## Background

This is the same symptom as the first time a new SDK outran conda's ld64. That
fix was `pyurbanair.utils.toolchain.apple_linker_flags`: it adds `-B/usr/bin/`
so the pixi gfortran/gcc driver links with Apple's ld instead of conda's
ld64-956.6, which cannot read `arm64e.x1` in `libSystem.tbd` (`docs/pylbm.md`,
"Platforms"). The failing link line does carry `-B/usr/bin/`, so the flag is
still applied. Now Apple's own ld cannot read the SDK it is given either.

## Likely cause (not verified, no Mac here)

The linker and the SDK come from different toolchains. The SDK in the error is
the Command Line Tools' `MacOSX27.0.sdk`
(`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`), whose `.tbd` stubs list
the new `arm64e.x1` slice. `/usr/bin/ld` is the `xcrun` shim, which runs the ld
of the active developer directory (`xcode-select -p`). On this image that is
probably an Xcode 26.x whose ld/libtapi predates the macOS 27 SDK. So an old ld
gets a new SDK. That fits `tapi error: malformed file`, which libtapi reports
for a stub it cannot parse, rather than ld64's own message.

## What to check on a Mac (with the macOS 27 SDK / CLT installed)

1. Reproduce: `pixi run -e dev python -m pytest tests/pylbm -m integration -x`
   (or build the LBM by hand and run the link line above).
2. Find which pieces are paired:
   - `xcode-select -p`, `xcrun --find ld`, `ld -v`
   - `xcrun --show-sdk-path`, `echo $SDKROOT`, `ls /Library/Developer/CommandLineTools/SDKs`
   - In the pixi env: `echo $CONDA_BUILD_SYSROOT $MACOSX_DEPLOYMENT_TARGET`. The
     conda compiler activation may be what selects the CLT SDK.
3. Candidate fixes, simplest first:
   - **Point SDK and ld at the same toolchain.** Either select the developer dir
     whose ld knows the SDK (`DEVELOPER_DIR=/Applications/Xcode_<new>.app`, or
     `sudo xcode-select -s /Library/Developer/CommandLineTools`), or pass the
     matching `-isysroot "$(xcrun --show-sdk-path)"`. If one setting is enough,
     put it in `apple_linker_flags` (the single place every solver build goes
     through) or in the CI setup action, not in each build.
   - **Link against an older SDK** if the runner still ships one (e.g. set
     `SDKROOT` to the Xcode 26 `MacOSX26.sdk`).
   - **A newer linker from conda-forge**: once an `ld64`/`cctools` release
     understands `arm64e.x1`, `apple_linker_flags` and `-B/usr/bin/` can be
     dropped entirely.
4. Check that all three solvers (LBM, uDALES, PALM) build and that
   `pixi run -e dev test-integration` passes. Then set the integration workflow
   back to `macos-26`, move this note to `implemented/`, and update the
   "Platforms" paragraph of `docs/pylbm.md` (plus the `apple_linker_flags`
   docstring) if the mechanism changes.

Users who update to the macOS 27 SDK locally will hit the same error. Until it
is fixed, the fallback is an older Command Line Tools / SDK.
