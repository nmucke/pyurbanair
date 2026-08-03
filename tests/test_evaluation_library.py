"""Import and leaf-library guards for ``libs/evaluation``.

The library is only useful as a *leaf*: metric and figure code that scripts
call, with no path back into Hydra, the run-directory layout, jax, or the
forward-model packages. That is easy to state and easy to break by a single
convenience import, so it is asserted here rather than left to review.

The checks run in subprocesses because ``sys.modules`` is shared across the
test session — by the time this file executes, another test may already have
imported jax or matplotlib, and an in-process assertion would be meaningless.
"""

import subprocess
import sys

import pytest

# Importing any of these from the library breaks the leaf rule: jax and the
# backends are heavyweight and drag the whole application stack behind them.
FORBIDDEN = (
    "jax",
    "jaxlib",
    "pyurbanair",
    "data_assimilation",
    "pylbm",
    "pyudales",
    "pypalm",
    "neural_surrogates",
    "hydra",
    "omegaconf",
)

# The numeric modules must also stay plotting-free: matplotlib belongs to
# ``style``/``figures`` alone, so metric computation never pays for it.
NUMERIC_MODULES = ("evaluation.scores", "evaluation.turbulence", "evaluation.sensors")

PLOTTING_MODULES = ("evaluation.style", "evaluation.figures")


def _import_and_report(modules: tuple[str, ...]) -> set[str]:
    """Import ``modules`` in a fresh interpreter; return the loaded top-level names."""
    code = (
        "import sys\n"
        f"for name in {modules!r}:\n"
        "    __import__(name)\n"
        "print('\\n'.join(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, f"importing {modules} failed:\n{completed.stderr}"
    return set(completed.stdout.split())


def test_package_and_all_modules_import() -> None:
    loaded = _import_and_report(("evaluation",) + NUMERIC_MODULES + PLOTTING_MODULES)
    assert "evaluation" in loaded


# pytest ships no stubs in this env, so its decorators read as untyped to mypy.
@pytest.mark.parametrize(  # type: ignore[misc]
    "module", ("evaluation",) + NUMERIC_MODULES + PLOTTING_MODULES
)
def test_no_application_stack_behind_the_library(module: str) -> None:
    loaded = _import_and_report((module,))
    assert not loaded & set(FORBIDDEN), (
        f"{module} pulled in {sorted(loaded & set(FORBIDDEN))}; "
        "libs/evaluation must stay a leaf (master plan invariant 5)"
    )


def test_numeric_modules_do_not_import_matplotlib() -> None:
    loaded = _import_and_report(NUMERIC_MODULES)
    assert (
        "matplotlib" not in loaded
    ), "matplotlib must be imported by evaluation.style/figures only"
