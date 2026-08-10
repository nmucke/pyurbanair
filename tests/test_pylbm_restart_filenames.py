"""The pylbm restart/output filenames must be spelled the way the solver reads them.

The LBM Fortran builds every restart and output filename from a fixed-width
iteration field (``character(len=6) cit`` / ``write(cit,'(i6.6)')``), opens the
name it built, and calls ``stop`` when that exact file is absent. Python writes
those files. If the two disagree on the width, the file Python writes is one the
solver never opens -- and because the solver's OWN restart from the previous run
sits in the same directory under the same iteration, it reads that instead. No
error anywhere; the warm start simply is not the state it was handed, which for
a state-bearing smoother means the Kalman update is discarded.

That is exactly what happened: Python carried a 9-digit field on the assumption
that an upstream "overflow fix" had widened the Fortran, and the pinned submodule
never carried it. These tests pin the agreement from BOTH ends -- the Fortran
sources are read and parsed, not restated -- so the next person to bump the
submodule finds out here rather than in a silently wrong run.
"""

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LBM_SRC = REPO / "libs" / "pylbm" / "LBM" / "src"

# Every Fortran unit that formats an iteration into a filename Python also has
# to spell: the restart writer and reader, the uvw dump, and the NetCDF output
# whose names the output collector globs.
_FORMATTING_SOURCES = (
    "m_readrestart.F90",
    "m_saverestart.F90",
    "m_save_uvw.F90",
    "m_diag.F90",
)

_ITERATION_FORMAT = re.compile(r"write\(cit,'\(?(?:a1,)?i(\d)\.(\d)\)?'\)")


requires_lbm_sources = pytest.mark.skipif(
    not LBM_SRC.is_dir(),
    reason="the LBM submodule is not checked out, so its formats cannot be read",
)


@requires_lbm_sources  # type: ignore[misc]
def test_the_fortran_still_formats_iterations_at_the_python_width() -> None:
    """Read the width out of the Fortran and compare it with the constant.

    The whole bug was two numbers that disagreed and never met. This is the
    place they meet. If a submodule bump widens the Fortran field, this fails
    with both numbers in the message -- update ``ITERATION_FIELD_WIDTH`` (and
    only then does writing wider names become correct).
    """
    from pylbm.utils.warm_start_utils import ITERATION_FIELD_WIDTH

    found: dict[str, set[int]] = {}
    for name in _FORMATTING_SOURCES:
        source = LBM_SRC / name
        assert source.is_file(), f"{name} is missing from the LBM sources"
        widths = {
            int(match.group(1))
            for match in _ITERATION_FORMAT.finditer(source.read_text())
        }
        assert widths, f"no iteration format found in {name}; the regex needs updating"
        found[name] = widths

    everything = set().union(*found.values())
    assert everything == {ITERATION_FIELD_WIDTH}, (
        f"the LBM Fortran formats iterations at width(s) {sorted(everything)} "
        f"({found}), but pylbm writes them at {ITERATION_FIELD_WIDTH}. A file "
        "written at the wrong width is one the solver never opens: it falls back "
        "to its own previous restart and the warm start silently loses the state "
        "it was handed."
    )


@requires_lbm_sources  # type: ignore[misc]
def test_the_fortran_iteration_buffer_is_exactly_the_field_width() -> None:
    """``character(len=N) cit`` must match the ``iN.N`` that writes into it.

    A buffer shorter than the format overflows to '******' rather than
    truncating, which produces a name matching neither restart pattern -- so it
    can never be pruned, and every later warm start at that iteration reads a
    permanently stale file. This is the declaration behind ``MAX_ITERATION``.
    """
    from pylbm.utils.warm_start_utils import ITERATION_FIELD_WIDTH

    declaration = re.compile(r"character\(len=(\d+)\)\s+cit")
    for name in ("m_readrestart.F90", "m_save_uvw.F90"):
        text = (LBM_SRC / name).read_text()
        lengths = {int(m.group(1)) for m in declaration.finditer(text)}
        assert lengths == {ITERATION_FIELD_WIDTH}, (
            f"{name} declares cit as {sorted(lengths)} but the iteration field is "
            f"{ITERATION_FIELD_WIDTH} wide"
        )


def test_max_iteration_is_the_largest_value_the_field_holds() -> None:
    from pylbm.utils.warm_start_utils import ITERATION_FIELD_WIDTH, MAX_ITERATION

    assert MAX_ITERATION == 10**ITERATION_FIELD_WIDTH - 1
    assert len(str(MAX_ITERATION)) == ITERATION_FIELD_WIDTH


def test_restart_names_are_zero_padded_to_the_field_width() -> None:
    from pylbm.utils.warm_start_utils import restart_file_name

    # The solver reads `restart_0000_000696.uf`. Anything else is invisible to it.
    assert restart_file_name(696) == "restart_0000_000696.uf"
    assert restart_file_name(0) == "restart_0000_000000.uf"
    assert restart_file_name(999_999) == "restart_0000_999999.uf"
    # Auxiliary companions (turbulence/theta/pottemp/tracer) share the field.
    assert (
        restart_file_name(12, prefix="turbulence", tile="0001")
        == "turbulence_0001_000012.uf"
    )


def test_an_iteration_too_large_for_the_field_is_refused_not_truncated() -> None:
    """Refusing beats writing ``restart_0000_******.uf``.

    Truncating would silently collide with another iteration; overflowing
    produces a name nothing can read or prune. The counter accumulates across
    warm starts, so a long rollout reaches this ceiling even when no single
    window is anywhere near it -- which is why the message says so.
    """
    from pylbm.utils.warm_start_utils import MAX_ITERATION, restart_file_name

    assert restart_file_name(MAX_ITERATION).count("*") == 0
    with pytest.raises(ValueError, match="does not fit"):
        restart_file_name(MAX_ITERATION + 1)


def test_the_forward_model_ceiling_is_the_filename_ceiling() -> None:
    """They were separate numbers once, and that is how they drifted apart."""
    import pylbm.forward_model as forward_model
    from pylbm.utils.warm_start_utils import MAX_ITERATION

    assert forward_model.MAX_ITERATION == MAX_ITERATION


def test_pruning_drops_a_wrong_width_duplicate_at_the_kept_iteration(
    tmp_path: pathlib.Path,
) -> None:
    """A run directory written before the fix holds both widths at one iteration.

    Both parse to the same iteration, so the "keep this iteration" rule used to
    keep both -- leaving a second restart the solver would never open sitting
    beside the one it does. Two restarts at one iteration is the ambiguity the
    whole bug was made of, so the stale width goes.
    """
    import types
    from typing import cast

    from pylbm.utils.dir_utils import DirectoryPaths
    from pylbm.utils.warm_start_utils import remove_old_restart_files

    restart_dir = tmp_path / "restart"
    restart_dir.mkdir()
    keep = restart_dir / "restart_0000_000696.uf"
    legacy = restart_dir / "restart_0000_000000696.uf"  # pre-fix, same iteration
    older = restart_dir / "restart_0000_000348.uf"
    companion = restart_dir / "turbulence_0000_000696.uf"
    for path in (keep, legacy, older, companion):
        path.write_bytes(b"x")

    # ``DirectoryPaths`` carries fourteen paths describing a whole build tree;
    # the pruning reads exactly one of them, so a stub keeps the test about
    # pruning rather than about constructing a build tree.
    dirs = cast(DirectoryPaths, types.SimpleNamespace(experiment_dir=tmp_path))
    remove_old_restart_files(dirs, keep_iteration=696)

    assert keep.exists(), "the canonical restart must survive"
    assert companion.exists(), "its auxiliary companion must survive"
    assert not legacy.exists(), "the wrong-width duplicate must be removed"
    assert not older.exists(), "a different iteration must still be removed"
