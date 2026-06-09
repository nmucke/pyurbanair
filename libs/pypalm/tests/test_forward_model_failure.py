"""Regression tests for pypalm member-failure reporting.

A PALM run that diverges terminates itself (exit 0) before writing the 3D
output dump. ``ForwardModel._locate_3d_output`` must report that missing output
as a ``subprocess.CalledProcessError`` -- the signal the ensemble layer's
resample-from-successes policy catches -- so a single diverged member is
resampled from a successful donor instead of aborting the whole ensemble run.
"""

import pickle
import subprocess
import types

import pytest

from pypalm.forward_model import ForwardModel


def _bare_forward_model(output_dir, experiment_name="000"):
    """A ForwardModel with only the attributes ``_locate_3d_output`` reads.

    Built via ``__new__`` to skip the expensive PALM staging that the real
    ``__init__`` performs; the method under test only touches ``dirs.output_dir``
    and ``experiment_name``.
    """
    fm = ForwardModel.__new__(ForwardModel)
    fm.dirs = types.SimpleNamespace(output_dir=output_dir)
    fm.experiment_name = experiment_name
    return fm


def test_missing_3d_output_raises_called_process_error(tmp_path):
    output_dir = tmp_path / "OUTPUT"
    output_dir.mkdir()
    # Mimic a diverged run: only the timeseries was written, no 3D field.
    (output_dir / "000_ts.nc").write_bytes(b"")

    fm = _bare_forward_model(output_dir)
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        fm._locate_3d_output()

    # The exception is raised in a ProcessPool worker and pickled back to the
    # parent, which only reconstructs from the *positional* constructor args
    # stored in ``self.args``. If those are missing, unpickling raises a
    # TypeError that breaks the pool instead of being caught as a member
    # failure -- so assert it round-trips through pickle with returncode/cmd.
    restored = pickle.loads(pickle.dumps(excinfo.value))
    assert restored.returncode == 1
    assert "000" in restored.cmd  # experiment name appears in the cmd label


def test_missing_3d_output_cmd_carries_experiment_name(tmp_path):
    output_dir = tmp_path / "OUTPUT"
    output_dir.mkdir()
    fm = _bare_forward_model(output_dir, experiment_name="042")
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        fm._locate_3d_output()
    assert "042" in excinfo.value.cmd


def test_present_3d_output_is_returned(tmp_path):
    output_dir = tmp_path / "OUTPUT"
    output_dir.mkdir()
    expected = output_dir / "000_3d.nc"
    expected.write_bytes(b"")

    fm = _bare_forward_model(output_dir)
    assert fm._locate_3d_output() == expected


def test_dot_pe_alternate_3d_output_is_returned(tmp_path):
    # palmrun/direct-run emit "<run>_3d.000.nc"; the glob fallback must hit it.
    output_dir = tmp_path / "OUTPUT"
    output_dir.mkdir()
    alternate = output_dir / "000_3d.000.nc"
    alternate.write_bytes(b"")

    fm = _bare_forward_model(output_dir)
    assert fm._locate_3d_output() == alternate
