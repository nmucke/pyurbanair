"""Rank-local restart staging preserves hidden fields and resets every clock."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest
import xarray as xr
from pyudales.utils.warm_start_utils import stage_discrepancy_warmstart
from scipy.io import FortranEOFError, FortranFile

from tests_new.pyudales.test_udales_output_cleanup import _make_dirs


def _records(path: pathlib.Path) -> list[np.ndarray]:
    result = []
    with FortranFile(path, "r") as handle:
        while True:
            try:
                result.append(handle.read_record(np.uint8))
            except FortranEOFError:
                return result


def _case(tmp_path: pathlib.Path) -> tuple[Any, xr.Dataset, list[pathlib.Path]]:
    dirs = _make_dirs(tmp_path, "000")
    (dirs.experiment_dir / "namoptions.000").write_text(
        "&RUN\nnprocx=2\nnprocy=1\ndtmax=0.2\n/\n"
        "&DOMAIN\nitot=8\njtot=6\nktot=4\n/\n"
        "&BC\nBCxm=1\nBCym=1\n/\n"
    )
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    paths = []
    for rank in range(2):
        path = template_dir / f"initd00000003_{rank:03d}_000.000"
        records = [
            np.arange(4 * 6 * 4, dtype=np.float64),
            np.arange(4 * 6 * 4 * 5, dtype=np.int32),
        ]
        records += [
            np.full((4 + 2) * (6 + 2) * (4 + 1), 10 * rank + index, dtype=np.float64)
            for index in range(2, 12)
        ]
        records += [np.array([3.0, 0.1])]
        with FortranFile(path, "w") as handle:
            for record in records:
                handle.write_record(record)
        paths.append(path)
    values = np.arange(4 * 6 * 8, dtype=float).reshape(4, 6, 8)
    state = xr.Dataset(
        {
            "u": (("zt", "yt", "xm"), values),
            "v": (("zt", "ym", "xt"), values + 1000),
            "w": (("zm", "yt", "xt"), values + 2000),
            "pres": (("zt", "yt", "xt"), values + 3000),
        }
    )
    return dirs, state, paths


def test_all_ranks_receive_local_flow_global_halos_and_same_clock(
    tmp_path: pathlib.Path,
) -> None:
    dirs, state, templates = _case(tmp_path)
    before = [path.read_bytes() for path in templates]
    representative = stage_discrepancy_warmstart(state, dirs, templates[0])
    assert representative == dirs.output_dir / "000" / templates[0].name
    for rank, template in enumerate(templates):
        original = _records(template)
        updated = _records(representative.parent / template.name)
        for index in [0, 1, *range(6, 12)]:
            np.testing.assert_array_equal(updated[index], original[index])
        np.testing.assert_array_equal(updated[12].view(np.float64), [0.0, 0.2])
        for index, name in enumerate(("u", "v", "w", "pres"), start=2):
            actual = updated[index].view(np.float64).reshape(6, 8, 5, order="F")
            np.testing.assert_array_equal(
                actual[1:5, 1:7, :4],
                state[name].values[:, :, rank * 4 : (rank + 1) * 4].transpose(2, 1, 0),
            )
            # Interface halo comes from the adjacent GLOBAL slab, never the
            # opposite edge of this rank's interior.
            np.testing.assert_array_equal(
                actual[0, 1:7, :4], state[name].values[:, :, (rank * 4 - 1) % 8].T
            )
            np.testing.assert_array_equal(
                actual[-1, 1:7, :4], state[name].values[:, :, ((rank + 1) * 4) % 8].T
            )
    assert [path.read_bytes() for path in templates] == before


def test_incomplete_rank_set_fails_before_staging(tmp_path: pathlib.Path) -> None:
    dirs, state, templates = _case(tmp_path)
    templates[1].unlink()
    with pytest.raises(ValueError, match="complete per-rank"):
        stage_discrepancy_warmstart(state, dirs, templates[0])
    assert not list((dirs.output_dir / "000").glob("initd*"))


def test_bad_second_rank_does_not_publish_first(tmp_path: pathlib.Path) -> None:
    dirs, state, templates = _case(tmp_path)
    with FortranFile(templates[1], "w") as handle:
        handle.write_record(np.zeros(1))
    with pytest.raises(ValueError, match="restart layout"):
        stage_discrepancy_warmstart(state, dirs, templates[0])
    assert not list((dirs.output_dir / "000").glob("initd*"))


def test_missing_velocity_rejected(tmp_path: pathlib.Path) -> None:
    dirs, state, templates = _case(tmp_path)
    with pytest.raises(ValueError, match="velocity field w"):
        stage_discrepancy_warmstart(state.drop_vars("w"), dirs, templates[0])


def test_auxiliary_restart_reads_rejected(tmp_path: pathlib.Path) -> None:
    from pyudales.utils.namoptions_utils import NamoptionsFile

    dirs, state, templates = _case(tmp_path)
    path = dirs.experiment_dir / "namoptions.000"
    baseline = path.read_text()
    for section, key in (
        ("SCALARS", "lreadscal"),
        ("RUN", "lreadmean"),
        ("INLET", "lreadminl"),
    ):
        path.write_text(baseline)
        options = NamoptionsFile(path)
        options.set_value(section, key, ".true.")
        options.set_value("SCALARS", "nsv", 1)
        options.write()
        with pytest.raises(ValueError, match="auxiliary scalar, mean, or inlet"):
            stage_discrepancy_warmstart(state, dirs, templates[0])
    assert not list((dirs.output_dir / "000").glob("initd*"))
