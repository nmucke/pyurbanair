"""
OpenMP runs of the LBM (``forward_model.ncpu``).

``ncpu > 1`` builds the gfortran binary with ``MP=1`` and runs it with
``OMP_NUM_THREADS=ncpu``; ``ncpu == 1`` must leave the build, its stamp and the
launch exactly as before. The unit tests mock ``make`` and the launch; the
``integration`` test runs the tiny case serially and on two threads.
"""

import pathlib
import re
import subprocess
import sys
from typing import Any, Optional

import numpy as np
import pytest
import xarray
from hydra.utils import instantiate
from pylbm import forward_model as forward_model_module
from pylbm.forward_model import ForwardModel
from pylbm.utils import compile_utils, infile_utils
from pylbm.utils.build_tree_utils import (
    compute_build_signature,
    read_build_stamp,
    write_build_stamp,
)

from tests.conftest import compose

STL_PATH = pathlib.Path("geometries/xie_and_castro/xie_castro_2008_STL.stl")


def _make_model(temp_dir: pathlib.Path, ncpu: int, cuda: Any = False) -> ForwardModel:
    """Construct a ForwardModel without compiling or running anything."""
    return ForwardModel(
        stl_path=STL_PATH,
        temp_dir=temp_dir,
        nx=12,
        ny=10,
        nz=4,
        bounds=((0.0, 20.0), (0.0, 20.0), (0.0, 10.0)),
        cuda=cuda,
        verbose=False,
        ncpu=ncpu,
    )


@pytest.mark.parametrize("ncpu", [0, -1, 1.5, True])  # type: ignore[misc]
def test_rejects_bad_ncpu(tmp_path: pathlib.Path, ncpu: Any) -> None:
    with pytest.raises(ValueError, match="ncpu"):
        _make_model(tmp_path, ncpu=ncpu)


class TestBuildSignature:
    def test_serial_signature_has_no_openmp_key(self, tmp_path: pathlib.Path) -> None:
        signature = compute_build_signature(tmp_path, "runcase", False, True)
        assert "openmp" not in signature
        assert signature == compute_build_signature(
            tmp_path, "runcase", False, True, openmp=False
        )

    def test_openmp_signature_records_it(self, tmp_path: pathlib.Path) -> None:
        signature = compute_build_signature(
            tmp_path, "runcase", False, True, openmp=True
        )
        assert signature["openmp"] is True


class TestPrebuiltBinaryStaleness:
    """``model.compile=false`` must not reuse a serial binary for ncpu > 1."""

    def _stamp(self, model: ForwardModel, openmp: bool, cuda: bool = False) -> None:
        model.dirs.executable_path.parent.mkdir(parents=True, exist_ok=True)
        model.dirs.executable_path.touch()
        write_build_stamp(
            build_root=model.dirs.lbm_src_path.parent,
            signature=compute_build_signature(
                src_path=model.dirs.lbm_src_path,
                experiment_name=model.dirs.experiment_name,
                enable_cuda=cuda,
                enable_netcdf=True,
                openmp=openmp,
            ),
        )

    @pytest.mark.parametrize("ncpu", [1, 4])  # type: ignore[misc]
    def test_matching_build_is_reused(self, tmp_path: pathlib.Path, ncpu: int) -> None:
        model = _make_model(tmp_path, ncpu=ncpu)
        self._stamp(model, openmp=ncpu > 1)
        model._verify_prebuilt_binary()

    def test_old_stamp_without_key_is_serial(self, tmp_path: pathlib.Path) -> None:
        model = _make_model(tmp_path, ncpu=1)
        self._stamp(model, openmp=False)
        recorded = read_build_stamp(model.dirs.lbm_src_path.parent)
        assert recorded is not None and "openmp" not in recorded
        model._verify_prebuilt_binary()

    @pytest.mark.parametrize(  # type: ignore[misc]
        "ncpu, built_openmp", [(4, False), (1, True)]
    )
    def test_mismatched_build_is_stale(
        self, tmp_path: pathlib.Path, ncpu: int, built_openmp: bool
    ) -> None:
        model = _make_model(tmp_path, ncpu=ncpu)
        self._stamp(model, openmp=built_openmp)
        with pytest.raises(RuntimeError, match="stale: openmp"):
            model._verify_prebuilt_binary()

    def test_cuda_build_is_reused_for_ncpu_above_one(
        self, tmp_path: pathlib.Path
    ) -> None:
        model = _make_model(tmp_path, ncpu=4, cuda="auto")
        self._stamp(model, openmp=False, cuda=True)
        model._verify_prebuilt_binary()


def _record_make(
    monkeypatch: pytest.MonkeyPatch, model: ForwardModel, cuda: bool
) -> list[list[str]]:
    """Stub out the toolchain and ``make``; return the recorded make commands."""
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: Any) -> subprocess.CompletedProcess:
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    env_path = pathlib.Path(model.dirs.pixi_env_path)
    monkeypatch.setattr(compile_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(compile_utils, "resolve_cuda", lambda *_: cuda)
    monkeypatch.setattr(
        compile_utils, "_resolve_build_environment", lambda **_: env_path
    )
    if cuda:
        nvfortran = env_path / ".nvhpc" / "Linux" / "24.1" / "compilers" / "bin"
        monkeypatch.setattr(
            compile_utils, "find_nvfortran", lambda _: nvfortran / "nvfortran"
        )
        monkeypatch.setattr(
            compile_utils, "_detect_gpu_compute_capability", lambda: None
        )
        monkeypatch.setattr(
            compile_utils, "_ensure_cuda_netcdf_fortran", lambda **_: env_path
        )
    return calls


class TestMakeArgs:
    @pytest.mark.parametrize("ncpu", [1, 2])  # type: ignore[misc]
    def test_mp_only_for_openmp_cpu_builds(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, ncpu: int
    ) -> None:
        model = _make_model(tmp_path, ncpu=ncpu)
        calls = _record_make(monkeypatch, model, cuda=False)
        compile_utils.compile_lbm(
            dirs=model.dirs, verbose=False, enable_cuda=False, openmp=ncpu > 1
        )

        # Both the depends.file priming pass and the -B build.
        assert [call[-1] for call in calls] == ["depends.file", "-B"]
        for call in calls:
            assert ("MP=1" in call) == (ncpu > 1)
        stamp = read_build_stamp(model.dirs.lbm_src_path.parent)
        assert stamp is not None
        assert stamp.get("openmp", False) == (ncpu > 1)
        assert ("openmp" in stamp) == (ncpu > 1)

    def test_cuda_build_warns_and_drops_openmp(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        model = _make_model(tmp_path, ncpu=4)
        calls = _record_make(monkeypatch, model, cuda=True)
        with caplog.at_level("WARNING", logger=compile_utils.logger.name):
            compile_utils.compile_lbm(
                dirs=model.dirs, verbose=False, enable_cuda=True, openmp=True
            )

        assert "OpenMP does not apply" in caplog.text
        assert len(calls) == 2
        for call in calls:
            assert "CUDA=1" in call and "MP=1" not in call
        stamp = read_build_stamp(model.dirs.lbm_src_path.parent)
        assert stamp is not None and "openmp" not in stamp


@pytest.mark.parametrize("ncpu", [1, 3])  # type: ignore[misc]
def test_run_sets_omp_num_threads_only_above_one(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, ncpu: int
) -> None:
    model = _make_model(tmp_path, ncpu=ncpu)
    launches: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        forward_model_module,
        "run_solver",
        lambda command, *, env, **_: launches.append((list(command), dict(env))),
    )
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    model.run()

    [(command, env)] = launches
    assert command == [
        "sh",
        "-c",
        "ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null; "
        f"{model.dirs.executable_path}",
    ]
    if ncpu == 1:
        assert "OMP_NUM_THREADS" not in env
    else:
        assert env["OMP_NUM_THREADS"] == str(ncpu)


@pytest.mark.parametrize("openmp", [False, True])  # type: ignore[misc]
def test_create_infile_runs_openmp_build_on_one_thread(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, openmp: bool
) -> None:
    model = _make_model(tmp_path, ncpu=2 if openmp else 1)
    model.dirs.executable_path.parent.mkdir(parents=True, exist_ok=True)
    model.dirs.executable_path.touch()
    envs: list[dict[str, str]] = []

    def fake_run(args: list[str], *, env: dict, **_: Any) -> Any:
        envs.append(dict(env))
        pathlib.Path("infile.in").touch()  # cwd is the experiment dir
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(infile_utils.subprocess, "run", fake_run)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    infile_utils.create_infile(dirs=model.dirs, verbose=False, openmp=openmp)

    [env] = envs
    assert env.get("OMP_NUM_THREADS") == ("1" if openmp else None)


def _links_openmp_runtime(binary: pathlib.Path) -> bool:
    """Whether ``binary`` links an OpenMP runtime (libomp or libgomp)."""
    tool = ["otool", "-L"] if sys.platform == "darwin" else ["ldd"]
    listing = subprocess.run(
        [*tool, str(binary)], capture_output=True, text=True, check=True
    ).stdout
    return re.search(r"lib(g)?omp\b", listing) is not None


def _tiny_run(
    root: pathlib.Path,
    ncpu: int,
    seed: Optional[str],
    turbulence: bool = True,
    threads: Optional[int] = None,
) -> tuple[xarray.Dataset, str]:
    """TEMPORARY diagnostic variant (CI Linux mismatch)."""
    cfg = compose(
        "forward",
        "+test=forward",
        "model=pylbm_tiny",
        f"model.forward_model.ncpu={ncpu}",
        f"model.forward_model.inlet_turbulence.enabled={str(turbulence).lower()}",
        root=root,
    )
    model = instantiate(cfg.model.forward_model)
    instantiate(cfg.model.prepare, forward_model=model)
    seed_file = model.dirs.experiment_dir / "seed_0000.orig"
    if seed is not None:
        seed_file.write_text(seed)
    import os

    saved = os.environ.get("OMP_NUM_THREADS")
    if threads is not None:
        model.ncpu = 1
        os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        params = xarray.Dataset(
            data_vars={"inflow_angle": 10.0, "velocity_magnitude": 5.0}
        )
        state = model.run_single(params=params).load()
    finally:
        if saved is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = saved
    return state, seed_file.read_text()


@pytest.mark.integration  # type: ignore[misc]
def test_openmp_run_matches_serial(tmp_path: pathlib.Path) -> None:
    """TEMPORARY diagnostic: report max diffs of several build/thread variants."""
    import os
    import platform

    ref, seed = _tiny_run(tmp_path / "S", 1, None)
    runs = {
        "serial repeat": _tiny_run(tmp_path / "S2", 1, seed)[0],
        "MP build, 1 thread": _tiny_run(tmp_path / "M1", 2, seed, threads=1)[0],
        "ncpu=2": _tiny_run(tmp_path / "T2", 2, seed)[0],
        "ncpu=2 repeat": _tiny_run(tmp_path / "T2b", 2, seed)[0],
        "ncpu=4": _tiny_run(tmp_path / "T4", 4, seed)[0],
    }
    ref_off, _ = _tiny_run(tmp_path / "Soff", 1, seed, turbulence=False)
    off = {
        "turb off, ncpu=2": _tiny_run(tmp_path / "T2off", 2, seed, turbulence=False)[0],
        "turb off, MP build 1 thread": _tiny_run(
            tmp_path / "M1off", 2, seed, turbulence=False, threads=1
        )[0],
    }

    def diff(a: xarray.Dataset, b: xarray.Dataset) -> str:
        return ", ".join(
            f"{v}={float(abs(a[v] - b[v]).max()):.3e}" f"/{int((a[v] != b[v]).sum())}"
            for v in ("u", "v", "w")
        )

    lines = [f"{k}: {diff(ref, v)}" for k, v in runs.items()]
    lines += [f"{k}: {diff(ref_off, v)}" for k, v in off.items()]
    lines.append(f"turb on vs off serial: {diff(ref, ref_off)}")
    cpu = ""
    if pathlib.Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line
                for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith(("model name", "flags"))
            ),
            "",
        )
    lines.append(f"platform: {platform.platform()} {cpu} ncpus={os.cpu_count()}")
    report = "\n".join(lines)
    print(report)
    assert all(
        float(abs(ref[v] - r[v]).max()) == 0.0 for r in runs.values() for v in "uvw"
    ), report
